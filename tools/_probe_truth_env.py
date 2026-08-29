r"""通用：按 env 组合切换行为，用真值判定「准确率 / 墙钟」的权衡。

凡是要判断某个开关"值不值"（有损优化、质量变更），都应该看**真值准确率**
而不是"文本有没有变"——文本变了可能是变好也可能是变坏，而置信度上升也
可能对应错字（实测 `羸弱 → 赢弱` 置信度反而从 0.9433 升到 0.9700）。

## ⚠️ 两个必须对齐的口径（2026-08-29 血的教训）
曾因这两点全错，把 pad 下限从 224 下调到 160，导致生产原始误读 7→26：

1. **`force_aspect` 必须与生产一致。** 生产（RaceVideoToLog）传
   `force_aspect=mw`（真值头里 test5 = 1.5），而引擎默认 0.0。
   两者下 pad 下限的作用方向**完全相反**：

   | pad | fa=0（内容 154px） | fa=1.5（内容压到 72px） |
   |---|---:|---:|
   | 160 | **6** | 26 |
   | 192 | 12 | 17 |
   | 224 | 30 | **7** |
   | 256 | 31 | 6 |
   | 320 | 29 | **2** |

   故本探针默认**从真值头读 force_aspect 并传给引擎**（`--force-aspect` 覆盖）。

2. **必须在"段代表帧"上比，不能逐帧比。** 一段跨约 2.9 帧，逐帧比会把每个
   段错误放大约 3 倍，还混入"段内真值跳变"的噪声（test5：逐帧 150 vs
   代表帧 31，生产口径为 7）。故 `--metric rep`（默认，代表帧 + 数值 tol）
   而非 `frame`（逐帧全等）。

## 用法
  python tools/_probe_truth_env.py --video test5 \
      --cases "基线:|改动:OCR_PAD_SMALL=160" [--metric rep|frame] [--tol 1]

cases 格式：`名称:ENV1=v1;ENV2=v2`，多个用 `|` 分隔；env 为空表示清空。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PY = sys.executable
GT = Path(r"D:\Videos\racelog_test\ground_truth_csv")
VID = Path(r"D:\Videos\racelog_test")

WORKER = r"""
import os, sys, time, json, statistics
sys.path.insert(0, os.getcwd())
os.environ['ENGINE_PROFILE'] = '1'
path, roi_s, start, end, dbe, obe, stride, fa = sys.argv[1:9]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
# force_aspect 必须与生产一致：默认从真值头读出后传进来。
# 用 0.0 测会把 pad 下限的结论导向反面（见文件头注释）。
ex = FieldExtractor(path, roi, frame_start=int(start), frame_end=int(end),
                    sample_stride=int(stride), decode_backend=dbe,
                    ocr_backend=obe, keep_crops=False,
                    force_aspect=float(fa))
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
got = {}
reps = {}
for s in r.segments:
    txt = s.text or ""
    conf = round(float(s.confidence), 5)
    reps[int(s.rep_frame if s.rep_frame is not None else s.start)] = (txt, conf)
    # 逐帧口径保留（--metric frame），但默认不用：一段跨约 2.9 帧会放大误差
    for f in (s.frames or (s.start,)):
        got[int(f)] = (txt, conf)
print(json.dumps({'wall': round(wall, 3), 'segs': len(r.segments),
                  'got': got, 'reps': reps, 'fps': r.fps,
                  'timing': {k: round(v, 3) for k, v in ex.timing.items()},
                  'mean_conf': (round(statistics.mean(
                      [v[1] for v in reps.values()]), 5) if reps else 0.0)}))
"""


def truth_meta(p: Path) -> dict:
    """从真值头解析 roi / frame_start / frame_end / force_aspect / fill_width。

    头行形如 `# roi=843,993,948,1025, format=km/h, frame_start=362, ...`
    以及 `# ... force_aspect=1.5, div=1, target_h=48, max_width=0, pad=0`。
    **不能按逗号切再找前缀**（会只拿到 "843"），要逐项正则。
    """
    out: dict = {}
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.search(r"roi\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
                          line)
            if m and "roi" not in out:
                out["roi"] = ",".join(m.groups())
            for key in ("frame_start", "frame_end", "force_aspect",
                        "fill_width"):
                m = re.search(rf"{key}\s*=\s*(-?[\d.]+)", line)
                if m:
                    v = float(m.group(1))
                    out[key] = int(v) if key in ("frame_start", "frame_end",
                                                 "fill_width") else v
    return out


def load_truth(p: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
                continue
            out[int(parts[0])] = parts[2].strip()
    return out


def num_eq_tol(x: str, y: str, tol: float) -> bool:
    """生产口径：数值差 <= tol 视为正确；非数值退化成字符串全等。"""
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return x == y
    return abs(fx - fy) <= tol


def num_eq(x: str, y: str) -> bool:
    """相对容差（2%）等价判定，用于速度读数的尾位抖动。"""
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return False
    return abs(fx - fy) <= max(1e-3, 0.02 * max(abs(fx), abs(fy), 1.0))


def parse_cases(spec: str) -> list[tuple[str, dict]]:
    out = []
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        name, _, envs = part.partition(":")
        env = {}
        for kv in envs.split(";"):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            k, _, v = kv.partition("=")
            env[k.strip()] = v.strip()
        out.append((name.strip() or "(默认)", env))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dbe", default="auto")
    ap.add_argument("--ocr", default="auto")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--metric", choices=("rep", "frame"), default="rep",
                    help="rep=段代表帧（生产口径，默认）；frame=逐帧（会放大误差）")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="数值容差（生产门禁用 1）")
    ap.add_argument("--force-aspect", type=float, default=None,
                    help="覆盖真值头里的 force_aspect（默认跟随真值头 = 生产口径）")
    a = ap.parse_args()

    tp = GT / f"{a.video}_ref.csv"
    if not tp.exists():
        tp = GT / f"{a.video}_truth.csv"
    if not tp.exists():
        print(f"无真值: {a.video}")
        return 2
    meta = truth_meta(tp)
    truth = load_truth(tp)
    vp = VID / f"{a.video}.mp4"
    cases = parse_cases(a.cases)
    fa = (a.force_aspect if a.force_aspect is not None
          else float(meta.get("force_aspect", 0.0)))
    print(f"=== {a.video}  roi={meta.get('roi')}  帧 {meta.get('start')}.."
          f"{meta.get('end')}  真值 {len(truth)} 帧  reps={a.reps} ===")
    print(f"  force_aspect={fa}（{'真值头' if a.force_aspect is None else '命令行覆盖'}）"
          f"  metric={a.metric}  tol={a.tol}  decode={a.dbe}")
    print(f"  {'用例':<14s} {'墙钟':>8s} {'段':>6s} {'比对':>7s} "
          f"{'误读':>6s} {'全等':>16s} {'数值容错':>16s} {'均置信':>8s}")

    res = {}
    for name, env in cases:
        e = dict(os.environ)
        for k in env:
            e.pop(k, None)
        e.update(env)
        best = None
        err = None
        for _ in range(a.reps):
            p = subprocess.run(
                [PY, "-c", WORKER, str(vp), meta["roi"],
                 str(meta.get("frame_start", 0)), str(meta.get("frame_end", 0)),
                 a.dbe, a.ocr, str(a.stride), str(fa)],
                capture_output=True, text=True, env=e)
            out = (p.stdout or "").strip().splitlines()
            if p.returncode != 0 or not out:
                err = (p.stderr or "").strip()[-300:]
                break
            d = json.loads(out[-1])
            if best is None or d["wall"] < best["wall"]:
                best = d
        if err:
            print(f"  {name:<14s} FAIL {err}")
            continue
        src = best["reps"] if a.metric == "rep" else best["got"]
        got = {int(k): v[0] for k, v in src.items()}
        both = [f for f in got if f in truth]
        ok = sum(1 for f in both if got[f] == truth[f])
        # 生产口径：数值 |diff| <= tol 视为正确（tol=1）
        bad = sum(1 for f in both
                  if not (num_eq_tol(got[f], truth[f], a.tol)))
        acc = ok / len(both) if both else 0.0
        accn = (len(both) - bad) / len(both) if both else 0.0
        res[name] = {"wall": best["wall"], "acc": acc, "accn": accn,
                     "misread": bad, "n": len(both), "segs": best["segs"],
                     "decode": best["timing"].get("decode", 0.0),
                     "mean_conf": best["mean_conf"]}
        print(f"  {name:<14s} {best['wall']:7.3f}s {best['segs']:6d} "
              f"{len(both):7d} {bad:6d} {ok:6d} ({acc:7.3%}) "
              f"{len(both)-bad:6d} ({accn:7.3%}) {best['mean_conf']:8.5f}")

    if len(res) >= 2:
        names = list(res)
        b = res[names[0]]
        print()
        for nm in names[1:]:
            g = res[nm]
            print(f"  {names[0]} → {nm}: 墙钟 {(g['wall']/b['wall']-1)*100:+.1f}%  "
                  f"解码 {(g['decode']/max(b['decode'],1e-6)-1)*100:+.1f}%  "
                  f"误读 {b['misread']}→{g['misread']}  "
                  f"数值容错 {(g['accn']-b['accn'])*100:+.2f}pp")
    outp = Path(__file__).with_name("_probe_truth_env.json")
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n明细落盘：{outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
