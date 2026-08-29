r"""通用：按 env 组合切换行为，用真值判定「准确率 / 墙钟」的权衡。

凡是要判断某个开关"值不值"（有损优化、质量变更），都应该看**真值准确率**
而不是"文本有没有变"——文本变了可能是变好也可能是变坏，而置信度上升也
可能对应错字（实测 `羸弱 → 赢弱` 置信度反而从 0.9433 升到 0.9700）。

用法：
  python tools/_probe_truth_env.py --video test5 --cases "关:;开:DECORD_SKIP_LOOP_FILTER=all"
      [--reps 3]

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
path, roi_s, start, end, dbe, obe, stride = sys.argv[1:8]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_start=int(start), frame_end=int(end),
                    sample_stride=int(stride), decode_backend=dbe,
                    ocr_backend=obe, keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
got = {}
for s in r.segments:
    for f in (s.frames or (s.start,)):
        got[int(f)] = (s.text or "", round(float(s.confidence), 5))
print(json.dumps({'wall': round(wall, 3), 'segs': len(r.segments),
                  'got': got, 'fps': r.fps,
                  'timing': {k: round(v, 3) for k, v in ex.timing.items()},
                  'mean_conf': (round(statistics.mean(
                      [v[1] for v in got.values()]), 5) if got else 0.0)}))
"""


def truth_meta(p: Path) -> dict:
    out: dict = {}
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.search(r"roi\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
                          line)
            if m and "roi" not in out:
                out["roi"] = ",".join(m.groups())
            m = re.search(r"frame_start\s*=\s*(-?\d+)", line)
            if m:
                out["start"] = int(m.group(1))
            m = re.search(r"frame_end\s*=\s*(-?\d+)", line)
            if m:
                out["end"] = int(m.group(1))
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


def num_eq(x: str, y: str) -> bool:
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
    ap.add_argument("--dbe", default="cpu")
    ap.add_argument("--ocr", default="auto")
    ap.add_argument("--stride", type=int, default=1)
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
    print(f"=== {a.video}  roi={meta.get('roi')}  帧 {meta.get('start')}.."
          f"{meta.get('end')}  真值 {len(truth)} 帧  reps={a.reps} ===")
    print(f"  {'用例':<14s} {'墙钟':>8s} {'段':>6s} {'比对帧':>7s} "
          f"{'全等':>16s} {'数值容错':>16s} {'均置信':>8s}")

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
                 str(meta.get("start", 0)), str(meta.get("end", 0)),
                 a.dbe, a.ocr, str(a.stride)],
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
        got = {int(k): v[0] for k, v in best["got"].items()}
        both = [f for f in got if f in truth]
        ok = sum(1 for f in both if got[f] == truth[f])
        okn = sum(1 for f in both if got[f] == truth[f] or num_eq(got[f], truth[f]))
        acc = ok / len(both) if both else 0.0
        accn = okn / len(both) if both else 0.0
        res[name] = {"wall": best["wall"], "acc": acc, "accn": accn,
                     "n": len(both), "segs": best["segs"],
                     "decode": best["timing"].get("decode", 0.0),
                     "mean_conf": best["mean_conf"]}
        print(f"  {name:<14s} {best['wall']:7.3f}s {best['segs']:6d} "
              f"{len(both):7d} {ok:6d} ({acc:7.3%}) {okn:6d} ({accn:7.3%}) "
              f"{best['mean_conf']:8.5f}")

    if len(res) >= 2:
        names = list(res)
        b = res[names[0]]
        print()
        for nm in names[1:]:
            g = res[nm]
            print(f"  {names[0]} → {nm}: 墙钟 {(g['wall']/b['wall']-1)*100:+.1f}%  "
                  f"解码 {(g['decode']/max(b['decode'],1e-6)-1)*100:+.1f}%  "
                  f"全等 {(g['acc']-b['acc'])*100:+.2f}pp  "
                  f"数值容错 {(g['accn']-b['accn'])*100:+.2f}pp")
    outp = Path(__file__).with_name("_probe_truth_env.json")
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n明细落盘：{outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
