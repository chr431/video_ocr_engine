r"""重新评估 OCR 输入 pad 宽下限（`OCR_PAD_WIDTH_MIN` / `DEFAULT_FILL_WIDTH` = 224）。

## 为什么要重测
`engine_config` 里的依据是**很久以前**的数据：
  窄图（48 高后 78-160 宽）在宽 pad 下 v6_small 更准
  （test6：224→err 0.09%，192→0.16%，48~96→0.69~1.19%；256 精度相同但更慢）
现在模型/预处理/分段都变过了，需要用**真值**重测。

## ⚠️ 前提：真正生效的旋钮是 `fill_width`，不是 `OCR_PAD_SMALL`
    extractor:  self._fill_width = fill_width or config.DEFAULT_FILL_WIDTH (=224)
    OcrEngine.__call__:
        if self._fill_width > 0:  _floor = self._fill_width     ← 默认走这条
        else:                     _floor = PAD_WIDTH_MIN_BY_MODEL; OCR_PAD_SMALL 覆盖
所以 **`OCR_PAD_SMALL` env 在默认配置下完全不生效**（README 把它列成了可调
旋钮，是错的）。本探针显式传 `fill_width` 来扫。

## 输出
每个 (视频, fill_width) 组合给出：按帧对齐真值的全等准确率、数值容错准确率
（速度读数有尾位误差）、墙钟。

用法：
  python tools/_probe_pad_width.py [--sweep 48,96,160,192,224,320]
      [--videos test5,test6] [--reps 2] [--force-aspect 0]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

# WORKER 以 `python -c` 执行，-c 下 __file__ 未定义，
# 故本进程算好根目录后经 PROBE_ROOT 传给子进程。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROBE_ROOT"] = ROOT  # 供 `python -c` 的 WORKER 子进程使用
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


PY = sys.executable
GT = _VIDEO_DIR / "ground_truth_csv"
VID = _VIDEO_DIR

WORKER = r"""
import os, sys, time, json, statistics
sys.path.insert(0, os.environ["PROBE_ROOT"])
os.environ['ENGINE_PROFILE'] = '1'
path, roi_s, start, end, dbe, obe, stride, fw, fa = sys.argv[1:10]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_start=int(start), frame_end=int(end),
                    sample_stride=int(stride), decode_backend=dbe,
                    ocr_backend=obe, keep_crops=False,
                    fill_width=int(fw), force_aspect=float(fa))
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
got = {}
for s in r.segments:
    for f in (s.frames or (s.start,)):
        got[int(f)] = (s.text or "", round(float(s.confidence), 5))
print(json.dumps({'wall': round(wall, 3), 'segs': len(r.segments),
                  'got': got, 'fps': r.fps,
                  'mean_conf': (round(statistics.mean(
                      [v[1] for v in got.values()]), 5) if got else 0.0),
                  'ocr_backend': r.meta.get('ocr_backend')}))
"""


def truth_meta(p: Path) -> dict:
    """从真值 CSV 头里取 roi / frame_start / frame_end。

    注意头行是 `# roi=843,993,948,1025, format=km/h, frame_start=362, ...`
    —— **不能按逗号切再找 `roi=` 前缀**（那样只会拿到 "843"）。
    用正则取 roi= 后面的四个整数。
    """
    import re as _re
    out: dict = {}
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = _re.search(
                r"roi\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", line)
            if m and "roi" not in out:
                out["roi"] = ",".join(m.groups())
            m = _re.search(r"frame_start\s*=\s*(-?\d+)", line)
            if m:
                out["start"] = int(m.group(1))
            m = _re.search(r"frame_end\s*=\s*(-?\d+)", line)
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


def run(video, roi, start, end, dbe, obe, stride, fw, fa, reps):
    best = None
    for _ in range(reps):
        p = subprocess.run(
            [PY, "-c", WORKER, str(video), roi, str(start), str(end), dbe, obe,
             str(stride), str(fw), str(fa)],
            capture_output=True, text=True)
        out = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not out:
            return {"err": (p.stderr or "").strip()[-400:]}
        d = json.loads(out[-1])
        if best is None or d["wall"] < best["wall"]:
            best = d
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="48,96,128,160,192,224,256,320")
    ap.add_argument("--videos", default="test5,test6,test,test2")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--dbe", default="cpu")
    ap.add_argument("--ocr", default="auto")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--force-aspect", type=float, default=0.0)
    a = ap.parse_args()
    sweep = [int(x) for x in a.sweep.split(",") if x.strip()]

    print(f"=== pad 宽下限重估：sweep={sweep}  decode={a.dbe} ocr={a.ocr} "
          f"stride={a.stride} force_aspect={a.force_aspect} reps={a.reps} ===")
    rows: dict[str, list] = {}
    for name in (x.strip() for x in a.videos.split(",") if x.strip()):
        tp = GT / f"{name}_ref.csv"
        if not tp.exists():
            tp = GT / f"{name}_truth.csv"
        if not tp.exists():
            print(f"  [skip] 无真值: {name}")
            continue
        meta = truth_meta(tp)
        truth = load_truth(tp)
        vp = VID / f"{name}.mp4"
        print(f"\n--- {name}  roi={meta.get('roi')}  帧 "
              f"{meta.get('start')}..{meta.get('end')}  真值 {len(truth)} 帧 ---")
        print(f"  {'fill_width':>10s} {'墙钟':>8s} {'段':>6s} "
              f"{'比对帧':>7s} {'全等':>16s} {'数值容错':>16s}")
        rows[name] = []
        for fw in sweep:
            d = run(vp, meta["roi"], meta.get("start", 0), meta.get("end", 0),
                    a.dbe, a.ocr, a.stride, fw, a.force_aspect, a.reps)
            if "err" in d:
                print(f"  {fw:>10d}  FAIL {d['err']}")
                continue
            got = {int(k): v[0] for k, v in d["got"].items()}
            both = [f for f in got if f in truth]
            ok = sum(1 for f in both if got[f] == truth[f])
            okn = sum(1 for f in both
                      if got[f] == truth[f] or num_eq(got[f], truth[f]))
            acc = ok / len(both) if both else 0.0
            accn = okn / len(both) if both else 0.0
            rows[name].append({"fw": fw, "wall": d["wall"], "segs": d["segs"],
                               "n": len(both), "acc": acc, "accn": accn,
                               "mean_conf": d["mean_conf"]})
            print(f"  {fw:>10d} {d['wall']:7.3f}s {d['segs']:6d} "
                  f"{len(both):7d} {ok:6d} ({acc:7.3%}) {okn:6d} ({accn:7.3%})")

    out = Path(__file__).with_name("_probe_pad_width.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n明细落盘：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
