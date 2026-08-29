r"""测量各视频 ROI 内的**留白量**（左/右/内容占比）。

背景：裁切（OCR_ROI_AUTOCROP）在 test5/test6 上大幅改善、在 test/test2 上
退化。force_aspect 与效果高度相关（fa=1.5 的两个都改善、fa=0 的都退化），
但 fa 可能只是混淆变量 —— 真正的判据可能是 **ROI 相对内容的宽裕程度**
（留白多 → 裁掉空白有益；ROI 已紧凑 → 裁切可能切到笔画）。

本探针直接用引擎抽代表帧（keep_crops），按与 `_crop_to_content` 相同的判据
（Otsu 阈值 + 每列墨迹数 ≥ 2）算内容列范围，输出留白分布。

用法：
  python tools/_probe_roi_whitespace.py --videos test5,test6,test,test2,test3
      [--frames 3000]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

GT = Path(r"D:\Videos\racelog_test\ground_truth_csv")
VID = Path(r"D:\Videos\racelog_test")

WORKER = r"""
import sys, json
sys.path.insert(0, r"D:\Repo\video_ocr_engine")
import numpy as np
from segmentation import _otsu
from video_ocr_engine import FieldExtractor

vp, roi_s, s, e, stride, nframes = sys.argv[1:7]
roi = tuple(int(x) for x in roi_s.split(','))
kw = dict(frame_start=int(s), sample_stride=int(stride),
          decode_backend='cpu', ocr_backend='auto',
          rep_crop_format='gray', keep_crops=True)
if int(nframes) > 0:
    kw['frame_end'] = int(s) + int(nframes)
else:
    kw['frame_end'] = int(e)
ex = FieldExtractor(vp, roi, **kw)
r = ex.extract()
out = []
for seg in r.segments:
    c = seg.rep_crop
    if c is None:
        continue
    g = c[..., 0] if c.ndim == 3 else c
    w = int(g.shape[1])
    if w <= 8 or float(g.std()) < 3.0:
        continue
    th = _otsu(g)
    br = float((g > th).sum())
    mask = (g > th) if br <= float((g <= th).sum()) else (g <= th)
    cols = np.nonzero(mask.sum(axis=0) >= 2)[0]
    if len(cols) == 0:
        continue
    out.append([w, int(cols[0]), int(cols[-1])])
print(json.dumps(out))
"""


def meta(p: Path) -> dict:
    o: dict = {}
    for line in open(p, encoding="utf-8-sig"):
        if not line.startswith("#"):
            break
        m = re.search(r"roi\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", line)
        if m and "roi" not in o:
            o["roi"] = ",".join(m.groups())
        for key in ("frame_start", "frame_end", "force_aspect"):
            m = re.search(rf"{key}\s*=\s*(-?[\d.]+)", line)
            if m:
                v = float(m.group(1))
                o[key] = int(v) if key != "force_aspect" else v
    return o


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="test5,test6,test,test2,test3")
    ap.add_argument("--frames", type=int, default=3000,
                    help="每个视频最多抽多少帧（0=全量）")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    print(f"每视频最多 {a.frames} 帧，stride={a.stride}")
    print()
    print(f"{'视频':<7s}{'fa':>5s}{'ROI宽高比':>10s}{'段数':>7s}  "
          f"{'内容占比p10':>11s}{'p50':>8s}{'p90':>8s}  "
          f"{'左留白p50':>10s}{'右留白p50':>10s}{'可裁比例':>9s}")
    print("-" * 96)
    res = {}
    for v in a.videos.split(","):
        v = v.strip()
        tp = GT / f"{v}_ref.csv"
        if not tp.exists():
            tp = GT / f"{v}_truth.csv"
        if not tp.exists():
            print(f"{v:<7s} (无真值)")
            continue
        M = meta(tp)
        x0, y0, x1, y1 = (int(t) for t in M["roi"].split(","))
        roi_w, roi_h = x1 - x0, y1 - y0
        p = subprocess.run(
            [sys.executable, "-c", WORKER, str(VID / f"{v}.mp4"), M["roi"],
             str(M.get("frame_start", 0)), str(M.get("frame_end", 0)),
             str(a.stride), str(a.frames)],
            capture_output=True, text=True)
        lines = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not lines:
            print(f"{v:<7s} FAIL {(p.stderr or '').strip()[-110:]}")
            continue
        data = json.loads(lines[-1])
        if not data:
            print(f"{v:<7s} (无有效段)")
            continue
        frac = [(last - first + 1) / w for w, first, last in data]
        left = [first / w for w, first, last in data]
        right = [(w - 1 - last) / w for w, first, last in data]
        # 可裁比例：按现役判据（内容满宽则不裁）有多少段会被裁
        crop_able = sum(1 for w, f, l in data if not (f == 0 and l == w - 1))
        res[v] = {"n": len(data), "roi_ar": round(roi_w / roi_h, 2),
                  "fa": M.get("force_aspect", 0),
                  "frac_p50": round(pct(frac, .5), 3),
                  "left_p50": round(pct(left, .5), 3),
                  "right_p50": round(pct(right, .5), 3),
                  "crop_able": round(crop_able / len(data), 3)}
        print(f"{v:<7s}{M.get('force_aspect', 0):>5}{roi_w / roi_h:>10.2f}"
              f"{len(data):>7d}  {pct(frac, .1):>11.3f}{pct(frac, .5):>8.3f}"
              f"{pct(frac, .9):>8.3f}  {pct(left, .5):>10.3f}{pct(right, .5):>10.3f}"
              f"{crop_able / len(data):>8.0%}")

    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
