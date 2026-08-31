r"""检测"误裁"：被裁掉的列里是否真的有笔画。

## 背景

裁切（`_crop_to_content`）的判据是 `g > _bin_thresh` 且**每列墨迹数 ≥ 2**，
两侧各留 `OCR_ROI_AUTOCROP_MARGIN`% 余量。若该判据把淡笔画/细边判成背景，
裁切就会切掉真实内容 —— 紧凑 ROI（留白本来就少）上尤其容易中招。

此前用"余量 10% → 20%"规避，但 20% 相当大：它会把宽 ROI 上的留白也放回来
（test5/test6 余量 30% 时误读反而 0→10）。更根本的做法应该是
**收益太小就不裁**（紧凑 ROI 自动不裁），而不是全局加大余量。

## 本探针做什么

1. 抽各视频代表帧，按引擎判据算内容列范围 (first, last)
2. 对给定余量算裁切区间，统计**裁切率**与**裁掉比例**分布
3. **误裁检测**：对被裁掉的左区间 [0,lo) 与右区间 [hi,w)，改用宽松判据
   （同一 Otsu 阈值但列墨迹数 ≥ 1，即只要有一行算墨迹就算）统计墨迹像素数。
   宽松判据下仍有墨迹 ⇒ 引擎判据漏掉了真实内容 ⇒ 存在误裁风险。

用法：
  python tools/_probe_crop_miscut.py --videos test,test2,test5,test6
      [--margins 10,20] [--frames 3000]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
import os

# WORKER 以 `python -c` 执行，-c 下 __file__ 未定义，
# 故本进程算好根目录后经 PROBE_ROOT 传给子进程。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROBE_ROOT"] = ROOT  # 供 `python -c` 的 WORKER 子进程使用
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


GT = _VIDEO_DIR / "ground_truth_csv"
VID = _VIDEO_DIR

WORKER = r"""
import sys, json
sys.path.insert(0, os.environ["PROBE_ROOT"])
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
    ink = mask.sum(axis=0)                       # 每列墨迹行数
    cols = np.nonzero(ink >= 2)[0]               # 引擎判据
    if len(cols) == 0:
        continue
    # 宽松判据：只要有一行算墨迹
    loose = np.nonzero(ink >= 1)[0]
    lo_loose = int(loose[0]) if len(loose) else int(cols[0])
    hi_loose = int(loose[-1]) if len(loose) else int(cols[-1])
    # 被裁区间内的墨迹像素数（区间 [a,b) 左闭右开）
    def ink_in(a, b):
        a = max(0, a); b = min(w, b)
        return int(mask[:, a:b].sum()) if b > a else 0
    out.append({
        "w": w,
        "first": int(cols[0]), "last": int(cols[-1]),
        "loose_first": lo_loose, "loose_last": hi_loose,
        "ink": [int(x) for x in ink],            # 逐列墨迹行数，供离线复算
    })
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


def crop_range(first, last, w, margin_pct, min_gain=0.0):
    """复刻 _content_range_to_crop：返回 (lo, hi) 或 None（不裁）。

    min_gain：裁掉比例低于此值则不裁（与 `_ocr_autocrop_min_gain` 同义）。
    """
    m = max(1, int(round(w * margin_pct / 100.0)))
    lo = max(0, first - m)
    hi = min(w, last + 1 + m)
    if lo == 0 and hi == w:
        return None
    if min_gain > 0.0 and (w - (hi - lo)) / w < min_gain:
        return None
    return lo, hi


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def analyse(data, margin_pct, min_gain=0.0):
    """统计给定余量下的裁切行为与误裁指标。"""
    n = len(data)
    cropped = []
    cut_frac = []
    miscut_l = miscut_r = 0          # 被裁区间内含墨迹（宽松判据）的段数
    tiny_cut = 0                     # 裁掉比例 < 10% 的段（收益极小）
    for d in data:
        w, ink = d["w"], d["ink"]
        rng = crop_range(d["first"], d["last"], w, margin_pct, min_gain)
        if rng is None:
            cut_frac.append(0.0)
            continue
        lo, hi = rng
        cropped.append((d, lo, hi))
        cut_frac.append((w - (hi - lo)) / w)
        # 被裁掉的左区间 [0,lo) 与右区间 [hi,w) 里是否混进了宽松判据的墨迹
        left = d["loose_first"] < lo          # 宽松起点在裁起点左边 ⇒ 左区间有内容
        right = d["loose_last"] >= hi         # 宽松终点在裁终点右边 ⇒ 右区间有内容
        if left:
            miscut_l += 1
        if right:
            miscut_r += 1
        if (w - (hi - lo)) / w < 0.10:
            tiny_cut += 1
    return {
        "n": n,
        "cropped": len(cropped),
        "crop_rate": round(len(cropped) / n, 3) if n else 0.0,
        "cut_p50": round(pct(cut_frac, .5), 3),
        "cut_p90": round(pct(cut_frac, .9), 3),
        "cut_max": round(max(cut_frac), 3) if cut_frac else 0.0,
        "miscut_left": miscut_l,
        "miscut_right": miscut_r,
        "tiny_cut": tiny_cut,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="test,test2,test5,test6")
    ap.add_argument("--margins", default="10,20")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-gain", default="0",
                    help="最小收益门槛 %%（逗号分隔可多组对比；0=不启用）")
    a = ap.parse_args()
    margins = [int(x) for x in a.margins.split(",")]
    gains = [int(x) for x in a.min_gain.split(",")]

    for v in a.videos.split(","):
        v = v.strip()
        tp = GT / f"{v}_ref.csv"
        if not tp.exists():
            tp = GT / f"{v}_truth.csv"
        if not tp.exists():
            print(f"{v}: 无真值"); continue
        M = meta(tp)
        p = subprocess.run(
            [sys.executable, "-c", WORKER, str(VID / f"{v}.mp4"), M["roi"],
             str(M.get("frame_start", 0)), str(M.get("frame_end", 0)),
             str(a.stride), str(a.frames)],
            capture_output=True, text=True)
        lines = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not lines:
            print(f"{v}: FAIL {(p.stderr or '').strip()[-110:]}"); continue
        data = json.loads(lines[-1])
        if not data:
            print(f"{v}: 无有效段"); continue

        w0 = data[0]["w"]
        print(f"\n=== {v}  ROI 宽 {w0}px  fa={M.get('force_aspect', 0)}  "
              f"段 {len(data)} ===")
        print(f"  {'余量':>4s} {'门槛':>6s} {'裁切率':>8s} {'裁掉p50':>9s} "
              f"{'p90':>7s} {'max':>7s} {'误裁左':>7s} {'误裁右':>7s} "
              f"{'微裁(<10%)':>11s}")
        for m in margins:
            for g in gains:
                st = analyse(data, m, g / 100.0)
                print(f"  {m:>3d}% {g:>5d}% {st['crop_rate']:>7.0%} "
                      f"{st['cut_p50']:>9.1%} {st['cut_p90']:>7.1%} "
                      f"{st['cut_max']:>7.1%} {st['miscut_left']:>7d} "
                      f"{st['miscut_right']:>7d} {st['tiny_cut']:>11d}")
        # 末组设置的裁掉量直方图，看是否集中在「几乎没裁」
        frs = []
        for d in data:
            rng = crop_range(d["first"], d["last"], d["w"], margins[-1],
                             gains[-1] / 100.0)
            frs.append(0.0 if rng is None
                       else (d["w"] - (rng[1] - rng[0])) / d["w"])
        buckets = [(0, .05), (.05, .10), (.10, .20), (.20, .40), (.40, 1.01)]
        hist = "  ".join(
            f"[{lo:.0%},{hi:.0%})={sum(1 for f in frs if lo <= f < hi)}"
            for lo, hi in buckets)
        print(f"  末组(余量{margins[-1]}%/门槛{gains[-1]}%) 裁掉量分布: {hist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
