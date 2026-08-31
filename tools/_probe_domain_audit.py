r"""专项审计：合并判据的口径问题——原始灰度域 vs OCR 输入域。

背景：merge_similar 在原始灰度 + 全局校准 Otsu 上比较两段 rep；OCR 吃的
是 48 高 resize + gamma 2.0 + force_aspect 压窄后的图（裁切路径已为此
改用逐图 Otsu，见 _crop_after_aspect 注释，合并路径未跟进）。

本探针把 92 个实际合并对（19 误 + 73 安全）的判据搬进 OCR 输入域重算：
  p = _preprocess_standard(crop, force_aspect=fa)  # 真实 OCR 输入形态
  - pbin：逐图 Otsu 二值化后的 mean/chg/max_block
  - pgray：直接灰度绝对差 mean（不做二值化——rec 消费的是连续灰度，
    二值判据对墨迹内部灰度级差异全盲，这是口径差异的另一面）
对照原始域指标，回答：换域能否把误合并类与安全类分开。

用法：
  python tools/_probe_domain_audit.py [--frames 3000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np

from _probe_det_crop_eval import GT, truth_meta
from _probe_merge_audit import actual_merges, build, load_xsg
from _probe_block_audit import max_block
import os


def pair_diff_domain(crops, reps, i, fa, mode):
    from video_utils import _preprocess_standard, _text_sep_gray
    from segmentation import _otsu
    c1, c2 = crops.get(reps[i]), crops.get(reps[i + 1])
    if c1 is None or c2 is None:
        return None
    p1 = _preprocess_standard(c1, force_aspect=fa)
    p2 = _preprocess_standard(c2, force_aspect=fa)
    g1 = p1[..., 0] if p1.ndim == 3 else p1
    g2 = p2[..., 0] if p2.ndim == 3 else p2
    if mode == "pgray":
        d = np.abs(np.asarray(g1, np.float32) - np.asarray(g2, np.float32))
        return dict(mean=float(d.mean()), chg=int((d > 10).sum()),
                    mb=0, nblocks=0, area=int(g1.size))
    o1 = _otsu(np.clip(g1, 0, 255).astype(np.uint8))
    o2 = _otsu(np.clip(g2, 0, 255).astype(np.uint8))
    ab = (g1 > o1).astype(np.float32) * 255.0
    bb = (g2 > o2).astype(np.float32) * 255.0
    d = np.abs(ab.astype(np.int16) - bb.astype(np.int16))
    mask = d > 10
    mb, sizes = max_block(mask)
    return dict(mean=float(d.mean()), chg=int(mask.sum()), mb=mb,
                nblocks=len(sizes), area=int(g1.size))


def audit(video, frames_n, xsg=False, stride=1):
    if xsg:
        meta, _ = load_xsg(video, frames_n)
    else:
        tp = GT / f"{video}_ref.csv"
        if not tp.exists():
            tp = GT / f"{video}_truth.csv"
        meta = truth_meta(tp)
    exr, fa = build(video, meta, 0 if xsg else frames_n, merge=False,
                    stride=stride, xsg=xsg)
    reps = [s.rep_frame for s in exr.extract().segments]
    crops = exr.crops
    exm, _ = build(video, meta, 0 if xsg else frames_n, merge=True,
                   stride=stride, xsg=xsg)
    merged = set(actual_merges(
        [(s.start, s.end) for s in exr.extract().segments],
        [(s.start, s.end) for s in exm.extract().segments]))
    rows = []
    for i in range(len(reps) - 1):
        if (i, i + 1) not in merged:
            continue
        for mode in ("pbin", "pgray"):
            m = pair_diff_domain(crops, reps, i, fa, mode)
            if m:
                m["i"] = i
                m["mode"] = mode
                m["ocr_diff_domain"] = fa
                rows.append(m)
    return rows, fa


def summarize(rows, key):
    v = sorted(r[key] for r in rows)
    if not v:
        return "无"
    return (f"min {v[0]:.2f} p50 {v[len(v)//2]:.2f} max {v[-1]:.2f}"
            if key == "mean" else
            f"min {v[0]} p50 {v[len(v)//2]} max {v[-1]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3000)
    a = ap.parse_args()

    mis, safe = [], []
    for video, kw in (("test5", {}), ("test6", {})):
        rows, fa = audit(video, a.frames, **kw)
        mis.extend(rows)
    for video, kw in (("test", {}), ("test2", {}),
                      ("新三国01", dict(xsg=True, stride=8))):
        rows, fa = audit(video, 30000 if kw else a.frames, **kw)
        safe.extend(rows)

    for mode in ("pbin", "pgray"):
        print(f"\n=== OCR 输入域（{mode}）===")
        mm = [r for r in mis if r["mode"] == mode]
        ss = [r for r in safe if r["mode"] == mode]
        print(f"  误合并 n={len(mm)}: mean {summarize(mm, 'mean')} | "
              f"chg {summarize(mm, 'chg')} | mb {summarize(mm, 'mb')}")
        print(f"  安全   n={len(ss)}: mean {summarize(ss, 'mean')} | "
              f"chg {summarize(ss, 'chg')} | mb {summarize(ss, 'mb')}")
        if mode == "pbin":
            hi_s = max(r["mb"] for r in ss)
            lo_m = min(r["mb"] for r in mm)
            print(f"  可分性：安全 mb≤{hi_s} vs 误合并 mb≥{lo_m} → "
                  + ("可分" if hi_s < lo_m else "重叠不可分"))
            for cap in (8, 10, 12, 15):
                blk = sum(1 for r in mm if r["mb"] > cap)
                kill = sum(1 for r in ss if r["mb"] > cap)
                print(f"    cap={cap}px：拦截误合并 {blk}/{len(mm)}"
                      f"  误杀安全 {kill}/{len(ss)}")
            # mean 阈值可分性
            hi_sm = max(r["mean"] for r in ss)
            lo_mm = min(r["mean"] for r in mm)
            print(f"  mean 可分性：安全 ≤{hi_sm:.2f} vs 误合并 ≥{lo_mm:.2f} → "
                  + ("可分" if hi_sm < lo_mm else "重叠不可分"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
