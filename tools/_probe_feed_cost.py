"""宿主分段状态机每帧成本分解（解释 producer 缺口 0.60s 的构成）。

背景：`_probe_producer_gap.py` 实测 h264-cpu 热轮
    producer 2.0512s − 已计 span(1.4489s) = **缺口 0.6023s（占 producer 29%）**
缺口只可能来自 `SegmentStateMachine.feed` 的未打桩部分：
    `d = self._prev_bin != bin` / `_cluster_win3(d)` / `on_similar` / `on_emit`
本探针在**真实 ROI 形状**上逐项微基准，再乘实际帧数核对是否能解释该缺口。

⚠️ `nocluster` 短路臂**不可用于归因**（改判据 → 每帧断段 → 段数 3000 →
工作量反而暴涨 +93%，见 `_probe_producer_gap.py`）。归因只能靠本探针的
「单帧成本 × 帧数」口径。

用法：
  python tools/_probe_feed_cost.py [--roi-w 106] [--roi-h 33] [--frames 2950]
      [--reps 200]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]


def bench(fn, reps: int) -> float:
    import numpy as np  # noqa: F401  # 供 lambda 使用
    for _ in range(5):
        fn()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-w", type=int, default=106)
    ap.add_argument("--roi-h", type=int, default=33)
    ap.add_argument("--frames", type=int, default=2950)
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()

    import numpy as np
    from video_ocr_engine.domain.segmentation import _cluster_win3
    from video_ocr_engine.domain import segmentation as seg

    h, w = args.roi_h, args.roi_w
    rng = np.random.default_rng(7)
    # 真实形态：绝大多数帧与前一帧完全相同（字幕不变），少量帧有变化
    base_bin = (rng.random((h, w)) > 0.5)
    changed_bin = base_bin.copy()
    changed_bin[10:14, 40:50] = ~changed_bin[10:14, 40:50]

    rows = {}
    rows["prev_bin != bin（相同帧）"] = bench(
        lambda: (base_bin != base_bin), args.reps)
    rows["prev_bin != bin（变化帧）"] = bench(
        lambda: (base_bin != changed_bin), args.reps)

    diff_same = base_bin != base_bin
    diff_chg = base_bin != changed_bin
    rows["_cluster_win3（全 0 提前返回）"] = bench(
        lambda: _cluster_win3(diff_same), args.reps)
    rows["_cluster_win3（有变化）"] = bench(
        lambda: _cluster_win3(diff_chg), args.reps)

    # 相似判定（合并门）——用真实签名
    try:
        rows["similar_decision"] = bench(
            lambda: seg.similar_decision(0.5, 12, 2.0, 35, dense=False),
            args.reps)
    except Exception as e:  # noqa: BLE001  # 签名不符则跳过该项
        rows["similar_decision"] = float("nan")
        print("similar_decision 跳过：%s" % e)

    total = 0.0
    print("ROI %dx%d（%d px）  reps=%d  实际帧数=%d\n"
          % (w, h, w * h, args.reps, args.frames))
    print("%-34s %10s %14s" % ("每帧操作", "µs", "全片折算(s)"))
    for k, us in rows.items():
        if us != us:  # NaN
            continue
        # 变化帧占比按 1090 段/2950 帧 ≈ 37% 估（段边界即变化帧 + 少量噪声）
        sec = us * 1e-6 * args.frames
        total += sec
        print("%-34s %10.2f %14.3f" % (k, us, sec))

    # 合成估计：全部帧做 diff + cluster；37% 帧触发 emit/similar
    diff_us = rows["prev_bin != bin（相同帧）"]
    clus_us = rows["_cluster_win3（全 0 提前返回）"]
    diff_c = rows["prev_bin != bin（变化帧）"]
    clus_c = rows["_cluster_win3（有变化）"]
    p = 0.37
    est_us = ((1 - p) * (diff_us + clus_us) + p * (diff_c + clus_c))
    print("\n加权每帧估计（变化帧占比 %.0f%%）= %.1f µs" % (p * 100, est_us))
    print("全片（%d 帧）折算 = %.3f s" % (args.frames, est_us * 1e-6 * args.frames))
    print("实测缺口          = 0.602 s（producer 2.051 − 已计 1.449）")
    print("→ 解释度 %.0f%%" % (est_us * 1e-6 * args.frames / 0.602 * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
