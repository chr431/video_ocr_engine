"""D1 调查第三刀：记录两管线状态机逐帧 changed 决策，定位首分歧。

RecMachine 包装 on_emit/on_similar 并在 feed 里记录决策用的 score 与
changed 布尔（宿主=win3(prev_bin!=bin)，GPU=cluster 直通），跑完 diff：
  - 首个 changed 分歧帧 + 两侧 score
  - segs 边界结构对比（首分歧段）
  - 校准窗（前 calib_n=50 帧）内的断段数两侧对比

用法：python tools/_probe_d1_trace2.py [--video test5] [--frames 3000]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VIDS = {
    "test5": (r"D:\Videos\racelog_test\test5.mp4", (843, 993, 948, 1025)),
    "test6": (r"D:\Videos\racelog_test\test6.mp4", (841, 994, 949, 1026)),
    "test": (r"D:\Videos\racelog_test\test.mp4", (841, 994, 949, 1026)),
}


def trace_pipeline(video: str, roi: tuple, frames: int, gpu_pl: str) -> dict:
    import video_ocr_engine._gpu_pipeline as gp
    import video_ocr_engine._host_pipeline as hp
    from segmentation import SegmentStateMachine, _cluster_win3
    os.environ["GPU_PIPELINE"] = gpu_pl

    rec: dict = {"feed": [], "emit": [], "similar": []}

    class RecMachine(SegmentStateMachine):
        def __init__(self, *a, **kw):
            orig_emit = kw.get("on_emit")

            def emit_rec(seg, rep, frac):
                rec["emit"].append(
                    (self._frames_of(seg)) if False else (seg[0], seg[-1]))
                orig_emit(seg, rep, frac)
            kw["on_emit"] = emit_rec
            orig_sim = kw.get("on_similar")

            def sim_rec(pa, pb):
                r = bool(orig_sim(pa, pb))
                rec["similar"].append(r)
                return r
            kw["on_similar"] = sim_rec
            super().__init__(*a, **kw)

        def feed(self, k, fi, sharp, payload, bin=None, cluster=None):
            if bin is not None:
                d = (self._prev_bin != bin) if (self._started
                                                and self._prev_bin is not None) \
                    else None
                score = float(_cluster_win3(d)) if d is not None else 0.0
                changed = (score >= self._C) if d is not None else False
                rec["feed"].append((k, fi, score, changed))
            else:
                score = float(cluster)
                changed = score >= self._C
                rec["feed"].append((k, fi, score, changed))
            return super().feed(k, fi, sharp, payload, bin=bin,
                                cluster=cluster)

        @staticmethod
        def _frames_of(seg):
            return seg

    gp.SegmentStateMachine = RecMachine
    hp.SegmentStateMachine = RecMachine
    try:
        from video_ocr_engine import FieldExtractor
        ex = FieldExtractor(video, roi, frame_end=frames, sample_stride=1,
                            decode_backend="nvdec", ocr_backend="auto",
                            keep_frames=True)
        res = ex.extract()
        rec["segs"] = [(s[0], s[-1]) for s in
                       (seg if isinstance(seg, (list, tuple)) else [seg])
                       for seg in res.segments] if False else None
        rec["n_segs"] = len(res.segments)
        rec["th"] = ex._bin_thresh
        # 段边界直接从 machine 无法拿（内部对象），用 res.segments 的帧区间
        try:
            rec["bounds"] = [(s.frame_start, s.frame_end) for s in res.segments]
        except Exception:
            rec["bounds"] = None
    finally:
        gp.SegmentStateMachine = SegmentStateMachine
        hp.SegmentStateMachine = SegmentStateMachine
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5", choices=sorted(VIDS))
    ap.add_argument("--frames", type=int, default=3000)
    args = ap.parse_args()
    video, roi = VIDS[args.video]

    g = trace_pipeline(video, roi, args.frames, "1")
    h = trace_pipeline(video, roi, args.frames, "0")
    print(f"gpu : segs={g['n_segs']} th={g['th']} emits={len(g['emit'])} "
          f"sims={len(g['similar'])}")
    print(f"host: segs={h['n_segs']} th={h['th']} emits={len(h['emit'])} "
          f"sims={len(h['similar'])}")

    # 校准窗内断段（前 50 帧）
    calib_n = 50
    g_calib_breaks = sum(1 for (k, fi, s, c) in g["feed"][:calib_n]
                         if c and k > 0)
    h_calib_breaks = sum(1 for (k, fi, s, c) in h["feed"][:calib_n]
                         if c and k > 0)
    print(f"校准窗(0..49)断段: gpu={g_calib_breaks} host={h_calib_breaks}")

    # 逐帧 changed 对比
    n = min(len(g["feed"]), len(h["feed"]))
    diffs = []
    for i in range(n):
        if g["feed"][i][3] != h["feed"][i][3]:
            diffs.append(i)
    print(f"changed 分歧帧数: {len(diffs)}（前 {n} 帧对齐比较）")
    for i in diffs[:8]:
        print(f"  k={i}: gpu(score={g['feed'][i][2]:.0f},"
              f"changed={g['feed'][i][3]}) host(score={h['feed'][i][2]:.0f},"
              f"changed={h['feed'][i][3]})")

    # similar 判定分歧
    m = min(len(g["similar"]), len(h["similar"]))
    sim_diffs = [i for i in range(m) if g["similar"][i] != h["similar"][i]]
    print(f"similar 判定分歧: {len(sim_diffs)}（公共长度 {m}，"
          f"长度 {len(g['similar'])} vs {len(h['similar'])}）")
    for i in sim_diffs[:5]:
        print(f"  #{i}: gpu={g['similar'][i]} host={h['similar'][i]}")


if __name__ == "__main__":
    main()
