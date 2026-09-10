"""D1 调查第二刀：全程跑 gpu/host 两管线，记录状态机 trace 找首分歧帧。

对 SegmentStateMachine 打记录子类（feed 输入 + emit 事件 + similar 判定），
分别跑 GPU_PIPELINE=1/0 的 nvdec extract，diff 两条 trace：
  - 首个 emit 边界分歧的段号/帧号
  - 首个 similar 判定分歧
  - feed 输入（sharp/cluster/bin）首个分歧

用法：python tools/_probe_d1_trace.py [--video test5] [--frames 3000]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VIDS = {
    "test5": (r"D:\Videos\racelog_test\test5.mp4", (843, 993, 948, 1025)),
    "test6": (r"D:\Videos\racelog_test\test6.mp4", (841, 994, 949, 1026)),
    "test": (r"D:\Videos\racelog_test\test.mp4", (841, 994, 949, 1026)),
}


def trace_pipeline(video: str, roi: tuple, frames: int, gpu_pl: str) -> dict:
    import video_ocr_engine.gpu.device as gp
    import video_ocr_engine._host_pipeline as hp
    from segmentation import SegmentStateMachine
    os.environ["GPU_PIPELINE"] = gpu_pl

    rec: dict = {"feed": [], "emit": [], "similar": []}

    class RecMachine(SegmentStateMachine):
        def feed(self, k, fi, sharp, payload, bin=None, cluster=None):
            c = (float(cluster) if cluster is not None else None)
            rec["feed"].append((k, fi, round(float(sharp), 6), c,
                                None if bin is None else int(bin.sum())))
            return super().feed(k, fi, sharp, payload, bin=bin,
                                cluster=cluster)

    orig_on_similar_cls = SegmentStateMachine
    gp.SegmentStateMachine = RecMachine
    hp.SegmentStateMachine = RecMachine
    # 记录 similar 判定：包一层 ex._segments_similar 与 _similar_device
    try:
        from video_ocr_engine import FieldExtractor
        ex = FieldExtractor(video, roi, frame_end=frames, sample_stride=1,
                            decode_backend="nvdec", ocr_backend="auto",
                            keep_frames=True)
        _orig_sim = ex._segments_similar

        def _rec_sim(a, b):
            r = bool(_orig_sim(a, b))
            rec["similar"].append(r)
            return r
        ex._segments_similar = _rec_sim
        res = ex.extract()
        rec["segs"] = len(res.segments)
        rec["th"] = ex._bin_thresh
    finally:
        gp.SegmentStateMachine = orig_on_similar_cls
        hp.SegmentStateMachine = orig_on_similar_cls
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5", choices=sorted(VIDS))
    ap.add_argument("--frames", type=int, default=3000)
    args = ap.parse_args()
    video, roi = VIDS[args.video]

    g = trace_pipeline(video, roi, args.frames, "1")
    h = trace_pipeline(video, roi, args.frames, "0")
    print(f"gpu: segs={g['segs']} th={g['th']} feeds={len(g['feed'])} "
          f"emits={len(g['emit'])} sims={len(g['similar'])}")
    print(f"host: segs={h['segs']} th={h['th']} feeds={len(h['feed'])} "
          f"sims={len(h['similar'])}")

    # feed 输入对比
    n = min(len(g["feed"]), len(h["feed"]))
    first_feed_diff = None
    for i in range(n):
        if g["feed"][i] != h["feed"][i]:
            first_feed_diff = i
            break
    if first_feed_diff is None and len(g["feed"]) != len(h["feed"]):
        first_feed_diff = n
    print(f"feed 首分歧: {first_feed_diff}")
    if first_feed_diff is not None and first_feed_diff < n:
        i = first_feed_diff
        print(f"  gpu feed[{i}]={g['feed'][i]}")
        print(f"  host feed[{i}]={h['feed'][i]}")

    # similar 序列对比
    m = min(len(g["similar"]), len(h["similar"]))
    for i in range(m):
        if g["similar"][i] != h["similar"][i]:
            print(f"similar 首分歧: #{i} gpu={g['similar'][i]} "
                  f"host={h['similar'][i]}（前缀一致 {i} 条）")
            break
    else:
        print(f"similar 序列前 {m} 条一致"
              + ("" if len(g["similar"]) == len(h["similar"])
                 else f"（长度不同 {len(g['similar'])} vs {len(h['similar'])}）"))


if __name__ == "__main__":
    main()
