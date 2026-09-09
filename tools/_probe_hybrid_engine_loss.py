"""H2/H3：hybrid 引擎内流失定位与修复验证（对照 2601fps 纯解码上限）。

三组：基线（fill_prev 已收拢）/ GPU_PIPELINE_STREAM=1 / 解码批 128。
每格 2 跑取最快；记录 wall/decode/stream_analyze/尾相 + sha 门禁。
用法：python tools/_probe_hybrid_engine_loss.py [--video test6]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run(video, roi, batch=None, stream=None, runs=2, frames=3000):
    import engine_config as config
    from video_ocr_engine import FieldExtractor
    saved_batch = config.GPU_PIPELINE_DECODE_BATCH
    if batch is not None:
        config.GPU_PIPELINE_DECODE_BATCH = int(batch)
    old = {}
    if stream is not None:
        old["GPU_PIPELINE_STREAM"] = os.environ.get("GPU_PIPELINE_STREAM")
        os.environ["GPU_PIPELINE_STREAM"] = str(stream)
    walls, shas, segs = [], [], []
    prof = {}
    try:
        for i in range(runs):
            ex = FieldExtractor(video, roi, frame_end=frames,
                                decode_backend="hybrid", ocr_backend="auto",
                                keep_frames=True)
            t0 = time.perf_counter()
            res = ex.extract()
            walls.append(time.perf_counter() - t0)
            texts = {s.text for s in res.segments if s.text}
            shas.append(hashlib.sha1(
                "\n".join(sorted(texts)).encode()).hexdigest()[:12])
            segs.append(len(res.segments))
            prof = {k: dict(v) for k, v in ex.profile.items()}
    finally:
        config.GPU_PIPELINE_DECODE_BATCH = saved_batch
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    t = res.timing
    p = prof.get("producer", {})
    dec = t.get("decode", 0)
    fps = frames / dec if dec else 0
    print(f"  batch={batch or config.GPU_PIPELINE_DECODE_BATCH} stream={stream}: "
          f"wall={min(walls):.3f} decode={dec:.3f}({fps:.0f}fps) "
          f"analyze={p.get('stream_analyze', 0):.3f} "
          f"tail={t.get('ocr_tail', 0):.3f} segs={segs} sha={shas}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test6.mp4")
    ap.add_argument("--roi", default="841,994,949,1026")
    args = ap.parse_args()
    roi = tuple(int(v) for v in args.roi.split(","))
    print("== hybrid 引擎内对照（av1 上限 2601fps）==", flush=True)
    run(args.video, roi)
    run(args.video, roi, stream=1)
    run(args.video, roi, batch=128)
    run(args.video, roi, batch=128, stream=1)
