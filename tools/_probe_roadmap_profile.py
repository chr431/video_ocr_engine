"""路线图轮 M4：ENGINE_PROFILE 分相打印驱动（单配置一次 extract）。

bench_hybrid 不打印 ex.profile，本驱动直接构造 FieldExtractor 并输出
timing + ENGINE_PROFILE 全部分相（producer.*/ocr.* 组），用于分解
「OCR 消费端墙钟 > TRT 推理天花板」的差额去向（engine_init /
q_get_wait / preprocess / infer / ctc_decode / autocrop…）。

用法：
  ENGINE_PROFILE=1 python tools/_probe_roadmap_profile.py --video test6 \
      --roi 841,994,949,1026 --backend nvdec --pipeline gpu [--frames 3000]
DLL 口径由外部 DECORD_LIBRARY_PATH 控制。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--backend", default="nvdec",
                    choices=["nvdec", "cpu", "hybrid"])
    ap.add_argument("--pipeline", default="gpu", choices=["gpu", "host"])
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    if args.pipeline == "host":
        os.environ["GPU_PIPELINE"] = "0"

    from video_ocr_engine import FieldExtractor
    ex = FieldExtractor(args.video, tuple(int(v) for v in args.roi.split(",")),
                        frame_end=args.frames, sample_stride=args.stride,
                        decode_backend=args.backend, ocr_backend="auto",
                        keep_frames=True)
    t0 = time.perf_counter()
    res = ex.extract()
    wall = time.perf_counter() - t0
    print(f"wall={wall:.3f}s segs={len(res.segments)} "
          f"timing={ {k: round(v, 3) for k, v in res.timing.items()} }",
          flush=True)
    prof = getattr(ex, "profile", None)
    if prof:
        for group, d in sorted(prof.items()):
            for k, v in sorted(d.items()):
                print(f"  {group}.{k} = {v:.3f}", flush=True)
    else:
        print("  (ENGINE_PROFILE 未开启或无数据)", flush=True)


if __name__ == "__main__":
    main()
