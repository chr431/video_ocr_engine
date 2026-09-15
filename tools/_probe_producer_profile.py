"""宿主生产者缺口定位：cProfile 找出 producer(2.05s) 里未打桩的 0.46~0.60s。

背景：`_probe_span_dump.py` 实测 h264-cpu 热轮
    pipeline.consumer(生产者总墙钟) 2.05s
    已计 span: decode.batch 1.35 / gray 0.05 / sharp 0.09 / bin 0.006 /
               calib 0.07 / open 0.02  ≈ 1.59s
    **缺口 ≈ 0.46s（156 µs/帧）**
`_probe_feed_cost.py` 证明分段状态机的逐帧成本只占 3%（0.018s）——
所以缺口在别处，只能靠 profiler 定位，不能靠猜。

本探针用 cProfile 跑**单次 extract（热池）**，按 tottime 排序打印主线程
热点（只列 video_ocr_engine 内的帧 + 少量 numpy 顶层），并特别报告
`segments_similar` / `SegmentStateMachine.feed` / `_emit_ocr` 的累计时间。

用法：
  python tools/_probe_producer_profile.py [--video test5] [--frames 3000]
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VID = {"test5": ("test5.mp4", (843, 993, 948, 1025)),
       "test6_hevc": ("test6_hevc.mp4", (841, 994, 949, 1026))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--decode", default="cpu")
    ap.add_argument("--ocr", default="cpu")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    name, roi = VID[args.video]
    path = str(Path(os.environ.get("RACELOG_VIDEO_DIR",
                                   r"D:\Videos\racelog_test")) / name)
    from video_ocr_engine import FieldExtractor

    # 先热身一轮（冷启动建引擎/内核不计入 profile）
    ex = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                        decode_backend=args.decode, ocr_backend=args.ocr,
                        keep_crops=False)
    r = ex.extract()
    print("热身：段数 %d" % len(r.segments))

    ex = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                        decode_backend=args.decode, ocr_backend=args.ocr,
                        keep_crops=False)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    r = ex.extract()
    pr.disable()
    wall = time.perf_counter() - t0
    print("profile 轮：墙钟 %.4fs  段数 %d" % (wall, len(r.segments)))

    st = pstats.Stats(pr)
    buf = io.StringIO()
    st.stream = buf
    st.sort_stats("tottime").print_stats(args.top)
    txt = buf.getvalue()
    print("\n=== tottime top %d ===" % args.top)
    keep = False
    shown = 0
    for line in txt.splitlines():
        if line.strip().startswith("ncalls"):
            keep = True
        if keep:
            print(line)
            shown += 1
            if shown > args.top + 6:
                break

    # 关注项（cumtime）
    print("\n=== 关注项（cumulative）===")
    for key in ("segments_similar", "feed", "similar_decision", "_cluster_win3",
                "_emit_ocr", "crop_luma", "batch_luma", "get_batch",
                "asnumpy", "copy", "preprocess_standard"):
        for func, stat in st.stats.items():
            cc, nc, tt, ct, _ = stat
            if key in func[2]:
                print("%-28s %-30s ncalls=%-8d tottime=%6.3f cumtime=%6.3f"
                      % (key, func[0][-34:], nc, tt, ct))
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
