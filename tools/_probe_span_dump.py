"""全 span 细目转储：热池单次 extract 的完整报告（找时间到底去了哪）。

`_probe_critical_path.py` 判定「解码非绑定（纯解码 1.24s vs 墙钟 2.28s）
但 OCR 消费者却空等 1.46s」——两个结论看似矛盾，必须看**完整 span 表**
才能定位（生产者内部某一段吃掉了差额）。本探针只做一件事：把热轮报告的
spans/counters/gauges 全表打印并按 sum 降序排列。

用法：
  python tools/_probe_span_dump.py [--video test5] [--frames 3000] [--ocr cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VID = {"test5": ("test5.mp4", (843, 993, 948, 1025)),
       "test6_hevc": ("test6_hevc.mp4", (841, 994, 949, 1026)),
       "test6_av1": ("test6.mp4", (841, 994, 949, 1026))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--decode", default="cpu")
    ap.add_argument("--ocr", default="cpu")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    name, roi = VID[args.video]
    path = str(Path(os.environ.get("RACELOG_VIDEO_DIR",
                                   r"D:\Videos\racelog_test")) / name)
    from video_ocr_engine import FieldExtractor

    rep = None
    wall = 0.0
    for _ in range(args.rounds):
        ex = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                            decode_backend=args.decode, ocr_backend=args.ocr,
                            keep_crops=False)
        t0 = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t0
        rep = r.meta.get("report") or {}
    segs = len(r.segments)

    print("热轮墙钟 %.4fs  段数 %d  decode=%s ocr=%s"
          % (wall, segs, args.decode, args.ocr))
    spans = rep.get("spans", {})
    print("\n%-34s %9s %6s %9s %9s %9s" % (
        "span", "sum(s)", "n", "min", "p50", "max"))
    for k, v in sorted(spans.items(), key=lambda kv: -(kv[1].get("sum") or 0)):
        print("%-34s %9.4f %6.0f %9.5f %9.5f %9.5f" % (
            k, v.get("sum") or 0, v.get("n") or 0, v.get("min") or 0,
            v.get("p50") or 0, v.get("max") or 0))

    print("\ncounter / gauge：")
    for k, v in sorted((rep.get("counters") or {}).items()):
        print("  C %-30s %s" % (k, v))
    for k, v in sorted((rep.get("gauges") or {}).items()):
        print("  G %-30s %s" % (k, v))

    out = Path(__file__).resolve().parents[1] / "bench" / "span_dump.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"video": name, "wall": wall, "segs": segs,
                               "decode": args.decode, "ocr": args.ocr,
                               "report": rep}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
