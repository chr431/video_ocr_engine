"""hybrid OCR（双车道 TRT+OpenVINO）A/B 与文本对照（2026-09-20）。

问题（用户）：OCR-bound 场景（batch_test 稠密字幕 stride=1）双 OCR 车道
相对单 TRT 的收益。v0 裁决：**回退 +51%**——双引擎放弃 raw 设备直通后，
串行 worker 的宿主预处理（resize 1.37ms/段，占 prep 86%）成为新瓶颈
（worker 98% 忙、两车道挨饿）。正确性面：段数恒等、文本 26461 段仅
3 段不一致（形近字）。v0.5 设计（设备 prep 共享 + D2H 已预处理张量）
见叙事；天花板 = decode-bound 地板。

用法：
  python tools/_probe_hybrid_ocr.py [--video 新三国01] [--frames 0=全片]
      [--rounds 2]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="新三国01")
    ap.add_argument("--frames", type=int, default=0, help="0=全片")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    bdir = os.environ.get("RACELOG_BATCH_DIR", r"D:\Videos\batch_test")
    path = str(Path(bdir) / (args.video + ".mkv"))
    roi = (144, 398, 551, 423)      # batch_params.txt 口径

    import logging
    logging.disable(logging.INFO)
    import warnings
    warnings.filterwarnings("ignore")
    from video_ocr_engine import FieldExtractor

    def once(backend):
        ex = FieldExtractor(path, roi, decode_backend="hybrid",
                            ocr_backend=backend, sample_stride=1,
                            **({"frame_end": args.frames} if args.frames else {}))
        t0 = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t0
        sp = (r.meta.get("report") or {}).get("spans", {})

        def s(k):
            return sp.get(k, {}).get("sum", 0)
        return dict(wall=wall, segs=len(r.segments),
                    txt=[x.text or "" for x in r.segments],
                    infer=s("ocr.infer"), decode=s("decode.batch"),
                    pre=s("ocr.preprocess"), backend=ex._ocr_backend_used)

    data = {"tensorrt": [], "hybrid": []}
    first = {}
    for rnd in range(args.rounds):
        for b in ("tensorrt", "hybrid"):
            d = once(b)
            data[b].append(d)
            first.setdefault(b, d)
            print("轮%d %-9s wall=%.1fs segs=%d infer=%.1f decode=%.1f pre=%.1f (%s)"
                  % (rnd + 1, b, d["wall"], d["segs"], d["infer"], d["decode"],
                     d["pre"], d["backend"]), flush=True)
    print("\n中位：单 TRT %.1fs | 双车道 %.1fs（%+.1f%%）"
          % (statistics.median([d["wall"] for d in data["tensorrt"]]),
             statistics.median([d["wall"] for d in data["hybrid"]]),
             (statistics.median([d["wall"] for d in data["hybrid"])
              - statistics.median([d["wall"] for d in data["tensorrt"]))
             / statistics.median([d["wall"] for d in data["tensorrt"]]) * 100))
    a, b = first["tensorrt"], first["hybrid"]
    if a["segs"] == b["segs"]:
        diff = [i for i, (x, y) in enumerate(zip(a["txt"], b["txt"])) if x != y]
        print("文本对照：段数 %d 一致，不一致 %d 段" % (a["segs"], len(diff)))
        for i in diff[:5]:
            print("  seg%d: TRT=%r 双=%r" % (i, a["txt"][i], b["txt"][i]))
    else:
        print("⚠️ 段数不一致：", a["segs"], b["segs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
