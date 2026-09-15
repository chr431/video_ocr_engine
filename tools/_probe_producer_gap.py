"""宿主生产者内部缺口分解：consumer_total 与已计 span 之和的 0.56s 去哪了。

背景（`_probe_span_dump.py` 实测，h264-cpu 热轮）：
    pipeline.consumer(生产者总墙钟)  2.10s
    decode.batch                      1.35s
    decode.sharp_batch                0.09s
    decode.luma/binarize              0.01s
    producer.calib_total              0.07s
    producer.open_and_fps             0.02s
    ────────────────────────────────────────
    已计之和                          ~1.54s
    **缺口                          ~0.56s（190 µs/帧 × 2950 帧）**

缺口只可能来自**未打桩的分段/合并/emit**：
    · `SegmentStateMachine.feed`（`d = prev_bin != bin` + `_cluster_win3`）
    · `on_similar` → `segments_similar`（相似段合并判定）
    · `on_emit` → `_emit_ocr`（SegmentTask 构造 + 队列 put）

本探针按「微基准 × 实际调用次数」分解该缺口，并**用短路对照验证**
（把合并判定短路成"恒不相似"、把簇判定短路成常量，看生产者墙钟是否真降）
——照 C-42 判例：不在关键路径的东西，短路它墙钟不动。

用法：
  python tools/_probe_producer_gap.py [--video test5] [--frames 3000]
      [--repeat 3]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VID = {"test5": ("test5.mp4", (843, 993, 948, 1025)),
       "test6_hevc": ("test6_hevc.mp4", (841, 994, 949, 1026))}

WORKER = r'''
import json, os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
import numpy as np

ARM = os.environ["PROBE_ARM"]

if ARM == "nocluster":
    # 簇判定短路：返回常量（分段判据退化为"永不变化"）
    import video_ocr_engine.domain.segmentation as seg
    seg._cluster_win3 = lambda diff: 9.0

if ARM == "nomerge":
    # 合并判定短路：恒不相似（段数应上升，验证真的走到了）
    import video_ocr_engine.domain.segmentation as seg
    seg.similar_decision = lambda *a, **k: False

from video_ocr_engine import FieldExtractor

vid = os.path.join(os.environ["RACELOG_VIDEO_DIR"], os.environ["PROBE_VIDEO"])
roi = tuple(int(x) for x in os.environ["PROBE_ROI"].split(","))
outs = []
for i in range(int(os.environ.get("PROBE_ROUNDS", "2"))):
    ex = FieldExtractor(vid, roi, frame_start=0,
                        frame_end=int(os.environ["PROBE_FRAMES"]),
                        decode_backend=os.environ["PROBE_DEC"],
                        ocr_backend=os.environ["PROBE_OCR"], keep_crops=False)
    t = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t
    rep = r.meta.get("report") or {}
    sp, ga = rep.get("spans", {}), rep.get("gauges", {})
    def g(k):
        return (sp.get(k) or {}).get("sum")
    outs.append({"wall": wall, "segs": len(r.segments),
                 "producer": g("pipeline.consumer"), "ocr": g("pipeline.ocr"),
                 "decode_batch": g("decode.batch"),
                 "sharp": g("decode.sharp_batch"),
                 "calib": g("producer.calib_total"),
                 "open": g("producer.open_and_fps"),
                 "infer": g("ocr.infer"), "preproc": g("ocr.preprocess"),
                 "q_get_wait": ga.get("pipeline.q_get_wait")})
print("PROBE_JSON " + json.dumps(outs))
'''


def one(arm: str, cfg: str, frames: int) -> list:
    import subprocess
    vid, roi, dec, ocr = VIDS[cfg]
    env = dict(os.environ)
    env.update({"PROBE_ROOT": str(ROOT), "PROBE_ARM": arm,
                "PROBE_VIDEO": vid, "PROBE_ROI": ",".join(map(str, roi)),
                "PROBE_DEC": dec, "PROBE_OCR": ocr,
                "PROBE_FRAMES": str(frames), "PROBE_ROUNDS": "2",
                "RACELOG_VIDEO_DIR": env.get("RACELOG_VIDEO_DIR",
                                             r"D:\Videos\racelog_test")})
    p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    raise SystemExit("worker(%s/%s) 失败：\n%s\n%s"
                     % (arm, cfg, p.stdout[-1200:], p.stderr[-1500:]))


VIDS = {"h264-cpu": ("test5.mp4", (843, 993, 948, 1025), "cpu", "cpu"),
        "hevc-cpu": ("test6_hevc.mp4", (841, 994, 949, 1026), "cpu", "cpu")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="h264-cpu")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=4.0)
    args = ap.parse_args()

    res: dict = {}
    for cfg in args.config.split(","):
        res[cfg] = {}
        for arm in ("base", "nocluster", "nomerge"):
            hot = []
            for _ in range(args.repeat):
                hot.append(one(arm, cfg, args.frames)[-1])
                time.sleep(args.cooldown)
            med = {k: statistics.median([h[k] or 0 for h in hot])
                   for k in ("wall", "producer", "ocr", "decode_batch",
                             "sharp", "calib", "open", "infer", "preproc",
                             "q_get_wait")}
            med["segs"] = hot[-1]["segs"]
            res[cfg][arm] = med

        b = res[cfg]["base"]
        accounted = (b["decode_batch"] + b["sharp"] + b["calib"] + b["open"])
        print("\n== %s ==" % cfg)
        print("  base: wall %.4f  producer %.4f  已计 %.4f  **缺口 %.4f (%.0f%%)**"
              % (b["wall"], b["producer"], accounted,
                 b["producer"] - accounted,
                 (b["producer"] - accounted) / b["producer"] * 100))
        print("  %-11s %9s %9s %9s %9s %8s" % (
            "arm", "wall", "producer", "缺口", "segs", "Δwall"))
        for arm in ("base", "nocluster", "nomerge"):
            r = res[cfg][arm]
            acc = r["decode_batch"] + r["sharp"] + r["calib"] + r["open"]
            print("  %-11s %9.4f %9.4f %9.4f %9d %+7.2f%%" % (
                arm, r["wall"], r["producer"], r["producer"] - acc, r["segs"],
                (r["wall"] - b["wall"]) / b["wall"] * 100))

    out = ROOT / "bench" / "producer_gap.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
