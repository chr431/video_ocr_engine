"""生产者侧 numpy 成本的**非嵌套**分解 + 吸收机制定位。

## 为什么要这个探针
`_probe_patch_verify.py` 的初版把 `_text_sep_binary`（0.088s）与
`_segments_similar`（0.255s）**并列相加**得到 0.53s——但
`_text_sep_binary` 是**在 `_segments_similar` 内部被调用**的（见
`extractor._segments_similar` L379-380），所以那是**重复计数**。
真实生产者 numpy 成本 ≈ `_segments_similar` + `_cluster_win3`
（后者既被 feed 调用，也被 `_segments_similar` 的稠密簇门调用）。

本探针按**调用栈归属**统计：分开记 feed 路径的 cluster 与
similar 路径的 cluster，并把 `_text_sep_binary` 标为 nested（不并入合计）。

## 吸收机制
`_probe_producer_binding2.py` 双侧结论：
  · 生产者 numpy ×5（+2.1s 成本）→ 墙钟仅 **+5.8%**（decode.batch 从
    1.37s 掉到 0.53s → 生产者被抢占，OCR 消费者反而更饿）；
  · 消费者预处理 ×5 → 墙钟 +5.1%（同为吸收）。
即两侧都有 ~90% 的余量被**对方的等待**吸收。本探针量化该余量：
输出 `q_get_wait`（消费者空等）与 producer 的差值。

用法：
  python tools/_probe_numpy_cost_split.py [--config h264-cpu] [--frames 3000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VID = {"h264-cpu": ("test5.mp4", (843, 993, 948, 1025), "cpu", "cpu"),
       "hevc-cpu": ("test6_hevc.mp4", (841, 994, 949, 1026), "cpu", "cpu")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="h264-cpu")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--roi", default="")
    args = ap.parse_args()

    import numpy as np
    import video_ocr_engine.domain.segmentation as seg
    import video_ocr_engine.extractor as ex_mod

    calls: dict = {}
    depth = {"sim": 0}

    def counted(name, fn):
        def inner(*a, **k):
            c = calls.setdefault(name, {"n": 0, "s": 0.0, "nested_n": 0,
                                        "nested_s": 0.0})
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                dt = time.perf_counter() - t
                c["n"] += 1
                c["s"] += dt
                if depth["sim"] > 0:      # 归属：出现在 sim 内部
                    c["nested_n"] += 1
                    c["nested_s"] += dt
        return inner

    _orig_cluster = seg._cluster_win3
    _orig_sep = ex_mod._text_sep_binary
    _orig_sim = ex_mod.FieldExtractor._segments_similar

    seg._cluster_win3 = counted("cluster_win3", _orig_cluster)
    ex_mod._text_sep_binary = counted("text_sep_binary", _orig_sep)

    def sim_wrapper(self, a, b):
        depth["sim"] += 1
        try:
            return _orig_sim(self, a, b)
        finally:
            depth["sim"] -= 1

    ex_mod.FieldExtractor._segments_similar = counted("segments_similar",
                                                      sim_wrapper)

    vid, roi, dec, ocr = VID[args.config]
    if args.roi:
        roi = tuple(int(x) for x in args.roi.split(","))
    path = str(Path(os.environ.get("RACELOG_VIDEO_DIR",
                                   r"D:\Videos\racelog_test")) / vid)
    from video_ocr_engine import FieldExtractor

    ex = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                        decode_backend=dec, ocr_backend=ocr, keep_crops=False)
    ex.extract()
    for c in calls.values():
        for k in ("n", "s", "nested_n", "nested_s"):
            c[k] = 0.0

    ex = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                        decode_backend=dec, ocr_backend=ocr, keep_crops=False)
    t0 = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t0
    rep = r.meta.get("report") or {}
    sp, ga = rep.get("spans", {}), rep.get("gauges", {})
    prod = (sp.get("pipeline.consumer") or {}).get("sum") or 0
    wait = ga.get("pipeline.q_get_wait") or 0
    decb = (sp.get("decode.batch") or {}).get("sum") or 0

    print("%s  ROI=%s  热轮墙钟 %.4fs  段数 %d"
          % (args.config, roi, wall, len(r.segments)))
    print("producer(consumer)=%.4f  decode.batch=%.4f  消费者空等 q_get_wait=%.4f"
          % (prod, decb, wait))
    print("\n%-20s %8s %9s %9s   %s" % ("函数", "次数", "总s", "其中嵌套s", "说明"))
    for k, c in sorted(calls.items(), key=lambda kv: -kv[1]["s"]):
        note = ("**嵌套在 segments_similar 内 → 不并入合计**"
                if c["nested_n"] == c["n"] and c["n"] else "")
        print("%-20s %8d %9.4f %9.4f   %s" % (k, c["n"], c["s"],
                                              c["nested_s"], note))

    # 非嵌套合计：segments_similar（含其内部的 cluster/sep）+ feed 路径的 cluster
    sim = calls.get("segments_similar", {"s": 0.0})
    clus = calls.get("cluster_win3", {"s": 0.0, "nested_s": 0.0})
    clus_outside = clus["s"] - clus["nested_s"]
    total = sim["s"] + clus_outside
    print("\n非嵌套生产者 numpy 合计 = segments_similar %.4f + "
          "feed 路径 cluster %.4f = **%.4f s（占墙钟 %.1f%%）**"
          % (sim["s"], clus_outside, total, total / wall * 100))
    print("（初版 0.53s 系把嵌套的 text_sep_binary 重复计入；正确值见上）")

    # 吸收余量
    print("\n吸收余量：producer %.3f > decode.batch %.3f，差 %.3f 由"
          "「喂给消费者的等待/调度」构成；消费者侧空等 %.3f"
          % (prod, decb, prod - decb, wait))

    out = ROOT / "bench" / "numpy_cost_split.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"config": args.config, "roi": list(roi), "wall": wall,
         "segs": len(r.segments), "producer": prod, "decode_batch": decb,
         "q_get_wait": wait, "calls": calls,
         "non_nested_total": total,
         "non_nested_pct_of_wall": total / wall * 100},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
