"""替换臂的**执行核实**：确认 monkeypatch 真的被调用（防止假阴性/假阳性）。

项目教训（`_probe_ov_prep_fusion.py`）：猴子补丁静默回落 legacy 会产生
**假胜利**（"胜利"与"miss 集全等"实为 legacy 对 legacy 噪声）。本轮的
风险是反向的——补丁没生效会产生**假阴性**（"换了没收益"其实是没换）。

所以任何 A/B 之前必须断言：补丁函数**真的被调用**，且调用次数与预期量级相符。

本探针在一个进程内装补丁 + 计数器，跑一次热 extract，报告：
  · 每个替换函数的调用次数与**内部累计耗时**；
  · 若调用次数为 0 → 补丁未生效（结论无效）。

用法：
  python tools/_probe_patch_verify.py [--config h264-cpu] [--frames 3000]
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
    ap.add_argument("--arm", default="B")
    args = ap.parse_args()

    import numpy as np
    import cv2
    import video_ocr_engine.domain.segmentation as seg
    import video_ocr_engine.extractor as ex_mod
    import video_ocr_engine.domain.video_utils as vu
    import video_ocr_engine.ocr.native as nat

    calls: dict = {}

    def wrap(name, fn):
        def inner(*a, **k):
            c = calls.setdefault(name, {"n": 0, "s": 0.0})
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                c["n"] += 1
                c["s"] += time.perf_counter() - t
        return inner

    # ── 基线（也计数，用于对比 numpy 版自身耗时）──
    base_cluster = wrap("BASE_cluster_win3", seg._cluster_win3)
    base_sep = wrap("BASE_text_sep_binary", ex_mod._text_sep_binary)
    base_sim = wrap("BASE_segments_similar",
                    ex_mod.FieldExtractor._segments_similar)
    base_resize = wrap("BASE_np_resize", vu._np_resize)
    seg._cluster_win3 = base_cluster
    ex_mod._text_sep_binary = base_sep
    # 实例方法：包一层（注意 self 由 __get__ 传入）
    ex_mod.FieldExtractor._segments_similar = (
        lambda self, a, b: base_sim(self, a, b))
    _orig_resize = vu._np_resize
    _wrapped_resize = wrap("BASE_np_resize_used", _orig_resize)
    for m in (vu, seg, nat):
        if hasattr(m, "_np_resize"):
            m._np_resize = _wrapped_resize

    vid, roi, dec, ocr = VID[args.config]
    path = str(Path(os.environ.get("RACELOG_VIDEO_DIR",
                                   r"D:\Videos\racelog_test")) / vid)
    from video_ocr_engine import FieldExtractor

    # 热一轮（不计）
    ex = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                        decode_backend=dec, ocr_backend=ocr, keep_crops=False)
    ex.extract()
    for c in calls.values():
        c["n"] = 0
        c["s"] = 0.0

    ex = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                        decode_backend=dec, ocr_backend=ocr, keep_crops=False)
    t0 = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t0
    rep = r.meta.get("report") or {}
    sp = rep.get("spans", {})
    print("arm=%s  %s  热轮墙钟 %.4fs  段数 %d"
          % (args.arm, args.config, wall, len(r.segments)))
    print("报告内 spans: producer(consumer)=%.4f  preproc_resize=%.4f"
          % ((sp.get("pipeline.consumer") or {}).get("sum") or -1,
             (sp.get("ocr.preproc_resize") or {}).get("sum") or -1))
    print("\n%-28s %8s %10s %10s" % ("被替换的函数", "调用次数", "累计耗时s", "占墙钟%"))
    tot = 0.0
    for k in sorted(calls, key=lambda x: -calls[x]["s"]):
        c = calls[k]
        tot += c["s"]
        print("%-28s %8d %10.4f %9.2f%%"
              % (k, c["n"], c["s"], c["s"] / wall * 100))
    print("%-28s %8s %10.4f %9.2f%%" % ("合计", "", tot, tot / wall * 100))

    zero = [k for k, c in calls.items() if c["n"] == 0]
    if zero:
        print("\n⚠️ 未被执行（补丁可能未生效）：%s" % ", ".join(zero))
    else:
        print("\n✓ 全部打桩点均被执行（替换臂结论有效）")

    out = ROOT / "bench" / "patch_verify.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"arm": args.arm, "config": args.config,
                               "wall": wall, "segs": len(r.segments),
                               "calls": calls}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
