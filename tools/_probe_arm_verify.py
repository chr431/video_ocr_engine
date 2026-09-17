"""替换臂有效性核实：确认 cv2 补丁**真的被调用**（而非静默回落 legacy）。

`_probe_numpy_replace_ab.py` 的 arm B/C 用 monkeypatch 换内核，且**指纹
与基线逐位相同**——这既可能是"换了且等价"（期望），也可能是"根本没换"
（假阴性），两者指纹表现一致。项目已有此类事故先例
（`_probe_ov_prep_fusion.py` 的静默回落）。

本探针在 arm B/C 的补丁函数里加计数器，跑一次热 extract，打印：
  · 每个 cv2 替换函数的调用次数与累计耗时；
  · 若次数为 0 → 该补丁未生效，对应 A/B 结论作废。

用法：
  python tools/_probe_arm_verify.py [--config h264-cpu] [--arm B] [--frames 3000]
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
       "hevc-cpu": ("test6_hevc.mp4", (841, 994, 949, 1026), "cpu", "cpu"),
       "h264-gpu": ("test5.mp4", (843, 993, 948, 1025), "nvdec", "tensorrt")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="h264-cpu")
    ap.add_argument("--arm", default="B", choices=["B", "C"])
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--roi", default="", help="覆盖 ROI（宽 ROI 场景）")
    args = ap.parse_args()

    import cv2
    import numpy as np
    import video_ocr_engine.domain.segmentation as seg
    import video_ocr_engine.extractor as ex_mod
    import video_ocr_engine.domain.video_utils as vu
    import video_ocr_engine.ocr.native as nat

    calls: dict = {}

    def counted(name, fn):
        def inner(*a, **k):
            c = calls.setdefault(name, {"n": 0, "s": 0.0})
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                c["n"] += 1
                c["s"] += time.perf_counter() - t
        return inner

    # ── 生产者侧 cv2（与 _probe_numpy_replace_ab.py 的 arm B 同实现）──
    def _box3(s, ddepth):
        padded = cv2.copyMakeBorder(s, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        w3 = cv2.boxFilter(padded, ddepth=ddepth, ksize=(3, 3),
                           normalize=False, borderType=cv2.BORDER_ISOLATED)
        return w3[1:-1, 1:-1]

    def cv_cluster_win3(diff):
        if not diff.any():
            return 0.0
        s = (diff.view(np.uint8) if diff.flags.c_contiguous
             else diff.astype(np.uint8))
        return float(_box3(s, cv2.CV_16U).max())

    def cv_text_sep_binary(gray, th):
        g8 = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
        return cv2.compare(g8, int(th), cv2.CMP_GT).astype(np.float32)

    def cv_segments_similar(self, a, b):
        _text_mode = self._merge_effective_mode()
        if _text_mode == 'binary':
            a = cv_text_sep_binary(a, self._bin_thresh)
            b = cv_text_sep_binary(b, self._bin_thresh)
        if a is None or b is None or a.shape != b.shape:
            return False
        d = cv2.absdiff(a.astype(np.int16), b.astype(np.int16))
        mean = float(cv2.mean(d)[0])
        changed = int(cv2.countNonZero(cv2.compare(d, 10, cv2.CMP_GT)))
        _dense = seg.dense_gate_hit(cv_cluster_win3(d > 10),
                                    self._merge_dense_gate)
        return seg.similar_decision(mean, changed,
                                    self._merge_similar_threshold,
                                    self._merge_max_changed_pixels,
                                    dense=_dense)

    def _cv2_resize(img, new_w, new_h):
        src_h, src_w = img.shape[:2]
        if new_w == src_w and new_h == src_h:
            return img.astype(np.float32)
        f = np.ascontiguousarray(img.astype(np.float32))
        out = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return out[..., None] if img.ndim == 3 and out.ndim == 2 else out

    seg._cluster_win3 = counted("CV2_cluster_win3", cv_cluster_win3)
    ex_mod._text_sep_binary = counted("CV2_text_sep_binary", cv_text_sep_binary)
    ex_mod.FieldExtractor._segments_similar = counted(
        "CV2_segments_similar", cv_segments_similar)
    if args.arm == "C":
        for m in (vu, seg, nat):
            if hasattr(m, "_np_resize"):
                m._np_resize = counted("CV2_resize", _cv2_resize)

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
        c["n"] = 0
        c["s"] = 0.0

    ex = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                        decode_backend=dec, ocr_backend=ocr, keep_crops=False)
    t0 = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t0

    print("arm=%s  %s  热轮墙钟 %.4fs  段数 %d\n"
          % (args.arm, args.config, wall, len(r.segments)))
    print("%-26s %8s %10s %10s" % ("cv2 替换函数", "调用次数", "累计s", "占墙钟%"))
    tot = 0.0
    for k in sorted(calls, key=lambda x: -calls[x]["s"]):
        c = calls[k]
        tot += c["s"]
        print("%-26s %8d %10.4f %9.2f%%" % (k, c["n"], c["s"], c["s"] / wall * 100))
    print("%-26s %8s %10.4f %9.2f%%" % ("合计", "", tot, tot / wall * 100))

    zero = [k for k, c in calls.items() if c["n"] == 0]
    if zero:
        print("\n✗ **补丁未生效**（未被执行）：%s —— 该臂结论作废" % ", ".join(zero))
        return 1
    print("\n✓ 全部 cv2 补丁均被执行（替换臂结论有效）")

    out = ROOT / "bench" / "arm_verify.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"arm": args.arm, "config": args.config,
                               "wall": wall, "segs": len(r.segments),
                               "calls": calls}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
