"""关键路径判定：当前最快配置（h264-cpu）的绑定约束是解码还是 OCR。

方法论（照 C-42 判例）：**不按 span 占比推算**，而是做「拆解对照」——
分别测①纯解码产能 ②全管线墙钟 ③生产者的队列阻塞时间，用三者的关系
判定谁在关键路径上。

  · 若 decode_only ≈ wall  → 解码绑定（OCR 有等待余量）
  · 若 decode_only << wall 且 q_put_block 大 → OCR 绑定（解码被背压）

用法：
  python tools/_probe_critical_path.py [--video test5.mp4] [--frames 3000]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VID = {"test5": ("test5.mp4", (843, 993, 948, 1025)),
       "test6_hevc": ("test6_hevc.mp4", (841, 994, 949, 1026))}


def decode_only(video: str, roi, frames: int, decode: str, repeats: int) -> dict:
    """纯解码产能：decord 直接按生产口径（ROI + uint8 gray）扫全片。"""
    os.environ.setdefault("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
    import numpy as np
    from decord import VideoReader, cpu
    t = []
    nbytes = 0
    for _ in range(repeats):
        vr = VideoReader(video, ctx=cpu(0), output_format="gray",
                         width=roi[2] - roi[0], height=roi[3] - roi[1],
                         roi=roi)
        t0 = time.perf_counter()
        got = 0
        for s in range(0, min(frames, len(vr)), 64):
            b = vr.get_batch(list(range(s, min(s + 64, min(frames, len(vr))))))
            a = b.asnumpy()
            got += a.shape[0]
            nbytes += int(a.nbytes)
        t.append(time.perf_counter() - t0)
        del vr
    return {"decode_only_s": round(min(t), 4), "frames": got,
            "fps": round(got / min(t), 1)}


def full_pipeline(video: str, roi, frames: int, decode: str,
                  ocr: str, repeats: int) -> dict:
    """全管线（生产口径）：两轮取热轮，回报 span 细目。"""
    from video_ocr_engine import FieldExtractor
    outs = []
    for _ in range(repeats):
        ex = FieldExtractor(video, roi, frame_start=0, frame_end=frames,
                            decode_backend=decode, ocr_backend=ocr,
                            keep_crops=False)
        t0 = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t0
        rep = r.meta.get("report") or {}
        sp = rep.get("spans", {})
        ga = rep.get("gauges", {})
        outs.append({
            "wall": round(wall, 4), "segs": len(r.segments),
            "consumer_total": (sp.get("pipeline.consumer") or {}).get("sum"),
            "decode_phase": (sp.get("pipeline.decode") or {}).get("sum"),
            "ocr_phase": (sp.get("pipeline.ocr") or {}).get("sum"),
            "ocr_tail": (sp.get("pipeline.ocr_tail") or {}).get("sum"),
            "q_put_block": ga.get("pipeline.q_put_block"),
            "q_get_wait": ga.get("pipeline.q_get_wait"),
            "preproc": (sp.get("ocr.preprocess") or {}).get("sum"),
            "infer": (sp.get("ocr.infer") or {}).get("sum"),
            "init": ga.get("ocr.engine_init"),
        })
    hot = outs[-1]
    return {"hot": hot, "all": outs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ocr", default="cpu")
    args = ap.parse_args()

    name, roi = VID[args.video]
    path = str(Path(os.environ.get("RACELOG_VIDEO_DIR",
                                   r"D:\Videos\racelog_test")) / name)

    dec = decode_only(path, roi, args.frames, "cpu", args.repeats)
    print("① 纯解码产能：%.4fs（%.1f fps，%d 帧）" % (
        dec["decode_only_s"], dec["fps"], dec["frames"]))

    full = full_pipeline(path, roi, args.frames, "cpu", args.ocr, args.repeats)
    h = full["hot"]
    print("② 全管线热轮墙钟：%.4fs（段数 %d）" % (h["wall"], h["segs"]))
    for k in ("consumer_total", "decode_phase", "ocr_phase", "ocr_tail",
              "preproc", "infer", "q_put_block", "q_get_wait", "init"):
        print("     %-16s %s" % (k, h[k]))
    print("③ 判定：")
    ratio = dec["decode_only_s"] / h["wall"]
    print("     纯解码/墙钟 = %.2f  → %s" % (
        ratio, "解码绑定" if ratio > 0.9 else "解码非绑定（有等待余量）"))
    qpb = h["q_put_block"] or 0.0
    print("     生产者队列阻塞 %.3fs（占墙钟 %.0f%%）→ %s" % (
        qpb, qpb / h["wall"] * 100,
        "消费者背压（OCR 绑定）" if qpb / h["wall"] > 0.2 else "背压不显著"))

    out = Path(__file__).resolve().parents[1] / "bench" / "critical_path.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"decode_only": dec, "full": full,
                               "video": name, "frames": args.frames,
                               "ocr_backend": args.ocr},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
