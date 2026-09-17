"""宽 ROI（字幕）场景的绑定判定 + resize 替换收益上限核算。

## 现象
`_probe_numpy_replace_ab.py --roi 841,994,1890,1026`（1049×32，字幕型宽 ROI）：
  · A（全 numpy）24.6701s / C（生产者+消费者 cv2）24.3950s → **仅 −1.12%**
    （符号 ----- 一致 = 真效应，但仍远小于"resize 占 33%"的直觉预期）
  · 该场景 resize 从 7.98s → 3.04s（省 **4.9s**），而墙钟只省 0.28s。

## 为什么 4.9s 只兑现 0.28s
短窗 span 转储（1200 帧）显示该场景瓶颈已**换人**：
    pipeline.consumer 7.78s（= decode 7.78s）
    ocr.infer 18.55s / 73 chunk（双 worker → 每 worker 9.3s）
    **pipeline.ocr_tail 1.76s**
即 trt/OV 推理在该宽度下成为真正绑定（resize 属于消费者线程内部，
消费者本身被推理占满 → 省下的 resize 时间只是让 worker 更快空转等下一批，
而**尾批**（ocr_tail）吃掉差额）。

本探针在**全片口径**复核该判定，并算出 resize 替换的**收益上限**
（= 消费者线程里 resize 占比 × 可兑现系数），用于回答
"换库能否缩减墙钟"在该场景下的上限。

用法：
  python tools/_probe_wide_roi_binding.py [--frames 3000] [--repeat 3]
      [--roi 841,994,1890,1026]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKER = r'''
import hashlib, json, os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
import numpy as np

ARM = os.environ["PROBE_ARM"]

if ARM == "noinfer":
    # 零成本 OCR 推理（保留预处理/CTC）：判定该场景是否推理绑定
    import video_ocr_engine.ocr.native as nat

    def _noop(self, batch_np):
        return np.zeros((len(batch_np), 40, 18710), dtype=np.float32)

    nat.OcrEngine._infer_locked = _noop

from video_ocr_engine import FieldExtractor

vid = os.path.join(os.environ["RACELOG_VIDEO_DIR"], os.environ["PROBE_VIDEO"])
roi = tuple(int(x) for x in os.environ["PROBE_ROI"].split(","))
outs = []
for i in range(2):
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
                 "producer": g("pipeline.consumer"),
                 "ocr_phase": g("pipeline.ocr"), "tail": g("pipeline.ocr_tail"),
                 "decode_batch": g("decode.batch"), "resize": g("ocr.preproc_resize"),
                 "preproc": g("ocr.preprocess"), "infer": g("ocr.infer"),
                 "chunks": (rep.get("counters") or {}).get("ocr.chunks")})
print("PROBE_JSON " + json.dumps(outs))
'''


def one(arm: str, video: str, roi, frames: int, dec: str, ocr: str) -> list:
    env = dict(os.environ)
    env.update({"PROBE_ROOT": str(ROOT), "PROBE_ARM": arm,
                "PROBE_VIDEO": video, "PROBE_ROI": ",".join(map(str, roi)),
                "PROBE_DEC": dec, "PROBE_OCR": ocr,
                "PROBE_FRAMES": str(frames),
                "RACELOG_VIDEO_DIR": env.get("RACELOG_VIDEO_DIR",
                                             r"D:\Videos\racelog_test")})
    p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    raise SystemExit("worker(%s) 失败：\n%s\n%s" % (arm, p.stdout[-1200:], p.stderr[-1500:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--roi", default="841,994,1890,1026")
    ap.add_argument("--video", default="test5.mp4")
    ap.add_argument("--decode", default="cpu")
    ap.add_argument("--ocr", default="cpu")
    ap.add_argument("--cooldown", type=float, default=4.0)
    args = ap.parse_args()

    roi = tuple(int(x) for x in args.roi.split(","))
    res: dict = {"roi": list(roi), "frames": args.frames, "arms": {}}
    print("宽 ROI 绑定判定：%s  ROI=%s（%dx%d）%d 帧\n"
          % (args.video, roi, roi[2] - roi[0], roi[3] - roi[1], args.frames))

    for arm in ("base", "noinfer"):
        rows = []
        for _ in range(args.repeat):
            rows.append(one(arm, args.video, roi, args.frames,
                            args.decode, args.ocr)[-1])
            time.sleep(args.cooldown)
        med = {k: statistics.median([r[k] or 0 for r in rows])
               for k in ("wall", "producer", "ocr_phase", "tail",
                         "decode_batch", "resize", "preproc", "infer")}
        med["segs"] = rows[-1]["segs"]
        med["chunks"] = rows[-1]["chunks"]
        res["arms"][arm] = med

    b, n = res["arms"]["base"], res["arms"]["noinfer"]
    print("%-9s %9s %9s %9s %9s %9s" % ("arm", "wall", "producer",
                                        "tail", "resize", "infer"))
    for k in ("base", "noinfer"):
        r = res["arms"][k]
        print("%-9s %9.4f %9.4f %9.4f %9.4f %9.4f"
              % (k, r["wall"], r["producer"], r["tail"], r["resize"], r["infer"]))
    print("\nΔwall（零成本推理）= %+.2f%%  段数 %d vs %d"
          % ((n["wall"] - b["wall"]) / b["wall"] * 100, b["segs"], n["segs"]))

    # 收益上限核算
    share = b["resize"] / b["wall"] * 100
    print("\nresize 占墙钟 %.1f%%（%.4fs）；替换可省 ~91%%（实测 −94~95%% 内核）"
          "⇒ 名义可省 %.3fs = %.1f%% 墙钟"
          % (share, b["resize"], b["resize"] * 0.91,
             b["resize"] * 0.91 / b["wall"] * 100))
    print("实测（_probe_numpy_replace_ab.py，全片）仅 −1.12% —— 差额被"
          "「推理绑定 + ocr_tail %.2fs」吸收" % b["tail"])

    out = ROOT / "bench" / "wide_roi_binding.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
