"""绑定约束判定：把 OCR 换成零成本，看墙钟是否变化（决定性实验）。

背景（`_probe_span_dump.py` 实测，h264-cpu 热轮）：
  wall 2.27s | producer(pipeline.consumer) 2.10s | ocr 相位 2.21s
  ocr.infer 2.96s（69 批，双 worker）| preproc_resize 0.51s | q_get_wait 1.51s
  —— 而 `_probe_critical_path.py` 测**纯解码只有 1.24s**。

即：解码不是瓶颈（1.24 < 2.10），OCR 也疑似不是（砍掉 59% 预处理后墙钟不动）。
按 C-42 判例「按占比推算收益是错的」，唯一可信的判定法是**把一侧拆掉看墙钟**：

  · 零成本 OCR（infer 直接返回常量）→ 墙钟若 ≈ producer(2.10s)，则**生产者绑定**，
    任何 OCR/预处理依赖替换都无墙钟空间；若墙钟大跌，则 OCR 绑定。
  · 同时报告 producer 内部缺口（consumer − 已计 span 之和）= 分段状态机 +
    emit + Python 循环的未计成本。

用法：
  python tools/_probe_binding.py [--video test5] [--frames 3000] [--repeat 3]
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
       "test6_hevc": ("test6_hevc.mp4", (841, 994, 949, 1026)),
       "test6_av1": ("test6.mp4", (841, 994, 949, 1026))}

WORKER = r'''
import json, os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
import numpy as np

ARM = os.environ["PROBE_ARM"]

if ARM == "noocr":
    # 零成本 OCR：把推理换成常量输出（预处理/CTC 仍在，只拆推理）。
    # 形状 = 生产 (B, seq, vocab)，CTC 解码得到空文本，段数不受影响
    # （段的定义来自像素分段，不来自 OCR）。
    import video_ocr_engine.ocr.native as nat
    _orig_init = nat.OcrEngine._init_onnx

    def _noop_infer(self, batch_np):
        return np.zeros((len(batch_np), 40, 18710), dtype=np.float32)

    nat.OcrEngine._infer_locked = _noop_infer

if ARM == "nopreproc":
    # 零成本预处理：跳过 resize（返回原图转置后的常量形状），看预处理是否绑定。
    import video_ocr_engine.ocr.native as nat
    _orig = nat.OcrEngine._resize_norm

    def _noop_resize_norm(img, max_wh_ratio, height=48):
        img_width = int(height * max_wh_ratio)
        return np.zeros((3, height, img_width), dtype=np.float32)

    nat.OcrEngine._resize_norm = staticmethod(_noop_resize_norm)

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
    outs.append({"wall": wall, "segs": len(r.segments),
                 "producer": (sp.get("pipeline.consumer") or {}).get("sum"),
                 "ocr_phase": (sp.get("pipeline.ocr") or {}).get("sum"),
                 "decode_batch": (sp.get("decode.batch") or {}).get("sum"),
                 "preproc": (sp.get("ocr.preprocess") or {}).get("sum"),
                 "preproc_resize": (sp.get("ocr.preproc_resize") or {}).get("sum"),
                 "infer": (sp.get("ocr.infer") or {}).get("sum"),
                 "q_get_wait": ga.get("pipeline.q_get_wait"),
                 "q_put_block": ga.get("pipeline.q_put_block")})
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
        "h264-gpu": ("test5.mp4", (843, 993, 948, 1025), "nvdec", "tensorrt"),
        "hevc-gpu": ("test6_hevc.mp4", (841, 994, 949, 1026), "nvdec", "tensorrt")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="h264-cpu")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=4.0)
    args = ap.parse_args()

    cfgs = args.config.split(",")
    res: dict = {}
    for cfg in cfgs:
        res[cfg] = {}
        for arm in ("base", "noocr", "nopreproc"):
            hot = []
            for _ in range(args.repeat):
                out = one(arm, cfg, args.frames)
                hot.append(out[-1])
                time.sleep(args.cooldown)
            res[cfg][arm] = {
                "wall": statistics.median([h["wall"] for h in hot]),
                "segs": hot[-1]["segs"],
                "producer": statistics.median([h["producer"] or 0 for h in hot]),
                "preproc": statistics.median([h["preproc"] or 0 for h in hot]),
                "infer": statistics.median([h["infer"] or 0 for h in hot]),
                "q_get_wait": statistics.median([h["q_get_wait"] or 0 for h in hot]),
            }
        b, o, p = (res[cfg][k] for k in ("base", "noocr", "nopreproc"))
        print("\n== %s ==" % cfg)
        print("%-10s %9s %9s %9s %9s %9s" % ("arm", "wall", "producer",
                                             "preproc", "infer", "q_get_wait"))
        for name, r in (("base", b), ("noocr", o), ("nopreproc", p)):
            print("%-10s %9.4f %9.4f %9.4f %9.4f %9.4f" % (
                name, r["wall"], r["producer"], r["preproc"], r["infer"],
                r["q_get_wait"]))
        print("  Δ wall: noocr %+.2f%%  nopreproc %+.2f%%（负=更快）"
              % ((o["wall"] - b["wall"]) / b["wall"] * 100,
                 (p["wall"] - b["wall"]) / b["wall"] * 100))
        print("  段数: base %d / noocr %d / nopreproc %d"
              % (b["segs"], o["segs"], p["segs"]))

    out = ROOT / "bench" / "binding.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
