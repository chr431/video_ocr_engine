"""预处理 resize 依赖替换的引擎级交错 A/B（cv2 vs 现役 numpy-take）。

背景：`tools/_probe_preproc_dep.py` 的模型级 µbench 显示 cv2.resize 在
**生产真实形状**（33×106 → 154×48，1090 次/3000 帧）上快 ~19×
（105.3 → 5.5 µs）。但 C-42 的判例是「按占比推算收益是错的」——
微基准差 ≠ 管线可回收（C-47 同判例）。所以必须落引擎口径：A/B/A/B
交错、独立子进程、热池轮、段数 + 唯一文本集硬门禁。

设计：
  - **A 臂** = 现役 numpy-take（零改动）
  - **B 臂** = 进程内 monkeypatch `_np_resize` 为 cv2 版（产品代码零开关）
  - 每臂跑 2 次，第 1 次冷（建引擎）、第 2 次热（生产口径）
  - 门禁：段数相等 + 唯一文本集相等（浮点差若改读数会在这里暴露）

用法：
  python tools/_probe_preproc_ab.py [--config h264-cpu] [--frames 3000] \
      [--repeat 3] [--roi 843,993,948,1025]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VIDS = {"h264-cpu": ("test5.mp4", (843, 993, 948, 1025), "cpu", "cpu"),
        "h264-gpu": ("test5.mp4", (843, 993, 948, 1025), "nvdec", "tensorrt"),
        "hevc-gpu": ("test6_hevc.mp4", (841, 994, 949, 1026), "nvdec", "tensorrt")}

#: B 臂：把 `_np_resize` 换成 cv2 版。**三处模块级引用都要打**——
#: 产品里是 `from ... import _np_resize`（函数级绑定），只补一个拦不住。
WORKER = r'''
import hashlib, json, os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
import numpy as np

ARM = os.environ["PROBE_ARM"]

if ARM == "B":
    import cv2
    import video_ocr_engine.domain.video_utils as vu

    def _cv2_resize(img, new_w, new_h):
        src_h, src_w = img.shape[:2]
        if new_w == src_w and new_h == src_h:
            return img.astype(np.float32)
        f = np.ascontiguousarray(img.astype(np.float32))
        out = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 3 and out.ndim == 2:
            out = out[..., None]
        return out

    # 关掉 lru_cache 版映射：cv2 路径不用它
    import video_ocr_engine.domain.segmentation as seg
    import video_ocr_engine.ocr.native as nat
    for mod in (vu, seg, nat):
        if hasattr(mod, "_np_resize"):
            mod._np_resize = _cv2_resize

from video_ocr_engine import FieldExtractor

vid = os.path.join(os.environ["RACELOG_VIDEO_DIR"], os.environ["PROBE_VIDEO"])
roi = tuple(int(x) for x in os.environ["PROBE_ROI"].split(","))
outs = []
for i in range(2):                      # 第 1 轮冷、第 2 轮热
    ex = FieldExtractor(vid, roi, frame_start=0,
                        frame_end=int(os.environ["PROBE_FRAMES"]),
                        decode_backend=os.environ["PROBE_DEC"],
                        ocr_backend=os.environ["PROBE_OCR"],
                        keep_crops=False)
    t = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t
    rep = r.meta.get("report") or {}
    # 唯一文本集（正确性门禁：只测"文本有没有变"会漏准确率退化，
    # 这里作为**结构性**门禁，真值准确率由 golden 单独守）
    texts = sorted({s.text for s in r.segments})
    sha = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    spans = rep.get("spans", {})
    outs.append({"wall": wall, "segs": len(r.segments), "texts": len(texts),
                 "text_sha": sha,
                 "preproc": (spans.get("ocr.preprocess") or {}).get("sum"),
                 "preproc_resize": (spans.get("ocr.preproc_resize") or {}).get("sum"),
                 "infer": (spans.get("ocr.infer") or {}).get("sum"),
                 "init": (rep.get("gauges") or {}).get("ocr.engine_init")})
print("PROBE_JSON " + json.dumps(outs))
'''


def one(arm: str, cfg: str, frames: int, roi_override: str) -> list:
    vid, roi, dec, ocr = VIDS[cfg]
    if roi_override:
        roi = tuple(int(x) for x in roi_override.split(","))
    env = dict(os.environ)
    env.update({"PROBE_ROOT": str(ROOT), "PROBE_ARM": arm,
                "PROBE_VIDEO": vid, "PROBE_ROI": ",".join(map(str, roi)),
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
    raise SystemExit("worker(%s) 失败：\n%s\n%s"
                     % (arm, p.stdout[-1500:], p.stderr[-1500:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="h264-cpu")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--roi", default="")
    args = ap.parse_args()

    rows: dict = {c: {"A": [], "B": []} for c in args.config.split(",")}
    for i in range(args.repeat):
        for cfg in args.config.split(","):
            for arm in ("A", "B"):
                out = one(arm, cfg, args.frames, args.roi)
                rows[cfg][arm].append(out)
                print("  [%d] %-10s %s hot=%.4f segs=%d texts=%d "
                      "preproc=%.3f resize=%.3f infer=%.3f"
                      % (i, cfg, arm, out[1]["wall"], out[1]["segs"],
                         out[1]["texts"], out[1]["preproc"] or -1,
                         out[1]["preproc_resize"] or -1, out[1]["infer"] or -1))
            time.sleep(args.cooldown)

    print("\n%-10s %10s %10s %9s  %-8s %s"
          % ("config", "A(hot)", "B(hot)", "Δ%", "逐对符号", "门禁"))
    summary = {}
    for cfg in args.config.split(","):
        pairs = [(rows[cfg]["A"][i][1]["wall"], rows[cfg]["B"][i][1]["wall"])
                 for i in range(args.repeat)]
        ma = statistics.median([p[0] for p in pairs])
        mb = statistics.median([p[1] for p in pairs])
        signs = "".join("+" if b > a else "-" for a, b in pairs)
        segs_ok = all(rows[cfg]["A"][i][1]["segs"] == rows[cfg]["B"][i][1]["segs"]
                      for i in range(args.repeat))
        sha_ok = all(rows[cfg]["A"][i][1]["text_sha"] == rows[cfg]["B"][i][1]["text_sha"]
                     for i in range(args.repeat))
        gate = ("段数OK" if segs_ok else "段数FAIL") + "/" + \
               ("文本OK" if sha_ok else "文本FAIL")
        print("%-10s %10.4f %10.4f %+8.2f%%  %-8s %s"
              % (cfg, ma, mb, (mb - ma) / ma * 100, signs, gate))
        pa = statistics.median([rows[cfg]["A"][i][1]["preproc_resize"] or 0
                                for i in range(args.repeat)])
        pb = statistics.median([rows[cfg]["B"][i][1]["preproc_resize"] or 0
                                for i in range(args.repeat)])
        print("%-10s %10.4f %10.4f %+8.2f%%   (ocr.preproc_resize)"
              % ("", pa, pb, (pb - pa) / pa * 100 if pa else 0))
        summary[cfg] = {"A_hot": ma, "B_hot": mb, "delta_pct": (mb - ma) / ma * 100,
                        "signs": signs, "segs_ok": segs_ok, "text_ok": sha_ok,
                        "preproc_a": pa, "preproc_b": pb}

    out = ROOT / "bench" / "preproc_ab.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "raw": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
