"""解码批大小 sweep（S6 后续：**所有配置都已解码受限**，故攻解码侧）。

背景：S6 之后 h264-gpu 95% / hevc 92% / av1 93% / h264-cpu 也转为解码侧
受限（OCR 相位 0.555s < decode 0.93s）。解码批从 64 起，每批一次
`get_batch` + 一次 analyze 同步；批越大，每次提交的固定开销摊得越薄。

**方法学（C-35）**：顺序 sweep（老 `_probe_perf_sweep.py` 的 `min(walls)`）
会把机器漂移记进结果——同码两次可差 7.7%。本探针按批大小**逐轮交错**、
每格独立子进程（各自含冷/热两轮），比较热轮中位 + 逐对符号一致性。

**正确性硬门**：PI-2（设备批缓冲必须 ≥ max(解码批, 校准帧)）历史事故是
批尺寸不符 → 越界 → 段数漂移。这里每个批大小都比对段数与唯一文本 sha，
不一致立即标红。

用法：
  python tools/_probe_decode_batch_ab.py --configs h264-gpu --batches 64,128,256
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
VIDS = {
    "h264-gpu": ("test5.mp4", (843, 993, 948, 1025), "nvdec", 3000),
    "h264-cpu": ("test5.mp4", (843, 993, 948, 1025), "cpu", 3000),
    "hevc-nvdec": ("test6_hevc.mp4", (841, 994, 949, 1026), "nvdec", 3000),
    "av1-nvdec": ("test6.mp4", (841, 994, 949, 1026), "nvdec", 3000),
}

WORKER = r'''
import hashlib, json, os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
from video_ocr_engine.config import constants as config
config.GPU_PIPELINE_DECODE_BATCH = int(os.environ["PROBE_BATCH"])
from video_ocr_engine import FieldExtractor
vid = os.path.join(os.environ["RACELOG_VIDEO_DIR"], os.environ["PROBE_VIDEO"])
roi = tuple(int(x) for x in os.environ["PROBE_ROI"].split(","))
outs = []
for i in range(2):
    ex = FieldExtractor(vid, roi, frame_start=0,
                        frame_end=int(os.environ["PROBE_FRAMES"]),
                        decode_backend=os.environ["PROBE_DEC"],
                        ocr_backend="tensorrt", keep_crops=False)
    t = time.perf_counter()
    r = ex.extract()
    rep = r.meta.get("report") or {}
    sp = rep.get("spans", {})
    sha = hashlib.sha256("".join(s.text or "" for s in r.segments)
                         .encode()).hexdigest()[:16]
    outs.append({"wall": time.perf_counter() - t, "segs": len(r.segments),
                 "sha": sha,
                 "decode": (sp.get("pipeline.decode") or {}).get("sum"),
                 "dcd_batch": (sp.get("decode.batch") or {}).get("sum"),
                 "dcd_n": (sp.get("decode.batch") or {}).get("n")})
print("PROBE_JSON " + json.dumps(outs))
'''


def one(cfg: str, batch: int, frames: int) -> list:
    vid, roi, dec, _f = VIDS[cfg]
    env = dict(os.environ)
    env.update({"PROBE_ROOT": str(ROOT), "PROBE_BATCH": str(batch),
                "PROBE_VIDEO": vid, "PROBE_ROI": ",".join(map(str, roi)),
                "PROBE_DEC": dec, "PROBE_FRAMES": str(frames),
                "RACELOG_VIDEO_DIR": env.get("RACELOG_VIDEO_DIR",
                                             r"D:\Videos\racelog_test")})
    p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    raise SystemExit("worker 失败：\n%s\n%s" % (p.stdout[-1200:], p.stderr[-1200:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="h264-gpu,hevc-nvdec,av1-nvdec")
    ap.add_argument("--batches", default="64,128,256")
    ap.add_argument("--frames", type=int, default=0, help="0=按配置默认")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=3.0)
    args = ap.parse_args()
    cfgs = args.configs.split(",")
    batches = [int(b) for b in args.batches.split(",")]
    res: dict = {(c, b): [] for c in cfgs for b in batches}
    for i in range(args.repeat):
        for cfg in cfgs:
            frames = args.frames or VIDS[cfg][3]
            for b in batches:
                out = one(cfg, b, frames)
                res[(cfg, b)].append(out)
                print("  [%d] %-11s batch=%-4d cold=%.3f hot=%.3f dcd=%.3f n=%s"
                      % (i, cfg, b, out[0]["wall"], out[1]["wall"],
                         out[1]["dcd_batch"] or -1, out[1]["dcd_n"]),
                      flush=True)
                time.sleep(args.cooldown)
    print("\n%-11s %-6s %9s %9s %8s  %s" % ("config", "batch", "热轮中位",
                                           "解码相位", "Δ%", "段数/文本 sha"))
    for cfg in cfgs:
        ref = statistics.median([r[1]["wall"] for r in res[(cfg, batches[0])]])
        shas = set()
        for b in batches:
            runs = res[(cfg, b)]
            hot = statistics.median([r[1]["wall"] for r in runs])
            dcd = statistics.median([r[1]["dcd_batch"] or 0 for r in runs])
            segs = {r[1]["segs"] for r in runs}
            shas |= {r[1]["sha"] for r in runs}
            print("%-11s %-6d %9.4f %9.4f %+7.2f%%  %s"
                  % (cfg, b, hot, dcd, (hot - ref) / ref * 100, sorted(segs)))
        print("%-11s %-6s 唯一文本 sha %d 个 %s"
              % ("", "", len(shas), "✓" if len(shas) == 1 else "⚠️漂移！"))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
