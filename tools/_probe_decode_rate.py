"""_probe_decode_rate.py —— 只测解码，按 ctx 分离 fork 侧吞吐（不经引擎）。

用途（工作项 0：hybrid 稳态封顶归因）
------------------------------------
三编码 hybrid 稳态速率趋同（2214~2285 fps）说明存在与编码无关的封顶。判定它
落在 **fork 解码路径** 还是 **引擎生产者链（分段/合并）**，需要一个"只解码"的
参照：

  · 本探针 = 与引擎 GPU 管线**同参数**地打开解码器（ctx / output_format='gray'
    / ROI-first / num_threads），并以同样的 `DECODE_BATCH` 粒度调 `get_batch`，
    但**不做任何 analyze / 分段 / merge / OCR**，也不额外消费返回数组。
  · 对照基线 = 引擎报告里的 `producer.decode_batch` span 速率（同机同刻）。

判读
----
- 若"只解码"速率 ≈ 引擎内 decode_batch 速率 → 封顶在 fork；
- 若"只解码"显著更高 → 封顶在引擎侧（生产者线程与消费/分段耦合）。

口径要点（照抄引擎，别自作聪明）
--------------------------------
- `roi` 只在打开时给（构造期 SetRoi）；**逐批 get_batch 不再传 roi**——引擎注释
  实测 hybrid av1 逐批传 roi 会触发 fork 侧池深重算，2487→2264 fps（−9%）。
- 首批单列（冷启/池分配），不计入稳态速率。
- 每臂独立子进程 + min-of-N（跨臂比较用差值，消掉共享的固定开销）。

用法
----
    python tools/_probe_decode_rate.py --ctx hybrid_gpu --video test6_hevc \
        --frames 6000 --threads 32 --reps 2
    python tools/_probe_decode_rate.py                 # 默认跑三 ctx × 三码矩阵
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

_VDIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))
OUT = ROOT / "bench" / "decode_rate.json"

VIDS = {"h264": ("test5.mp4", (843, 993, 948, 1025)),
        "hevc": ("test6_hevc.mp4", (841, 994, 949, 1026)),
        "av1": ("test6.mp4", (841, 994, 949, 1026))}
# 引擎 §14 的 hybrid/cpu 臂线程档（codec 感知）：h264/hevc=32、av1=24
THREADS = {"h264": 32, "hevc": 32, "av1": 24}

WORKER = r'''
import json, os, sys, time
ROOT = os.environ["PROBE_ROOT"]
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import decord
from decord import VideoReader
from video_ocr_engine.config import constants as cfg

ctx_name, path, roi_s, n, threads, batch = sys.argv[1:7]
roi = tuple(int(v) for v in roi_s.split(","))
n, threads, batch = int(n), int(threads), int(batch)
if ctx_name == "hybrid_gpu":
    ctx = decord.hybrid_gpu(0)
elif ctx_name == "hybrid":
    ctx = decord.hybrid(0)
elif ctx_name == "gpu":
    ctx = decord.gpu(0)
else:
    ctx = decord.cpu(0)
kw = {"num_threads": threads} if ctx_name in ("hybrid", "cpu") else {}
vr = VideoReader(path, ctx=ctx, output_format="gray", roi=roi, **kw)
frames = list(range(n))
first = None
lst = []
for bstart in range(0, len(frames), batch):
    bend = min(bstart + batch, len(frames))
    t0 = time.perf_counter()
    nds = vr.get_batch(frames[bstart:bend])
    dt = time.perf_counter() - t0
    nfr = len(nds) if nds is not None else (bend - bstart)
    if first is None:
        first = dt
    else:
        lst.append((nfr, dt))
try:
    vr.close()
except Exception:
    pass
tot_f = sum(f for f, _ in lst)
tot_t = sum(t for _, t in lst)
print(json.dumps({"ctx": ctx_name, "n_first_drop": first,
                  "frames": tot_f, "secs": tot_t,
                  "fps": (tot_f / tot_t) if tot_t else 0.0,
                  "batches": len(lst)}))
'''


def run_arm(ctx_name: str, codec: str, frames: int, reps: int) -> dict:
    vid, roi = VIDS[codec]
    path = str(_VDIR / vid)
    threads = THREADS[codec]
    # DECODE_BATCH 由引擎常量给出（与 device.py 同粒度）；缺省 200
    try:
        from video_ocr_engine.config import constants as cfg
        batch = int(cfg.DECODE_BATCH)
    except Exception:
        batch = 200
    env = dict(os.environ)
    env["PROBE_ROOT"] = str(ROOT)
    rows = []
    for _ in range(reps):
        p = subprocess.run([sys.executable, "-c", WORKER, ctx_name, path,
                            ",".join(map(str, roi)), str(frames), str(threads),
                            str(batch)],
                           capture_output=True, text=True, encoding="utf-8",
                           env=env, timeout=900)
        if p.returncode != 0:
            return {"ctx": ctx_name, "codec": codec, "error": (p.stderr or "")[-400:]}
        rows.append(json.loads(p.stdout.strip().splitlines()[-1]))
    best = max(rows, key=lambda r: r["fps"])
    best["codec"] = codec
    best["threads"] = threads
    best["batch"] = batch
    best["reps"] = reps
    best["fps_all"] = [round(r["fps"], 1) for r in rows]
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="hybrid_gpu")
    ap.add_argument("--video", default="hevc", choices=sorted(VIDS))
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--matrix", action="store_true",
                    help="跑三 ctx × 三码全矩阵（默认只跑单个组合）")
    args = ap.parse_args()

    combos = ([("hybrid_gpu", c) for c in VIDS] + [("gpu", c) for c in VIDS]
              + [("cpu", c) for c in VIDS]) if args.matrix else \
             [(args.ctx, args.video)]
    out = {}
    print("只解码速率（%d 帧/臂，min-of-%d，ROI-first + gray + DECODE_BATCH 粒度）"
          % (args.frames, args.reps))
    print("%-12s %-6s %8s %8s  %s" % ("ctx", "codec", "fps", "线程", "各次"))
    for ctx_name, codec in combos:
        r = run_arm(ctx_name, codec, args.frames, args.reps)
        if "error" in r:
            print("%-12s %-6s  失败: %s" % (ctx_name, codec, r["error"][:80]))
        else:
            print("%-12s %-6s %8.0f %8d  %s" % (ctx_name, codec, r["fps"],
                                                r["threads"], r["fps_all"]))
            print("             （首批 %.4fs 已单列/丢弃，批大小 %d）"
                  % (r["n_first_drop"], r["batch"]))
        out["%s/%s" % (ctx_name, codec)] = r
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("落盘", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
