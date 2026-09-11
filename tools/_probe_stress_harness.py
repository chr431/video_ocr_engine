"""hybrid 死锁/性能压测 harness：每 trial 独立子进程 + 硬超时。

教训（第三次同款）：ad-hoc 压测脚本自身**没有超时**、stdout 走管道被
缓冲——一个挂死的 trial 烧掉 1543s CPU 且不可见。纪律：**任何**解码压
测都按 _probe_upload_chain 的规格来——子进程隔离 + timeout + 进度逐行
落文件（本脚本即该规格的通用化）。

用法（DECORD_LIBRARY_PATH 指向被测 fork dll）：
  python tools/_probe_stress_harness.py --cases hevc-cpu,hevc-gpu,av1-gpu --trials 4 --timeout 90
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKER = r'''
import os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
from decord import VideoReader, hybrid, hybrid_gpu
ctxf = hybrid if os.environ["PROBE_CTX"] == "cpu" else hybrid_gpu
vid = os.path.join(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"),
                   os.environ["PROBE_VID"])
roi = (841, 994, 950, 1027) if "hevc" in os.environ["PROBE_VID"] or "test6" in os.environ["PROBE_VID"] else (843, 993, 949, 1026)
vr = VideoReader(vid, ctx=ctxf(0), output_format="gray", num_threads=16, roi=roi)
n = len(vr); idx = list(range(n)); t = time.perf_counter(); held = []
for s in range(0, min(n, int(os.environ["PROBE_FRAMES"])), 64):
    b = vr.get_batch(idx[s:s+64])
    held.append(b.asnumpy().copy()[:2])
    if len(held) > 6: held.pop(0)
print("WALL " + str(round(time.perf_counter() - t, 2)))
del vr, held
'''

# name -> (ctx, video, extra env)。历史注：fork 4ccf889 起 CPU 产能读数
# 收敛为滑窗持续产能（唯一口径），DECORD_CPU_RATE_SUSTAINED env 已删除
# ——此前用该 env 区分 sustained/EWMA 臂的 -sust 后缀机制一并移除。
CASES = {
    "hevc-cpu": ("cpu", "test6_hevc.mp4", {}),
    "hevc-gpu": ("gpu", "test6_hevc.mp4", {}),
    "av1-gpu": ("gpu", "test6.mp4", {}),
    "av1-cpu": ("cpu", "test6.mp4", {}),
    "h264-gpu": ("gpu", "test5.mp4", {}),
    "h264-cpu": ("cpu", "test5.mp4", {}),
}


def resolve_case(name: str):
    """返回 (ctx, video, extra env) 三元组。"""
    return CASES[name]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="hevc-cpu,hevc-gpu,av1-gpu")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--log", default=str(ROOT / "bench" / "stress_harness.log"))
    args = ap.parse_args()
    Path(args.log).parent.mkdir(exist_ok=True)
    log = open(args.log, "w", encoding="utf-8", buffering=1)
    bad = 0
    for name in args.cases.split(","):
        ctx, vid, extra = resolve_case(name)
        for i in range(args.trials):
            env = dict(os.environ)
            env.update({"PROBE_ROOT": str(ROOT), "PROBE_VID": vid,
                        "PROBE_CTX": ctx, "PROBE_FRAMES": str(args.frames)})
            env.pop("DECORD_HYBRID_DEBUG", None)
            env.update(extra)
            line = ""
            try:
                p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                                   capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   timeout=args.timeout)
                wall = next((ln[5:] for ln in p.stdout.splitlines()
                             if ln.startswith("WALL ")), None)
                line = "%-12s t%-2d %s" % (name, i,
                                           ("wall " + wall + "s") if wall
                                           else "FAIL:" + (p.stderr.strip().splitlines() or ["?"])[-1][:60])
                if not wall:
                    bad += 1
            except subprocess.TimeoutExpired:
                line = "%-12s t%-2d TIMEOUT/挂死" % (name, i)
                bad += 1
            print(line, flush=True)
            log.write(line + "\n")
    log.close()
    print("== bad=%d ==" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
