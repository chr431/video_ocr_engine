"""K2：hybrid 调度轨迹剖析（DECORD_HYBRID_DEBUG=1 + 逐批耗时捕获）。

捕获每 chunk 的调度决策（rc/rg/侧选择/在途）与逐批 get_batch 耗时，
输出：分侧帧数、rc/rg 稳态值、逐批耗时分布、慢批与侧切换的对应关系。

用法：python tools/_probe_hybrid_trace.py [--video test6] [--n 6000]
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys

WORKER = r'''
import sys, time, json
sys.stdout.reconfigure(encoding="utf-8")
from decord import VideoReader, hybrid_gpu
path, roi, nt, n = sys.argv[1], (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])), int(sys.argv[6]), int(sys.argv[7])
roi_hw = (roi[0], roi[1], roi[2]+1, roi[3]+1)
vr = VideoReader(path, ctx=hybrid_gpu(0), output_format='yuv420', roi=roi_hw, num_threads=nt)
vr.seek(0)
got, t0, batches = 0, time.perf_counter(), []
while got < n:
    e = min(got + 64, n)
    tb = time.perf_counter()
    b = vr.get_batch(list(range(got, e)))
    batches.append((got, time.perf_counter() - tb))
    got += b.shape[0]
    del b
vr.close()
wall = time.perf_counter() - t0
print("BATCHES " + json.dumps(batches))
print(f"FPS {got/wall:.0f}")
'''


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test6.mp4")
    ap.add_argument("--roi", default="842,995,950,1027")
    ap.add_argument("--nt", type=int, default=24)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    roi = tuple(int(v) for v in args.roi.split(","))

    env = dict(os.environ)
    env["DECORD_HYBRID_DEBUG"] = "1"
    for r in range(args.runs):
        proc = subprocess.run(
            [sys.executable, "-c", WORKER, args.video,
             str(roi[0]), str(roi[1]), str(roi[2]), str(roi[3]),
             str(args.nt), str(args.n)],
            capture_output=True, text=True, timeout=300, env=env)
        err = proc.stderr
        # 调度行解析
        sched = re.findall(
            r"\[hybrid-sched\] key=(-?\d+) rc=([\d.]+) rg=([\d.]+) "
            r"side=(\d) q=(-?\d+) pend=(-?\d+) spt=(-?\d+) gp=(-?\d+) "
            r"tc=([\d.]+) tg=([\d.]+) -> (\d)", err)
        final = re.findall(r"\[hybrid\] chunks cpu=(\d+) gpu=(\d+) "
                           r"frames cpu=(\d+) gpu=(\d+)", err)
        budget = re.findall(r"\[hybrid-budget\] (.+)", err)
        print(f"── run{r+1} ──")
        for b in budget:
            print(f"  budget: {b}")
        if final:
            print(f"  final: chunks cpu={final[0][0]} gpu={final[0][1]} "
                  f"frames cpu={final[0][2]} gpu={final[0][3]} "
                  f"(份额 cpu={int(final[0][3]) and int(final[0][2]) / (int(final[0][2]) + int(final[0][3])):.2f})")
        # 稳态 rc/rg
        rcs = [float(s[1]) for s in sched if float(s[1]) > 0]
        rgs = [float(s[2]) for s in sched if float(s[2]) > 0]
        if rcs:
            print(f"  rc 稳态(后5次): {[f'{v:.0f}' for v in rcs[-5:]]}")
            print(f"  rg 稳态(后5次): {[f'{v:.0f}' for v in rgs[-5:]]}")
        sides = [(int(s[0]), s[10], s[1], s[2]) for s in sched]
        switches = sum(1 for i in range(1, len(sides))
                       if sides[i][1] != sides[i - 1][1])
        print(f"  chunks={len(sides)} 侧切换次数={switches}")
        # 逐批耗时
        for line in proc.stdout.splitlines():
            if line.startswith("BATCHES "):
                import json
                batches = json.loads(line[8:])
                durs = [d for _, d in batches]
                durs_ms = [d * 1000 for d in durs]
                durs_ms.sort()
                p50 = durs_ms[len(durs_ms) // 2]
                p90 = durs_ms[int(len(durs_ms) * 0.9)]
                p99 = durs_ms[int(len(durs_ms) * 0.99)]
                print(f"  批耗时 ms: p50={p50:.1f} p90={p90:.1f} p99={p99:.1f} "
                      f"max={durs_ms[-1]:.1f}（理论 {64 / 2400 * 1000:.1f}ms@2400fps）")
                slow = [(idx, d * 1000) for idx, d in batches
                        if d * 1000 > p50 * 3]
                print(f"  慢批(>3×p50): {len(slow)} 个 → 前 10: "
                      f"{[(i, f'{d:.0f}ms') for i, d in slow[:10]]}")
            if line.startswith("FPS "):
                print(f"  {line}")


if __name__ == "__main__":
    main()
