"""隔离测量：CPU 软解下「打开时 SetRoi」vs「每次 get_batch 传 roi」的速率差。

起因
----
h264 上 hybrid 比纯 CPU 后端慢 2.17×（decode 3.067s vs 1.415s）。查代码发现
两条路径打开 CPU reader 的方式不同：

    纯 CPU 路径  _open_decord_reader(_cpu(0), roi_kw, num_threads=...)   roi_kw={'roi': roi}
    hybrid 路径  _open_decord_reader(_cpu(0), {}, num_threads=...)       ← 空！

hybrid 改为每次 `get_batch(..., roi=roi)` 传。C++ 侧（src/video/video_reader.cc）：
  · 带 SetRoi：has_roi_=true，解码器按 ROI 尺寸工作
  · 不带：NextFrameImpl() 出整帧后 CropRoi(frame, ...) 再裁

本探针把两种打开方式与同一 roi 交叉，直接量化差异。

用法
----
    python tools/_probe_roi_decode.py --video X --roi A,B,C,D --frames 2000 [--threads 24]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decord import VideoReader, cpu  # noqa: E402


def bench(video: str, roi: tuple, frames: int, threads: int,
          roi_at_open: bool, roi_per_call: bool, runs: int = 3) -> dict:
    kw = {"roi": roi} if roi_at_open else {}
    nt = {"num_threads": threads} if threads > 0 else {}
    walls, rates = [], []
    for _ in range(runs):
        vr = VideoReader(video, ctx=cpu(0), **nt, **kw)
        n = min(frames, len(vr))
        idx = list(range(n))
        # 预热（首批含 decoder 初始化）
        vr.get_batch(idx[:32], **({"roi": roi} if roi_per_call else {})).asnumpy()
        t0 = time.perf_counter()
        done = 0
        i = 32
        while i < n:
            be = min(i + 256, n)
            vr.get_batch(idx[i:be],
                         **({"roi": roi} if roi_per_call else {})).asnumpy()
            done += be - i
            i = be
        dt = time.perf_counter() - t0
        walls.append(dt)
        rates.append(done / dt)
        del vr
    return {"wall": statistics.median(walls), "fps": statistics.median(rates)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    roi = tuple(int(v) for v in args.roi.split(","))
    roi = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)   # 引擎内部用半开区间
    print(f"视频={Path(args.video).name} roi={roi} frames={args.frames} "
          f"threads={args.threads} runs={args.runs}\n")

    cases = [
        ("打开时 SetRoi + 调用传 roi (纯CPU路径)", True, True),
        ("打开时不设 + 调用传 roi (hybrid路径)  ", False, True),
        ("打开时 SetRoi + 调用不传             ", True, False),
        ("都不设（解整帧，基线）                ", False, False),
    ]
    out = {}
    for label, at_open, per_call in cases:
        try:
            r = bench(args.video, roi, args.frames, args.threads, at_open, per_call,
                      args.runs)
            out[label] = r["fps"]
            print("  %s  %6.0f fps  (%.3fs)" % (label, r["fps"], r["wall"]))
        except Exception as e:
            print("  %s  ✗ %s" % (label, e))
            out[label] = 0.0

    a = out.get(cases[0][0], 0)
    b = out.get(cases[1][0], 0)
    if a and b:
        print("\n  纯CPU路径 / hybrid路径 = %.2f×" % (a / b))
        print("  → 若显著 >1，说明 hybrid 的 CPU reader 打开方式漏了优化")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
