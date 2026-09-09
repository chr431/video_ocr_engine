"""hybrid 增益逐轴复现探针（联调深挖 F4：定位 decord 侧增益在引擎口径下死在哪一层）。

从 decord/tests/bench_hybrid_gpu.py 的口径（yuv420/无 ROI/fork 默认线程/批 250）
出发，逐轴加引擎条件，观察 hybrid_gpu/gpu 相对增益的衰减：
  A bench 原口径            B +ROI
  C +ROI+引擎线程档         D +ROI+引擎线程档+批64（=引擎解码口径）

用法：
  python tools/_probe_hybrid_axis.py --video test6 [--n 3000]
"""
from __future__ import annotations

import argparse
import sys
import time

from decord import VideoReader, gpu, hybrid_gpu

VIDS = {
    "test6": (r"D:\Videos\racelog_test\test6.mp4", (842, 995, 950, 1027), "av1", 24),
    "test5": (r"D:\Videos\racelog_test\test5.mp4", (844, 994, 949, 1026), "h264", 12),
    "test":  (r"D:\Videos\racelog_test\test.mp4",  (842, 995, 950, 1027), "hevc", 12),
}


def run(path, roi_hw, nt, block, n):
    vr = VideoReader(path, ctx=hybrid_gpu(0), output_format='yuv420',
                     roi=roi_hw, num_threads=nt)
    vr.seek(0)
    t0, got = time.perf_counter(), 0
    while got < n:
        e = min(got + block, n)
        b = vr.get_batch(list(range(got, e)))
        got += b.shape[0]
        del b
    vr.close()
    return got / (time.perf_counter() - t0)


def run_gpu(path, roi_hw, n):
    vr = VideoReader(path, ctx=gpu(0), output_format='yuv420', roi=roi_hw)
    vr.seek(0)
    t0, got = time.perf_counter(), 0
    while got < n:
        e = min(got + 64, n)
        b = vr.get_batch(list(range(got, e)))
        got += b.shape[0]
        del b
    vr.close()
    return got / (time.perf_counter() - t0)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, choices=list(VIDS))
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    path, roi, codec, nt = VIDS[args.video]
    x1, y1, x2, y2 = roi
    roi_hw = (x1, y1, x2 + 1, y2 + 1)
    n = args.n

    print(f"== {args.video} ({codec}) roi={roi} 引擎线程档={nt} ==", flush=True)
    # A: bench 原口径（无 ROI，fork 默认线程=不传，批 250）
    vr = VideoReader(path, ctx=hybrid_gpu(0), output_format='yuv420')
    vr.seek(0)
    t0, got = time.perf_counter(), 0
    while got < n:
        e = min(got + 250, n)
        b = vr.get_batch(list(range(got, e)))
        got += b.shape[0]
        del b
    vr.close()
    a = got / (time.perf_counter() - t0)
    # B: +ROI（仍 fork 默认线程、批 250）
    vr = VideoReader(path, ctx=hybrid_gpu(0), output_format='yuv420', roi=roi_hw)
    vr.seek(0)
    t0, got = time.perf_counter(), 0
    while got < n:
        e = min(got + 250, n)
        b = vr.get_batch(list(range(got, e)))
        got += b.shape[0]
        del b
    vr.close()
    b_ = got / (time.perf_counter() - t0)
    # C: +引擎线程档（仍批 250）
    c = run(path, roi_hw, nt, 250, n)
    # D: +批 64（=引擎解码口径）
    d = run(path, roi_hw, nt, 64, n)
    g = run_gpu(path, roi_hw, n)
    print(f"  A bench口径(无ROI,默认nt,批250) hybrid_gpu {a:.0f} fps", flush=True)
    print(f"  B +ROI                        hybrid_gpu {b_:.0f} fps", flush=True)
    print(f"  C +引擎线程档nt={nt}            hybrid_gpu {c:.0f} fps", flush=True)
    print(f"  D +批64 (=引擎解码口径)        hybrid_gpu {d:.0f} fps", flush=True)
    print(f"  纯gpu(NVDEC)+ROI+批64          gpu        {g:.0f} fps", flush=True)
    print(f"  增益: B/A={b_/a:.2f} C/A={c/a:.2f} D/A={d/a:.2f}  "
          f"D vs gpu: {d/g:.2f}x", flush=True)


if __name__ == "__main__":
    main()
