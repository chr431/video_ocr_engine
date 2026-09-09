"""K1：hybrid 与两侧解码器速率之和的精确差值测量。

同条件（同 ROI/yuv420/批64/帧窗口）下测 NVDEC 单独、dav1d 单独（与
hybrid 相同 nt）、hybrid 合跑，N 轮取中位，输出理想和与实测和的差。

用法：python tools/_probe_hybrid_sum_gap.py [--video test6] [--n 6000] [--runs 4]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from decord import VideoReader, gpu, cpu, hybrid_gpu

VIDS = {
    "test6": (r"D:\Videos\racelog_test\test6.mp4", (842, 995, 950, 1027), "av1", 24),
    "test5": (r"D:\Videos\racelog_test\test5.mp4", (844, 994, 949, 1026), "h264", 12),
    "test":  (r"D:\Videos\racelog_test\test.mp4",  (842, 995, 950, 1027), "hevc", 12),
}


def run(path, roi_hw, kind, nt, block, n):
    if kind == "gpu":
        vr = VideoReader(path, ctx=gpu(0), output_format='yuv420', roi=roi_hw)
    elif kind == "cpu":
        vr = VideoReader(path, ctx=cpu(0), output_format='yuv420', roi=roi_hw,
                         num_threads=nt)
    else:
        vr = VideoReader(path, ctx=hybrid_gpu(0), output_format='yuv420',
                         roi=roi_hw, num_threads=nt)
    vr.seek(0)
    got, t0 = 0, time.perf_counter()
    while got < n:
        e = min(got + block, n)
        b = vr.get_batch(list(range(got, e)))
        got += b.shape[0]
        del b
    vr.close()
    return got / (time.perf_counter() - t0)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test6", choices=list(VIDS))
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--runs", type=int, default=4)
    args = ap.parse_args()
    path, roi, codec, nt = VIDS[args.video]
    x1, y1, x2, y2 = roi
    roi_hw = (x1, y1, x2 + 1, y2 + 1)
    # 帧窗口不得越界（探测总帧数，留 64 余量）
    vr0 = VideoReader(path, ctx=cpu(0))
    n = min(args.n, len(vr0) - 64)
    vr0.close()

    def med(kind):
        rs = []
        for i in range(args.runs):
            rs.append(run(path, roi_hw, kind, nt, 64, n))
            print(f"  {kind} run{i+1}: {rs[-1]:.0f} fps", flush=True)
        return statistics.median(rs)

    print(f"== {args.video} ({codec}) roi={roi} nt={nt} n={n} "
          f"batch=64 runs={args.runs} ==", flush=True)
    g = med("gpu")
    c = med("cpu")
    h = med("hybrid")
    ideal = g + c
    print(f"\nNVDEC={g:.0f}  dav1d(nt{nt})={c:.0f}  理想和={ideal:.0f}  "
          f"hybrid={h:.0f}")
    print(f"损耗 = {ideal - h:.0f} fps ({(1 - h / ideal) * 100:.1f}%)；"
          f"hybrid/NVDEC = {h / g:.2f}x")


if __name__ == "__main__":
    main()
