"""K6：hybrid 输出逐位等价 + 深 prefetch 安全性检验。

以 NVDEC 输出为参照，逐帧比对 hybrid 的 yuv420 ROI 输出字节；
同时在不同 DECORD_HYBRID_PREFETCH 下检验（深 lead 会放大 av1 跨侧
交界竞态——若出现错帧/坏帧，字节比对立即暴露）。

用法：python tools/_probe_hybrid_bitwise.py [--n 6000]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from decord import VideoReader, gpu, hybrid_gpu

PATH = r"D:\Videos\racelog_test\test6.mp4"
ROI_HW = (842, 995, 950, 1027)
BLOCK = 64


def grab(kind, n, pf=None):
    if pf:
        os.environ["DECORD_HYBRID_PREFETCH"] = pf
    if kind == "gpu":
        vr = VideoReader(PATH, ctx=gpu(0), output_format='yuv420', roi=ROI_HW)
    else:
        vr = VideoReader(PATH, ctx=hybrid_gpu(0), output_format='yuv420',
                         roi=ROI_HW, num_threads=24)
    vr.seek(0)
    frames = []
    got = 0
    while got < n:
        e = min(got + BLOCK, n)
        b = vr.get_batch(list(range(got, e))).asnumpy()
        frames.append(b)
        got += b.shape[0]
    vr.close()
    os.environ.pop("DECORD_HYBRID_PREFETCH", None)
    return np.concatenate(frames, axis=0)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    args = ap.parse_args()
    ref = grab("gpu", args.n)
    print(f"参照 NVDEC: {ref.shape} {ref.dtype}", flush=True)
    for tag, pf in (("hybrid 默认prefetch", None),
                    ("hybrid prefetch=1536", "1536"),
                    ("hybrid prefetch=2980", "2980")):
        out = grab("hybrid", args.n, pf)
        same_shape = out.shape == ref.shape
        identical = same_shape and bool((out == ref).all())
        if identical:
            print(f"{tag}: 逐位一致 ✓ ({args.n} 帧)", flush=True)
        else:
            diff_frames = []
            for i in range(min(len(out), len(ref))):
                if out[i].tobytes() != ref[i].tobytes():
                    diff_frames.append(i)
            print(f"{tag}: ⚠️ 不一致！shape={out.shape}/{ref.shape} "
                  f"差异帧数={len(diff_frames)} 首5个={diff_frames[:5]}",
                  flush=True)


if __name__ == "__main__":
    main()
