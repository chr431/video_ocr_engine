"""临时探针 B：拆解"每帧固定开销"来源（分析用）。

测量：
  1. 批大小 B 扫描：若固定开销是"每批 Python/调用"级，大 B 应显著更快。
  2. ROI 尺寸扫描：若固定开销是解码器每帧固定成本，ROI 大小无关。
  3. next_roi 顺序流 vs get_batch 索引批。
  4. 只解码不取数据（不 asnumpy）时的上限。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def open_vr(path, ctx_tag, roi=None, fmt="gray"):
    from decord import VideoReader, cpu, gpu as _g
    ctx = _g(0) if ctx_tag == "NVDEC" else cpu(0)
    kw = {}
    if roi is not None:
        kw["roi"] = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)
    return VideoReader(path, ctx=ctx, output_format=fmt, **kw)


def sweep_batch(path, roi, nframes, ctx_tag):
    print(f"  --- 批大小扫描 ({ctx_tag}, ROI={roi}) ---")
    for B in (1, 8, 16, 64, 256, 1024):
        vr = open_vr(path, ctx_tag, roi)
        frames = list(range(0, nframes))
        vr.get_batch(frames[:min(16, B)], roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1)).asnumpy()
        t0 = time.perf_counter()
        got = 0
        for i in range(0, len(frames), B):
            nds = vr.get_batch(frames[i:i + B],
                               roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1))
            got += len(nds.asnumpy())
        dt = time.perf_counter() - t0
        del vr
        print(f"    B={B:5d}: {got} 帧 / {dt:.3f}s = {got/dt:7.0f} fps "
              f"({dt/got*1e3:.4f} ms/帧)")


def sweep_roi(path, nframes, ctx_tag, base_roi):
    print(f"  --- ROI 尺寸扫描 ({ctx_tag}) ---")
    x1, y1, x2, y2 = base_roi
    for scale, name in ((1, "33x106 原"), (4, "132x424"), (12, "396x1272")):
        w = min(1920 - x1, (x2 - x1 + 1) * scale)
        h = min(1080 - y1, (y2 - y1 + 1) * scale)
        roi = (x1, y1, x1 + w - 1, y1 + h - 1)
        try:
            vr = open_vr(path, ctx_tag, roi)
        except Exception as e:
            print(f"    {name}: open fail {e}")
            continue
        frames = list(range(0, nframes))
        rk = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)
        vr.get_batch(frames[:16], roi=rk).asnumpy()
        t0 = time.perf_counter()
        for i in range(0, len(frames), 64):
            vr.get_batch(frames[i:i + 64], roi=rk).asnumpy()
        dt = time.perf_counter() - t0
        del vr
        print(f"    ROI {w}x{h:4d} ({name}): {len(frames)/dt:7.0f} fps "
              f"({dt/len(frames)*1e3:.4f} ms/帧)")


def bench_next_roi(path, roi, nframes, ctx_tag):
    print(f"  --- next_roi 顺序流 ({ctx_tag}) ---")
    vr = open_vr(path, ctx_tag, roi)
    rk = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)
    for _ in range(16):
        vr.next_roi(*rk)
    t0 = time.perf_counter()
    n = 0
    for _ in range(nframes - 16):
        vr.next_roi(*rk).asnumpy()
        n += 1
    dt = time.perf_counter() - t0
    del vr
    print(f"    next_roi 逐帧: {n/dt:7.0f} fps ({dt/n*1e3:.4f} ms/帧)")


def bench_nodata(path, roi, nframes, ctx_tag):
    """只提交解码、不取回像素（asnumpy 跳过）—— 量化 D2H/转换占比。"""
    print(f"  --- 不取像素 ({ctx_tag}) ---")
    vr = open_vr(path, ctx_tag, roi)
    frames = list(range(0, nframes))
    rk = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)
    vr.get_batch(frames[:16], roi=rk).asnumpy()
    t0 = time.perf_counter()
    for i in range(0, len(frames), 64):
        vr.get_batch(frames[i:i + 64], roi=rk)
    dt = time.perf_counter() - t0
    del vr
    print(f"    仅 get_batch 无 asnumpy: {len(frames)/dt:7.0f} fps "
          f"({dt/len(frames)*1e3:.4f} ms/帧)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test5.mp4")
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--ctx", default="NVDEC")
    args = ap.parse_args()
    roi = tuple(int(x) for x in args.roi.split(","))
    print(f"=== {os.path.basename(args.video)} / {args.ctx} ===")
    bench_nodata(args.video, roi, args.frames, args.ctx)
    sweep_batch(args.video, roi, args.frames, args.ctx)
    bench_next_roi(args.video, roi, args.frames, args.ctx)
    if args.ctx == "NVDEC":
        sweep_roi(args.video, args.frames, args.ctx, roi)


if __name__ == "__main__":
    main()
