"""解码速率/seek 开销微基准探针（hybrid 问题定位用，临时工具）。

测量：
  1. 视频元数据（codec/fps/frames/keyframes）
  2. GPU(NVDEC) 与 CPU 软解在 ROI 下的顺序扫掠吞吐（fps）
  3. 乱序 seek 代价（跳片场景）
  4. 关键帧分布（片界吸附可行性）
"""
from __future__ import annotations

import argparse
import bisect
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def probe(video: str, roi: tuple, n_frames: int = 4000, batch: int = 64):
    from decord import VideoReader, cpu, gpu

    print(f"视频: {video}")
    print(f"ROI: {roi}")
    x1, y1, x2, y2 = roi
    roi_hw = (x1, y1, x2 + 1, y2 + 1)

    for label, ctx in (("GPU(NVDEC)", gpu(0)), ("CPU", cpu(0))):
        t0 = time.perf_counter()
        vr = VideoReader(video, ctx=ctx, output_format="gray",
                         num_threads=(4 if label == "CPU" else 0))
        t_open = time.perf_counter() - t0
        total = len(vr)
        codec = vr.get_codec()
        fps = vr.get_avg_fps()
        keys = list(vr.get_key_indices())
        print(f"\n[{label}] open={t_open*1000:.0f}ms frames={total} codec={codec} "
              f"avg_fps={fps} keyframes={len(keys)}")
        if keys:
            gaps = np.diff(np.asarray(keys))
            print(f"  关键帧间隔: min={gaps.min()} med={int(np.median(gaps))} "
                  f"max={gaps.max()}")
        n = min(n_frames, total)
        fr = list(range(0, n))
        # 顺序扫掠
        t0 = time.perf_counter()
        got = 0
        for i in range(0, n, batch):
            b = fr[i:i + batch]
            arr = vr.get_batch(b, roi=roi_hw).asnumpy()
            got += arr.shape[0]
        t_seq = time.perf_counter() - t0
        print(f"  [顺序扫掠] {got} 帧 {t_seq:.3f}s → {got/t_seq:.0f} fps")
        # 乱序 seek（模拟分片竞争：每片起点 seek + 批读）
        t0 = time.perf_counter()
        n_chunks = 8
        step = n // n_chunks
        for c in range(n_chunks):
            s = min(c * step, n - 1)
            vr.seek_accurate(s)
            b = list(range(s, min(s + batch, n)))
            vr.get_batch(b, roi=roi_hw).asnumpy()
        t_seek = time.perf_counter() - t0
        seq_per_chunk = batch / (got / t_seq) if got else 0.001
        print(f"  [8 片 seek+读] {t_seek:.3f}s"
              f"（≈{(t_seek - 8*seq_per_chunk)*1000/8:.0f} ms/次 seek 净额）")
        # 相邻片（连续扫掠）对照
        t0 = time.perf_counter()
        pos = 0
        for c in range(n_chunks):
            s = min(c * step, n - 1)
            if s != pos:
                vr.seek_accurate(s)
            b = list(range(s, min(s + batch, n)))
            vr.get_batch(b, roi=roi_hw).asnumpy()
            pos = s + len(b)
        t_contig = time.perf_counter() - t0
        print(f"  [连续扫掠 8 片] {t_contig:.3f}s（{got/t_contig:.0f} fps）")
        del vr

    # 关键帧 vs 采样网格吸附
    vr = VideoReader(video, ctx=cpu(0), num_threads=4)
    keys = list(vr.get_key_indices())
    stride = 1
    frames = list(range(0, min(n_frames, len(vr)), stride))
    near = 0
    for k in keys:
        if 0 <= k < frames[-1]:
            i = bisect.bisect_left(frames, k)
            cand = [i - 1, i]
            cand = [c for c in cand if 0 <= c < len(frames)]
            d = min(abs(frames[c] - k) for c in cand)
            if d <= stride // 2 + 1:
                near += 1
    print(f"\n[关键帧吸附] 前 {frames[-1]} 帧内 {len(keys)} 个关键帧，"
          f"其中 {near} 个离采样网格 ≤stride/2（吸附成本低）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--frames", type=int, default=4000)
    args = ap.parse_args()
    roi = tuple(int(v) for v in args.roi.split(","))
    probe(args.video, roi, args.frames)


if __name__ == "__main__":
    main()
