"""seek 成本分解探针：把 seek_accurate 的净额拆成固定开销 × 关键帧距离斜率。

对每个编码取关键帧后 {0,30,60,120,240} 帧距离的目标点，测：
  A) seek_accurate(s) 纯调用（不读帧，测 demuxer/flush 开销）
  B) seek_accurate(s) + get_batch([s])（完整精确取帧）
  C) seek 落到关键帧后连续取 100 帧的逐帧速率（post-seek 解码是否变慢）
  D) 无 seek 顺序取 100 帧对照
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decord import VideoReader, cpu  # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r"D:\Videos\racelog_test\test6.mp4"
ROI = (841, 994, 950, 1027)
DISTS = (0, 30, 60, 120, 240)
REP = 5


def bench_cpu():
    vr = VideoReader(VIDEO, ctx=cpu(0), num_threads=4, output_format="gray")
    keys = list(vr.get_key_indices())
    print(f"frames={len(vr)} keyframes={len(keys)} first_keys={keys[:6]}")
    # 只用前 3 个关键帧（避开前百帧的编码器预热段），每个距离 REP 轮取中位
    rows = []
    for k in keys[1:4]:
        for d in DISTS:
            s = min(k + d, len(vr) - 1)
            tA, tB = [], []
            for _ in range(REP):
                t0 = time.perf_counter()
                vr.seek_accurate(s)
                tA.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                arr = vr.get_batch([s], roi=ROI).asnumpy()
                tB.append(time.perf_counter() - t0)
            rows.append((d, sorted(tA)[REP // 2], sorted(tB)[REP // 2]))
    print("d(dist)  seek_only(ms)  seek+fetch(ms)  fetch_net(ms)")
    for d, ta, tb in rows:
        print(f"{d:4d}  {ta*1000:10.1f}  {tb*1000:13.1f}  {(tb-ta)*1000:12.1f}")
    # 线性拟合 fetch_net ~ a*d + b（最小二乘）
    n = len(rows)
    sx = sum(r[0] for r in rows); sy = sum((r[2]-r[1])*1000 for r in rows)
    sxx = sum(r[0]*r[0] for r in rows); sxy = sum(r[0]*(r[2]-r[1])*1000 for r in rows)
    a = (n*sxy - sx*sy) / (n*sxx - sx*sx)
    b = (sy - a*sx) / n
    print(f"拟合: fetch_net ≈ {a:.2f} ms/帧 × d + {b:.1f} ms 固定")

    # C: seek 后连续解码 100 帧 vs D: 顺序对照
    k = keys[2]
    vr.seek_accurate(k)
    vr.get_batch([k], roi=ROI).asnumpy()
    t0 = time.perf_counter()
    vr.get_batch(list(range(k, k+100)), roi=ROI).asnumpy()
    t_post = time.perf_counter() - t0
    t0 = time.perf_counter()
    vr.get_batch(list(range(k+200, k+300)), roi=ROI).asnumpy()
    t_seq = time.perf_counter() - t0
    print(f"C post-seek 连续100帧: {t_post*1000:.0f}ms ({100/t_post:.0f} fps)")
    print(f"D 顺序     连续100帧: {t_seq*1000:.0f}ms ({100/t_seq:.0f} fps)")


if __name__ == "__main__":
    bench_cpu()
