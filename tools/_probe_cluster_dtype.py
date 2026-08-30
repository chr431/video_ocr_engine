"""临时探针：_cluster_win3 的 uint8 变体 —— 等价性与加速比微基准。

现役实现（segmentation.py:97）走 int32：
    s = diff.astype(np.int32)   # 分配 + 类型转换（4 字节/px）
    c3 = s.copy(); 2×切片加法
    w3 = c3.copy(); 2×切片加法
    float(w3.max())

窗口和的数学上界 = 9（3×3 全 1），uint8 完全够用。且 np.bool_ 与 np.uint8
同为 1 字节 → `diff.view(np.uint8)` 是**零拷贝**，连转换都省掉。
内存带宽降到 1/4，大 ROI 上应显著更快。

本探针验证两件事：
  Stage A 等价性：随机掩码（多种密度）逐位对比，必须完全一致。
  Stage B 加速比：四个真实 ROI 尺寸各 2000 次，测两个版本耗时。
"""
from __future__ import annotations

import time

import numpy as np

from segmentation import _cluster_win3 as orig


def cluster_u8(diff: np.ndarray) -> float:
    """uint8 变体（候选实现）。"""
    if not diff.any():
        return 0.0
    s = diff.view(np.uint8) if diff.flags.c_contiguous else diff.astype(np.uint8)
    c3 = s.copy()
    c3[:, 1:] += s[:, :-1]
    c3[:, :-1] += s[:, 1:]
    w3 = c3.copy()
    w3[1:, :] += c3[:-1, :]
    w3[:-1, :] += c3[1:, :]
    return float(w3.max())


def stage_a():
    print("=== Stage A 等价性（随机掩码，逐位对比）===")
    rng = np.random.default_rng(0)
    bad = 0
    total = 0
    for (h, w) in ((33, 106), (52, 800), (200, 800), (600, 1600), (7, 3), (1, 1)):
        for p in (0.0, 0.001, 0.01, 0.05, 0.2, 0.5, 0.9, 1.0):
            d = rng.random((h, w)) < p
            a, b = orig(d), cluster_u8(d)
            total += 1
            if a != b:
                bad += 1
                print(f"  不一致 {h}x{w} p={p}: {a} != {b}")
    # 结构化图案（真实帧的 diff 不是纯随机）
    for (h, w) in ((33, 106), (52, 800), (200, 800)):
        for pat in ("block", "row", "col", "cross", "corner"):
            d = np.zeros((h, w), dtype=bool)
            if pat == "block":
                d[h // 3:2 * h // 3, w // 3:2 * w // 3] = True
            elif pat == "row":
                d[h // 2, :] = True
            elif pat == "col":
                d[:, w // 2] = True
            elif pat == "cross":
                d[h // 2, :] = True
                d[:, w // 2] = True
            else:
                d[:3, :3] = True
                d[-3:, -3:] = True
            a, b = orig(d), cluster_u8(d)
            total += 1
            if a != b:
                bad += 1
                print(f"  不一致 {h}x{w} {pat}: {a} != {b}")
    print(f"  {total} 组：不一致 {bad} 组 → {'等价' if bad == 0 else '❌ 不等价'}")
    return bad == 0


def stage_b():
    print("\n=== Stage B 加速比（各 2000 次）===")
    rng = np.random.default_rng(1)
    print(f"  {'ROI':>12s} {'int32(现役)':>12s} {'uint8':>10s} {'加速':>7s}")
    for (h, w) in ((33, 106), (52, 800), (200, 800), (600, 1600)):
        # 用接近真实的稀疏密度（变帧的 diff 通常集中在字形上）
        d = rng.random((h, w)) < 0.05
        for fn in (orig, cluster_u8):        # 预热
            fn(d)
        n = 2000
        t0 = time.perf_counter()
        for _ in range(n):
            orig(d)
        t1 = time.perf_counter()
        for _ in range(n):
            cluster_u8(d)
        t2 = time.perf_counter()
        a = (t1 - t0) / n * 1e6
        b = (t2 - t1) / n * 1e6
        print(f"  {w:5d}x{h:<5d} {a:11.2f}µs {b:9.2f}µs {a / b:6.2f}×")


if __name__ == "__main__":
    ok = stage_a()
    stage_b()
    print("\n结论：", "uint8 变体逐位等价，可落地" if ok
          else "❌ 不等价，不可落地")
