"""GPU 指令账本（周期计数测量首轮，2026-10-09）。

问题：GPU 侧墙钟受频率漂移/争用干扰；假设：同一 kernel 对同一输入的
指令数（sm__inst_executed）是确定性的工作量度量——同码重复逐位相等，
且与 CPU 背景负载无关。用 Nsight Compute（ncu）验证；ncu 在 GeForce
上需要管理员令牌（ERR_NVGPUCTRPERM），由外部提权运行（UAC 静默）。

工作负载 = 引擎自带的 NVRTC kernel（GpuFrameAnalyzer.analyze_batch，
B×1080×1920 灰度），预热臂用 B_warm=8（grid=8，CSV 里按 grid 与正式
臂 grid=B 区分）。

用法：
  python tools/_probe_gpu_inst.py --reps 10                # 纯墙钟（ncu 外）
  python tools/_probe_gpu_inst.py --reps 10 --load 32      # 墙钟 + CPU 负载
  # ncu 提权（由 shell 发起，UAC 静默）：
  #   ncu --csv --metrics sm__inst_executed.sum,gpu__time_duration.sum,launch__grid_size \
  #       --target-processes all --log-file bench\gpu_inst.csv \
  #       python tools\_probe_gpu_inst.py --reps 8
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 跨探针 import 惯例

B = 64
H, W = 1080, 1920
B_WARM = 8


def gpu_work(reps: int, load: int) -> int:
    """跑负载：预热 1×(B_warm) + reps×(B)。返回 checksum（确定性自检）。"""
    from _probe_cycle_ledger import LoadGroup
    from cuda.bindings import runtime as cudart
    from video_ocr_engine.ocr.trt import GpuFrameAnalyzer

    n = B * H * W
    _e, a_ptr = cudart.cudaMalloc(n)
    _e, b_ptr = cudart.cudaMalloc(n)
    _e, s = cudart.cudaStreamCreate()
    cudart.cudaMemsetAsync(a_ptr, 0x5A, n, s)
    cudart.cudaMemsetAsync(b_ptr, 0xA5, n, s)
    cudart.cudaStreamSynchronize(s)

    an = GpuFrameAnalyzer()
    chk = 0
    walls: list[float] = []
    with LoadGroup(load):
        an.analyze_batch(a_ptr, b_ptr, B_WARM, H, W, 128.0)   # 预热（grid=B_warm）
        for i in range(reps):
            t0 = time.perf_counter()
            out = an.analyze_batch(a_ptr, b_ptr, B, H, W, 128.0)
            walls.append((time.perf_counter() - t0) * 1000)
            chk += int(out.sum())
            print("    rep%-2d wall=%7.2fms out_sum=%d"
                  % (i, walls[-1], int(out.sum())))
    cv = (statistics.stdev(walls) / statistics.mean(walls) * 100
          if len(walls) > 1 else 0.0)
    print("  [gpu load=%d] wall %.2fms CV %.2f%% checksum=%d"
          % (load, statistics.mean(walls), cv, chk))
    an.release()
    cudart.cudaFree(a_ptr)
    cudart.cudaFree(b_ptr)
    cudart.cudaStreamDestroy(s)
    return chk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--load", type=int, default=0)
    args = ap.parse_args()
    gpu_work(args.reps, args.load)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
