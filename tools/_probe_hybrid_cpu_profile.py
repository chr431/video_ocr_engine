"""直接测量 hybrid 运行期间的真实 CPU 占用（不靠反推）。

为什么需要
----------
前面几轮都是用"对照组"反推拖慢源（忙等线程 / BLAS hog / 带宽 hog），
但对照组本身会被污染（例：numpy BLAS 是**多线程**的，号称"2 个 hog"
实际可能吃满 32 核）。用户实测任务管理器里 hybrid 只有 **~50% 占用**，
这与"host CPU 争用"的结论直接冲突。

别再反推了 —— 直接采样：
  1. 后台线程每 200ms 采一次 per-core CPU%
  2. 同时跑真实的 hybrid 提取
  3. 报告解码期间的 CPU 占用分布

若 CPU 占用确实只有 ~50%，则"算力争用"不成立，需另找机理；
若接近 100%，则争用成立，用户的观测可能只是采样时机问题（如看了 OCR 阶段）。

用法
----
    python tools/_probe_hybrid_cpu_profile.py --video X --roi A,B,C,D
        [--frames 4000] [--backend hybrid] [--interval 0.2]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import psutil
except ImportError:
    psutil = None                                   # type: ignore[assignment]


class CpuSampler(threading.Thread):
    """后台采样 per-core CPU%。"""

    def __init__(self, interval: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[list[float]] = []
        self._stop = threading.Event()
        self._proc = psutil.Process() if psutil else None

    def run(self) -> None:
        if not psutil:
            return
        while not self._stop.is_set():
            if self._proc is not None:
                # 进程自身的 CPU%（多核可 >100）
                self.samples.append([float(self._proc.cpu_percent(interval=None))])
            else:
                self.samples.append([0.0])
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=3.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--backend", default="hybrid",
                    choices=["nvdec", "cpu", "hybrid"])
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    if psutil is None:
        print("psutil 未装，无法采样。装：pip install psutil")
        return 1

    roi = tuple(int(v) for v in args.roi.split(","))
    print(f"视频={Path(args.video).name} backend={args.backend} "
          f"frames={args.frames} runs={args.runs}\n")
    print("  逻辑核 = %d, 物理核 = %d"
          % (psutil.cpu_count(), psutil.cpu_count(logical=False)))

    from video_ocr_engine import FieldExtractor  # noqa: E402

    for r in range(args.runs):
        sampler = CpuSampler(args.interval)
        sys_cpu: list[float] = []
        sampler.start()
        t0 = time.perf_counter()
        ex = FieldExtractor(args.video, roi, frame_end=args.frames,
                            keep_frames=True, decode_backend=args.backend,
                            ocr_backend="auto")
        res = ex.extract()
        wall = time.perf_counter() - t0
        sampler.stop()
        if not sampler.samples:
            continue
        # 进程 cpu_percent 是"相对单核"的，多核可 >100%
        vals = [s[0] for s in sampler.samples if s[0] > 0]
        if not vals:
            continue
        peak = max(vals)
        med = statistics.median(vals)
        print("  run%d: 墙钟 %.3fs | 进程 CPU 中位 %.0f%% 峰值 %.0f%% "
              "| 占全部逻辑核 %.0f%% | segs=%d"
              % (r + 1, wall, med, peak, med / psutil.cpu_count(),
                 len(res.segments)))
        sys_cpu.append(med)

    if sys_cpu:
        avg = statistics.median(sys_cpu)
        print("\n  %s 期间进程 CPU 占用中位数 ≈ %.0f%%（单核口径）"
              % (args.backend, avg))
        print("  → 占全部 %d 逻辑核的 %.0f%%"
              % (psutil.cpu_count(), avg / psutil.cpu_count()))
        print("\n  判读：")
        print("    · 若 ≈100%×核数（即吃满）→ 算力争用成立")
        print("    · 若明显偏低（如 ~50%）→ CPU 没吃满，争用另有机理")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
