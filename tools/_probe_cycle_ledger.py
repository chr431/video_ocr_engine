"""CPU 周期账本（周期计数测量首轮，2026-10-09）。

问题：墙钟 A/B 受机器漂移/后台负载干扰（AGENTS：A/B 必须交错；同码两次
可差 7.7%）。假设：QueryProcessCycleTime（进程全线程的 TSC 周期，只计
在 CPU 上运行的时间）对频率漂移/被调度挤出的空转天然免疫 → 同一负载
重复的 cycles CV 应远小于 wall CV；负载开/关之间 cycles 均值应近似守恒
（wall 则被拉伸）。

用法：
  python tools/_probe_cycle_ledger.py --calib           # 合成矩阵乘（不引引擎）
  python tools/_probe_cycle_ledger.py                    # 引擎矩阵（默认 5 条件）
  python tools/_probe_cycle_ledger.py --backend cpu --load 32 --reps 8   # 单条件

判据（先量后做）：
  - 免疫性：cycles CV(max over load) ≪ wall CV(max over load)
  - 守恒性：load>0 与 load=0 的 cycles 均值比 ≈ 1（SMT 争用会带来多少抬升
    是待测项，不是假设）
  - 区分度：cpu 与 nvdec 两臂的 cycles 差远超各自噪声（判据可用性下限）
"""
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import statistics
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Win32 周期账本 ────────────────────────────────────────────────────
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.QueryProcessCycleTime.argtypes = (wintypes.HANDLE,
                                       ctypes.POINTER(ctypes.c_ulonglong))
# QueryProcessCycleTime = 该进程所有线程的周期计数（不变 TSC 刻度），
# 只计在 CPU 上运行的时间；粒度即 TSC，远细于 GetProcessTimes 的
# 15.6ms 量子。


def process_cycles() -> int:
    c = ctypes.c_ulonglong(0)
    if not _k32.QueryProcessCycleTime(_k32.GetCurrentProcess(),
                                      ctypes.byref(c)):
        raise ctypes.WinError(ctypes.get_last_error())
    return c.value


def machine_busy(t0, t1) -> float:
    """两次 psutil.cpu_times() 快照间的整机忙时占比（背景干扰的语境列）。

    Windows 的 scputimes 没有 nice/iowait 等字段 → 按 t0 的实际字段名取交集。
    """
    d = [getattr(t1, n) - getattr(t0, n) for n in t0._fields]
    total = sum(d)
    idle = getattr(t1, "idle", 0.0) - getattr(t0, "idle", 0.0)
    return 1.0 - (idle / total if total > 0 else 0.0)


# ── 负载子进程（独立进程：只抢 CPU，不进本进程周期账本）──────────────
def _spin_forever() -> None:  # pragma: no cover - 子进程
    # 不能用 `while not event.is_set(): pass`——Event.is_set() 走 Condition
    # 锁（Windows 信号量 acquire），实测每迭代 ~80% 时间阻塞在等待里，
    # 4 个"自旋"只烧 ~20% LP（2026-10-09 _tmp_spintest3/4 对比实测）。
    while True:
        pass


class LoadGroup:
    def __init__(self, n: int):
        self.n = n
        self._ctx = mp.get_context("spawn")
        self._ps: list = []

    def __enter__(self):
        for _ in range(self.n):
            p = self._ctx.Process(target=_spin_forever, daemon=True)
            p.start()
            self._ps.append(p)
        time.sleep(1.0)   # 让负载进程上 CPU 后再开测
        return self

    def __exit__(self, *exc):
        for p in self._ps:
            p.terminate()
            p.join(timeout=3)
        self._ps.clear()
        return False


# ── 工作负载 ─────────────────────────────────────────────────────────
def calib_work() -> int:
    """合成负载：固定种子矩阵乘（200 次 512³）+ 纯 Python 单线程变体。"""
    import numpy as np
    rng = np.random.default_rng(0)
    a = rng.standard_normal((512, 512), dtype=np.float32)
    b = a.copy()
    chk = 0.0
    for _ in range(200):
        c = a @ b
        chk += float(c[0, 0])
    s = 0
    for i in range(1_000_000):
        s += i * i
    return int(chk) + (s & 1)


def engine_work(vid: str, backend: str, frames: int) -> float:
    """真实负载：FieldExtractor 短窗口（stride=1 铁律，构造计入门内）。"""
    from video_ocr_engine import FieldExtractor
    roi = (843, 993, 948, 1025)
    ex = FieldExtractor(vid, roi, frame_start=0, frame_end=frames,
                        decode_backend=backend, ocr_backend="cpu")
    r = ex.extract()
    return float(len(r.segments))


# ── 测量循环 ─────────────────────────────────────────────────────────
def measure(label: str, work, reps: int, load: int) -> dict:
    walls: list[float] = []
    cycles: list[int] = []
    busy: list[float] = []
    with LoadGroup(load):
        import psutil
        work()   # 预热：进程内首次运行的冷效应（频率爬升/线程池展开）不进测量
        for i in range(reps):
            psutil.cpu_times()   # 清内核累计快照缓存（psutil 内部有 interval 缓存）
            c0, t0, b0 = process_cycles(), time.perf_counter(), psutil.cpu_times()
            work()
            c1, t1, b1 = process_cycles(), time.perf_counter(), psutil.cpu_times()
            walls.append((t1 - t0) * 1000)
            cycles.append(c1 - c0)
            busy.append(machine_busy(b0, b1))
            print("    rep%-2d wall=%8.1fms cycles=%12d effGHz=%.2f busy=%.0f%%"
                  % (i, walls[-1], cycles[-1],
                     cycles[-1] / ((t1 - t0) * 1e9), busy[-1] * 100))
    def cv(xs):
        return statistics.stdev(xs) / statistics.mean(xs) * 100 if len(xs) > 1 else 0.0
    row = dict(label=label, load=load,
               wall_ms=statistics.mean(walls), wall_cv=cv(walls),
               cycles=statistics.mean(cycles), cycles_cv=cv(cycles),
               busy=statistics.mean(busy))
    print("  [%s load=%d] wall %.0fms CV %.2f%% | cycles %.3fG CV %.2f%% | busy %.0f%%"
          % (label, load, row["wall_ms"], row["wall_cv"],
             row["cycles"] / 1e9, row["cycles_cv"], row["busy"] * 100))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true", help="合成负载（不引引擎）")
    ap.add_argument("--backend", choices=("cpu", "nvdec"), default=None)
    ap.add_argument("--load", type=int, default=None)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    if args.calib:
        rows = [measure("matmul", calib_work, args.reps, load)
                for load in (0, 8, 32)]
    else:
        import os
        vid = args.video or os.path.join(
            os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"),
            "test5.mp4")
        if args.backend is not None and args.load is not None:
            conds = [(args.backend, args.load)]
        elif args.backend is not None:
            conds = [(args.backend, l) for l in (0, 8, 32)]
        else:
            conds = [("cpu", 0), ("cpu", 8), ("cpu", 32),
                     ("nvdec", 0), ("nvdec", 32)]
        rows = []
        for backend, load in conds:
            w = (lambda b: lambda: engine_work(vid, b, args.frames))(backend)
            # 预热一次（引擎池/NVRTC 冷启动不进测量）
            engine_work(vid, backend, args.frames)
            rows.append(measure(backend, w, args.reps, load))

    print("\n== 汇总（负载免疫性判据：cycles CV ≪ wall CV；守恒性：load 臂 cycles 均值 ≈ idle 臂）==")
    base = {r["label"]: r["cycles"] for r in rows if r["load"] == 0}
    for r in rows:
        ratio = (r["cycles"] / base[r["label"]]
                 if r["label"] in base and r["load"] else 1.0)
        print("%-8s load=%-2d wall %7.0fms CV %5.2f%% | cycles %6.3fG CV %4.2f%% | cycles/idle=%.3f"
              % (r["label"], r["load"], r["wall_ms"], r["wall_cv"],
                 r["cycles"] / 1e9, r["cycles_cv"], ratio))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mp.freeze_support()
    raise SystemExit(main())
