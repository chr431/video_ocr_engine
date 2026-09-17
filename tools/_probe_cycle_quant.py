"""测量学证据（2026-09-17 §8 固化）：CPU 时间读数的量化与 cycle 计数语义。

三段证据，供性能监测系统重设计（resources.py 换 cycle 计数、bench 门禁）引用：

A. `time.process_time()` 的 tick 量化：高频重采只出现一个非零增量
   （本机预期 15.625ms = 1/64s，Windows 调度器节拍）——短相位 dcpu 只有
   1~2 个 tick 时 cores_avg 是量化伪影（±49%~100%）。
B. `QueryProcessCycleTime` / `QueryThreadCycleTime` 可用性与语义：
   伪句柄直传（进程 -1 / 线程 -2，绕开 GetCurrentProcess restype 截断陷阱）；
   sleep 不增（只计实际执行）；多线程是各线程之和；分辨率 ≈ 单周期。
C. cycles/wall 的稳定性（§8.1 复测）：固定负载下 wall 与 cycles 各自波动大
   （调度/抢占），但比值几乎恒定——证明「换时钟消不掉噪声」且 cycle 计数
   与墙钟在本机是同一个量的两种单位；同时验证「全程 cycles ÷ 全程
   process_time」的自校准频率换算法在两个窗口给出一致的 GHz。

产物：bench/cycle_quant.json
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parents[1] / "bench" / "cycle_quant.json"


def _kernel32():
    import ctypes
    return ctypes.windll.kernel32  # type: ignore[attr-defined]


def process_cycles() -> int:
    """进程级 cycle 计数（伪句柄 -1 直传，不经过 GetCurrentProcess）。"""
    import ctypes
    k32 = _kernel32()
    k32.QueryProcessCycleTime.restype = ctypes.c_int
    c = ctypes.c_ulonglong()
    if not k32.QueryProcessCycleTime(ctypes.c_void_p(-1), ctypes.byref(c)):
        raise OSError("QueryProcessCycleTime 失败")
    return c.value


def thread_cycles() -> int:
    """当前线程 cycle 计数（伪句柄 -2 直传）。"""
    import ctypes
    k32 = _kernel32()
    k32.QueryThreadCycleTime.restype = ctypes.c_int
    c = ctypes.c_ulonglong()
    if not k32.QueryThreadCycleTime(ctypes.c_void_p(-2), ctypes.byref(c)):
        raise OSError("QueryThreadCycleTime 失败")
    return c.value


# ── A. process_time tick 量化 ───────────────────────────────────────────
def probe_process_time_ticks(n: int = 200_000) -> dict:
    t0 = time.process_time()
    deltas: dict[float, int] = {}
    last = t0
    for _ in range(n):
        t = time.process_time()
        d = t - last
        if d > 0:
            deltas[round(d, 9)] = deltas.get(round(d, 9), 0) + 1
            last = t
    min_tick = min(deltas) if deltas else None
    return {"samples": n, "distinct_deltas": len(deltas),
            "min_nonzero_tick_s": min_tick,
            "tick_counts": {str(k): v for k, v in
                            sorted(deltas.items())[:8]}}


# ── B. cycle 计数语义 ───────────────────────────────────────────────────
def _spin(iterations: int) -> int:
    s = 0
    for i in range(iterations):
        s += i * i
    return s


def probe_cycle_semantics() -> dict:
    out: dict = {}
    # sleep 不增（只计实际执行，语义=CPU 时间而非墙钟）
    c0, w0 = process_cycles(), time.perf_counter()
    time.sleep(0.5)
    out["sleep_0p5s_cycle_delta"] = process_cycles() - c0
    out["sleep_0p5s_wall_s"] = round(time.perf_counter() - w0, 3)
    # 单线程负载：推算隐含 GHz（cycle/s）
    c0, w0 = process_cycles(), time.perf_counter()
    _spin(3_000_000)
    dc, dw = process_cycles() - c0, time.perf_counter() - w0
    out["single_thread"] = {"cycles": dc, "wall_s": round(dw, 4),
                            "implied_ghz": round(dc / dw / 1e9, 3)}
    # 多线程求和：4 线程各忙 ~0.5s → 总 cycle ≈ 4×单线程速率×wall
    N, iters = 4, 8_000_000
    c0, w0 = process_cycles(), time.perf_counter()
    ts = [threading.Thread(target=_spin, args=(iters,)) for _ in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dc, dw = process_cycles() - c0, time.perf_counter() - w0
    out["four_threads"] = {"cycles": dc, "wall_s": round(dw, 4),
                           "implied_cores_ghz": round(dc / dw / 1e9, 3),
                           "per_thread_ghz": round(dc / dw / 1e9 / N, 3)}
    # 线程级计数：主线程在别人忙时 sleep → 主线程增量≈0、进程增量>0
    tc0, pc0 = thread_cycles(), process_cycles()
    worker = threading.Thread(target=_spin, args=(4_000_000,))
    worker.start()
    time.sleep(0.2)
    worker.join()
    out["thread_isolation"] = {
        "main_thread_cycles_while_worker_busy": thread_cycles() - tc0,
        "process_cycles_delta": process_cycles() - pc0}
    return out


# ── C. cycles/wall 稳定性 + 自校准频率换算 ─────────────────────────────
def probe_cycles_wall_cv(reps: int = 15) -> dict:
    rows = []
    for _ in range(reps):
        c0, w0 = process_cycles(), time.perf_counter()
        _spin(5_000_000)
        rows.append((time.perf_counter() - w0, process_cycles() - c0))

    def cv(xs):
        m = statistics.fmean(xs)
        return statistics.pstdev(xs) / m * 100 if m else 0.0

    walls = [r[0] for r in rows]
    cycs = [r[1] for r in rows]
    ratios = [c / w for w, c in rows]

    def corr(a, b):
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den else 0.0

    # 自校准频率换算：两个独立窗口各算 implied GHz，看一致性
    # （resources.py 的 per_phase 换算 = Δcycles/Δwall ÷ f_run，f_run 就这么来）
    half = reps // 2
    g1 = sum(cycs[:half]) / sum(walls[:half]) / 1e9
    g2 = sum(cycs[half:]) / sum(walls[half:]) / 1e9
    return {"reps": reps, "wall_cv_pct": round(cv(walls), 3),
            "cycles_cv_pct": round(cv(cycs), 3),
            "cycles_per_wall_cv_pct": round(cv(ratios), 3),
            "corr_cycles_wall": round(corr(walls, cycs), 4),
            "selfcal_gHz_window1": round(g1, 3),
            "selfcal_gHz_window2": round(g2, 3),
            "selfcal_rel_diff_pct": round(abs(g1 - g2) / ((g1 + g2) / 2) * 100, 3)}


def main() -> int:
    print("A. process_time tick 量化 …")
    a = probe_process_time_ticks()
    print("  最小非零增量 = %s s（%d 次采样中 %d 种增量）"
          % (a["min_nonzero_tick_s"], a["samples"], a["distinct_deltas"]))
    print("B. cycle 计数语义 …")
    b = probe_cycle_semantics()
    for k, v in b.items():
        print("  %-38s %s" % (k, v))
    print("C. cycles/wall 稳定性（固定负载 15 次）…")
    c = probe_cycles_wall_cv()
    for k, v in c.items():
        print("  %-38s %s" % (k, v))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(
        {"process_time_ticks": a, "cycle_semantics": b,
         "cycles_wall_cv": c, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("→ %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
