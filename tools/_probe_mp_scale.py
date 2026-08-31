"""多进程流式带宽 worker：在**固定时间窗口**内持续施压，报告搬运的总字节数。

为什么必须这样：早期用「各进程各自计时再相加」的写法是错的 —— 父进程
`communicate()` 会阻塞等第一个进程跑完，第二个进程才刚做完 1.5GB 的数组
分配与预热，两者实际**串行**，于是"2 进程 107.7 GB/s"是假象（真实重叠只有
~65 GB/s）。正确做法是让所有进程在**同一个时间窗口**内各自累计字节数，
聚合带宽 = Σbytes / 窗口长度。

## 两种负载口径（用于判别"内存带宽争用" vs "CPU 核争用"）

同一个 kernel，只改 `--total-mb` 就得到两种负载：

| 口径 | --total-mb | 工作集落点 | DRAM 流量 | CPU 占用 |
| --- | --- | --- | --- | --- |
| `dram` | 512 | 远超 64MB L3，必落 DRAM | 满 | 高 |
| `l2`   | 4   | 每线程 256KB，落在 1MB/核 的 L2 里 | ≈0 | 高 |
| `l3`   | 64  | 每线程 4MB，落在共享 64MB L3 里 | ≈0 | 高 |

`l2`/`l3` 是**对照组**：CPU 一样忙、DRAM 流量为零。若它们不伤 TRT 而
`dram` 伤，则退化确实由 DRAM 带宽引起；若一样伤，则只是核争用。
注意 `l2`/`l3` 报告的"字节数"是缓存层次内部的流量，**不代表 DRAM 流量**，
父进程对这两种口径不应把它当带宽用。

## ⚠️ 第二个陷阱：只靠 READY 同步还不够，必须统一「绝对起跑时刻」

第一版修法（子进程打 READY、父进程逐个 readline 后计时）在进程数少时看
不出问题，进程数一多就崩：父进程**串行**读 N 个 READY，每个子进程的
Python 启动 + 数组分配要几百毫秒，于是第 1 个进程的窗口早就跑完了，
父进程还在等第 N 个。各进程实际错开执行，聚合值虚高 —— 实测 24 进程
读出 **137 GB/s**，而本机双通道 DDR5-6000 理论峰值只有 96 GB/s，
物理上不可能。

正确做法两步：
1. 父进程收齐 READY 后，通过 **stdin 广播一个未来的绝对时刻** `START <epoch>`，
   所有子进程忙等到该时刻才开窗（严格同时起跑）。
2. 子进程回报 `bytes start_epoch end_epoch`，聚合带宽 = **Σbytes ÷
   所有窗口的并集跨度**，而不是 Σ(各进程自己的带宽)。后者在部分重叠时
   会静默虚高，并集口径则不会。
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

GB = float(1 << 30)
MB = float(1 << 20)


class Streamer:
    """流式 kernel，数组只分配一次（避免分配/缺页噪声）。"""

    # 口径：每元素搬多少字节进/出 DRAM
    #   copy  = 1 读 1 写           -> 2N
    #   triad = 2 读 1 写           -> 3N
    #   sum   = 纯读（归约）        -> 1N（通常能跑到比 copy 更高，用来交叉
    #                                  验证 55 GB/s 到底是不是 DRAM 天花板）
    #   scale = 原地乘（1 读 1 写） -> 2N
    MULT = {"copy": 2, "triad": 3, "sum": 1, "scale": 2}

    def __init__(self, total_mb: float, kernel: str = "copy"):
        self.kernel = kernel
        self.mult = self.MULT[kernel]
        n = int(total_mb * MB / 4)
        self.n = n
        self.a = np.ones(n, np.float32)
        self.b = np.full(n, 2.0, np.float32)
        self.c = np.empty(n, np.float32)
        if kernel == "copy":
            self.op = lambda A, B, C: np.copyto(C, A)      # noqa: E731
        elif kernel == "sum":
            self.op = lambda A, B, C: np.add(A.sum(), 0.0)  # noqa: E731
        elif kernel == "scale":
            self.op = lambda A, B, C: np.multiply(A, 1.000001, out=A)  # noqa: E731
        else:
            self.op = lambda A, B, C: np.add(A, B, out=C)  # noqa: E731
        self.op(self.a[:4096], self.b[:4096], self.c[:4096])   # 预热/缺页

    def run(self, nthread: int, iters: int = 6) -> tuple[float, float]:
        """返回 (搬运字节数, 耗时秒)。"""
        n = self.n
        step = (n // nthread) & ~16383
        op = self.op

        def w(lo: int, hi: int) -> None:
            A, B, C = self.a[lo:hi], self.b[lo:hi], self.c[lo:hi]
            for _ in range(iters):
                op(A, B, C)

        ts = [threading.Thread(
            target=w,
            args=(i * step, n if i == nthread - 1 else (i + 1) * step))
            for i in range(nthread)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        dt = time.perf_counter() - t0
        return self.mult * 4 * step * nthread * iters, dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-mb", type=float, default=512.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--kernel", default="copy", choices=["copy", "triad", "sum", "scale"])
    ap.add_argument("--window", type=float, default=0.0,
                    help=">0：在该时间窗口内持续施压，输出 '总字节 窗口秒'；"
                         "=0：单次测量，输出 GB/s")
    ap.add_argument("--loop", type=float, default=0.0,
                    help="--window 的别名（自检里当作「持续施压 N 秒」使用）")
    ap.add_argument("--ready", action="store_true",
                    help="分配完毕后打印 READY，然后从 stdin 等 START <epoch>")
    ap.add_argument("--timeseries", action="store_true",
                    help="按 --slice 秒切片输出瞬时带宽，看负载是不是突发式的")
    ap.add_argument("--slice", type=float, default=0.2)
    args = ap.parse_args()

    st = Streamer(args.total_mb, args.kernel)

    win = args.window if args.window > 0 else args.loop
    if win <= 0:
        best = 0.0
        for _ in range(3):
            b, t = st.run(args.threads)
            best = max(best, b / GB / t)
        print("%.2f" % best, flush=True)
        return 0

    if args.ready:
        print("READY", flush=True)
        deadline = float(sys.stdin.readline().split()[1])
        while time.time() < deadline:      # busy-wait 到统一时刻，保证严格同时起跑
            pass

    total = 0.0
    t0 = time.perf_counter()
    t0e = time.time()
    if args.timeseries:
        # 时间序列模式：把窗口切成 --slice 秒的小片，逐片报告"探针在这片里
        # 拿到了多少带宽"。探针是恒定满速的，所以瞬时带宽的**下跌部分**就
        # 对应同时段被别的负载抢走的量 —— 用它判断负载是平滑还是突发。
        # 发射条件必须是「距上次发射的累计时长 ≥ slice」。第一版错写成
        # 判断单次 run 调用的时长，而单次 run 只有 ~0.07s < slice，于是一条
        # TS 都没发出来（父进程收到 n=0）。
        sl = 0.0
        seg_t = time.perf_counter()
        seg_e = time.time()
        while time.perf_counter() - t0 < win:
            b, _ = st.run(args.threads, iters=2)
            total += b
            sl += b
            now = time.perf_counter()
            if now - seg_t >= args.slice:
                print("TS %.6f %.6f %.0f"
                      % (seg_e, seg_e + (now - seg_t), sl), flush=True)
                seg_t, seg_e, sl = now, time.time(), 0.0
    else:
        while time.perf_counter() - t0 < win:
            b, _ = st.run(args.threads, iters=4)
            total += b
    t1e = time.time()
    print("%.0f %.6f %.6f" % (total, t0e, t1e), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
