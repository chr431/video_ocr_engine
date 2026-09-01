"""隔离测量：并发解码的**相互拖慢**到底来自哪一层。

问题
----
hybrid 里 CPU 生产者只有 ~966fps，而同样线程数单跑 ~1855fps（−48%）。
但 §22.6 实测：CPU ∥ NVDEC 只掉 **3.3%**，CPU ∥ 2 个忙等线程却掉 41.9%。

这两个数**互相矛盾** —— 若真凶是"host CPU 被抢"，NVDEC 那侧也该掉很多。
所以"抢核"解释不通，需要把候选一层层分开：

    候选 1  纯 host 抢核（忙等线程）            → D 组
    候选 2  NVDEC 路径的特定开销                → C 组
    候选 3  **并发解码本身**的共享资源竞争      → B 组 ← 本探针的重点
    候选 4  内存带宽饱和                        → E 组

B 组是判据：让**两个 CPU reader** 并发解（对端完全不是 NVDEC）。
· 若每路掉到 ~1000fps → 拖慢来自 CPU 解码路径的**共享资源**（带宽 / decord
  内部锁 / 分配器），与 NVDEC 无关 → 支持"decord 侧更底层的原因"。
· 若每路仍 ~1855fps → 那 hybrid 里的 966fps 必然另有解释（如流水线）。

用法
----
    python tools/_probe_decode_contention.py --video X --roi A,B,C,D
        [--frames 2000] [--threads 12] [--runs 2]
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

from decord import VideoReader, cpu, gpu  # noqa: E402

BARRIER_READY = "ready"


def decode_loop(video: str, roi: tuple, n: int, threads: int, ctx,
                stop: threading.Event, out: dict, key: str,
                budget: int = 0) -> None:
    """连续解码；`budget>0` 时解够这么多帧就返回，否则一直解到 stop 置位。

    ⚠️ 必须区分"被测主线程"与"背景陪跑线程"：主线程要能**自己结束**，
    否则 `stop.set()` 写在它返回之后 → 互相等待 → 死锁（第一版就挂了 10 分钟）。
    """
    try:
        kw = {"num_threads": threads} if threads > 0 else {}
        vr = VideoReader(video, ctx=ctx, roi=roi, **kw)
        idx = list(range(min(n, len(vr))))
        vr.get_batch(idx[:32], roi=roi).asnumpy()
        t0 = time.perf_counter()
        i, done = 32, 0
        while not stop.is_set():
            if budget and done >= budget:
                break
            if i >= len(idx):
                i = 0                       # 循环回放，保持持续负载
            be = min(i + 256, len(idx))
            if budget:
                be = min(be, i + max(1, budget - done))
            vr.get_batch(idx[i:be], roi=roi).asnumpy()
            done += be - i
            i = be
        out[key] = (done, time.perf_counter() - t0)
    except Exception as e:                  # 单路失败不应拖垮整轮测量
        out[key] = (0, 0.0)
        out[key + "_err"] = str(e)


def blas_hog(stop: threading.Event, size: int = 1200) -> None:
    """**释放 GIL** 的 CPU 负载：numpy BLAS 在 C 层运算，不吃 GIL。

    与 `spin()`（纯 Python 循环 → 持续持有 GIL）形成对照。两组都大约吃掉
    2 个核的算力，唯一差别就是**抢不抢 GIL**。
    """
    try:
        import numpy as np
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        while not stop.is_set():
            _ = a @ b
    except ImportError:
        return


def spin(stop: threading.Event, nthreads: int) -> None:
    """纯 Python 忙等 —— **持有 GIL**（每 switchinterval 才让出一次）。"""


def bandwidth_hog(stop: threading.Event, mb: int = 512) -> None:
    """顺序读写一块大内存，制造内存带宽压力（不吃满 CPU 算力）。"""
    try:
        import numpy as np
        buf = np.ones(mb * 1024 * 1024 // 8, dtype=np.float64)
        while not stop.is_set():
            buf *= 1.0000001               # 必须真正读写，不能被优化掉
            buf.sum()
    except ImportError:
        return


def run_case(threads: int, video: str, roi: tuple, n: int, runs: int,
             dual_cpu: bool, with_nvdec: bool, nspin: int, hog: bool,
             nblas: int = 0):
    per_thread = []
    for _ in range(runs):
        stop = threading.Event()
        out: dict = {}
        bg = []
        if dual_cpu:
            bg.append(threading.Thread(
                target=decode_loop,
                args=(video, roi, n, threads, cpu(0), stop, out, "peer"), daemon=True))
        if with_nvdec:
            bg.append(threading.Thread(
                target=decode_loop,
                args=(video, roi, n, 0, gpu(0), stop, out, "nv"), daemon=True))
        if nspin:
            bg.append(threading.Thread(target=spin, args=(stop, nspin), daemon=True))
        if hog:
            bg.append(threading.Thread(
                target=bandwidth_hog, args=(stop,), daemon=True))
        for _ in range(nblas):
            bg.append(threading.Thread(
                target=blas_hog, args=(stop,), daemon=True))
        for t in bg:
            t.start()
        time.sleep(0.2)                      # 让背景负载先进入稳态
        decode_loop(video, roi, n, threads, cpu(0), stop, out, "main",
                    budget=n)                # 主线程自己解够 n 帧就返回
        stop.set()
        for t in bg:
            t.join(timeout=3.0)
        done, dt = out.get("main", (0, 0.0))
        per_thread.append(done / dt if dt > 0 else 0.0)
    return statistics.median(per_thread)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=12, help="每路 CPU 解码线程数")
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    roi = tuple(int(v) for v in args.roi.split(","))
    roi = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)
    v, n, T, R = args.video, args.frames, args.threads, args.runs
    print(f"视频={Path(v).name} roi={roi} frames={n} 每路CPU线程={T} runs={R}\n")

    def c(**kw):
        return run_case(T, v, roi, n, R, **kw)

    a = c(dual_cpu=False, with_nvdec=False, nspin=0, hog=False)
    print("  A  CPU 单跑                       %6.0f fps   (基准)" % a)
    b = c(dual_cpu=True, with_nvdec=False, nspin=0, hog=False)
    print("  B  CPU ∥ CPU（两路并发）           %6.0f fps   %+.1f%%"
          % (b, (b / a - 1) * 100))
    cc = c(dual_cpu=False, with_nvdec=True, nspin=0, hog=False)
    print("  C  CPU ∥ NVDEC                    %6.0f fps   %+.1f%%"
          % (cc, (cc / a - 1) * 100))
    d = c(dual_cpu=False, with_nvdec=False, nspin=2, hog=False)
    print("  D  CPU ∥ 2 个忙等线程              %6.0f fps   %+.1f%%"
          % (d, (d / a - 1) * 100))
    e = c(dual_cpu=False, with_nvdec=False, nspin=0, hog=True)
    print("  E  CPU ∥ 内存带宽 hog              %6.0f fps   %+.1f%%"
          % (e, (e / a - 1) * 100))
    f = c(dual_cpu=False, with_nvdec=False, nspin=0, hog=False, nblas=2)
    print("  F  CPU ∥ 2 个 BLAS hog（**放 GIL**）%6.0f fps   %+.1f%%"
          % (f, (f / a - 1) * 100))

    print("\n  判读：")
    print("    D 与 F 都约吃 2 个核，唯一差别是**抢不抢 GIL**：")
    if d < a * 0.75 and f > a * 0.9:
        print("      → D 大跌、F 几乎不跌 ⇒ **元凶是 GIL 争抢，不是 CPU 算力**。")
        print("        这解释了「任务管理器只看到 ~50% 占用却有争用」：")
        print("        被 GIL 阻塞的线程在**等待**，不产生 CPU 占用。")
    elif abs(d - f) < a * 0.1:
        print("      → D 与 F 跌幅相近 ⇒ 是真实算力竞争，与 GIL 无关。")
    if b < a * 0.75 and cc > a * 0.9:
        print("    B 大跌而 C 几乎不跌 → 第二路 CPU 解码的代价远大于 NVDEC。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
