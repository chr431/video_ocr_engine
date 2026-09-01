"""隔离测量：NVDEC 解码对 CPU 软解吞吐的干扰（host CPU 争用）。

为什么需要
----------
hybrid 里 CPU 生产者只有 ~1000fps，而同样线程数**单跑**是 ~1807fps
（`_probe_roi_decode.py` 实测）。差的 47% 到底是谁造成的？候选：

    (a) NVDEC 的 host 侧（demux / 提交 / D2H 拷贝）抢 host CPU
    (b) TRT 消费者抢 host CPU
    (c) 两者叠加

本探针在同一进程内跑：
    A 组  CPU 软解单跑                → cpu_alone
    B 组  CPU 软解 ∥ NVDEC 解码        → cpu_with_nvdec
    C 组  CPU 软解 ∥ 一个纯 host 忙等负载 → cpu_with_spin（对照：模拟"抢核"）

B 组相对 A 组的跌幅 = NVDEC 的 host 干扰成本。

用法
----
    python tools/_probe_nvdec_interference.py --video X --roi A,B,C,D
        [--frames 2000] [--cpu-threads 12] [--runs 2]
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


def cpu_decode(video: str, roi: tuple, n: int, threads: int,
               stop: threading.Event | None = None) -> tuple[float, float]:
    """连续解 n 帧，返回 (总耗时, fps)。stop 置位时提前退出。"""
    vr = VideoReader(video, ctx=cpu(0), num_threads=threads, roi=roi)
    idx = list(range(min(n, len(vr))))
    vr.get_batch(idx[:32], roi=roi).asnumpy()
    t0 = time.perf_counter()
    i, done = 32, 0
    while i < len(idx):
        if stop is not None and stop.is_set():
            break
        be = min(i + 256, len(idx))
        vr.get_batch(idx[i:be], roi=roi).asnumpy()
        done += be - i
        i = be
    dt = time.perf_counter() - t0
    return dt, (done / dt if dt > 0 else 0.0)


def nvdec_decode(video: str, roi: tuple, n: int, stop: threading.Event) -> float:
    """NVDEC 解码循环，直到 stop 置位。返回解码帧数。"""
    vr = VideoReader(video, ctx=gpu(0), roi=roi)
    idx = list(range(min(n, len(vr))))
    i, done = 0, 0
    while not stop.is_set():
        be = min(i + 64, len(idx))
        if be <= i:
            i = 0                      # 循环回放，保持 NVDEC 持续占用
            continue
        vr.get_batch(idx[i:be], roi=roi).asnumpy()
        done += be - i
        i = be
    return done


def spin(stop: threading.Event, nthreads: int) -> None:
    """纯 host 忙等负载（对照组）：模拟"被别的线程抢核"。"""
    def one():
        x = 0
        while not stop.is_set():
            x = (x + 1) % 1000003
    ts = [threading.Thread(target=one, daemon=True) for _ in range(nthreads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--cpu-threads", type=int, default=12)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    roi = tuple(int(v) for v in args.roi.split(","))
    roi = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)
    print(f"视频={Path(args.video).name} roi={roi} frames={args.frames} "
          f"cpuT={args.cpu_threads} runs={args.runs}\n")

    def measure(with_nvdec: bool, with_spin: int = 0) -> float:
        rates = []
        for _ in range(args.runs):
            stop = threading.Event()
            bg = []
            if with_nvdec:
                bg.append(threading.Thread(
                    target=nvdec_decode,
                    args=(args.video, roi, args.frames, stop), daemon=True))
            if with_spin:
                bg.append(threading.Thread(
                    target=spin, args=(stop, with_spin), daemon=True))
            for t in bg:
                t.start()
            time.sleep(0.15)          # 让背景负载先起来
            _, fps = cpu_decode(args.video, roi, args.frames,
                                args.cpu_threads, stop)
            stop.set()
            for t in bg:
                t.join(timeout=2.0)
            rates.append(fps)
        return statistics.median(rates)

    alone = measure(False)
    print("  A  CPU 软解单跑                    %6.0f fps" % alone)
    with_nv = measure(True)
    print("  B  CPU 软解 ∥ NVDEC               %6.0f fps" % with_nv)
    with_sp = measure(False, with_spin=2)
    print("  C  CPU 软解 ∥ 2 个忙等线程（对照）  %6.0f fps" % with_sp)

    print("\n  NVDEC 干扰成本：CPU 吞吐 %+.1f%%" % ((with_nv / alone - 1) * 100))
    print("  对照（等量的纯 host 抢核）：%+.1f%%" % ((with_sp / alone - 1) * 100))
    print("\n  判读：若 B 的跌幅接近 C，说明 NVDEC 的开销主要就是抢 host 核；")
    print("        若 B 跌幅远大于 C，说明还有别的机制（如内存带宽 / PCIe）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
