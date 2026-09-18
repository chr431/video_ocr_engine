"""hard-window 回归复现器（2026-09-19 窗口缺陷轮的取证工具）。

用法：python tools/_probe_window_repro.py <视频路径> <帧数> [win]
ctx = hybrid（CPU-out，gold A 组口径）。win = set_decode_window(n)。
健康判据：got==n（少交付=缺陷）；60~120s 不返回=挂死。
覆盖矩阵（全部应 rc=0 且 got==n）：test5/3000, test6_h264/{3000,6660,
12000,23970}, test6_hevc/12000, test6.mp4/12000 × {win, 无窗}。
"""
import ctypes, os, sys, time
ctypes.windll.kernel32.SetErrorMode(0x0004)
sys.stdout.reconfigure(encoding="utf-8")
from decord import VideoReader, hybrid, hybrid_gpu
path, n = sys.argv[1], int(sys.argv[2])
win = len(sys.argv) > 3 and sys.argv[3] == "win"
t0 = time.perf_counter()
print("ctor...", flush=True)
vr = VideoReader(path, ctx=(hybrid_gpu if len(sys.argv) > 4 and sys.argv[4] == "gpu" else hybrid)(0), output_format="gray",
                 roi=(843, 993, 949, 1026), num_threads=32)
print("ctor done %.3f" % (time.perf_counter() - t0), flush=True)
if win:
    vr.set_decode_window(n)
vr.seek(0)
print("seek done %.3f" % (time.perf_counter() - t0), flush=True)
got = 0
while got < n:
    e = min(got + 64, n)
    b = vr.get_batch(list(range(got, e)))
    got += b.shape[0]
    del b
    print("  got %d %.3f" % (got, time.perf_counter() - t0), flush=True)
print("CPU-OUT %s n=%d win=%s wall=%.3f got=%d" % (
    os.path.basename(path), n, win, time.perf_counter() - t0, got))
vr.close()
