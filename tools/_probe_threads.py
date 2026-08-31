"""临时探针 D：decord CPU 软解线程数扫描（对照 ffmpeg）。

对照：ffmpeg 8 线程 1195 fps / 16 线程 1884 fps / auto 1938 fps。
decord 默认 = DECORD_FFMPEG_THREAD_COUNT = clamp(hw/4, 2, 8) = 8（32 线程机）。
验证：把线程数提到 16/0 是否能拿到 ffmpeg 同级的 ~1900fps。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# WORKER 以 `python -c` 执行，-c 下 __file__ 未定义，
# 故本进程算好根目录后经 PROBE_ROOT 传给子进程。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROBE_ROOT"] = ROOT  # 供 `python -c` 的 WORKER 子进程使用
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


PY = sys.executable
WORKER = r"""
import os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
nt = sys.argv[1]          # 传给 VideoReader 的 num_threads（"none" = 不传）
path, roi = sys.argv[2], tuple(int(x) for x in sys.argv[3].split(","))
n = int(sys.argv[4])
from decord import VideoReader, cpu
kw = {} if nt == "none" else {"num_threads": int(nt)}
vr = VideoReader(path, ctx=cpu(0), output_format="gray",
                 roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1), **kw)
frames = list(range(0, n))
vr.get_batch(frames[:16], roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1)).asnumpy()
t0 = time.perf_counter()
for i in range(0, len(frames), 64):
    vr.get_batch(frames[i:i+64], roi=(roi[0], roi[1], roi[2] + 1, roi[3] + 1)).asnumpy()
dt = time.perf_counter() - t0
print(f"{len(frames)/dt:.0f}")
"""


def run(nt, video, roi, n, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run([PY, "-c", WORKER, nt, video, roi, str(n)],
                       capture_output=True, text=True, env=e)
    out = (p.stdout or "").strip().splitlines()
    if p.returncode != 0 or not out:
        return f"FAIL {p.stderr.strip()[-200:]}"
    return f"{float(out[-1]):.0f} fps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(_VIDEO_DIR / "test5.mp4"))
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--frames", type=int, default=3000)
    a = ap.parse_args()
    roi = ",".join(str(x) for x in
                   tuple(int(x) for x in a.roi.split(",")))
    print(f"=== decord CPU 软解线程扫描：{os.path.basename(a.video)} ===")
    print("  -- num_threads 显式传入 --")
    for nt in ("none", "2", "4", "8", "12", "16", "24"):
        print(f"    num_threads={nt:>4s}: "
              f"{run(nt, a.video, roi, a.frames)}")


if __name__ == "__main__":
    main()
