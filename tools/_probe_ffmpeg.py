"""临时探针 C：用系统 ffmpeg 测量解码天花板（对照 decord）。

每种模式跑两轮取较快者，解析 ffmpeg -stats 的 frame= 计数与墙钟。
模式：
  nvdec_hw   -hwaccel cuda -hwaccel_output_format cuda   （帧留显存）
  nvdec_host -hwaccel cuda                               （帧回内存）
  cpu1..cpuN 纯软解（-threads N）
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time

FFMPEG = r"D:\Software\ffmpeg8\bin\ffmpeg.exe"


def run(args, label):
    t0 = time.perf_counter()
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error",
                        "-stats"] + args + ["-f", "null", "-"],
                       capture_output=True, text=True, errors="ignore")
    dt = time.perf_counter() - t0
    err = p.stderr or ""
    m = re.findall(r"frame=\s*(\d+)", err)
    n = int(m[-1]) if m else 0
    # -stats 最终行含 speed
    sp = re.findall(r"speed=\s*([\d.]+)x", err)
    print(f"  {label:26s}: {n:6d} 帧 / {dt:6.3f}s = {n/dt:8.0f} fps "
          f"({dt/max(n,1)*1e3:.4f} ms/帧)  rc={p.returncode}"
          + (f"  speed={sp[-1]}x" if sp else ""))
    if p.returncode != 0 and n == 0:
        print(f"      stderr: {err.strip()[-300:]}")
    return n / dt if dt > 0 and n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test5.mp4")
    ap.add_argument("--frames", type=int, default=3000)
    a = ap.parse_args()
    v = ["-i", a.video, "-frames:v", str(a.frames), "-an", "-sn"]
    print(f"=== ffmpeg 解码吞吐：{a.video} ({a.frames} 帧) ===")
    run(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] + v,
        "NVDEC(帧留显存)")
    run(["-hwaccel", "cuda"] + v, "NVDEC(帧回内存)")
    for t in (1, 4, 8, 16, 0):
        run(["-threads", str(t)] + v, f"CPU 软解 threads={t or 'auto'}")


if __name__ == "__main__":
    main()
