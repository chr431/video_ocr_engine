"""零开销确认：busy 计数器 dll vs wheel 交错配对（hevc/h264 hybrid decode-only）。

每对内轮转先后顺序（防位置偏差——2026-09-12 的教训：固定先 A 后 B 会把
漂移记到 B 头上）。各 4 对，独立子进程（每次 del vr 出一份 stats）。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# fork 构建树（含 busy 计数器）：与 _probe_hol_stats 同一 env 约定，不写死
FORK = os.environ.get("DECORD_FORK_BUILD", "")
sys.stdout.reconfigure(encoding="utf-8")
if not FORK:
    print("需 DECORD_FORK_BUILD 指向含 busy 计数器的 fork 构建树"
          "（≥6da2957）")
    raise SystemExit(2)

WORKER = r"""
import os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
import decord
from decord import VideoReader, cpu
vid = os.environ["PROBE_VID"]
vr = VideoReader(vid, ctx=cpu(0), num_threads=32)  # 占位；实际 ctx 在下方
"""
# 直接复用 _probe_hol_stats 的 worker 口径太重；这里用引擎无关的
# decode-only：gray + 批 64 + num_threads=32（与 hol_stats 同口径）。
WORKER = r"""
import os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
import decord
from decord import VideoReader
vid = os.environ["PROBE_VID"]
ROI = tuple(int(x) for x in os.environ["PROBE_ROI"].split(","))
ctx = decord.hybrid(0)
vr = VideoReader(vid, ctx=ctx, num_threads=32,
                 output_format="gray")
x1, y1, x2, y2 = ROI
n = len(vr)
t0 = time.perf_counter()
got = 0
for s in range(0, n, 64):
    idx = list(range(s, min(s + 64, n)))
    arr = vr.get_batch(idx).asnumpy()
    got += len(idx)
wall = time.perf_counter() - t0
print("RES %d %.4f" % (got, wall))
"""

CASES = {"h264": (r"D:\Videos\racelog_test\test5.mp4", "843,993,949,1026"),
         "hevc": (r"D:\Videos\racelog_test\test6_hevc.mp4", "841,994,950,1027")}


def run(case, use_fork):
    vid, roi = CASES[case]
    env = dict(os.environ)
    env["PROBE_ROOT"] = str(ROOT)
    env["PROBE_VID"] = vid
    env["PROBE_ROI"] = roi
    if use_fork:
        env["DECORD_LIBRARY_PATH"] = FORK
    else:
        env.pop("DECORD_LIBRARY_PATH", None)
    p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                       capture_output=True, text=True, timeout=300)
    for ln in p.stdout.splitlines():
        if ln.startswith("RES "):
            got, wall = ln[4:].split()
            return int(got) / float(wall)
    return None


def main():
    for case in CASES:
        a_l, b_l, sign = [], [], 0
        for i in range(4):
            fa = (i % 2 == 0)          # 轮转先后
            r1 = run(case, fa)
            r2 = run(case, not fa)
            v_fork = r1 if fa else r2
            v_whl = r2 if fa else r1
            a_l.append(v_fork); b_l.append(v_whl)
            if i:
                sign += 1 if v_fork > v_whl else 0
        med = lambda v: sorted(v)[len(v) // 2]
        print("%-5s fork=%-7.0f wheel=%-7.0f fps  Δ=%+.2f%%  fork更快 %d/3"
              % (case, med(a_l), med(b_l),
                 (med(a_l) - med(b_l)) / med(b_l) * 100, sign))


if __name__ == "__main__":
    main()
