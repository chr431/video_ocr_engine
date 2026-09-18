"""_probe_ffmpeg_min_perf.py —— FFmpeg 极简集 vs GPL 全家桶的解码性能对照。

命题
----
RVTL frozen 用极简 FFmpeg（--disable-everything + 4 解码器 + libdav1d）
替换了 BtbN gpl-shared 全家桶（decord 部分 157→13MB）。需排除性能回退：
两个构建的解码内核（nasm SIMD 路径）与组件启用面不同。

口径
----
- 两份 decord 包副本（同 py、不同 FFmpeg DLL 集）：gpl = wheel 原装
  （BtbN gpl-shared），min = 极简重建（h264/hevc/vp9/libdav1d）。
- worker 子进程：sys.path 指向副本父目录，全片 ROI gray 批环
  （与 _probe_decode_rate 同口径：DECODE_BATCH_SIZE 粒度、首批剔除），
  输出 fps。
- 控制器：三码 × 两臂 × reps 交错（臂序逐 rep 正反交替），报告
  min-of-N 与逐轮对；NVDEC（gpu ctx）另测（demux 走 avformat，解码
  为原生 cuvid——预期零差异，作健全性对照）。

用法
----
    python tools/_probe_ffmpeg_min_perf.py [--reps 3] [--gpu-reps 1]
前提：D:/Software/ffmpeg_ab/decord_gpl、decord_min 两份副本已备
（生成段见 build 函数——直接 copy + DLL 覆盖）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

AB = Path(r"D:/Software/ffmpeg_ab")
VIDS = {"h264": (r"D:\Videos\racelog_test\test5.mp4", (843, 993, 948, 1025), 7761),
        "hevc": (r"D:\Videos\racelog_test\test6_hevc.mp4", (841, 994, 949, 1026), 23970),
        "av1": (r"D:\Videos\racelog_test\test6.mp4", (841, 994, 949, 1026), 23970)}

WORKER = r"""
import json, sys, time
pkg_parent, path, roi_s, n, ctx_name, nt = sys.argv[1:7]
nt = int(nt)
sys.path.insert(0, pkg_parent)
sys.stdout.reconfigure(encoding="utf-8")
import decord
from decord import VideoReader
roi = tuple(int(v) for v in roi_s.split(","))
n = int(n)
if ctx_name == "hybrid":
    ctx = decord.hybrid_gpu(0)
elif ctx_name == "gpu":
    ctx = decord.gpu(0)
else:
    ctx = decord.cpu(0)
# 口径轮：线程档由父进程按引擎策略传入（原硬编码 16/32）
kw = {} if ctx_name == "gpu" else {"num_threads": nt}
vr = VideoReader(path, ctx=ctx, output_format="gray", roi=roi, **kw)
from video_ocr_engine.config import constants as cfg
batch = int(cfg.DECODE_BATCH_SIZE)
first = None
t_sum = 0.0
f_sum = 0
for s in range(0, n, batch):
    e = min(s + batch, n)
    t0 = time.perf_counter()
    nd = vr.get_batch(list(range(s, e)))
    dt = time.perf_counter() - t0
    if first is None:
        first = dt
    else:
        t_sum += dt
        f_sum += (e - s)
print(json.dumps({"fps": f_sum / t_sum if t_sum else 0.0}))
"""


_RE_FRAMES = None


def run_arm(pkg_parent: str, codec: str, ctx_name: str):
    vid, roi, n = VIDS[codec]
    import os, re
    global _RE_FRAMES
    if _RE_FRAMES is None:
        _RE_FRAMES = re.compile(r"frames c=(\d+) g=(\d+)")
    env = {"PYTHONPATH": pkg_parent, "PROBE_ROOT": str(ROOT)}
    env = {**os.environ, **env}
    if ctx_name == "hybrid":
        env["DECORD_HYBRID_STATS"] = "1"
    from video_ocr_engine.config.decode_caliber import (
        decode_num_threads, roi_for_decord)
    roi = roi_for_decord(roi)   # 真值口径 → decord 半开（曾少 1px）
    nt = decode_num_threads(codec)
    p = subprocess.run([sys.executable, "-c", WORKER, pkg_parent, vid,
                        ",".join(map(str, roi)), str(n), ctx_name,
                        str(nt)],
                       capture_output=True, text=True, encoding="utf-8",
                       env=env, timeout=600)
    fps = None
    for ln in p.stdout.splitlines():
        if ln.startswith("{"):
            fps = json.loads(ln)["fps"]
    if fps is None:
        raise RuntimeError(p.stderr[-300:])
    m = _RE_FRAMES.search(p.stderr or "")
    cg = (int(m.group(1)), int(m.group(2))) if m else None
    return fps, cg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gpu-reps", type=int, default=1)
    args = ap.parse_args()
    gpl = str(AB / "A")          # A/decord = wheel 原装（GPL 全家桶）
    mn = str(AB / "B")           # B/decord = 极简 FFmpeg 集
    assert (AB / "A/decord").is_dir(), "先准备 A/decord 副本"
    assert (AB / "B/decord").is_dir(), "先准备 B/decord 副本"

    print("== CPU 软解（FFmpeg 解码内核——本轮验证对象）==")
    print("%-5s %-5s %10s %10s %8s   逐轮(min,gpl)" % ("codec", "arm", "min fps", "gpl fps", "min/gpl"))
    for codec in VIDS:
        per = {"min": [], "gpl": []}
        order = [("min", mn), ("gpl", gpl)]
        for r in range(args.reps):
            seq = order if r % 2 == 0 else order[::-1]
            for arm, pp in seq:
                fps, _cg = run_arm(pp, codec, "cpu")
                per[arm].append(fps)
        mn_fps, gpl_fps = min(per["min"]), min(per["gpl"])
        ratio = mn_fps / gpl_fps
        pairs = " ".join("(%.0f,%.0f)" % (a, b)
                         for a, b in zip(per["min"], per["gpl"]))
        print("%-5s %-5s %10.0f %10.0f %7.1f%%   %s"
              % (codec, "cpu", mn_fps, gpl_fps, ratio * 100, pairs), flush=True)

    print("== hybrid 混跑（调度按实测 rc/rg 供水；对照分臂交付）==")
    for codec in VIDS:
        per = {"min": [], "gpl": []}
        cg = {"min": None, "gpl": None}
        order = [("min", mn), ("gpl", gpl)]
        for r in range(args.reps):
            seq = order if r % 2 == 0 else order[::-1]
            for arm, pp in seq:
                fps, c = run_arm(pp, codec, "hybrid")
                per[arm].append(fps)
                cg[arm] = c
        mn_fps, gpl_fps = min(per["min"]), min(per["gpl"])
        pairs = " ".join("(%.0f,%.0f)" % (a, b)
                         for a, b in zip(per["min"], per["gpl"]))
        print("%-5s %-5s %10.0f %10.0f %7.1f%%   %s   交付c/g min=%s gpl=%s"
              % (codec, "hyb", mn_fps, gpl_fps,
                 (mn_fps / gpl_fps) * 100, pairs, cg["min"], cg["gpl"]),
              flush=True)

    print("== NVDEC（demux=avformat；解码=原生 cuvid，健全性对照）==")
    for codec in VIDS:
        if args.gpu_reps <= 0:
            continue
        per = {"min": [], "gpl": []}
        order = [("min", mn), ("gpl", gpl)]
        for r in range(args.gpu_reps):
            seq = order if r % 2 == 0 else order[::-1]
            for arm, pp in seq:
                fps, _cg = run_arm(pp, codec, "gpu")
                per[arm].append(fps)
        mn_fps, gpl_fps = min(per["min"]), min(per["gpl"])
        print("%-5s %-5s %10.0f %10.0f %7.1f%%"
              % (codec, "gpu", mn_fps, gpl_fps,
                 (mn_fps / gpl_fps) * 100), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
