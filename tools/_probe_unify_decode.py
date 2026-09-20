"""decode 统一设想（hybrid+FORCE_SIDE 替代纯臂）可行性测量（2026-09-20）。

问题（用户）：只保留 hybrid 管线、用 DECORD_HYBRID_FORCE_SIDE=cpu/gpu
实现 decode_backend=cpu/gpu——forced-hybrid 相对纯臂的解码速率差、
构造（冷启）差、输出等价性是多少？

口径：fork 级纯解码（gray+ROI+引擎档线程数 32），子进程自调用
（FORCE_SIDE 为进程级 static 缓存，同进程不可切换）。
已知前置（fork 注释）：force=cpu 在非 IDR 边界编码（AV1）下退化为
GPU——本探针在 av1 上实测验证该退化。

用法：python tools/_probe_unify_decode.py            # 父进程：交错矩阵
      python tools/_probe_unify_decode.py --child <ctx> <force> <video> <mode>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VIDS = {
    "test5": ("test5.mp4", (843, 993, 949, 1026), 7761),
    "test6_hevc": ("test6_hevc.mp4", (841, 994, 949, 1026), 23970),
    "test6_av1": ("test6.mp4", (841, 994, 949, 1026), 23970),
}


def child(ctx_name: str, force: str, video: str, mode: str) -> str:
    import warnings
    warnings.filterwarnings("ignore")
    import decord
    import numpy as np
    name, roi, total = VIDS[video]
    vdir = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
    path = str(Path(vdir) / name)
    ctx = {"cpu": decord.cpu(0), "gpu": decord.gpu(0),
           "hybrid": decord.hybrid(0), "hybrid_gpu": decord.hybrid_gpu(0)}[ctx_name]
    t0 = time.perf_counter()
    vr = decord.VideoReader(path, ctx=ctx, output_format="gray",
                            roi=roi, num_threads=32)
    ctor = time.perf_counter() - t0
    n = total if mode == "full" else 1000
    lo = 0 if mode == "full" else 5000
    parts = []
    t1 = time.perf_counter()
    got = 0
    while got < n:
        e = min(got + 64, n)
        b = vr.get_batch(list(range(lo + got, lo + e))).asnumpy()
        if mode == "hash":
            parts.append(b)
        got += b.shape[0]
    decode = time.perf_counter() - t1
    h = hashlib.md5(np.concatenate(parts, 0).tobytes()).hexdigest()[:12] if parts else "-"
    vr.close()
    return ("RESULT ctx=%s force=%s video=%s mode=%s ctor=%.3f decode=%.3f "
            "fps=%.0f got=%d md5=%s"
            % (ctx_name, force or "-", video, mode, ctor, decode, got / decode,
               got, h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=4, default=None,
                    help="[ctx force video mode]（内部自调用）")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    if args.child:
        print(child(*args.child))
        return 0

    py = sys.executable
    me = str(Path(__file__).resolve())

    def run(ctx, force, video, mode):
        env = dict(os.environ)
        env.pop("DECORD_HYBRID_FORCE_SIDE", None)
        if force:
            env["DECORD_HYBRID_FORCE_SIDE"] = force
        r = subprocess.run([py, me, "--child", ctx, force or "", video, mode],
                           capture_output=True, text=True, encoding="utf-8",
                           env=env, timeout=300)
        for line in r.stdout.splitlines():
            if line.startswith("RESULT"):
                return line
        raise RuntimeError("child failed: %s" % r.stderr[-300:])

    # 等价性（一次）：forced vs plain 的窗口哈希
    print("== 输出等价性（[5000,6000) 窗口 md5）==")
    for video in ("test5", "test6_hevc"):
        for ctx_plain, ctx_hyb, force in (("cpu", "hybrid", "cpu"),
                                          ("gpu", "hybrid_gpu", "gpu")):
            a = run(ctx_plain, "", video, "hash")
            b = run(ctx_hyb, force, video, "hash")
            ha = a.split("md5=")[1]
            hb = b.split("md5=")[1]
            print("  %-9s %-4s: plain=%s forced=%s %s"
                  % (video, force, ha, hb, "一致" if ha == hb else "不一致"))

    # 速率矩阵（交错：每 rep 轮流跑全部配置，配对比较）
    print("\n== 纯解码矩阵（交错 ×%d，fps / ctor ms）==" % args.reps)
    cfgs = []
    for video in ("test5", "test6_hevc", "test6_av1"):
        cfgs.append((video, "cpu", ""))
        cfgs.append((video, "hybrid", "cpu"))     # FORCE_CPU
        cfgs.append((video, "gpu", ""))
        cfgs.append((video, "hybrid_gpu", "gpu")) # FORCE_GPU
    acc = {c: [] for c in cfgs}
    ctor_acc = {c: [] for c in cfgs}
    for rep in range(args.reps):
        for c in cfgs:               # 顺序即轮转（每 rep 同序，配置间隔相等）
            line = run(c[1], c[2], c[0], "full")
            fps = float(line.split("fps=")[1].split()[0])
            ctor = float(line.split("ctor=")[1].split()[0])
            acc[c].append(fps)
            ctor_acc[c].append(ctor)
    for c in cfgs:
        import statistics
        print("  %-9s %-10s force=%-4s fps中位=%6.0f ctor=%4.0fms"
              % (c[0], c[1], c[2] or "-",
                 statistics.median(acc[c]), statistics.median(ctor_acc[c]) * 1000))
    print("\n配对（forced 相对纯臂，同 rep）：")
    import statistics as st
    for video in ("test5", "test6_hevc", "test6_av1"):
        for plain, hyb, force, tag in (
                (("cpu", ""), ("hybrid", "cpu"), "cpu", "FORCE_CPU vs cpu"),
                (("gpu", ""), ("hybrid_gpu", "gpu"), "gpu", "FORCE_GPU vs gpu")):
            key_p = (video, *plain)
            key_h = (video, *hyb)
            diffs = [(acc[key_p][i] - acc[key_h][i]) / acc[key_p][i] * 100
                     for i in range(args.reps)]
            print("  %-9s %-20s forced 慢 %+.1f%%（各 rep %s）"
                  % (video, tag, st.fmean(diffs),
                     " ".join("%+.1f" % d for d in diffs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
