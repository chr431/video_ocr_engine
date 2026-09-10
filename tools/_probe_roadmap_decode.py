"""路线图轮 M1：三编码 × 解码后端 × CPU 线程档的顺序解码吞吐矩阵。

口径与 _probe_hybrid_sum_gap.py 对齐（同 ROI / yuv420 / 批64 / 每轮重开），
差异：本探针把 CPU 线程数做成 sweep 维度，且 hybrid 也扫同一 nt 网格，
用于回答「hybrid 增益随 CPU 线程档如何变化」。
DLL 口径由外部 DECORD_LIBRARY_PATH 控制：
  不设            = pip wheel 0.8.2（用户现状，不含 fork 未发布修复）
  build-081fix    = 本地 fork（含 73e5540/d94d92e/3b96c6f）
pip wheel 下 av1 hybrid close 可能崩（析构 UAF，fork 73e5540 才修），
故 rate 先打印再 close，close 异常单独记一行。

用法：
  python tools/_probe_roadmap_decode.py --video test6 --nts 12,24 --n 6000 --runs 3
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_VDIR = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
VIDS = {
    # roi_hw 与 _probe_hybrid_sum_gap.py 完全一致（引擎 roi 的 +1 形式）；
    # 第三位之后第 4 元 = 片长上限（test.mp4 只有 4718 帧，n 必须钳到片长内）
    "test":  (os.path.join(_VDIR, "test.mp4"),  (842, 995, 950, 1027), "hevc", 4718),
    "test5": (os.path.join(_VDIR, "test5.mp4"), (844, 994, 950, 1026), "h264", 7761),
    "test6": (os.path.join(_VDIR, "test6.mp4"), (842, 995, 950, 1027), "av1", 23970),
}


def run_once(path: str, roi_hw: tuple, kind: str, nt: int, block: int, n: int) -> float:
    from decord import VideoReader, cpu, gpu, hybrid_gpu
    if kind == "gpu":
        vr = VideoReader(path, ctx=gpu(0), output_format="yuv420", roi=roi_hw)
    elif kind == "cpu":
        vr = VideoReader(path, ctx=cpu(0), output_format="yuv420",
                         roi=roi_hw, num_threads=nt)
    else:
        vr = VideoReader(path, ctx=hybrid_gpu(0), output_format="yuv420",
                         roi=roi_hw, num_threads=nt)
    vr.seek(0)
    got, t0 = 0, time.perf_counter()
    while got < n:
        e = min(got + block, n)
        b = vr.get_batch(list(range(got, e)))
        got += b.shape[0]
        del b
    fps = got / (time.perf_counter() - t0)
    print(f"    RUN {kind} nt={nt}: {fps:.0f} fps", flush=True)
    try:
        vr.close()
    except BaseException as exc:  # pip wheel av1 hybrid 析构 UAF（fork 73e5540 修复）
        print(f"    [close 异常] {type(exc).__name__}: {exc}", flush=True)
    return fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, choices=sorted(VIDS))
    ap.add_argument("--kinds", default="cpu,gpu,hybrid")
    ap.add_argument("--nts", default="12")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    path, roi_hw, codec, vmax = VIDS[args.video]
    n = min(args.n, vmax)
    dll = os.environ.get("DECORD_LIBRARY_PATH", "<pip-wheel>")
    print(f"[{args.video} {codec}] dll={dll} n={n} block={args.block} "
          f"runs={args.runs}", flush=True)

    import decord
    print(f"  decord {decord.__version__} @ {decord.__file__}", flush=True)

    nts = [int(x) for x in args.nts.split(",")]
    for kind in args.kinds.split(","):
        if kind == "gpu":
            fps = [run_once(path, roi_hw, "gpu", 0, args.block, n)
                   for _ in range(args.runs)]
            print(f"  RESULT {args.video} nvdec nt=0 med={statistics.median(fps):.0f} "
                  f"all={[round(f) for f in fps]}", flush=True)
        else:
            for nt in nts:
                fps = [run_once(path, roi_hw, kind, nt, args.block, n)
                       for _ in range(args.runs)]
                print(f"  RESULT {args.video} {kind} nt={nt} "
                      f"med={statistics.median(fps):.0f} "
                      f"all={[round(f) for f in fps]}", flush=True)


if __name__ == "__main__":
    main()
