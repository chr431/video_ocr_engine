"""hybrid 与理论性能的差距分解（用户点名方向的前置诊断）。

三层对照（同一视频、同一 ROI、同 output_format=gray、同批 64）：

  A. **解码器自身**：纯 `get_batch` 扫掠（无分段/无 OCR）—— cpu / nvdec / hybrid
     三个 ctx 各测一条速率曲线；
  B. **理论理想和**：1/(1/cpu + 1/nvdec)（两侧完全并行、零协调开销的上界）；
  C. **引擎口径**：真实 extract 的 `pipeline.decode` 相位（含 analyze/emit/OCR 重叠）。

  B − A(hybrid) = **解码器/调度侧差距**（fork 的 chunk 预路由、池深、分片比例）；
  A(hybrid) − C = **引擎消费侧差距**（生产者/消费者/OCR 抽取）。

用法：
  python tools/_probe_hybrid_gap.py --videos h264,hevc,av1 [--frames 3000]
  # 换 fork dll：DECORD_LIBRARY_PATH=D:\\Repo\\decord\\build-081fix
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEOS = {
    "h264": (r"D:\Videos\racelog_test\test5.mp4", (843, 993, 948, 1025)),
    "hevc": (r"D:\Videos\racelog_test\test6_hevc.mp4", (841, 994, 949, 1026)),
    "av1": (r"D:\Videos\racelog_test\test6.mp4", (841, 994, 949, 1026)),
}
BATCH = 64


def decode_rate(video: str, roi: tuple, ctx_name: str, frames: int,
                reps: int = 2) -> float:
    """纯解码速率（帧/s）。ctx_name ∈ {cpu, nvdec, hybrid, hybrid_gpu}。"""
    import decord
    from decord import cpu, gpu, hybrid, hybrid_gpu
    ctx = {"cpu": lambda: cpu(0), "nvdec": lambda: gpu(0),
           "hybrid": lambda: hybrid(0),
           "hybrid_gpu": lambda: hybrid_gpu(0)}[ctx_name]()
    x1, y1, x2, y2 = roi
    nt = 32 if ctx_name in ("cpu", "hybrid") else 0
    vr = decord.VideoReader(video, ctx=ctx, output_format="gray",
                            num_threads=nt, roi=(x1, y1, x2 + 1, y2 + 1))
    n = min(frames, len(vr))
    idx = list(range(n))
    best = 0.0
    for _ in range(reps):
        t = time.perf_counter()
        got = 0
        for s in range(0, n, BATCH):
            got += vr.get_batch(idx[s:s + BATCH]).shape[0]
        r = got / (time.perf_counter() - t)
        best = max(best, r)
    del vr
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="h264,hevc,av1")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    import decord
    from decord._ffi import base as _B
    try:
        lib = Path(_B._LIB._name).parent.name
    except Exception:  # noqa: BLE001
        lib = "?"
    print("decord %s | dll 目录 %s | 批 %d | %d 帧 | output=gray\n"
          % (decord.__version__, lib, BATCH, args.frames))
    # 引擎口径热轮墙钟（S6 后实测；hybrid 走 GPU 管线 → hybrid_gpu ctx）
    ENGINE_WALL = {"h264": {"nvdec": 3.107, "cpu": 0.994, "hybrid": 1.281},
                   "av1": {"nvdec": 1.818, "cpu": 2.6, "hybrid": 1.372},
                   "hevc": {"nvdec": 1.530, "cpu": 3.1, "hybrid": None}}
    print("%-6s %8s %8s %8s %10s %11s %12s" % (
        "视频", "cpu", "nvdec", "hybrid", "hybrid_gpu", "理想(并行和)",
        "hybrid_gpu/理想"))
    for name in args.videos.split(","):
        video, roi = VIDEOS[name]
        r = {c: decode_rate(video, roi, c, args.frames, args.reps)
             for c in ("cpu", "nvdec", "hybrid", "hybrid_gpu")}
        # 并行分摊：两路吞吐相加（不是调和平均——那是串行分摊的口径）
        ideal = r["cpu"] + r["nvdec"]
        print("%-6s %8.0f %8.0f %8.0f %10.0f %11.0f %11.0f%%" % (
            name, r["cpu"], r["nvdec"], r["hybrid"], r["hybrid_gpu"],
            ideal, r["hybrid_gpu"] / ideal * 100))
        w = ENGINE_WALL.get(name, {})
        if w.get("hybrid"):
            eng = args.frames / w["hybrid"]
            print("       引擎口径：hybrid %.3fs（%.0f fps）／解码器 %.3fs"
                  "（%.0f fps）→ 引擎吃掉 %.0f%%；单侧 nvdec 引擎 %.0f fps"
                  " vs 解码器 %.0f fps" % (
                      w["hybrid"], eng, args.frames / r["hybrid_gpu"],
                      r["hybrid_gpu"], (1 - eng / r["hybrid_gpu"]) * 100,
                      args.frames / w["nvdec"], r["nvdec"]))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
