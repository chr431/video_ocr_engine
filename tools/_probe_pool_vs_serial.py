"""批量三码族：serial-hybrid vs pool pair vs pool-hybrid 并发（交错配对）。

问题（2026-09-20）：hybrid 单流已达并联和 96/92/96%，decode-bound 批量
下 pool 相比 hybrid 串行是否仍有显著收益？C-52 的 −20.6% 基线是「全
nvdec」，同期 hybrid 串行三码之和（单臂墙钟相加）≈27.9s vs pair 27.0s，
账面差 ~3%——本探针实测裁决。

三臂（同一进程，引擎池共享，臂序每轮轮转对冲漂移）：
  serial-hybrid  三文件顺序 extract（每文件 decode_backend=hybrid）
  pool-pair      pool.run(backends='pair')（h264→cpu，hevc/av1→nvdec，2 工人）
  pool-hybrid    pool.run(backends=['hybrid']×3)（2 个并发 hybrid reader =
                 方向一的核心未知量：单 NVDEC 下双会话退化率）

判据：每轮配对差分 + 符号一致性（首轮冷启弃置）。段数恒定 8340/文件
为正确性副断言（并发 hybrid 出错会漂）。

用法：python tools/_probe_pool_vs_serial.py [--rounds 5] [--warmup 1]
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

ROI = (841, 994, 949, 1026)
FILES = ["test6_h264.mp4", "test6_hevc.mp4", "test6.mp4"]   # h264/hevc/av1


def _paths():
    vdir = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
    return [str(Path(vdir) / f) for f in FILES]


def run_serial(paths) -> tuple[float, list]:
    from video_ocr_engine import FieldExtractor
    segs = []
    t0 = time.perf_counter()
    for p in paths:
        ex = FieldExtractor(p, ROI, decode_backend="hybrid",
                            ocr_backend="tensorrt")
        r = ex.extract()
        segs.append(len(r.segments))
        del ex, r
    return time.perf_counter() - t0, segs


def run_pool(paths, backends) -> tuple[float, list]:
    from video_ocr_engine.pipeline import pool as _pool
    items = [dict(video_path=p, roi=ROI, ocr_backend="tensorrt")
             for p in paths]
    t0 = time.perf_counter()
    results = _pool.run(items, backends=backends, max_workers=2)
    wall = time.perf_counter() - t0
    return wall, [len(r.segments) for r in results]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    paths = _paths()
    # 引擎池/GPU 时钟预热（warmup 轮数据弃置）
    from video_ocr_engine import FieldExtractor
    ex = FieldExtractor(paths[0], ROI, frame_start=0, frame_end=2000,
                        decode_backend="hybrid", ocr_backend="tensorrt")
    ex.extract()
    del ex

    arms = [
        ("serial-hybrid", lambda: run_serial(paths)),
        ("pool-pair", lambda: run_pool(paths, "pair")),
        ("pool-hybrid", lambda: run_pool(paths, ["hybrid"] * 3)),
    ]
    data = {name: [] for name, _ in arms}
    seg_ok = True
    total_rot = 0
    for rnd in range(args.warmup + args.rounds):
        # 臂序轮转：round r 从臂 (r mod 3) 起
        order = [arms[(i + rnd) % 3] for i in range(3)]
        for name, fn in order:
            wall, segs = fn()
            if rnd >= args.warmup:
                data[name].append(wall)
                if segs != [8340, 8340, 8340]:
                    seg_ok = False
                    print("⚠️ 段数漂移 %s: %s" % (name, segs))
            print("轮%-2d %-14s wall=%.2fs segs=%s%s"
                  % (rnd + 1, name, wall, segs,
                     "" if rnd >= args.warmup else "（冷启弃置）"),
                  flush=True)
        total_rot += 1

    print("\n== 汇总（热轮，中位/散布）==")
    base = statistics.median(data["serial-hybrid"])
    for name, walls in data.items():
        med = statistics.median(walls)
        spread = (max(walls) - min(walls)) / med * 100
        print("  %-14s 中位=%.2fs 散布=%.1f%% 各轮=%s vs 串行=%+.1f%%"
              % (name, med, spread,
                 " ".join("%.2f" % w for w in walls),
                 (med - base) / base * 100))
    # 配对差分（同轮内 pair/serial 与 hybpool/serial——臂序轮转下同轮
    # 三臂共享热状态，配对成立）
    n = min(len(data["serial-hybrid"]),
            len(data["pool-pair"]), len(data["pool-hybrid"]))
    for other in ("pool-pair", "pool-hybrid"):
        diffs = [(data[other][i] - data["serial-hybrid"][i])
                 / data["serial-hybrid"][i] * 100 for i in range(n)]
        signs = sum(1 for d in diffs if d < 0)
        print("  %s − serial：逐轮 %s 符号快 %d/%d 均值 %+.2f%%"
              % (other, " ".join("%+.1f" % d for d in diffs),
                 signs, n, statistics.fmean(diffs)))
    print("段数副断言：%s" % ("全部 8340 ✓" if seg_ok else "有漂移 ✗"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
