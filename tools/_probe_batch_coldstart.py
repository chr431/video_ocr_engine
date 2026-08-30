"""临时探针：批量场景的单视频墙钟分布与冷启动占比（prewarm 立项输入）。

§17.4 的问题：§15 量出每进程一次性成本 ~1.8s（CUDA 上下文 0.42s +
decord/NVDEC 首次 ~1.0s + TRT 反序列化 0.31s + NVRTC 0.068s），但要判断
`prewarm()` 值不值得做，还需要知道**批量任务里单个视频的墙钟量级**：
单视频墙钟越小，一次性成本占比越高，预热才越有意义。

方法：单进程顺序处理 batch_test 5 集（标清字幕，stride=8），记录每集
墙钟 / 段数 / 唯一文本。第 1 集承担全部一次性成本，第 2 集起走暖路径
（B5 的进程级 OCR 引擎池 + NVRTC 模块缓存命中）。

    Δ = 第1集 − 后续中位数  ≈  本进程可消除的一次性成本
    占比 = Δ / 单视频墙钟中位数  →  决定 prewarm 的价值

注意：prewarm 只把成本**提前**，不改变总量；它只在"冷启动可与别的工作
重叠"或"避免多实例并发重复构建"时才有净收益。本探针量化的是上界。
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_ocr_engine import FieldExtractor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\Videos\batch_test")
    ap.add_argument("--pattern", default="*.mkv")
    ap.add_argument("--roi", default="144,398,551,423")
    ap.add_argument("--frames", type=int, default=30000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--dbe", default="auto")
    ap.add_argument("--obe", default="auto")
    a = ap.parse_args()

    roi = tuple(int(x) for x in a.roi.split(","))
    videos = sorted(glob.glob(os.path.join(a.dir, a.pattern)))
    if not videos:
        print(f"未找到视频: {a.dir}/{a.pattern}")
        return

    rows = []
    for i, v in enumerate(videos):
        name = os.path.basename(v)
        try:
            ex = FieldExtractor(v, roi, frame_end=a.frames,
                                sample_stride=a.stride,
                                decode_backend=a.dbe, ocr_backend=a.obe,
                                keep_crops=False)
            t0 = time.perf_counter()
            r = ex.extract()
            wall = time.perf_counter() - t0
            uniq = len({s.text for s in r.segments if s.text})
            rows.append((name, wall, len(r.segments), uniq,
                         r.meta.get("backend"), r.meta.get("ocr_backend")))
            print(f"[{i + 1}/{len(videos)}] {name}: wall={wall:.3f}s "
                  f"段数={len(r.segments)} uniq={uniq}")
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(videos)}] {name}: 失败 {type(e).__name__}: "
                  f"{str(e)[:200]}")
            return

    if len(rows) < 2:
        print("视频不足 2 个，无法分离冷启动")
        return

    print(f"\n=== 批量 {len(rows)} 集（{os.path.basename(a.dir)}，"
          f"frame_end={a.frames} stride={a.stride} dbe={a.dbe} obe={a.obe}）===")
    print(f"后端: {rows[0][4]} / {rows[0][5]}")
    first = rows[0][1]
    rest = [r[1] for r in rows[1:]]
    med_rest = statistics.median(rest)
    med_all = statistics.median([r[1] for r in rows])
    delta = first - med_rest
    print(f"  第 1 集（含冷启动）: {first:.3f}s")
    print(f"  第 2~N 集中位数    : {med_rest:.3f}s   逐集: "
          f"{[round(x, 3) for x in rest]}")
    print(f"  单视频墙钟中位数    : {med_all:.3f}s")
    print(f"\n  一次性成本 Δ = {delta:+.3f}s  = 单视频墙钟的 "
          f"{delta / med_all * 100:.1f}%")
    print(f"  §15 独立实测的一次性账单 ≈ 1.8s（其中引擎可管部分 ≈ 0.38s："
          f"TRT 反序列化 0.31s + NVRTC 0.068s）")
    if delta > 0.05:
        managed = min(delta, 0.38)
        print(f"  → prewarm 可回收上限 ≈ {managed:.2f}s/进程 "
              f"（= 批量总墙钟的 {managed / (med_all * len(rows)) * 100:.1f}%）")
    print("\n  注：prewarm 只把成本提前，不改变总量。仅当冷启动可与别的工作")
    print("  重叠（GUI 启动、多实例并发避免重复构建）时才有净收益；")
    print("  纯顺序批量的吞吐不受 prewarm 影响。")


if __name__ == "__main__":
    main()
