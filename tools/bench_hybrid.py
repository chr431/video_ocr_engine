"""hybrid 解码 A/B 基准：纯 NVDEC / 纯 CPU / hybrid 同窗口墙钟对比。

测量前会做**空闲检查**（--allow-busy 可跳过）：残留进程会污染结果且不会报错。

用法：
  python tools/bench_hybrid.py --video X --roi A,B,C,D --frames 3000
    [--backends nvdec,cpu,hybrid] [--runs 2] [--envs GPU_PIPELINE=0]

每个后端跑 runs 次取中位；打印 timing 分相（decode/ocr/ocr_tail）与
分段数、唯一文本集（一致性校验）。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_ocr_engine import FieldExtractor  # noqa: E402

BACKENDS = {
    "nvdec": dict(decode_backend="nvdec", ocr_backend="auto"),
    "cpu": dict(decode_backend="cpu", ocr_backend="auto"),
    "hybrid": dict(decode_backend="hybrid", ocr_backend="auto"),
}


def run_once(video, roi, frames, cfg, stride=1, envs=None):
    old = {}
    if envs:
        for k, v in envs.items():
            old[k] = os.environ.get(k)
            os.environ[k] = v
    try:
        ex = FieldExtractor(
            video, roi, frame_end=frames, sample_stride=stride,
            keep_frames=True, **cfg)
        t0 = time.perf_counter()
        res = ex.extract()
        wall = time.perf_counter() - t0
    finally:
        if envs:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return ex, res, wall


def check_idle(max_pct: float = 20.0, strict: bool = False) -> bool:
    """测量前确认机器空闲；不空闲就大声报警（或中止）。

    为什么需要（2026-09-01 真踩过）：
    一个探针被中断后留下**孤儿进程**，它的解码线程继续跑，占了 ~48% CPU
    并把 CPU 从 2501MHz 压到 1987MHz（降频）。之后十几轮 benchmark 全部
    偏慢 ~2.4s，而**输出看起来完全正常** —— 差点据此得出错误结论。
    数字会骗人，但"开跑前先看看机器忙不忙"不会。
    """
    try:
        import psutil
    except ImportError:
        print("  [空闲检查] psutil 未装，跳过")
        return True
    p = psutil.Process()
    p.cpu_percent(interval=None)          # 先给自身计数器打基线
    busy = psutil.cpu_percent(interval=1.5)
    try:
        freq = psutil.cpu_freq()
        fs = " 频率 %.0f/%.0f MHz" % (freq.current, freq.max) if freq else ""
    except Exception:
        fs = ""
    ok = busy <= max_pct
    print("  [空闲检查] CPU 占用 %.1f%%%s → %s"
          % (busy, fs, "OK" if ok else "**偏高**"))
    if not ok:
        print("     ⚠️ 机器不空闲：可能有残留的进程在跑（上次中断的探针？）。")
        print("     测量值会系统性偏慢，且**不会有任何报错**。")
        print("     Windows: Stop-Process -Name python -Force")
        print("     或加 --allow-busy 忽略（不推荐）。")
    if not ok and strict:
        raise SystemExit("机器不空闲，已中止（去掉 --strict 或先清理进程）")
    return ok


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--backends", default="nvdec,cpu,hybrid")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--envs", default="GPU_PIPELINE=0",
                    help="逗号分隔 K=V（如 GPU_PIPELINE=0,HYBRID_PROBE=1）")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--hybrid-max-chunks", type=int, default=16)
    ap.add_argument("--affinity", type=int, default=0,
                    help="把进程绑定到前 N 个逻辑 CPU（弱 CPU 模拟；0=不限制）")
    ap.add_argument("--allow-busy", action="store_true",
                    help="跳过测量前的空闲检查（不推荐）")
    ap.add_argument("--strict", action="store_true",
                    help="机器不空闲时直接中止")
    ap.add_argument("--idle-max", type=float, default=20.0,
                    help="空闲检查阈值（CPU 占用 %%，默认 20）")
    args = ap.parse_args()

    if not args.allow_busy:
        check_idle(args.idle_max, args.strict)

    if args.affinity > 0:
        try:
            import psutil
            ids = list(range(min(args.affinity, psutil.cpu_count() or 1)))
            psutil.Process().cpu_affinity(ids)
            print(f"affinity → {ids}")
        except Exception as e:
            print(f"affinity 设置失败: {e}")

    roi = tuple(int(v) for v in args.roi.split(","))
    envs = {}
    if args.envs:
        for kv in args.envs.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                envs[k.strip()] = v.strip()
    if args.hybrid_max_chunks != 16:
        envs["HYBRID_MAX_CHUNKS"] = str(args.hybrid_max_chunks)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    print(f"视频={args.video} ROI={roi} frames={args.frames} "
          f"backends={backends} runs={args.runs} envs={envs}")

    results = {}
    for bk in backends:
        walls, segs_n, texts, metas, timings = [], [], set(), [], []
        for i in range(args.runs):
            ex, res, wall = run_once(args.video, roi, args.frames,
                                     BACKENDS[bk], stride=args.stride,
                                     envs=envs)
            walls.append(wall)
            segs_n.append(len(res.segments))
            texts |= {s.text for s in res.segments if s.text}
            metas.append(res.meta)
            timings.append(dict(res.timing))
            print(f"  [{bk}] run{i+1}: {wall:.3f}s segs={len(res.segments)} "
                  f"meta={res.meta} timing={ {k: round(v,3) for k,v in res.timing.items()} }")
        med = statistics.median(walls)
        results[bk] = dict(walls=walls, segs=segs_n, texts=texts)
        print(f"  [{bk}] 中位墙钟={med:.3f}s min={min(walls):.3f}s "
              f"segs={segs_n} 唯一文本={len(texts)}")

    # cross-backend text consistency
    if len(results) > 1:
        base = results[backends[0]]["texts"]
        for bk in backends[1:]:
            s2 = results[bk]["texts"]
            inter = len(base & s2)
            print(f"  文本一致: {backends[0]} vs {bk} = {inter}/{max(1,len(base))} "
                  f"({inter/max(1,len(base)):.1%})")


if __name__ == "__main__":
    main()
