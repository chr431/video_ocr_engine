"""hybrid 解码 A/B 基准：纯 NVDEC / 纯 CPU / hybrid 同窗口墙钟对比。

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
    ap.add_argument("--hybrid-max-chunks", type=int, default=16)
    args = ap.parse_args()

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
                                     BACKENDS[bk], envs=envs)
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
