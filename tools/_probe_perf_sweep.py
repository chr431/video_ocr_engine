"""解码路径参数 sweep 探针（性能优化会话用）。

矩阵化测试 GPU_PIPELINE_DECODE_BATCH / GPU_PIPELINE_STREAM /
DECODE_THREADS / HYBRID_CPU_THREADS 对墙钟的影响；每格 2 跑取最快，
记录段数+唯一文本 sha 作正确性参照。

用法：
  python tools/_probe_perf_sweep.py --video test6 [--sweep batch] [--sweep stream]
  python tools/_probe_perf_sweep.py --video test5 --sweep threads
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEOS = {
    "test6": dict(path=r"D:\Videos\racelog_test\test6.mp4",
                  roi=(841, 994, 949, 1026), backend="nvdec"),
    "test5": dict(path=r"D:\Videos\racelog_test\test5.mp4",
                  roi=(843, 993, 948, 1025), backend="cpu"),
}


def run_once(path, roi, bk, envs, frames=3000):
    import engine_config as config
    from video_ocr_engine import FieldExtractor
    # GPU_PIPELINE_DECODE_BATCH 是模块常量（非 env）→ sweep 用 monkey-patch
    #（调用点读的是 config 属性，patch 即生效）。
    batch_val = envs.get("__GPU_PIPELINE_DECODE_BATCH__")
    saved_batch = config.GPU_PIPELINE_DECODE_BATCH
    if batch_val is not None:
        config.GPU_PIPELINE_DECODE_BATCH = int(batch_val)
    old = {}
    for k, v in envs.items():
        if k.startswith("__"):
            continue
        old[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        ex = FieldExtractor(path, roi, frame_end=frames,
                            decode_backend=bk, ocr_backend="auto",
                            keep_frames=True)
        t0 = time.perf_counter()
        res = ex.extract()
        wall = time.perf_counter() - t0
    finally:
        config.GPU_PIPELINE_DECODE_BATCH = saved_batch
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    texts = {s.text for s in res.segments if s.text}
    sha = hashlib.sha1("\n".join(sorted(texts)).encode()).hexdigest()[:12]
    return wall, len(res.segments), sha, dict(res.timing)


def sweep(video, what, values, runs=2):
    v = VIDEOS[video]
    ref_sha = None
    print(f"== {video} sweep {what} ==", flush=True)
    for val in values:
        if what == "batch":
            envs = {"__GPU_PIPELINE_DECODE_BATCH__": str(val)}
        elif what == "stream":
            envs = {"GPU_PIPELINE_STREAM": str(val)}
        elif what == "threads":
            envs = {"DECODE_THREADS": str(val)}
        elif what == "hybthreads":
            envs = {"HYBRID_CPU_THREADS": str(val)}
        else:
            raise SystemExit(f"unknown sweep {what}")
        walls, segs, shas = [], [], []
        for i in range(runs):
            w, n, sha, tm = run_once(v["path"], v["roi"], v["backend"],
                                     envs)
            walls.append(w)
            segs.append(n)
            shas.append(sha)
        if ref_sha is None:
            ref_sha = shas[0]
        ok = all(s == ref_sha for s in shas)
        print(f"  {what}={val}: {min(walls):.3f}s segs={segs} "
              f"{'✓' if ok else '⚠️漂移 ' + str(shas)}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, choices=list(VIDEOS))
    ap.add_argument("--sweep", required=True,
                    choices=["batch", "stream", "threads", "hybthreads"])
    ap.add_argument("--values", default="")
    ap.add_argument("--backend", default="")
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    if args.backend:
        VIDEOS[args.video]["backend"] = args.backend
    defaults = dict(
        batch=[32, 64, 128, 256],
        stream=[0, 1],
        threads=[8, 16, 24, 32, 48],
        hybthreads=[12, 16, 24, 32],
    )
    vals = ([int(v) for v in args.values.split(",")] if args.values
            else defaults[args.sweep])
    sweep(args.video, args.sweep, vals, args.runs)
