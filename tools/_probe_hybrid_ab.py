"""临时探针 F：hybrid 混合解码 × CPU 线程数 A/B。

hybrid 的安全门：stride==1、NVDEC 可用、未开 GPU 全驻留管线。
CPU 侧线程数由 HYBRID_CPU_THREADS 控制（0 = 核数//2）；
DECORD_FFMPEG_THREAD_COUNT 决定其默认回退值。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path

# WORKER 以 `python -c` 执行，-c 下 __file__ 未定义，
# 故本进程算好根目录后经 PROBE_ROOT 传给子进程。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROBE_ROOT"] = ROOT  # 供 `python -c` 的 WORKER 子进程使用
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


PY = sys.executable

WORKER = r"""
import os, sys, time, json
sys.path.insert(0, os.environ["PROBE_ROOT"])
os.environ['ENGINE_PROFILE'] = '1'
path, roi_s, n, dbe, obe, st = sys.argv[1:7]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_end=int(n), sample_stride=int(st),
                    decode_backend=dbe, ocr_backend=obe, keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
texts = sorted({s.text for s in r.segments if s.text})
print(json.dumps({
    'wall': round(wall, 3), 'segs': len(r.segments), 'uniq': len(texts),
    'timing': {k: round(v, 3) for k, v in ex.timing.items()},
    'producer': {k: round(v, 3) for k, v in
                 sorted(ex.profile.get('producer', {}).items(),
                        key=lambda kv: -kv[1])[:5]},
    'ocr': {k: round(v, 3) for k, v in
            sorted(ex.profile.get('ocr', {}).items(),
                   key=lambda kv: -kv[1])[:4]},
    'backend': r.meta['backend'], 'ocr_backend': r.meta['ocr_backend'],
}))
"""


def run(video, roi, n, dbe, obe, env=None, reps=2, stride=1):
    e = dict(os.environ)
    if env:
        e.update(env)
    best = None
    for _ in range(reps):
        p = subprocess.run([PY, "-c", WORKER, video, roi, str(n), dbe, obe,
                            str(stride)],
                           capture_output=True, text=True, env=e)
        out = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not out:
            return {"err": (p.stderr or "").strip()[-300:]}
        d = json.loads(out[-1])
        if best is None or d["wall"] < best["wall"]:
            best = d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(_VIDEO_DIR / "test5.mp4"))
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    roi = ",".join(str(x) for x in (int(x) for x in a.roi.split(",")))
    print(f"=== hybrid A/B: {os.path.basename(a.video)} {a.frames}帧 "
          f"stride={a.stride} (取 {a.reps} 轮最快) ===")
    # 注：DECODE_THREADS 是引擎侧钩子（extractor._decode_num_threads 覆盖），
    # 只作用于 decode_backend="cpu"；HYBRID_CPU_THREADS 才是 hybrid 里
    # CPU 生产者的线程数。两者独立，不要混用（旧版这里错用了
    # DECORD_FFMPEG_THREAD_COUNT，对 hybrid 完全无效）。
    cases = [
        ("NVDEC+TRT (默认)", "auto", "auto", {}),
        ("CPU+TRT (引擎分档)", "cpu", "auto", {}),
        ("hybrid  cpuT=0（现役→fork 8线程）", "hybrid", "auto", {}),
        ("hybrid  cpuT=16", "hybrid", "auto",
         {"HYBRID_CPU_THREADS": "16"}),
        ("hybrid  cpuT=24", "hybrid", "auto",
         {"HYBRID_CPU_THREADS": "24"}),
        ("hybrid  cpuT=32", "hybrid", "auto",
         {"HYBRID_CPU_THREADS": "32"}),
        ("hybrid  cpuT=32 chunks=32", "hybrid", "auto",
         {"HYBRID_CPU_THREADS": "32", "HYBRID_MAX_CHUNKS": "32"}),
        ("hybrid  cpuT=32 chunks=64", "hybrid", "auto",
         {"HYBRID_CPU_THREADS": "32", "HYBRID_MAX_CHUNKS": "64"}),
    ]
    base = None
    for name, dbe, obe, env in cases:
        d = run(a.video, roi, a.frames, dbe, obe, env, reps=a.reps,
                stride=a.stride)
        if "err" in d:
            print(f"  {name:26s}: FAIL {d['err']}")
            continue
        if base is None:
            base = d["wall"]
        print(f"  {name:26s}: wall={d['wall']:6.3f}s "
              f"({d['wall']/base*100:5.1f}%)  segs={d['segs']:5d} "
              f"uniq={d['uniq']:4d}  decode={d['timing'].get('decode',0):.3f} "
              f"ocr_tail={d['timing'].get('ocr_tail',0):.3f} "
              f"[{d['backend']}]")
        print(f"      producer: {d['producer']}")


if __name__ == "__main__":
    main()
