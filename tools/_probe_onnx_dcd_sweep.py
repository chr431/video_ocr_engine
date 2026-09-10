"""ONNX OCR 场景的解码线程数 sweep（DECODE_THREADS）。

场景：decode_backend=cpu + ocr_backend=cpu（宿主管线，ORT 在 CPU 上与
FFmpeg 抢核）。矩阵：视频 × stride(1/8) × 线程数，每格 3 跑取
min/median；sha 按 (video,stride) 内部一致作门禁（线程数不应改变输出）。

用法：
  python tools/_probe_onnx_dcd_sweep.py [--videos test5,test6] [--runs 3]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

VIDEOS = {
    "test5": (r"D:\Videos\racelog_test\test5.mp4", (843, 993, 948, 1025)),
    "test6": (r"D:\Videos\racelog_test\test6.mp4", (841, 994, 949, 1026)),
    "test":  (r"D:\Videos\racelog_test\test.mp4",  (842, 995, 950, 1027)),
}


def run(path, roi, stride, threads, frames=3000):
    os.environ["DECODE_THREADS"] = str(threads)
    from video_ocr_engine import FieldExtractor
    ex = FieldExtractor(path, roi, frame_end=frames, sample_stride=stride,
                        decode_backend="cpu", ocr_backend="cpu",
                        keep_frames=True)
    t0 = time.perf_counter()
    res = ex.extract()
    wall = time.perf_counter() - t0
    texts = {s.text for s in res.segments if s.text}
    sha = hashlib.sha1("\n".join(sorted(texts)).encode()).hexdigest()[:12]
    return wall, len(res.segments), sha


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="test5,test6")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    for vname in args.videos.split(","):
        path, roi = VIDEOS[vname]
        for stride in (1, 8):
            threads_list = ([8, 10, 12, 16, 20, 24, 32] if stride == 1
                            else [8, 16, 24, 32, 48])
            ref = None
            rows = []
            print(f"== {vname} stride={stride} (ONNX OCR) ==", flush=True)
            for th in threads_list:
                walls, shas, segs = [], [], []
                for i in range(args.runs):
                    w, n, sha = run(path, roi, stride, th)
                    walls.append(w)
                    shas.append(sha)
                    segs.append(n)
                if ref is None:
                    ref = shas[0]
                ok = all(s == ref for s in shas)
                mn = min(walls)
                md = statistics.median(walls)
                rows.append((th, mn, md))
                print(f"  threads={th:2}: min={mn:.3f} med={md:.3f} "
                      f"segs={segs} sha={'✓' if ok else '⚠️' + str(shas)}",
                      flush=True)
            best = min(rows, key=lambda r: r[1])
            print(f"  → 最优 threads={best[0]} (min {best[1]:.3f}s)", flush=True)


if __name__ == "__main__":
    main()
