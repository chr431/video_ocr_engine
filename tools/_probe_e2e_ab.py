"""临时探针 E：端到端 A/B（解码线程数 / 后端矩阵）。

对照现役默认（NVDEC+TRT）评估：
  - CPU 软解 + 高线程数（DECORD_FFMPEG_THREAD_COUNT 覆盖）
  - CPU + ONNX（争核场景是否仍成立）
输出墙钟 + 分相 + 段数/唯一文本（校验一致性）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PY = sys.executable

WORKER = r"""
import os, sys, time
sys.path.insert(0, r"D:\Repo\video_ocr_engine")
os.environ['ENGINE_PROFILE'] = '1'
path, roi_s, nframes, stride, dbe, obe, keep = sys.argv[1:8]
roi = tuple(int(x) for x in roi_s.split(','))
n = int(nframes); st = int(stride)
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_end=n, sample_stride=st,
                    decode_backend=dbe, ocr_backend=obe,
                    keep_crops=(keep == '1'))
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
texts = sorted({s.text for s in r.segments if s.text})
prof = ex.profile.get('producer', {})
ocr = ex.profile.get('ocr', {})
import json
print(json.dumps({
    'wall': round(wall, 3),
    'segs': len(r.segments),
    'uniq': len(texts),
    'timing': {k: round(v, 3) for k, v in ex.timing.items()},
    'producer': {k: round(v, 3) for k, v in
                 sorted(prof.items(), key=lambda kv: -kv[1])[:6]},
    'ocr': {k: round(v, 3) for k, v in
            sorted(ocr.items(), key=lambda kv: -kv[1])[:6]},
    'ocr_backend': r.meta['ocr_backend'],
    'backend': r.meta['backend'],
}))
"""


def run(video, roi, nframes, stride, dbe, obe, env=None, keep='0', reps=1):
    e = dict(os.environ)
    if env:
        e.update(env)
    best = None
    for _ in range(reps):
        p = subprocess.run(
            [PY, "-c", WORKER, video, roi, str(nframes), str(stride),
             dbe, obe, keep],
            capture_output=True, text=True, env=e)
        out = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not out:
            return {"err": (p.stderr or "").strip()[-300:]}
        import json
        d = json.loads(out[-1])
        if best is None or d["wall"] < best["wall"]:
            best = d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test5.mp4")
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    roi = ",".join(str(x) for x in (int(x) for x in a.roi.split(",")))
    print(f"=== 端到端 A/B: {os.path.basename(a.video)} "
          f"{a.frames}帧 stride={a.stride} ===")
    cases = [
        ("NVDEC+TRT   (现役默认)", "auto", "auto", {}),
        ("NVDEC+ONNX", "auto", "cpu", {}),
        ("CPU+TRT     (默认8线程)", "cpu", "auto", {}),
        ("CPU+TRT     dcdT=16", "cpu", "auto",
         {"DECORD_FFMPEG_THREAD_COUNT": "16"}),
        ("CPU+TRT     dcdT=24", "cpu", "auto",
         {"DECORD_FFMPEG_THREAD_COUNT": "24"}),
        ("CPU+TRT     dcdT=32", "cpu", "auto",
         {"DECORD_FFMPEG_THREAD_COUNT": "32"}),
        ("CPU+ONNX    (默认8线程)", "cpu", "cpu", {}),
        ("CPU+ONNX    dcdT=16", "cpu", "cpu",
         {"DECORD_FFMPEG_THREAD_COUNT": "16"}),
        ("CPU+ONNX    dcdT=24", "cpu", "cpu",
         {"DECORD_FFMPEG_THREAD_COUNT": "24"}),
    ]
    base = None
    for name, dbe, obe, env in cases:
        d = run(a.video, roi, a.frames, a.stride, dbe, obe, env,
                reps=a.reps)
        if "err" in d:
            print(f"  {name:26s}: FAIL {d['err']}")
            continue
        if base is None:
            base = d["wall"]
        print(f"  {name:26s}: wall={d['wall']:6.3f}s "
              f"({d['wall']/base*100:5.1f}%)  segs={d['segs']:5d} "
              f"uniq={d['uniq']:4d}  decode={d['timing'].get('decode',0):.3f} "
              f"ocr_tail={d['timing'].get('ocr_tail',0):.3f}  [{d['backend']}/{d['ocr_backend']}]")
        print(f"      producer: {d['producer']}")
        print(f"      ocr:      {d['ocr']}")


if __name__ == "__main__":
    main()
