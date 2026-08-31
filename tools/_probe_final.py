"""临时探针 H：收口测量 —— 全片对照 + 弱 CPU（绑核）敏感性。

用法：--affinity N 把子进程绑到前 N 个逻辑核，模拟弱 CPU。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# WORKER 以 `python -c` 执行，-c 下 __file__ 未定义，
# 故本进程算好根目录后经 PROBE_ROOT 传给子进程。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROBE_ROOT"] = ROOT  # 供 `python -c` 的 WORKER 子进程使用
_BATCH_DIR = Path(os.environ.get("RACELOG_BATCH_DIR", r"D:\Videos\batch_test"))
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


PY = sys.executable

WORKER = r"""
import os, sys, time, json
sys.path.insert(0, os.environ["PROBE_ROOT"])
aff = int(sys.argv[1])
if aff > 0:
    import psutil
    psutil.Process().cpu_affinity(list(range(aff)))
os.environ['ENGINE_PROFILE'] = '1'
path, roi_s, n, stride, dbe, obe = sys.argv[2:8]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_end=int(n), sample_stride=int(stride),
                    decode_backend=dbe, ocr_backend=obe, keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
print(json.dumps({
    'wall': round(wall, 3), 'segs': len(r.segments),
    'uniq': len({s.text for s in r.segments if s.text}),
    'timing': {k: round(v, 3) for k, v in ex.timing.items()},
    'decode_batch': round(ex.profile.get('producer', {}).get('decode_batch', 0), 3),
    'infer': round(ex.profile.get('ocr', {}).get('infer', 0), 3),
    'cores': len(psutil.Process().cpu_affinity()) if aff > 0 else os.cpu_count(),
    'backend': r.meta['backend'], 'ocr_backend': r.meta['ocr_backend'],
}))
"""


def run(video, roi, n, stride, dbe, obe, env=None, reps=2, aff=0):
    e = dict(os.environ)
    if env:
        e.update(env)
    best = None
    for _ in range(reps):
        p = subprocess.run([PY, "-c", WORKER, str(aff), video, roi, str(n),
                            str(stride), dbe, obe],
                           capture_output=True, text=True, env=e)
        out = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not out:
            return {"err": (p.stderr or "").strip()[-300:]}
        d = json.loads(out[-1])
        if best is None or d["wall"] < best["wall"]:
            best = d
    return best


def suite(name, video, roi, frames, stride, cases, reps=2, aff=0):
    print(f"\n### {name}  ({os.path.basename(video)}, "
          f"{frames} 源帧 stride={stride}"
          + (f", 绑核={aff}" if aff else "") + ")")
    base = None
    for cname, dbe, obe, env in cases:
        d = run(video, roi, frames, stride, dbe, obe, env, reps, aff)
        if "err" in d:
            print(f"  {cname:28s}: FAIL {d['err']}")
            continue
        if base is None:
            base = d["wall"]
        print(f"  {cname:28s}: wall={d['wall']:7.3f}s "
              f"({d['wall']/base*100:5.1f}%)  segs={d['segs']:5d} "
              f"uniq={d['uniq']:4d}  decode={d['decode_batch']:6.3f} "
              f"infer={d['infer']:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--affinity", type=int, default=0)
    a = ap.parse_args()

    T5 = str(_VIDEO_DIR / "test5.mp4")
    T5ROI = "843,993,948,1025"
    SUB = str(_BATCH_DIR / "新三国01.mkv")
    SUBROI = "144,398,551,423"

    h264_cases = [
        ("NVDEC+TRT (现役默认)", "auto", "auto", {}),
        ("CPU+TRT dcdT=8 (现役默认)", "cpu", "auto", {}),
        ("CPU+TRT dcdT=16", "cpu", "auto", {"DECORD_FFMPEG_THREAD_COUNT": "16"}),
        ("CPU+TRT dcdT=24", "cpu", "auto", {"DECORD_FFMPEG_THREAD_COUNT": "24"}),
        ("CPU+TRT dcdT=32", "cpu", "auto", {"DECORD_FFMPEG_THREAD_COUNT": "32"}),
    ]
    suite("h264 全片", T5, T5ROI, 7223, 1, h264_cases, a.reps, a.affinity)
    suite("h264 字幕整集 stride8", SUB, SUBROI, 73430, 8, h264_cases,
          a.reps, a.affinity)


if __name__ == "__main__":
    main()
