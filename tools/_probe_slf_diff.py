"""P0-6 skip_loop_filter 的逐帧真值差异分析（临时探针）。

用法: python tools/_probe_slf_diff.py <video>
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# WORKER 以 `python -c` 执行，-c 下 __file__ 未定义，
# 故本进程算好根目录后经 PROBE_ROOT 传给子进程。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROBE_ROOT"] = ROOT  # 供 `python -c` 的 WORKER 子进程使用
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


GT = _VIDEO_DIR / "ground_truth_csv"
VID = _VIDEO_DIR
video = sys.argv[1] if len(sys.argv) > 1 else 'test4'

tp = GT / f"{video}_ref.csv"
if not tp.exists():
    tp = GT / f"{video}_truth.csv"
meta = {}
with open(tp, encoding="utf-8-sig") as f:
    for line in f:
        if not line.startswith("#"):
            break
        m = re.search(r"roi\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", line)
        if m and "roi" not in meta:
            meta["roi"] = tuple(int(x) for x in m.groups())
        m = re.search(r"frame_start\s*=\s*(-?\d+)", line)
        if m:
            meta["start"] = int(m.group(1))
        m = re.search(r"frame_end\s*=\s*(-?\d+)", line)
        if m:
            meta["end"] = int(m.group(1))
truth = {}
with open(tp, encoding="utf-8-sig") as f:
    for line in f:
        line = line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
            continue
        truth[int(parts[0])] = parts[2].strip()

def num_eq(x, y):
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return False
    return abs(fx - fy) <= max(1e-3, 0.02 * max(abs(fx), abs(fy), 1.0))

WORKER = r'''
import os, sys, json
sys.path.insert(0, os.environ["PROBE_ROOT"])
path, roi_s, start, end = sys.argv[1:5]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_start=int(start), frame_end=int(end),
                    sample_stride=1, decode_backend='cpu', ocr_backend='auto',
                    keep_crops=False)
r = ex.extract()
got = {}
for s in r.segments:
    for f in (s.frames or (s.start,)):
        got[int(f)] = s.text or ""
print(json.dumps({'got': got, 'n_segs': len(r.segments)}))
'''

def run_case(env):
    e = dict(os.environ)
    e.pop("DECORD_SKIP_LOOP_FILTER", None)
    e.update(env)
    p = subprocess.run([sys.executable, "-c", WORKER, str(VID / f"{video}.mp4"),
                        ",".join(map(str, meta["roi"])),
                        str(meta.get("start", 0)), str(meta.get("end", 0))],
                       capture_output=True, text=True, env=e)
    assert p.returncode == 0, p.stderr[-500:]
    d = json.loads(p.stdout.strip().splitlines()[-1])
    return {int(k): v for k, v in d["got"].items()}, d["n_segs"]

off, segs_off = run_case({})
on, segs_on = run_case({"DECORD_SKIP_LOOP_FILTER": "all"})
print("diag: len(truth)=%d  len(off)=%d(segs=%d)  len(on)=%d(segs=%d)" % (
    len(truth), len(off), segs_off, len(on), segs_on))
print("diag: off 缺失=%d 多余=%d | on 缺失=%d 多余=%d" % (
    len(set(truth) - set(off)), len(set(off) - set(truth)),
    len(set(truth) - set(on)), len(set(on) - set(truth))))
both = [f for f in off if f in truth]
def acc(got):
    ok = sum(1 for f in both if got[f] == truth[f])
    okn = sum(1 for f in both if got[f] == truth[f] or num_eq(got[f], truth[f]))
    return ok, okn
o, on_ = acc(off), acc(on)
print("关 : 全等 %d/%d (%.3f%%)  容错 %d (%.3f%%)" % (
    o[0], len(both), 100 * o[0] / len(both), o[1], 100 * o[1] / len(both)))
print("开 : 全等 %d/%d (%.3f%%)  容错 %d (%.3f%%)" % (
    on_[0], len(both), 100 * on_[0] / len(both), on_[1], 100 * on_[1] / len(both)))
fix = [f for f in both if off[f] != truth[f] and on[f] == truth[f]]
brk = [f for f in both if off[f] == truth[f] and on[f] != truth[f]]
fixn = [f for f in both if not num_eq(off[f], truth[f]) and num_eq(on[f], truth[f])]
brkn = [f for f in both if num_eq(off[f], truth[f]) and not num_eq(on[f], truth[f])]
print("纠错(关错→开对): %d 帧   退化(关对→开错): %d 帧" % (len(fix), len(brk)))
print("数值域纠错: %d   数值域退化: %d" % (len(fixn), len(brkn)))
print("退化样例(最多 12):")
for f in brk[:12]:
    print("  f=%d 真值=%r 关=%r 开=%r" % (f, truth[f], off[f], on[f]))
print("纠错样例(最多 6):")
for f in fix[:6]:
    print("  f=%d 真值=%r 关=%r 开=%r" % (f, truth[f], off[f], on[f]))
