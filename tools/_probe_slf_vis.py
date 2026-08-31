"""P0-6 test4 三方分歧帧定位 + 抽帧清单（视觉裁定用，临时探针）。

跑 关/开 两路提取，对每帧给出 (truth, 关, 开)，列出所有分歧帧及邻域。
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
video = 'test4'
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
print(json.dumps({'got': got}))
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
    return {int(k): v for k, v in json.loads(p.stdout.strip().splitlines()[-1])["got"].items()}

off = run_case({})
on = run_case({"DECORD_SKIP_LOOP_FILTER": "all"})

def num_eq(x, y):
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return False
    return abs(fx - fy) <= max(1e-3, 0.02 * max(abs(fx), abs(fy), 1.0))

both = sorted(f for f in truth if f in off and f in on)
# 分歧帧：关/开/真值 三方互不相同的帧
diff = [f for f in both if len({truth[f], off[f], on[f]}) > 1]
print("三方分歧帧数: %d / %d" % (len(diff), len(both)))
# 按连续段聚簇
clusters = []
for f in diff:
    if clusters and f - clusters[-1][-1] <= 3:
        clusters[-1].append(f)
    else:
        clusters.append([f])
print("分歧簇: %d 个" % len(clusters))
for c in clusters:
    print("\n== 簇 f%d..f%d (%d 帧) ==" % (c[0], c[-1], len(c)))
    for f in c:
        mark = []
        print("  f=%d truth=%r 关=%r 开=%r" % (f, truth[f], off[f], on[f]))
OUT = Path(__file__).resolve().parent / "_slf_vis"
OUT.mkdir(exist_ok=True)
out = {"frames": {str(f): {"truth": truth[f], "off": off[f], "on": on[f]} for f in diff},
       "clusters": [c for c in clusters]}
with open(OUT / "diff.json", 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n已写入 tools/_slf_vis/diff.json")
