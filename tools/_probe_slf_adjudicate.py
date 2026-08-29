"""P0-6 test4 关≠开 帧的拼图生成（视觉裁定，临时探针）。

两路提取 → 枚举 关≠开 帧 → 从原视频抽 ROI → 4x 最近邻放大 →
5列网格拼图（cell 上方黑条标注帧号+关/开读数）→ 供逐格视觉裁定。
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

GT = Path(r"D:\Videos\racelog_test\ground_truth_csv")
VID = Path(r"D:\Videos\racelog_test")
OUT = Path(r"D:\Repo\video_ocr_engine\tools\_slf_vis")
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
sys.path.insert(0, os.getcwd())
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

cache = OUT / "cases.json"
if cache.exists():
    d = json.loads(cache.read_text(encoding="utf-8"))
    off = {int(k): v for k, v in d["off"].items()}
    on = {int(k): v for k, v in d["on"].items()}
else:
    off = run_case({})
    on = run_case({"DECORD_SKIP_LOOP_FILTER": "all"})
    cache.write_text(json.dumps({"off": off, "on": on}, ensure_ascii=False), encoding="utf-8")

both = sorted(f for f in truth if f in off and f in on)
ne = [f for f in both if off[f] != on[f]]
print("关≠开 帧数: %d / %d" % (len(ne), len(both)))
# 按 ≤3 帧间距聚簇，簇代表帧抽图
clusters = []
for f in ne:
    if clusters and f - clusters[-1][-1] <= 3:
        clusters[-1].append(f)
    else:
        clusters.append([f])
print("簇数: %d" % len(clusters))
for c in clusters:
    ex_f = c[len(c) // 2]
    print("  f%s 代表=%d truth=%r 关=%r 开=%r" % (
        ("%d..%d" % (c[0], c[-1])) if len(c) > 1 else str(c[0]),
        ex_f, truth.get(c[0]), off[c[0]], on[c[0]]))

# ── 抽帧（ffmpeg 一次进程抽一帧）→ 4x 放大 → 5 列网格拼图 ──
x1, y1, x2, y2 = meta["roi"]
W, H = x2 - x1 + 1, y2 - y1 + 1
S = 4
CW, CH = W * S, H * S + 22   # cell = 放大图 + 22px 标注条
COLS = 5
tmp = OUT / "cells"
tmp.mkdir(exist_ok=True)
reps = [c[len(c) // 2] for c in clusters]
for f in reps:
    fp = tmp / ("f%06d.png" % f)
    if not fp.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(VID / f"{video}.mp4"),
                        "-vf", "select=eq(n\,%d),crop=%d:%d:%d:%d,scale=%d:%d:flags=neighbor"
                        % (f, W, H, x1, y1, W * S, H * S),
                        "-vsync", "0", "-frames:v", "1", str(fp)], check=True)

from PIL import Image, ImageDraw
rows = (len(reps) + COLS - 1) // COLS
per_sheet = 25   # 5x5
for si in range(0, len(reps), per_sheet):
    chunk = reps[si:si + per_sheet]
    r = (len(chunk) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * CW, r * CH), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for i, f in enumerate(chunk):
        im = Image.open(tmp / ("f%06d.png" % f))
        cx, cy = (i % COLS) * CW, (i // COLS) * CH
        sheet.paste(im, (cx, cy + 22))
        dr.text((cx + 4, cy + 4),
                "f%d T=%s A=%s B=%s" % (f, truth.get(f), off[f], on[f]),
                fill=(255, 255, 0))
    outp = OUT / ("sheet_%02d.png" % (si // per_sheet))
    sheet.save(outp)
    print("已生成", outp)
