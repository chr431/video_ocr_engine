"""hybrid CPU 臂线程档位的引擎 e2e 配对 A/B（§14 线程消融的引擎复验）。

decode-only 结论（_probe_hol_stats --nts）：24T 甜点（av1 +10%、h264lg
+7~10%、hevc 平、32T 无增益）。但引擎 e2e 有分段/OCR 线程抢核，档位最优
必须在引擎口径复验（S6 续轮 12→16 即引擎口径定的）。本探针按 C-05 统一
口径跑全片 e2e：独立子进程槽位、配对交错、段数 + 唯一文本集做正确性门禁。

用法（DECORD_LIBRARY_PATH 由 --fork 注入子进程）：
  python tools/_probe_hybrid_threads_e2e.py --videos test6_h264.mp4 \
      --arms 16,24 --pairs 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKER = r'''
import os, sys, time, json, hashlib
sys.path.insert(0, os.environ["PROBE_ROOT"])
sys.stdout.reconfigure(encoding="utf-8")
from video_ocr_engine.extractor import FieldExtractor
vid = os.environ["PROBE_VID"]
roi = ((841, 994, 950, 1027) if ("hevc" in vid or "test6" in vid)
       else (843, 993, 949, 1026))
ex = FieldExtractor(video_path=vid, roi=roi,
                    decode_backend="hybrid", ocr_backend="tensorrt",
                    keep_crops=False)
t = time.perf_counter()
res = ex.extract()
wall = time.perf_counter() - t
texts = sorted({s.text for s in res.segments})
print("E2EJSON " + json.dumps({
    "wall": wall, "segs": len(res.segments),
    "texts_sha": hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest(),
    "cpuT": os.environ.get("HYBRID_CPU_THREADS", "?")}))
'''

_VDIR = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")


def run_slot(vid: str, arm: str, fork: str) -> dict | None:
    env = dict(os.environ)
    env.update({"PROBE_ROOT": str(ROOT), "PROBE_VID": vid,
                "HYBRID_CPU_THREADS": arm})
    if fork:
        env["DECORD_LIBRARY_PATH"] = fork
    try:
        p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        return {"wall": None, "err": "TIMEOUT"}
    for ln in p.stdout.splitlines():
        if ln.startswith("E2EJSON "):
            return json.loads(ln[len("E2EJSON "):])
    return {"wall": None, "err": "WORKER_FAIL:" + p.stderr.strip()[-200:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="test6_h264.mp4")
    ap.add_argument("--arms", default="16,24")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--fork", default=os.environ.get("DECORD_FORK_BUILD", ""))
    args = ap.parse_args()

    vids = [os.path.join(_VDIR, v) for v in args.videos.split(",")]
    arms = args.arms.split(",")
    cells = [(v, a) for v in vids for a in arms]
    res: dict[tuple, list] = {}
    for i in range(args.pairs):
        rot = cells[i % len(cells):] + cells[:i % len(cells)]
        for vid, arm in rot:
            t0 = time.perf_counter()
            r = run_slot(vid, arm, args.fork)
            res.setdefault((vid, arm), []).append(r)
            if r.get("wall") is None:
                print("  [%d] %-16s arm=%s %s" % (i, os.path.basename(vid),
                                                  arm, r.get("err")), flush=True)
                continue
            print("  [%d] %-16s arm=%s %7.3fs segs=%d %s"
                  % (i, os.path.basename(vid), arm, r["wall"], r["segs"],
                     r["texts_sha"][:10]), flush=True)
    print()
    ok = True
    for vid in vids:
        walls = {}
        for arm in arms:
            v = [x["wall"] for x in res.get((vid, arm), []) if x.get("wall")]
            if not v:
                print("%-16s arm=%s 全挂" % (os.path.basename(vid), arm))
                ok = False
                continue
            walls[arm] = v
            shas = {x["texts_sha"] for x in res[(vid, arm)]
                    if x.get("wall")}
            segs = {x["segs"] for x in res[(vid, arm)] if x.get("wall")}
            if len(shas) > 1 or len(segs) > 1:
                print("⚠️ %s arm=%s 段数/文本集不稳: segs=%s shas=%s"
                      % (os.path.basename(vid), arm, segs,
                         [s[:8] for s in shas]))
                ok = False
        if len(walls) == len(arms) and len(arms) == 2:
            a, b = arms
            med_a = statistics.median(walls[a])
            med_b = statistics.median(walls[b])
            pair_diffs = []
            for wa, wb in zip(walls[a], walls[b]):
                pair_diffs.append((wb - wa) / wa * 100)
            shas_a = {x["texts_sha"] for x in res[(vid, a)]
                      if x.get("wall")}
            shas_b = {x["texts_sha"] for x in res[(vid, b)]
                      if x.get("wall")}
            same = shas_a == shas_b
            print("%-16s %s=%6.3fs %s=%6.3fs  Δ(B−A)/A=%.2f%%"
                  "  配对差%s  文本集一致=%s"
                  % (os.path.basename(vid), a, med_a, b, med_b,
                     (med_b - med_a) / med_a * 100,
                     ["%+.2f%%" % d for d in pair_diffs], same))
            if not same:
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
