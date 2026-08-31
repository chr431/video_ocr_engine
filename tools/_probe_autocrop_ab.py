"""OCR 输入宽度自适应裁切：端到端 A/B（墙钟 + 文本一致）。

对照组（全 env 可控）：
  A  OCR_ROI_AUTOCROP=0 OCR_REORDER_WINDOW=1   ← 旧行为（全宽 + 顺序分批）
  B  OCR_ROI_AUTOCROP=1 OCR_REORDER_WINDOW=1   ← 只裁不分组（预期几乎无收益）
  C  OCR_ROI_AUTOCROP=1 OCR_REORDER_WINDOW=64  ← 裁 + 跨批按宽分组（现役默认）

一致性门：**全段文本**必须与 A 一致（置信度允许小数点后抖动——裁切会改变
CTC 序列长度，故文本是硬门、置信度是软门）。

用法：
  python tools/_probe_autocrop_ab.py --video X --roi a,b,c,d [--frames N]
      [--stride S] [--ocr auto|cpu] [--reps 3]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

# WORKER 以 `python -c` 执行，-c 下 __file__ 未定义，
# 故本进程算好根目录后经 PROBE_ROOT 传给子进程。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROBE_ROOT"] = ROOT  # 供 `python -c` 的 WORKER 子进程使用
_BATCH_DIR = Path(os.environ.get("RACELOG_BATCH_DIR", r"D:\Videos\batch_test"))


PY = sys.executable

WORKER = r"""
import os, sys, time, json, statistics
sys.path.insert(0, os.environ["PROBE_ROOT"])
os.environ['ENGINE_PROFILE'] = '1'
path, roi_s, n, dbe, obe, stride = sys.argv[1:7]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_end=int(n), sample_stride=int(stride),
                    decode_backend=dbe, ocr_backend=obe, keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
texts = [s.text for s in r.segments]
confs = [round(float(s.confidence), 5) for s in r.segments]
print(json.dumps({
    'wall': round(wall, 3), 'segs': len(r.segments),
    'texts': texts, 'confs': confs,
    'mean_conf': round(statistics.mean(confs), 5) if confs else 0.0,
    'timing': {k: round(v, 3) for k, v in ex.timing.items()},
    'ocr': {k: round(v, 3) for k, v in
            sorted(ex.profile.get('ocr', {}).items(),
                   key=lambda kv: -kv[1])[:6]},
    'backend': r.meta['backend'], 'ocr_backend': r.meta.get('ocr_backend'),
}))
"""

CASES = [
    ("A 旧行为(全宽+顺序)", {"OCR_ROI_AUTOCROP": "0", "OCR_REORDER_WINDOW": "1"}),
    ("B 只裁不分组", {"OCR_ROI_AUTOCROP": "1", "OCR_REORDER_WINDOW": "1"}),
    ("C 裁+按宽分组(默认)", {"OCR_ROI_AUTOCROP": "1", "OCR_REORDER_WINDOW": "64"}),
]


def run(video, roi, n, dbe, obe, env, reps, stride):
    e = dict(os.environ)
    e.pop("OCR_ROI_AUTOCROP", None)
    e.pop("OCR_REORDER_WINDOW", None)
    e.pop("OCR_ROI_AUTOCROP_MARGIN", None)
    e.update(env)
    best = None
    for _ in range(reps):
        p = subprocess.run(
            [PY, "-c", WORKER, video, roi, str(n), dbe, obe, str(stride)],
            capture_output=True, text=True, env=e)
        out = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not out:
            return {"err": (p.stderr or "").strip()[-500:]}
        d = json.loads(out[-1])
        if best is None or d["wall"] < best["wall"]:
            best = d
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(_BATCH_DIR / "新三国01.mkv"))
    ap.add_argument("--roi", default="144,398,551,423")
    ap.add_argument("--frames", type=int, default=30000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--dbe", default="cpu")
    ap.add_argument("--ocr", default="auto")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    roi = ",".join(str(x) for x in (int(x) for x in a.roi.split(",")))
    print(f"=== 宽度自适应裁切 A/B：{Path(a.video).name} {a.frames}帧 "
          f"stride={a.stride} decode={a.dbe} ocr={a.ocr} ===")

    res = {}
    for name, env in CASES:
        d = run(a.video, roi, a.frames, a.dbe, a.ocr, env, a.reps, a.stride)
        if "err" in d:
            print(f"  {name}: FAIL {d['err']}")
            continue
        res[name] = d
        print(f"  {name:20s} wall={d['wall']:6.3f}s  segs={d['segs']}  "
              f"mean_conf={d['mean_conf']:.5f}  "
              f"decode={d['timing'].get('decode', 0):.3f}  "
              f"[{d['ocr_backend']}]")
        print(f"      ocr: {d['ocr']}")

    if "A 旧行为(全宽+顺序)" not in res:
        return 2
    ref = res["A 旧行为(全宽+顺序)"]
    ok = True
    print()
    for name, d in res.items():
        if name.startswith("A"):
            continue
        same = sum(1 for x, y in zip(ref["texts"], d["texts"]) if x == y)
        rate = same / max(1, len(ref["texts"]))
        if rate < 1.0:
            ok = False
        dc = [abs(x - y) for x, y in zip(ref["confs"], d["confs"])]
        print(f"  {name:20s} 墙钟 {(d['wall']/ref['wall']-1)*100:+6.1f}%  "
              f"文本一致 {same}/{len(ref['texts'])} ({rate:.2%})  "
              f"|Δconf| 中位 {statistics.median(dc):.1e} max {max(dc):.1e}")
        if rate < 1.0:
            bad = [(i, x, y) for i, (x, y) in enumerate(zip(ref["texts"], d["texts"]))
                   if x != y][:6]
            for i, x, y in bad:
                print(f"      差异[{i}] A={x!r} → {name[0]}={y!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
