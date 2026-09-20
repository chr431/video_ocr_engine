"""裁切（OCR ROI 宽自适应）复测探针（2026-09-20 复测轮）。

对 log/2026-09-20-裁切与左对齐实验.md 的复测与归因，臂按 env 参数化：

  A1   AUTOCROP=0 REORDER=1    ← 旧 A 臂（不裁 + 顺序分批）
  A64  AUTOCROP=0 REORDER=64   ← **拆混变量**：不裁但按宽分组（A vs C 差的不只裁切）
  C    AUTOCROP=1 REORDER=64   ← 现役默认
  C781 C + OCR_PAD_SMALL=781   ← **尺度归因**：裁切内容强制 pad 回 781（几何不变、
                                  只改尺度；A1 全宽即 781）
  C639 C + OCR_PAD_SMALL=639   ← 尺度响应曲线中点

两种模式：
  accuracy  顺序单跑全片，落 JSON 到 _scratch，逐臂 diff（段级文本硬门）+
            指定段号区间打印（真值核对用）
  perf      两臂交错配对（臂序逐轮轮转），子进程 reps 取暖轮，配对差分
            wall / ocr.infer / ocr.q_get_wait（q_get_wait = OCR worker 等
            任务的阻塞时长，此前 21s→0.4s 流水线效应的复核口径）

用法：
  python tools/_probe_autocrop_retest.py --mode accuracy \
      --arms A1,A64,C,C781,C639 --groups 7591-7607,15413-15424,12478-12479
  python tools/_probe_autocrop_retest.py --mode perf --a A1 --b C --pairs 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["PROBE_ROOT"] = str(ROOT)   # `python -c` WORKER 无 __file__
_BATCH_DIR = Path(os.environ.get("RACELOG_BATCH_DIR", r"D:\Videos\batch_test"))
PY = sys.executable

ARMS: dict[str, dict[str, str]] = {
    "A1": {"OCR_ROI_AUTOCROP": "0", "OCR_REORDER_WINDOW": "1"},
    "A64": {"OCR_ROI_AUTOCROP": "0", "OCR_REORDER_WINDOW": "64"},
    "C": {"OCR_ROI_AUTOCROP": "1", "OCR_REORDER_WINDOW": "64"},
    "C781": {"OCR_ROI_AUTOCROP": "1", "OCR_REORDER_WINDOW": "64",
             "OCR_PAD_SMALL": "781"},
    "C639": {"OCR_ROI_AUTOCROP": "1", "OCR_REORDER_WINDOW": "64",
             "OCR_PAD_SMALL": "639"},
}

WORKER = r"""
import os, sys, time, json, statistics
sys.path.insert(0, os.environ["PROBE_ROOT"])
os.environ['ENGINE_PROFILE'] = '1'
path, roi_s, n, dbe, obe, stride = sys.argv[1:7]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
out = []
for rep in range(int(sys.argv[7])):
    ex = FieldExtractor(path, roi, frame_end=int(n) or None,
                        sample_stride=int(stride),
                        decode_backend=dbe, ocr_backend=obe, keep_crops=False)
    t0 = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t0
    out.append({
        'wall': round(wall, 3), 'segs': len(r.segments),
        'texts': [s.text for s in r.segments],
        'confs': [round(float(s.confidence), 5) for s in r.segments],
        'timing': {k: round(v, 3) for k, v in ex.timing.items()},
        'ocr': {k: round(v, 3) for k, v in ex.profile.get('ocr', {}).items()},
        'backend': r.meta['backend'], 'ocr_backend': r.meta.get('ocr_backend'),
    })
print(json.dumps(out))
"""

_CLEAR = ("OCR_ROI_AUTOCROP", "OCR_REORDER_WINDOW", "OCR_PAD_SMALL")


def run_arm(video, roi, n, dbe, obe, stride, arm, reps):
    e = dict(os.environ)
    for k in _CLEAR:
        e.pop(k, None)
    e.update(ARMS[arm])
    p = subprocess.run(
        [PY, "-c", WORKER, video, roi, str(n), dbe, obe, str(stride),
         str(reps)],
        capture_output=True, text=True, env=e)
    out = (p.stdout or "").strip().splitlines()
    if p.returncode != 0 or not out:
        raise RuntimeError(f"{arm} FAIL: {(p.stderr or '').strip()[-400:]}")
    return json.loads(out[-1])


def _fmt(d):
    return (f"wall={d['wall']:7.3f}  segs={d['segs']}  "
            f"decode={d['timing'].get('decode', 0):6.2f}  "
            f"infer={d['ocr'].get('infer', 0):6.2f}  "
            f"q_get_wait={d['ocr'].get('q_get_wait', 0):6.2f}  "
            f"[{d['ocr_backend']}/{d['backend']}]")


def mode_accuracy(a):
    out_dir = Path(os.environ.get("AUTOCROP_RETEST_OUT",
                                  r"D:\Repo\_scratch\autocrop_retest"))
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = a.arms.split(",")
    res = {}
    for arm in arms:
        runs = run_arm(a.video, a.roi, a.frames, a.dbe, a.ocr, a.stride,
                       arm, reps=1)
        res[arm] = runs[0]
        (out_dir / f"{Path(a.video).stem}_{arm}.json").write_text(
            json.dumps(runs[0], ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"  {arm:5s} {_fmt(runs[0])}")
    ref = res[arms[0]]
    print(f"\n  以 {arms[0]} 为基准的逐段文本差异：")
    for arm in arms[1:]:
        d = res[arm]
        diffs = [(i, x, y) for i, (x, y) in enumerate(
            zip(ref["texts"], d["texts"])) if x != y]
        print(f"  {arm:5s} vs {arms[0]}: {len(diffs)} / {len(ref['texts'])}")
        for i, x, y in diffs[:20]:
            print(f"        [{i}] {x!r} → {y!r}")
        if len(diffs) > 20:
            print(f"        ……（另 {len(diffs) - 20} 段，见 JSON）")
    if a.groups:
        print("\n  指定段号区间（真值核对）：")
        for g in a.groups.split(","):
            lo, hi = (int(v) for v in g.split("-"))
            for arm in arms:
                ts = res[arm]["texts"]
                uniq = sorted(set(ts[lo:hi + 1]))
                print(f"    [{lo}-{hi}] {arm:5s}: {uniq}")
    return 0


def mode_perf(a):
    arms = (a.a, a.b)
    data = {arm: [] for arm in arms}
    for i in range(a.pairs):
        order = list(arms) if i % 2 == 0 else list(arms)[::-1]
        for arm in order:
            runs = run_arm(a.video, a.roi, a.frames, a.dbe, a.ocr, a.stride,
                           arm, reps=a.reps)
            d = runs[-1]                      # 取暖轮
            data[arm].append(d)
            print(f"  p{i} {arm:5s} {_fmt(d)}")
            time.sleep(a.cooldown)
    print()
    for key in ("wall", "infer", "q_get_wait"):
        def get(d, k=key):
            return d[k] if k in d else d["ocr"].get(k, 0.0)
        ds = [(get(b) / get(x) - 1.0) * 100.0
              for x, b in zip(data[arms[0]], data[arms[1]])]
        mean = statistics.mean(ds)
        se = (statistics.stdev(ds) / len(ds) ** 0.5) if len(ds) > 1 else 0.0
        print(f"  {key:11s}（{arms[1]} vs {arms[0]}）%：均值 {mean:+7.3f}"
              f"  SE {se:5.3f}  逐对 {['%+.2f' % d for d in ds]}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("accuracy", "perf"), required=True)
    ap.add_argument("--arms", default="A1,A64,C,C781,C639")
    ap.add_argument("--a", default="A1")
    ap.add_argument("--b", default="C")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--video", default=str(_BATCH_DIR / "新三国01.mkv"))
    ap.add_argument("--roi", default="144,398,551,423")
    ap.add_argument("--frames", type=int, default=73500)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dbe", default="cpu")
    ap.add_argument("--ocr", default="auto")
    ap.add_argument("--groups", default="",
                    help="段号区间（真值核对），如 7591-7607,15413-15424")
    a = ap.parse_args()
    print(f"=== 裁切复测 [{a.mode}] {Path(a.video).name} "
          f"{a.frames}帧 stride={a.stride} decode={a.dbe} ocr={a.ocr} ===")
    return (mode_accuracy(a) if a.mode == "accuracy" else mode_perf(a))


if __name__ == "__main__":
    sys.exit(main())
