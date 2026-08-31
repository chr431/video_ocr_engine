r"""用 ground truth 判定「宽度自适应裁切」对识别准确率的影响。

离线对照（tools/_probe_roi_crop_ocr.py）只能看"文本是否变化"，看不出
变化是变好还是变坏。本探针按帧对齐真值 CSV，给出**准确率**，
才能判定裁切到底是增益还是回归。

真值格式（RaceVideoToLog 输出）：'#' 头 + 数据行 `frame,time,text,conf`。

用法：
  python tools/_probe_autocrop_truth.py \
      --video D:\Videos\racelog_test\test5.mp4 --roi 843,993,948,1025 \
      --truth D:\Videos\racelog_test\ground_truth_csv\test5_ref.csv \
      [--frames 7223] [--stride 1] [--ocr auto] [--reps 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
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
path, roi_s, n, dbe, obe, stride, fa = sys.argv[1:8]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_end=int(n), sample_stride=int(stride),
                    decode_backend=dbe, ocr_backend=obe, keep_crops=False,
                    force_aspect=float(fa))
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
got = {}
for s in r.segments:
    for f in (s.frames or (s.start,)):
        got[int(f)] = (s.text or "", round(float(s.confidence), 5))
print(json.dumps({'wall': round(wall, 3), 'segs': len(r.segments),
                  'got': got, 'backend': r.meta['backend'],
                  'ocr_backend': r.meta.get('ocr_backend')}))
"""


def load_truth(p: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
                continue
            out[int(parts[0])] = parts[2].strip()
    return out


def run(video, roi, n, dbe, obe, stride, fa, env, reps):
    e = dict(os.environ)
    e.pop("OCR_ROI_AUTOCROP", None)
    e.pop("OCR_REORDER_WINDOW", None)
    e.update(env)
    best = None
    for _ in range(reps):
        p = subprocess.run(
            [PY, "-c", WORKER, video, roi, str(n), dbe, obe, str(stride),
             str(fa)], capture_output=True, text=True, env=e)
        out = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not out:
            return {"err": (p.stderr or "").strip()[-500:]}
        d = json.loads(out[-1])
        if best is None or d["wall"] < best["wall"]:
            best = d
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(_VIDEO_DIR / "test5.mp4"))
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--truth",
                    default=r"D:\Videos\racelog_test\ground_truth_csv\test5_ref.csv")
    ap.add_argument("--frames", type=int, default=7223)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dbe", default="cpu")
    ap.add_argument("--ocr", default="auto")
    ap.add_argument("--force-aspect", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    truth = load_truth(Path(a.truth))
    print(f"=== 裁切 × 真值准确率：{Path(a.video).name} {a.frames}帧 "
          f"stride={a.stride} force_aspect={a.force_aspect} ===")
    print(f"    真值帧数 {len(truth)}"
          + ("；force_aspect>0 → 宽度自适应裁切自动跳过，无对照意义"
             if a.force_aspect > 0 else ""))

    cases = [
        ("A 旧行为(全宽+顺序)", {"OCR_ROI_AUTOCROP": "0", "OCR_REORDER_WINDOW": "1"}),
        ("C 裁+按宽分组", {"OCR_ROI_AUTOCROP": "1", "OCR_REORDER_WINDOW": "64"}),
    ]
    res = {}
    for name, env in cases:
        d = run(a.video, a.roi, a.frames, a.dbe, a.ocr, a.stride,
                a.force_aspect, env, a.reps)
        if "err" in d:
            print(f"  {name}: FAIL {d['err']}")
            continue
        got = {int(k): v[0] for k, v in d["got"].items()}
        both = [f for f in got if f in truth]
        ok = sum(1 for f in both if got[f] == truth[f])
        # 数值容错（速度读数常见尾位误差）
        def num_eq(x, y):
            try:
                return abs(float(x) - float(y)) <= max(
                    1e-3, 0.02 * max(abs(float(x)), abs(float(y)), 1.0))
            except ValueError:
                return False
        ok_num = sum(1 for f in both if got[f] == truth[f] or num_eq(got[f], truth[f]))
        d["acc"] = ok / len(both) if both else 0.0
        d["acc_num"] = ok_num / len(both) if both else 0.0
        d["n_cmp"] = len(both)
        res[name] = d
        print(f"  {name:20s} wall={d['wall']:7.3f}s  段={d['segs']:5d}  "
              f"比对帧={len(both)}  全等={ok} ({d['acc']:.3%})  "
              f"数值容错={ok_num} ({d['acc_num']:.3%})")

    if len(res) == 2:
        (n0, d0), (n1, d1) = list(res.items())
        print(f"\n  墙钟 {(d1['wall']/d0['wall']-1)*100:+.1f}%   "
              f"准确率 {d0['acc']:.3%} → {d1['acc']:.3%} "
              f"({(d1['acc']-d0['acc'])*100:+.2f} 个百分点)")
        print(f"  数值容错准确率 {d0['acc_num']:.3%} → {d1['acc_num']:.3%} "
              f"({(d1['acc_num']-d0['acc_num'])*100:+.2f} 个百分点)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
