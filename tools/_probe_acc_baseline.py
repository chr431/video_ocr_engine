"""准确率基线 + 误差分类（2026-09-12 准确项第一步：先量后做）。

按真值 CSV 头（# roi=/frame_start=/frame_end=）严格对齐跑引擎默认配置
（decode/ocr auto、sample_stride=1），逐帧比对给出：

  1. 基线准确率（exact + 剥前导零两种口径——已知显示伪影，见 DECISIONS
     P0-6 翻案）；
  2. 误差分类：**边界类**（真值文本变化点 ±k 帧内 = 分段时机/裁切问题）
     vs **段中文本类**（OCR/预处理/裁切内容问题）；
  3. 段中文本类的模式归类（剥零即同 / 1 字符距离 / 多字符）与样例帧，
     落盘 `bench/acc_baseline.json` 供后续归因探针消费。

**真值可靠性（2026-09-12 实测校准，勿再当作绝对真值）**：truth 各片人工
校对、ref 只保证 ±1，两者可信度不同所以分列；但**真值本身也会严于可见
像素**——抽检原始 ROI 证实：test4 f4652-4665 有一条移动的过曝白带扫过，
三位真值的**首位在像素上不可见**（引擎只能读到两位），真值显然来自序列
连续性；test2 f740-744 的**前导零是褪色幽灵**（真值记 `37` 而显示 `037`）。
故「边界类/段中类」只是**定位标签不是成因定论**，疑点必须用
`_probe_roi_dump.py` 导原始像素目视裁定（本轮即据此把 test5 f687/688
判为引擎真错、把 test4 的丢位判为真值严于可见）。用法：

  python tools/_probe_acc_baseline.py                 # 全部 6 片
  python tools/_probe_acc_baseline.py --videos test5,test6
  python tools/_probe_acc_baseline.py --boundary-k 2   # 边界类带宽（默认 1）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

_VDIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))
_TRUTH = _VDIR / "ground_truth_csv"
OUT = ROOT / "bench" / "acc_baseline.json"

WORKER = r"""
import os, sys, time, json
sys.path.insert(0, os.environ["PROBE_ROOT"])
sys.stdout.reconfigure(encoding="utf-8")
path, roi_s, fs, fe = sys.argv[1:5]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_start=int(fs), frame_end=int(fe),
                    sample_stride=1, decode_backend="auto",
                    ocr_backend="auto", keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
got = {}
for s in r.segments:
    for f in (s.frames or (s.start,)):
        got[int(f)] = (s.text or "")
print("ACCJSON " + json.dumps({"wall": round(wall, 2),
      "segs": len(r.segments), "got": got}, ensure_ascii=False))
"""

# (视频, 真值, 可信度)
PAIRS = [("test.mp4", "test_truth.csv", "truth"),
         ("test2.mp4", "test2_truth.csv", "truth"),
         ("test3.mp4", "test3_truth.csv", "truth"),
         ("test4.mp4", "test4_truth.csv", "truth"),
         ("test5.mp4", "test5_ref.csv", "ref"),
         ("test6.mp4", "test6_ref.csv", "ref")]


def load_truth(p: Path):
    roi, fs, fe = None, 0, None
    rows: dict[int, str] = {}
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("#"):
            m = re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", line)
            if m:
                roi = tuple(int(x) for x in m.groups())
            m = re.search(r"frame_start=(\d+)", line)
            if m:
                fs = int(m.group(1))
            m = re.search(r"frame_end=(\d+)", line)
            if m:
                fe = int(m.group(1))
            continue
        parts = line.split(",")
        if len(parts) >= 3 and parts[0].lstrip("-").isdigit():
            rows[int(parts[0])] = parts[2].strip()
    return roi, fs, fe, rows


def strip_zeros(t: str) -> str:
    return t.lstrip("0") or "0"


def run_case(video: str, roi, fs, fe) -> dict | None:
    env = dict(os.environ)
    env["PROBE_ROOT"] = str(ROOT)
    try:
        p = subprocess.run(
            [sys.executable, "-c", WORKER, str(_VDIR / video),
             ",".join(map(str, roi)), str(fs), str(fe)],
            env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        return None
    for ln in p.stdout.splitlines():
        if ln.startswith("ACCJSON "):
            return json.loads(ln[8:])
    print("    WORKER_FAIL: " + ((p.stderr or "").strip().splitlines() or ["?"])[-1][:100])
    return None


def classify(truth: dict[int, str], got: dict[int, str], k: int):
    frames = sorted(truth)
    # 真值文本变化点（含首帧）
    changes = {f for i, f in enumerate(frames)
               if i == 0 or truth[f] != truth[frames[i - 1]]}
    exact = strip0 = 0
    stripz, boundary, mid = [], [], []
    for f in frames:
        t, g = truth[f], got.get(f)
        if g is None:
            near = any(abs(f - c) <= k for c in changes)
            (boundary if near else mid).append((f, t, "<缺段>"))
            continue
        if g == t:
            exact += 1
            strip0 += 1
            continue
        if strip_zeros(g) == strip_zeros(t):
            # 口径伪影（truth 剥零 vs 显示忠实）：不算引擎真错，单列
            strip0 += 1
            stripz.append((f, t, g))
            continue
        rec = (f, t, g)
        near = any(abs(f - c) <= k for c in changes)
        (boundary if near else mid).append(rec)
    return exact, strip0, stripz, boundary, mid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="")
    ap.add_argument("--boundary-k", type=int, default=1)
    args = ap.parse_args()
    want = set(args.videos.split(",")) if args.videos else None

    report = {"boundary_k": args.boundary_k, "cases": {}}
    for video, truthf, kind in PAIRS:
        if want and video not in want:
            continue
        roi, fs, fe, truth = load_truth(_TRUTH / truthf)
        if roi is None or fe is None:
            print("跳过 %s：头缺 roi/帧窗" % video)
            continue
        n = fe - fs + 1
        if len(truth) < n * 0.5:
            print("跳过 %s：真值行 %d < 帧窗一半" % (video, len(truth)))
            continue
        r = run_case(video, roi, fs, fe)
        if r is None:
            report["cases"][video] = {"err": "FAIL"}
            continue
        got = {int(k): v for k, v in r["got"].items()}
        exact, strip0, stripz, boundary, mid = classify(truth, got, args.boundary_k)
        # 段中文本类模式归类
        pats: dict[str, int] = {}
        samples: dict[str, list] = {}
        for f, t, g in mid:
            if strip_zeros(g) == strip_zeros(t):
                key = "剥零即同"
            elif abs(len(g) - len(t)) <= 1 and sum(
                    1 for a, b in zip(g, t) if a != b) + abs(len(g) - len(t)) <= 1:
                key = "1字符距离"
            elif g == "":
                key = "空读"
            else:
                key = "多字符差"
            pats[key] = pats.get(key, 0) + 1
            samples.setdefault(key, []).append([f, t, g])
        n_truth = len(truth)
        rep = {
            "kind": kind, "frames": n_truth, "segs": r["segs"],
            "wall": r["wall"],
            "acc_exact": round(exact / n_truth, 5),
            "acc_strip0": round(strip0 / n_truth, 5),
            "strip0_n": len(stripz),
            "boundary_n": len(boundary),
            "mid_n": len(mid),
            "mid_patterns": pats,
            "strip0_all": stripz,
            "boundary_all": boundary,
            "mid_all": mid,
        }
        report["cases"][video] = rep
        print("%-11s[%5s] 帧=%d %5.1fs  exact=%.4f strip0=%.4f  剥零=%d "
              "真边界=%d 真段中=%d %s" % (
                  video, kind, n_truth, r["wall"],
                  rep["acc_exact"], rep["acc_strip0"], len(stripz),
                  len(boundary), len(mid),
                  {k: v for k, v in sorted(pats.items(), key=lambda x: -x[1])}))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("落盘 %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
