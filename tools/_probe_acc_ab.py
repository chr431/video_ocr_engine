"""准确率 A/B：env 旋钮各档位跑真实引擎全片，按真值逐帧比对。

与 _probe_acc_baseline.py 的区别：本探针**参数化 env 旋钮**，用于验证
"离线单帧扫描"给出的候选（如 OCR_GAMMA=1.0、SEG_MERGE_DENSE_GATE）在
**真实管线**（GPU/TRT 默认路径、force_aspect、代表帧、合并）下是否成立。
离线扫描用 crop_to_content + 单帧 Otsu，与生产口径不同，不能直接下结论。

用法：
  python tools/_probe_acc_ab.py --knob OCR_GAMMA=2.0,1.0
  python tools/_probe_acc_ab.py --knob SEG_MERGE_DENSE_GATE=5,0 --videos test5,test6
  python tools/_probe_acc_ab.py --knob OCR_GAMMA=2.0,1.0 --baseline 2.0

落盘 bench/acc_ab_<knob>.json。误差分类口径与基线探针一致（边界类/段中类）。
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

WORKER = r"""
import os, sys, time, json
sys.path.insert(0, os.environ["PROBE_ROOT"])
sys.stdout.reconfigure(encoding="utf-8")
path, roi_s, fs, fe = sys.argv[1:5]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
# 填充宽度扫描入口（2026-09-13）：fill_width 无 env 旋钮（只能改常量），
# 故经本 env 传给构造参数；0/未设 = 用引擎默认。配合 --knob PROBE_FILL_WIDTH=...
_fw = int(os.environ.get("PROBE_FILL_WIDTH", "0") or 0)
_fw_kw = {"fill_width": _fw} if _fw > 0 else {}
ex = FieldExtractor(path, roi, frame_start=int(fs), frame_end=int(fe),
                    sample_stride=1, decode_backend="auto",
                    ocr_backend="auto", keep_crops=False,
                    merge_similar=os.environ.get(
                        "PROBE_MERGE_SIMILAR", "1") == "1", **_fw_kw)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
got = {}
for s in r.segments:
    for f in (s.frames or (s.start,)):
        got[int(f)] = (s.text or "")
segs = [[int(s.start), int(s.end), int(s.rep_frame), s.text or ""]
        for s in r.segments]
print("ACCJSON " + json.dumps({"wall": round(wall, 2),
      "segs": len(r.segments), "got": got, "seg_list": segs},
      ensure_ascii=False))
"""

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


def run_case(video: str, roi, fs, fe, envx: dict, timeout: int = 900):
    env = dict(os.environ)
    env["PROBE_ROOT"] = str(ROOT)
    env.update(envx)
    try:
        p = subprocess.run(
            [sys.executable, "-c", WORKER, str(_VDIR / video),
             ",".join(map(str, roi)), str(fs), str(fe)],
            env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    for ln in p.stdout.splitlines():
        if ln.startswith("ACCJSON "):
            return json.loads(ln[8:])
    print("    WORKER_FAIL: " + ((p.stderr or "").strip().splitlines() or ["?"])[-1][:160])
    return None


def classify(truth: dict[int, str], got: dict[int, str], k: int):
    frames = sorted(truth)
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
            strip0 += 1
            stripz.append((f, t, g))
            continue
        rec = (f, t, g)
        near = any(abs(f - c) <= k for c in changes)
        (boundary if near else mid).append(rec)
    return exact, strip0, stripz, boundary, mid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True,
                    help="NAME=v1,v2,...（env 值型，逗号分隔档位）")
    ap.add_argument("--videos", default="")
    ap.add_argument("--boundary-k", type=int, default=1)
    ap.add_argument("--baseline", default="",
                    help="基线档位值（默认第一个），用于打印相对 delta")
    args = ap.parse_args()

    name, _, vals = args.knob.partition("=")
    arms = [v.strip() for v in vals.split(",") if v.strip()]
    if not arms:
        print("--knob 需给出至少一个档位值")
        return 2
    base = args.baseline.strip() or arms[0]
    # 允许 --videos test5 或 test5.mp4 两种写法
    want = ({t.strip() if t.strip().endswith(".mp4") else t.strip() + ".mp4"
             for t in args.videos.split(",")} if args.videos else None)

    report = {"knob": name, "arms": arms, "baseline": base,
              "boundary_k": args.boundary_k, "cases": {}}
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
        per_arm = {}
        for a in arms:
            r = run_case(video, roi, fs, fe, {name: a})
            if r is None:
                per_arm[a] = None
                continue
            got = {int(k): v for k, v in r["got"].items()}
            exact, strip0, stripz, boundary, mid = classify(
                truth, got, args.boundary_k)
            per_arm[a] = {
                "frames": len(truth), "segs": r["segs"], "wall": r["wall"],
                "exact_n": exact, "acc_exact": round(exact / len(truth), 5),
                "acc_strip0": round(strip0 / len(truth), 5),
                "strip0_n": len(stripz), "boundary_n": len(boundary),
                "mid_n": len(mid),
                "mid_all": mid, "boundary_all": boundary,
            }
        report["cases"][video] = {"kind": kind, "arms": per_arm}

        b = per_arm.get(base)
        print("== %s (%s) n=%d" % (video, kind,
                                   b["frames"] if b else -1))
        for a in arms:
            c = per_arm[a]
            if c is None:
                print("   %-6s FAIL" % a)
                continue
            d = ("" if b is None or a == base else
                 "  Δexact=%+.4f" % (c["acc_exact"] - b["acc_exact"]))
            print("   %-6s exact=%.4f strip0=%.4f 边界=%3d 段中=%3d "
                  "segs=%4d %.1fs%s" % (
                      a, c["acc_exact"], c["acc_strip0"], c["boundary_n"],
                      c["mid_n"], c["segs"], c["wall"], d))
    out = ROOT / "bench" / ("acc_ab_%s.json" % name.lower().replace(".", "_"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("落盘 %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
