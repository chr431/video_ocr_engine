"""字幕场景 A/B：稠密簇门对字幕（保持型内容）准确率与成本的影响。

`text_test.mp4` 的真值是**时间戳式**（time_hms,text），不是逐帧式——本
探针按 fps 把每条字幕的生存区间映射到帧，取该区间内引擎的**多数文本**
与真值比对（区间 = [t_i, t_{i+1})）。这是字幕场景唯一可用的准确率口径。

用法：python tools/_probe_text_ab.py [--gates 0,5]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

VID = r"D:\Videos\text_video_test\text_test.mp4"
TRUTH = r"D:\Videos\text_video_test\text_test_truth.csv"
ROI = (266, 989, 1569, 1058)


def load_truth(path: str):
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("time_hms") or not line.strip():
            continue
        t, _, txt = line.partition(",")
        h, m, s = (int(x) for x in t.split(":"))
        rows.append((h * 3600 + m * 60 + s, txt.strip()))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="0,5")
    args = ap.parse_args()

    from video_ocr_engine import FieldExtractor
    truth = load_truth(TRUTH)
    for gate in (int(x) for x in args.gates.split(",")):
        os.environ["SEG_MERGE_DENSE_GATE"] = str(gate)
        ex = FieldExtractor(VID, ROI, decode_backend="auto",
                            ocr_backend="auto", keep_crops=False)
        t0 = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t0
        fps = r.fps
        got: dict[int, str] = {}
        for s in r.segments:
            for f in (s.frames or (s.start,)):
                got[int(f)] = s.text or ""
        # 每条字幕的生存区间 → 多数文本
        hit = tot = 0
        misses = []
        for i, (t, txt) in enumerate(truth):
            t_end = truth[i + 1][0] if i + 1 < len(truth) else t + 3
            fs, fe = int(t * fps), max(int(t * fps), int(t_end * fps) - 1)
            if fe < fs:
                continue
            cnt = Counter(got.get(f, "<缺段>") for f in range(fs, fe + 1))
            maj, _n = cnt.most_common(1)[0]
            tot += 1
            if maj == txt:
                hit += 1
            else:
                misses.append((fs, txt, maj))
        print("gate=%-2d segs=%4d wall=%.2fs  字幕行 %d/%d = %.4f" % (
            gate, len(r.segments), wall, hit, tot, hit / max(1, tot)))
        for f, t, g in misses[:6]:
            print("      f%-6d 真值=%r 引擎=%r" % (f, t, g))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
