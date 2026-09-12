"""merge_similar 决策插桩：全片跑引擎，记录每次 on_similar 调用的
(mean, changed_px, diff 的 win3 簇分, 判定)，并输出逐帧 got。

目的：验证「merge 吞真实短状态」机制覆盖率——边界误差里有多少帧
是被 win3 稠密（真实变化）但 changed_px ≤ 上限的合并吞掉的。

  python tools/_probe_merge_log.py test6
  python tools/_probe_merge_log.py test5
落盘 bench/merge_log_<video>.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from video_ocr_engine import FieldExtractor
from video_ocr_engine.domain.segmentation import (_cluster_win3,
                                                   _text_sep_binary)

_VDIR = Path(r"D:\Videos\racelog_test")
_TRUTH = _VDIR / "ground_truth_csv"
OUT = ROOT / "bench"

PAIRS = {"test": "test_truth.csv", "test2": "test2_truth.csv",
         "test3": "test3_truth.csv", "test4": "test_truth.csv",  # test4 同 ROI 不需要
         "test5": "test5_ref.csv", "test6": "test6_ref.csv"}
PAIRS["test4"] = "test4_truth.csv"


def load_header(p: Path):
    txt = p.read_text(encoding="utf-8-sig")
    roi = tuple(int(x) for x in re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", txt).groups())
    fs = int(re.search(r"frame_start=(\d+)", txt).group(1))
    fe = int(re.search(r"frame_end=(\d+)", txt).group(1))
    return roi, fs, fe


def main() -> int:
    vid = sys.argv[1] if len(sys.argv) > 1 else "test6"
    mp4 = f"{vid}.mp4"
    roi, fs, fe = load_header(_TRUTH / PAIRS[vid])

    logs: list[dict] = []
    ex = FieldExtractor(str(_VDIR / mp4), roi, frame_start=fs, frame_end=fe,
                        sample_stride=1, decode_backend="auto",
                        # 强制宿主管线：GPU 路径的合并在设备侧，不走
                        # _segments_similar；分段/合并语义两管线逐位一致（C-32）
                        ocr_backend="cpu", keep_crops=False)
    orig = ex._segments_similar
    th_holder = ex

    def patched(a, b, _pl=[None]):
        ba = _text_sep_binary(a, th_holder._bin_thresh)
        bb = _text_sep_binary(b, th_holder._bin_thresh)
        diff = np.abs(ba.astype(np.int16) - bb.astype(np.int16))
        mean = float(diff.mean())
        npx = int(np.sum(diff > 10))
        win3 = _cluster_win3(diff > 10)
        verdict = orig(a, b)
        logs.append({"mean": round(mean, 3), "npx": npx,
                     "win3": int(win3), "merged": bool(verdict),
                     "gray_a_std": round(float(a.std()), 2),
                     "gray_b_std": round(float(b.std()), 2)})
        return verdict

    ex._segments_similar = patched
    r = ex.extract()

    got: dict[int, str] = {}
    rep_by_seg = []
    for s in r.segments:
        for f in (s.frames or (s.start,)):
            got[int(f)] = s.text or ""
        rep_by_seg.append([s.start, s.end, s.rep_frame, s.text, s.confidence])

    OUT.mkdir(exist_ok=True)
    out = OUT / f"merge_log_{vid}.json"
    out.write_text(json.dumps({
        "video": vid, "n_merge_calls": len(logs),
        "n_merged": sum(1 for L in logs if L["merged"]),
        "swallow_win3ge5": sum(1 for L in logs if L["merged"] and L["win3"] >= 5),
        "swallow_win3ge5_win3ge8": sum(1 for L in logs if L["merged"] and L["win3"] >= 8),
        "logs": logs, "got": {str(k): v for k, v in got.items()},
        "segments": rep_by_seg}, ensure_ascii=False), encoding="utf-8")
    n_sw = sum(1 for L in logs if L["merged"] and L["win3"] >= 5)
    print(f"{vid}: merge 调用 {len(logs)} 次，合并 {sum(1 for L in logs if L['merged'])} 次，"
          f"其中 win3≥5 的可疑吞并 {n_sw} 次（win3≥8: "
          f"{sum(1 for L in logs if L['merged'] and L['win3'] >= 8)}）")
    print(f"落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
