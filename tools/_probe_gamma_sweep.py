"""误读帧的预处理变体扫描（离线：单帧 → 预处理变体 → ONNX OCR）。

对基线里确诊的 OCR 误读帧（8→日 / 1→曰 / 5→S 等），逐帧扫 gamma 与
对比度拉伸变体，看哪个变体能纠正、哪个变体会引入新误读：
  - gamma 档：1.0 / 1.5 / 1.75 / 2.0(现役) / 2.25 / 2.5 / 3.0
  - 对比度拉伸：p2-p98 百分位归一（gamma=1）
  - 对照组：同视频抽 N 个基线读对帧（防"修 8 破 6"）

用法：python tools/_probe_gamma_sweep.py [test5|test6|all]
落盘 bench/gamma_sweep.json
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

_VDIR = Path(r"D:\Videos\racelog_test")
_TRUTH = _VDIR / "ground_truth_csv"

# 确诊误读帧（来自 bench/acc_baseline.json mid+boundary 复核）与对照帧
CASES = {
    "test6": {   # (frame, 真值, 引擎读数)
        "misread": [(431, "88", "8日"), (548, "81", "曰1"), (549, "81", "曰1"),
                    (5725, "148", "14日"), (5933, "88", "8日"),
                    (5934, "88", "8日"), (6063, "88", "8日"),
                    (6064, "88", "8日"), (13207, "81", "曰1"),
                    (13890, "88", "8日"), (14009, "78", "7日"),
                    (19840, "118", "11日")],
        "ctrl": [(500, None), (1000, None), (5000, None),
                 (10000, None), (15000, None), (20000, None)],
    },
    "test5": {
        "misread": [(2531, "55", "5S"), (2533, "55", "5S"),
                    (2535, "55", "5S"), (2537, "55", "5S"),
                    (2576, "55", "SS"), (3851, "115", "11S"),
                    (6603, "55", "SS"), (6714, "51", "S1")],
        "ctrl": [(600, None), (1500, None), (3000, None),
                 (4500, None), (6000, None)],
    },
}
GAMMAS = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]


def roi_of(csv: str):
    t = (_TRUTH / csv).read_text(encoding="utf-8-sig")
    return tuple(int(x) for x in re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", t).groups())


def truth_row(csv: str, f: int) -> str:
    for line in (_TRUTH / csv).read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("#"):
            continue
        p = line.split(",")
        if len(p) >= 3 and p[0].lstrip("-").isdigit() and int(p[0]) == f:
            return p[2].strip()
    return ""


def main() -> int:
    import decord
    from video_ocr_engine.domain.segmentation import (_otsu, crop_to_content,
                                                       preprocess_standard)
    from video_ocr_engine.ocr.native import OcrEngine

    eng = OcrEngine(variant="v6_small", engine_type="onnxruntime",
                    num_threads=8)
    report = {}
    for vid, spec in CASES.items():
        csv = f"{vid}_ref.csv"
        roi = roi_of(csv)
        vr = decord.VideoReader(str(_VDIR / f"{vid}.mp4"), num_threads=2)
        frames = sorted({f for f, *_ in spec["misread"]}
                        | {f for f, *_ in spec["ctrl"]})
        arr = vr.get_batch(frames, roi=(roi[0], roi[1], roi[2] + 1,
                                        roi[3] + 1)).asnumpy()
        gray_all = {f: (arr[i].astype(np.float32)
                        @ np.array([0.299, 0.587, 0.114],
                                   dtype=np.float32)).astype(np.uint8)
                    for i, f in enumerate(frames)}

        rep = {}
        for kind in ("misread", "ctrl"):
            for case in spec[kind]:
                f = case[0]
                t = case[1] or truth_row(csv, f)   # ctrl 真值从 CSV 取
                g = gray_all[f]
                th = _otsu(g)          # 单帧近似（引擎用 50 帧中位，裁切判据同构）
                base = crop_to_content(g, th)
                row = {"truth": t}
                for gm in GAMMAS:
                    p = preprocess_standard(base, gamma=gm)
                    r0 = eng([p])
                    txt = r0[0].txts[0] if isinstance(r0, list) else r0.txts[0]
                    row[f"g{gm}"] = txt
                # 对比度拉伸变体：p2-p98 线性归一到 0-255，gamma=1
                lo, hi = np.percentile(base, (2, 98))
                if hi > lo:
                    st = np.clip((base.astype(np.float32) - lo)
                                 / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
                    p = preprocess_standard(st, gamma=1.0)
                    r0 = eng([p])
                    row["stretch"] = (r0[0].txts[0] if isinstance(r0, list)
                                      else r0.txts[0])
                rep[f"{kind}:{f}"] = row
        report[vid] = rep

        # 汇总本片
        for kind in ("misread", "ctrl"):
            tot = ok = {gm: 0 for gm in GAMMAS} | {}, {gm: 0 for gm in GAMMAS} | {"stretch": 0}
        print(f"== {vid}")
        for kind in ("misread", "ctrl"):
            for key, row in rep.items():
                if not key.startswith(kind):
                    continue
                t = row["truth"]
                marks = " ".join(
                    f"{k.replace('g', 'γ') if k.startswith('g') else k}"
                    f"={'✓' if v == t else '✗' + v}"
                    for k, v in row.items() if k != "truth")
                print(f"  [{kind}] f{key.split(':')[1]} truth={t}: {marks}")
    out = ROOT / "bench" / "gamma_sweep.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
