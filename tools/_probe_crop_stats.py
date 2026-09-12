"""代表帧裁切图的强度分布对比（回答「逐图自适应曲线是否可能可分」）。

动机：真实管线 A/B 显示 test5 与 test6 在**每一个全局色调旋钮**上都反向
（γ=1.0：−44/+23；逐图拉伸：−45/+17），γ=2.0 是折中。若存在可用的逐图
自适应规则，其判别量必须在**单张裁切图内**可测——本探针把各片代表帧裁切
图的强度统计打出来看两类是否重叠。重叠 ⇒ 逐图自适应在像素层无判别信息。

用法：python tools/_probe_crop_stats.py [test5,test6,test,test4]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

_VDIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))
PAIRS = {"test": "test_truth.csv", "test2": "test2_truth.csv",
         "test3": "test3_truth.csv", "test4": "test4_truth.csv",
         "test5": "test5_ref.csv", "test6": "test6_ref.csv"}


def main() -> int:
    import re
    from video_ocr_engine import FieldExtractor
    from video_ocr_engine.domain.segmentation import _otsu

    names = (sys.argv[1].split(",") if len(sys.argv) > 1
             else ["test", "test4", "test5", "test6"])
    for nm in names:
        txt = (_VDIR / "ground_truth_csv" / PAIRS[nm]).read_text(
            encoding="utf-8-sig")
        roi = tuple(int(x) for x in re.search(
            r"roi=(\d+),(\d+),(\d+),(\d+)", txt).groups())
        fs = int(re.search(r"frame_start=(\d+)", txt).group(1))
        ex = FieldExtractor(str(_VDIR / f"{nm}.mp4"), roi, frame_start=fs,
                            frame_end=fs + 3000, sample_stride=1,
                            keep_crops=True, rep_crop_format="gray")
        r = ex.extract()
        rows = []
        for s in r.segments:
            c = s.rep_crop
            if c is None:
                continue
            g = np.asarray(c, dtype=np.float32)
            g = g[..., 0] if g.ndim == 3 else g
            if g.size < 64 or float(g.std()) < 3.0:
                continue
            g8 = np.clip(g, 0, 255).astype(np.uint8)
            th = _otsu(g8)
            hi = np.percentile(g, 99.5)          # 笔画峰值
            lo = np.percentile(g, 50)            # 背景
            fg = float((g > th).mean())          # 文字像素占比
            rows.append((lo, hi, hi - lo, th, fg, float(g.std())))
        a = np.array(rows)
        if not len(a):
            print("%-7s 无可用裁切图" % nm)
            continue
        med = np.median(a, axis=0)
        p10, p90 = np.percentile(a, 10, axis=0), np.percentile(a, 90, axis=0)
        print("%-7s n=%4d | 背景p50 %5.1f[%4.1f,%5.1f] 笔画p99.5 %5.1f"
              "[%4.1f,%5.1f] 对比 %5.1f[%4.1f,%5.1f] Otsu %5.1f 文字占比 "
              "%.3f std %5.1f" % (
                  nm, len(a), med[0], p10[0], p90[0], med[1], p10[1], p90[1],
                  med[2], p10[2], p90[2], med[3], med[4], med[5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
