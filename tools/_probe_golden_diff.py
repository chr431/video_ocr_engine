"""金标差异定位（S6 轮）：对指定用例重跑并与已录向量逐段对比。

用法：python tools/_probe_golden_diff.py C-h264-nvdec [前缀段数]
打印首个差异段、两侧段数、以及 rep_frame/text/conf 的差异分布——
用于判断"是结构漂移还是数值末位"（§8.5：段数与文本 sha 永不豁免）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "golden"))

from record import GOLDEN, MATRIX, load_case, summarize  # noqa: E402


def main() -> int:
    cid = sys.argv[1] if len(sys.argv) > 1 else "C-h264-nvdec"
    case = next(c for c in MATRIX if c["id"] == cid)
    calib_old, ocr_old = load_case(case)
    # 同进程连跑两次：第 1 次池冷（TRT 反序列化 0.31s，前几十段在
    # raw_ready 置位前 emit → 走宿主预处理），第 2 次池热（0.07s →
    # 全程 GPU raw 直通）。两条预处理的浮点流差异即 F-7 的敏感段。
    for run in (1, 2):
        calib_new, ocr_new = summarize(case)
        sn, so = ocr_new["segments"], ocr_old["segments"]
        n = min(len(sn), len(so))
        diffs = [i for i in range(n) if sn[i] != so[i]]
        print("[run %d %s] 段数 %d/%d struct_sha %s text_sha %s 置信度差异 %d 段"
              % (run, "冷" if run == 1 else "热", len(sn), len(so),
                 "同" if ocr_new["seg_struct_sha"] == ocr_old["seg_struct_sha"]
                 else "异",
                 "同" if ocr_new["text_sha"] == ocr_old["text_sha"] else "异",
                 len(diffs)))
        for i in diffs[:4]:
            print("   [%d] new=%r old=%r" % (i, sn[i], so[i]))
        for k in calib_new:
            if calib_new[k] != calib_old.get(k):
                print("   calib.%s new=%r old=%r"
                      % (k, calib_new[k], calib_old.get(k)))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
