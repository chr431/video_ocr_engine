"""C-68 修复效果测量：三码 hybrid 缺口（与 2026-10-10 矩阵同口径）。

口径：test6 同内容三码，3000 帧，TRT OCR，热池（每臂预热 1 次后 3 reps
取中位）；GPU_PIPELINE=1。臂 = nvdec 纯 / cpu 纯 / hybrid。
产出：三码 × {纯臂×2, hybrid, 理论和, 缺口%, CPU 臂实得}。
"""
import os
import statistics
import sys
import time

sys.path.insert(0, "tools")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("GPU_PIPELINE", "1")

from _measure_lib import ROI, VIDS  # noqa: E402
from video_ocr_engine import FieldExtractor  # noqa: E402

CASES = [("hevc", "test6_hevc"), ("h264", "test6_h264"), ("av1", "test6_av1")]
WIN = 3000


def run(vid: str, backend: str) -> float:
    ex = FieldExtractor(VIDS[vid], ROI["test6"], frame_start=0, frame_end=WIN,
                        decode_backend=backend, ocr_backend="tensorrt",
                        sample_stride=1)
    ex.extract()  # 预热
    walls = []
    for _ in range(3):
        ex2 = FieldExtractor(VIDS[vid], ROI["test6"], frame_start=0, frame_end=WIN,
                             decode_backend=backend, ocr_backend="tensorrt",
                             sample_stride=1)
        t0 = time.perf_counter()
        r = ex2.extract()
        walls.append(time.perf_counter() - t0)
        n = len(r.segments)
    return WIN / statistics.median(walls), n


print("code   nvdec    cpu   理论和  hybrid   缺口    cpu实得")
for name, vid in CASES:
    r_g, _ = run(vid, "nvdec")
    r_c, _ = run(vid, "cpu")
    r_h, _ = run(vid, "hybrid")
    theo = r_g + r_c
    # CPU 臂实得：hybrid 墙钟内 CPU 臂交付帧数（hybrid_stats 穿透略，
    # 用缺口反解口径与 10-10 矩阵一致：实得 = 理论和比例无——直接给 hybrid
    # 速率与缺口；臂内分率由 fork stats 在引擎 report 里，此处从略）
    print("%-5s %6.0f %6.0f %7.0f %7.0f %+6.1f%%" % (
        name, r_g, r_c, theo, r_h, (r_h - theo) / theo * 100))
