"""每 run 固定成本分解（S6 后续：短任务/批量场景的主要开销）。

现象：600 帧跑 h264-cpu 墙钟 0.30s，其中 `pipeline.calibrate` **0.091s**——
而校准批只有 50 帧（CPU 解码 ~15ms / NVDEC ~49ms）。两个后端都稳定 0.09s
说明主体不是解码，而是**每 run 重建的设备侧对象**（analyzer / 池 / 首次 kernel）。

本探针在**热进程**里逐项计时（CUDA 上下文已建立），给出可池化项的清单。

用法：python tools/_probe_run_setup_cost.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tick(label: str, t0: float) -> float:
    d = time.perf_counter() - t0
    print("  %-34s %7.1f ms" % (label, d * 1000))
    return d


def main() -> int:
    from video_ocr_engine import FieldExtractor
    vid = r"D:\Videos\racelog_test\test5.mp4"
    roi = (843, 993, 948, 1025)
    # 先跑一次把 CUDA 上下文/引擎池预热（此后测到的都是"每 run 重复支付"的部分）
    ex = FieldExtractor(vid, roi, frame_start=0, frame_end=600,
                        decode_backend="cpu", ocr_backend="tensorrt")
    t = time.perf_counter()
    ex.extract()
    print("预热 run 墙钟 %.3fs" % (time.perf_counter() - t))

    print("\n[CUDA 已热] 每 run 重建项：")
    from video_ocr_engine.ocr.trt import GpuFrameAnalyzer
    for i in (1, 2):
        t = time.perf_counter()
        a = GpuFrameAnalyzer()
        tick("GpuFrameAnalyzer() 第 %d 次" % i, t)
        a.release()
    from video_ocr_engine._gpu_kernels import GpuPreprocessor
    t = time.perf_counter()
    p = GpuPreprocessor()
    tick("GpuPreprocessor()（OCR 预处理，懒加载）", t)
    p.release()

    from cuda.bindings import runtime as cudart
    t = time.perf_counter()
    _e, ptr = cudart.cudaMalloc(1 << 20)
    tick("cudaMalloc 1MB", t)
    cudart.cudaFree(ptr)
    t = time.perf_counter()
    _e, s = cudart.cudaStreamCreate()
    tick("cudaStreamCreate", t)
    cudart.cudaStreamDestroy(s)

    print("\n[对照] 再两次完整 extract 的相位：")
    for rep in range(2):
        ex = FieldExtractor(vid, roi, frame_start=0, frame_end=600,
                            decode_backend="cpu", ocr_backend="tensorrt")
        t = time.perf_counter()
        r = ex.extract()
        w = time.perf_counter() - t
        sp = r.meta["report"]["spans"]
        print("  rep%d wall=%.3f  setup=%.3f cal=%.3f tail=%.3f init=%.4f"
              % (rep, w, sp.get("pipeline.setup", {}).get("sum", 0),
                 sp.get("pipeline.calibrate", {}).get("sum", 0),
                 sp.get("pipeline.ocr_tail", {}).get("sum", 0),
                 r.meta["report"]["gauges"].get("ocr.engine_init", 0)))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
