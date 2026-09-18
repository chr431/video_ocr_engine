"""解码口径单一事实源：引擎与 tools/ 探针共用的打开参数派生。

背景（2026-09-18 口径轮）：fork 级探针曾各自硬编码线程档与 ROI 换算
——实测 5 个 live 探针 ROI 少 ``+1``（解码窗比引擎小 1px）、THREADS
表复制策略且不吃 ``DECODE_THREADS`` 覆盖、新探针 format 误用 yuv420
（引擎 GPU 管线缺省 gray）。引擎档位一改探针即漂移，「探针口径 ≠
生产口径」的测量结论不可比。本模块把派生逻辑收敛到一处：探针
import 这里，永远与 ``FieldExtractor`` 同口径。

探针用法::

    from video_ocr_engine.config.decode_caliber import (
        decode_num_threads, roi_for_decord)
    nt = decode_num_threads(codec)                    # cpu/hybrid CPU 臂
    roi = roi_for_decord((843, 993, 948, 1025))        # 真值口径 → decord 半开

- GPU（NVDEC）路径引擎不传 ``num_threads``（硬件解码），对照臂同此。
- 引擎 decord output_format 缺省 **gray**（keep_crops=False 的生产
  口径）；keep_crops+yuv 时才是 'yuv420'——fork 级探针缺省应对齐 gray。
- 窗口运行（frame_end < 全长）时 fork ≥硬窗界版应 ``set_decode_window``。
"""
from __future__ import annotations

import os


def cpu_physical_cores() -> int:
    """物理核数（psutil 缺失时用逻辑核/2 估算，最小 2）。

    2026-09-18 自 ``ocr.native`` 迁入（单一事实源；native 反向引用保持
    兼容）：解码口径模块不得依赖 OCR 装载链（native→trt→cuda），
    探针 worker 才能零负担 import 本模块。
    """
    try:
        import psutil  # type: ignore[import-not-found]
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        physical = None
    if not physical:
        physical = max(2, (os.cpu_count() or 8) // 2)
    return max(2, int(physical))


def roi_for_decord(roi) -> tuple:
    """真值口径 ROI (x1, y1, x2, y2 闭区间) → decord roi (x2/y2 半开)。

    探针直接把真值表塞给 ``VideoReader(roi=...)`` 曾系统性少 1px
    （105×32 vs 引擎 106×33）——一律经此换算。
    """
    return (roi[0], roi[1], roi[2] + 1, roi[3] + 1)


def decode_num_threads(codec: str | None = None, *,
                       sample_stride: int = 1,
                       ocr_on_gpu: bool = True,
                       override: int = 0,
                       cores: int | None = None,
                       logical: int | None = None) -> int | None:
    """CPU 软解（cpu 后端 / hybrid CPU 臂）的 decord FFmpeg 帧线程数。

    完整分档依据与实测表见 ``FieldExtractor._decode_num_threads``
    （extractor.py，2026-09-18 起委托本函数——同源派生）。要点：
    - ``override > 0``（``DECODE_THREADS`` env / RunConfig）直返；
    - av1：逻辑核 3/4 钳 [8,24]（stride>1 放宽到 48 上限）；
    - hevc：钳 [8,32]（stride>1 放宽到 48；不分 OCR 位置）；
    - 其余：GPU-OCR = 逻辑核钳 [GPU_OCR_MIN, GPU_OCR_MAX]；
      CPU-OCR 按物理核阈值/stride 分档（extractor 表）。

    探针调用的缺省（stride=1, ocr_on_gpu=True）= 引擎 GPU 管线口径。
    """
    if override > 0:
        return override
    from video_ocr_engine.config import constants as config
    if cores is None:
        cores = cpu_physical_cores()
    if logical is None:
        logical = os.cpu_count() or cores
    if codec == 'av1':
        if sample_stride > 1:
            return max(8, min(48, logical))
        return max(8, min(24, logical * 3 // 4))
    if codec == 'hevc':
        if sample_stride > 1:
            return max(8, min(48, logical))
        return max(8, min(32, logical))
    if ocr_on_gpu:
        return max(config.DECODE_THREADS_GPU_OCR_MIN,
                   min(config.DECODE_THREADS_GPU_OCR_MAX, logical))
    if cores <= config.CPU_CORES_SPLIT_THRESHOLD:
        return max(2, cores // 2)
    if sample_stride > 1:
        return max(8, min(config.DECODE_THREADS_CPU_OCR_MAX,
                          logical * 3 // 4))
    return max(8, min(config.DECODE_THREADS_CPU_OCR_STRIDE1_MAX,
                      logical // 3))
