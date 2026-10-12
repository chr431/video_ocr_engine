"""GPU 可用性探测与设备侧门面。

本模块只剩**探测面**：_cuda_python_available 与 monkeypatch 点
（nvdec_available / tensorrt_available 的模块属性——tests/ 探针经
`video_ocr_engine.gpu.device` 命名空间 monkeypatch，extractor
`_gpu_pipeline_enabled` 经模块属性解析，两者都依赖本模块属性）。
池机制在 gpu/pools.py、协作流在 gpu/streams.py；下方 re-export 保旧
导入路径（frozen 探针 / tests / gpu_backend 懒导入）不断。
"""

import logging

from video_ocr_engine.domain.video_utils import nvdec_available, tensorrt_available  # noqa: F401 —— monkeypatch 点（经 extractor._gpu_pipeline_enabled 使用）

# 旧路径兼容 re-export（规范位置：pools/streams；零逻辑改动搬迁）
from .pools import (  # noqa: F401
    _CpuFrameRef,
    _DevBatch,
    _DevBatchPool,
    _YFrame,
    _YFramePool,
)
from .streams import (  # noqa: F401
    DevHooks,
    _GpuRunCtx,
    _gpu_fill_prev,
    _gpu_frame_stream_cpu,
    _gpu_frame_stream_nvdec,
    _gpu_prepare_calibration,
    _gpu_release_partial,
)

logger = logging.getLogger(__name__)


def _cuda_python_available() -> bool:
    """cuda-python（cuda.core / cuda.bindings）是否可导入。

    GPU 分段/校准/CTC kernel 依赖它；缺失时 GPU 管线会初始化失败——
    门控直接判不可用（避免带 NVDEC 但无 cuda-python 的环境崩在
    GpuFrameAnalyzer()）。
    """
    try:
        import importlib.util as _u

        return _u.find_spec("cuda") is not None
    except Exception:
        return False
