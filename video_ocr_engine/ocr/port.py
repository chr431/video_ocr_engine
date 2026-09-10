"""OcrBackend 端口（v2 §6.3 / D2）——OCR 后端的依赖注入契约。

S3-2 起定义；S3-3 起由 onnx_backend / trt_backend（v1 ocr_native /
ocr_trt 的移植）实现。池的 checkout 契约（§9）：同一 backend 实例
同时只被一个线程使用。
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class OcrBackend(Protocol):
    backend_name: str
    max_batch: int
    supports_device_input: bool

    def call(self, images: Sequence[np.ndarray]) -> list: ...
    def release(self) -> None: ...
