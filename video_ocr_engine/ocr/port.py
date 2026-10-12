"""OcrBackend 端口 —— OCR 后端的依赖注入契约。

池的 checkout 契约：同一 backend 实例同时只被一个线程使用。
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
