"""FrameSource 端口（v2 §6.3 / D2）——解码后端的依赖注入契约。

S3-2 起定义；S3-3 起由 decord 适配器（decode/decord_source.py）实现。
语义要点来自 S0 冻结的 decoder_contract.yaml（DC-01..10）——尤其
roi_format（ROI-first 是引擎性能地基）与 color_range（决定 Y 平面展开）。
"""
from __future__ import annotations

from typing import Literal, Protocol, Sequence, runtime_checkable


@runtime_checkable
class FrameSource(Protocol):
    """一段视频的解码访问面（一次 run 绑定一个实例）。"""

    def __len__(self) -> int: ...
    @property
    def fps(self) -> float: ...
    @property
    def codec(self) -> str: ...
    @property
    def color_range(self) -> int: ...
    def roi_format(self) -> Literal["gray", "yuv420"]: ...
    def batch(self, frames: Sequence[int]): ...
    def close(self) -> None: ...
