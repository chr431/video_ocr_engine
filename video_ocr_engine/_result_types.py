"""引擎通用结果类型（从 extractor.py 拆出，无类依赖）。

ExtractedSegment / ExtractionResult；extractor 与 __init__ 从这里 re-export。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class ExtractedSegment:
    """引擎输出的单个文本字段段（原有字段区间 + 代表帧 + 原始文本）。"""

    start: int                      # 段首帧号
    end: int                        # 段末帧号
    frames: tuple = ()              # 段内帧号序列
    rep_frame: int = -1             # 代表帧号（段内最清晰帧）
    text: Optional[str] = None      # OCR 原始文本（None=未读出）
    confidence: float = 0.0         # OCR 置信度 0-1
    rep_crop: Any = None            # 代表帧 ROI 图像（YUV420 或 RGB）


@dataclass
class ExtractionResult:
    """引擎通用提取结果（无领域语义）。"""

    segments: list = field(default_factory=list)  # list[ExtractedSegment]
    frames: list = field(default_factory=list)     # 全部采样帧号
    fps: float = 0.0                               # 自测帧率
    timing: dict = field(default_factory=dict)     # 各阶段耗时
    meta: dict = field(default_factory=dict)       # backend/codec/引擎版本等
