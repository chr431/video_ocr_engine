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
    rep_crop: Any = None            # 代表帧 ROI 图像（rep_crop_format 决定：
                                    # "yuv"=packed NV12，外部用 nv12_to_rgb
                                    # 转 RGB；"gray"=灰度 (H,W) uint8）


@dataclass
class ExtractionResult:
    """引擎通用提取结果（无领域语义）。"""

    segments: list = field(default_factory=list)  # list[ExtractedSegment]
    frames: list = field(default_factory=list)     # 全部采样帧号
    fps: float = 0.0                               # 自测帧率
    timing: dict = field(default_factory=dict)     # 各阶段耗时
    meta: dict = field(default_factory=dict)       # backend/codec/参数/降级原因等

    def rep_crop_rgb(self, seg):
        """段代表帧 → RGB (H, W, 3) uint8 预览（DESIGN-REVIEW D4 helper）。

        布局按 meta['rep_crop_format'] 解释（"yuv"=packed NV12 / "gray"；
        引擎写入的是实际生效格式，yuv 不被支持而降级时会是 "gray"）；
        NV12→RGB 按 meta['color_range']（缺省 limited）展开。
        返回新数组，不修改 rep_crop 本体；rep_crop 为 None 时返回 None。
        """
        import numpy as np
        c = seg.rep_crop
        if c is None:
            return None
        meta = self.meta or {}
        fmt = meta.get('rep_crop_format') or 'yuv'
        color_range = int(meta.get('color_range') or 0)
        if fmt == 'yuv' and c.ndim == 2:
            from video_utils import nv12_to_rgb
            return nv12_to_rgb(c, color_range)
        g = c[..., 0] if c.ndim == 3 else c
        return np.repeat(g[..., None], 3, axis=-1)
