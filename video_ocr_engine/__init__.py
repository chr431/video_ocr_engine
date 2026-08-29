"""video_ocr_engine — 从视频固定区域提取文本的通用引擎。

职责：video → ROI → 像素分段 → 代表帧 → OCR 文本。
识别层只输出"文本及置信度"，不做任何领域后处理（速度/数值/纠错由
上层应用完成）。本引擎与 decord fork（chr431/decord）协作解码。

用法：
    from video_ocr_engine import FieldExtractor
    ex = FieldExtractor(video_path, roi, frame_start=..., frame_end=...)
    result = ex.extract()
    for seg in result.segments:
        print(seg.text, seg.confidence)
"""
import os as _os

# P0-6（2026-08-29 视觉裁定后翻默认开）：CPU 软解关去块滤波。
# HEVC 墙钟 -13%~-18%、h264 -5%~-13%（仅 CPU 软解；NVDEC 由硬件管不受影响）。
# 六片真值 + test4 逐帧视觉裁定（tools/_probe_slf_adjudicate.py）确认对 OCR
# 结果无负面影响：5 片全等准确率 +0.00~+0.08pp；test4 账面 -0.19pp 经证明是
# 真值伪影（显示为三位补零 "020"、前导 0 更暗，真值剥掉了前导零 —— 关滤波
# 恰好漏读暗淡前导零而"账面对"，视觉裁定按显示忠实度开反而略优）。
# 行为提示：显示为 2 位数字时（如 "020"）OCR 输出会带前导零（更忠实于显示），
# 下游做字符串匹配需注意。opt-out：预先设置 DECORD_SKIP_LOOP_FILTER=none
# （AVDiscard 语义，恢复完整去块滤波）；rep_crop 预览在关滤波下有块状伪影。
# 需要 decord fork ≥v0.7.13 的透传支持（旧版忽略该 env，行为不变）。
_os.environ.setdefault("DECORD_SKIP_LOOP_FILTER", "all")

from video_ocr_engine.extractor import (  # noqa: F401
    FieldExtractor, ExtractedSegment, ExtractionResult,
)
from video_ocr_engine import _version  # noqa: F401

__version__ = _version.__version__
