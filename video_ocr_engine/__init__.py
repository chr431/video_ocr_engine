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
from video_ocr_engine.extractor import (  # noqa: F401
    FieldExtractor, ExtractedSegment, ExtractionResult,
)
from video_ocr_engine import _version  # noqa: F401

__version__ = _version.__version__
