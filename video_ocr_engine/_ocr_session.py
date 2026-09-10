"""OcrSession 已迁至 pipeline/ocr_stage.py（S3-3b，显式 SessionSpec 契约）。

本文件保留为兼容 shim：旧导入路径 `from video_ocr_engine._ocr_session
import OcrSession` 仍可用；实现与契约见新址。
"""
from __future__ import annotations

from .pipeline.ocr_stage import OcrSession, SessionSpec  # noqa: F401
