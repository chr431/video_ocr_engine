"""ocr —— OCR 后端层（v2 §5）：端口协议 + ONNX/TRT 实现 + 引擎池。

S9 起自根模块迁入（`ocr_native` → `ocr/native.py`、`ocr_trt` → `ocr/trt.py`）；
根处保留兼容 shim 至 0.14.0。
"""
from __future__ import annotations
