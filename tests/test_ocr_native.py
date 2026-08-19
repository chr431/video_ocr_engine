"""OCR 推理冒烟（ONNX CPU 后端；需仓库内模型资产，无需 decord/GPU）。"""
from __future__ import annotations

import numpy as np
import pytest

from ocr_native import OcrEngine


@pytest.fixture(scope="module")
def engine():
    return OcrEngine("v6_small", "onnxruntime", fill_width=224, num_threads=2)


def test_backend_name_is_onnxruntime(engine):
    assert engine.backend_name == "onnxruntime"


def test_onnx_backend_returns_one_result_per_input(engine):
    # 输入为 _preprocess_standard 输出风格 (48, w, 3) float32
    img = np.zeros((48, 96, 3), dtype=np.float32)
    res = engine([img, img.copy()])
    assert len(res) == 2
    for r in res:
        assert isinstance(r.txts, tuple)
        assert len(r.txts) == 1
        assert len(r.scores) == 1
        assert 0.0 <= r.scores[0] <= 1.0


def test_onnx_empty_input(engine):
    assert engine([]) == []


def test_ctc_decode_blank_only(engine):
    # 全 blank(logits 0 = blank) → 空文本、置信度 0
    pred = np.zeros((1, 8, 6906), dtype=np.float32)
    pred[:, 1:, 0] = 1.0  # 除首帧外选 blank
    out = engine._ctc_decode_batch(pred)
    assert len(out) == 1
    assert out[0].txts == ("",)
    assert out[0].scores[0] == 0.0
