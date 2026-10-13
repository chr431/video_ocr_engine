"""OCR 推理冒烟（CPU/OpenVINO 后端；需仓库内模型资产，无需 decord/GPU）。"""

from __future__ import annotations

import numpy as np
import pytest

from video_ocr_engine.ocr.native import OcrEngine


@pytest.fixture(scope="module")
def engine():
    return OcrEngine("v6_small", "openvino", fill_width=224, num_threads=2)


def test_backend_name_is_openvino(engine):
    # ORT 后端已移除（历史见 git 档案）；CPU OCR 引擎必须为 openvino
    # 引擎标识为 'openvino'（历史迁移表见 git 档案 archive/pre-rework:docs/MIGRATION.md）：
    # CPU 路径唯一引擎 = OpenVINO。
    assert engine.backend_name == "openvino"


def test_openvino_backend_returns_one_result_per_input(engine):
    # 输入为 _preprocess_standard 输出风格 (48, w, 3) float32
    img = np.zeros((48, 96, 3), dtype=np.float32)
    res = engine([img, img.copy()])
    assert len(res) == 2
    for r in res:
        assert isinstance(r.txts, tuple)
        assert len(r.txts) == 1
        assert len(r.scores) == 1
        assert 0.0 <= r.scores[0] <= 1.0


def test_openvino_empty_input(engine):
    assert engine([]) == []


def test_ctc_decode_blank_only(engine):
    # 全 blank(logits 0 = blank) → 空文本、置信度 0
    pred = np.zeros((1, 8, 6906), dtype=np.float32)
    pred[:, 1:, 0] = 1.0  # 除首帧外选 blank
    out = engine._ctc_decode_batch(pred)
    assert len(out) == 1
    assert out[0].txts == ("",)
    assert out[0].scores[0] == 0.0
