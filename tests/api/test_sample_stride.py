"""sample_stride（分频采样）参数测试：构造校验（解码路径需 decord，不在此测）。"""
from __future__ import annotations

from video_ocr_engine import FieldExtractor
import engine_config as config


def test_default_is_one():
    ex = FieldExtractor("x.mp4", (0, 0, 10, 10))
    assert ex._sample_stride == 1
    assert config.DEFAULT_SAMPLE_STRIDE == 1


def test_explicit_stride():
    ex = FieldExtractor("x.mp4", (0, 0, 10, 10), sample_stride=3)
    assert ex._sample_stride == 3


def test_stride_forced_ge_one():
    assert FieldExtractor("x.mp4", (0, 0, 10, 10), sample_stride=0)._sample_stride == 1
    assert FieldExtractor("x.mp4", (0, 0, 10, 10), sample_stride=-2)._sample_stride == 1
