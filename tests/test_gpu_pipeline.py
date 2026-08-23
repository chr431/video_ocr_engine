"""GPU 全驻留管线默认门控与合并模式的单元测试（无需视频/GPU）。"""
from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor
from video_ocr_engine import extractor as _ext


def _make(**kwargs):
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **kwargs)


@pytest.fixture
def gpu_ok(monkeypatch):
    """模拟 NVDEC+TRT 均可用。"""
    monkeypatch.setattr(_ext, "nvdec_available", lambda p: True)
    monkeypatch.setattr(_ext, "tensorrt_available", lambda: True)


def test_gpu_pipeline_default_on_for_gray_nvdec_trt(gpu_ok):
    ex = _make(gray_output=True)
    assert ex._gpu_pipeline_enabled() is True


def test_gpu_pipeline_opt_out_env(gpu_ok, monkeypatch):
    monkeypatch.setenv("RVTOL_GPU_PIPELINE", "0")
    ex = _make(gray_output=True)
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_requires_gray(gpu_ok):
    assert _make()._gpu_pipeline_enabled() is False          # rgb 输出
    assert _make(yuv_output=True)._gpu_pipeline_enabled() is False


def test_gpu_pipeline_requires_nvdec_decode(gpu_ok):
    ex = _make(gray_output=True, decode_backend="cpu")
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_requires_trt_ocr(gpu_ok):
    ex = _make(gray_output=True, ocr_backend="cpu")
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_unavailable_backends(monkeypatch):
    monkeypatch.setattr(_ext, "nvdec_available", lambda p: False)
    monkeypatch.setattr(_ext, "tensorrt_available", lambda: False)
    ex = _make(gray_output=True)
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_dual_takes_precedence(gpu_ok):
    ex = _make(gray_output=True, dual_pipeline=True)
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_contrast_merge_falls_back(gpu_ok, monkeypatch):
    monkeypatch.setenv("RVTOL_TEXT_SEP", "contrast")
    ex = _make(gray_output=True)
    assert ex._gpu_pipeline_enabled() is False


def test_merge_effective_mode_resolution(monkeypatch):
    ex = _make()
    # 引擎默认 binary
    assert ex._merge_effective_mode() == "binary"
    # TEXT_SEP 覆盖引擎默认
    monkeypatch.setenv("RVTOL_TEXT_SEP", "contrast")
    assert ex._merge_effective_mode() == "contrast"
    monkeypatch.setenv("RVTOL_TEXT_SEP", "1")
    assert ex._merge_effective_mode() == "contrast"
    monkeypatch.setenv("RVTOL_TEXT_SEP", "2")
    assert ex._merge_effective_mode() == "binary"
    # TEXT_SEP_MERGE 最高优先级
    monkeypatch.setenv("RVTOL_TEXT_SEP_MERGE", "off")
    assert ex._merge_effective_mode() == ""
    monkeypatch.delenv("RVTOL_TEXT_SEP_MERGE")
    monkeypatch.delenv("RVTOL_TEXT_SEP")
    # 显式构造参数 merge_text_sep
    ex2 = _make(merge_text_sep="")
    assert ex2._merge_effective_mode() == ""
