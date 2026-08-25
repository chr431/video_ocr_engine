"""GPU 全驻留管线默认门控与合并模式的单元测试（无需视频/GPU）。"""
from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor
# GPU 门控方法在 _gpu_pipeline mixin 模块；模拟可用性需 patch 该模块的
# nvdec_available / tensorrt_available（split 后不再位于 extractor 命名空间）。
from video_ocr_engine import _gpu_pipeline as _gpu


def _make(**kwargs):
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **kwargs)


@pytest.fixture
def gpu_ok(monkeypatch):
    """模拟 NVDEC 可用（分段/校准 kernel 只依赖 CUDA，不依赖 TRT）。"""
    monkeypatch.setattr(_gpu, "nvdec_available", lambda p: True)


def test_gpu_pipeline_default_on_for_gray_nvdec_trt(gpu_ok):
    ex = _make(gray_output=True)
    assert ex._gpu_pipeline_enabled() is True


def test_gpu_pipeline_opt_out_env(gpu_ok, monkeypatch):
    monkeypatch.setenv("GPU_PIPELINE", "0")
    ex = _make(gray_output=True)
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_default_yuv_supported(gpu_ok):
    # 默认 rep_crop_format="yuv"（keep_crops 时内部 yuv420，GPU 侧用
    # luma_nv12 kernel 提取 Y 平面）——GPU 管线应启用（此前 yuv 门控回退宿主）。
    ex = _make()
    assert ex._rep_crop_format == "yuv"
    assert ex._yuv_output is True
    assert ex._gpu_pipeline_enabled() is True
    assert _make(yuv_output=True)._gpu_pipeline_enabled() is True


def test_gpu_pipeline_keep_crops_false_no_yuv(gpu_ok):
    # keep_crops=False：无 UV 需求 → decord 直接 gray 输出（省 0.5B/px 传输）
    ex = _make(rep_crop_format="yuv", keep_crops=False)
    assert ex._yuv_output is False
    assert ex._gpu_pipeline_enabled() is True


def test_gpu_pipeline_ocr_backend_independent(gpu_ok):
    # GPU 分段与 OCR 后端解耦：ocr_backend="cpu"（ONNX）也走 GPU 分段
    #（OCR 阶段代表帧 D2H + 宿主预处理）。
    ex = _make(gray_output=True, ocr_backend="cpu")
    assert ex._gpu_pipeline_enabled() is True


def test_gpu_pipeline_requires_nvdec_decode(gpu_ok):
    ex = _make(gray_output=True, decode_backend="cpu")
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_unavailable_backends(monkeypatch):
    monkeypatch.setattr(_gpu, "nvdec_available", lambda p: False)
    ex = _make(gray_output=True)
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_contrast_merge_falls_back(gpu_ok, monkeypatch):
    monkeypatch.setenv("TEXT_SEP_MERGE", "contrast")
    ex = _make(gray_output=True)
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_force_aspect_falls_back(gpu_ok):
    # GPU raw 直通（process_gray_raw）按自然宽高比缩放，不支持强制宽高比；
    # 有 force_aspect 时必须走宿主路径，否则两路径 OCR 输入/结果不一致。
    ex = _make(gray_output=True, force_aspect=2.0)
    assert ex._gpu_pipeline_enabled() is False


def test_merge_effective_mode_resolution(monkeypatch):
    ex = _make()
    # 引擎默认 binary
    assert ex._merge_effective_mode() == "binary"
    # TEXT_SEP_MERGE 覆盖引擎默认
    monkeypatch.setenv("TEXT_SEP_MERGE", "contrast")
    assert ex._merge_effective_mode() == "contrast"
    monkeypatch.setenv("TEXT_SEP_MERGE", "1")
    assert ex._merge_effective_mode() == "contrast"
    monkeypatch.setenv("TEXT_SEP_MERGE", "2")
    assert ex._merge_effective_mode() == "binary"
    monkeypatch.setenv("TEXT_SEP_MERGE", "off")
    assert ex._merge_effective_mode() == ""
    monkeypatch.delenv("TEXT_SEP_MERGE")
    # 显式构造参数 merge_text_sep
    ex2 = _make(merge_text_sep="")
    assert ex2._merge_effective_mode() == ""
