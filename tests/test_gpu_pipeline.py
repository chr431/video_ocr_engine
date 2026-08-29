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
    """模拟 NVDEC + TensorRT 可用（默认规则要求两者，全程 raw 才启用）。"""
    monkeypatch.setattr(_gpu, "nvdec_available", lambda p: True)
    monkeypatch.setattr(_gpu, "tensorrt_available", lambda: True)


def test_gpu_pipeline_default_on_for_gray_nvdec_trt(gpu_ok):
    ex = _make(rep_crop_format="gray")
    assert ex._gpu_pipeline_enabled() is True


def test_gpu_pipeline_opt_out_env(gpu_ok, monkeypatch):
    monkeypatch.setenv("GPU_PIPELINE", "0")
    ex = _make(rep_crop_format="gray")
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_default_yuv_supported(gpu_ok):
    # 默认 rep_crop_format="yuv"（keep_crops 时内部 yuv420，GPU 侧用
    # luma_nv12 kernel 提取 Y 平面）——GPU 管线应启用。
    ex = _make()
    assert ex._rep_crop_format == "yuv"
    assert ex._yuv_output is True
    assert ex._gpu_pipeline_enabled() is True
    assert _make(rep_crop_format="yuv")._gpu_pipeline_enabled() is True


def test_gpu_pipeline_keep_crops_false_no_yuv(gpu_ok):
    # keep_crops=False：无 UV 需求 → decord 直接 gray 输出（省 0.5B/px 传输）
    ex = _make(rep_crop_format="yuv", keep_crops=False)
    assert ex._yuv_output is False
    assert ex._gpu_pipeline_enabled() is True


def test_gpu_pipeline_default_requires_trt(gpu_ok, monkeypatch):
    # 默认规则：无 TRT → GPU 分段+ONNX 实测无净收益 → 走宿主管线
    monkeypatch.setattr(_gpu, "tensorrt_available", lambda: False)
    ex = _make(rep_crop_format="gray")
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_forced_without_trt(gpu_ok, monkeypatch):
    # GPU_PIPELINE=1 强制尝试：允许 GPU 分段+ONNX 等实验组合
    monkeypatch.setattr(_gpu, "tensorrt_available", lambda: False)
    monkeypatch.setenv("GPU_PIPELINE", "1")
    ex = _make(rep_crop_format="gray", ocr_backend="cpu")
    assert ex._gpu_pipeline_enabled() is True


def test_gpu_pipeline_ocr_cpu_host_by_default(gpu_ok):
    # 默认规则：ocr_backend="cpu"（ONNX）→ 宿主管线（无全程 raw 收益）
    ex = _make(rep_crop_format="gray", ocr_backend="cpu")
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_cpu_decode_enabled_p13(gpu_ok, monkeypatch):
    # P1-3 解耦：decode_backend="cpu"（显式 CPU 软解）+ TRT → GPU 管线
    # （每批 H2D 进 GPU 分段/raw OCR，CPU 解码收益与零拷贝 OCR 叠加）
    ex = _make(rep_crop_format="gray", decode_backend="cpu")
    assert ex._gpu_pipeline_enabled() is True
    # CPU 解码分支跳过 NVDEC 探测：nvdec_available 为假也应启用
    monkeypatch.setattr(_gpu, "nvdec_available", lambda p: False)
    assert _make(rep_crop_format="gray", decode_backend="cpu")._gpu_pipeline_enabled() is True


def test_gpu_pipeline_hybrid_enabled_p83(gpu_ok):
    # §8.3 合并：hybrid 走 GPU 管线 CPU 分支（消费 HybridDecoder 宿主数组，
    # 双解码收益 + 零拷贝 OCR 叠加；原互斥门控已移除）
    ex = _make(rep_crop_format="gray", decode_backend="hybrid")
    assert ex._gpu_pipeline_enabled() is True


def test_content_range_to_crop_margin_math():
    # 宿主 _crop_to_content 与 GPU 直通共用的余量数学：
    # m = max(1, round(w * margin% / 100))、lo = max(0, first-m)、
    # hi = min(w, last+1+m)、满宽 → None（不裁）。
    #
    # 期望值**按 config 当前余量算出，不硬编码数字** —— 余量默认值是会调的
    # （2026-08-30 由 10 改为 20），写死会让本测试在调参时假失败。
    ex = _make(rep_crop_format="gray")
    w = 100
    m = max(1, int(round(w * ex._ocr_autocrop_margin_pct / 100.0)))

    assert ex._content_range_to_crop(0, w - 1, w) is None        # 满宽
    # 内容离边界不足 m → 加余量后仍满宽 → 不裁
    assert ex._content_range_to_crop(m - 1, w - m, w) is None
    # 内容离边界足够远 → 裁到 [first-m, last+1+m)
    first, last = m + 10, w - m - 11
    lo, cw = ex._content_range_to_crop(first, last, w)
    assert (lo, cw) == (first - m, last + 1 + m - (first - m))
    # 单列内容
    mid = w // 2
    lo2, cw2 = ex._content_range_to_crop(mid, mid, w)
    assert (lo2, cw2) == (mid - m, 1 + 2 * m)
    # 下限钳制：余量不小于 1px
    ex2 = FieldExtractor("dummy.mp4", (0, 0, 4, 2), rep_crop_format="gray")
    assert ex2._content_range_to_crop(2, 2, 4) == (1, 3)


def test_autocrop_margin_follows_config(monkeypatch):
    # 余量来自 config 且可被 env 覆盖（避免"改了默认值但没生效"的静默失败）。
    monkeypatch.setenv("OCR_ROI_AUTOCROP_MARGIN", "30")
    ex = _make(rep_crop_format="gray")
    assert ex._ocr_autocrop_margin_pct == 30
    # w=100、余量 30 → first=29 时 lo=0、hi=min(100, last+31)
    assert ex._content_range_to_crop(29, 69, 100) is None


def test_gpu_pipeline_unavailable_backends(monkeypatch):
    monkeypatch.setattr(_gpu, "nvdec_available", lambda p: False)
    ex = _make(rep_crop_format="gray")
    assert ex._gpu_pipeline_enabled() is False


def test_gpu_pipeline_contrast_supported(gpu_ok, monkeypatch):
    # contrast 合并判定：边界时 D2H 两帧 → 宿主 _segments_similar（已支持）
    monkeypatch.setenv("TEXT_SEP_MERGE", "contrast")
    ex = _make(rep_crop_format="gray")
    assert ex._gpu_pipeline_enabled() is True


def test_gpu_pipeline_force_aspect_supported(gpu_ok):
    # force_aspect：process_gray_raw 支持强制 content 宽（round 语义）
    ex = _make(rep_crop_format="gray", force_aspect=1.5)
    assert ex._gpu_pipeline_enabled() is True


def test_merge_effective_mode_resolution(monkeypatch):
    ex = _make()
    # 引擎默认 binary
    assert ex._merge_effective_mode() == "binary"
    # TEXT_SEP_MERGE 覆盖引擎默认（contrast 已于 0.9.0 删除 → 归一 binary）
    monkeypatch.setenv("TEXT_SEP_MERGE", "contrast")
    assert ex._merge_effective_mode() == "binary"
    monkeypatch.setenv("TEXT_SEP_MERGE", "1")
    assert ex._merge_effective_mode() == "binary"
    monkeypatch.setenv("TEXT_SEP_MERGE", "2")
    assert ex._merge_effective_mode() == "binary"
    monkeypatch.setenv("TEXT_SEP_MERGE", "off")
    assert ex._merge_effective_mode() == ""
    monkeypatch.delenv("TEXT_SEP_MERGE")
    # 显式构造参数 merge_text_sep
    ex2 = _make(merge_text_sep="")
    assert ex2._merge_effective_mode() == ""
