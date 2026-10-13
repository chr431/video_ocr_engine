"""GPU 全驻留管线默认门控与合并模式的单元测试（无需视频/GPU）。"""

from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor

# GPU 门控方法在 _gpu_pipeline mixin 模块；模拟可用性需 patch 该模块的
# nvdec_available / tensorrt_available（split 后不再位于 extractor 命名空间）。
from video_ocr_engine.gpu import device as _gpu


def _make(**kwargs):
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **kwargs)


@pytest.fixture
def gpu_ok(monkeypatch):
    """模拟 NVDEC + TensorRT + cuda-python 可用（默认规则要求三者；
    门控 _cuda_python_available 在无 GPU runner 上恒 False，故补 patch）。"""
    monkeypatch.setattr(_gpu, "nvdec_available", lambda p: True)
    monkeypatch.setattr(_gpu, "tensorrt_available", lambda: True)
    monkeypatch.setattr(_gpu, "_cuda_python_available", lambda: True)


# 门控判据矩阵：同一谓词 _gpu_pipeline_enabled() 在 (patch, env, 构造 kwargs)
# 组合下的期望值。单列成表便于逐行读"什么组合该启用 GPU 管线"。
_GATE_CASES = [
    ("default_gray", {}, {}, {"rep_crop_format": "gray"}, True),
    ("opt_out_env", {}, {"GPU_PIPELINE": "0"}, {"rep_crop_format": "gray"}, False),
    ("requires_trt", {"tensorrt_available": lambda: False}, {}, {"rep_crop_format": "gray"}, False),
    (
        "forced_without_trt",
        {"tensorrt_available": lambda: False},
        {"GPU_PIPELINE": "1"},
        {"rep_crop_format": "gray", "ocr_backend": "cpu"},
        True,
    ),
    ("ocr_cpu_host_by_default", {}, {}, {"rep_crop_format": "gray", "ocr_backend": "cpu"}, False),
    # CPU 解码分支跳过 NVDEC 探测：nvdec_available 为假也应启用
    (
        "cpu_decode_ignores_nvdec_probe",
        {"nvdec_available": lambda p: False},
        {},
        {"rep_crop_format": "gray", "decode_backend": "cpu"},
        True,
    ),
    ("hybrid_enabled", {}, {}, {"rep_crop_format": "gray", "decode_backend": "hybrid"}, True),
    (
        "nvdec_unavailable",
        {"nvdec_available": lambda p: False},
        {},
        {"rep_crop_format": "gray"},
        False,
    ),
    ("contrast_supported", {}, {"TEXT_SEP_MERGE": "contrast"}, {"rep_crop_format": "gray"}, True),
    ("force_aspect_supported", {}, {}, {"rep_crop_format": "gray", "force_aspect": 1.5}, True),
]


@pytest.mark.parametrize(
    "patches,env,kwargs,expected",
    [c[1:] for c in _GATE_CASES],
    ids=[c[0] for c in _GATE_CASES],
)
def test_gpu_pipeline_gate(gpu_ok, monkeypatch, patches, env, kwargs, expected):
    for name, fn in patches.items():
        monkeypatch.setattr(_gpu, name, fn)
    for name, val in env.items():
        monkeypatch.setenv(name, val)
    assert _make(**kwargs)._gpu_pipeline_enabled() is expected


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


def test_content_range_to_crop_margin_math():
    # 宿主 _crop_to_content 与 GPU 直通共用的余量数学：
    # m = max(1, round(w * margin% / 100))、lo = max(0, first-m)、
    # hi = min(w, last+1+m)、满宽 → None（不裁）。
    #
    # 期望值**按 config 当前余量算出，不硬编码数字** —— 余量默认值是可调的，
    # 写死会让本测试在调参时假失败。
    ex = _make(rep_crop_format="gray")
    w = 100
    m = max(1, int(round(w * ex._ocr_autocrop_margin_pct / 100.0)))

    assert ex._content_range_to_crop(0, w - 1, w) is None  # 满宽
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


def test_autocrop_min_gain(monkeypatch):
    # 最小收益门槛：裁掉比例低于门槛就整段不裁。
    # 这条比"加大余量"更根本 —— 余量是全局的，为规避紧凑 ROI 的误裁而
    # 调大会连宽 ROI 的收益一起削掉；门槛则是逐段自适应。
    monkeypatch.setenv("OCR_ROI_AUTOCROP_MARGIN", "10")
    monkeypatch.setenv("OCR_ROI_AUTOCROP_MIN_GAIN", "10")
    ex = _make(rep_crop_format="gray")
    assert ex._ocr_autocrop_min_gain == 0.10
    w = 100
    m = 10  # = round(100 * 10%)，由上面的 env 固定
    # 收益足够（裁掉 20%）→ 裁
    rng = ex._content_range_to_crop(2 * m, w - 1 - 2 * m, w)
    assert rng is not None
    assert (w - rng[1]) / w >= 0.10
    # 收益不足（内容几乎满宽，裁掉 3%）→ 不裁
    assert ex._content_range_to_crop(12, 88, w) is None
    # 门槛为 0 时退化成"只要不满宽就裁"
    monkeypatch.setenv("OCR_ROI_AUTOCROP_MIN_GAIN", "0")
    ex2 = _make(rep_crop_format="gray")
    assert ex2._content_range_to_crop(12, 88, w) is not None


def test_merge_effective_mode_resolution(monkeypatch):
    """env 在**构造期**解析冻结；构造后改 env 不再生效。"""
    ex = _make()
    assert ex._merge_effective_mode() == "binary"
    monkeypatch.setenv("TEXT_SEP_MERGE", "off")
    assert ex._merge_effective_mode() == "binary"  # 冻结
    monkeypatch.setenv("TEXT_SEP_MERGE", "contrast")
    assert _make()._merge_effective_mode() == "binary"  # 构造前生效:归一
    monkeypatch.setenv("TEXT_SEP_MERGE", "2")
    assert _make()._merge_effective_mode() == "binary"
    monkeypatch.setenv("TEXT_SEP_MERGE", "off")
    assert _make()._merge_effective_mode() == ""
    monkeypatch.delenv("TEXT_SEP_MERGE")
    # 显式构造参数 merge_text_sep
    ex2 = _make(merge_text_sep="")
    assert ex2._merge_effective_mode() == ""
