"""S3 行为缺陷修复测试（Q8 裁决的 B1/B3，v2 §13.3）。

B1：同实例第二次 extract 不再带上一次的降级原因/计时/剖面。
B3：未知 decode_backend / ocr_backend 构造期 ValueError（v1 静默兜底）。
两个修复都是 §10.2 声明的**有意行为变更**。
"""
from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor


def _make(**kw):
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **kw)


def test_b3_unknown_decode_backend_raises():
    with pytest.raises(ValueError, match="decode_backend"):
        _make(decode_backend="cuda")


def test_b3_unknown_ocr_backend_raises():
    with pytest.raises(ValueError, match="ocr_backend"):
        _make(ocr_backend="tesseract")


def test_b3_valid_backends_accepted():
    for dec in ("auto", "cpu", "nvdec", "hybrid"):
        _make(decode_backend=dec)
    for ocr in ("auto", "cpu", "tensorrt"):
        _make(ocr_backend=ocr)


def test_b1_second_extract_resets_run_state(monkeypatch):
    ex = _make()
    # 污染运行态（模拟上一次 extract 的残留）
    ex._degraded = ["上次的原因"]
    ex.timing = {"decode": 9.9}
    ex.profile = {"ocr": {"x": 1}}
    ex._backend = "decord/CPU"
    ex._ocr_backend_used = "tensorrt"
    ex._bin_thresh = 77
    ex._n_segments = 5
    # _fps 不在重置范围（B2：同实例同视频的文档化缓存）

    monkeypatch.setattr(
        FieldExtractor, "_run_pipelined",
        lambda self: ([], [], [], [], []))
    r = ex.extract()
    assert ex._degraded == []
    assert ex.timing == {}
    assert ex.profile == {}
    assert ex._backend == ""
    assert ex._ocr_backend_used == ""
    assert ex._bin_thresh == 0
    assert r.meta["degraded_reason"] is None
    assert len(r.segments) == 0
