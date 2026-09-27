"""hybrid OCR（双车道 TRT+OpenVINO，2026-09-20）的回归钉子。

覆盖：
- ocr_backend='hybrid' 合法（构造期不炸）且映射 engine_type='hybrid'
- acquire_engines('hybrid') 取双擎（tensorrt + onnxruntime 各一，池 key 同参）
（pool.run 的废弃钉子随 pool.py 于 0.16.0 删除一并退场）
性能/正确性实测（v0 = OCR-bound 负载 +51% 回归、文本 3/26461 差异）由
tools/_probe_hybrid_ocr.py 承载，不入单测。
"""
from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor


def _make(**kwargs):
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **kwargs)


def test_ocr_backend_hybrid_accepted():
    """'hybrid' 合法且 engine_type 映射正确（auto/tensorrt/cpu 不回归）。"""
    ex = _make(ocr_backend="hybrid")
    assert ex._ocr_engine_type() == 'hybrid'
    assert _make()._ocr_engine_type() == 'tensorrt'
    assert _make(ocr_backend="auto")._ocr_engine_type() == 'tensorrt'
    assert _make(ocr_backend="cpu")._ocr_engine_type() == 'onnxruntime'
    with pytest.raises(ValueError, match="ocr_backend"):
        _make(ocr_backend="bogus")


def test_hybrid_ocr_on_gpu_counts_as_gpu():
    """hybrid OCR 含 TRT 车道 → 解码线程预算按 GPU-OCR 档（与 tensorrt 同）。"""
    assert _make(ocr_backend="hybrid")._ocr_on_gpu() is True
    assert _make(ocr_backend="cpu")._ocr_on_gpu() is False


def test_acquire_engines_hybrid_takes_two(monkeypatch):
    """'hybrid' → tensorrt + onnxruntime 各一引擎（同池 key 参数）。"""
    import video_ocr_engine.ocr.native as _native   # monkeypatch 目标先导入
    from video_ocr_engine.pipeline import ocr_stage

    calls = []

    class _FakeEng:
        def __init__(self, t):
            self.backend_name = t

    def fake_acquire(variant, engine_type, **kw):
        calls.append((variant, engine_type, kw))
        return _FakeEng(engine_type)

    monkeypatch.setattr(_native, "acquire_ocr_engine", fake_acquire)

    class _Spec:
        model = "v6_small"
        fill_width = None
        pad_floor_env = None
        gamma = None
        gpu_ctc = None
        ocr_instances = None
        num_threads_fn = staticmethod(lambda: 8)
        engine_type_fn = staticmethod(lambda: "hybrid")

    engines, etype = ocr_stage.acquire_engines(_Spec())
    assert etype == 'hybrid'
    assert [e.backend_name for e in engines] == ['tensorrt', 'onnxruntime']
    assert len(calls) == 2
    # 两擎同池 key 参数（fill_width/threads/gamma 一致）——除 engine_type 外
    assert calls[0][2] == calls[1][2]

