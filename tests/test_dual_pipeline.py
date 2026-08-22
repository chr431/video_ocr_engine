"""单实例双完整流水线并行的单元测试（无需视频/GPU）。

只覆盖构造/参数/后端组合/分发逻辑；真实解码与 OCR 由集成冒烟负责。
"""
from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor


def _make(**kwargs):
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **kwargs)


def test_dual_pipeline_default_off():
    ex = _make()
    assert ex._dual_pipeline is False
    assert ex._dual_pipeline_chunks == 0
    assert ex._dual_backends is None


def test_dual_pipeline_env_enabled(monkeypatch):
    monkeypatch.setenv("RVTOL_DUAL_PIPELINE", "1")
    ex = _make()
    assert ex._dual_pipeline is True


def test_dual_pipeline_explicit_override(monkeypatch):
    monkeypatch.setenv("RVTOL_DUAL_PIPELINE", "1")
    ex = _make(dual_pipeline=False)
    assert ex._dual_pipeline is False
    ex2 = _make(dual_pipeline=True, dual_pipeline_chunks=6)
    assert ex2._dual_pipeline is True
    assert ex2._dual_pipeline_chunks == 6


def test_dual_backend_pairs_default_opposite():
    ex = _make(decode_backend="auto", ocr_backend="auto")
    assert ex._dual_backend_pairs() == [("auto", "auto"), ("cpu", "cpu")]
    ex2 = _make(decode_backend="cpu", ocr_backend="cpu")
    assert ex2._dual_backend_pairs() == [("cpu", "cpu"), ("auto", "auto")]
    ex3 = _make(decode_backend="nvdec", ocr_backend="tensorrt")
    assert ex3._dual_backend_pairs() == [("nvdec", "tensorrt"), ("cpu", "cpu")]


def test_dual_backends_custom_and_duplicate_single():
    ex = _make(dual_backends=[("cpu", "auto"), ("auto", "cpu")])
    assert ex._dual_backend_pairs() == [("cpu", "auto"), ("auto", "cpu")]
    ex2 = _make(dual_backends=[("cpu", "auto")])
    assert ex2._dual_backend_pairs() == [("cpu", "auto"), ("cpu", "auto")]
    ex3 = _make(dual_backends=[("cpu", "auto")], decode_backend="auto")
    assert ex3._dual_backend_pairs() == [("cpu", "auto"), ("cpu", "auto")]


def test_extract_dispatches_to_parallel_when_enabled(monkeypatch):
    ex = _make(dual_pipeline=True)

    def fake_parallel(self):
        self.crops = {1: "crop1", 2: "crop2"}
        return ([0, 1, 2], [[0, 1], [2]], ["a", "b"], [0.9, 0.8], [1, 2])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined_parallel", fake_parallel)
    result = ex.extract()
    assert result.frames == [0, 1, 2]
    assert result.segments[0].frames == (0, 1)
    assert result.segments[0].rep_crop == "crop1"
    assert result.meta["ocr_backend"] == ""


def test_extract_does_not_dispatch_when_disabled(monkeypatch):
    ex = _make(dual_pipeline=False)

    def fake_parallel(self):
        raise AssertionError("不应进入并行路径")

    monkeypatch.setattr(FieldExtractor, "_run_pipelined_parallel", fake_parallel)

    def fake_run(self):
        self.crops = {}
        return ([0], [[0]], ["x"], [0.5], [0])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined", fake_run)
    result = ex.extract()
    assert result.segments[0].text == "x"
