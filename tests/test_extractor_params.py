"""FieldExtractor 参数校验与内存选项测试（无需解码器/视频）。"""
from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor


def _make(**kwargs):
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **kwargs)


def test_roi_validation():
    with pytest.raises(ValueError, match="四元组"):
        FieldExtractor("x.mp4", (0, 0, 100))
    with pytest.raises(ValueError, match="不能为负"):
        FieldExtractor("x.mp4", (-1, 0, 100, 50))
    with pytest.raises(ValueError, match="x2 > x1"):
        FieldExtractor("x.mp4", (0, 0, 0, 50))
    with pytest.raises(ValueError, match="y2 > y1"):
        FieldExtractor("x.mp4", (0, 0, 100, 0))


def test_frame_range_validation():
    with pytest.raises(ValueError, match="frame_start"):
        _make(frame_start=-1)
    with pytest.raises(ValueError, match="frame_end"):
        _make(frame_start=10, frame_end=10)
    with pytest.raises(ValueError, match="frame_end"):
        _make(frame_start=10, frame_end=5)
    # 0 作为“到末尾”的兼容写法不应报错
    _make(frame_start=10, frame_end=0)


def test_extract_sets_frames_and_defaults_keep(monkeypatch):
    ex = _make()

    def fake_run(self):
        self.crops = {1: "crop1", 2: "crop2"}
        return ([0, 1, 2], [[0, 1], [2]], ["a", "b"], [0.9, 0.8], [1, 2])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined", fake_run)
    result = ex.extract()
    assert ex.frames == [0, 1, 2]
    assert result.frames == [0, 1, 2]
    assert result.segments[0].frames == (0, 1)
    assert result.segments[1].frames == (2,)
    assert result.segments[0].rep_crop == "crop1"
    assert result.segments[1].rep_crop == "crop2"


def test_extract_keep_crops_false(monkeypatch):
    ex = _make(keep_crops=False)

    def fake_run(self):
        self.crops = {}
        return ([0, 1, 2], [[0, 1], [2]], ["a", "b"], [0.9, 0.8], [1, 2])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined", fake_run)
    result = ex.extract()
    assert all(seg.rep_crop is None for seg in result.segments)


def test_extract_keep_frames_false(monkeypatch):
    ex = _make(keep_frames=False)

    def fake_run(self):
        self.crops = {1: "crop1", 2: "crop2"}
        return ([0, 1, 2], [[0, 1], [2]], ["a", "b"], [0.9, 0.8], [1, 2])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined", fake_run)
    result = ex.extract()
    assert result.frames == []
    assert all(seg.frames == () for seg in result.segments)
    # rep_crop 默认仍保留
    assert result.segments[0].rep_crop == "crop1"
