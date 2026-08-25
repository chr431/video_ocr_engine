"""FieldExtractor 参数校验与内存选项测试（无需解码器/视频）。"""
from __future__ import annotations

import numpy as np
import pytest

from video_ocr_engine import FieldExtractor
from video_ocr_engine.extractor import _gray_mean_abs_diff


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


def test_merge_similar_default_on_and_custom():
    ex = _make()
    assert ex._merge_similar is True
    assert ex._merge_similar_threshold == 3.0
    assert ex._merge_text_sep == "binary"
    ex2 = _make(merge_similar=False, merge_similar_threshold=5.0,
                merge_text_sep="contrast")
    assert ex2._merge_similar is False
    assert ex2._merge_similar_threshold == 5.0
    assert ex2._merge_text_sep == "contrast"


def test_rep_crop_format_resolution_and_alias():
    # 新默认 = "yuv"（旧 gray_output=False 曾是 RGB，内部已无 RGB 链路）
    assert _make()._rep_crop_format == "yuv"
    assert _make()._yuv_output is True
    # yuv_output=True / gray_output=True 为 deprecated 别名
    assert _make(yuv_output=True)._rep_crop_format == "yuv"
    assert _make(gray_output=True)._rep_crop_format == "gray"
    assert _make(gray_output=True)._yuv_output is False
    # rep_crop_format 显式优先
    assert _make(rep_crop_format="gray")._rep_crop_format == "gray"
    assert _make(rep_crop_format="yuv")._rep_crop_format == "yuv"
    # keep_crops=False 时无 UV 需求 → 内部退化为 gray 输出
    assert _make(rep_crop_format="yuv", keep_crops=False)._yuv_output is False
    with pytest.raises(ValueError, match="rep_crop_format"):
        _make(rep_crop_format="rgb")


def test_gray_mean_abs_diff():
    a = np.zeros((4, 5), dtype=np.uint8)
    b = a.copy()
    assert _gray_mean_abs_diff(a, b) == 0.0
    b[0, 0] = 10
    assert _gray_mean_abs_diff(a, b) == pytest.approx(10.0 / 20)
    assert _gray_mean_abs_diff(a, None) == float("inf")
    assert _gray_mean_abs_diff(a, np.zeros((3, 5), dtype=np.uint8)) == float("inf")


def test_segments_similar_requires_small_changed_area():
    # _make 的 ROI=(0,0,100,50)，面积 5050，max_changed=50
    ex = _make(merge_similar=True)
    a = np.zeros((50, 100), dtype=np.uint8)
    b_small = a.copy()
    b_small[:10, :10] = 200          # 100 像素显著变化 > 50 -> 不相似
    assert not ex._segments_similar(a, b_small)
    b_tiny = a.copy()
    b_tiny[:5, :5] = 200             # 25 像素显著变化 <= 50 -> 相似
    assert ex._segments_similar(a, b_tiny)
    # 平均差超阈值也判不相似
    b_mean = a.copy()
    b_mean[:, :] = 30                # 整体抬升，平均差 30 > 3
    assert not ex._segments_similar(a, b_mean)
