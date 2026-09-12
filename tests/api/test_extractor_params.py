"""FieldExtractor 参数校验与内存选项测试（无需解码器/视频）。"""
from __future__ import annotations

import numpy as np
import pytest

from video_ocr_engine import FieldExtractor
from video_ocr_engine.pipeline import RunOutcome


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
        return RunOutcome([0, 1, 2], [[0, 1], [2]], ["a", "b"],
                          [0.9, 0.8], [1, 2])

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
        return RunOutcome([0, 1, 2], [[0, 1], [2]], ["a", "b"],
                          [0.9, 0.8], [1, 2])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined", fake_run)
    result = ex.extract()
    assert all(seg.rep_crop is None for seg in result.segments)


def test_extract_keep_frames_false(monkeypatch):
    ex = _make(keep_frames=False)

    def fake_run(self):
        self.crops = {1: "crop1", 2: "crop2"}
        return RunOutcome([0, 1, 2], [[0, 1], [2]], ["a", "b"],
                          [0.9, 0.8], [1, 2])

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
                merge_text_sep="off")
    assert ex2._merge_similar is False
    assert ex2._merge_similar_threshold == 5.0
    assert ex2._merge_text_sep == "off"


def test_rep_crop_format_resolution_and_alias():
    # 新默认 = "yuv"（旧 gray_output=False 曾是 RGB，内部已无 RGB 链路）
    assert _make()._rep_crop_format == "yuv"
    assert _make()._yuv_output is True
    # rep_crop_format="yuv" / rep_crop_format="gray" 为 deprecated 别名
    assert _make(rep_crop_format="yuv")._rep_crop_format == "yuv"
    assert _make(rep_crop_format="gray")._rep_crop_format == "gray"
    assert _make(rep_crop_format="gray")._yuv_output is False
    # rep_crop_format 显式优先
    assert _make(rep_crop_format="gray")._rep_crop_format == "gray"
    assert _make(rep_crop_format="yuv")._rep_crop_format == "yuv"
    # keep_crops=False 时无 UV 需求 → 内部退化为 gray 输出
    assert _make(rep_crop_format="yuv", keep_crops=False)._yuv_output is False
    with pytest.raises(ValueError, match="rep_crop_format"):
        _make(rep_crop_format="rgb")


def test_segments_similar_requires_small_changed_area():
    # _make 的 ROI=(0,0,100,50)，面积 5050，max_changed=50
    ex = _make(merge_similar=True)
    a = np.zeros((50, 100), dtype=np.uint8)
    b_small = a.copy()
    b_small[:10, :10] = 200          # 100 像素显著变化 > 50 -> 不相似
    assert not ex._segments_similar(a, b_small)
    # 稀疏小变化（不构成 3×3 稠密簇）<= 50 -> 相似：像素数上限本身仍生效
    b_sparse = a.copy()
    for i in range(0, 50, 2):
        b_sparse[i, 0] = 200         # 25 像素显著变化，1px 宽 → win3 = 3
    assert ex._segments_similar(a, b_sparse)
    # 平均差超阈值也判不相似
    b_mean = a.copy()
    b_mean[:, :] = 30                # 整体抬升，平均差 30 > 3
    assert not ex._segments_similar(a, b_mean)


def test_segments_similar_dense_gate(monkeypatch):
    """稠密簇门（segment.merge_dense_gate）：紧凑 3×3 簇恒判不相似。

    动机（实测）：末位单数字变化在紧凑 ROI 上只动 12-14px，低于
    max_changed 上限（面积 1%）而被合并吞掉整段短状态。断段判据用
    win3 ≥ C 定义「内容变了」，合并判据必须自洽。
    """
    a = np.zeros((50, 100), dtype=np.uint8)
    b_dense = a.copy()
    b_dense[:4, :4] = 200        # 16px ≤ max_changed(51)，但 3×3 窗口和=9
    monkeypatch.delenv("SEG_MERGE_DENSE_GATE", raising=False)
    assert not _make(merge_similar=True)._segments_similar(a, b_dense)
    # gate=0 关闭 → 回到纯两阈值判定（16 ≤ 51 → 相似）
    monkeypatch.setenv("SEG_MERGE_DENSE_GATE", "0")
    assert _make(merge_similar=True)._segments_similar(a, b_dense)
