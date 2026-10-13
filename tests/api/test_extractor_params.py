"""FieldExtractor 参数校验 / 构造期冻结 / 内存与合并选项（无需解码器/视频）。"""

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
        self._crops = {1: "crop1", 2: "crop2"}
        return RunOutcome([0, 1, 2], [[0, 1], [2]], ["a", "b"], [0.9, 0.8], [1, 2])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined", fake_run)
    result = ex.extract()
    # ex.frames 兼容读面已删：帧号只看 result.frames
    assert result.frames == [0, 1, 2]
    assert result.segments[0].frames == (0, 1)
    assert result.segments[1].frames == (2,)
    assert result.segments[0].rep_crop == "crop1"
    assert result.segments[1].rep_crop == "crop2"


def test_extract_keep_crops_false(monkeypatch):
    ex = _make(keep_crops=False)

    def fake_run(self):
        self._crops = {}
        return RunOutcome([0, 1, 2], [[0, 1], [2]], ["a", "b"], [0.9, 0.8], [1, 2])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined", fake_run)
    result = ex.extract()
    assert all(seg.rep_crop is None for seg in result.segments)


def test_extract_keep_frames_false(monkeypatch):
    ex = _make(keep_frames=False)

    def fake_run(self):
        self._crops = {1: "crop1", 2: "crop2"}
        return RunOutcome([0, 1, 2], [[0, 1], [2]], ["a", "b"], [0.9, 0.8], [1, 2])

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
    ex2 = _make(merge_similar=False, merge_similar_threshold=5.0, merge_text_sep="off")
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
    # _make 的 ROI=(0,0,100,50) → 101×51，面积 5151，max_changed=51
    ex = _make(merge_similar=True)
    a = np.zeros((50, 100), dtype=np.uint8)
    b_small = a.copy()
    b_small[:10, :10] = 200  # 100 像素显著变化 > 51 -> 不相似
    assert not ex._segments_similar(a, b_small)
    # 稀疏小变化（不构成 3×3 稠密簇）<= 51 -> 相似：像素数上限本身仍生效
    b_sparse = a.copy()
    for i in range(0, 50, 2):
        b_sparse[i, 0] = 200  # 25 像素显著变化，1px 宽 → win3 = 2
    assert ex._segments_similar(a, b_sparse)
    # 平均差超阈值也判不相似
    b_mean = a.copy()
    b_mean[:, :] = 30  # 整体抬升，平均差 30 > 3
    assert not ex._segments_similar(a, b_mean)


def test_segments_similar_dense_gate(monkeypatch):
    """稠密簇门（segment.merge_dense_gate）：紧凑 3×3 簇恒判不相似。

    动机（实测）：末位单数字变化在紧凑 ROI 上只动 12-14px，低于
    max_changed 上限（面积 1%）而被合并吞掉整段短状态。断段判据用
    win3 ≥ C 定义「内容变了」，合并判据必须自洽。
    """
    a = np.zeros((50, 100), dtype=np.uint8)
    b_dense = a.copy()
    b_dense[:4, :4] = 200  # 16px ≤ max_changed(51)，但 3×3 窗口和=9
    monkeypatch.delenv("SEG_MERGE_DENSE_GATE", raising=False)
    assert not _make(merge_similar=True)._segments_similar(a, b_dense)
    # gate=0 关闭 → 回到纯两阈值判定（16 ≤ 51 → 相似）
    monkeypatch.setenv("SEG_MERGE_DENSE_GATE", "0")
    assert _make(merge_similar=True)._segments_similar(a, b_dense)


# ── 构造期冻结与"显式参数 > env > 默认"优先级（v1 env-wins 不得复活）──


def test_env_frozen_at_construction(monkeypatch):
    ex = _make()
    assert ex._ocr_autocrop is True  # 默认
    monkeypatch.setenv("OCR_ROI_AUTOCROP", "0")
    assert ex._ocr_autocrop is True  # 构造后改 env：不再生效
    monkeypatch.setenv("OCR_REORDER_WINDOW", "8")
    assert ex._ocr_reorder_window == 64  # 冻结
    monkeypatch.setenv("GPU_PIPELINE", "1")
    assert ex._rc.pipeline_gpu is None  # 三态仍为"规则"
    # 构造前设 env → 新实例参与解析
    fresh = _make()
    assert fresh._ocr_reorder_window == 8
    assert fresh._rc.pipeline_gpu is True
    monkeypatch.setenv("OCR_REORDER_WINDOW", "16")
    assert _make()._ocr_reorder_window == 16


def test_fill_width_param_beats_env(monkeypatch):
    monkeypatch.setenv("OCR_PAD_SMALL", "320")
    ex = _make(fill_width=224)
    assert ex._fill_width == 224  # 显式参数锁定（v1 会变 320）
    assert ex._pad_floor_env == 0


def test_fill_width_env_applies_when_param_absent(monkeypatch):
    monkeypatch.setenv("OCR_PAD_SMALL", "320")
    ex = _make()  # fill_width 缺省
    assert ex._fill_width == 224  # 参数缺省 → 默认 224
    assert ex._pad_floor_env == 320  # env 抬升下限（在引擎内生效）


def test_merge_text_sep_param_beats_env(monkeypatch):
    monkeypatch.setenv("TEXT_SEP_MERGE", "off")
    ex = _make(merge_text_sep="binary")
    assert ex._merge_effective_mode() == "binary"  # 参数锁定
    assert _make()._merge_effective_mode() == ""  # 缺省 → env 生效


def test_session_spec_carries_frozen_rc(monkeypatch):
    """注入口贯通（rc → SessionSpec，纯构建无线程）。"""
    monkeypatch.setenv("OCR_GAMMA", "1.8")
    monkeypatch.setenv("OCR_BATCH", "24")
    ex = _make()
    spec = ex._build_session_spec()
    assert spec.gamma == 1.8
    assert spec.ocr_batch == 24
    # reorder_window 是**生效值**——本 ROI（101×51）的宽高比 1.98 够不到
    # pad 下限比 224/48=4.67 → 按宽分组不可能改变 pad 宽 → 收敛到 1
    assert spec.reorder_window == 1
    assert ex._ocr_reorder_window == 64  # 旋钮本身仍解析为 64


def test_reorder_window_kept_for_wide_roi():
    """判据是"ROI 上界宽高比"：宽 ROI（内容能超过下限）保留原窗口。"""
    from video_ocr_engine.pipeline.ocr_stage import effective_reorder_window

    # 407×25 字幕 ROI：16.3 > 4.67 → 分组有效，保留 64
    assert effective_reorder_window(407, 25, 224, 0.0, 64) == 64
    # force_aspect>0：输入被压到固定宽高比 → 分组无意义
    assert effective_reorder_window(407, 25, 224, 1.5, 64) == 1


# ── 未知后端构造期报错 + 二次 extract 重置运行态 ──────────────────


@pytest.mark.parametrize(
    "kw,match",
    [({"decode_backend": "cuda"}, "decode_backend"), ({"ocr_backend": "tesseract"}, "ocr_backend")],
)
def test_unknown_backend_raises(kw, match):
    """未知 decode_backend / ocr_backend 构造期 ValueError（v1 静默兜底已废除）。"""
    with pytest.raises(ValueError, match=match):
        _make(**kw)


def test_valid_backends_accepted():
    for dec in ("auto", "cpu", "nvdec", "hybrid"):
        _make(decode_backend=dec)
    for ocr in ("auto", "cpu", "tensorrt"):
        _make(ocr_backend=ocr)


def test_second_extract_resets_run_state(monkeypatch):
    """同实例第二次 extract 不再带上一次的降级原因/计时/剖面。"""
    ex = _make()
    # 污染运行态（模拟上一次 extract 的残留；profile 读面已为只读
    # property——直接污染 spine 内字典）
    ex._degraded = ["上次的原因"]
    ex._timing = {"decode": 9.9}
    ex._spine.profile["ocr"] = {"x": 1}
    ex._backend = "decord/CPU"
    ex._ocr_backend_used = "tensorrt"
    ex._bin_thresh = 77
    ex._n_segments = 5
    # _fps 不在重置范围（同实例同视频的文档化缓存）

    monkeypatch.setattr(FieldExtractor, "_run_pipelined", lambda self: RunOutcome())
    r = ex.extract()
    assert ex._degraded == []
    assert ex._timing == {}
    assert ex.profile == {}
    assert ex._backend == ""
    assert ex._ocr_backend_used == ""
    assert ex._bin_thresh == 0
    assert r.meta["degraded_reason"] is None
    assert len(r.segments) == 0
