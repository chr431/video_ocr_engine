"""S9-2 行为契约：D6（构造期冻结）与 Q5（显式参数 > env > 默认）。

v1 语义（README 曾承诺"构造之后再改 env 同样生效"、env 盖过构造参数）
已按 §10.2/Q5 裁决废除——迁移说明见 docs/MIGRATION.md。
"""
from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor


def _make(**kw):
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **kw)


# ── D6：构造期冻结 ──────────────────────────────────────────────
def test_env_frozen_at_construction(monkeypatch):
    ex = _make()
    assert ex._ocr_autocrop is True          # 默认
    monkeypatch.setenv("OCR_ROI_AUTOCROP", "0")
    assert ex._ocr_autocrop is True          # 构造后改 env：不再生效
    monkeypatch.setenv("OCR_REORDER_WINDOW", "8")
    assert ex._ocr_reorder_window == 64      # 冻结
    monkeypatch.setenv("GPU_PIPELINE", "1")
    assert ex._rc.pipeline_gpu is None       # 三态仍为"规则"
    # 构造前设 env → 新实例参与解析
    fresh = _make()
    assert fresh._ocr_reorder_window == 8
    assert fresh._rc.pipeline_gpu is True
    monkeypatch.setenv("OCR_REORDER_WINDOW", "16")
    assert _make()._ocr_reorder_window == 16


# ── Q5：显式参数 > env > 默认（v1 相反）────────────────────────
def test_q5_fill_width_param_beats_env(monkeypatch):
    monkeypatch.setenv("OCR_PAD_SMALL", "320")
    ex = _make(fill_width=224)
    assert ex._fill_width == 224             # 显式参数锁定（v1 会变 320）
    assert ex._pad_floor_env == 0


def test_q5_env_applies_when_param_absent(monkeypatch):
    monkeypatch.setenv("OCR_PAD_SMALL", "320")
    ex = _make()                             # fill_width 缺省
    assert ex._fill_width == 224             # 参数缺省 → 默认 224
    assert ex._pad_floor_env == 320          # env 抬升下限（在引擎内生效）


def test_q5_escape_hatch_restores_v1(monkeypatch):
    monkeypatch.setenv("OCR_PAD_SMALL", "320")
    monkeypatch.setenv("VOE_ENV_WINS", "1")
    with pytest.warns(DeprecationWarning):
        ex = _make(fill_width=224)
    assert ex._fill_width == 224
    assert ex._pad_floor_env == 320          # v1：env 恒先于参数


def test_q5_merge_text_sep_param_beats_env(monkeypatch):
    monkeypatch.setenv("TEXT_SEP_MERGE", "off")
    ex = _make(merge_text_sep="binary")
    assert ex._merge_effective_mode() == "binary"   # 参数锁定
    assert _make()._merge_effective_mode() == ""    # 缺省 → env 生效


# ── 注入口贯通（rc → SessionSpec，纯构建无线程）────────────────
def test_session_spec_carries_frozen_rc(monkeypatch):
    monkeypatch.setenv("OCR_GAMMA", "1.8")
    monkeypatch.setenv("OCR_BATCH", "24")
    ex = _make()
    spec = ex._build_session_spec()
    assert spec.gamma == 1.8
    assert spec.ocr_batch == 24
    # S6-f：reorder_window 是**生效值**——本 ROI（101×51）的宽高比 1.98 够不到
    # pad 下限比 224/48=4.67 → 按宽分组不可能改变 pad 宽 → 收敛到 1
    assert spec.reorder_window == 1
    assert ex._ocr_reorder_window == 64           # 旋钮本身仍解析为 64


def test_reorder_window_kept_for_wide_roi(monkeypatch):
    """判据是"ROI 上界宽高比"：宽 ROI（内容能超过下限）保留原窗口。"""
    from video_ocr_engine.pipeline.ocr_stage import effective_reorder_window
    # 407×25 字幕 ROI：16.3 > 4.67 → 分组有效，保留 64
    assert effective_reorder_window(407, 25, 224, 0.0, 64) == 64
    # force_aspect>0：输入被压到固定宽高比 → 分组无意义
    assert effective_reorder_window(407, 25, 224, 1.5, 64) == 1
