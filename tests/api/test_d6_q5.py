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
    assert spec.reorder_window == 64
