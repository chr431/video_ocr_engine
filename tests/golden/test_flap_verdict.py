"""金标翻动判定的负测（R-golden 的"能失败"面）。

判据来自 F-12：一次性差异（复跑一致）是运行间翻动，放行并计数；差异
可复现才是真回归。三种输入必须分别落到"一致 / 翻动 / 真回归"。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tests" / "golden"))

from record import _verdict  # noqa: E402


def test_verdict_clean() -> None:
    """首跑无差异 → 一致（不红、不计翻动）。"""
    assert _verdict([], []) == ([], False)


def test_verdict_flap() -> None:
    """坏例边界：首跑差而复跑一致 → 翻动（放行但计数；不是真回归）。"""
    diffs, flap = _verdict(["ocr.segments"], [])
    assert diffs == ["ocr.segments"] and flap is True


def test_verdict_real_regression() -> None:
    """坏例：两轮都差 → 真回归（必红——这是判据要抓的形态）。"""
    diffs, flap = _verdict(["ocr.segments"], ["ocr.segments"])
    assert diffs == ["ocr.segments"] and flap is False
