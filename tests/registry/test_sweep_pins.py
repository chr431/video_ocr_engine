"""sweep 判据单测——显式前缀才算钉版，裸数字是注记；墓碑匹配须带词边界。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from conclusions_sweep import _PIN, _tombstone_pattern  # noqa: E402


def test_prefixed_pins_captured() -> None:
    blob = "4060；fork 0.9.8；引擎 0.26.1；wheel 0.8.2；engine v1.2.3"
    got = [(m.group(1), m.group(2)) for m in _PIN.finditer(blob)]
    assert ("fork", "0.9.8") in got
    assert ("引擎", "0.26.1") in got
    assert ("wheel", "0.8.2") in got
    assert ("engine", "1.2.3") in got


def test_bare_versions_are_not_pins() -> None:
    """负测：注记里的时代数字/日期版本不是钉版（上下文词启发式的替代伤）。"""
    blob = "可见性 2.45× 为 0.8.3 时代口径；复核于 2026-10-12；0.9.4-0.9.8 历代复验"
    assert list(_PIN.finditer(blob)) == []


def test_tombstone_pattern_word_boundary() -> None:
    """负测：`DC-06` 是解码契约用例名，不得被判为死者 `C-06` 的引用（子串误报）。"""
    pat = _tombstone_pattern({"C-06", "C-24"})
    assert pat.findall("DC-06") == []
    assert pat.findall("见 C-06 与 C-24") == ["C-06", "C-24"]
