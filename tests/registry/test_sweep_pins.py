"""sweep 判据单测——钉版前缀/墓碑词边界/注释坐标模式（本文件是扫描器自证材料，
列入 _COMMENT_SCAN_EXEMPT）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from conclusions_sweep import (  # noqa: E402
    _PIN,
    _comment_hits,
    _hygiene_rows,
    _tombstone_pattern,
)


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
    """负测：词边界防子串误报——相邻编号与前缀串不得命中（真实事故：解码
    契约用例名 DC-06 曾被误判为死者编号的引用）。用活结论 id 做测例，
    避免测试串引用死者、在墓碑清单里常驻噪音。"""
    pat = _tombstone_pattern({"C-53", "C-08"})
    assert pat.findall("DC-530") == []
    assert pat.findall("C-5300") == []
    assert pat.findall("见 C-53 与 C-08") == ["C-53", "C-08"]


def test_comment_coordinate_patterns_hit() -> None:
    """正测：旧编号/裁决铁律/日期/出树文档四类坐标必须命中。"""
    kinds = {k for k, _ in _comment_hits("见 S9-3 与 R4；裁决 12 与 铁律 9；Phase D；第三轮")}
    assert "旧编号" in kinds
    assert _comment_hits("复核于 2026-09-19")[0][0] == "日期坐标"
    assert _comment_hits("证据见 docs/log/xxx.md")[0][0] == "出树文档"


def test_comment_coordinate_patterns_spared() -> None:
    """负测：活标识与档案锚点不得误报。"""
    assert _comment_hits("见 PI-10、S0-C1、fork@C-77、decord@0.9.8、R-perf-quiet") == []
    assert _comment_hits("迁移表见 git 档案 archive/pre-rework:docs/MIGRATION.md") == []


def test_worktree_hygiene_age_filter() -> None:
    """卫生清单只列超龄物（忽略≠退出），新物不列。"""
    now = 1_000_000_000.0
    day = 86400.0
    cands = [
        ("tools/_old.log", now - 30 * day, 2_000_000),
        ("tools/_fresh.log", now - 1 * day, 10),
    ]
    rows = _hygiene_rows(cands, now, days=14)
    assert len(rows) == 1 and rows[0].startswith("tools/_old.log（2.0 MB，30 天）")
