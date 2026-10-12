"""version_advice 的判据负测——档位是计算结果（R-version-advice 零裁量）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from version_advice import _diff_keys, api_surface, verdict  # noqa: E402


def test_verdict_table() -> None:
    assert verdict(api_diff=True, golden_diff=False) == "minor"
    assert verdict(api_diff=False, golden_diff=True) == "minor"
    assert verdict(api_diff=True, golden_diff=True) == "minor"
    assert verdict(api_diff=False, golden_diff=False) == "patch"


def test_diff_keys_semantics() -> None:
    cur = {"extract": {"kind": "func", "signature": "(a, b=1)"}, "new": {"kind": "other"}}
    prev = {"extract": {"kind": "func", "signature": "(a)"}, "gone": {"kind": "other"}}
    assert _diff_keys(cur, prev) == ["extract", "gone", "new"]


def test_api_surface_deterministic_and_public() -> None:
    a, b = api_surface(), api_surface()
    assert a == b, "同一解释器两次导出必须一致（对照基线的前提）"
    assert a["version"] and a["symbols"]
    assert all(not k.startswith("_") for k in a["symbols"])
    assert "FieldExtractor" in a["symbols"]  # 公开主入口类必须在面上
