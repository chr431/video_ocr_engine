"""health --check 的负测：KPI 对账必须能红。

`check()` 比对「落盘 kpi.yaml」与「重算实态」的稳定指标；两者不一致即非零
退出。这里用最小 kpi 结构 + 替换 compute() 分别构造一致/漂移两种输入。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import health  # noqa: E402

_NAME = "注入核字节（≤6144=宪章不变量）"


def _stored(tmp_path: Path, value: int) -> Path:
    p = tmp_path / "kpi.yaml"
    p.write_text(
        "metrics:\n- name: %s\n  value: %d\n  target: '≤6144'\n" % (_NAME, value),
        encoding="utf-8",
    )
    return p


def _fresh(value: int) -> dict:
    return {"metrics": [{"name": _NAME, "value": value, "target": "≤6144"}]}


def test_check_detects_drift(tmp_path, monkeypatch) -> None:
    """坏例：落盘值与实态不符 → 非零退出（KPI 对账能红）。"""
    monkeypatch.setattr(health, "KPI", _stored(tmp_path, 1))
    monkeypatch.setattr(health, "compute", lambda: _fresh(2))
    assert health.check() == 1


def test_check_passes_when_consistent(tmp_path, monkeypatch) -> None:
    """正例：一致 → 0。"""
    monkeypatch.setattr(health, "KPI", _stored(tmp_path, 2))
    monkeypatch.setattr(health, "compute", lambda: _fresh(2))
    assert health.check() == 0
