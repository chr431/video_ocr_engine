"""知识库预算与一致性门禁（v2 §14.2 四预算之 knowledge 两项 + 渲染一致性）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_conclusions_budget():
    size = (ROOT / "knowledge" / "conclusions.yaml").stat().st_size
    assert size <= 8 * 1024, "conclusions.yaml %d B 超 8 KB 预算" % size


def test_knobs_budget():
    size = (ROOT / "knowledge" / "knobs.yaml").stat().st_size
    assert size <= 24 * 1024, "knobs.yaml %d B 超 24 KB 预算" % size


def test_render_check_passes():
    r = subprocess.run(
        [sys.executable, str(ROOT / "knowledge" / "render.py")],
        capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_agents_budget():
    size = (ROOT / "AGENTS.md").stat().st_size
    assert size <= 12 * 1024, "AGENTS.md %d B 超 12 KB 注入预算" % size
