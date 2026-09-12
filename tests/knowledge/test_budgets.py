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


def test_superseded_pointers_resolve():
    """`superseded →` 的指针必须落在现存结论 id 上。

    防「C-27 → ?」式悬空：本轮实测曾有一条 2026-09 起就挂着 `?` 的记录，
    因为 replaced_by 只在渲染时拼成一行、没有任何门禁看它是否解析得到。
    """
    import re
    text = (ROOT / "knowledge" / "conclusions.yaml").read_text(encoding="utf-8")
    ids: set[str] = set()
    sup: dict[str, str] = {}
    cur: str | None = None
    for line in text.split("\n"):
        m = re.match(r"- id: (\S+)", line)
        if m:
            cur = m.group(1)
            ids.add(cur)
            continue
        m = re.match(r"  status: (\S+)", line)
        if m and cur and m.group(1) == "superseded":
            sup[cur] = "?"
            continue
        m = re.match(r"  replaced_by: (.*)", line)
        if m and cur in sup:
            sup[cur] = m.group(1).strip()
    assert sup, "未解析到 superseded 条目——解析逻辑与 yaml 格式已脱节"
    bad = {k: v for k, v in sup.items() if v == "?" or v not in ids}
    assert not bad, "悬空 replaced_by（应为现存 id）：%s" % bad
