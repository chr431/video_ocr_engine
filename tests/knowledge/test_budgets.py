"""知识库预算与一致性门禁（v2 §14.2 四预算之 knowledge 两项 + 渲染一致性）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_conclusions_budget():
    # 14 KB（2026-09-20 稳健性轮由 12KB 上调）：CONCLUSIONS.md 按需读取
    # （非会话注入），14KB≈3.5k tok/次定位成本；12KB 饱和后每次新条目
    # 都要修剪旧规范性文本（实测三轮修剪触碰 20+ 条措辞）——审查负担
    # 与漂移风险大于读取成本。超预算时的压缩顺序：evidence 字段
    # （机械可压 ~15%）→ premises → conclusion 正文最后动。
    size = (ROOT / "knowledge" / "conclusions.md").stat().st_size
    assert size <= 14 * 1024, "conclusions.md %d B 超 14 KB 预算" % size


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
    assert size <= 14 * 1024, "AGENTS.md %d B 超 14 KB 注入预算" % size


def test_superseded_pointers_resolve():
    """`superseded →` 的指针必须落在现存结论 id 上。

    防「C-27 → ?」式悬空：本轮实测曾有一条 2026-09 起就挂着 `?` 的记录，
    因为 replaced_by 只在渲染时拼成一行、没有任何门禁看它是否解析得到。
    """
    import re
    text = (ROOT / "knowledge" / "conclusions.md").read_text(encoding="utf-8")
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
    assert sup, "未解析到 superseded 条目——解析逻辑与源文件格式已脱节"
    bad = {k: v for k, v in sup.items() if v == "?" or v not in ids}
    assert not bad, "悬空 replaced_by（应为现存 id）：%s" % bad
