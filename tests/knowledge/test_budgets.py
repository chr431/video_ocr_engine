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
    """死档（conclusions-history）里 `superseded → X` 指针必须可解析。

    2026-10-09 清扫起 superseded 不再住 knowledge 源（仅 active），
    指针解析义务随全文迁到死档：第一跳必须落在 active 源或死档自身的
    id 上。防「C-27 → ?」式悬空（历史上 replaced_by 无门禁真实出现过）。
    """
    import re
    src = (ROOT / "knowledge/conclusions.md").read_text(encoding="utf-8")
    active = set(re.findall(r"- id: (\S+)", src))
    hist = (ROOT / "docs/log/2026-09-10-conclusions-history.md"
            ).read_text(encoding="utf-8")
    hist_ids = set(re.findall(r"^## (C-\d+)", hist, re.M))
    bad = []
    for m in re.finditer(r"^## (C-\d+)（superseded → ([^）]+)）", hist, re.M):
        first = m.group(2).split("→")[0].strip()
        if first not in active and first not in hist_ids:
            bad.append((m.group(1), first))
    assert not bad, "死档悬空指针：%s" % bad
