"""知识库预算与一致性门禁（v2 §14.2 四预算之 knowledge 两项 + 渲染一致性）。

预算常量唯一事实源 = tools/_doc_guards.py（2026-10-10 治理改革轮收敛：
此前本文件与 _audit_ext 各写一遍上限，两处迟早分叉）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from _doc_guards import (  # noqa: E402
    AGENTS_MAX_BYTES,
    CONCLUSIONS_SRC_MAX_BYTES,
    KNOBS_YAML_MAX_BYTES,
)


def test_conclusions_budget():
    # 16 KB（2026-10-10 治理改革轮由 14KB 上调）：CONCLUSIONS 按需读取
    # （非会话注入），14KB 饱和后（实测 14,241/14,336 只剩 95B）每条新
    # 结论都要仓促修剪旧规范性文本——审查负担与漂移风险大于读取成本。
    # 超预算时的压缩顺序：evidence 字段（机械可压 ~15%）→ premises →
    # conclusion 正文最后动。
    size = (ROOT / "knowledge" / "conclusions.md").stat().st_size
    assert size <= CONCLUSIONS_SRC_MAX_BYTES, (
        "conclusions.md %d B 超 %d B 预算" % (size, CONCLUSIONS_SRC_MAX_BYTES))


def test_knobs_budget():
    size = (ROOT / "knowledge" / "knobs.yaml").stat().st_size
    assert size <= KNOBS_YAML_MAX_BYTES, (
        "knobs.yaml %d B 超 %d B 预算" % (size, KNOBS_YAML_MAX_BYTES))


def test_render_check_passes():
    r = subprocess.run(
        [sys.executable, str(ROOT / "knowledge" / "render.py")],
        capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_agents_budget():
    size = (ROOT / "AGENTS.md").stat().st_size
    assert size <= AGENTS_MAX_BYTES, (
        "AGENTS.md %d B 超 %d B 注入预算" % (size, AGENTS_MAX_BYTES))


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
