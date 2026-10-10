"""文档守卫的单一实现 —— 同一条规则只写一遍（2026-10-10 治理改革轮）。

为什么需要
----------
此前「文档裸 CR / AGENTS.md 注入预算 / 各文档预算」在三个地方各实现一遍：

  · tools/_probe_discipline_audit.py  #9（裸 CR）、#10（AGENTS 预算）
  · tools/_audit_ext.py               #18（文档预算，常量内嵌）
  · tests/test_docs_hygiene.py + tests/knowledge/test_budgets.py（常量再抄一遍）

两处常量、两处文件清单，迟早分叉——AGENTS.md 里 mypy 检查面的描述就是
这样漂移的（写了 domain+pipeline+decode，pyproject 实际已含 gpu）。
本模块是这些规则与常量的**唯一事实源**：审计脚本与 pytest 都从这里取数。

消费方
------
  · tools/_probe_discipline_audit.py   #9 / #10（薄委托）
  · tools/_audit_ext.py                #18（薄委托）
  · tests/test_docs_hygiene.py         （预算与裸 CR 断言）
  · tests/knowledge/test_budgets.py    （knowledge 两项预算）
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ── 预算（唯一事实源；改这里即同时改审计与 pytest 两处门禁）────────────
AGENTS_MAX_BYTES = 14 * 1024
# knowledge 源：2026-10-10 由 14 KB 上调至 16 KB——14,241/14,336 只剩 95 B，
# 饱和后每条新结论都要仓促修剪旧文本（审查负担与漂移风险大于按需读取成本）。
CONCLUSIONS_SRC_MAX_BYTES = 16 * 1024
CONCLUSIONS_MD_MAX_BYTES = 13 * 1024   # 渲染产物 L1（按需读取，非注入）
KNOBS_YAML_MAX_BYTES = 24 * 1024
DOCS_TOP_TOTAL_MAX_BYTES = 60 * 1024   # docs/*.md 合计（不含 log/ 大档案）

# 裸 CR 检查覆盖的治理文档（审计旧表 7 份 + CONCLUSIONS/log README，取并集）
DOCS = ["README.md", "AGENTS.md", "docs/log/PERFORMANCE.md", "docs/log/DECISIONS.md",
        "docs/DEPENDENCIES.md", "docs/log/ARCHIVE.md", "tools/INDEX.md",
        "docs/CONCLUSIONS.md", "docs/log/README.md", "docs/MIGRATION.md",
        "knowledge/conclusions.md", "knowledge/rules.yaml"]

_BARE_CR = re.compile(rb"\r(?!\n)")


def bare_cr_violations() -> list[tuple[str, int]]:
    """各治理文档的裸 CR 计数（\r 不跟 \n；git 会判二进制、CommonMark 渲染错）。"""
    bad: list[tuple[str, int]] = []
    for rel in DOCS:
        p = ROOT / rel
        if not p.is_file():
            continue
        n = len(_BARE_CR.findall(p.read_bytes()))
        if n:
            bad.append((rel, n))
    return bad


def budget_violations() -> list[tuple[str, int, int]]:
    """超出预算的 (文件, 实测字节, 上限)。空列表 = 全部在预算内。"""
    out: list[tuple[str, int, int]] = []
    caps = (("AGENTS.md", AGENTS_MAX_BYTES),
            ("knowledge/conclusions.md", CONCLUSIONS_SRC_MAX_BYTES),
            ("docs/CONCLUSIONS.md", CONCLUSIONS_MD_MAX_BYTES),
            ("knowledge/knobs.yaml", KNOBS_YAML_MAX_BYTES))
    for rel, cap in caps:
        p = ROOT / rel
        if p.is_file() and p.stat().st_size > cap:
            out.append((rel, p.stat().st_size, cap))
    total = sum(p.stat().st_size for p in (ROOT / "docs").glob("*.md"))
    if total > DOCS_TOP_TOTAL_MAX_BYTES:
        out.append(("docs/*.md 合计", total, DOCS_TOP_TOTAL_MAX_BYTES))
    return out


# ── 大文档可分章（N2：防整读的使能结构）───────────────────────────────
# 活文档一旦超过阈值，必须带 ≥3 个章节标题——否则 _doc_section.py 无法
# --toc/--find/单章定位，agent 只能整读（PERFORMANCE 42k tok 的教训）。
# 冻结大档案（PERFORMANCE/ARCHIVE/DECISIONS）天然满足；本项防的是**新增**。
BIG_DOC_THRESHOLD = 24 * 1024
_HEADING = re.compile(r"^#{2,3} ", re.M)


def big_doc_section_violations() -> list[tuple[str, int, int]]:
    """(文件, 字节, 标题数)——超阈值且标题不足 3 个的活文档。"""
    bad: list[tuple[str, int, int]] = []
    targets = ["README.md", "AGENTS.md", "docs/DEPENDENCIES.md", "docs/MIGRATION.md",
               "docs/CONCLUSIONS.md", "tools/INDEX.md"]
    log_dir = ROOT / "docs" / "log"
    if log_dir.is_dir():
        targets += sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                          for p in log_dir.glob("*.md"))
    for rel in targets:
        p = ROOT / rel
        if not p.is_file():
            continue
        size = p.stat().st_size
        if size <= BIG_DOC_THRESHOLD:
            continue
        n_head = len(_HEADING.findall(p.read_text(encoding="utf-8", errors="replace")))
        if n_head < 3:
            bad.append((rel, size, n_head))
    return bad


if __name__ == "__main__":
    sys_out = [("裸 CR", bare_cr_violations()), ("预算", budget_violations()),
               ("大文档分章", big_doc_section_violations())]
    for label, hits in sys_out:
        print("%s: %s" % (label, hits if hits else "✓"))
    raise SystemExit(1 if any(hits for _, hits in sys_out) else 0)
