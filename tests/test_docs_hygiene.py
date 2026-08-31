"""文档卫生回归防护 —— 守住两条会静默劣化的规矩。

为什么需要
----------
1. **裸 `\\r` 地雷**：`docs/PERFORMANCE.md` 历史上攒了 934 个裸 CR，
   它们把文件变成"二进制"（git diff 只输出 `Binary files differ`，没法 review），
   还让 CommonMark 在 934 处渲染错误。2026-08-31 一次性清除（提交 3086a92）。
   这类残留是**静默**回来的：某次跨编辑器复制就可能重新引入，没人会在
   code review 里发现。
2. **CLAUDE.md 注入预算**：本文件被部分 harness 在**每个会话开头全量注入**，
   所以它有一个硬上限。历史教训是它会自己长到 70 KB（≈15–21K tokens），
   靠"自觉"是守不住的 —— 必须有测试拦。

本文件只做**结构性**校验，不校验文档内容对错。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from _paths import ROOT

# 硬上限：CLAUDE.md 在每个会话开头被注入，超过就必须迁内容到 docs/DECISIONS.md
CLAUDE_MD_MAX_BYTES = 12 * 1024

DOCS = ["README.md", "CLAUDE.md", "docs/PERFORMANCE.md", "docs/DECISIONS.md",
        "docs/DEPENDENCIES.md", "docs/ARCHIVE.md", "tools/INDEX.md"]


def _existing(rel: str) -> Path | None:
    p = ROOT / rel
    return p if p.is_file() else None


@pytest.mark.parametrize("rel", DOCS)
def test_no_bare_cr(rel: str) -> None:
    """文档里不得有裸 `\\r`。

    裸 CR（不跟 `\\n` 的 CR）会让 git 把文件判为二进制，且被 CommonMark
    当作行结束符 → 块引用被劈段、意外硬换行。
    """
    p = _existing(rel)
    if p is None:
        pytest.skip(f"{rel} 尚不存在")
    data = p.read_bytes()
    # 真 CRLF 是允许的（虽然本项目约定 LF，但 CRLF 不会造成上面的危害）；
    # 禁止的是"\r 后面不跟 \n"。
    bare = len(re.findall(rb"\r(?!\n)", data))
    assert bare == 0, (
        f"{rel} 含 {bare} 个裸 CR（不跟 LF 的 CR）。\n"
        f"这会让 git 把文件判为二进制（diff 只显示 Binary files differ），"
        f"并被 CommonMark 当成行结束符造成渲染错误。\n"
        f"修法：re.sub(r'\\r(>?[ ]*)(?=\\n)', '', txt)"
    )


def test_claude_md_within_injection_budget() -> None:
    """CLAUDE.md 必须在注入预算内 —— 它是每个会话开头全量注入的。

    超了就把内容迁到 `docs/DECISIONS.md`，这里只留指针。
    """
    p = _existing("CLAUDE.md")
    if p is None:
        pytest.skip("CLAUDE.md 不存在")
    size = p.stat().st_size
    assert size <= CLAUDE_MD_MAX_BYTES, (
        f"CLAUDE.md 已 {size} 字节，超过注入预算 {CLAUDE_MD_MAX_BYTES} 字节"
        f"（硬上限 12 KB）。\n"
        f"它在每个会话开头被全量注入，涨上去等于每个会话都付 token。\n"
        f"处理：把历史/过程性章节迁到 docs/DECISIONS.md，本文件只留指针。"
    )


def test_doc_map_targets_exist() -> None:
    """CLAUDE.md「文档地图」里列出的每个文件都必须真实存在。

    防止文档地图指向尚未创建（或已被删/改名）的文件 —— 那会让新会话
    按图索骥扑空。
    """
    p = _existing("CLAUDE.md")
    if p is None:
        pytest.skip("CLAUDE.md 不存在")
    text = p.read_text(encoding="utf-8")
    missing = []
    for rel in re.findall(r"`((?:docs|tools)/[A-Za-z0-9_.-]+\.md)`", text):
        if not (ROOT / rel).is_file():
            missing.append(rel)
    assert not missing, (
        f"CLAUDE.md 文档地图指向了不存在的文件：{missing}\n"
        f"要么创建它，要么把地图里的那一行改掉。"
    )
