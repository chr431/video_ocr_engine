"""`tools/_doc_section.py` 的守卫测试。

为什么值得测
------------
这个工具是**省 token 的关键路径**：用它读单章 ≈ 2,800 tokens，整文件读
42,470~46,253 tokens。它要是坏了，agent 会静默退回"整文件读"——
不会报错，只是每次查询多花十几倍 token。这种退化不会有任何测试失败，
只能靠这里守住。

测试只覆盖**契约**（能不能用、找不找得到），不覆盖具体文案。
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from _paths import ROOT

TOOL = ROOT / "tools" / "_doc_section.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args],
                          cwd=str(ROOT), capture_output=True, text=True)


def test_toc_lists_sections_with_token_counts() -> None:
    """`--toc` 必须列出章节并带 token 数。"""
    r = run("--toc", "docs/PERFORMANCE.md")
    assert r.returncode == 0, r.stderr
    assert "tokens" in r.stdout
    # 至少列出 PERFORMANCE.md 的 21 个 `## ` 章节
    assert r.stdout.count("\n   ") >= 15


def test_toc_all_docs_no_crash() -> None:
    """不给参数时列全部文档，不能崩。"""
    r = run("--toc")
    assert r.returncode == 0, r.stderr
    assert "PERFORMANCE.md" in r.stdout


def test_find_locates_known_section() -> None:
    """`--find` 按标题关键词跨文档定位。"""
    r = run("--find", "NVDEC")
    assert r.returncode == 0, r.stderr
    assert "ARCHIVE.md" in r.stdout or "DECISIONS.md" in r.stdout


def test_find_no_match_is_explicit() -> None:
    """搜不到要明说，并提示 --find 只搜标题。"""
    r = run("--find", "绝不存在的关键词ZZZ")
    assert r.returncode == 0
    assert "没找到" in r.stdout
    assert "只搜" in r.stdout or "标题" in r.stdout


def test_read_section_by_number() -> None:
    """按章节号读，输出里得带上来源标注和正文。"""
    r = run("docs/PERFORMANCE.md", "5")
    assert r.returncode == 0, r.stderr
    assert "docs/PERFORMANCE.md" in r.stdout        # 来源标注
    assert "## 5." in r.stdout


def test_read_section_by_alphanumeric_number() -> None:
    """`4.4b` 这种带字母的子章节号要能读（不能只支持纯数字）。"""
    r = run("docs/ARCHIVE.md", "4.4b")
    assert r.returncode == 0, r.stderr
    assert "4.4b" in r.stdout


def test_read_section_by_title_keyword() -> None:
    """按标题关键词读。"""
    r = run("docs/DECISIONS.md", "设计审查结论")
    assert r.returncode == 0, r.stderr
    assert "设计审查" in r.stdout


def test_unknown_section_is_explicit() -> None:
    """章节号不存在要提示看目录，而不是空输出。"""
    r = run("docs/PERFORMANCE.md", "999")
    assert "找不到" in r.stdout
    assert "--toc" in r.stdout


def test_toc_is_cheap_compared_to_full_file() -> None:
    """目录本身必须远小于全文 —— 否则"先看目录"就不划算了。

    实测：PERFORMANCE.md 全文 42,470 tokens，目录输出约 838 tokens（2%）。
    这里放宽到 15%，防止目录被塞进正文后悄悄膨胀。
    """
    toc = run("--toc", "docs/PERFORMANCE.md").stdout
    full = (ROOT / "docs" / "PERFORMANCE.md").read_text(encoding="utf-8")
    assert len(toc) < len(full) * 0.15, (
        "目录输出 %d 字符，已达全文 %d 字符的 %.0f%%（应 <15%%）—— "
        "目录膨胀会让「先看目录再读单章」失去意义"
        % (len(toc), len(full), len(toc) / len(full) * 100)
    )
