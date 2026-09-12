"""文档卫生回归防护 —— 守住两条会静默劣化的规矩。

为什么需要
----------
1. **裸 `\\r` 地雷**：`docs/log/PERFORMANCE.md` 历史上攒了 934 个裸 CR，
   它们把文件变成"二进制"（git diff 只输出 `Binary files differ`，没法 review），
   还让 CommonMark 在 934 处渲染错误。2026-08-31 一次性清除（提交 3086a92）。
   这类残留是**静默**回来的：某次跨编辑器复制就可能重新引入，没人会在
   code review 里发现。
2. **AGENTS.md 注入预算**：本文件被部分 harness 在**每个会话开头全量注入**，
   所以它有一个硬上限。历史教训是它会自己长到 70 KB（≈15–21K tokens），
   靠"自觉"是守不住的 —— 必须有测试拦。

本文件只做**结构性**校验，不校验文档内容对错。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from _paths import ROOT

# 硬上限：AGENTS.md 在每个会话开头被注入，超过就必须迁内容到 docs/log/DECISIONS.md
CLAUDE_MD_MAX_BYTES = 12 * 1024

DOCS = ["README.md", "AGENTS.md", "docs/log/PERFORMANCE.md", "docs/log/DECISIONS.md",
        "docs/DEPENDENCIES.md", "docs/log/ARCHIVE.md", "tools/INDEX.md",
        "docs/CONCLUSIONS.md", "docs/log/README.md"]


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
    """AGENTS.md 必须在注入预算内 —— 它是每个会话开头全量注入的。

    超了就把内容迁到 `docs/log/DECISIONS.md`，这里只留指针。
    """
    p = _existing("AGENTS.md")
    if p is None:
        pytest.skip("AGENTS.md 不存在")
    size = p.stat().st_size
    assert size <= CLAUDE_MD_MAX_BYTES, (
        f"AGENTS.md 已 {size} 字节，超过注入预算 {CLAUDE_MD_MAX_BYTES} 字节"
        f"（硬上限 12 KB）。\n"
        f"它在每个会话开头被全量注入，涨上去等于每个会话都付 token。\n"
        f"处理：把历史/过程性章节迁到 docs/log/DECISIONS.md，本文件只留指针。"
    )


def test_doc_map_targets_exist() -> None:
    """AGENTS.md「文档地图」里列出的每个文件都必须真实存在。

    防止文档地图指向尚未创建（或已被删/改名）的文件 —— 那会让新会话
    按图索骥扑空。
    """
    p = _existing("AGENTS.md")
    if p is None:
        pytest.skip("AGENTS.md 不存在")
    text = p.read_text(encoding="utf-8")
    missing = []
    for rel in re.findall(r"`((?:docs|tools)/[A-Za-z0-9_.-]+\.md)`", text):
        if not (ROOT / rel).is_file():
            missing.append(rel)
    assert not missing, (
        f"AGENTS.md 文档地图指向了不存在的文件：{missing}\n"
        f"要么创建它，要么把地图里的那一行改掉。"
    )


# ════════ 文档分层管理守护（2026-09-08，见 DECISIONS「文档分层管理」）════════

CONCLUSIONS_MD_MAX_BYTES = 12 * 1024  # L1 索引的体积上限（现约 6.6KB）

# "现役"字样只允许出现在 AGENTS.md（注入核/总真相）与 docs/CONCLUSIONS.md
# （L1 索引）。其余文档按「存量豁免、增量严格」：基线以下只许下降不许上涨。
# （基线 = 2026-09-08 文档分层改造时点的计数。）
XIANYI_BASELINE = {
    "README.md": 3,
    "tools/INDEX.md": 1,
}

# 前瞻性文档：读者默认其中的文件引用可直接跟进去。历史档案
# （PERFORMANCE/DECISIONS/ARCHIVE 正文）不检查——那里合法地引用已删除的文件。
DANGLING_CHECK_FILES = ["README.md", "AGENTS.md", "docs/CONCLUSIONS.md",
                        "docs/DEPENDENCIES.md", "tools/INDEX.md"]

_FILE_REF = re.compile(r"`([^`\n]+?\.(?:py|md|json))`")
# 合法引用形如 `tools/x.py` / `engine_config.py`：纯 ASCII 路径、无空格、无通配符
_SANE_NAME = re.compile(r"^[\w./-]+$", re.ASCII)
# 行级豁免：历史注记合法地提及已删除文件（含跨行换行时的近义表述）
_HIST_MARKS = ("已删除", "删除", "历史", "dead")


def test_conclusions_within_budget() -> None:
    """docs/CONCLUSIONS.md 必须在预算内 —— 动手前第一个要读的就是它，
    涨上去等于每次开发都付税。超了就精简措辞或把细节退回证据章节。"""
    p = _existing("docs/CONCLUSIONS.md")
    if p is None:
        pytest.skip("docs/CONCLUSIONS.md 不存在")
    size = p.stat().st_size
    assert size <= CONCLUSIONS_MD_MAX_BYTES, (
        f"docs/CONCLUSIONS.md 已 {size} 字节，超过预算 {CONCLUSIONS_MD_MAX_BYTES}。\n"
        f"L1 索引只放一行式结论；长证据/叙事退回 docs/log/ 或 PERF 对应章节。"
    )


def test_xianyi_outside_l1_not_growing() -> None:
    """「现役」表述只许住 AGENTS.md 与 docs/CONCLUSIONS.md。

    历史文档里的存量按基线豁免，但**只许减不许增**——新增规范性表述
    必须写进 L1 索引，而不是散落在叙事里（hybrid 迁移清扫的教训）。
    docs/log/ 是新叙事区，直接禁止。
    """
    for rel, baseline in XIANYI_BASELINE.items():
        p = _existing(rel)
        if p is None:
            continue
        n = p.read_text(encoding="utf-8").count("现役")
        assert n <= baseline, (
            f"{rel} 的「现役」计数 {n} 超过基线 {baseline}。\n"
            f"新的现役/规范性表述请写入 docs/CONCLUSIONS.md（一行一条，带状态）。"
        )
    log_dir = ROOT / "docs" / "log"
    _DEMOTED = {"PERFORMANCE.md", "ARCHIVE.md", "DECISIONS.md"}  # S2:冻结历史
    if log_dir.is_dir():
        for p in sorted(log_dir.glob("*.md")):
            if p.name in _DEMOTED:
                continue
            n = p.read_text(encoding="utf-8").count("现役")
            assert n == 0, (
                f"docs/log/{p.name} 含 {n} 处「现役」——叙事目录禁止规范性语句，"
                f"结论请一行写进 docs/CONCLUSIONS.md。"
            )


def test_no_dangling_file_refs() -> None:
    """前瞻性文档中反引号引用的仓库文件必须真实存在。

    抓 `hybrid_decode.py` 式悬空引用：文件删了、文档还让人"跟进去"。
    豁免：所在行含「已删除」（历史注记合法提及死者）；git 不可用时跳过。
    """
    import subprocess

    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, shell=True,
                                 capture_output=True, text=True, timeout=30,
                                 check=True).stdout.split()
    except Exception:
        pytest.skip("git ls-files 不可用")
    tracked_set = set(tracked)
    base_set = {p.rsplit("/", 1)[-1] for p in tracked_set}

    files = list(DANGLING_CHECK_FILES)
    log_dir = ROOT / "docs" / "log"
    _DEMOTED = {"PERFORMANCE.md", "ARCHIVE.md", "DECISIONS.md"}  # S2:历史档案
    if log_dir.is_dir():
        files.extend(str(p.relative_to(ROOT)).replace("\\", "/")
                     for p in sorted(log_dir.glob("*.md"))
                     if p.name not in _DEMOTED)

    missing = []
    for rel in files:
        p = _existing(rel)
        if p is None:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if any(mark in line for mark in _HIST_MARKS):
                continue
            for name in _FILE_REF.findall(line):
                if not _SANE_NAME.match(name):
                    continue  # 命令片段、glob 通配符、占位模板
                if name.startswith(("http://", "https://")):
                    continue
                if "/" in name:
                    ok = name in tracked_set or (ROOT / name).is_file()
                else:
                    ok = name in base_set
                if not ok:
                    missing.append(f"{rel}:{i} `{name}`")
    assert not missing, (
        f"前瞻性文档引用了仓库中不存在的文件：{missing}\n"
        f"要么改指现存文件，要么在行内注明「已删除」转为历史注记。"
    )


def test_param_defaults_consistent_with_engine_config() -> None:
    """README 宣称的参数默认值必须与 engine_config 一致。

    防「engine 已改、文档没改」式漂移（实例：HYBRID 分档 README 写 [8, 24]
    而 engine_config 已是 [8, 16]，漂移存活了两周）。清单可按需追加。
    """
    import engine_config as config

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expectations = [
        ("HYBRID_CPU_THREADS 自动分档",
         "自动：与 CPU 软解后端同一 codec 感知策略"),
        ("OCR pad 下限", f"默认 {config.OCR_PAD_WIDTH_MIN}"),
        ("OCR 批大小", f"默认 {config.OCR_BATCH_SIZE}"),
    ]
    broken = [desc for desc, needle in expectations if needle not in readme]
    assert not broken, (
        f"README 的默认值表述与 engine_config 不一致：{broken}。\n"
        f"参数值的唯一事实源是 engine_config.py；改引擎默认值时必须同步 README。"
    )
