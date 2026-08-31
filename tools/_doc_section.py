"""文档章节级检索引擎 —— 让 agent 只读需要的那一章，而不是整个文件。

为什么需要
----------
实测（tiktoken cl100k_base，2026-09-01）：

| 文件 | 整文件 tokens | 章节数 | 单章均值 | 目录 tokens |
|---|---:|---:|---:|---:|
| docs/PERFORMANCE.md | 42,470 | 21 | 1,988 | 2,622 |
| docs/ARCHIVE.md     | 46,253 | 14 | 3,256 |   524 |
| docs/DECISIONS.md   | 29,332 | 26 | 1,073 |   928 |

对比一下量级：**CLAUDE.md 每会话注入才 3,636 tokens**，而误读一次
ARCHIVE.md 就是 46,253 —— 相当于 **12.7 倍的注入成本**。

所以降低 agent token 消耗的最大杠杆**不是压缩 CLAUDE.md**，而是避免整文件
读取：先看目录（几百 tokens）定位，再只读那一章（1~3K tokens），
**省 85~96%**。

用法
----
    # 1) 看目录（带 token 数，好判断值不值得读）
    python tools/_doc_section.py --toc                     # 全部文档
    python tools/_doc_section.py --toc docs/PERFORMANCE.md # 单份

    # 2) 按标题关键词定位（不知道在哪个文件/哪一章时用）
    python tools/_doc_section.py --find NVDEC
    python tools/_doc_section.py --find 带宽 --max-tokens 3000

    # 3) 读指定章节
    python tools/_doc_section.py docs/PERFORMANCE.md 21
    python tools/_doc_section.py docs/DECISIONS.md 设计审查结论
    python tools/_doc_section.py docs/ARCHIVE.md 16.8      # 支持子章节号

输出都带 token 计数，方便判断这一章够不够便宜。

依赖
----
`tiktoken` 用于精确计数；没装时自动退化为「按 CJK/ASCII 拆分」的估算，
并在输出里标注是估算值（不静默给出假精度）。
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DOCS = ["docs/PERFORMANCE.md", "docs/ARCHIVE.md", "docs/DECISIONS.md",
        "README.md", "CLAUDE.md", "docs/DEPENDENCIES.md"]

# 每份文档按哪个标题层级切分（DECISIONS.md 主体是 ###，不是 ##）
LEVEL = {
    "docs/PERFORMANCE.md": 2,
    "docs/ARCHIVE.md": 3,
    "docs/DECISIONS.md": 3,
    "README.md": 2,
    "CLAUDE.md": 2,
    "docs/DEPENDENCIES.md": 2,
}

_enc = None
_exact = True


def tok(text: str) -> int:
    """token 计数。tiktoken 不可用时退化为估算，并标出来。"""
    global _enc, _exact
    if _enc is None:
        try:
            import tiktoken
            _enc = tiktoken.get_encoding("cl100k_base")
            _exact = True
        except Exception:
            _enc = None
            _exact = False
    if _enc is not None:
        return len(_enc.encode(text))
    # 估算：CJK 约 0.6~0.9 token/字，ASCII 约 0.22~0.30 token/字符
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return int(cjk * 0.75 + (len(text) - cjk) * 0.26)


def split_key(h: str) -> tuple[str, str]:
    r"""从标题里拆出 (章节号, 标题)。

    章节号要能吃下 `16` / `16.8` / `4.4b` 这三种形态 —— 只取 `\d+(\.\d+)*`
    会把 §4.4b 和 §4.4c 都截成 `4.4`，目录里就分不清了。
    """
    m = re.match(r"^#+\s*(\d+(?:\.\d+)*[a-z]?)[.\s]*(.*)$", h)
    if m:
        return m.group(1), m.group(2).strip()
    bare = h.lstrip("# ").strip()
    return (bare.split()[0] if bare else "?"), bare


def split_sections(rel: str) -> list[tuple[str, str, int, int]]:
    """返回 [(标题, 正文, token数, 起始行号)]。"""
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        lines = f.read().split("\n")
    lvl = LEVEL.get(rel, 2)
    pat = re.compile(r"^%s " % ("#" * lvl))
    heads = [(i, l) for i, l in enumerate(lines) if pat.match(l)]
    out = []
    for n, (i, h) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        body = "\n".join(lines[i:end]).rstrip()
        out.append((h.strip(), body, tok(body), i + 1))
    return out


def show_toc(only: str | None) -> None:
    targets = [only] if only else [d for d in DOCS
                                   if os.path.isfile(os.path.join(ROOT, d))]
    print("token 计数：%s\n" % ("tiktoken 精确值" if _exact else "估算值（未装 tiktoken）"))
    grand = 0
    for rel in targets:
        secs = split_sections(rel)
        if not secs:
            continue
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            total = tok(f.read())
        grand += total
        print("── %s   全文 %s tokens / %d 章" % (rel, format(total, ","), len(secs)))
        for h, body, n, ln in secs:
            key, title = split_key(h)
            print("   %-10s %6s  L%-5d %s" % (key[:10], format(n, ","), ln, title[:54]))
        print()
    print("合计 %s tokens" % format(grand, ","))


def show_find(pattern: str, max_tok: int | None) -> None:
    pat = re.compile(pattern, re.I)
    hits = []
    for rel in DOCS:
        if not os.path.isfile(os.path.join(ROOT, rel)):
            continue
        for h, body, n, ln in split_sections(rel):
            if pat.search(h):
                if max_tok and n > max_tok:
                    continue
                hits.append((rel, h, n, ln))
    if not hits:
        print("没找到标题匹配 %r 的章节。" % pattern)
        print("提示：--find 只搜**标题**。要搜正文请用 grep。")
        return
    print("标题匹配 %r 的章节 %d 个：\n" % (pattern, len(hits)))
    for rel, h, n, ln in hits:
        key, title = split_key(h)
        print("  %-20s §%-10s %6s tokens  L%d"
              % (rel.replace("docs/", ""), key[:10], format(n, ","), ln))
        print("      %s" % (title or h.lstrip("# "))[:66])
    print("\n读取：python tools/_doc_section.py <文件> <章节号>")


def show_section(rel: str, key: str) -> None:
    secs = split_sections(rel)
    if not secs:
        print("找不到文档 %s" % rel)
        return
    # 1) 按章节号匹配（支持 16 / 16.8 这种前缀）
    cand = [s for s in secs
            if re.match(r"^#+\s*%s(?:\.|\s|$)" % re.escape(key), s[0])]
    # 2) 按标题关键词匹配
    if not cand:
        cand = [s for s in secs if key.lower() in s[0].lower()]
    if not cand:
        print("在 %s 里找不到章节 %r。用 --toc 看目录。" % (rel, key))
        return
    if len(cand) > 1:
        print("[匹配到 %d 章，全部输出]\n" % len(cand))
    for h, body, n, ln in cand:
        print("<!-- %s  L%d  %s tokens -->" % (rel, ln, format(n, ",")))
        print(body)
        print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="文档章节级检索 —— 先看目录，再只读需要的那一章。")
    ap.add_argument("--toc", nargs="?", const="", metavar="FILE",
                    help="打印目录（带 token 数）；不给 FILE 则列全部文档")
    ap.add_argument("--find", metavar="PATTERN",
                    help="按**标题**关键词跨文档定位章节")
    ap.add_argument("--max-tokens", type=int, metavar="N",
                    help="配合 --find：只显示不超过 N tokens 的章节")
    ap.add_argument("rest", nargs="*", help="<文件> <章节号或标题关键词>")
    args = ap.parse_args()

    if args.toc is not None:
        show_toc(args.toc or None)
        return 0
    if args.find:
        show_find(args.find, args.max_tokens)
        return 0
    if len(args.rest) >= 2:
        show_section(args.rest[0], args.rest[1])
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
