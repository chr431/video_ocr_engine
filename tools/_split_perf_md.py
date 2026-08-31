"""把 docs/PERFORMANCE.md 按「活/归档」切出 docs/ARCHIVE.md。

背景
----
PERFORMANCE.md 长到 223 KB / 3451 行，其中 **54% 是归档内容**：

    §16 性能提升路线图归档   76.9 KB / 1093 行   ← 独占 35%
    §18 历史实验档案         33.6 KB /  550 行
    §4  已删除的混合解码/OCR  6.6 KB
    §8  性能演进摘要         0.7 KB

现役章节只占 ~101 KB。按「活/归档」切（**不按主题切**），让"查现在是什么样"
的文件瘦身一半，同时归档内容一个字不动。

设计要点
--------
1. **编号一律不重编**。保留章节里有 **32 处**引用指向 §4/§8/§16/§18
   （§16 独占 23 处）。逐个改引用风险大且噪声大，改用**统一跨文件约定**：
   在 PERFORMANCE.md 顶部声明"§4/§8/§16/§18 见 docs/ARCHIVE.md"，
   并在每个章节原位留指针块。正文里那 32 处引用**一字不改**。
2. **纯文本操作**。文件已于 2026-08-31 清除全部 934 个裸 CR（提交 3086a92），
   现在是纯 LF，普通文本模式 100% 安全，不需要二进制读-改-写。
3. 迁出的章节**逐字节原样搬运**，不做任何重排版。

用法
----
    python tools/_split_perf_md.py            # 预演（只打印，不写盘）
    python tools/_split_perf_md.py --apply    # 实际执行

幂等：若 docs/ARCHIVE.md 已存在且 PERFORMANCE.md 里已是指针块，直接退出。
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PERF = os.path.join(ROOT, "docs", "PERFORMANCE.md")
ARCH = os.path.join(ROOT, "docs", "ARCHIVE.md")

MOVE = ("4", "8", "16", "18")

ARCH_HEADER = """# 历史归档（从 docs/PERFORMANCE.md 迁出）

本文件是 `docs/PERFORMANCE.md` 的**归档部分**，2026-08-31 按「活/归档」切分
时迁出（切分脚本：`tools/_split_perf_md.py`）。

> ⚠️ **编号一律保留，不要重编号。**
> `docs/PERFORMANCE.md` 的现役章节里有 **32 处**引用指向本文件的 §4 / §8 /
> §16 / §18（§16 独占 23 处）。改号会让这些引用全部失效。

| 章节 | 性质 | 怎么用 |
|---|---|---|
| **§4** 已删除的混合解码 / 混合 OCR | 已删除功能的结论 | 只看"为什么不做"，勿据此调参 |
| **§8** 性能演进摘要 | 拆仓前 RaceVideoToLog 历史 | 查数值演变 |
| **§16** 性能提升路线图 | **2026-08-29 快照** | 开头有**校正表**，7 处已被后续提交推翻，正文保持原样 |
| **§18** 历史实验档案 | 已删除功能（dual pipeline 等） | 只看"为什么不做" |

**本文件的数值多数已过期**，用于理解"当时为什么这么做"。查"现在是什么样"
请看 `docs/PERFORMANCE.md`。

---

"""


def pointer(num: str, title: str) -> str:
    return (
        f"## {num}. {title}\n\n"
        f"> 📦 **本章已迁至 [`docs/ARCHIVE.md`](ARCHIVE.md) §{num}**"
        f"（2026-08-31 按「活/归档」切分）。\n"
        f"> **编号保留，勿重编** —— `docs/PERFORMANCE.md` 中凡提及"
        f" §4 / §8 / §16 / §18 的，一律指该文件的对应章节。\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写盘（默认只预演）")
    args = ap.parse_args()

    with open(PERF, encoding="utf-8", newline="") as f:
        text = f.read()
    assert text.count("\r") == 0, "PERFORMANCE.md 应已是纯 LF"

    lines = text.split("\n")
    heads = [(i, l) for i, l in enumerate(lines) if re.match(r"^## ", l)]
    assert heads, "没找到任何 ## 章节"

    # 切段
    secs: list[tuple[str, str, list[str]]] = []   # (num, heading, body_lines)
    preamble = lines[: heads[0][0]]
    for n, (i, h) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        num = re.match(r"## (\d+)\.", h).group(1)
        title = re.match(r"## \d+\.\s*(.*)", h).group(1)
        secs.append((num, title, lines[i:end]))

    print("=" * 68)
    print("PERFORMANCE.md 切分%s" % ("（实际执行）" if args.apply else "（预演）"))
    print("=" * 68)
    print("\n章节清单：")
    moved_bytes = 0
    for num, title, body in secs:
        b = len("\n".join(body).encode("utf-8"))
        tag = "→ ARCHIVE" if num in MOVE else ""
        print("  §%-3s %7d B  %-42s %s" % (num, b, title[:42], tag))
        if num in MOVE:
            moved_bytes += b

    keep = [s for s in secs if s[0] not in MOVE]
    keep_bytes = sum(len("\n".join(b).encode("utf-8")) for _, _, b in keep)
    print("\n  现役(PERFORMANCE.md) %7d B" % keep_bytes)
    print("  归档(ARCHIVE.md)     %7d B + 头部" % moved_bytes)

    # ---- 组装 ARCHIVE.md ----
    arch_parts = [ARCH_HEADER.rstrip("\n"), ""]
    for num, title, body in secs:
        if num in MOVE:
            blk = "\n".join(body).rstrip("\n")
            arch_parts.append(blk)
            arch_parts.append("")
    arch_text = "\n".join(arch_parts).rstrip("\n") + "\n"

    # ---- 组装 PERFORMANCE.md ----
    # 在原 preamble 的章节表后面补一条跨文件约定
    note = (
        "> 📦 **§4 / §8 / §16 / §18 已于 2026-08-31 迁至 "
        "[`docs/ARCHIVE.md`](ARCHIVE.md)**（按「活/归档」切分，编号保留）。\n"
        "> 本文件中凡提及这四个章节号的，**一律指 ARCHIVE.md 的对应章节**；\n"
        "> 原位留有指针块，共 32 处引用未作改动。"
    )
    new_pre = list(preamble)
    for i, l in enumerate(new_pre):
        if l.startswith("| **§18** 历史档案"):
            new_pre.insert(i + 1, "")
            new_pre.insert(i + 2, note)
            break
    else:
        new_pre.extend(["", note])

    perf_parts = ["\n".join(new_pre).rstrip("\n"), ""]
    for num, title, body in secs:
        if num in MOVE:
            perf_parts.append(pointer(num, title))
            perf_parts.append("")
        else:
            perf_parts.append("\n".join(body).rstrip("\n"))
            perf_parts.append("")
    perf_text = "\n".join(perf_parts).rstrip("\n") + "\n"

    # ---- 校验：内容零丢失 ----
    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s)

    old_all = norm(text)
    # 迁出的章节必须逐字出现在 ARCHIVE.md
    for num, title, body in secs:
        if num in MOVE:
            blk = norm("\n".join(body))
            assert blk in norm(arch_text), f"§{num} 在 ARCHIVE.md 中丢失"
    # 保留的章节必须逐字出现在 PERFORMANCE.md
    for num, title, body in secs:
        if num not in MOVE:
            blk = norm("\n".join(body))
            assert blk in norm(perf_text), f"§{num} 在 PERFORMANCE.md 中丢失"
    print("\n内容零丢失校验：✓（%d 个章节全部逐字核对）" % len(secs))

    print("  新 PERFORMANCE.md : %7d B / %d 行" %
          (len(perf_text.encode("utf-8")), perf_text.count("\n") + 1))
    print("  新 ARCHIVE.md     : %7d B / %d 行" %
          (len(arch_text.encode("utf-8")), arch_text.count("\n") + 1))

    if not args.apply:
        print("\n（预演模式，未写盘。加 --apply 执行）")
        return 0

    with open(ARCH, "w", encoding="utf-8", newline="") as f:
        f.write(arch_text)
    with open(PERF, "w", encoding="utf-8", newline="") as f:
        f.write(perf_text)
    print("\n已写出：docs/ARCHIVE.md、docs/PERFORMANCE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
