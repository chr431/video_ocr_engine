"""维护工具：把 PERFORMANCE.md 中的历史档案章节迁移到 docs/ARCHIVE.md。

用途：本仓库约定"PERFORMANCE.md 只保留现役结论，历史档案统一放
docs/ARCHIVE.md"。当新实验把旧章节标记为"历史"时，可运行本脚本把
指定章节从 PERFORMANCE.md 剪切到 ARCHIVE.md 末尾，避免正文膨胀。

用法：
  python tools/trim_perf_doc.py --section "## 4.5" [--target "## 5."]

--section：要迁移的章节标题前缀（从该行开始剪切）。
--target：迁移的结束边界（该行之前的全部内容属于被迁移章节，含该行前
  一行的尾部）；缺省 = 迁移到文件末尾。

注意：仅支持顶层 `## ` 章节边界；剪切后会在原位置插入一行
"（已归档至 docs/ARCHIVE.md）"占位。请勿在 CI 中使用——人工核对迁移
边界后再提交。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", required=True, help="被迁移章节标题前缀（含 ## 前缀）")
    ap.add_argument("--target", default=None,
                    help="迁移结束边界标题前缀（不含该行）；缺省迁移到文件末尾")
    args = ap.parse_args()

    perf = ROOT / "docs" / "PERFORMANCE.md"
    arch = ROOT / "docs" / "ARCHIVE.md"
    lines = perf.read_text(encoding="utf-8").splitlines(keepends=True)

    start = None
    for i, ln in enumerate(lines):
        if ln.startswith(args.section):
            start = i
            break
    if start is None:
        print(f"未找到章节 {args.section!r}", file=sys.stderr)
        return 2

    end = len(lines)
    if args.target:
        for i in range(start + 1, len(lines)):
            if lines[i].startswith(args.target):
                end = i
                break
        else:
            print(f"未找到结束边界 {args.target!r}", file=sys.stderr)
            return 2

    body = lines[start:end]
    # 剪切 body 尾部空行（保留占位后的单空行）
    while body and body[-1].strip() == "":
        body.pop()
    body_text = "".join(body).rstrip("\n")

    # 归档追加（带分隔线）
    with arch.open("a", encoding="utf-8") as f:
        f.write("\n\n---\n\n" + body_text + "\n")

    # 原位置占位
    new_lines = lines[:start] + [f"（已归档至 docs/ARCHIVE.md）\n\n"] + lines[end:]
    # 压缩连续空行
    out: list[str] = []
    prev_blank = False
    for ln in new_lines:
        blank = ln.strip() == ""
        if blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = blank
    perf.write_text("".join(out), encoding="utf-8")

    print(f"已迁移 {body_text.splitlines()[0]!r} 到 docs/ARCHIVE.md（{len(body)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
