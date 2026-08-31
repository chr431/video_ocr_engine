"""tools/INDEX.md 一致性审计 —— 索引里的每个数字都要跟磁盘对齐。

动机
----
`tools/INDEX.md` 是手写的索引，它声称的数字（文件数、行数、依赖关系、
孤儿集合）会随探针增删而漂移。本脚本用实测核对这些数字，避免索引变成
"看起来很全、其实是过期信息"的第二个 CLAUDE.md。

检查项
------
1. 文件总数 / 总行数 / 总字节，与 INDEX 头部声称值比对
2. 索引覆盖度：实际存在但未索引、索引了但不存在的文件
3. 索引里的文件名拼写（是否存在 `_probe_sk?ip_frame.py` 这类损坏）
4. 依赖关系复核：谁 import / 子进程调用谁（实测，不信任索引）
5. 孤儿集合复核：既无文档引用、也无代码引用的探针
6. 索引表格里每个 (文件, 行数) 对是否与磁盘一致

用法
----
    python tools/_probe_index_audit.py
    python tools/_probe_index_audit.py --fix-hint   # 额外打印可直接粘贴的修正值

退出码 0 = 全部一致；1 = 发现不一致（CI 可直接用）。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

INDEX = os.path.join(HERE, "INDEX.md")
DOCS = ["README.md", "CLAUDE.md", "docs/PERFORMANCE.md", "docs/DECISIONS.md",
        "docs/DEPENDENCIES.md", "docs/ARCHIVE.md"]


def read_bytes(p: str) -> bytes:
    with open(p, "rb") as f:
        return f.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-hint", action="store_true",
                    help="额外打印可直接粘贴回 INDEX.md 的修正值")
    args = ap.parse_args()

    problems: list[str] = []

    # ---------- 实测 ----------
    files: dict[str, bytes] = {}
    for f in glob.glob(os.path.join(HERE, "*.py")):
        files[os.path.basename(f)] = read_bytes(f)
    lines_of = {k: v.count(b"\n") + 1 for k, v in files.items()}
    bytes_of = {k: len(v) for k, v in files.items()}
    n_file, n_line, n_byte = len(files), sum(lines_of.values()), sum(bytes_of.values())

    idx = read_bytes(INDEX).decode("utf-8")

    print("=" * 68)
    print("tools/INDEX.md 一致性审计")
    print("=" * 68)

    # ---------- [1] 头部声称值 ----------
    print("\n【1】文件总量")
    print("    实测      : %d 个 .py / %d 行 / %s 字节" %
          (n_file, n_line, format(n_byte, ",")))
    m = re.search(r"\*\*(\d+)\s*个\s*`\.py`\*\*[^\n]*?([\d,]+)\s*行", idx)
    if not m:
        problems.append("INDEX 头部没匹配到 (文件数, 行数) 声称值")
        print("    INDEX    : 匹配失败")
    else:
        cn, cl = int(m.group(1)), int(m.group(2).replace(",", ""))
        print("    INDEX    : %d 个 .py / %s 行" % (cn, format(cl, ",")))
        if cn != n_file:
            problems.append("头部文件数不符：声称 %d，实测 %d" % (cn, n_file))
        if cl != n_line:
            problems.append("头部行数不符：声称 %d，实测 %d" % (cl, n_line))

    # ---------- [2] 覆盖度 ----------
    # 注意：§16 那张表是「双列并排」（一行里有两个 | `x.py` | 行 | 改于 |），
    # 只取第一个会漏掉右半列。必须扫描整行的所有反引号文件名。
    print("\n【2】索引覆盖度")
    tbl: set[str] = set()
    for line in idx.split("\n"):
        for mm in re.finditer(r"`([^`]+\.py)`", line):
            # 取 basename：正文里可能出现 `python tools/_probe_x.py` 这种命令行
            tbl.add(os.path.basename(mm.group(1)))
    missing = sorted(set(files) - tbl)      # 有文件没索引
    # 排除 glob 模式（如 `tools/_probe_*.py` 出现在正文里是正常的，不是文件名）
    ghost = sorted(x for x in tbl - set(files) if "*" not in x)
    print("    表格内文件数 : %d" % len(tbl))
    print("    未索引      : %s" % (missing or "（无）"))
    print("    不存在/拼错 : %s" % (ghost or "（无）"))
    if missing:
        problems.append("有 %d 个文件未进索引: %s" % (len(missing), ", ".join(missing)))
    if ghost:
        problems.append("索引里有 %d 个不存在的文件名（很可能是拼写损坏）: %s"
                        % (len(ghost), ", ".join(ghost)))

    # ---------- [3] 依赖关系实测 ----------
    # 三种引用形式都要算：import / 子进程调用 / docstring 里的点名。
    # 只看前两种会漏判（`_probe_roi_segcost` 就只在 `_probe_seg_share` 的
    # docstring 里被点名，但那足以说明"别删"）。
    print("\n【3】依赖关系（实测，不信任索引）")
    dep: dict[str, list[str]] = {}
    for name, data in files.items():
        txt = data.decode("utf-8", "replace")
        base = name[:-3]
        for other in files:
            ob = other[:-3]
            if ob == base:
                continue
            if re.search(r"^\s*from\s+%s\s+import" % re.escape(ob), txt, re.M):
                dep.setdefault(other, []).append("%s (import)" % name)
            elif re.search(r"[\"']%s\.py[\"']" % re.escape(ob), txt):
                dep.setdefault(other, []).append("%s (子进程)" % name)
            elif ob in txt:
                dep.setdefault(other, []).append("%s (提及)" % name)
    if not dep:
        print("    （无跨探针依赖）")
    for k in sorted(dep):
        print("    %-32s ← %s" % (k, ", ".join(sorted(set(dep[k])))))

    # ---------- [4] D 节「孤儿」表的自洽性 ----------
    # 注意别写成循环定义：D 节自己列出了这些文件，所以"是否孤儿"不能靠
    # "有没有出现在 INDEX 里"来判断。这里只校验**D 节声称的合计**是否等于
    # **D 节列出的那些文件**的实测之和 —— 至于哪些文件该归 D，是人工判断。
    print("\n【4】D 节「孤儿」表自洽性")
    d_sec = re.search(r"## D\..*?(?=\n## E\.)", idx, re.S)
    listed_d: list[str] = []
    if d_sec:
        # 只扫表格行（以 | 开头）：D 节正文里的命令行示例（`python tools/x.py`）
        # 不该被当成"D 节列出的文件"。
        for line in d_sec.group(0).split("\n"):
            if not line.lstrip().startswith("|"):
                continue
            for mm in re.finditer(r"`([^`]+\.py)`", line):
                n = os.path.basename(mm.group(1))
                if n in files and n not in listed_d:
                    listed_d.append(n)
    tl = sum(lines_of.get(o, 0) for o in listed_d)
    tb = sum(bytes_of.get(o, 0) for o in listed_d)
    for o in listed_d:
        print("    %-32s %4d 行 %7d B" % (o, lines_of[o], bytes_of[o]))
    print("    D 节列出: %d 个 / %d 行 / %s B (%.1f KB)" %
          (len(listed_d), tl, format(tb, ","), tb / 1024))
    m = re.search(r"共\s*(\d+)\s*个\s*/\s*([\d,]+)\s*行\s*/\s*([\d.]+)\s*KB", idx)
    if m:
        co, clo, ckb = int(m.group(1)), int(m.group(2)), float(m.group(3))
        print("    INDEX 声称: %d 个 / %s 行 / %.1f KB" % (co, format(clo, ","), ckb))
        if (co, clo) != (len(listed_d), tl):
            problems.append("D 节合计不符：声称 %d 个/%d 行，D 表实测 %d 个/%d 行"
                            % (co, clo, len(listed_d), tl))
        if abs(ckb - tb / 1024) > 0.15:
            problems.append("D 节 KB 不符：声称 %.1f，实测 %.1f" % (ckb, tb / 1024))
    else:
        problems.append("D 节没匹配到 (个数, 行数, KB) 声称值")

    # ---------- [4b] 完全未归类（真·漏网） ----------
    docs_txt = ""
    for d in DOCS:
        p = os.path.join(ROOT, d)
        if os.path.isfile(p):
            docs_txt += read_bytes(p).decode("utf-8", "replace") + "\n"
    uncategorized = [n for n in sorted(files)
                     if n[:-3] not in docs_txt and n not in dep and n not in tbl]
    print("\n【4b】完全未归类（文档/代码/本索引里都查不到）")
    print("    %s" % (", ".join(uncategorized) or "（无）"))
    if uncategorized:
        problems.append("有 %d 个文件完全没被索引到: %s"
                        % (len(uncategorized), ", ".join(uncategorized)))

    # ---------- [5] 表格内逐行 (文件, 行数) ----------
    print("\n【5】索引表格内行数逐项核对")
    bad_rows = []
    for line in idx.split("\n"):
        mm = re.match(r"\|\s*`([^`]+\.py)`\s*\|\s*(\d+)\s*\|", line)
        if not mm:
            continue
        name, claimed = mm.group(1), int(mm.group(2))
        if name in lines_of and lines_of[name] != claimed:
            bad_rows.append((name, claimed, lines_of[name]))
    if bad_rows:
        for name, claimed, actual in bad_rows:
            print("    ✗ %-32s 声称 %4d 行，实测 %4d 行" % (name, claimed, actual))
        problems.append("表格内有 %d 行行数不符" % len(bad_rows))
    else:
        print("    全部一致 ✓")

    # ---------- 结论 ----------
    print("\n" + "=" * 68)
    if problems:
        print("✗ 发现 %d 处不一致：" % len(problems))
        for p in problems:
            print("   -", p)
        if args.fix_hint:
            print("\n可直接粘贴回 INDEX.md 的修正值：")
            print("  头部    : **%d 个 `.py`**（%s 行 / ~%d KB）"
                  % (n_file, format(n_line, ","), n_byte // 1024))
            print("  D 节    : 共 %d 个 / %d 行 / %.1f KB"
                  % (len(orphans), tl, tb / 1024))
            for name, _, actual in bad_rows:
                print("  行数    : %-32s %d" % (name, actual))
        return 1
    print("✓ INDEX.md 与磁盘完全一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
