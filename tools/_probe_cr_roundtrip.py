"""裸 CR 往返保真性探针 —— 判定 docs/PERFORMANCE.md 是否真的"只能二进制改"。

调查动机
--------
项目 MEMORY.md 长期记着一条规矩：

    ⚠️ 禁止用 Python 文本模式（open(..., encoding=...)）读-改-写这个文件：
    通用换行会把 934 个裸 \r 转成 \n。改它只能用二进制模式。

但 Python 的 open() 有 newline 参数：默认的 newline=None 才开启"通用换行翻译"，
newline='' 与 newline='\\n' 都不翻译。所以"必须用二进制"可能是条伪规矩。
本探针用实测判定：

  1. 裸 CR 到底是"行尾"还是"行中残留"？（决定它该不该被翻译）
  2. 各 newline 模式的读-写往返是否 100% 保真？
  3. git 视角下，CRLF 行尾到底算不算"被改动"？（.gitattributes 是 * text=auto eol=lf）

用法
----
    python tools/_probe_cr_roundtrip.py                 # 默认测 docs/PERFORMANCE.md
    python tools/_probe_cr_roundtrip.py --file CLAUDE.md
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def classify(data: bytes) -> dict:
    """把裸 CR 按「后跟字节」分类：行尾 vs 行中残留。"""
    cnt = collections.Counter()
    for i, b in enumerate(data):
        if b == 0x0D:
            cnt[data[i + 1 : i + 2]] += 1
    return {
        "total": data.count(b"\r"),
        "crlf": cnt[b"\n"],           # 真正的 CRLF 行尾
        "cr_space": cnt[b" "],        # 行中残留：CR 后跟空格（markdown 硬换行）
        "cr_gt": cnt[b">"],           # 行中残留：CR 后跟 '>'（引用块续行）
        "cr_other": sum(v for k, v in cnt.items() if k not in (b"\n", b" ", b">")),
    }


def roundtrip(data: bytes, mode: str) -> tuple[bool, int, int]:
    """用给定 newline 模式做 读→写 往返，返回 (是否保真, 写回后字节数, 写回后 CR 数)。

    mode: 'none' | 'empty' | 'lf' | 'binary'
    """
    path = os.path.join(tempfile.gettempdir(), "_cr_rt_%s.md" % mode)
    with open(path, "wb") as f:
        f.write(data)

    if mode == "binary":
        with open(path, "rb") as f:
            buf = f.read()
        with open(path, "wb") as f:
            f.write(buf)
    else:
        nl = {"none": None, "empty": "", "lf": "\n"}[mode]
        # 读：显式 newline；写：也显式 newline，避免 os.linesep 干扰
        with open(path, "r", encoding="utf-8", newline=nl) as f:
            text = f.read()
        with open(path, "w", encoding="utf-8", newline=nl) as f:
            f.write(text)

    with open(path, "rb") as f:
        out = f.read()
    os.unlink(path)
    return out == data, len(out), out.count(b"\r")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join("docs", "PERFORMANCE.md"))
    args = ap.parse_args()

    path = args.file if os.path.isabs(args.file) else os.path.join(ROOT, args.file)
    data = open(path, "rb").read()

    print("=" * 68)
    print("裸 CR 往返保真性探针 —— %s" % args.file)
    print("=" * 68)
    print("全文 %d 字节 / %d 个 \\n" % (len(data), data.count(b"\n")))
    print()

    # ---------- 1. 裸 CR 分类 ----------
    k = classify(data)
    print("【1】裸 CR 分类（共 %d 个）" % k["total"])
    print("    后跟 \\n（真 CRLF 行尾）      %4d" % k["crlf"])
    print("    后跟空格（行中残留·硬换行）  %4d" % k["cr_space"])
    print("    后跟 '>'（行中残留·引用续行）%4d" % k["cr_gt"])
    print("    其他                        %4d" % k["cr_other"])
    inline = k["total"] - k["crlf"]
    print("    → 行中残留 %d 个（%.1f%%），行尾 %d 个（%.1f%%）"
          % (inline, 100.0 * inline / max(k["total"], 1),
             k["crlf"], 100.0 * k["crlf"] / max(k["total"], 1)))
    print()

    # ---------- 2. 各 newline 模式的往返 ----------
    print("【2】读-写往返保真性（读 newline == 写 newline）")
    print("    %-10s %-8s %-10s %-10s" % ("模式", "保真", "字节数", "CR 数"))
    results = {}
    for mode, label in (("none", "None(默认)"), ("empty", "''"),
                        ("lf", "'\\n'"), ("binary", "二进制")):
        ok, n, cr = roundtrip(data, mode)
        results[mode] = ok
        print("    %-10s %-8s %-10d %-10d" % (label, "是" if ok else "否", n, cr))
    print()

    # ---------- 3. 只做 CRLF→LF 规范化会丢多少 ----------
    norm = data.replace(b"\r\n", b"\n")
    print("【3】若只做 CRLF→LF 规范化（多数编辑器/工具的默认行为）")
    print("    字节 %d → %d（-%d）" % (len(data), len(norm), len(data) - len(norm)))
    print("    CR   %d → %d（-%d）" % (k["total"], norm.count(b"\r"),
                                       k["total"] - norm.count(b"\r")))
    print("    丢掉的 %d 个全是「行尾 CR」；行中残留 %d 个不受影响。"
          % (k["total"] - norm.count(b"\r"), inline))
    print()

    # ---------- 4. git 视角 ----------
    print("【4】git 视角：CRLF 行尾算不算「改动」？")
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    r = subprocess.run(["git", "check-attr", "text", "eol", "--", rel],
                       cwd=ROOT, capture_output=True, text=True)
    print("    git check-attr: %s" % r.stdout.strip().replace("\n", " | "))
    r2 = subprocess.run(["git", "diff", "--numstat", "--", rel],
                        cwd=ROOT, capture_output=True, text=True)
    print("    当前 git diff --numstat: %s" % (r2.stdout.strip() or "（干净）"))

    # 实测：把规范化后的内容写进临时索引，看 git 是否认为有变化
    tmp = os.path.join(tempfile.gettempdir(), "_cr_gitprobe.md")
    with open(tmp, "wb") as f:
        f.write(norm)
    r3 = subprocess.run(
        ["git", "hash-object", "--path", rel, "--", tmp],
        cwd=ROOT, capture_output=True, text=True)
    h_norm = r3.stdout.strip()
    r4 = subprocess.run(["git", "hash-object", "--path", rel, "--", path],
                        cwd=ROOT, capture_output=True, text=True)
    h_orig = r4.stdout.strip()
    print("    blob hash 原文   : %s" % h_orig[:16])
    print("    blob hash 规范化后: %s" % h_norm[:16])
    print("    → git 认为%s" % ("**有变化**（CRLF 未归一化，会被 diff 看到）"
                               if h_norm != h_orig else "**无变化**（归一化后一致）"))
    if os.path.exists(tmp):
        os.unlink(tmp)
    print()

    # ---------- 判定 ----------
    print("=" * 68)
    print("判定")
    print("=" * 68)
    safe_text = results.get("empty") and results.get("lf")
    if safe_text:
        print("✅ “必须二进制”是伪规矩。文本模式显式 newline='' 或 '\\n' 即 100% 保真。")
        print("   唯一致命的是默认 newline=None（通用换行翻译），避开即可。")
    else:
        print("❌ 文本模式确实不保真，继续用二进制。")
    if k["crlf"]:
        print("⚠️ 另有 %d 处真 CRLF 行尾。按 .gitattributes(* text=auto eol=lf)，"
              % k["crlf"])
        print("   这些本就该是 LF；规范化掉它们是正确的、且是 git 期望的。")
        print("   护栏数字应改为「行中 CR == %d」，而非「全文 CR == %d」。"
              % (inline, k["total"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
