"""项目纪律审计 —— 把"靠自觉"的规矩变成"跑一下就知道"。

动机
----
本项目已经踩过三次"规矩写在文档里但没人守"的坑：

1. CLAUDE.md 长到 70 KB（≈15–21K tokens）才被发现 —— 它每个会话开头被注入。
2. "PERFORMANCE.md 只能用二进制编辑"传了几轮，实测是**伪规矩**，
   而真正的病灶（934 个裸 CR）一直没人拆。
3. tools/INDEX.md 手写的数字不到一天就漂了（文件数、行数、依赖、孤儿集合）。

共同点：**没有自动化检查**。文档里写的规矩，人不会自觉遵守，
直到它造成损失。所以本工具把纪律检查做成一个可执行的命令。

检查项
------
    [1]  硬编码绝对路径（D:\\ / C:\\Users\\）
    [2]  裸 except / 静默异常吞噬（except: pass 或 except Exception: pass）
    [3]  未使用的 import（尊重 `# noqa: F401` 与 `__all__` 这两种有意 re-export）
    [4]  未门控的 print（产品代码里不受 debug 开关保护的打印）
    [5]  TODO / FIXME / HACK / XXX 残留
    [6]  文件名拼写损坏（含 `?` 等异常字符）
    [7]  版本号一致性（engine_config.__version__ ↔ 最新 git tag）
    [8]  文档里指向仓库内文件的引用是否悬空
    [9]  文档裸 CR（会让 git 判为二进制 + CommonMark 渲染错误）
    [10] CLAUDE.md 注入预算（12 KB 硬上限）
    [11] 未跟踪的产物文件（该进 .gitignore 或该提交）
    [12] 测试纪律：依赖真实视频/真值的测试必须有 skip 保护

用法
----
    python tools/_probe_discipline_audit.py            # 全量检查
    python tools/_probe_discipline_audit.py --only 1,3,7
    python tools/_probe_discipline_audit.py --strict   # 警告也当失败

退出码 0 = 通过；1 = 有违规。（CI 可直接用；关键项另有单测守护）
"""

from __future__ import annotations
import os

import argparse
import ast
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# 产品代码（不含 tools/ 探针与 tests/）
PRODUCT = ["engine_config.py", "gpu_setup.py", "hybrid_decode.py", "ocr_native.py",
           "ocr_trt.py", "segmentation.py", "video_utils.py"]
PRODUCT += [os.path.join("video_ocr_engine", f)
            for f in sorted(os.listdir(os.path.join(ROOT, "video_ocr_engine")))
            if f.endswith(".py")]
ALL_PY = PRODUCT + [os.path.join("tools", f)
                    for f in sorted(os.listdir(HERE)) if f.endswith(".py")]
ALL_PY += [os.path.join("tests", f)
           for f in sorted(os.listdir(os.path.join(ROOT, "tests"))) if f.endswith(".py")]
for sub in ("api", "decode", "pipeline", "segment", "utils"):
    d = os.path.join(ROOT, "tests", sub)
    if os.path.isdir(d):
        ALL_PY += [os.path.join("tests", sub, f)
                   for f in sorted(os.listdir(d)) if f.endswith(".py")]

DOCS = ["README.md", "CLAUDE.md", "docs/PERFORMANCE.md", "docs/ARCHIVE.md",
        "docs/DECISIONS.md", "docs/DEPENDENCIES.md", "tools/INDEX.md"]

CLAUDE_MD_MAX_BYTES = 12 * 1024

errors: list[str] = []
warns: list[str] = []


def fail(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warns.append(msg)


def rp(rel: str) -> str:
    return os.path.join(ROOT, rel)


def read(rel: str) -> str:
    with open(rp(rel), encoding="utf-8") as f:
        return f.read()


def check_hardcoded_paths() -> None:
    """[1] 硬编码绝对路径。"""
    pat = re.compile(r'["\'][A-Za-z]:[\\/](?:Repo|Users)[\\/]')
    hits = []
    for rel in ALL_PY:
        for i, line in enumerate(read(rel).split("\n"), 1):
            if pat.search(line):
                hits.append((rel, i, line.strip()[:80]))
    print("    命中 %d 处" % len(hits))
    for rel, ln, txt in hits[:12]:
        print("      %-46s L%-5d %s" % (rel, ln, txt))
    if hits:
        # 探针是调查工具，允许硬编码（一次性使用）；产品代码与测试不允许。
        prod = [h for h in hits if not h[0].startswith("tools" + os.sep)]
        if prod:
            fail("产品/测试代码含硬编码绝对路径 %d 处（tools/ 探针豁免）" % len(prod))
        else:
            warn("tools/ 探针含硬编码绝对路径 %d 处（豁免，但换机器会坏）" % len(hits))


BASELINE = os.path.join(HERE, "_discipline_baseline.json")


def _load_baseline() -> set[str]:
    import json
    if not os.path.isfile(BASELINE):
        return set()
    with open(BASELINE, encoding="utf-8") as f:
        return set(json.load(f).get("except_pass", []))


def check_bare_except() -> None:
    """[2] 裸 except 与静默异常吞噬。

    判据（三者满足其一即合规）：
      1. `pass` 行带行尾注释（`pass  # 为何可忽略`）；
      2. `except` 上一行是注释（解释性说明）；
      3. 登记在 baseline 里（**存量豁免、增量严格**）。

    为什么要 baseline：全项目有 61 处 `except … pass`，绝大多数在清理/释放
    路径（cudaFree / release / rollback），逐条改要动产品代码行为、风险不小。
    但"以后别再新增静默吞噬"必须立刻生效 —— 所以存量登记豁免，新增一律报。
    改到某个文件时顺手补注释，补了之后它自然从 baseline 里脱出。
    """
    base = _load_baseline()
    bare, swallow, exempted = [], [], 0
    for rel in ALL_PY:
        src = read(rel)
        lines = src.split("\n")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.ExceptHandler):
                continue
            if n.type is None:
                bare.append((rel, n.lineno))
                continue
            if len(n.body) != 1 or not isinstance(n.body[0], ast.Pass):
                continue
            key = "%s:%d" % (rel.replace(os.sep, "/"), n.lineno)
            # 1) pass 行自带注释
            pl = lines[n.body[0].lineno - 1] if n.body[0].lineno <= len(lines) else ""
            if "#" in pl:
                exempted += 1
                continue
            # 2) except 上一行是注释
            if n.lineno >= 2 and lines[n.lineno - 2].strip().startswith("#"):
                exempted += 1
                continue
            # 3) baseline 存量豁免
            if key in base:
                exempted += 1
                continue
            swallow.append((rel, n.lineno, ast.unparse(n.type)))
    print("    裸 except %d 处；静默吞噬 %d 处（另有 %d 处已带注释或登记豁免）"
          % (len(bare), len(swallow), exempted))
    for rel, ln in bare[:10]:
        print("      裸 %-42s L%-5d" % (rel, ln))
    for rel, ln, t in swallow[:15]:
        print("      吞 %-42s L%-5d except %s" % (rel, ln, t))
    if bare:
        fail("裸 except %d 处 —— 会连 KeyboardInterrupt / SystemExit 一起吞" % len(bare))
    if swallow:
        fail("新增静默吞噬 %d 处：异常消失得无声无息。要么记 logger.debug，"
             "要么 `pass  # 说明为何可忽略`；确属存量可 --update-baseline 登记"
             % len(swallow))


def check_unused_imports() -> None:
    """[3] 未使用的 import（尊重 noqa F401 与 __all__）。"""
    unused = []
    for rel in PRODUCT:
        src = read(rel)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        exported: set[str] = set()
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(
                getattr(t, "id", "") == "__all__" for t in n.targets
            ):
                try:
                    exported |= set(ast.literal_eval(n.value))
                except Exception:
                    pass
        for n in ast.walk(tree):
            if not isinstance(n, (ast.Import, ast.ImportFrom)):
                continue
            # `from __future__ import annotations` 是编译期指令，不是"用了的名字"
            if isinstance(n, ast.ImportFrom) and (n.module or "") == "__future__":
                continue
            # 有 noqa: F401 视为有意 re-export，跳过
            seg = src.split("\n")[n.lineno - 1]
            if "noqa" in seg.lower() and "F401" in seg:
                continue
            # 多行 import：noqa 可能在首行
            if n.lineno >= 2 and "noqa" in src.split("\n")[n.lineno - 2].lower() \
                    and "F401" in src.split("\n")[n.lineno - 2]:
                continue
            for alias in n.names:
                name = alias.asname or alias.name.split(".")[0]
                if name == "*" or name in exported:
                    continue
                if len(re.findall(r"\b%s\b" % re.escape(name), src)) <= 1:
                    unused.append((rel, n.lineno, name))
    print("    未使用 import %d 处" % len(unused))
    for rel, ln, nm in unused:
        print("      %-46s L%-5d %s" % (rel, ln, nm))
    if unused:
        fail("产品代码未使用 import %d 处（有意 re-export 请加 `# noqa: F401`）" % len(unused))


def check_unguarded_print() -> None:
    """[4] 未门控的 print（产品代码）。

    用 AST 找真正的 `print(...)` 调用 —— 文本匹配会把 **docstring 里的用法示例**
    也算进去（`video_ocr_engine/__init__.py` 的 docstring 就有一行
    `print(seg.text, ...)`，那是文档不是代码）。

    判据：print 的任一祖先 `if` 条件含 env_bool / _probe / debug 等门控关键字，
    或所在函数/方法名自带 probe / debug / dump 语义，即视为已门控。
    """
    gate = re.compile(r"env_bool|_probe|debug|DEBUG|verbose|VERBOSE|__main__")
    hits = []
    for rel in PRODUCT:
        src = read(rel)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        parents: dict[int, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[id(child)] = node
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                continue
            guarded = False
            cur: ast.AST | None = node
            while cur is not None:
                if isinstance(cur, ast.If):
                    try:
                        if gate.search(ast.unparse(cur.test)):
                            guarded = True
                            break
                    except Exception:
                        # unparse 在极老语法上可能失败；判不出来就当没门控，
                        # 继续上溯，不影响正确性
                        pass
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)) and re.search(
                        r"probe|debug|dump|cli|main", cur.name, re.I):
                    guarded = True
                    break
                cur = parents.get(id(cur))
            if not guarded:
                hits.append((rel, node.lineno,
                             src.split("\n")[node.lineno - 1].strip()[:70]))
    print("    未门控 print %d 处" % len(hits))
    for rel, ln, txt in hits[:10]:
        print("      %-46s L%-5d %s" % (rel, ln, txt))
    if hits:
        warn("产品代码有 %d 处 print 未见 debug 门控（确认是日志就该走 logging）" % len(hits))


def check_todo() -> None:
    """[5] TODO / FIXME / HACK / XXX 残留。"""
    pat = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b")
    # 本脚本源码里必然含 "TODO|FIXME" 这些词（检查项本身就要写它们），
    # 自己扫自己会永远命中 —— 排除。
    SELF = os.path.join("tools", "_probe_discipline_audit.py")
    hits = []
    for rel in ALL_PY + DOCS:
        p = rp(rel)
        if rel == SELF or not os.path.isfile(p):
            continue
        for i, line in enumerate(read(rel).split("\n"), 1):
            if pat.search(line):
                hits.append((rel, i, line.strip()[:70]))
    print("    命中 %d 处" % len(hits))
    for rel, ln, txt in hits[:8]:
        print("      %-46s L%-5d %s" % (rel, ln, txt))
    if hits:
        warn("TODO/FIXME/HACK %d 处（不是错误，但别让它烂在那里）" % len(hits))


def check_filename_corruption() -> None:
    """[6] 文件名拼写损坏（如 `_probe_sk?ip_frame.py`）。"""
    bad = []
    for rel in ALL_PY:
        base = os.path.basename(rel)
        if re.search(r"[?*<>\"|]", base):
            bad.append(base)
    print("    异常文件名 %d 个" % len(bad))
    for b in bad:
        print("      %s" % b)
    if bad:
        fail("文件名含非法字符 %d 个" % len(bad))

    # 顺带：文档里引用的仓库内 .py 是否存在。
    # 注意排除**历史语境**：文档经常要写"已删除的 tests/test_dual_pipeline.py"、
    # "当时的维护工具 tools/trim_perf_doc.py"，这类提到不存在的文件是对的，
    # 不是悬空引用。只有"当成现役文件在指路"才算违规。
    HIST = re.compile(r"已删除|不存在|已移除|已废弃|历史|当时|原\s|建议|曾|废弃|遗留|归档|没有")
    # ARCHIVE.md / DECISIONS.md 整体就是历史档案，里面的路径指的是"当时那个文件"，
    # 后来被删/被移是常态 —— 不做存在性要求。只有现役文档必须指得通。
    LIVE_DOCS = ["README.md", "CLAUDE.md", "docs/PERFORMANCE.md", "tools/INDEX.md"]
    ghost = []
    for rel in LIVE_DOCS:
        p = rp(rel)
        if not os.path.isfile(p):
            continue
        for i, line in enumerate(read(rel).split("\n"), 1):
            for m in re.finditer(r"`((?:tools|tests|docs)/[A-Za-z0-9_./-]+\.py)`", line):
                t = m.group(1)
                if "*" in t or os.path.isfile(rp(t)):
                    continue
                if HIST.search(line):
                    continue
                ghost.append((rel, i, t))
    if ghost:
        for rel, ln, t in ghost[:8]:
            print("      文档 %s L%d 指向不存在的 %s" % (rel, ln, t))
        fail("文档引用了 %d 个不存在的 .py（历史语境已排除）" % len(ghost))


def check_version() -> None:
    """[7] 版本号一致性：engine_config.__version__ ↔ 最新 git tag。"""
    src = read("engine_config.py")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', src, re.M)
    if not m:
        fail("engine_config.py 里找不到 __version__")
        return
    ver = m.group(1)
    try:
        tags = subprocess.run(["git", "tag", "--sort=-v:refname"],
                              cwd=ROOT, capture_output=True, text=True).stdout.split()
    except Exception as e:
        warn("读 git tag 失败：%s" % e)
        return
    latest = tags[0].lstrip("v") if tags else None
    print("    engine_config.__version__ = %s ；最新 git tag = %s" % (ver, latest))
    if latest is None:
        warn("仓库还没有任何 git tag")
    elif ver != latest:
        fail("版本号不一致：engine_config=%s，最新 tag=%s（铁律：改版本必须打同名 tag）"
             % (ver, latest))


def check_doc_refs() -> None:
    """[8] 文档里指向仓库内 md 的引用是否悬空。"""
    dangling = []
    for rel in DOCS:
        p = rp(rel)
        if not os.path.isfile(p):
            continue
        for m in re.finditer(r"\]\(([A-Za-z0-9_./-]+\.md)\)", read(rel)):
            t = m.group(1)
            target = rp(os.path.join(os.path.dirname(rel), t))
            if not os.path.isfile(target):
                dangling.append((rel, t))
    print("    悬空 md 链接 %d 处" % len(dangling))
    for rel, t in dangling[:8]:
        print("      %s → %s" % (rel, t))
    if dangling:
        fail("文档有 %d 处指向不存在文件的链接" % len(dangling))


def check_bare_cr() -> None:
    """[9] 文档裸 CR。"""
    bad = []
    for rel in DOCS:
        p = rp(rel)
        if not os.path.isfile(p):
            continue
        with open(p, "rb") as f:
            d = f.read()
        n = len(re.findall(rb"\r(?!\n)", d))
        if n:
            bad.append((rel, n))
    print("    含裸 CR 的文档 %d 个" % len(bad))
    for rel, n in bad:
        print("      %-24s %d 个" % (rel, n))
    if bad:
        fail("文档含裸 CR（git 会判为二进制、CommonMark 渲染错误）")


def check_claude_budget() -> None:
    """[10] CLAUDE.md 注入预算。"""
    p = rp("CLAUDE.md")
    if not os.path.isfile(p):
        return
    size = os.path.getsize(p)
    print("    CLAUDE.md %d 字节 / 上限 %d" % (size, CLAUDE_MD_MAX_BYTES))
    if size > CLAUDE_MD_MAX_BYTES:
        fail("CLAUDE.md %d 字节，超 12 KB 注入预算" % size)


def check_untracked() -> None:
    """[11] 未跟踪文件（该进 .gitignore 或该提交）。"""
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             cwd=ROOT, capture_output=True, text=True).stdout
    except Exception:
        return
    unt = [l[3:] for l in out.split("\n") if l.startswith("?? ")]
    print("    未跟踪条目 %d 个" % len(unt))
    for u in unt:
        print("      %s" % u)
    # .claude/ 与 .workbuddy/ 是会话态，按惯例豁免
    real = [u for u in unt if u not in (".claude/", ".workbuddy/")]
    if real:
        warn("有 %d 个未跟踪条目长期挂着（要么提交，要么进 .gitignore）" % len(real))


def check_test_discipline() -> None:
    """[12] 依赖真实视频/真值的测试必须有 skip 保护。"""
    risky = []
    for rel in [r for r in ALL_PY if r.startswith(("tests" + os.sep, "tests/"))]:
        src = read(rel)
        needs = bool(re.search(r"racelog_test|ground_truth|D:\\Videos|batch_test", src))
        if not needs:
            continue
        if "pytest.skip" in src or "pytest.mark.skipif" in src or "skipif" in src:
            continue
        risky.append(rel)
    print("    依赖真实资源但无 skip 保护的测试 %d 个" % len(risky))
    for r in risky:
        print("      %s" % r)
    if risky:
        warn("这些测试在没视频的机器上会直接失败（CI / 新机器）")


CHECKS = {
    1: ("硬编码绝对路径", check_hardcoded_paths),
    2: ("裸 except / 静默吞噬", check_bare_except),
    3: ("未使用的 import", check_unused_imports),
    4: ("未门控的 print", check_unguarded_print),
    5: ("TODO / FIXME / HACK", check_todo),
    6: ("文件名 / 引用损坏", check_filename_corruption),
    7: ("版本号一致性", check_version),
    8: ("文档 md 链接悬空", check_doc_refs),
    9: ("文档裸 CR", check_bare_cr),
    10: ("CLAUDE.md 注入预算", check_claude_budget),
    11: ("未跟踪文件", check_untracked),
    12: ("测试纪律（真实资源保护）", check_test_discipline),
}


def update_baseline() -> None:
    """把当前所有 `except … pass` 登记为存量豁免。

    只在"全绿之后重新打底"或"接手一批历史代码"时用。之后新增的静默吞噬
    不在 baseline 里，审计会报。
    """
    import json
    items = []
    for rel in ALL_PY:
        src = read(rel)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.ExceptHandler) or n.type is None:
                continue
            if len(n.body) == 1 and isinstance(n.body[0], ast.Pass):
                items.append("%s:%d" % (rel.replace(os.sep, "/"), n.lineno))
    with open(BASELINE, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"except_pass": sorted(items)}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("已登记 %d 处存量 `except … pass` → %s" % (len(items), BASELINE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只跑指定检查，逗号分隔，如 1,3,7")
    ap.add_argument("--strict", action="store_true", help="警告也当失败")
    ap.add_argument("--update-baseline", action="store_true",
                    help="把当前所有 except…pass 登记为存量豁免后退出")
    args = ap.parse_args()

    if args.update_baseline:
        update_baseline()
        return 0

    todo = [int(x) for x in args.only.split(",")] if args.only else sorted(CHECKS)

    print("=" * 70)
    print("项目纪律审计（%d 项）" % len(todo))
    print("=" * 70)
    for k in todo:
        name, fn = CHECKS[k]
        print("\n[%2d] %s" % (k, name))
        try:
            fn()
        except Exception as e:                      # 单项失败不应中断全量
            print("    检查本身异常：%r" % (e,))
            warn("第 %d 项检查抛异常：%r" % (k, e))

    print("\n" + "=" * 70)
    for w in warns:
        print("⚠️  %s" % w)
    for e in errors:
        print("✗  %s" % e)
    if errors:
        print("\n✗ 违规 %d 项，警告 %d 项" % (len(errors), len(warns)))
        return 1
    if warns and args.strict:
        print("\n✗ --strict：警告也当失败（%d 项）" % len(warns))
        return 1
    print("✓ 全部通过（警告 %d 项）" % len(warns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
