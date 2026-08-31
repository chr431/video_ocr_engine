"""一次性迁移：把 tools/ 探针里不可移植的路径改成可移植写法。

三个阶段针对**同一文件里不同性质的代码段**：

    阶段 1  模块自身代码   sys.path.insert(0, r"D:\\Repo\\...")  →  __file__ 推导
    阶段 2  测试视频常量   GT = Path(r"D:\\Videos\\...")        →  环境变量可覆盖
    阶段 3  子进程模板     WORKER = r\"\"\"...\"\"\" 里的路径      →  环境变量传递

⛔ 贯穿全程的陷阱：`WORKER = r\"\"\"...\"\"\"` 子进程模板
--------------------------------------------------------
模板里的代码会被 `python -c` 执行，于是：

1. **`-c` 模式下 `__file__` 未定义** —— 模板里不能用 `os.path.dirname(
   os.path.abspath(__file__))` 推导路径。第一版把模板里的硬编码也换成
   __file__ 推导，14 个模板在子进程里直接 NameError。
2. **模板内部的行也算"顶层 import"** —— 注入 `import os` / 常量声明时若按
   "最后一条 import 之后"定位，会算到模板里的 `from pathlib import Path`，
   把声明插进模板字符串内部，父进程反而拿不到。第二版就栽在这里。

对策：先算模板区间，其后**所有按行定位的操作只在非模板行里选点**。
模板内的路径一律走 `os.environ["PROBE_ROOT"]`，由父进程注入。

实测规模（2026-08-31）
--------------------
    阶段 1  24 处 / 23 个文件（硬编码仓库路径 15 + os.getcwd() 9）
    阶段 2  28 处 / 15 个文件
    阶段 3  14 处 / 14 个文件（_probe_cpu_onnx 除外，见下）

已知例外
--------
`_probe_cpu_onnx.py` 的 WORKER **必须插 cwd 而非固定 ROOT**：它做新旧版本
A/B 对比，子进程 cwd 被设成旧版本的 worktree，插 ROOT 会变成"新代码 vs
新代码"而静默失效。该模板保留 `os.getcwd()`。

安全性
------
- 只做字面量替换，不碰其它逻辑。
- **预演也编译校验**（写临时文件 py_compile），不过就中止。
- `--apply` 时先写盘再编译（有一版只 compile 不 write，报"已修改 30 个
  文件"而 `git diff` 是空的）。

用法
----
    python tools/_fix_probe_paths.py            # 预演
    python tools/_fix_probe_paths.py --apply    # 写盘
"""

from __future__ import annotations

import argparse
import os
import py_compile
import re
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

TOOLS = os.path.dirname(os.path.abspath(__file__))

Q = chr(34)
BS = chr(92)
HARD_TOOLS = 'sys.path.insert(0, r%sD:%sRepo%svideo_ocr_engine%stools%s)' % (Q, BS, BS, BS, Q)
HARD_ROOT = 'sys.path.insert(0, r%sD:%sRepo%svideo_ocr_engine%s)' % (Q, BS, BS, Q)
CWD = "sys.path.insert(0, os.getcwd())"

DEF_HERE = "HERE = os.path.dirname(os.path.abspath(__file__))"
DEF_ROOT = "ROOT = os.path.dirname(HERE)"

DV = "D:%sVideos" % BS
DECL_VIDEO = '_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"%s%sracelog_test"))' % (DV, BS)
DECL_BATCH = '_BATCH_DIR = Path(os.environ.get("RACELOG_BATCH_DIR", r"%s%sbatch_test"))' % (DV, BS)

RULES = [
    ('Path(r"%s%sracelog_test%sground_truth_csv")' % (DV, BS, BS),
     '_VIDEO_DIR / "ground_truth_csv"', "video"),
    ('Path(r"%s%sracelog_test")' % (DV, BS), '_VIDEO_DIR', "video"),
    ('r"%s%sracelog_test%stest5.mp4"' % (DV, BS, BS),
     'str(_VIDEO_DIR / "test5.mp4")', "video"),
    ('r"%s%sracelog_test%stest6.mp4"' % (DV, BS, BS),
     'str(_VIDEO_DIR / "test6.mp4")', "video"),
    ('Path(r"%s%sbatch_test")' % (DV, BS), '_BATCH_DIR', "batch"),
    ('r"%s%sbatch_test%s新三国01.mkv"' % (DV, BS, BS),
     'str(_BATCH_DIR / "新三国01.mkv")', "batch"),
]

WORKER_INSERT = 'sys.path.insert(0, os.environ["PROBE_ROOT"])'
# A/B 对比必须插 cwd（子进程 cwd = 旧版本 worktree），不能插固定 ROOT
KEEP_CWD = {"_probe_cpu_onnx.py"}


# ───────────────────────── 模板区间基础设施 ─────────────────────────

def template_spans(src: str) -> list[tuple[int, int]]:
    """子进程模板 `WORKER = r\"\"\"...\"\"\"` 的字符区间。

    ⚠️ 必须同时支持 `\"\"\"` 和 `'''`：项目里两种都有（`_probe_slf_vis.py` 用的
    就是 `r'''`）。只认双引号时该模板会被当成普通代码，于是常量声明被插进
    模板字符串内部、`sys.path.insert` 被换成 __file__ 推导 —— 而模板是
    `python -c` 执行的，__file__ 未定义 → 运行时 NameError（第四版踩的）。
    """
    spans = []
    for m in re.finditer(r"(?:WORKER|worker|CHILD|SCRIPT)\s*=\s*r?(\"\"\"|''')", src):
        quote = m.group(1)
        close = src.find(quote, m.end())
        if close < 0:
            continue
        spans.append((m.start(), close + len(quote)))
    return spans


def in_spans(off: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= off < b for a, b in spans)


def line_starts(src: str) -> list[int]:
    offs = [0]
    for i, ch in enumerate(src):
        if ch == "\n":
            offs.append(i + 1)
    return offs


def last_import_before(src: str, limit_off: int) -> int:
    """返回 `limit_off` **之前**最后一条非模板顶层 import 的行号；没有则 -1。

    为什么是"之前"而不是"文件里最后一条"：有些探针是**扁平脚本**，
    中间还会再 import（如 `_probe_slf_adjudicate.py` 在 L117 才
    `from PIL import ...`），而常量在 L15 就已经用上了。按"文件里最后一条
    import"定位会把声明插到使用点之后 → NameError（第五版踩的）。
    """
    spans = template_spans(src)
    starts = line_starts(src)
    last = -1
    for i, off in enumerate(starts):
        if off >= limit_off:
            break
        if in_spans(off, spans):          # 模板里的 import 不算
            continue
        line = src[off:off + 120]
        if line.startswith("import ") or line.startswith("from "):
            last = i
    return last


def first_use_offset(src: str, names: list[str]) -> int:
    """names 中任一名字**首次出现**（模板外）的字符偏移；都没出现返回 len(src)。"""
    spans = template_spans(src)
    best = len(src)
    for name in names:
        for m in re.finditer(r"\b%s\b" % re.escape(name), src):
            if not in_spans(m.start(), spans):
                best = min(best, m.start())
                break
    return best


def insert_before_first_use(src: str, names: list[str], block: list[str]) -> str:
    """把 block 插到 names 首次使用之前、且尽量靠后的 import 之后。"""
    limit = first_use_offset(src, names)
    if limit >= len(src):                 # 没有使用点：退化为插到最后一条 import 后
        limit = len(src) + 1
    li = last_import_before(src, limit)
    lines = src.split("\n")
    if li < 0:
        lines[0:0] = block
    else:
        lines[li + 1:li + 1] = block
    return "\n".join(lines)


def insert_after_imports(src: str, block: list[str]) -> str:
    """兼容旧调用：插到最后一条非模板 import 之后。"""
    li = last_import_before(src, len(src) + 1)
    lines = src.split("\n")
    if li < 0:
        return "\n".join(block) + "\n" + src
    lines[li + 1:li + 1] = block
    return "\n".join(lines)


def ensure_import(src: str, stmt: str) -> str:
    """确保 import 存在；缺失时插在最后一条非模板 import 之后。"""
    if re.search(r"^%s$" % re.escape(stmt), src, re.M):
        return src
    return insert_after_imports(src, [stmt])


def write_and_check(path: str, src: str, apply: bool) -> None:
    if apply:
        target = path
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
    else:
        fd, target = tempfile.mkstemp(suffix=".py", prefix="_fixcheck_")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
    try:
        py_compile.compile(target, doraise=True, cfile=target + "c")
    except py_compile.PyCompileError as e:
        raise SystemExit("\n✗ 生成结果无法编译，已中止：\n%s" % e)
    finally:
        try:
            os.unlink(target + "c")
        except OSError:
            # 清理路径：编译产物可能本来就没生成，删不掉无所谓
            pass
        if not apply:
            try:
                os.unlink(target)
            except OSError:
                # 同上：临时文件删不掉不影响校验结果
                pass


# ───────────────────────── 阶段 1：模块自身代码 ─────────────────────────

def fix_module_paths(path: str, apply: bool) -> tuple[bool, list[str]]:
    src = open(path, encoding="utf-8").read()
    orig = src
    notes: list[str] = []
    spans = template_spans(src)

    def repl(seg: str) -> str:
        nonlocal notes
        for old, label in ((HARD_TOOLS, "HERE"), (HARD_ROOT, "ROOT"), (CWD, "ROOT")):
            c = seg.count(old)
            if c:
                seg = seg.replace(old, "sys.path.insert(0, %s)" % label)
                notes.append("硬编码/%s ×%d → %s" % (
                    "cwd" if old is CWD else "路径", c, label))
        return seg

    pieces, prev = [], 0
    for a, b in spans:
        pieces.append(repl(src[prev:a]))
        pieces.append(src[a:b])          # 模板段原样，阶段 3 处理
        prev = b
    pieces.append(repl(src[prev:]))
    src = "".join(pieces)

    if src == orig:
        return False, []

    need_root = "sys.path.insert(0, ROOT)" in src and "ROOT =" not in src
    need_here = ("sys.path.insert(0, HERE)" in src or need_root) and "HERE =" not in src
    if need_here or need_root:
        lines = src.split("\n")
        idx = next(i for i, l in enumerate(lines) if "sys.path.insert" in l)
        block = []
        if need_here:
            block.append(DEF_HERE)
        if need_root:
            block.append(DEF_ROOT)
        lines[idx:idx] = block
        src = "\n".join(lines)
        notes.append("插入 %s" % " + ".join(block))

    if "os.path" in src:
        src = ensure_import(src, "import os")

    write_and_check(path, src, apply)
    return True, notes


# ───────────────────────── 阶段 2：测试视频常量 ─────────────────────────

def fix_video_paths(path: str, apply: bool) -> tuple[bool, list[str]]:
    src = open(path, encoding="utf-8").read()
    orig = src
    notes: list[str] = []
    need: set[str] = set()

    spans = template_spans(src)
    for old, new, kind in RULES:
        c = src.count(old)
        if not c:
            continue
        # 模板内的出现留给阶段 3 的语义（那里走 PROBE_ROOT + 子进程自己的常量）
        keep = []
        for m in re.finditer(re.escape(old), src):
            if not in_spans(m.start(), spans):
                keep.append(m.start())
        if not keep:
            continue
        src = src.replace(old, new)
        need.add(kind)
        notes.append("%s ×%d → 环境变量可覆盖" % (
            old.rsplit(BS, 1)[-1].rstrip(Q) or "batch_test", len(keep)))

    if src == orig:
        return False, []

    block = []
    if "batch" in need:
        block.append(DECL_BATCH)
    if "video" in need:
        block.append(DECL_VIDEO)
    block.append("")
    src = insert_before_first_use(src, ["_VIDEO_DIR", "_BATCH_DIR"], block)

    if "os.environ" in src:
        src = ensure_import(src, "import os")
    if "Path(" in src and "from pathlib import" not in src:
        src = ensure_import(src, "from pathlib import Path")

    write_and_check(path, src, apply)
    return True, notes


# ───────────────────────── 阶段 3：子进程模板 ─────────────────────────

def fix_worker_templates(path: str, apply: bool) -> tuple[bool, list[str]]:
    src = open(path, encoding="utf-8").read()
    notes: list[str] = []
    spans = template_spans(src)
    if not spans:
        return False, []

    if os.path.basename(path) in KEEP_CWD:
        notes.append("跳过（A/B 对比需插 cwd）")
        return False, notes

    out, prev, touched = [], 0, False
    for a, b in spans:
        blk = src[a:b]
        n = blk.count(HARD_ROOT) + blk.count(HARD_TOOLS) + blk.count(CWD)
        if n:
            blk = blk.replace(HARD_TOOLS, WORKER_INSERT)
            blk = blk.replace(HARD_ROOT, WORKER_INSERT)
            blk = blk.replace(CWD, WORKER_INSERT)
            touched = True
            notes.append("WORKER 模板路径 ×%d → PROBE_ROOT" % n)
        out.append(src[prev:a])
        out.append(blk)
        prev = b
    out.append(src[prev:])
    src = "".join(out)

    if not touched:
        return False, []

    if 'os.environ["PROBE_ROOT"] = ROOT' not in src:
        # ⚠️ 必须**一次**插入：分两次调用 insert_after_imports 会插到同一位置，
        # 后插的反而排在前面，变成先用 ROOT 后定义 → NameError（第三版踩的）。
        block = []
        if re.search(r"^ROOT\s*=", src, re.M) is None:
            block += [
                "",
                "# WORKER 以 `python -c` 执行，-c 下 __file__ 未定义，",
                "# 故本进程算好根目录后经 PROBE_ROOT 传给子进程。",
                "ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
            ]
        block.append('os.environ["PROBE_ROOT"] = ROOT  '
                     "# 供 `python -c` 的 WORKER 子进程使用")
        # 必须在**首个 subprocess.run / WORKER 使用之前**注入，否则子进程
        # 启动时环境变量还没设（扁平脚本里 subprocess 可能出现在文件中部）。
        limit = first_use_offset(src, ["WORKER", "subprocess.run"])
        li = last_import_before(src, limit)
        lines = src.split("\n")
        if li < 0:
            lines[0:0] = block
        else:
            lines[li + 1:li + 1] = block
        src = "\n".join(lines)
        notes.append("父进程注入 PROBE_ROOT")
    if "import os" not in src:
        src = ensure_import(src, "import os")

    write_and_check(path, src, apply)
    return True, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写盘（默认预演）")
    args = ap.parse_args()

    changed: list[str] = []
    for name in sorted(os.listdir(TOOLS)):
        if not name.endswith(".py") or name.startswith("_fix_"):
            continue
        p = os.path.join(TOOLS, name)
        notes: list[str] = []
        for fn in (fix_module_paths, fix_video_paths, fix_worker_templates):
            _, n = fn(p, args.apply)
            notes += n
        if not notes:
            continue
        changed.append(p)
        print("  %-38s %s" % (name, "；".join(notes)))

    if not args.apply:
        print("\n将修改 %d 个文件（预演+编译校验已过，未写盘。加 --apply 执行）"
              % len(changed))
        return 0

    print("\n已修改 %d 个文件，复核编译：" % len(changed))
    bad = []
    for p in changed:
        r = subprocess.run([sys.executable, "-m", "py_compile", p],
                           capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(p)
            print("  ✗ %s\n%s" % (os.path.basename(p), r.stderr[-400:]))
    if bad:
        print("\n✗ %d 个文件编译失败" % len(bad))
        return 1
    print("  ✓ %d 个文件全部编译通过" % len(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
