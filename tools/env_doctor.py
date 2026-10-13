"""环境体检 / 部署同步 / 硬编码路径审计。

三类事故的根治工具：
  ① 找不到 ffmpeg      → `env_doctor.py` 打印每个依赖的**解析来源+版本+md5**，
                          找不到时列出全部已探测候选；
  ② DLL 非最新         → `--deploy` 把 fork 构建产物同步进 site-packages 并
                          校验 md5；`--check` 检出不一致即非零退出；
  ③ 硬编码路径         → `--audit` 扫描仓库内的绝对路径字面量（纪律项）。

用法：
  python tools/env_doctor.py            # 体检（依赖事实 + DLL 一致性）
  python tools/env_doctor.py --deploy   # 同步 fork 构建产物 → site-packages
  python tools/env_doctor.py --audit    # 硬编码路径审计（CI/纪律审计调用）
  python tools/env_doctor.py --json     # 机器可读（探针/CI 消费）
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from env_probe import (
    decord_fact,
    ffmpeg_fact,
    fork_build_dll,
    video_dir,
)

ROOT = Path(__file__).resolve().parents[1]

# 审计：绝对路径字面量（Windows 盘符 / UNC）。白名单按「文件 + 行内容」放行。
# ⚠️ 预筛字符类须写 [\\/]：[\/] 里反斜杠转义了正斜杠，类中只剩 `/`——
# 纯反斜杠路径的文件（bench_baseline.py 等）被预筛整文件跳过。
_ABS_PATH_RE_PLAIN = re.compile(r"[A-Za-z]:[\\/]")  # 文本级预筛（快判有无盘符）
# ⚠️ 负向后顾不得含引号：Python 路径字面量必然紧随引号，含引号的后顾会把
# 全部真实字面量排除、只匹配语法非法的裸路径（审计恒假绿）。自检样例见
# _PATH_REGEX_SELFTEST。
_ABS_PATH_RE = re.compile(r"""(?<![\w#])[A-Za-z]:[\\/](?:[\w.\- ]+[\\/])*[\w.\- ]+""")
#: 审计判据自检（audit 每次先跑）：已知坏的引号内字面量必须命中。
_PATH_REGEX_SELFTEST_BAD = r'VIDS = {"k": r"D:\Videos\racelog_test\x.mp4"}'
#: 无路径行不得命中（防正则过宽）。
_PATH_REGEX_SELFTEST_CLEAN = "out = compute(frames[0], n=3)"
_AUDIT_SKIP_DIRS = {
    ".git",
    "build",
    "dist",
    "__pycache__",
    ".venv",
    "node_modules",
    "bench",
    "docs",
}  # docs 内的叙事可提及路径
_AUDIT_EXTS = {".py", ".bat", ".ps1", ".cmd"}
# 允许的例外：环境事实源本身（它就是放默认路径/正则自检样例的地方）。
# 行级豁免：命中行行尾注释含 `hardcode-ok: <非空理由>` 才放行（内联豁免，
# 无登记账本；无理由的裸 hardcode-ok 不豁免——豁免必须携带依据）。
_HARDCODE_OK = re.compile(r"#\s*hardcode-ok:\s*\S")
_AUDIT_ALLOW_FILES = {"tools/env_probe.py", "tools/env_doctor.py"}


def _iter_audit_files():
    # rglob("*") 全树枚举 + 逐项 suffix/parts 过滤
    # （Windows 上 ≈0.28s）改 os.walk 剪枝（目录不入栈，≈0.01s）；行为
    # 等价（同跳过集、同扩展名集）。
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in _AUDIT_SKIP_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() in _AUDIT_EXTS:
                yield Path(dirpath) / f


# 硬编码路径审计的执行形态：正则自检（失明/过宽都拦）+ AST 级 docstring/
# env-defaults 豁免 + 行级内联豁免（`# hardcode-ok:` + 理由）。无存量登记
# 账本——审计是全量严格的；"存量豁免"叙事会让增量违规看起来被赦免。


def _env_default_lines(tree) -> set[int]:
    """AST 标记「env 带默认值」调用跨及的行（豁免形态：可被覆盖）。

    os.environ.get/setdefault 与 getenv 的路径实参即纪律允许的形态；
    按**单行**子串豁免会漏掉多行调用的字面量行（常在下一行）——这里按
    调用节点整段标行。
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = (
            getattr(f, "attr", "")
            if isinstance(f, ast.Attribute)
            else (f.id if isinstance(f, ast.Name) else "")
        )
        if name not in ("get", "setdefault", "getenv"):
            continue
        val = getattr(f, "value", None)
        is_env = (isinstance(val, ast.Attribute) and val.attr == "environ") or name == "getenv"
        if not is_env:
            continue
        end = getattr(node, "end_lineno", node.lineno)
        lines.update(range(node.lineno, end + 1))
    return lines


def _abs_path_selftest_failed() -> str | None:
    """正则判据自检：失明/过宽都返回错误描述。

    正则对已知坏例失明时整个审计项都是假绿灯（后顾含引号即此态）。
    """
    if not _ABS_PATH_RE.search(_PATH_REGEX_SELFTEST_BAD):
        return "env_doctor._ABS_PATH_RE 自检失败：引号内路径字面量未命中（正则失明）"
    if _ABS_PATH_RE.search(_PATH_REGEX_SELFTEST_CLEAN):
        return "env_doctor._ABS_PATH_RE 自检失败：无路径行被误命中（正则过宽）"
    return None


def _abs_path_hits_raw() -> list[tuple[str, int, str]]:
    """全部命中 (文件, 行号, 行文本)——未经 baseline 过滤。

    结构化返回：命中行内含 `D:` 盘符冒号，字符串拼接后再 split 会把
    文件键污染成行内容，计数一律按元组做。
    """
    bad: list[tuple[str, int, str]] = []
    for p in _iter_audit_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in _AUDIT_ALLOW_FILES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # 读不了就跳过（非源码）
            continue
        # 全文无盘符字面量时 AST 解析（≈0.2s/246 文件）
        # 纯属浪费——预筛不过直接跳 doc 行标记（正则逐行仍执行，漏报面=0：
        # 命中必然含盘符，预筛同字符集）。
        if not _ABS_PATH_RE_PLAIN.search(text):
            continue
        doc_lines: set[int] = set()
        env_lines: set[int] = set()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                body = getattr(node, "body", None)
                if isinstance(body, list) and body:
                    first = body[0]
                    if (
                        isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)
                    ):
                        end = getattr(first.value, "end_lineno", first.value.lineno)
                        doc_lines.update(range(first.value.lineno, end + 1))
            env_lines = _env_default_lines(tree)
        for i, line in enumerate(text.splitlines(), 1):
            if i in doc_lines:
                continue
            if line.lstrip().startswith("#"):
                continue  # 整行注释
            if _HARDCODE_OK.search(line):
                continue  # 行级内联豁免——必须带非空理由（无理由不豁免，见 R-exemption-live）
            code_part = line.split("#", 1)[0]  # 行尾注释也剔除
            if not _ABS_PATH_RE.search(code_part):
                continue
            # 例外：env 默认值（可被覆盖）——纪律允许的形态（整段豁免：
            # 多行 os.environ.get/setdefault 的字面量行在上，见 _env_default_lines）
            if i in env_lines:
                continue
            bad.append((rel, i, line.strip()[:110]))
    return bad


def audit_hardcoded_paths() -> list[str]:
    """扫描**可执行代码**中的绝对路径字面量（docstring/注释不算）。

    判据：AST 标记 docstring 所在行并豁免，再剔除纯注释行——docstring 里
    的用法示例被排除，真正写死在代码里的路径会被抓到。判据可信度由
    _abs_path_selftest_failed 自检兜底（正则失明/过宽即整项红）。
    """
    fail = _abs_path_selftest_failed()
    if fail:
        return [fail]  # 自检失败（伪命中）原样上报，不做存量过滤
    return ["%s:%d: %s" % (rel, i, line) for rel, i, line in _abs_path_hits_raw()]


def deploy_fork_dll(dry: bool = False) -> int:
    """把 fork 构建产物 decord.dll 同步到 site-packages（含 md5 校验）。

    取代手工 `cp`——这是「实际使用的 DLL 非最新」的根治手段。
    """
    # 不 import decord（import 会把 decord.dll 载入本进程
    # 并锁住，--deploy 随即无法覆盖它——实测"目标 DLL 被占用"）。find_spec
    # 只解析路径不执行包代码。
    import importlib.util

    spec = importlib.util.find_spec("decord")
    if spec is None or not spec.submodule_search_locations:
        print("✗ decord 未安装（site-packages 无该包）")
        return 2
    src = fork_build_dll()
    if src is None:
        print(
            "✗ 找不到 fork 构建产物（decord 仓 build-*\\decord.dll）\n"
            "  先构建：cd /d <decord repo> && rebuild_dev.bat"
        )
        return 2
    dst = Path(list(spec.submodule_search_locations)[0]) / "decord.dll"
    if not dst.parent.is_dir():
        print("✗ site-packages decord 目录不存在：%s" % dst.parent)
        return 2
    src_md5 = _md5(src)
    dst_md5 = _md5(dst) if dst.is_file() else ""
    if src_md5 == dst_md5:
        print("✓ 已是最新（构建产物与已部署 DLL 的 md5 一致：%s）" % src_md5[:12])
        return _sync_libinfo(src, dst)
    print("构建产物: %s  (%s)" % (src, src_md5[:12]))
    print("已部署  : %s  (%s)" % (dst, dst_md5[:12] or "不存在"))
    if dry:
        print("（--dry-run：未执行拷贝）")
        return 0
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        print("✗ 目标 DLL 被占用——先关掉所有 python 进程再试：\n  taskkill /F /IM python.exe")
        return 3
    now_md5 = _md5(dst)
    if now_md5 != src_md5:
        print("✗ 拷贝后 md5 仍不一致（%s vs %s）" % (now_md5[:12], src_md5[:12]))
        return 3
    print("✓ 已同步（md5 %s）" % now_md5[:12])
    return _sync_libinfo(src, dst)


def _sync_libinfo(src: Path, dst: Path) -> int:
    """版本元数据（libinfo.py）同步。

    只拷 DLL 会让 site-packages 版本号滞后，须一并同步。DLL 已最新而
    libinfo 滞后的形态同样要修（早退路径也要走这里）。源 = fork 仓
    python/decord/_ffi/libinfo.py。
    """
    libinfo_src = src.parents[1] / "python" / "decord" / "_ffi" / "libinfo.py"
    if not libinfo_src.is_file():
        return 0
    libinfo_dst = dst.parent / "_ffi" / "libinfo.py"
    try:
        if not libinfo_dst.is_file() or _md5(libinfo_src) != _md5(libinfo_dst):
            shutil.copy2(libinfo_src, libinfo_dst)
            print("✓ 版本元数据已同步（libinfo.py）")
    except (PermissionError, OSError) as e:
        print("✗ libinfo.py 同步失败（%s）——decord_pin 审计将报版本漂移" % e)
        return 3
    return 0


def _md5(p: Path) -> str:
    import hashlib

    h = hashlib.md5()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _drive_root_tmp_notes() -> list[str]:
    """盘符根 `<drive>:\\tmp` 非空提示。

    字面 /tmp 在原生 Windows 解析为**当前盘符根**（与 Git Bash 的 /tmp =
    %TEMP% 是不同目录）——误写产物落在仓外，任何仓内守卫都看不见。
    这里只提示（非失败）：发现即清空，并把写方改为 tempfile.gettempdir()。
    """
    import os as _os

    drives = list(getattr(_os, "listdrives", lambda: [])())
    if not drives:
        drives = list({Path.cwd().drive, Path(tempfile.gettempdir()).drive} - {""})
    notes = []
    for d in drives:
        p = Path(d) / "tmp"
        try:
            if p.is_dir() and any(p.iterdir()):
                n = sum(1 for q in p.rglob("*") if q.is_file())
                sz = sum(q.stat().st_size for q in p.rglob("*") if q.is_file())
                notes.append(
                    "%s\\tmp 非空（%d 文件 / %.1f MB）——疑似误写（临时目录应经 tempfile.gettempdir()）"
                    % (d, n, sz / 1e6)
                )
        except OSError:
            continue
    return notes


def health_report() -> dict:
    """依赖事实汇总（供打印 / --json / CI）。"""
    ff = ffmpeg_fact()
    build = fork_build_dll()
    dc = decord_fact(expect_dll=build)
    vd = video_dir()
    return {
        "ffmpeg": {
            "path": str(ff.path) if ff.ok else None,
            "source": ff.source,
            "version": ff.version,
            "ok": ff.ok,
            "note": ff.note,
        },
        "decord": {
            "path": str(dc.path) if dc.ok else None,
            "version": dc.version,
            "md5": dc.md5,
            "ok": dc.ok and not dc.note,
            "note": dc.note,
        },
        "fork_build": {"path": str(build) if build else None, "md5": _md5(build) if build else ""},
        "video_dir": {"path": str(vd), "exists": vd.is_dir()},
        "drive_tmp_notes": _drive_root_tmp_notes(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--deploy", action="store_true", help="把 fork 构建产物同步到 site-packages（md5 校验）"
    )
    ap.add_argument("--audit", action="store_true", help="硬编码绝对路径审计（违规→非零退出）")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--dry-run", action="store_true", help="只报告不落地")
    args = ap.parse_args()

    if args.deploy:
        return deploy_fork_dll(dry=args.dry_run)

    if args.audit:
        bad = audit_hardcoded_paths()
        if args.json:
            print(json.dumps({"hardcoded_paths": bad}, ensure_ascii=False, indent=1))
        elif bad:
            print(
                "✗ 发现 %d 处硬编码绝对路径（纪律：外部依赖路径须经 "
                "tools/env_probe.py 解析）：" % len(bad)
            )
            for b in bad:
                print("   " + b)
            print(
                "\n修法：from env_probe import ffmpeg_bin; "
                "ffmpeg_bin('ffmpeg')（或 os.environ.get 带默认值）"
            )
        else:
            print("✓ 无硬编码绝对路径")
        return 1 if bad else 0

    rep = health_report()
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0 if (rep["ffmpeg"]["ok"] and rep["decord"]["ok"]) else 1

    print("=" * 68)
    print("环境体检（env_doctor）")
    print("=" * 68)
    ff = rep["ffmpeg"]
    print("  %-10s %s  %s" % ("ffmpeg", "✓" if ff["ok"] else "✗", ff["path"] or ff["note"]))
    if ff["version"]:
        print("            版本 %s（来源 %s）" % (ff["version"], ff["source"]))
    dc = rep["decord"]
    print("  %-10s %s  %s" % ("decord", "✓" if dc["ok"] else "✗", dc["path"] or dc["note"]))
    if dc["version"]:
        print("            版本 %s  md5 %s" % (dc["version"], dc["md5"][:12]))
    fb = rep["fork_build"]
    print(
        "  %-10s %s  %s"
        % ("fork构建", "✓" if fb["path"] else "✗", fb["path"] or "未找到（未构建？）")
    )
    if fb["md5"]:
        print("            md5 %s" % fb["md5"][:12])
    if dc["note"]:
        print("\n  ⚠️ %s" % dc["note"])
        print("  修复：python tools/env_doctor.py --deploy")
    vd = rep["video_dir"]
    print("  %-10s %s  %s" % ("视频目录", "✓" if vd["exists"] else "✗", vd["path"]))
    for note in rep.get("drive_tmp_notes") or []:
        print("  %-10s ⚠️ %s" % ("盘符根tmp", note))
    ok = ff["ok"] and dc["ok"]
    print("\n%s" % ("✓ 环境就绪" if ok else "✗ 环境有问题（见上）"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
