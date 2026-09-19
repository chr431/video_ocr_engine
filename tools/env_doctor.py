"""环境体检 / 部署同步 / 硬编码路径审计（2026-09-19 纪律轮）。

三类事故的根治工具：
  ① 找不到 ffmpeg      → `env_doctor.py` 打印每个依赖的**解析来源+版本+md5**，
                          找不到时列出全部已探测候选；
  ② DLL 非最新         → `--deploy` 把 fork 构建产物同步进 site-packages 并
                          校验 md5；`--check` 检出不一致即非零退出；
  ③ 硬编码路径回潮     → `--audit` 扫描仓库内的绝对路径字面量（新纪律项）。

用法：
  python tools/env_doctor.py            # 体检（依赖事实 + DLL 一致性）
  python tools/env_doctor.py --deploy   # 同步 fork 构建产物 → site-packages
  python tools/env_doctor.py --audit    # 硬编码路径审计（CI/纪律审计调用）
  python tools/env_doctor.py --json     # 机器可读（探针/CI 消费）
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from env_probe import (  # noqa: E402
    decord_fact, ffmpeg_fact, fork_build_dll, video_dir)

ROOT = Path(__file__).resolve().parents[1]

# 审计：绝对路径字面量（Windows 盘符 / UNC）。白名单按「文件 + 行内容」放行。
_ABS_PATH_RE = re.compile(r"""(?<![\w"'#])[A-Za-z]:[\\/](?:[\w.\- ]+[\\/])*[\w.\- ]+""")
_AUDIT_SKIP_DIRS = {".git", "build", "dist", "__pycache__", ".venv", "node_modules",
                    "bench", "docs"}          # docs/log 的历史叙事可提及路径
_AUDIT_EXTS = {".py", ".bat", ".ps1", ".cmd"}
# 允许的例外：环境事实源本身（它就是放默认路径的地方）+ 文档生成器
# （_split_claude_md.py 把 AGENTS.md 内容作为字符串数据持有，非代码路径）
_AUDIT_ALLOW_FILES = {"tools/env_probe.py", "tools/_split_claude_md.py"}


def _iter_audit_files():
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _AUDIT_EXTS:
            continue
        if any(part in _AUDIT_SKIP_DIRS for part in p.parts):
            continue
        yield p


def audit_hardcoded_paths() -> list[str]:
    """扫描**可执行代码**中的绝对路径字面量（docstring/注释不算）。

    判据：用 AST 标记 docstring 所在行并豁免，再剔除纯注释行——这样
    docstring 里的用法示例（占误报多数）被排除，而真正写死在代码里的
    路径会被抓到。
    """
    import ast
    bad: list[str] = []
    for p in _iter_audit_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in _AUDIT_ALLOW_FILES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — 读不了就跳过（非源码）
            continue
        doc_lines: set[int] = set()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                body = getattr(node, "body", None)
                if isinstance(body, list) and body:
                    first = body[0]
                    if (isinstance(first, ast.Expr)
                            and isinstance(first.value, ast.Constant)
                            and isinstance(first.value.value, str)):
                        end = getattr(first.value, "end_lineno",
                                      first.value.lineno)
                        doc_lines.update(range(first.value.lineno, end + 1))
        for i, line in enumerate(text.splitlines(), 1):
            if i in doc_lines:
                continue
            if line.lstrip().startswith("#"):
                continue          # 整行注释
            code_part = line.split("#", 1)[0]   # 行尾注释也剔除
            if not _ABS_PATH_RE.search(code_part):
                continue
            # 例外：env 默认值（可被覆盖）——纪律允许的形态
            if "environ.get(" in code_part or "getenv(" in code_part:
                continue
            bad.append("%s:%d: %s" % (rel, i, line.strip()[:110]))
    return bad


def deploy_fork_dll(dry: bool = False) -> int:
    """把 fork 构建产物 decord.dll 同步到 site-packages（含 md5 校验）。

    取代手工 `cp`——这是「实际使用的 DLL 非最新」的根治手段。
    """
    # 不 import decord（2026-09-19 纪律轮：import 会把 decord.dll 载入本进程
    # 并锁住，--deploy 随即无法覆盖它——实测"目标 DLL 被占用"）。find_spec
    # 只解析路径不执行包代码。
    import importlib.util
    spec = importlib.util.find_spec("decord")
    if spec is None or not spec.submodule_search_locations:
        print("✗ decord 未安装（site-packages 无该包）")
        return 2
    src = fork_build_dll()
    if src is None:
        print("✗ 找不到 fork 构建产物（decord 仓 build-*\\decord.dll）\n"
              "  先构建：cd /d <decord repo> && rebuild_dev.bat")
        return 2
    dst = Path(list(spec.submodule_search_locations)[0]) / "decord.dll"
    if not dst.parent.is_dir():
        print("✗ site-packages decord 目录不存在：%s" % dst.parent)
        return 2
    src_md5 = _md5(src)
    dst_md5 = _md5(dst) if dst.is_file() else ""
    if src_md5 == dst_md5:
        print("✓ 已是最新（构建产物与已部署 DLL 的 md5 一致：%s）" % src_md5[:12])
        return 0
    print("构建产物: %s  (%s)" % (src, src_md5[:12]))
    print("已部署  : %s  (%s)" % (dst, dst_md5[:12] or "不存在"))
    if dry:
        print("（--dry-run：未执行拷贝）")
        return 0
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        print("✗ 目标 DLL 被占用——先关掉所有 python 进程再试：\n"
              "  taskkill /F /IM python.exe")
        return 3
    now_md5 = _md5(dst)
    if now_md5 != src_md5:
        print("✗ 拷贝后 md5 仍不一致（%s vs %s）" % (now_md5[:12], src_md5[:12]))
        return 3
    print("✓ 已同步（md5 %s）" % now_md5[:12])
    return 0


def _md5(p: Path) -> str:
    import hashlib
    h = hashlib.md5()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return ""


def health_report() -> dict:
    """依赖事实汇总（供打印 / --json / CI）。"""
    ff = ffmpeg_fact()
    build = fork_build_dll()
    dc = decord_fact(expect_dll=build)
    vd = video_dir()
    return {
        "ffmpeg": {"path": str(ff.path) if ff.ok else None,
                   "source": ff.source, "version": ff.version,
                   "ok": ff.ok, "note": ff.note},
        "decord": {"path": str(dc.path) if dc.ok else None,
                   "version": dc.version, "md5": dc.md5,
                   "ok": dc.ok and not dc.note, "note": dc.note},
        "fork_build": {"path": str(build) if build else None,
                       "md5": _md5(build) if build else ""},
        "video_dir": {"path": str(vd), "exists": vd.is_dir()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true",
                    help="把 fork 构建产物同步到 site-packages（md5 校验）")
    ap.add_argument("--audit", action="store_true",
                    help="硬编码绝对路径审计（违规→非零退出）")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--dry-run", action="store_true", help="只报告不落地")
    args = ap.parse_args()

    if args.deploy:
        return deploy_fork_dll(dry=args.dry_run)

    if args.audit:
        bad = audit_hardcoded_paths()
        if args.json:
            print(json.dumps({"hardcoded_paths": bad}, ensure_ascii=False,
                             indent=1))
        elif bad:
            print("✗ 发现 %d 处硬编码绝对路径（纪律：外部依赖路径须经 "
                  "tools/env_probe.py 解析）：" % len(bad))
            for b in bad:
                print("   " + b)
            print("\n修法：from env_probe import ffmpeg_bin; "
                  "ffmpeg_bin('ffmpeg')（或 os.environ.get 带默认值）")
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
    print("  %-10s %s  %s" % ("ffmpeg", "✓" if ff["ok"] else "✗",
                              ff["path"] or ff["note"]))
    if ff["version"]:
        print("            版本 %s（来源 %s）" % (ff["version"], ff["source"]))
    dc = rep["decord"]
    print("  %-10s %s  %s" % ("decord", "✓" if dc["ok"] else "✗",
                              dc["path"] or dc["note"]))
    if dc["version"]:
        print("            版本 %s  md5 %s" % (dc["version"], dc["md5"][:12]))
    fb = rep["fork_build"]
    print("  %-10s %s  %s" % ("fork构建", "✓" if fb["path"] else "✗",
                              fb["path"] or "未找到（未构建？）"))
    if fb["md5"]:
        print("            md5 %s" % fb["md5"][:12])
    if dc["note"]:
        print("\n  ⚠️ %s" % dc["note"])
        print("  修复：python tools/env_doctor.py --deploy")
    vd = rep["video_dir"]
    print("  %-10s %s  %s" % ("视频目录", "✓" if vd["exists"] else "✗",
                              vd["path"]))
    ok = ff["ok"] and dc["ok"]
    print("\n%s" % ("✓ 环境就绪" if ok else "✗ 环境有问题（见上）"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
