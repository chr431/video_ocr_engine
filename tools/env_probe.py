"""环境事实源（environment provenance）：外部依赖的**唯一解析入口**。

背景（2026-09-19 纪律轮）：反复发生两类事故——
  ① 「找不到 ffmpeg」：路径硬编码散落 4+ 处，目录改名/迁移后无人察觉；
  ② 「用的 decord DLL 不是最新」：构建产物 → site-packages 的部署靠手工
     `cp`，改了 C++ 却忘了部署，测量结果静默用旧 DLL。

根治思路（本模块 + 配套工具）：
  · **单一解析源**：所有外部依赖路径只在这里解析（env 覆盖 → 候选列表
    依次探测），工具/探针一律 `from env_probe import ffmpeg_bin`；
  · **显式校验**：解析结果带 `source`（来自 env / 哪个候选目录）+ 版本
    + md5，可打印、可断言；
  · **部署链自动化**：`tools/env_doctor.py --deploy` 把构建产物同步到
    site-packages 并校验 md5 一致（消除手工 cp）；
  · **审计守门**：`tools/env_doctor.py` 与 `tools/_probe_discipline_audit.py`
    检查硬编码路径（新纪律：仓库内不得出现绝对路径字面量）。

设计原则：**不引入新旋钮**（复用既有 FFMPEG_DIR / RACELOG_* env）；
找不到就**显式失败并列出已探测的候选**，绝不静默回落。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── 候选目录（按优先级；env 覆盖永远第一）─────────────────────────────
# 每项 = (标签, 目录)。标签用于报错时告诉用户"我找过哪些地方"。
_FFMPEG_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("env:FFMPEG_DIR", "{FFMPEG_DIR}"),
    ("BtbN-n9.0", r"D:\Software\ffmpeg-n9.0-latest-win64-gpl-shared-9.0\bin"),
    ("ffmpeg-9.0-sdk", r"D:\Software\ffmpeg-9.0-sdk\bin"),
    ("ffmpeg-9.0", r"D:\Software\ffmpeg-9.0"),
    ("PATH", "{PATH}"),
)


class EnvFact:
    """一个已解析的外部依赖事实。"""

    __slots__ = ("kind", "path", "source", "version", "md5", "note")

    def __init__(self, kind: str, path: Path | None, source: str,
                 version: str = "", md5: str = "", note: str = "") -> None:
        self.kind = kind
        self.path = path
        self.source = source
        self.version = version
        self.md5 = md5
        self.note = note

    @property
    def ok(self) -> bool:
        return self.path is not None

    def line(self) -> str:
        if not self.ok:
            return "%-10s ✗ %s" % (self.kind, self.note or "未找到")
        parts = ["%-10s ✓ %s" % (self.kind, self.path)]
        if self.version:
            parts.append("版本=%s" % self.version)
        if self.md5:
            parts.append("md5=%s" % self.md5[:12])
        parts.append("(来源: %s)" % self.source)
        return "  ".join(parts)


def _probe_exe(name: str) -> tuple[Path | None, str, str]:
    """按候选列表找可执行文件；返回 (path, source, note)。

    找不到时 note 列出全部已探测候选（用户据此知道"该放哪"）。
    """
    tried: list[str] = []
    for label, tmpl in _FFMPEG_CANDIDATES:
        if tmpl == "{FFMPEG_DIR}":
            d = os.environ.get("FFMPEG_DIR", "").strip()
            if not d:
                continue
            cand = Path(d) / name
            if cand.is_file():
                return cand, label, ""
            tried.append("%s=%s（无 %s）" % (label, d, name))
        elif tmpl == "{PATH}":
            found = shutil.which(name)
            if found:
                return Path(found), label, ""
            tried.append("%s（which 未命中）" % label)
        else:
            cand = Path(tmpl) / name
            if cand.is_file():
                return cand, label, ""
            tried.append("%s=%s" % (label, tmpl))
    return None, "", "已探测：%s" % "; ".join(tried)


def _version_of(exe: Path, args: tuple[str, ...] = ("-version",)) -> str:
    """取版本首行（ffmpeg -version / ffprobe -version）。"""
    try:
        out = subprocess.run([str(exe), *args], capture_output=True, text=True,
                             timeout=15, encoding="utf-8", errors="replace")
        first = (out.stdout or out.stderr or "").strip().splitlines()
        if first:
            # "ffmpeg version n9.0.1-26-g5c8e7e2433 Copyright ..." → 取第二词
            toks = first[0].split()
            for i, t in enumerate(toks):
                if t == "version" and i + 1 < len(toks):
                    return toks[i + 1]
            return toks[0][:40]
    except Exception:  # noqa: BLE001 — 版本探测失败不影响可用性判定
        pass  # 无版本号只影响显示，不改变依赖可用性结论
    return ""


def _md5(path: Path) -> str:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # noqa: BLE001 — 读不到就不给摘要
        return ""


def ffmpeg_bin(name: str = "ffmpeg") -> Path:
    """ffmpeg / ffprobe 可执行文件路径；找不到抛 FileNotFoundError。

    所有工具/探针必须经此获取——**不得再写绝对路径字面量**（纪律项）。
    """
    if name not in ("ffmpeg", "ffprobe", "ffplay"):
        raise ValueError("未知可执行名: %r" % name)
    exe = name + (".exe" if os.name == "nt" else "")
    path, source, note = _probe_exe(exe)
    if path is None:
        raise FileNotFoundError(
            "找不到 %s。%s\n修复：设置 FFMPEG_DIR 指向含 %s 的目录，"
            "或把该目录加入 PATH。" % (name, note, exe))
    return path


def ffmpeg_fact() -> EnvFact:
    """ffmpeg 事实（含版本与来源），供 env_doctor 打印。"""
    path, source, note = _probe_exe("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if path is None:
        return EnvFact("ffmpeg", None, "", note=note)
    return EnvFact("ffmpeg", path, source, version=_version_of(path))


def decord_fact(expect_dll: Path | None = None) -> EnvFact:
    """decord 包与其 DLL 的事实；``expect_dll`` 给定时校验 md5 一致。

    这是「实际使用的 DLL 非最新」的根治点：构建产物 md5 与
    site-packages 里的 DLL md5 必须一致，不一致即显式报错。
    """
    # ⚠️ 用 find_spec 定位而**不 import**（2026-09-19 纪律轮）：import decord
    # 会把 decord.dll 载入本进程并锁住，--deploy 随即无法覆盖它
    # （实测："目标 DLL 被占用"）。find_spec 只解析路径不执行包代码。
    import importlib.util
    try:
        spec = importlib.util.find_spec("decord")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("decord 未安装")
        pkg = Path(list(spec.submodule_search_locations)[0])
    except Exception as e:  # noqa: BLE001
        return EnvFact("decord", None, "", note="定位 decord 失败: %r" % (e,))
    dll = pkg / "decord.dll"
    if not dll.is_file():
        # 非 Windows 或静态链接构建
        dll = pkg / "libdecord.so"
    try:
        from importlib.metadata import version as _v
        ver = _v("decord")
    except Exception:  # noqa: BLE001 — 元数据缺失不影响事实记录
        ver = "?"
    fact = EnvFact("decord", dll if dll.is_file() else pkg, "site-packages",
                   version=ver, md5=_md5(dll) if dll.is_file() else "")
    if expect_dll is not None and dll.is_file():
        want = _md5(expect_dll)
        if want and fact.md5 and want != fact.md5:
            fact.note = ("**DLL 不一致**：构建产物 %s (%s) ≠ 已部署 %s (%s)"
                         "——运行 `tools/env_doctor.py --deploy` 同步"
                         % (expect_dll, want[:12], dll, fact.md5[:12]))
    return fact


def fork_build_dll(repo: str = r"D:\Repo\decord") -> Path | None:
    """decord fork 的构建产物 DLL（build-*/decord.dll，取最新修改的）。"""
    root = Path(repo)
    if not root.is_dir():
        return None
    cands = sorted(root.glob("build-*/decord.dll"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def video_dir() -> Path:
    """测试视频目录（RACELOG_VIDEO_DIR 覆盖）。"""
    return Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


def python_exe() -> Path:
    """项目解释器（本机绝对路径；PATH 上的 python 缺依赖）。"""
    return Path(sys.executable)
