"""引用解析器——注册名 token 的解析与全量校验（R-refs-resolve 执行体）。

token 语法（宪章不变量 2）：
  C-NN            结论（registry/conclusions.yaml；superseded 在 conclusions-history）
  R-名 / adv-名   规则（registry/rules.yaml）
  decord@x.y.z    外部钉版（须与 registry/environment.yaml 一致）
  archive/<tag>   档案锚点（须为已存在的 git tag）

校验错误四类，每类有负测样例（tests/registry/test_refs.py）：
  UNKNOWN          文中出现 token 形态但注册表中无定义（悬空引用）
  VERSION_MISMATCH decord@ 版本与钉版不符
  BARE_SECTION     md 文本中出现裸 §N（章号仅允许 log 叙事内部，禁入基底）
  BARE_VERSION     md 文本中出现裸 x.y.z（版本只住 environment 数据或带前缀）

扫描范围：substrate = charter.md + registry/*.yaml。Phase C/E 视图切换后扩面。
用法：
  python tools/refs.py resolve C-53
  python tools/refs.py check
  python tools/refs.py check --files <path>...
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
CHARTER = ROOT / "charter.md"
HISTORY = ROOT / "docs" / "log" / "2026-09-10-conclusions-history.md"
HISTORY_ARCHIVE = "archive/pre-rework:docs/log/2026-09-10-conclusions-history.md"

TOKEN_CONCLUSION = re.compile(r"(?<![A-Za-z0-9@-])C-\d{2,3}\b")
TOKEN_RULE = re.compile(r"(?<![A-Za-z0-9-])(?:R|adv)-[a-z0-9]+(?:-[a-z0-9]+)*")
TOKEN_DECORD = re.compile(r"decord@(\d+\.\d+\.\d+)")
TOKEN_FORK = re.compile(r"fork@C-\d{2,3}\b")  # 外部命名空间：decord fork 仓自己的结论编号
TOKEN_ARCHIVE = re.compile(r"archive/([A-Za-z0-9_.-]+)")
BARE_SECTION = re.compile(r"§\d")
BARE_VERSION = re.compile(r"(?<![@\w.])\d+\.\d+\.\d+(?![\w.])")


def _load_yaml(path: Path) -> Any:
    if not path.is_file():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def _git_tags() -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "tag", "--list"], capture_output=True, text=True, encoding="utf-8"
    )
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


class Index:
    """全部命名空间的解析索引（从各注册表派生，不另立事实源）。"""

    def __init__(self) -> None:
        self.conclusions: dict[str, dict] = {}
        for entry in _load_yaml(REGISTRY / "conclusions.yaml"):
            if isinstance(entry, dict) and "id" in entry:
                self.conclusions[entry["id"]] = entry
        self.rules: dict[str, dict] = {}
        for entry in _load_yaml(REGISTRY / "rules.yaml"):
            if isinstance(entry, dict) and "id" in entry:
                self.rules[entry["id"]] = entry
        self.historical: set[str] = set()
        if HISTORY.is_file():  # 过渡期本地文件；出树后经档案 tag 取
            self.historical = set(
                re.findall(r"(?m)^## (C-\d+)", HISTORY.read_text(encoding="utf-8"))
            )
        else:
            r = subprocess.run(
                ["git", "-C", str(ROOT), "show", HISTORY_ARCHIVE],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if r.returncode == 0:
                self.historical = set(re.findall(r"(?m)^## (C-\d+)", r.stdout))
        env = _load_yaml(REGISTRY / "environment.yaml")
        self.decord_pin = str(env.get("dependencies", {}).get("decord_pin", ""))
        self.tags = _git_tags()

    def resolve(self, token: str) -> str:
        """单 token 解析，返回人读结果；不可解析返回 'UNKNOWN: <token>'。"""
        if token in self.conclusions:
            entry = self.conclusions[token]
            return f"{token} [active] {entry.get('conclusion', '')}"
        if token in self.historical:
            return f"{token} [superseded/dead] 全文经 {HISTORY_ARCHIVE} 取回（git show）"
        if token in self.rules:
            entry = self.rules[token]
            return f"{token} [{entry.get('enforcement', '')}] {entry.get('statement', '')}"
        m = TOKEN_DECORD.fullmatch(token)
        if m:
            state = (
                "一致"
                if token == f"decord@{self.decord_pin}"
                else (f"VERSION_MISMATCH（钉版为 decord@{self.decord_pin}）")
            )
            return f"{token} [钉版对账] {state}"
        if TOKEN_FORK.fullmatch(token):
            return f"{token} [外部命名空间] decord fork 仓结论（其仓内取全文）"
        if token.startswith("archive/"):
            tag = token[len("archive/") :]
            state = "存在" if tag in self.tags else "UNKNOWN（tag 不存在）"
            return f"{token} [档案 tag] {state}"
        return f"UNKNOWN: {token}"


def _substrate_files(extra: list[str]) -> list[Path]:
    files = [CHARTER] + sorted(REGISTRY.glob("*.yaml")) if REGISTRY.is_dir() else [CHARTER]
    if not CHARTER.is_file():
        files = sorted(REGISTRY.glob("*.yaml"))
    files += [Path(f) for f in extra]
    return [f for f in files if f.is_file()]


def scan_file(path: Path, index: Index) -> list[str]:
    """单文件扫描：token 逐一解析 + md 文本的裸 §N/裸版本检查。"""
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        rel = path.relative_to(ROOT)
    except ValueError:  # 仓外文件（负测夹具）按原路径显示
        rel = path
    for lineno, line in enumerate(text.splitlines(), 1):
        where = f"{rel}:{lineno}"
        for m in TOKEN_CONCLUSION.finditer(line):
            tok = m.group(0)
            if tok not in index.conclusions and tok not in index.historical:
                errors.append(f"{where}  UNKNOWN  {tok}")
        for m in TOKEN_RULE.finditer(line):
            tok = m.group(0)
            if tok not in index.rules:
                errors.append(f"{where}  UNKNOWN  {tok}")
        for m in TOKEN_DECORD.finditer(line):
            if m.group(0) != f"decord@{index.decord_pin}":
                errors.append(f"{where}  VERSION_MISMATCH  {m.group(0)}")
        for m in TOKEN_ARCHIVE.finditer(line):
            if m.group(0) not in index.tags:
                errors.append(f"{where}  UNKNOWN  {m.group(0)}")
        if path.suffix == ".md":  # 裸章号/裸版本仅查 prose 基底（数据住 yaml）
            if BARE_SECTION.search(line):
                errors.append(f"{where}  BARE_SECTION  裸 §N 禁入基底")
            for m in BARE_VERSION.finditer(line):
                errors.append(f"{where}  BARE_VERSION  {m.group(0)}（版本须前缀或住 environment）")
    return errors


def check(extra_files: list[str] | None = None) -> list[str]:
    """substrate 全量校验；返回错误清单（空=通过）。"""
    index = Index()
    errors: list[str] = []
    for path in _substrate_files(extra_files or []):
        errors.extend(scan_file(path, index))
    return errors


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "resolve":
        if len(argv) != 3:
            print("用法：refs resolve <token>")
            return 2
        print(Index().resolve(argv[2]))
        return 0
    if len(argv) >= 2 and argv[1] == "check":
        extra = []
        if "--files" in argv:
            rest = argv[argv.index("--files") + 1 :]
            extra = rest
        errors = check(extra)
        if errors:
            print(f"refs check：{len(errors)} 处违规")
            for e in errors:
                print(f"  {e}")
            return 1
        print("refs check：substrate 引用全部可解析 ✓")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
