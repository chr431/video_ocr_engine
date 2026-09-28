"""审计哨兵 —— 2026-09-28 审计轮的防复发机制。

该轮审计发现四类静默劣化，各自留下机器哨兵：

1. **死代码**（`_decord_format` 委托链、`DEFAULT_*` 孤儿常量）：靠 review
   拦不住——AST 扫描「全仓（包+tests+tools）词法引用计数 ≤ 定义行」
   即红。误报走 ALLOWLIST（逐条写理由）。
2. **命名与行为不符**（`'onnxruntime'` 标签在 ORT 移除后残留一轮，
   C-48）：产品码与测试面禁该标识符——改名轮（0.17.0，MIGRATION §6）
   之后任何回潮即红。
3. **architecture.svg 版本漂移**（停在 0.13.3 连续三轮）：图内版本标记
   必须 == `__version__`（AGENTS 规矩「改架构时同步刷新版本号」的
   机器化——发版忘刷图不再可能静默通过）。
4. **AGENTS.md 硬编码快照计数**（live/frozen、审计项数——漂移了两轮
   才被审计发现）：禁止在 AGENTS 里写易漂移的数字快照，指针式表述
   （「以 INDEX/脚本输出为准」）才是合法形态。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from _paths import ROOT

PKG = ROOT / "video_ocr_engine"
CORPUS_DIRS = (PKG, ROOT / "tests", ROOT / "tools")

# ── 1. 死代码哨兵 ────────────────────────────────────────────────────
# 词法引用计数会把以下情况算作"有引用"（偏保守、少误报）：
# 同名方法在别的类里被用 / 字符串 getattr 动态取用。真动态面走 allowlist。
ALLOWLIST: dict[str, str] = {
    # 名字: 理由（每条必须写清为什么词法计数不可信）
}


def _corpus() -> dict[str, str]:
    out = {}
    for base in CORPUS_DIRS:
        for p in base.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            out[str(p)] = p.read_text(encoding="utf-8", errors="replace")
    return out


def _dead_candidates() -> list[str]:
    corpus = _corpus()
    counts: dict[str, int] = {}

    def mentions(name: str) -> int:
        if name not in counts:
            pat = re.compile(r"\b" + re.escape(name) + r"\b")
            counts[name] = sum(len(pat.findall(s)) for s in corpus.values())
        return counts[name]

    findings = []
    for path, src in corpus.items():
        if not path.startswith(str(PKG)):
            continue
        tree = ast.parse(src)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Assign):
                # 模块级 UPPER_CASE 常量（死常量类：DEFAULT_* 孤儿即此类）
                for t in node.targets:
                    if (isinstance(t, ast.Name)
                            and re.fullmatch(r"[A-Z][A-Z0-9_]{3,}", t.id)):
                        names.append(t.id)
            for name in names:
                if name.startswith("__") or name in ALLOWLIST:
                    continue
                if mentions(name) <= 1:
                    findings.append(f"{name}  ({Path(path).relative_to(ROOT)})")
    return sorted(set(findings))


def test_no_dead_definitions_or_constants() -> None:
    """产品包内定义（函数/类/模块级常量）在全仓词法引用 ≤1 即死代码。"""
    findings = [f for f in _dead_candidates()
                if f.split("  ")[0] not in ALLOWLIST]
    assert not findings, (
        "疑似死代码（全仓零引用；动态面请登记 tests/test_audit_sentinels.py"
        f" 的 ALLOWLIST 并写理由）: {findings}")


# ── 2. 旧 OCR 标识禁串（0.17.0 改名轮，MIGRATION §6）────────────────
BANNED_TOKENS = ("onnxruntime", "OCR_ONNX_CHUNK", "_init_onnx")


def test_no_legacy_ort_labels() -> None:
    """产品码与测试不得再出现旧 ORT 时代标识（历史叙事在 docs/log）。"""
    hits = []
    for base in (PKG, ROOT / "tests"):
        for p in base.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            if p.name == "test_audit_sentinels.py":
                continue   # 哨兵自身定义禁串表，不算回潮
            src = p.read_text(encoding="utf-8", errors="replace")
            for tok in BANNED_TOKENS:
                if tok in src:
                    hits.append(f"{p.relative_to(ROOT)}: {tok}")
    assert not hits, (
        f"旧 OCR 标识回潮（0.17.0 起为 openvino/OCR_OV_CHUNK/_init_ov，"
        f"见 MIGRATION §6）: {hits}")


# ── 3. architecture.svg 版本哨兵 ────────────────────────────────────
def test_architecture_svg_version_matches() -> None:
    """README 配图版本标记必须 == 当前版本（AGENTS「改架构同步刷版本号」）。"""
    import video_ocr_engine
    svg = (ROOT / "docs" / "architecture.svg").read_text(encoding="utf-8")
    assert f"（{video_ocr_engine.__version__}）" in svg, (
        f"docs/architecture.svg 版本标记未随 {video_ocr_engine.__version__} "
        "刷新（标题文本内更新版本号）")


# ── 4. AGENTS.md 禁硬编码快照计数 ───────────────────────────────────
_AGENTS_SNAPSHOT_PATTERNS = (
    re.compile(r"live \d+ / frozen \d+"),
    re.compile(r"\d+ 项 = \d+ 基础"),
)


def test_agents_has_no_stale_snapshot_counts() -> None:
    """AGENTS 是注入核，硬编码的计数快照必然漂移——只许指针式表述。"""
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for pat in _AGENTS_SNAPSHOT_PATTERNS:
        m = pat.search(agents)
        assert m is None, (
            f"AGENTS.md 含易漂移的硬编码计数 {m.group()!r}——"
            "改为指针式（「以 tools/INDEX.md / 脚本输出为准」）")
