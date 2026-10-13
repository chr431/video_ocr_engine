"""审计哨兵 —— 机器化的防复发机制。

四类静默劣化各自留下机器哨兵：

1. **死代码**（`_decord_format` 委托链、`DEFAULT_*` 孤儿常量）：靠 review
   拦不住——AST 扫描「全仓（包+tests+tools）词法引用计数 ≤ 定义行」
   即红。误报走 ALLOWLIST（逐条写理由）。
2. **命名与行为不符**（`'onnxruntime'` 等旧 ORT 标签，C-48）：产品码与
   测试面禁该标识符——任何回潮即红。
3. **architecture.svg 版本漂移**：图内版本标记必须 == `__version__`
   （「改架构时同步刷新版本号」的机器化——发版忘刷图不再可能静默通过）。
4. **AGENTS.md 硬编码快照计数**（live/frozen、审计项数——易漂移）：
   禁止在 AGENTS 里写易漂移的数字快照，指针式表述
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
    # 单遍合并 alternation + 命中计数（行为等价——\b 分词语义不变，
    # 命中归属由捕获组还原）；逐名单正则扫描需 O(names×corpus)
    # （525 名单 × 244 文件 ≈ 12s），太慢。
    corpus = _corpus()
    pkg_names: set[str] = set()
    for path, src in corpus.items():
        if not path.startswith(str(PKG)):
            continue
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                pkg_names.add(node.name)
            elif isinstance(node, ast.Assign):
                # 模块级 UPPER_CASE 常量（死常量类：DEFAULT_* 孤儿即此类）
                for t in node.targets:
                    if isinstance(t, ast.Name) and re.fullmatch(r"[A-Z][A-Z0-9_]{3,}", t.id):
                        pkg_names.add(t.id)
    pkg_names = {n for n in pkg_names if not n.startswith("__") and n not in ALLOWLIST}
    if not pkg_names:
        return []
    pat = re.compile(r"\b(" + "|".join(sorted(re.escape(n) for n in pkg_names)) + r")\b")
    counts: dict[str, int] = {}
    for src in corpus.values():
        for m in pat.finditer(src):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1

    findings = []
    for path, src in corpus.items():
        if not path.startswith(str(PKG)):
            continue
        tree = ast.parse(src)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and re.fullmatch(r"[A-Z][A-Z0-9_]{3,}", t.id):
                        names.append(t.id)
            for name in names:
                if name.startswith("__") or name in ALLOWLIST:
                    continue
                if counts.get(name, 0) <= 1:
                    findings.append(f"{name}  ({Path(path).relative_to(ROOT)})")
    return sorted(set(findings))


def test_no_dead_definitions_or_constants() -> None:
    """产品包内定义（函数/类/模块级常量）在全仓词法引用 ≤1 即死代码。"""
    findings = [f for f in _dead_candidates() if f.split("  ")[0] not in ALLOWLIST]
    assert not findings, (
        "疑似死代码（全仓零引用；动态面请登记 tests/test_audit_sentinels.py"
        f" 的 ALLOWLIST 并写理由）: {findings}"
    )


# ── 2. 旧 OCR 标识禁串（任何回潮即红）────────────────────────────────
BANNED_TOKENS = ("onnxruntime", "OCR_ONNX_CHUNK", "_init_onnx")


def test_no_legacy_ort_labels() -> None:
    """产品码、测试与手写文档（README）不得出现旧 ORT 标识。

    面含 README：它是唯一手写散文文档，"旧能力标签任何回潮即红"必须
    覆盖它——本轮实测曾漏（README 把 onnxruntime 写成运行时依赖）。
    """
    hits = []
    scanned = [p for base in (PKG, ROOT / "tests") for p in base.rglob("*.py")]
    scanned.append(ROOT / "README.md")
    for p in scanned:
        if "__pycache__" in p.parts or not p.is_file():
            continue
        if p.name == "test_audit_sentinels.py":
            continue  # 哨兵自身定义禁串表，不算回潮
        src = p.read_text(encoding="utf-8", errors="replace")
        for tok in BANNED_TOKENS:
            if tok in src:
                hits.append(f"{p.relative_to(ROOT)}: {tok}")
    assert not hits, f"旧 OCR 标识回潮（现行为 openvino/OCR_OV_CHUNK/_init_ov）: {hits}"


# ── 3. architecture.svg 版本哨兵 ────────────────────────────────────
def test_architecture_svg_version_matches() -> None:
    """README 配图版本标记必须 == 当前版本（AGENTS「改架构同步刷版本号」）。"""
    import video_ocr_engine

    svg = (ROOT / "docs" / "architecture.svg").read_text(encoding="utf-8")
    assert f"（{video_ocr_engine.__version__}）" in svg, (
        f"docs/architecture.svg 版本标记未随 {video_ocr_engine.__version__} "
        "刷新（标题文本内更新版本号）"
    )


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
            "改为指针式（「以 tools/INDEX.md / 脚本输出为准」）"
        )


# ── 负测（C2 / R-guard-reliable）：四类哨兵必须能失败 ─────────────────
# 手法：把哨兵读的根（ROOT/PKG/CORPUS_DIRS）指到临时坏树，调用哨兵本体，
# 断言它确实红——"守卫能失败"的实体证据，不重构哨兵逻辑。
import sys as _sys  # noqa: E402

import pytest  # noqa: E402

_ME = _sys.modules[__name__]


def test_negative_dead_sentinel_catches_orphan(tmp_path, monkeypatch) -> None:
    """坏例：孤立定义（全仓零引用）必须被抓。"""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "m.py").write_text("ORPHAN_CONST = 1\n\n\ndef used():\n    return 1\n", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "t.py").write_text("from pkg.m import used\n\nused()\n", encoding="utf-8")
    monkeypatch.setattr(_ME, "PKG", pkg)
    monkeypatch.setattr(_ME, "CORPUS_DIRS", (pkg, other))
    monkeypatch.setattr(_ME, "ROOT", tmp_path)  # 报告用相对路径，须同根
    findings = _dead_candidates()
    assert any("ORPHAN_CONST" in f for f in findings), findings


def test_negative_ort_sentinel_catches_label(tmp_path, monkeypatch) -> None:
    """坏例：产品面出现旧 ORT 标签（哨兵本体必须 AssertionError）。"""
    pkg = tmp_path / "video_ocr_engine"
    pkg.mkdir()
    (pkg / "bad.py").write_text('BACKEND = "onnxruntime"\n', encoding="utf-8")
    monkeypatch.setattr(_ME, "PKG", pkg)
    monkeypatch.setattr(_ME, "ROOT", tmp_path)
    with pytest.raises(AssertionError, match="旧 OCR 标识"):
        test_no_legacy_ort_labels()


def test_negative_svg_sentinel_catches_stale_version(tmp_path, monkeypatch) -> None:
    """坏例：架构图版本标记与当前版本不符。"""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "architecture.svg").write_text("<svg>（0.0.0）</svg>", encoding="utf-8")
    monkeypatch.setattr(_ME, "ROOT", tmp_path)
    with pytest.raises(AssertionError, match="版本标记"):
        test_architecture_svg_version_matches()


def test_negative_agents_sentinel_catches_snapshot_count(tmp_path, monkeypatch) -> None:
    """坏例：AGENTS 里写死易漂移的计数快照。"""
    (tmp_path / "AGENTS.md").write_text("standing live 5 / frozen 75\n", encoding="utf-8")
    monkeypatch.setattr(_ME, "ROOT", tmp_path)
    with pytest.raises(AssertionError, match="硬编码计数"):
        test_agents_has_no_stale_snapshot_counts()
