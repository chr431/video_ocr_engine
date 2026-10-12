"""refs 解析器的正负测——R-guard-negative 的先行样板：每个错误类都有必须命中的坏例。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))  # tools/ 导入约定（探针/测试同款）

from refs import Index, check, scan_file  # noqa: E402


def _scan_content(tmp_path: Path, name: str, content: str, index: Index) -> list[str]:
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return scan_file(f, index)


def test_substrate_clean() -> None:
    """正测：真实基底（charter + registry）必须零违规。"""
    errors = check()
    assert errors == [], "substrate 应全部可解析，实际：\n" + "\n".join(errors)


def test_resolve_known_tokens() -> None:
    idx = Index()
    assert "[active]" in idx.resolve("C-53")
    assert "superseded" in idx.resolve("C-57")  # 历史结论可解（tombstone）
    assert "[local-gate]" in idx.resolve("R-ab-interleave")
    assert "钉版对账" in idx.resolve(f"decord@{idx.decord_pin}")
    assert "UNKNOWN" in idx.resolve("C-999")


def test_unknown_conclusion_and_rule(tmp_path: Path) -> None:
    """负测：悬空 C-/R- token 必须报 UNKNOWN。"""
    idx = Index()
    errors = _scan_content(tmp_path, "bad1.md", "见 C-999 与 R-nope-thing", idx)
    assert any("C-999" in e and "UNKNOWN" in e for e in errors)
    assert any("R-nope-thing" in e and "UNKNOWN" in e for e in errors)


def test_s0_prefix_not_token(tmp_path: Path) -> None:
    """负测边界：S0-C1 是痛点编号，不得误判为悬空结论。"""
    idx = Index()
    errors = _scan_content(tmp_path, "ok.md", "prevents S0-C1 S0-D2", idx)
    assert errors == []


def test_decord_version_mismatch(tmp_path: Path) -> None:
    idx = Index()
    errors = _scan_content(tmp_path, "bad2.yaml", "pin: decord@0.0.1", idx)
    assert any("VERSION_MISMATCH" in e for e in errors)


def test_bare_section_and_version_in_md(tmp_path: Path) -> None:
    """负测：md 基底中裸 §N 与裸 x.y.z 必须命中；yaml 中不查（数据值）。"""
    idx = Index()
    errors = _scan_content(tmp_path, "bad3.md", "见 §12 与 1.2.3 两个问题", idx)
    assert any("BARE_SECTION" in e for e in errors)
    assert any("BARE_VERSION" in e for e in errors)
    ok = _scan_content(tmp_path, "data.yaml", "mypy: 2.3.1", idx)
    assert ok == []


def test_bare_section_in_py_comment(tmp_path: Path) -> None:
    """负测：py 注释层的裸 §N 同样命中（注释轮后 md/py 一视同仁）。"""
    idx = Index()
    errors = _scan_content(tmp_path, "bad.py", "# 依据 §8.6 N-2 与 2026-09-12 §3", idx)
    assert any("BARE_SECTION" in e for e in errors)
    # py 里的版本字面量不判（常量/钉版值不可判定）——只禁 §N
    ok = _scan_content(tmp_path, "ok.py", "V = '0.26.2'  # 当前版本", idx)
    assert ok == []


def test_archive_tag(tmp_path: Path) -> None:
    idx = Index()
    errors = _scan_content(tmp_path, "bad4.md", "证据在 archive/not-a-tag", idx)
    assert any("UNKNOWN" in e and "archive/not-a-tag" in e for e in errors)


@pytest.mark.parametrize("tok", ["C-01", "C-80", "R-refs-resolve", "adv-commit-slim"])
def test_registry_ids_resolve(tok: str) -> None:
    """注册表内全部 id 形态可被 resolve（防 id 手写漂移）。"""
    assert "UNKNOWN" not in Index().resolve(tok)
