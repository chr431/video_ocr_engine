"""_audit_ext 逐项负测（R-guard-negative：每项审计必须带"它抓得住的坏例"）。

模式：不重构审计本体，用 monkeypatch 供给坏输入——ROOT 指向临时坏树
（文件读取型审计）、替换被导入的数据源（产品注册表型）、替换 subprocess
返回（包装型）。判据=该项必须返回非 None 的违规消息。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import _audit_ext as ae  # noqa: E402


def _mini_root(tmp_path: Path) -> Path:
    pkg = tmp_path / "video_ocr_engine"
    pkg.mkdir(parents=True)
    return tmp_path


def test_layering_catches_cycle(tmp_path, monkeypatch):
    """坏例：a ↔ b 模块级导入环。"""
    root = _mini_root(tmp_path)
    (root / "video_ocr_engine" / "a.py").write_text("import b\n", encoding="utf-8")
    (root / "video_ocr_engine" / "b.py").write_text("import a\n", encoding="utf-8")
    monkeypatch.setattr(ae, "ROOT", root)
    assert "导入环" in (ae.check_layering() or "")


def test_no_shim_duplication_catches_revival(tmp_path, monkeypatch):
    """坏例：已清除的转发壳 similar_binary 复活。"""
    root = _mini_root(tmp_path)
    (root / "video_ocr_engine" / "_helpers.py").write_text(
        "def similar_binary(x):\n    return x\n", encoding="utf-8"
    )
    monkeypatch.setattr(ae, "ROOT", root)
    assert "similar_binary" in (ae.check_no_shim_duplication() or "")


def test_dead_code_regression_catches_execute(tmp_path, monkeypatch):
    """坏例：ocr_trt.py 复活零调用方法 execute。"""
    root = _mini_root(tmp_path)
    (root / "ocr_trt.py").write_text("def execute(x):\n    return x\n", encoding="utf-8")
    monkeypatch.setattr(ae, "ROOT", root)
    assert "execute" in (ae.check_dead_code_regression() or "")


def test_knob_coverage_catches_missing_rationale(monkeypatch):
    """坏例：注册表里混入缺 rationale_id 的旋钮。"""
    import video_ocr_engine.config as C
    from video_ocr_engine.config import knobs as K

    bad = SimpleNamespace(name="x.no_anchor", rationale_id=None)
    fake = SimpleNamespace(knobs=list(K.KNOBS.knobs) + [bad])
    monkeypatch.setattr(C, "KNOBS", fake)  # KNOBS 是 frozen dataclass，换模块出口
    assert "rationale_id" in (ae.check_knob_coverage() or "")


def test_no_import_side_effects_catches_environ_write(tmp_path, monkeypatch):
    """坏例：产品模块 import 期改写 os.environ。"""
    root = _mini_root(tmp_path)
    (root / "video_ocr_engine" / "dirty.py").write_text(
        'import os\nos.environ["VOE_X"] = "1"\n', encoding="utf-8"
    )
    monkeypatch.setattr(ae, "ROOT", root)
    assert "import 期改写" in (ae.check_no_import_side_effects() or "")


def test_metrics_coverage_catches_lost_key(monkeypatch):
    """坏例：现役指标 pipeline.decode 从注册表丢失。"""

    class _FakeMetrics:
        def names(self):
            return ["pipeline.ocr"]

        def __contains__(self, name):
            return name == "pipeline.ocr"

    import video_ocr_engine.domain.metrics as M

    monkeypatch.setattr(M, "METRICS", _FakeMetrics())
    assert "pipeline.decode" in (ae.check_metrics_coverage() or "")


def test_rules_schema_catches_missing_executor(tmp_path, monkeypatch):
    """坏例：ci 规则缺 executor（无执行体的不是规则）。"""
    root = _mini_root(tmp_path)
    (root / "registry").mkdir()
    (root / "registry" / "rules.yaml").write_text(
        "- id: TEST_RULE\n  statement: s\n  enforcement: ci\n  prevents: S0-A1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ae, "ROOT", root)
    assert "executor" in (ae.check_rules_schema() or "") or "TEST_RULE" in (
        ae.check_rules_schema() or ""
    )


def test_probe_hooks_resolve_catches_dead_point(monkeypatch):
    """坏例：打桩点表出现失效点位。"""
    import _probe_hooks

    monkeypatch.setattr(_probe_hooks.hooks, "audit", lambda: ["fake.point 已失效"])
    assert "打桩点失效" in (ae.check_probe_hooks_resolve() or "")


def test_render_consistency_catches_render_red(monkeypatch):
    """坏例：render --check 非零退出（包装逻辑须把红透传）。"""
    fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="x: 与源不一致", stderr="")
    monkeypatch.setattr(ae.subprocess, "run", lambda *a, **k: fake)
    assert "render --check 未过" in (ae.check_render_consistency() or "")


def test_decord_pin_catches_mismatch(tmp_path, monkeypatch):
    """坏例：environment 钉版 ≠ 本机安装版本。"""
    decord = pytest.importorskip("decord")  # CI 无 decord=环境事实，按环境跳过
    root = _mini_root(tmp_path)
    (root / "registry").mkdir()
    (root / "registry" / "environment.yaml").write_text(
        "dependencies:\n  decord_pin: 0.0.0\n", encoding="utf-8"
    )
    monkeypatch.setattr(ae, "ROOT", root)
    msg = ae.check_decord_pin() or ""
    assert "0.0.0" in msg and decord.__version__ in msg


def test_deprecation_due_catches_overdue_env(tmp_path, monkeypatch):
    """坏例：remove_in 已过而 env 旋钮仍活。"""
    root = _mini_root(tmp_path)
    (root / "registry").mkdir()
    from video_ocr_engine.config import KNOBS

    live_env = next(k.env for k in KNOBS.knobs if k.env)
    (root / "registry" / "deprecations.yaml").write_text(
        "- name: %s\n  kind: env\n  remove_in: 0.0.1\n" % live_env, encoding="utf-8"
    )
    monkeypatch.setattr(ae, "ROOT", root)
    assert live_env in (ae.check_deprecation_due() or "")


def test_gate_manifest_catches_missing(tmp_path, monkeypatch):
    """坏例：bump 后未重跑金标（manifest 缺失）。"""
    root = _mini_root(tmp_path)
    monkeypatch.setattr(ae, "ROOT", root)
    assert "golden.yaml 缺失" in (ae.check_gate_manifest() or "")


def test_injection_budget_catches_oversized_charter(tmp_path, monkeypatch):
    """坏例：charter.md 超 2048B 预算。"""
    root = _mini_root(tmp_path)
    (root / "charter.md").write_bytes(b"x" * 2500)
    (root / "AGENTS.md").write_bytes(b"ok")
    monkeypatch.setattr(ae, "ROOT", root)
    assert "2048" in (ae.check_injection_budget() or "")


def test_dependency_floors_catches_floor_above_pin(tmp_path, monkeypatch):
    """坏例：floor 高于钉版（声明了钉版不具备的能力）。"""
    root = _mini_root(tmp_path)
    (root / "registry").mkdir()
    (root / "registry" / "environment.yaml").write_text(
        "dependencies:\n  decord_pin: 0.9.8\ndependency_floors:\n  x: 9.9.9\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ae, "ROOT", root)
    assert "9.9.9" in (ae.check_dependency_floors() or "")


def test_negative_fixtures_catches_dead_member(monkeypatch):
    """坏例：豁免集合里出现不存在的成员（豁免失去依据）。"""
    import refs

    monkeypatch.setattr(refs, "NEGATIVE_FIXTURES", {"no_such_fixture_test.py"})
    assert "no_such_fixture_test.py" in (ae.check_negative_fixtures() or "")


def test_probe_age_catches_stale():
    """坏例：探针最后触达后已过 ≥2 个发版 tag（超龄即红）。"""
    counts = {"_probe_old.py": 2, "_probe_fresh.py": 1, "_probe_new.py": 0}
    assert ae._probe_stale(counts) == ["_probe_old.py"]


def test_worktree_clean_catches_untracked(monkeypatch):
    """坏例：工作树出现未纳管文件（porcelain 的 `?? ` 行）。"""
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="?? scratch_note.py\n M a.py\n", stderr=""
    )
    monkeypatch.setattr(ae.subprocess, "run", lambda *a, **k: fake)
    assert "scratch_note.py" in (ae.check_worktree_clean() or "")


def test_no_literal_tmp_catches_literal_string(tmp_path, monkeypatch):
    """坏例：产品文件里出现字面临时目录路径（拼接构造，测试自身不含该串）。"""
    root = _mini_root(tmp_path)
    (root / "video_ocr_engine" / "leaky.py").write_text(
        'import os\nP = "' + "/" + "tmp" + '/x"\n', encoding="utf-8"
    )
    monkeypatch.setattr(ae, "ROOT", root)
    assert "leaky.py" in (ae.check_no_literal_tmp() or "")


def test_no_literal_tmp_spares_docstring(tmp_path, monkeypatch):
    """精度：docstring 里提及该路径不算违规（禁的是字符串用法）。"""
    root = _mini_root(tmp_path)
    (root / "video_ocr_engine" / "clean.py").write_text(
        '"""说明：不要写 ' + "/" + "tmp" + ' 字面量。"""\n', encoding="utf-8"
    )
    monkeypatch.setattr(ae, "ROOT", root)
    assert ae.check_no_literal_tmp() is None


def test_ignore_liveness_catches_dead_entry(tmp_path, monkeypatch):
    """坏例：具体路径型 ignore 条目已无目标且无行内理由。"""
    root = _mini_root(tmp_path)
    (root / ".gitignore").write_text("tools/_gone_probe_dir/\n", encoding="utf-8")
    monkeypatch.setattr(ae, "ROOT", root)
    assert "_gone_probe_dir" in (ae.check_ignore_liveness() or "")


def test_ignore_liveness_reason_spares(tmp_path, monkeypatch):
    """精度：带行内理由的条目即使暂无目标也放行（豁免内联化）。"""
    root = _mini_root(tmp_path)
    (root / ".gitignore").write_text("tools/_future_out/  # 新写方落脚点\n", encoding="utf-8")
    monkeypatch.setattr(ae, "ROOT", root)
    assert ae.check_ignore_liveness() is None


def test_guard_declarations_catches_missing_negative(tmp_path, monkeypatch):
    """坏例：非 advisory 规则缺 negative（守卫声明不完备）。"""
    root = _mini_root(tmp_path)
    (root / "registry").mkdir()
    (root / "registry" / "rules.yaml").write_text(
        "- id: TEST_RULE\n  statement: s\n  enforcement: ci\n"
        '  executor: x\n  prevents: S0-A1\n  env: "无"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(ae, "ROOT", root)
    assert "缺 negative" in (ae.check_guard_declarations() or "")


def test_guard_declarations_catches_dead_negative_path(tmp_path, monkeypatch):
    """坏例：negative 声明指向不存在的测试文件。"""
    root = _mini_root(tmp_path)
    (root / "registry").mkdir()
    (root / "registry" / "rules.yaml").write_text(
        "- id: TEST_RULE\n  statement: s\n  enforcement: ci\n  executor: x\n"
        '  prevents: S0-A1\n  env: "无"\n  negative: "tests/nope.py"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(ae, "ROOT", root)
    assert "路径不存在" in (ae.check_guard_declarations() or "")
