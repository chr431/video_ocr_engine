"""新制度守卫的测试接线——refs / render / 结构化审计（取代旧 test_discipline）。

设计取舍沿用旧件：子进程隔离（守卫会 git/遍历文件树），失败时整段输出贴进
断言消息。这三条是 check.py 链的快检步，接进 pytest 让 CI（无 decord/GPU）
也能守住制度面。
"""

from __future__ import annotations

import subprocess
import sys

from _paths import ROOT

GUARDS = [
    ("tools/refs.py", ["check"], "引用解析（substrate）"),
    ("tools/render.py", ["--check"], "渲染一致性（四视图）"),
    ("tools/_audit_ext.py", [], "结构化审计（按名注册）"),
]


def _run(rel: str, args: list) -> tuple[int, str]:
    r = subprocess.run(
        [sys.executable, str(ROOT / rel), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_refs_check_passes() -> None:
    code, out = _run(*GUARDS[0][:2])
    assert code == 0, "refs check 未通过：\n%s" % out


def test_render_check_passes() -> None:
    code, out = _run(*GUARDS[1][:2])
    assert code == 0, "render --check 未通过（重渲染：python tools/render.py --write）：\n%s" % out


def test_audit_ext_passes() -> None:
    code, out = _run(*GUARDS[2][:2])
    assert code == 0, "结构化审计未通过：\n%s" % out


def test_guards_can_fail() -> None:
    """负测的负测：守卫对已知坏输入必须非 0（R-guard-negative 的自证）。

    refs check --files 指向一个含悬空 token 的临时文件，必须报 UNKNOWN。
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.md"
        bad.write_text("见 C-999 与 R-nope", encoding="utf-8")
        code, out = _run("tools/refs.py", ["check", "--files", str(bad)])
        assert code != 0 and "C-999" in out, "refs 对坏样例必须失败，实际输出：\n%s" % out
