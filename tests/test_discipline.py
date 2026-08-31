"""把两个审计脚本接进测试 —— 让纪律由 CI 守，而不是靠自觉。

为什么需要
----------
本项目已经三次栽在"规矩写在文档里但没人守"上：

1. CLAUDE.md 长到 70 KB（≈15–21K tokens）才被发现 —— 它每个会话开头被注入。
2. "PERFORMANCE.md 只能用二进制编辑"传了几轮，实测是**伪规矩**，
   真正的病灶（934 个裸 CR）一直没人拆。
3. tools/INDEX.md 手写的数字不到一天就漂了。

共同点是**没有自动化检查**。所以这里把 `tools/_probe_discipline_audit.py`
（12 项）和 `tools/_probe_index_audit.py`（索引数字一致性）接成测试，
退出码非 0 即失败。

设计取舍
--------
- 用子进程跑而不是 import 后调函数：审计脚本会 `git status` / 遍历文件树，
  子进程隔离更干净，也不污染本进程的 `sys.path`。
- 失败时把审计输出**整段贴进断言消息**：光看"退出码 1"没法定位，
  审计脚本自己已经把违规项列得很清楚。
- 不做 `pytest.mark.slow`：两个加起来约 0.9s，够快。
"""

from __future__ import annotations

import subprocess
import sys

from _paths import ROOT

AUDITS = [
    ("tools/_probe_discipline_audit.py", "项目纪律审计（12 项）"),
    ("tools/_probe_index_audit.py", "tools/INDEX.md 数字一致性"),
]


def _run(rel: str) -> tuple[int, str]:
    p = ROOT / rel
    r = subprocess.run([sys.executable, str(p)],
                       cwd=str(ROOT), capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _check(rel: str, title: str) -> None:
    code, out = _run(rel)
    assert code == 0, (
        "%s 未通过（退出码 %d）。\n"
        "修掉下面列出的项，或确认属存量后重新登记 baseline。\n"
        "手动复现：python %s\n\n%s" % (title, code, rel, out)
    )


def test_discipline_audit_passes() -> None:
    """项目纪律审计全绿。"""
    _check(*AUDITS[0])


def test_tools_index_audit_passes() -> None:
    """tools/INDEX.md 里的每个数字都与磁盘一致（防止索引漂移）。"""
    _check(*AUDITS[1])
