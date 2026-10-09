"""bench 周期账本接线（C-63 产品化，2026-10-09）的单测。

`_flatten` 提取 resources.per_phase 的原始 cycles / cycles_e2e；
`_cycle_verdict` 的判读门槛（CI 排零 ∧ 符号多数）——纯 dict 口径，
不依赖 Windows / 真实引擎。
"""
from __future__ import annotations

import sys

from _paths import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import bench  # noqa: E402  tools/ 惯例：无包结构，路径注入后直名导入


def _rep(cyc_decode: int, cyc_e2e: int) -> dict:
    return {"resources": {"per_phase": {
        "calibrate": {"wall": 0.1, "threads": 1, "cycles": cyc_e2e // 10},
        "decode": {"wall": 1.0, "threads": 10, "cycles": cyc_decode},
        "ocr": {"wall": 0.5, "threads": 4, "cycles": cyc_e2e - cyc_decode},
        # 非 dict / 无 cycles 的行必须被跳过（Linux=来源缺席，无键≠0）
        "checkpoints": ["open", "calibrate", "decode", "ocr"],
        "_at_first_checkpoint": {"cpu_total_s": 0.0},
        "cycles_e2e": {"total": cyc_e2e, "span": "open..ocr"},
    }}}


def test_flatten_extracts_cycle_ledger():
    flat = bench._flatten(_rep(100, 1_000))
    assert flat["cyc:decode"] == 100
    assert flat["cyc:e2e"] == 1_000
    assert flat["cyc:calibrate"] == 100          # cyc_e2e // 10
    assert "cyc:checkpoints" not in flat          # list 行不进表
    assert "cyc:_at_first_checkpoint" not in flat


def test_flatten_without_cycles_is_silent():
    rep = {"resources": {"per_phase": {"decode": {"wall": 1.0}}}}
    assert not {k for k in bench._flatten(rep) if k.startswith("cyc:")}


def _arm(e2e_fn, decode_fn, pairs: int = 6) -> dict:
    """构造 _arm_reports 同构输入：config → [(label, round, valid, report)]。"""
    return {"h264-cpu": [
        ("lbl#%d@t" % i, 2, True,
         _rep(decode_fn(i), e2e_fn(i))) for i in range(pairs)]}


def test_cycle_verdict_significant_less(capsys):
    ra = _arm(lambda i: 1_000_000, lambda i: 600_000)
    rb = _arm(lambda i: 950_000, lambda i: 560_000)   # 每对都 −5% 上下
    bench._cycle_verdict("h264-cpu", ra, rb)
    out = capsys.readouterr().out
    assert "B 周期显著少" in out
    # 行名打印时剥掉 "cyc:" 前缀（k[4:]）
    assert " e2e " in out and " decode " in out


def test_cycle_verdict_indecisive_on_noise(capsys):
    # 符号交替（偶对 B 多、奇对 B 少）→ 方向不稳 → 不可判定
    ra = _arm(lambda i: 1_000_000, lambda i: 600_000)
    rb = _arm(lambda i: 1_000_000 + (50_000 if i % 2 == 0 else -50_000),
              lambda i: 600_000)
    bench._cycle_verdict("h264-cpu", ra, rb)
    out = capsys.readouterr().out
    assert "不可判定" in out


def test_cycle_verdict_no_keys_is_silent(capsys):
    ra = {"h264-cpu": [("lbl#0@t", 2, True, {"spans": {}})]}
    rb = {"h264-cpu": [("lbl#0@t", 2, True, {"spans": {}})]}
    bench._cycle_verdict("h264-cpu", ra, rb)
    assert capsys.readouterr().out == ""
