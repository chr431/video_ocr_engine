"""bench 周期账本的单测。

`_flatten` 提取 resources.per_phase 的原始 cycles / cycles_e2e；
`_cycle_verdict` 的判读门槛（CI 排零 ∧ 符号多数）——纯 dict 口径，
不依赖 Windows / 真实引擎。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from _paths import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import bench  # tools/ 惯例：无包结构，路径注入后直名导入


def _rep(cyc_decode: int, cyc_e2e: int, thr: dict | None = None) -> dict:
    return {
        "resources": {
            "per_phase": {
                "calibrate": {"wall": 0.1, "threads": 1, "cycles": cyc_e2e // 10},
                "decode": {"wall": 1.0, "threads": 10, "cycles": cyc_decode},
                "ocr": {"wall": 0.5, "threads": 4, "cycles": cyc_e2e - cyc_decode},
                # 非 dict / 无 cycles 的行必须被跳过（Linux=来源缺席，无键≠0）
                "checkpoints": ["open", "calibrate", "decode", "ocr"],
                "_at_first_checkpoint": {"cpu_total_s": 0.0},
                "cycles_e2e": {
                    "total": cyc_e2e,
                    "span": "open..ocr",
                    **({"threads": thr} if thr else {}),
                },
            }
        }
    }


def test_flatten_extracts_cycle_ledger():
    flat = bench._flatten(_rep(100, 1_000, thr={"ocr": 300, "consumer": 700}))
    assert flat["cyc:decode"] == 100
    assert flat["cyc:e2e"] == 1_000
    assert flat["cyc:calibrate"] == 100  # cyc_e2e // 10
    assert flat["cyct:ocr"] == 300 and flat["cyct:consumer"] == 700
    assert "cyc:checkpoints" not in flat  # list 行不进表
    assert "cyc:_at_first_checkpoint" not in flat


def test_flatten_without_cycles_is_silent():
    rep = {"resources": {"per_phase": {"decode": {"wall": 1.0}}}}
    assert not {k for k in bench._flatten(rep) if k.startswith("cyc:")}


def _arm(e2e_fn, decode_fn, pairs: int = 6, thr=None) -> dict:
    """构造 _arm_reports 同构输入：config → [(label, round, valid, report)]。"""
    return {
        "h264-cpu": [
            ("lbl#%d@t" % i, 2, True, _rep(decode_fn(i), e2e_fn(i), thr=thr(i) if thr else None))
            for i in range(pairs)
        ]
    }


def test_cycle_verdict_significant_less(capsys):
    ra = _arm(lambda i: 1_000_000, lambda i: 600_000)
    rb = _arm(lambda i: 950_000, lambda i: 560_000)  # 每对都 −5% 上下
    bench._cycle_verdict("h264-cpu", ra, rb)
    out = capsys.readouterr().out
    assert "B 周期显著少" in out
    # 行名打印时剥掉 "cyc:"/"cyct:" 前缀（split(":",1)[1]）
    assert " e2e " in out and " decode " in out


def test_cycle_verdict_includes_thread_keys(capsys):
    ra = _arm(
        lambda i: 1_000_000, lambda i: 600_000, thr=lambda i: {"ocr": 100_000, "consumer": 800_000}
    )
    rb = _arm(
        lambda i: 900_000, lambda i: 540_000, thr=lambda i: {"ocr": 100_000, "consumer": 700_000}
    )
    bench._cycle_verdict("h264-cpu", ra, rb)
    out = capsys.readouterr().out
    assert "consumer" in out and "B 周期显著少" in out


def test_cycle_verdict_indecisive_on_noise(capsys):
    # 符号交替（偶对 B 多、奇对 B 少）→ 方向不稳 → 不可判定
    ra = _arm(lambda i: 1_000_000, lambda i: 600_000)
    rb = _arm(lambda i: 1_000_000 + (50_000 if i % 2 == 0 else -50_000), lambda i: 600_000)
    bench._cycle_verdict("h264-cpu", ra, rb)
    out = capsys.readouterr().out
    assert "不可判定" in out


def test_cycle_verdict_no_keys_is_silent(capsys):
    ra = {"h264-cpu": [("lbl#0@t", 2, True, {"spans": {}})]}
    rb = {"h264-cpu": [("lbl#0@t", 2, True, {"spans": {}})]}
    bench._cycle_verdict("h264-cpu", ra, rb)
    assert capsys.readouterr().out == ""


# ── P0-4：diff 逐指标配对口径（与 ab 同款统计纪律）──────────────────


def test_sig_stats_empty_on_single_sample():
    """n<2 无 SE 可算 → 返空 dict（调用方据此只报幅值，不伪造统计量）。"""
    assert bench._sig_stats([1.5]) == {}
    assert bench._sig_stats([]) == {}


def test_sig_stats_consistent_with_band():
    """_sig_stats 的 mean/se 与 _band 同源（两处判据不得各算一套）。

    `_band` 是展示口径（round 到 3 位）；`_sig_stats` 是内部判据口径
    （不 round）。故用近似比对而非相等。
    """
    vals = [1.0, -1.0, 2.0, -0.5, 0.5]
    st = bench._sig_stats(vals)
    b = bench._band(vals)
    assert st["mean"] == pytest.approx(b["mean"], abs=0.001)
    assert st["se"] == pytest.approx(b["se"], abs=0.001)
    assert st["pos"] == b["pos"]
    assert st["n"] == b["n"]


def test_load_keeps_round_reports_for_pairing():
    """_load 保留逐轮报告（P0-4 的 diff 配对键）。"""
    import bench as _b

    _b.REGISTRY.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "label": "p0-a",
                        "config": "h264-cpu",
                        "round": i,
                        "wall": 1.0 + i * 0.01,
                        "n_segments": 10,
                        "gpu": {"valid": True},
                        "report": {"spans": {"pipeline.run": {"sum": 1.0, "n": 1}}},
                    }
                )
                for i in (1, 2, 3)
            ]
        ),
        encoding="utf-8",
    )
    try:
        d = _b._load("p0-a")
        rr = d["h264-cpu"]["round_reports"]
        assert [r[0] for r in rr] == [1, 2, 3]
        assert all(r[1] for r in rr)
        assert rr[0][2]["spans"]["pipeline.run"]["sum"] == 1.0
    finally:
        _b.REGISTRY.unlink(missing_ok=True)


# ── P0-1：health 红旗 → 退出码 ───────────────────────────────────────


def _pi_report(engine_init: float, reuse: int = 1) -> dict:
    """最小 PI 报告（health 段按产品 health() 口径构造）。"""
    from video_ocr_engine.pipeline.report import health

    return {
        "spans": {},
        "counters": {"ocr.engine_reuse": reuse},
        "gauges": {"ocr.engine_init": engine_init},
        "health": health(
            {"gauges": {"ocr.engine_init": engine_init}, "counters": {"ocr.engine_reuse": reuse}}
        ),
    }


def test_health_gate_empty_when_clean():
    """无红旗 → 空表（不因"判据缺席"误报）。"""
    assert bench._health_gate([_pi_report(0.0001)], "L/cfg") == []
    assert bench._health_gate([], "L/cfg") == []
    assert bench._health_gate([None], "L/cfg") == []


def test_health_gate_dedups_with_count():
    """逐轮重复命中去重但保留次数（N 轮全红 ≠ 偶发一轮）。"""
    hits = bench._health_gate([_pi_report(0.3)] * 3, "L/cfg")
    assert len(hits) == 1
    lbl, k, v, lim, n = hits[0]
    assert lbl == "L/cfg" and k == "PI-10" and v == 0.3 and n == 3
    assert lim == 0.1


def test_health_gate_reads_product_single_source():
    """阈值来自产品 PI_LIMITS（bench 不得复制——防双事实源）。"""
    from video_ocr_engine.pipeline.report import PI_LIMITS

    hits = bench._health_gate([_pi_report(0.3)], "L/cfg")
    assert hits[0][3] == PI_LIMITS["PI-10"][1]


def test_diff_exit_code_fails_on_health_flag():
    """P0-1 端到端：红旗独立于墙钟判失败（registry 合成数据）。"""
    import argparse

    recs = []
    for i in (1, 2, 3):
        recs.append(
            {
                "label": "A-fail",
                "config": "h264-cpu",
                "round": i,
                "wall": 1.0,
                "n_segments": 10,
                "gpu": {"valid": True},
                "report": _pi_report(0.3),
            }
        )
        recs.append(
            {
                "label": "B-ok",
                "config": "h264-cpu",
                "round": i,
                "wall": 1.0,
                "n_segments": 10,
                "gpu": {"valid": True},
                "report": _pi_report(0.0001),
            }
        )
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        reg = Path(td) / "registry.jsonl"
        reg.write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8", newline="\n"
        )
        saved = bench.REGISTRY
        bench.REGISTRY = reg
        try:
            rc = bench.cmd_diff(
                argparse.Namespace(a="A-fail", b="B-ok", hard=5.0, warn=1.0, noise=1.3)
            )
        finally:
            bench.REGISTRY = saved
    assert rc == 1


def test_diff_exit_code_clean_when_no_flags():
    """对照：两臂都干净 → 墙钟判定为准（退出码 0）。"""
    import argparse
    import tempfile

    recs = []
    for lbl in ("C", "D"):
        for i in (1, 2, 3):
            recs.append(
                {
                    "label": lbl,
                    "config": "h264-cpu",
                    "round": i,
                    "wall": 1.0,
                    "n_segments": 10,
                    "gpu": {"valid": True},
                    "report": _pi_report(0.0001 if lbl == "C" else 0.0002),
                }
            )
    with tempfile.TemporaryDirectory() as td:
        reg = Path(td) / "registry.jsonl"
        reg.write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8", newline="\n"
        )
        saved = bench.REGISTRY
        bench.REGISTRY = reg
        try:
            rc = bench.cmd_diff(argparse.Namespace(a="C", b="D", hard=5.0, warn=1.0, noise=1.3))
        finally:
            bench.REGISTRY = saved
    assert rc == 0


def test_ab_health_gate_fires_even_without_hot_pairs():
    """P0-1 回归钉：热轮配对被时钟门禁全剔除时，health 红旗仍须判失败。

    真机实测背景：本轮 `ab` 输出的热轮因 GPU 掉频全被剔除（"无有效配对
    轮"），而健康判据若挂在墙钟判定之后就被连带跳过——引擎真跑了、
    report 真产了，正确性判据没理由缺席。
    """

    import bench as _b

    class _Args:
        telemetry = "full"
        a = "A"
        b = "B"

    # 两臂各 1 份报告，全部 gpu.valid=False（模拟全剔除）+ 一条 PI 红旗
    bad = _pi_report(0.3)  # engine_init 超限
    ra = {"h264-cpu": [("a#0@x", 2, False, bad)]}
    rb = {"h264-cpu": [("b#0@x", 2, False, bad)]}
    rc = _b._ab_health_gate(_Args, "h264-cpu", ra, rb, 0)
    assert rc == 1, "热轮全剔除时 health 红旗必须仍能判失败"
    # 干净对照：同结构无红旗 → rc 不变
    ok = _pi_report(0.0001)
    ra2 = {"h264-cpu": [("a#0@x", 2, False, ok)]}
    rb2 = {"h264-cpu": [("b#0@x", 2, False, ok)]}
    assert _b._ab_health_gate(_Args, "h264-cpu", ra2, rb2, 0) == 0
    # telemetry=off（无报告）→ 跳过，不改 rc
    _Args.telemetry = "off"
    assert _b._ab_health_gate(_Args, "h264-cpu", ra, rb, 0) == 0


# ── P2-1：多文件批量视图的单文件摘要（纯函数）────────────────────────


def _bulk_round(rnd: int, wall: float, valid: bool, text_sha: str = "T", segs: int = 10) -> dict:
    return {
        "round": rnd,
        "wall": wall,
        "n_segments": segs,
        "gpu": {"valid": valid},
        "shas": {"text_sha": text_sha, "seg_struct_sha": "S" + text_sha},
    }


def test_file_summary_hot_pool_excludes_cold_and_invalid():
    """热轮池 = round>1 ∧ 门禁 valid；冷轮与低时钟轮不得进中位与散布。"""
    rounds = [
        _bulk_round(1, 9.0, True),  # 冷轮（round=1）恒不进
        _bulk_round(2, 1.00, True),
        _bulk_round(3, 1.02, False),  # 低时钟：不进
    ]
    s = bench._file_summary(rounds)
    assert s["hot"] == [1.00] and s["median"] == 1.0 and not s["fell_back"]
    assert s["spread"] == 0.0  # 单热轮无散布


def test_file_summary_fallback_is_flagged():
    """门禁全剔 → 回退全收但标记 fell_back（散布不可比必须可见）。"""
    rounds = [_bulk_round(1, 9.0, True), _bulk_round(2, 1.0, False)]
    s = bench._file_summary(rounds)
    assert s["fell_back"] is True
    assert s["median"] == 5.0  # 冷轮混入——正是要标记的原因


def test_file_summary_detects_fingerprint_instability():
    """跨轮指纹不一致（含冷轮）→ 确定性缺陷的直接读数。"""
    rounds = [
        _bulk_round(1, 1.0, True, text_sha="A"),
        _bulk_round(2, 1.0, True, text_sha="A"),
        _bulk_round(3, 1.0, True, text_sha="B"),  # 同参数不同文本 = 缺陷
    ]
    s = bench._file_summary(rounds)
    assert s["text_stable"] is False and s["struct_stable"] is False
    ok = [_bulk_round(i, 1.0, True, text_sha="A") for i in (1, 2, 3)]
    s2 = bench._file_summary(ok)
    assert s2["text_stable"] and s2["struct_stable"] and s2["segs_stable"]


def test_file_summary_spread_uses_hot_pool():
    """散布仅在同池热轮间计算（冷轮不得污染——实测散布虚高 88% 的来源）。"""
    rounds = [
        _bulk_round(1, 1.26, True),  # 冷 1.26s
        _bulk_round(2, 0.50, True),
        _bulk_round(3, 0.52, True),
    ]
    s = bench._file_summary(rounds)
    assert s["spread"] == pytest.approx((0.52 - 0.50) / 0.51 * 100, rel=1e-3)


# ── P2-2：A/A 标定账本（指纹比对 + 漂移提示）──────────────────────────


def test_calibration_context_no_ledger(monkeypatch):
    """无标定账本 → 提示先跑 --aa（不冒充"一致"）。"""
    from pathlib import Path as _P

    monkeypatch.setattr(bench, "CALIBRATION", _P("does/not/exist.jsonl"))
    out = bench._calibration_context("h264-cpu", 1500)
    assert "无" in out and "--aa" in out


def test_calibration_context_fingerprint_mismatch(monkeypatch, tmp_path):
    """账本指纹与当前机器不符（换机）→ 显式告警"不可比、必须重标"。"""
    import json

    led = tmp_path / "cal.jsonl"
    led.write_text(
        json.dumps(
            {
                "ts": "2026-10-01T00:00:00",
                "fingerprint": "OTHER-GPU|1.0|cpu8",
                "config": "h264-cpu",
                "window": 1500,
                "n": 6,
                "mean": 0.1,
                "sd": 1.0,
                "se": 0.4,
                "p95abs": 1.5,
                "suggested_hard": 1.3,
                "suggested_hard_cold": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bench, "CALIBRATION", led)
    out = bench._calibration_context("h264-cpu", 1500)
    assert "指纹不符" in out and "--aa" in out


def test_calibration_context_same_machine_reports_age(monkeypatch, tmp_path):
    """同机 → 报年龄与建议阈值（供读者核对手里 --hard 是否为本机标定）。"""
    import json

    led = tmp_path / "cal.jsonl"
    led.write_text(
        json.dumps(
            {
                "ts": "2026-10-01T00:00:00",
                "fingerprint": bench._fp_key(bench._machine_fingerprint()),
                "config": "h264-cpu",
                "window": 1500,
                "n": 6,
                "mean": 0.1,
                "sd": 1.0,
                "se": 0.4,
                "p95abs": 1.5,
                "suggested_hard": 1.3,
                "suggested_hard_cold": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bench, "CALIBRATION", led)
    out = bench._calibration_context("h264-cpu", 1500)
    assert "同机" in out and "hard=1.3%" in out and "指纹不符" not in out


# ── P2-3：矩阵屏（全因子 + 差中差交互）────────────────────────────────


def test_matrix_cells_cartesian():
    """全因子格：2 旋钮 × 2 水平 = 4 格，按名排序。"""
    cells = bench._matrix_cells({"fill_width": [0, 224], "force_aspect": [0.0, 1.5]})
    assert len(cells) == 4
    keys = {bench._cell_key(c) for c in cells}
    assert keys == {
        "fill_width=0,force_aspect=0.0",
        "fill_width=0,force_aspect=1.5",
        "fill_width=224,force_aspect=0.0",
        "fill_width=224,force_aspect=1.5",
    }
    # 单旋钮：格数 = 水平数
    assert len(bench._matrix_cells({"fill_width": [0, 224]})) == 2


def _walls(cells: list, fn) -> dict:
    """合成 walls：fn(cell) -> 每轮墙钟列表。"""
    return {bench._cell_key(c): fn(c) for c in cells}


def test_matrix_effects_detects_pure_interaction():
    """构造式负测：无主效应、只有交互 → 差中差显著，主效应≈0。

    模型：wall = 1.0 × (1 + 0.02·(fw==0)·(fa==0))，即效应只在 (fw0,fa0) 出现。
    """
    cells = bench._matrix_cells({"fill_width": [0, 224], "force_aspect": [0.0, 1.5]})

    def fn(c):
        bump = 0.02 if (c["fill_width"] == 0 and c["force_aspect"] == 0.0) else 0.0
        return [1.0 * (1 + bump + 0.001 * i) for i in range(6)]

    eff = bench._matrix_effects(cells, _walls(cells, fn))
    d = eff["interaction"]["fill_width×force_aspect"]["diff_of_diff"]
    assert abs(d["mean"]) > 1.0, d  # 真交互必须被标出（幅值 %）
    assert d["pos"] == d["n"], "同向 6/6"  # 符号一致


def test_matrix_effects_no_interaction_when_additive():
    """对照：纯加性效应（各自独立）→ 差中差 ≈ 0（不误报交互）。"""
    cells = bench._matrix_cells({"fill_width": [0, 224], "force_aspect": [0.0, 1.5]})

    def fn(c):
        w = 1.0
        if c["fill_width"] == 0:
            w *= 1.03  # +3%
        if c["force_aspect"] == 0.0:
            w *= 1.05  # +5%
        return [w + 0.0005 * i for i in range(6)]  # 微扰防零方差

    eff = bench._matrix_effects(cells, _walls(cells, fn))
    d = eff["interaction"]["fill_width×force_aspect"]["diff_of_diff"]
    assert abs(d["mean"]) < 0.2, d  # 加性 ⇒ 差中差在噪声内
    # 主效应键 = 非基线水平（基线 = 各旋钮最小水平）。本模型里 fw=0 与
    # fa=0 各扣一个乘性因子，故 224 相对 0 为负（≈−2.9%）、1.5 相对
    # 0.0 为负（≈−4.8%）——两者都被标出即"主效应可读"。
    assert eff["main"]["fill_width=224"]["mean"] < -2.5
    assert eff["main"]["force_aspect=1.5"]["mean"] < -4.0


def test_matrix_cell_key_is_order_stable():
    """格键按名排序（同一格不同字典序产出同键——哈希/配对的前提）。"""
    a = bench._cell_key({"force_aspect": 1.5, "fill_width": 0})
    b = bench._cell_key({"fill_width": 0, "force_aspect": 1.5})
    assert a == b == "fill_width=0,force_aspect=1.5"


def test_sd_drift_note_fires_only_on_widened_band(monkeypatch, tmp_path):
    """P2-2 散布漂移提示：显著变宽才报；同量级/标定 sd=0 时静默。"""
    import json

    led = tmp_path / "cal.jsonl"
    led.write_text(
        json.dumps(
            {
                "ts": "2026-10-15T00:00:00",
                "fingerprint": bench._fp_key(bench._machine_fingerprint()),
                "config": "h264-cpu",
                "window": 3000,
                "n": 6,
                "mean": 0.1,
                "sd": 1.0,
                "se": 0.2,
                "p95abs": 1.5,
                "suggested_hard": 1.3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bench, "CALIBRATION", led)

    def _band(sd):
        return {"sd": sd, "se": 0.3, "mean": 0.0, "n": 5, "pos": 3}

    # 显著变宽（sd 1.0 → 3.0，阈 = 1.0 + max(0.6, 0.2) = 1.6）→ 报警
    assert "漂移" in bench._sd_drift_note("h264-cpu", 3000, _band(3.0)) or "散布" in (
        bench._sd_drift_note("h264-cpu", 3000, _band(3.0))
    )
    # 同量级 → 静默（不误报）
    assert bench._sd_drift_note("h264-cpu", 3000, _band(1.1)) == ""
    # 配置/窗口不符 → 不比
    assert bench._sd_drift_note("hevc-cpu", 3000, _band(9.0)) == ""
    assert bench._sd_drift_note("h264-cpu", 1500, _band(9.0)) == ""
    # 标定 sd=0（样本过少）→ 不比（避免 0 阈把抖动当漂移）
    led.write_text(
        json.dumps({"config": "h264-cpu", "window": 3000, "sd": 0.0, "se": 0.0}) + "\n",
        encoding="utf-8",
    )
    assert bench._sd_drift_note("h264-cpu", 3000, _band(9.0)) == ""
