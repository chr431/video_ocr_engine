"""自诊断（挂死/崩溃现场）门禁。

对应本轮需求「不依赖额外探针就能做死锁/崩溃诊断」：RunReport 只在 extract
正常返回时组装，挂死/硬崩溃时**什么都不输出**——本模块给两条独立通道，
这里逐条钉死其行为（全部合成，不需要视频/GPU）。
"""
from __future__ import annotations

import json
import time

from video_ocr_engine.domain.diagnostics import (
    NULL_DIAG, Progress, RunJournal, StallWatchdog, open_diagnostics)


def test_null_diag_is_a_noop():
    """未 opt-in 的路径必须零成本、零副作用。"""
    assert NULL_DIAG.armed is False
    assert NULL_DIAG.progress is None
    assert NULL_DIAG.tick("any.phase") is None
    NULL_DIAG.watch_depth("q", lambda: 0)      # 不抛、不记录
    assert NULL_DIAG.report() == {}
    assert NULL_DIAG.stop() == {}
    assert open_diagnostics(None) is NULL_DIAG
    assert open_diagnostics("") is NULL_DIAG


def test_progress_is_monotonic():
    p = Progress()
    assert p.seq == 0
    assert p.tick("a") == 1
    assert p.tick("b") == 2
    assert p.seq == 2


def test_watchdog_does_not_start_until_first_tick(tmp_path):
    """从未跑过 run → 不起线程（不能只为构造就留一个后台线程）。"""
    diag = open_diagnostics(str(tmp_path / "r.json"))
    try:
        assert diag.watchdog._th is None
        diag.tick("pipeline.decode")
        assert diag.watchdog._th is not None
    finally:
        diag.stop()


def test_watchdog_dumps_stall_scene(tmp_path):
    """核心行为：无进展超过阈值 → 自动落现场，且现场可读、含线程栈。"""
    diag = open_diagnostics(str(tmp_path / "r.json"))
    diag.watchdog._stall_s = 0.25          # 测试内收紧（不改模块默认）
    diag.watchdog._poll_s = 0.05
    try:
        diag.tick("pipeline.decode")       # 只打一次心跳，然后"挂住"
        deadline = time.monotonic() + 3.0
        dumps = []
        while time.monotonic() < deadline and not dumps:
            dumps = list(tmp_path.glob("r.json.diag.stall-*.json"))
            time.sleep(0.05)
        assert dumps, "看门狗未在阈值内落盘"
        payload = json.loads(dumps[0].read_text(encoding="utf-8"))
        assert payload["kind"] == "stall"
        assert payload["age_s"] >= 0.25
        assert payload["progress_seq"] == 1
        assert payload["threads"], "现场必须带线程栈"
        # 主线程至少要出现，且栈不可为空（否则现场没用）
        assert any(v["stack"] for v in payload["threads"].values())
        assert "depths" in payload
    finally:
        rep = diag.stop()
    assert rep["stalls"] >= 1
    assert rep["dumps"][0]["path"].endswith(".json")


def test_watchdog_silent_while_progressing(tmp_path):
    """持续有进展时**不得**误报——否则现场文件会被垃圾淹没。"""
    diag = open_diagnostics(str(tmp_path / "r.json"))
    diag.watchdog._stall_s = 0.35
    diag.watchdog._poll_s = 0.05
    try:
        end = time.monotonic() + 0.8
        while time.monotonic() < end:
            diag.tick("pipeline.producer")
            time.sleep(0.03)
    finally:
        rep = diag.stop()
    assert rep["stalls"] == 0
    assert not list(tmp_path.glob("*.stall-*.json"))


def test_depth_probe_only_called_on_stall(tmp_path):
    """深度探针不得进稳态路径：正常推进时调用次数为 0。"""
    diag = open_diagnostics(str(tmp_path / "r.json"))
    diag.watchdog._stall_s = 10.0
    calls = {"n": 0}

    def qsize():
        calls["n"] += 1
        return 7

    diag.watch_depth("ocr_in", qsize)
    try:
        for _ in range(5):
            diag.tick("pipeline.ocr")
            time.sleep(0.02)
    finally:
        diag.stop()
    assert calls["n"] == 0


def test_journal_is_append_only_jsonl(tmp_path):
    path = tmp_path / "j.jsonl"
    j = RunJournal(path)
    j.add("run_start", pid=1234)
    j.add("phase", detail="pipeline.decode")
    assert j.flush() == 2
    j.add("phase", detail="pipeline.ocr")
    j.close()
    lines = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]
    assert len(lines) == 3
    recs = [json.loads(ln) for ln in lines]
    assert recs[0]["event"] == "run_start" and recs[0]["pid"] == 1234
    assert recs[1]["detail"] == "pipeline.decode"
    assert all("t" in r and "wall" in r for r in recs)
    assert j.report()["dropped"] == 0


def test_journal_is_bounded(tmp_path):
    """崩溃日志必须有界：不能因为长期跑而吃光内存。"""
    j = RunJournal(tmp_path / "b.jsonl")
    for i in range(5000):
        j.add("bulk", i=i)
    assert len(j._buf) <= 4096           # 环形缓冲上限
    j.flush()
    assert j.report()["written"] == 4096
    assert j.report()["dropped"] >= 904
    j.close()


def test_tick_overhead_is_sub_microsecond_scale():
    """PI-15：心跳必须廉价（这是逐相位的热路径调用）。"""
    p = Progress()
    n = 20000
    t0 = time.perf_counter()
    for i in range(n):
        p.tick("pipeline.decode")
    per = (time.perf_counter() - t0) / n * 1e6
    assert per < 20.0, "Progress.tick %.2f µs/次，超出预算" % per


def test_report_gains_diagnostics_key_only_when_armed():
    """schema 演进只加键、且未 arming 时不写空值冒充。"""
    from video_ocr_engine.domain.metrics import make_metrics
    from video_ocr_engine.pipeline.report import REPORT_VERSION, build_report
    m = make_metrics("std")
    m.record_span("pipeline.run", 1.0)
    plain = build_report(m, wall=1.0)
    assert REPORT_VERSION == 4
    assert "diagnostics" not in plain
    armed = build_report(m, wall=1.0, diagnostics={"armed": True, "stalls": 0})
    assert armed["diagnostics"]["armed"] is True
