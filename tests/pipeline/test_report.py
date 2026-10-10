"""RunReport schema 快照 + §8.6 r5 资源层（L1 差分 / L2 NVML）守卫。

对应 §12 验收项"运行报告 schema"：**report_version 快照一致**、schema 演进
必须 bump、`metrics_coverage` 零未注册指标；以及资源层的两条实测踩坑防线：

1. 产品代码**不得 import psutil**——Windows 下 `Process.threads()` /
   `num_threads()` 实测 **43ms / 5.1ms 每次**（本机 `_probe_run_setup_cost`
   同批数据），一次就吃满 PI-15 的 0.1% 预算；资源层因此只用 stdlib。
2. 不可用的读数必须写成 `"unavailable:…"`，**不得用 None/0 冒充真值**。
"""
from __future__ import annotations

import sys
import time

import pytest

from _paths import PKG
from video_ocr_engine.domain.metrics import NULL_METRICS, Metrics
from video_ocr_engine.domain.resources import (NvmlSampler, ResourceProbe,
                                               _HostCounters)
from video_ocr_engine.pipeline.report import (REPORT_VERSION, RunReport,
                                              build_report, health)

# v2 快照：新增键只允许**加**，改语义/删键必须 bump REPORT_VERSION。
# v3（2026-09-13）：加 `diagnostics`（自诊断，仅 opt-in 时出现）→ 已 bump。
REPORT_KEYS = {
    "report_version", "tier", "wall_s", "spans", "counters", "gauges",
    "health", "environment", "pipeline", "degradations",
}
RESOURCE_KEYS = {"sources", "per_phase", "notes"}


def _std_metrics():
    m = Metrics("full")
    with m.span("pipeline.run"):
        m.record_span("pipeline.decode", 0.12)
        m.record_span("pipeline.ocr", 0.05)
        m.counter("ocr.chunks", 3)
        m.counter("ocr.syncs", 6)
        m.gauge("ocr.engine_init", 0.0002)
        m.checkpoint("open")
        m.checkpoint("calibrate")
        m.checkpoint("decode")
    return m


# ── schema 快照 ────────────────────────────────────────────────────────
def test_report_version_is_pinned():
    # v1→v2：+resources/+hardware；v2→v3：+diagnostics；v3→v4：+span_relations
    # 与 `<parent>_other` 派生 span；v4→v5：+histograms/histograms_meta
    # （full 档专属）；v5→v6：+hybrid（fork 遥测直通）；v6→v7：resources
    # per_phase 行 +cycles 原始差分与 cycles_e2e（C-63 判据用；均只加键）
    assert REPORT_VERSION == 10  # v8 +thr；v9 cores_avg 回退；v10 +thr_foreign


def test_report_schema_snapshot():
    rep = build_report(_std_metrics(), wall=0.5, config_digest="deadbeef",
                       n_segments=7, backend="decord/GPU",
                       ocr_backend="tensorrt")
    assert set(rep) == REPORT_KEYS | {"resources", "span_relations"}
    assert rep["report_version"] == REPORT_VERSION
    assert rep["tier"] == "full"
    assert rep["wall_s"] == pytest.approx(0.5)
    assert rep["spans"]["pipeline.decode"]["sum"] == pytest.approx(0.12)
    assert rep["spans"]["pipeline.run"]["n"] == 1
    assert rep["counters"]["ocr.chunks"] == 3
    assert rep["gauges"]["ocr.engine_init"] == pytest.approx(0.0002)
    assert rep["pipeline"]["n_segments"] == 7
    assert rep["pipeline"]["config_digest"] == "deadbeef"
    # v4：关系表自述（路径 + 派生口径），供读者复核 _other 算术
    assert rep["span_relations"]["path"] == "host"
    assert "pipeline.consumer" in rep["span_relations"]["parent_children"]
    # health 每项都是 {metric, value, limit, ok} 形态（§13.2 N-5 可判定）
    assert "PI-10" in rep["health"]
    for k, v in rep["health"].items():
        assert {"metric", "value", "limit", "ok"} <= set(v), k
    assert rep["environment"]["python"].startswith("3.")
    assert set(rep["resources"]) == RESOURCE_KEYS


def test_runreport_view_and_result_report():
    """R2（0.16.0）：RunReport 类型化只读视图 + result.report 派生读面。

    视图零复制（嵌套 section 与 data 共享）；序列化唯一形态仍是 dict
    （to_dict() 与 meta['report'] 同源）；可选段缺席 = None ≠ 空值冒充。
    """
    from video_ocr_engine._result_types import ExtractionResult

    rep = build_report(_std_metrics(), wall=0.5, config_digest="deadbeef",
                       n_segments=7, backend="decord/GPU",
                       ocr_backend="tensorrt")
    view = RunReport(rep)
    assert view.report_version == REPORT_VERSION
    assert view.tier == "full"
    assert view.wall_s == pytest.approx(0.5)
    assert view.spans["pipeline.decode"]["sum"] == pytest.approx(0.12)
    assert view.counters["ocr.chunks"] == 3
    assert view.pipeline["n_segments"] == 7
    assert view.degradations == []
    assert view.health is rep["health"]            # 零复制：section 共享
    assert view.to_dict() == rep                   # 序列化同源（v6 dict）
    # 可选段（本构造无 hardware/histograms/hybrid）缺席 = None
    assert view.hardware is None
    assert view.histograms is None
    assert view.hybrid is None
    with pytest.raises(AttributeError):
        view.tier = "full"                          # frozen：只读

    r = ExtractionResult(meta={"report": rep})
    assert r.report is not None
    assert r.report.tier == "full"
    # off 档（meta 无 report 键）：None，不冒充空值
    assert ExtractionResult(meta={}).report is None
    assert ExtractionResult().report is None


def test_other_span_derivation_host_vs_gpu():
    """v4 缺口派生：host 才有 consumer 关系；子项缺席记 0；不越出父和。"""
    m = _std_metrics()
    m.record_span("pipeline.consumer", 2.0)
    m.record_span("decode.batch", 1.2)
    rep = build_report(m, wall=2.5, span_path="host")
    other = rep["spans"]["pipeline.consumer_other"]
    assert other["n"] == 1
    assert other["sum"] == pytest.approx(0.8)        # 2.0 − 1.2 − 缺席记 0
    assert "min" not in other and "p50" not in other  # 缺口不伪造分位数
    # gpu 路径无 consumer 关系 → 不派生该键
    rep2 = build_report(_std_metrics(), wall=2.5, span_path="gpu")
    assert "pipeline.consumer_other" not in rep2["spans"]
    # ocr.infer ⊃ ctc_decode 两路径同构
    m2 = _std_metrics()
    m2.record_span("ocr.infer", 0.5)
    m2.record_span("ocr.ctc_decode", 0.05)
    rep3 = build_report(m2, wall=1.0, span_path="gpu")
    assert rep3["spans"]["ocr.infer_other"]["sum"] == pytest.approx(0.45)


def test_hybrid_section_from_fork_passthrough():
    """v6：fork 遥测直通段——extra 注入即出现；缺席≠空值。"""
    m = _std_metrics()
    plain = build_report(m, wall=1.0)
    assert "hybrid" not in plain
    st = {"frames_c": 100, "frames_g": 200, "hol_hist_c": [0] * 24,
          "busy_cpu_us": 12345}
    rep = build_report(m, wall=1.0, extra={"hybrid": st})
    assert rep["hybrid"] is st and rep["hybrid"]["frames_g"] == 200


def test_host_consume_feed_relations_and_histograms():
    """W1/W2：consume_feed ⊃ merge_pair+q_put（宿主）；full 档直方图段。"""
    m = _std_metrics()
    m.record_span("pipeline.consume_feed", 0.60)
    m.gauge("segment.merge_pair", 0.25)          # TOTALS 汇总为 gauge
    m.gauge("pipeline.q_put_block", 0.05)
    rep = build_report(m, wall=1.0, span_path="host")
    other = rep["spans"]["pipeline.consume_feed_other"]
    assert other["sum"] == pytest.approx(0.30)
    # std 档不写 histograms（缺席 ≠ 空值）
    assert "histograms" not in rep
    mf = Metrics("full")
    mf.histogram("pipeline.consume_feed_hist", 0.001)
    mf.histogram("pipeline.consume_feed_hist", 0.002)
    mf.histogram("pipeline.consume_feed_hist", 0.5)
    repf = build_report(mf, wall=1.0, span_path="host")
    h = repf["histograms"]["pipeline.consume_feed_hist"]
    assert h["n"] == 3 and sum(h["buckets"]) == 3
    assert repf["histograms_meta"]["n_buckets"] == len(h["buckets"])
    # 1ms 与 2ms 同桶（×2 分箱），0.5s 远离
    assert h["buckets"] == h["buckets"] and h["buckets"][-1] == 0


def test_off_tier_emits_no_report():
    assert build_report(NULL_METRICS, wall=1.0) == {}


def test_hardware_section_only_when_sampled():
    m = _std_metrics()
    assert "hardware" not in build_report(m, wall=1.0)
    rep = build_report(m, wall=1.0, hardware={"sources": "nvml", "n": 12})
    assert rep["hardware"]["sources"] == "nvml"


# ── L1 资源差分 ────────────────────────────────────────────────────────
def test_checkpoint_is_noop_off_and_lazy_std():
    assert NULL_METRICS.resource_report() is None
    NULL_METRICS.checkpoint("decode")            # 不得抛
    fresh = Metrics("full")
    # 惰性：没采过边界就不建探针（宿主/短路径零成本）
    assert fresh._resources is False
    assert fresh.resource_report() is None
    fresh.checkpoint("open")
    assert isinstance(fresh._resources, ResourceProbe)


def test_per_phase_deltas_are_sane():
    p = ResourceProbe()
    p.checkpoint("open")
    t = time.perf_counter()
    while time.perf_counter() - t < 0.05:        # 制造一段真实 CPU 占用
        sum(range(5000))
    p.checkpoint("decode")
    r = p.per_phase()
    assert r["checkpoints"] == ["open", "decode"]
    row = r["decode"]
    assert row["wall"] > 0
    # v9：cores_avg（tick 口径）是 cycles 缺席时的回退键——Windows 上
    # cycles 在场故 cores_avg 不发（权威=cores_avg_cycles）
    if sys.platform == "win32":
        assert "cores_avg" not in row
    else:
        assert row["cores_avg"] >= 0.0
    # P2a：cycle 口径的核数（无 15.625ms tick 量化）；单线程忙等相位 ≈1 核，
    # 量测含主线程以外的极小开销，放宽到 (0, 2)。
    if sys.platform == "win32":  # QueryProcessCycleTime 仅 Windows
        assert 0.0 < row.get("cores_avg_cycles", 0.0) < 2.0
        # v7 周期账本（C-63）：原始 cycles 差分与全程账本必须在场
        assert isinstance(row["cycles"], int) and row["cycles"] > 0
        e2e = r["cycles_e2e"]
        assert e2e["total"] == row["cycles"] and e2e["span"] == "open..decode"
    assert row["threads"] >= 1
    if "rss_delta_mib" in row:
        assert isinstance(row["rss_delta_mib"], float)


class _StubCountersThr:
    """per_phase 线程差分的台架：thr 样本按调用序给定。"""

    def __init__(self, steps):
        self._steps = list(steps)
        self._i = 0

    def sample(self):
        t = {"t": float(self._i), "cpu": 1.0 * self._i,
             "threads": 1, "thr": dict(self._steps[self._i]),
             "cycles": 1_000_000 * self._i,
             "rss": None, "rb": None, "wb": None, "vram": None,
             "sm_mhz": None}
        self._i += 1
        return t


def test_per_phase_thread_deltas_and_duty():
    # 进程级账本单例会被同进程先前跑过引擎的测试注册污染（consumer/ocr
    # 等名字残留）→ 换上隔离空账本，让 stub 的 thr 样本不被真实采样覆盖。
    # （仅 win32：非 Windows 无账本，thread_ledger() 恒 None 本就不覆盖。）
    import video_ocr_engine.domain.resources as _res
    _saved = _res._THREAD_LEDGER
    if sys.platform == "win32":
        _res._THREAD_LEDGER = _res.ThreadLedger()
    try:
        p = ResourceProbe()
        p._counters = _StubCountersThr(
            [{"ocr": 100_000, "consumer": 500_000},
             {"ocr": 300_000, "consumer": 900_000}])
        p.checkpoint("open", threads=True)
        p.checkpoint("decode", threads=True)
        r = p.per_phase()
    finally:
        if sys.platform == "win32":
            _res._THREAD_LEDGER = _saved
    row = r["decode"]
    assert row["thr"] == {"ocr": 200_000, "consumer": 400_000}
    # f_run = Δcycles/Δcpu = 1_000_000 → duty = Δthr/(f_run×Δwall)
    assert row["thr_duty"] == {"ocr": 0.2, "consumer": 0.4}
    assert r["cycles_e2e"]["threads"] == {"ocr": 200_000,
                                          "consumer": 400_000}


def test_thread_ledger_roundtrip_win32():
    if sys.platform != "win32":
        import video_ocr_engine.domain.resources as _res
        _res.register_thread("x")            # 非 Windows：静默 no-op
        return
    import threading
    import time as _time
    import video_ocr_engine.domain.resources as _res
    led = _res.thread_ledger()
    assert led is not None
    stop = threading.Event()
    t = threading.Thread(target=stop.wait)
    t.start()
    try:
        # 句柄须在存活期打开（死线程 TID 不可再 OpenThread）——懒开设计
        # 依赖"采样发生在 run 边界、线程存活"这一生命周期事实。
        _res.register_thread("probe-tmp", t)
        s0 = led.sample()
        _time.sleep(0.02)
        s1 = led.sample()
        assert s1["probe-tmp"] >= s0["probe-tmp"] >= 0
    finally:
        stop.set()
        t.join()
    led.register("probe-tmp", 0)          # 换 ident → 缓存句柄作废 → 打不开除名
    assert "probe-tmp" not in led.sample()


def test_thread_ledger_sample_all_win32():
    """v10 全线程快照：含自身线程，且与命名采样同源可互校。"""
    if sys.platform != "win32":
        return
    import threading
    import video_ocr_engine.domain.resources as _res
    led = _res.thread_ledger()
    assert led is not None
    stop = threading.Event()
    t = threading.Thread(target=stop.wait)
    t.start()
    try:
        _res.register_thread("probe-all", t)
        all1 = led.sample_all()
        assert threading.current_thread().ident in all1
        assert t.ident in all1
        named = led.sample()
        # 命名线程在两套采样里都出现（同 TID 同值口径）
        assert all1[t.ident] == named["probe-all"]
    finally:
        stop.set()
        t.join()


def test_phase_cap_is_honored():
    p = ResourceProbe()
    for i in range(p._MAX_PHASES + 50):
        p.checkpoint("p%d" % i)
    assert len(p._rows) == p._MAX_PHASES


def test_unavailable_sources_are_labeled_not_silent():
    src = _HostCounters().sources
    # 每个来源都必须有"真值出处"或"unavailable:原因"，不许空白/None
    for key in ("cpu", "rss", "disk"):
        assert key in src
        assert src[key] and src[key] != "None"
    assert src["cpu"] == "time.process_time"
    probe_src = ResourceProbe().sources
    assert probe_src["threads"] == "threading.active_count"
    assert probe_src["vram"].startswith(("cudaMemGetInfo", "unavailable",
                                         "not-probed"))


# ── L2 NVML 降级 ───────────────────────────────────────────────────────
def test_nvml_sampler_degrades_without_nvml(monkeypatch):
    import ctypes as _c
    from video_ocr_engine.domain import resources as RS

    def boom(_name):
        raise OSError("no nvml on this box")

    # NVML 会话是**进程级**的（`_NVML` 缓存），先清掉才会真的重开→走失败分支
    monkeypatch.setitem(RS._NVML, "state", None)
    monkeypatch.setitem(RS._NVML, "why", "")
    monkeypatch.setattr(_c, "CDLL", boom)

    s = NvmlSampler(interval_s=0.05)
    assert s._th is None
    s.start()
    assert s._th is None                          # 绝不留下半死线程
    out = s.stop()
    assert out["sources"] == "nvml_unavailable"
    assert "no nvml" in out["error"]
    assert out["n"] == 0
    # 失败判定同样落在环境指纹上：不抛异常，回退 nvidia-smi
    monkeypatch.setitem(RS._NVML, "state", None)
    assert RS.gpu_identity() is None


def test_nvml_sampler_stops_thread():
    s = NvmlSampler(interval_s=0.05)
    s.start()
    time.sleep(0.25)
    out = s.stop()
    assert s._th is None
    assert out["sources"] in ("nvml", "nvml_unavailable")
    if out["sources"] == "nvml":
        # 有卡即应采到 GPU 利用率；NVDEC 可能为 0（非解码期）但字段必须存在
        assert out["gpu_util_pct"] is not None
        assert "nvdec_util_pct" in out


def test_full_tier_only_starts_sampler():
    # 两档化后：off 不得建采样线程；full 允许（L2 语义）
    m = Metrics("off")
    m.start_hardware()
    assert m._hw is None
    assert m.hardware_report() is None


# ── 产品代码依赖纪律 ───────────────────────────────────────────────────
_PSUTIL_HOT_CALLS = {"threads", "num_threads"}


def test_no_psutil_thread_enumeration_in_product_code():
    """实测地雷防线：`psutil.Process.threads()`/`num_threads()` 本机 **43ms /
    5.1ms 每次**（Windows 逐线程开句柄），一次就吃满 PI-15 的 0.1% 预算。

    资源层（`domain/resources.py`）因此一律 stdlib，线程数用
    `threading.active_count()`（0.1µs）。注意 `num_threads=` 作为 **decord
    关键字参数**是合法的，这里只禁"属性调用"形态。
    """
    import ast as _ast
    hits = []
    for p in sorted(PKG.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        tree = _ast.parse(p.read_text(encoding="utf-8"))
        for n in _ast.walk(tree):
            if (isinstance(n, _ast.Call)
                    and isinstance(n.func, _ast.Attribute)
                    and n.func.attr in _PSUTIL_HOT_CALLS):
                hits.append("%s:%d .%s()" % (p.name, n.lineno, n.func.attr))
    assert not hits, "产品代码调用了昂贵线程枚举：%s" % hits


def test_resource_layer_is_dependency_free():
    """D7（无新依赖）：资源层与遥测/报告不得引入 psutil。"""
    for sub in ("domain", "pipeline"):
        for p in sorted((PKG / sub).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            assert "import psutil" not in p.read_text(encoding="utf-8"), p.name


def test_health_judges_only_self_evident_items():
    # 缺项不炸：什么都没采到时 health 为空，但报告仍能出
    assert health({"gauges": {}, "counters": {}}) == {}
    # PI-10 只在**热池**时判（冷启动 0.31–0.42s 是既有事实，不是回归）
    cold = health({"gauges": {"ocr.engine_init": 0.39},
                   "counters": {"ocr.engine_reuse": 0}})
    assert cold["PI-10"]["ok"] is True and cold["PI-10"]["hot_pool"] is False
    slow_hot = health({"gauges": {"ocr.engine_init": 0.39},
                       "counters": {"ocr.engine_reuse": 1}})
    assert slow_hot["PI-10"]["ok"] is False
    from video_ocr_engine.pipeline.report import red_flags
    assert red_flags({"health": slow_hot}) == ["PI-10"]
    assert red_flags({"health": cold}) == []
    # PI-3：每 chunk 同步次数 >3 判失败
    sync = health({"gauges": {"ocr.syncs_per_chunk": 6.0}, "counters": {}})
    assert sync["PI-3"]["ok"] is False
