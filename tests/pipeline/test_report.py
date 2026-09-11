"""RunReport schema 快照 + §8.6 r5 资源层（L1 差分 / L2 NVML）守卫。

对应 §12 验收项"运行报告 schema"：**report_version 快照一致**、schema 演进
必须 bump、`metrics_coverage` 零未注册指标；以及资源层的两条实测踩坑防线：

1. 产品代码**不得 import psutil**——Windows 下 `Process.threads()` /
   `num_threads()` 实测 **43ms / 5.1ms 每次**（本机 `_probe_run_setup_cost`
   同批数据），一次就吃满 PI-15 的 0.1% 预算；资源层因此只用 stdlib。
2. 不可用的读数必须写成 `"unavailable:…"`，**不得用 None/0 冒充真值**。
"""
from __future__ import annotations

import time

import pytest

from _paths import PKG
from video_ocr_engine.domain.metrics import NULL_METRICS, Metrics
from video_ocr_engine.domain.resources import (NvmlSampler, ResourceProbe,
                                               _HostCounters)
from video_ocr_engine.pipeline.report import (REPORT_VERSION, build_report,
                                              health)

# v2 快照：新增键只允许**加**，改语义/删键必须 bump REPORT_VERSION。
REPORT_KEYS = {
    "report_version", "tier", "wall_s", "spans", "counters", "gauges",
    "health", "environment", "pipeline", "degradations",
}
RESOURCE_KEYS = {"sources", "per_phase", "notes"}


def _std_metrics():
    m = Metrics("std")
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
    assert REPORT_VERSION == 2


def test_report_schema_snapshot():
    rep = build_report(_std_metrics(), wall=0.5, config_digest="deadbeef",
                       n_segments=7, backend="decord/GPU",
                       ocr_backend="tensorrt")
    assert set(rep) == REPORT_KEYS | {"resources"}
    assert rep["report_version"] == REPORT_VERSION
    assert rep["tier"] == "std"
    assert rep["wall_s"] == pytest.approx(0.5)
    assert rep["spans"]["pipeline.decode"]["sum"] == pytest.approx(0.12)
    assert rep["spans"]["pipeline.run"]["n"] == 1
    assert rep["counters"]["ocr.chunks"] == 3
    assert rep["gauges"]["ocr.engine_init"] == pytest.approx(0.0002)
    assert rep["pipeline"]["n_segments"] == 7
    assert rep["pipeline"]["config_digest"] == "deadbeef"
    # health 每项都是 {metric, value, limit, ok} 形态（§13.2 N-5 可判定）
    assert "PI-10" in rep["health"]
    for k, v in rep["health"].items():
        assert {"metric", "value", "limit", "ok"} <= set(v), k
    assert rep["environment"]["python"].startswith("3.")
    assert set(rep["resources"]) == RESOURCE_KEYS


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
    fresh = Metrics("std")
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
    assert row["cores_avg"] >= 0.0               # Δcpu/Δwall：本机应 ≈1
    assert row["threads"] >= 1
    if "rss_delta_mib" in row:
        assert isinstance(row["rss_delta_mib"], float)


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
    m = Metrics("std")
    m.start_hardware()                            # std 档不得建线程
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
