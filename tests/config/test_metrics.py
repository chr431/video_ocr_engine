"""指标注册表与 Metrics/NullMetrics 测试。

metrics_coverage 的核心断言：未注册名上报 → KeyError（硬失败）；
NullMetrics 全空调用；v1 现役键全部在注册表（观测不丢）。
"""

from __future__ import annotations

import threading

import pytest

from video_ocr_engine.domain.metrics import METRICS, Metrics, MetricSpec, NullMetrics


def test_registry_covers_v1_keys():
    # pipeline.producer 已出清（全仓零产出点），
    # v1 观测键里补一个仍然现役的无界键 consume_feed。
    for name in (
        "pipeline.decode",
        "pipeline.ocr",
        "pipeline.ocr_tail",
        "pipeline.consumer",
        "pipeline.consume_feed",
        "pipeline.q_get_wait",
        "pipeline.q_put_block",
    ):
        assert name in METRICS


def test_unregistered_name_fails_hard():
    m = Metrics("full")
    with pytest.raises(KeyError):
        with m.span("nope.span"):
            pass
    with pytest.raises(KeyError):
        m.counter("nope.counter")


def test_null_metrics_all_noop():
    m = NullMetrics()
    assert not m.enabled
    with m.span("pipeline.decode"):  # 未启用：不触注册表查询以外的任何工作
        pass
    m.counter("ocr.syncs_per_chunk")
    assert m.snapshot() == {}


def test_counter_accumulates_and_merges():
    m = Metrics("full")

    def bump(n):
        for _ in range(n):
            m.counter("ocr.syncs_per_chunk")

    ts = [threading.Thread(target=bump, args=(50,)) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert m.snapshot()["counters"]["ocr.syncs_per_chunk"] == 200


def test_span_records_duration():
    m = Metrics("full")
    with m.span("pipeline.decode"):
        pass
    agg = m.snapshot()["spans"]["pipeline.decode"]
    assert agg["n"] == 1 and agg["sum"] >= 0.0 and agg["max"] >= 0.0


def test_snapshot_aggregates_spans():
    """drain 语义——多线程样本在 snapshot 合并为 n/sum/p50/p99。"""
    m = Metrics("full")
    m.record_span("decode.batch", 0.1)
    m.record_span("decode.batch", 0.3)

    def worker():
        m.record_span("decode.batch", 0.2)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    agg = m.snapshot()["spans"]["decode.batch"]
    assert agg["n"] == 3
    assert abs(agg["sum"] - 0.6) < 1e-9
    assert agg["min"] == 0.1 and agg["max"] == 0.3
    assert m.snapshot()["spans"] == {}  # drain 后再取为空


def test_off_tier_is_silent_singleton():
    """off 档不触注册表、不计时、不组装快照。"""
    from video_ocr_engine.domain.metrics import NULL_METRICS, make_metrics

    assert make_metrics("off") is NULL_METRICS
    assert NULL_METRICS.tier == "off" and not NULL_METRICS.detailed
    NULL_METRICS.counter("完全未注册的名字")  # off 档不校验（无上报路径）
    assert NULL_METRICS.snapshot() == {}


def test_detailed_only_in_full_tier():
    assert Metrics("full").detailed and not Metrics("off").detailed


def test_profile_mapping_targets_are_registered():
    """单一计时脊柱的映射表只能指向注册名（防映射漂移）。"""
    from video_ocr_engine.domain.metrics import (
        PROFILE_GAUGES,
        PROFILE_SPANS,
        PROFILE_TOTALS,
        PROFILE_TOTALS_MAX,
        PROFILE_TOTALS_N,
    )

    for name in (
        list(PROFILE_SPANS.values())
        + list(PROFILE_TOTALS.values())
        + list(PROFILE_GAUGES.values())
        + list(PROFILE_TOTALS_N.values())
        + list(PROFILE_TOTALS_MAX.values())
    ):
        assert name in METRICS, name


def test_totals_n_max_cover_all_totals_keys():
    """TOTALS 全键分诊——N/MAX/HIST 映射与 TOTALS 键集合一致。"""
    from video_ocr_engine.domain.metrics import (
        HIST_N_BUCKETS,
        PROFILE_TOTALS,
        PROFILE_TOTALS_HIST,
        PROFILE_TOTALS_MAX,
        PROFILE_TOTALS_N,
        hist_bucket,
    )

    assert set(PROFILE_TOTALS_N) == set(PROFILE_TOTALS)
    assert set(PROFILE_TOTALS_MAX) == set(PROFILE_TOTALS)
    # 直方图映射同样全键覆盖
    assert set(PROFILE_TOTALS_HIST) == set(PROFILE_TOTALS)
    # 桶数学：×2 分箱、边界与溢出（桶号 = frexp 指数 + HIST_EXP_BIAS）
    assert hist_bucket(0.0) == 0 and hist_bucket(-1.0) == 0
    assert hist_bucket(2.0**-21) == 0  # 下溢进 0 桶
    assert hist_bucket(2.0**-20) == 1  # frexp 指数 -19
    assert hist_bucket(1.0) == 21  # frexp(1.0)=0.5×2^1
    assert hist_bucket(0.5) == 20  # 1.0 与 0.5 分属两桶
    assert hist_bucket(0.25) == 19
    assert hist_bucket(1e6) == HIST_N_BUCKETS - 1  # 上溢进末桶
    assert 0 <= hist_bucket(1e-9) < HIST_N_BUCKETS
    # ×2 分箱：同一 [2^k, 2^(k+1)) 区间内的值同桶
    assert hist_bucket(0.001) == hist_bucket(0.0019)  # 都在 [2^-10,2^-9)
    assert hist_bucket(0.001) != hist_bucket(0.0021)  # 跨到 [2^-9,2^-8)


def test_hist_bucket_lower_bounds():
    """桶界表须与 hist_bucket 自洽（旧表曾整表偏 2×）。

    旧式 `2^(i-BIAS)` 把桶的**上界**当下界发布（report.bucket_lower_s
    据此判读会把每个时长放错一倍）。这里逐桶断言 x=下界 落在该桶。
    """
    from video_ocr_engine.domain.metrics import HIST_BUCKET_LOWER, HIST_N_BUCKETS, hist_bucket

    for k in range(1, HIST_N_BUCKETS - 1):
        x = HIST_BUCKET_LOWER[k]
        assert hist_bucket(x) == k, (k, x, hist_bucket(x))
        assert HIST_BUCKET_LOWER[k] <= x < HIST_BUCKET_LOWER[k + 1]
    assert HIST_BUCKET_LOWER[20] == 0.5
    assert HIST_BUCKET_LOWER[21] == 1.0


def test_gauge_max_merges_by_max_across_threads():
    """*_max 型 gauge 跨线程合并取最大（末位写者胜会让门禁静默放行）。

    桶序 = 线程首触顺序，故先注册的线程写大值：若被后注册线程的
    小值覆盖（0.9→0.1），报告的单次最长偏低 = 门禁静默放行。
    """
    m = Metrics("full")
    name = "pipeline.q_put_block_max"

    def worker(v):
        m.gauge(name, v)

    ta = threading.Thread(target=worker, args=(0.9,))  # 先触桶：写大值
    tb = threading.Thread(target=worker, args=(0.1,))
    ta.start()
    ta.join()  # noqa: E702 — 启动即汇合的紧凑写法
    tb.start()
    tb.join()  # noqa: E702
    assert m.snapshot()["gauges"][name] == pytest.approx(0.9)


def test_registry_cap():
    from video_ocr_engine.domain.metrics import METRIC_CAP, MetricRegistry

    specs = tuple(MetricSpec("x.%d" % i, "counter", "", "x", "") for i in range(METRIC_CAP + 1))
    with pytest.raises(AssertionError):
        MetricRegistry(specs)
    # 现役注册表不得逼近上限（留 20% 余量给后续扩展）
    from video_ocr_engine.domain.metrics import METRICS

    assert len(METRICS.names()) <= METRIC_CAP * 0.8
