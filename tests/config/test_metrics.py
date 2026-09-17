"""S1 骨架测试：指标注册表与 Metrics/NullMetrics（v2 §8.6）。

metrics_coverage 的核心断言：未注册名上报 → KeyError（硬失败）；
NullMetrics 全空调用；v1 现役键全部在注册表（观测不丢）。
"""
from __future__ import annotations

import threading

import pytest

from video_ocr_engine.domain.metrics import (
    METRICS, Metrics, MetricSpec, NullMetrics)


def test_registry_covers_v1_keys():
    # pipeline.producer 已出清（2026-09-17：全仓零产出点），
    # v1 观测键里补一个仍然现役的无界键 consume_feed。
    for name in ("pipeline.decode", "pipeline.ocr", "pipeline.ocr_tail",
                 "pipeline.consumer", "pipeline.consume_feed",
                 "pipeline.q_get_wait", "pipeline.q_put_block"):
        assert name in METRICS


def test_unregistered_name_fails_hard():
    m = Metrics()
    with pytest.raises(KeyError):
        with m.span("nope.span"):
            pass
    with pytest.raises(KeyError):
        m.counter("nope.counter")


def test_null_metrics_all_noop():
    m = NullMetrics()
    assert not m.enabled
    with m.span("pipeline.decode"):        # 未启用：不触注册表查询以外的任何工作
        pass
    m.counter("ocr.syncs_per_chunk")
    assert m.snapshot() == {}


def test_counter_accumulates_and_merges():
    m = Metrics()

    def bump(n):
        for _ in range(n):
            m.counter("ocr.syncs_per_chunk")

    ts = [threading.Thread(target=bump, args=(50,)) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert m.snapshot()["counters"]["ocr.syncs_per_chunk"] == 200


def test_span_records_duration():
    m = Metrics()
    with m.span("pipeline.decode"):
        pass
    agg = m.snapshot()["spans"]["pipeline.decode"]
    assert agg["n"] == 1 and agg["sum"] >= 0.0 and agg["max"] >= 0.0


def test_snapshot_aggregates_spans():
    """S6-0：drain 语义——多线程样本在 snapshot 合并为 n/sum/p50/p99。"""
    m = Metrics()
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
    assert m.snapshot()["spans"] == {}          # drain 后再取为空


def test_off_tier_is_silent_singleton():
    """PI-15：off 档不触注册表、不计时、不组装快照。"""
    from video_ocr_engine.domain.metrics import NULL_METRICS, make_metrics
    assert make_metrics("off") is NULL_METRICS
    assert NULL_METRICS.tier == "off" and not NULL_METRICS.detailed
    NULL_METRICS.counter("完全未注册的名字")     # off 档不校验（无上报路径）
    assert NULL_METRICS.snapshot() == {}


def test_detailed_only_in_full_tier():
    assert Metrics("full").detailed and not Metrics("std").detailed


def test_profile_mapping_targets_are_registered():
    """§8.6 N-2：单一计时脊柱的映射表只能指向注册名（防映射漂移）。"""
    from video_ocr_engine.domain.metrics import (
        PROFILE_GAUGES, PROFILE_SPANS, PROFILE_TOTALS,
        PROFILE_TOTALS_MAX, PROFILE_TOTALS_N)
    for name in (list(PROFILE_SPANS.values()) + list(PROFILE_TOTALS.values())
                 + list(PROFILE_GAUGES.values())
                 + list(PROFILE_TOTALS_N.values())
                 + list(PROFILE_TOTALS_MAX.values())):
        assert name in METRICS, name


def test_totals_n_max_cover_all_totals_keys():
    """P2b：TOTALS 全键分诊——N/MAX/HIST 映射与 TOTALS 键集合一致。"""
    from video_ocr_engine.domain.metrics import (
        HIST_N_BUCKETS, PROFILE_TOTALS, PROFILE_TOTALS_HIST,
        PROFILE_TOTALS_MAX, PROFILE_TOTALS_N, hist_bucket)
    assert set(PROFILE_TOTALS_N) == set(PROFILE_TOTALS)
    assert set(PROFILE_TOTALS_MAX) == set(PROFILE_TOTALS)
    # W1：直方图映射同样全键覆盖
    assert set(PROFILE_TOTALS_HIST) == set(PROFILE_TOTALS)
    # 桶数学：×2 分箱、边界与溢出（桶号 = frexp 指数 + HIST_EXP_BIAS）
    assert hist_bucket(0.0) == 0 and hist_bucket(-1.0) == 0
    assert hist_bucket(2.0 ** -21) == 0            # 下溢进 0 桶
    assert hist_bucket(2.0 ** -20) == 1            # frexp 指数 -19
    assert hist_bucket(1.0) == 21                  # frexp(1.0)=0.5×2^1
    assert hist_bucket(0.5) == 20                  # 1.0 与 0.5 分属两桶
    assert hist_bucket(0.25) == 19
    assert hist_bucket(1e6) == HIST_N_BUCKETS - 1  # 上溢进末桶
    assert 0 <= hist_bucket(1e-9) < HIST_N_BUCKETS
    # ×2 分箱：同一 [2^k, 2^(k+1)) 区间内的值同桶
    assert hist_bucket(0.001) == hist_bucket(0.0019)    # 都在 [2^-10,2^-9)
    assert hist_bucket(0.001) != hist_bucket(0.0021)    # 跨到 [2^-9,2^-8)


def test_registry_cap():
    from video_ocr_engine.domain.metrics import METRIC_CAP, MetricRegistry
    specs = tuple(MetricSpec("x.%d" % i, "counter", "", "x", "")
                  for i in range(METRIC_CAP + 1))
    with pytest.raises(AssertionError):
        MetricRegistry(specs)
    # 现役注册表不得逼近上限（留 20% 余量给后续扩展）
    from video_ocr_engine.domain.metrics import METRICS
    assert len(METRICS.names()) <= METRIC_CAP * 0.8
