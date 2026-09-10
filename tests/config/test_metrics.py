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
    for name in ("pipeline.decode", "pipeline.ocr", "pipeline.ocr_tail",
                 "pipeline.producer", "pipeline.q_get_wait",
                 "pipeline.q_put_block"):
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
        PROFILE_GAUGES, PROFILE_SPANS, PROFILE_TOTALS)
    for name in (list(PROFILE_SPANS.values()) + list(PROFILE_TOTALS.values())
                 + list(PROFILE_GAUGES.values())):
        assert name in METRICS, name


def test_registry_cap():
    from video_ocr_engine.domain.metrics import MetricRegistry
    specs = tuple(MetricSpec("x.%d" % i, "counter", "", "x", "")
                  for i in range(65))
    with pytest.raises(AssertionError):
        MetricRegistry(specs)
