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
    assert m.snapshot()["spans"]["pipeline.decode"][0] >= 0.0


def test_registry_cap():
    from video_ocr_engine.domain.metrics import MetricRegistry
    specs = tuple(MetricSpec("x.%d" % i, "counter", "", "x", "")
                  for i in range(65))
    with pytest.raises(AssertionError):
        MetricRegistry(specs)
