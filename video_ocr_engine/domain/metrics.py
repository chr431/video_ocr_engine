"""指标注册表与记录器骨架（v2 §8.6 N-1/N-2，S1 落地、S3 接线）。

规则（§8.6）：未注册的名字不得上报；注册表上限 64；粗档 span（相位级）
随 std 档恒开；counter 恒开；telemetry=off 时一切空调用（NullMetrics，
PI-15 由 bench 三档互比门禁背书）。线程口径：每线程局部累积、drain 时合并
（消灭 v1 的无锁共享写，P1-9 同类）。

S1 只交付骨架与种子指标（v1 现役键全部迁入，不丢观测）；编排接线在 S3。
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

MetricKind = Literal["span", "counter", "gauge", "histogram"]
METRIC_CAP = 64

# v1 现役观测键（meta.timing / ex.profile / 剖面键）——迁入注册表，命名空间
# = 阶段（§8.6 N-1）；pi_binding 指向 §13.2 不变式（能绑的先绑）。
_SEED = (
    ("pipeline.decode", "span", "s", "pipeline", ""),
    ("pipeline.ocr", "span", "s", "pipeline", ""),
    ("pipeline.ocr_tail", "span", "s", "pipeline", ""),
    ("pipeline.producer", "span", "s", "pipeline", ""),
    ("pipeline.q_get_wait", "gauge", "s", "pipeline", "PI-5"),
    ("pipeline.q_put_block", "gauge", "s", "pipeline", "PI-5"),
    ("ocr.engine_init", "gauge", "s", "ocr", "PI-10"),
    ("ocr.syncs_per_chunk", "counter", "次/chunk", "ocr", "PI-3"),
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: MetricKind
    unit: str
    stage: str
    pi_binding: str


class MetricRegistry:
    def __init__(self, specs: tuple[MetricSpec, ...]) -> None:
        assert len(specs) <= METRIC_CAP, "指标注册表超过 %d 上限" % METRIC_CAP
        names = [s.name for s in specs]
        assert len(set(names)) == len(names), "指标名重复"
        self._by_name = {s.name: s for s in specs}

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def spec(self, name: str) -> MetricSpec:
        if name not in self._by_name:
            raise KeyError("未注册指标: %s（§8.6：未注册名不得上报）" % name)
        return self._by_name[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)


METRICS = MetricRegistry(tuple(MetricSpec(*row) for row in _SEED))


class Metrics:
    """线程安全（局部累积 + 合并）的指标记录器。

    tier：off/std/full（§8.6 r5 分层）。骨架阶段 span 仅计时入快照，
    L1 资源差分与 CUDA event 在 S3 接线时并入。
    """

    def __init__(self, tier: str = "std", registry: MetricRegistry = METRICS,
                 clock=time.perf_counter) -> None:
        assert tier in ("off", "std", "full")
        self._tier = tier
        self._registry = registry
        self._clock = clock
        self._lock = threading.Lock()
        self._master: dict = {}
        self._local = threading.local()

    @property
    def enabled(self) -> bool:
        return self._tier != "off"

    def _bucket(self) -> dict:
        b = getattr(self._local, "bucket", None)
        if b is None:
            b = self._local.bucket = {"spans": {}, "counters": {}, "gauges": {}}
        return b

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        self._registry.spec(name)          # 未注册即 KeyError——硬失败
        if not self.enabled:
            yield
            return
        t0 = self._clock()
        try:
            yield
        finally:
            d = self._clock() - t0
            with self._lock:
                self._bucket()["spans"].setdefault(name, []).append(d)
                # 局部桶在快照时合并（drain 语义），此处仅登记线程归属
                self._merge_locked()

    def counter(self, name: str, n: int = 1) -> None:
        if not self.enabled:
            return
        self._registry.spec(name)
        b = self._bucket()
        b["counters"][name] = b["counters"].get(name, 0) + n
        with self._lock:
            self._merge_locked()

    def gauge(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        self._registry.spec(name)
        self._bucket()["gauges"][name] = value
        with self._lock:
            self._merge_locked()

    def _merge_locked(self) -> None:
        b = self._bucket()
        for k, v in b["spans"].items():
            self._master.setdefault("spans", {}).setdefault(k, []).extend(v)
        for k, v in b["counters"].items():
            self._master.setdefault("counters", {})[k] = \
                self._master.setdefault("counters", {}).get(k, 0) + v
        for k, v in b["gauges"].items():
            self._master.setdefault("gauges", {})[k] = v
        b["spans"].clear(); b["counters"].clear(); b["gauges"].clear()

    def snapshot(self) -> dict:
        with self._lock:
            self._merge_locked()
            return {k: dict(v) for k, v in self._master.items()}


class NullMetrics(Metrics):
    """telemetry=off（§8.6 r5）：全部空调用、不组装快照。"""

    def __init__(self) -> None:
        super().__init__(tier="off")

    def snapshot(self) -> dict:
        return {}
