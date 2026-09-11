"""指标注册表与记录器（v2 §8.6 N-1/N-2，S1 骨架 → S6-0 接线）。

规则（§8.6）：未注册的名字不得上报；注册表上限 64；粗档 span（相位级）
随 std 档恒开；counter 恒开；telemetry=off 时一切空调用（NullMetrics，
PI-15 由 bench 三档互比门禁背书）。线程口径：**每线程局部累积、drain 时
合并**（消灭 v1 的无锁共享写，P1-9 同类）。

性能口径（S6-0 实测修正）：v1 骨架版在**每次 span 结束都取锁并合并**——
插桩点一多就变成自伤（PI-15 的 std vs off ≤ +0.1% 会当场失败）。现改为
本地桶 append + `snapshot()` 一次性合并；线程桶在**首次使用时**登记，
snapshot 时遍历登记表（每线程一次加锁，之后全无锁）。
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

from .resources import NvmlSampler, ResourceProbe

MetricKind = Literal["span", "counter", "gauge", "histogram"]
METRIC_CAP = 64

# 注册表（§8.6 N-1）：命名空间 = 阶段；pi_binding 指向 §13.2 不变式。
# 元组 = (name, kind, unit, stage, pi_binding)。
_SEED = (
    # ── pipeline：编排三段 + v1 现役键（迁移不丢观测）──
    ("pipeline.run", "span", "s", "pipeline", ""),
    ("pipeline.setup", "span", "s", "pipeline", ""),
    ("pipeline.calibrate", "span", "s", "pipeline", ""),
    ("pipeline.decode", "span", "s", "pipeline", ""),
    ("pipeline.producer", "span", "s", "pipeline", ""),
    ("pipeline.consumer", "span", "s", "pipeline", ""),
    ("pipeline.ocr", "span", "s", "pipeline", ""),
    ("pipeline.ocr_tail", "span", "s", "pipeline", ""),
    ("pipeline.q_get_wait", "gauge", "s", "pipeline", "PI-5"),
    ("pipeline.q_put_block", "gauge", "s", "pipeline", "PI-5"),
    ("pipeline.consume_feed", "span", "s", "pipeline", ""),
    # ── decode：批级（std 档不含 per-frame）──
    ("decode.batch", "span", "s", "decode", ""),
    ("decode.batches", "counter", "批", "decode", ""),
    ("decode.frames", "counter", "帧", "decode", ""),
    ("decode.luma_batch", "span", "s", "decode", ""),
    ("decode.sharp_batch", "span", "s", "decode", ""),
    ("decode.binarize_batch", "span", "s", "decode", ""),
    ("decode.stream_analyze", "span", "s", "decode", "PI-13"),
    # ── segment ──
    ("segment.emit", "span", "s", "segment", ""),
    ("segment.emit_batch", "span", "s", "segment", ""),
    ("segment.segments", "counter", "段", "segment", ""),
    ("segment.merge_pair", "span", "s", "segment", "PI-1"),
    ("segment.merges", "counter", "次", "segment", ""),
    # ── ocr ──
    ("ocr.engine_init", "gauge", "s", "ocr", "PI-10"),
    ("ocr.engine_reuse", "counter", "次", "ocr", "PI-10"),
    ("ocr.infer", "span", "s", "ocr", ""),
    ("ocr.preprocess", "span", "s", "ocr", ""),
    ("ocr.ctc_decode", "span", "s", "ocr", ""),
    ("ocr.chunks", "counter", "chunk", "ocr", ""),
    ("ocr.sub_chunks", "counter", "子批", "ocr", "PI-3"),
    ("ocr.syncs", "counter", "次", "ocr", "PI-3"),
    ("ocr.syncs_per_chunk", "gauge", "次/chunk", "ocr", "PI-3"),
    # 填充率（R5/S6-f）：pad 宽是 rec 模型的计算宽度，content 是真实内容宽度
    ("ocr.rows", "counter", "行", "ocr", ""),
    ("ocr.padded_cols", "counter", "列", "ocr", ""),
    ("ocr.content_cols", "counter", "列", "ocr", ""),
    ("ocr.fill_pct", "gauge", "%", "ocr", ""),
    # ── emit：消费端提交（第一阶段性能目标，§7.6）──
    ("emit.d2h_bytes", "counter", "B", "emit", "PI-14"),
    ("emit.h2d_bytes", "counter", "B", "emit", "PI-14"),
    ("emit.d2h_calls", "counter", "次", "emit", "PI-9"),
    ("emit.launches", "counter", "次", "emit", "PI-8"),
    ("emit.syncs", "counter", "次", "emit", "PI-8"),
    ("emit.keep_crops_d2h", "counter", "次", "emit", "PI-9"),
    ("emit.keep_crops_batched", "counter", "次", "emit", "PI-9"),
    ("emit.autocrop", "span", "s", "emit", ""),
    ("emit.put", "span", "s", "emit", ""),
    ("emit.d2h", "span", "s", "emit", ""),
    # ── 池与帧契约 ──
    ("pools.high_water", "gauge", "个", "pools", "PI-5"),
    ("frame_batch.inflight", "gauge", "个", "frame_batch", "PI-6"),
    ("frame_batch.leaks", "counter", "个", "frame_batch", "PI-6"),
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

# ── 单一计时脊柱（§8.6 N-2）──────────────────────────────────────────
# 编排三段与 Protocol 边界的既有计时点（`_prof_end(group, key, t0)`）直接
# 映射为注册指标名——后端不再自建字典，v1 的 13 个细粒度相位零改动迁入。
PROFILE_SPANS = {
    # 有界（每次 run ≤ ~200 次）：std 档即逐次采样
    ("producer", "open_and_fps"): "pipeline.setup",
    ("producer", "calib_total"): "pipeline.calibrate",
    ("producer", "gpu_calib_total"): "pipeline.calibrate",
    ("producer", "consumer_total"): "pipeline.consumer",
    ("producer", "decode_batch"): "decode.batch",
    ("producer", "stream_analyze"): "decode.stream_analyze",
    ("producer", "gray_batch"): "decode.luma_batch",
    ("producer", "sharp_batch"): "decode.sharp_batch",
    ("producer", "bin_batch"): "decode.binarize_batch",
    ("ocr", "infer"): "ocr.infer",
    ("ocr", "preprocess"): "ocr.preprocess",
    ("ocr", "ctc_decode"): "ocr.ctc_decode",
}
#: 无界（每段/每帧调用）→ std 档只累加总量（`totals` 段）；full 档另采样
PROFILE_TOTALS = {
    ("producer", "consume_feed"): "pipeline.consume_feed",
    ("producer", "q_put_block"): "pipeline.q_put_block",
    ("producer", "emit_autocrop"): "emit.autocrop",
    ("producer", "emit_put"): "emit.put",
    ("producer", "emit_d2h"): "emit.d2h",
    ("ocr", "q_get_wait"): "pipeline.q_get_wait",
    ("producer", "merge_pair"): "segment.merge_pair",
}
#: 单值量（只保留末次）：engine_init 是 PI-10 的判据本体
PROFILE_GAUGES = {
    ("ocr", "engine_init"): "ocr.engine_init",
}


def _empty_bucket() -> dict:
    return {"spans": {}, "counters": {}, "gauges": {}}


class Metrics:
    """线程安全的指标记录器（局部累积 + drain 合并）。

    tier：off/std/full（§8.6 r5 分层）。std = 相位级粗档 span + 全部
    counter/gauge；full 另含细档 span（per-batch/per-chunk 由调用方按
    tier 判定是否插桩，见 `detailed`）。
    """

    __slots__ = ("_tier", "_registry", "_clock", "_lock", "_local",
                 "_buckets", "_master", "_checked", "_resources", "_hw")

    def __init__(self, tier: str = "std", registry: MetricRegistry = METRICS,
                 clock=time.perf_counter) -> None:
        assert tier in ("off", "std", "full")
        self._tier = tier
        self._registry = registry
        self._clock = clock
        self._lock = threading.Lock()
        self._local = threading.local()
        self._buckets: list = []
        self._master: dict = _empty_bucket()
        self._checked: set = set()     # 已校验名（首次上报校验，之后集合命中）
        # §8.6 r5 资源层：L1 相位边界差分（std+，纯 stdlib 计数，见
        # domain/resources.py 的成本实测）；L2 设备峰值采样仅 full 档显式启动。
        # None = off 档永不采样；False = std+ 但本 run 还没到边界（惰性建探针：
        # 探针构造含 ctypes 结构定义，宿主/短路径不必付这笔钱）。
        self._resources = None if tier == "off" else False
        self._hw = None

    def _check(self, name: str) -> None:
        """N-1：未注册的名字不得上报（校验结果缓存，热路径只做集合命中）。"""
        if name not in self._checked:
            self._registry.spec(name)
            self._checked.add(name)

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def enabled(self) -> bool:
        return self._tier != "off"

    @property
    def detailed(self) -> bool:
        """细档（per-batch / per-chunk / CUDA event）是否插桩。"""
        return self._tier == "full"

    def _bucket(self) -> dict:
        b = getattr(self._local, "bucket", None)
        if b is None:
            b = self._local.bucket = _empty_bucket()
            with self._lock:          # 每线程一次
                self._buckets.append(b)
        return b

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._check(name)                  # 未注册即 KeyError——硬失败
        t0 = self._clock()
        try:
            yield
        finally:
            self.record_span(name, self._clock() - t0)

    def checkpoint(self, phase: str) -> None:
        """L1 相位边界资源采样（std+；off 一行不执行）。"""
        r = self._resources
        if r is None:
            return
        if r is False:
            r = self._resources = ResourceProbe()
        r.checkpoint(phase)

    def resource_report(self) -> dict | None:
        """`{"sources":…, "per_phase":…}`；off 档 / 未建探针 → None。"""
        r = self._resources
        if not r:
            return None
        return {"sources": r.sources, "per_phase": r.per_phase()}

    def start_hardware(self) -> None:
        """L2 设备峰值采样：**仅 full 档**、仅显式调用时建线程（B6）。"""
        if self._tier != "full" or self._hw is not None:
            return
        sampler = NvmlSampler()
        sampler.start()
        self._hw = sampler

    def hardware_report(self) -> dict | None:
        """停采样并返回摘要（未启动 / 非 full 档 → None）。"""
        s = self._hw
        if s is None:
            return None
        self._hw = None
        return s.stop()

    def record_span(self, name: str, seconds: float) -> None:
        """已计时点的上报口（`_prof_end` 单一时序脊柱走这里，零额外时钟）。"""
        if not self.enabled:
            return                    # off 档：不查注册表、不取桶、不分配（PI-15）
        self._check(name)
        b = self._bucket()
        s = b["spans"]
        lst = s.get(name)
        if lst is None:
            s[name] = [seconds]
        else:
            lst.append(seconds)

    def counter(self, name: str, n: int = 1) -> None:
        if not self.enabled:
            return
        self._check(name)
        b = self._bucket()
        c = b["counters"]
        c[name] = c.get(name, 0) + n

    def gauge(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        self._check(name)
        self._bucket()["gauges"][name] = value

    def snapshot(self) -> dict:
        """合并全部线程桶（drain 语义）并返回聚合快照。

        返回 {"spans": {name: {n,sum,min,max,p50,p99}}, "counters": {...},
        "gauges": {...}}；spans 聚合后丢弃原始样本（报告不需要时间线）。
        """
        with self._lock:
            buckets, self._buckets = self._buckets, []
            self._local = threading.local()          # 丢弃本线程桶引用
        spans: dict = {}
        counters: dict = {}
        gauges: dict = {}
        for b in buckets:
            for k, lst in b["spans"].items():
                agg = spans.get(k)
                if agg is None:
                    spans[k] = list(lst)
                else:
                    agg.extend(lst)
            for k, v in b["counters"].items():
                counters[k] = counters.get(k, 0) + v
            for k, v in b["gauges"].items():
                gauges[k] = v
        out_spans = {}
        for k, lst in spans.items():
            if not lst:
                continue
            xs = sorted(lst)
            n = len(xs)
            out_spans[k] = {
                "n": n,
                "sum": sum(xs),
                "min": xs[0],
                "max": xs[-1],
                "p50": xs[n // 2],
                "p99": xs[min(n - 1, int(n * 0.99))],
            }
        return {"spans": out_spans, "counters": counters, "gauges": gauges}


class NullMetrics(Metrics):
    """telemetry=off（§8.6 r5）：全部空调用、不组装快照。"""

    def __init__(self) -> None:
        super().__init__(tier="off")

    def snapshot(self) -> dict:
        return {}


#: off 档共享单例（无状态，避免每 run 新建对象）
NULL_METRICS = NullMetrics()


def make_metrics(tier: str) -> Metrics:
    """按 telemetry 档构造记录器（off 复用单例）。"""
    if tier == "off":
        return NULL_METRICS
    return Metrics(tier=tier)
