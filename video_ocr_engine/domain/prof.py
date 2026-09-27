"""ProfSpine —— 分相计时脊柱（v2 §8.6 N-2 的实现体）。

2026-09-27 R1 自 extractor._prof_end/_metric_from_profile 下放：同一 t0
同时喂 profile（v1 的 13 相位原始字典，diagnostics.profile）与指标注册表
（std/full 档，指标名映射表在 domain/metrics.py 单一出处）；P4 trace 的
t1 复用已算好的 t0+elapsed，不另取时钟。两档全关时 end() 只做两次属性
判断、连 perf_counter 都不调用——PI-15 的 off 档"一行关闭"靠这里兑现。

每次 run 新建一个实例（B1 同类重置：profile/totals/maxes 均为 run 内
状态）；diag 的进展心跳（挂死时可归因到相位）挂在 armed 判定下。
"""
from __future__ import annotations

import threading
import time
from typing import Any

from .metrics import (
    PROFILE_GAUGES, PROFILE_SPANS, PROFILE_TOTALS,
    PROFILE_TOTALS_HIST, PROFILE_TOTALS_MAX, PROFILE_TOTALS_N,
)


class ProfSpine:
    """单次 run 的计时脊柱：profile 字典 + 注册指标 + trace/diag 心跳。"""

    def __init__(self, profile_enabled: bool, metrics: Any,
                 trace: Any = None, diag: Any = None) -> None:
        self._profile_enabled = profile_enabled
        self._metrics = metrics
        self._trace = trace
        self._diag = diag
        self.profile: dict = {}
        self._lock = threading.Lock() if profile_enabled else None
        # 单次 run 的累计/单次最长（report 收敛时冲刷进指标，见
        # pipeline/report.finalized_report）
        self.totals: dict = {}
        self.maxes: dict = {}

    def end(self, group: str, key: str, t0: float) -> None:
        """一个计时相位的收口：同一 t0 喂 profile 与指标。"""
        prof = self._profile_enabled
        met = self._metrics.enabled
        if not (prof or met):
            return
        elapsed = time.perf_counter() - t0
        if prof:
            assert self._lock is not None   # _profile_enabled 与锁同构造
            with self._lock:
                d = self.profile.setdefault(group, {})
                d[key] = d.get(key, 0.0) + elapsed
        if met:
            name = self._metric_from_profile(group, key, elapsed)
            # P4 trace（默认关）：关闭时这条 = 一次 is-not-None 判断
            # （成本守卫背书零成本）。
            if self._trace is not None and name is not None:
                self._trace.record(name, t0, t0 + elapsed)
            if self._diag is not None and self._diag.armed:
                # 进展心跳：挂死时可归因到相位
                self._diag.tick("%s.%s" % (group, key), "%.4fs" % elapsed)

    def _metric_from_profile(self, group: str, key: str,
                             elapsed: float) -> str | None:
        """(group, key) → 注册指标名（映射表在 domain/metrics.py，单一出处）。

        返回主指标名（P4 trace 用）；未映射键返回 None。
        """
        m = self._metrics
        name = PROFILE_GAUGES.get((group, key))
        if name is not None:
            m.gauge(name, elapsed)          # 单值量：末次即本 run 的 engine_init
            return name
        name = PROFILE_SPANS.get((group, key))
        if name is not None:
            m.record_span(name, elapsed)    # 有界相位：std 档逐次采样
            return name
        name = PROFILE_TOTALS.get((group, key))
        if name is not None:
            t = self.totals
            t[name] = t.get(name, 0.0) + elapsed
            # std 档也留「次数 + 单次最长」：背压分诊的最小充分集
            _n = PROFILE_TOTALS_N.get((group, key))
            if _n is not None:
                m.counter(_n)
                _mx = PROFILE_TOTALS_MAX[(group, key)]
                if elapsed > self.maxes.get(_mx, 0.0):
                    self.maxes[_mx] = elapsed
            if m.detailed:
                m.record_span(name, elapsed)   # full 档另留逐次样本
                # W1：分布形状（full 档专属；每事件 0.165µs，std 预算付不起）
                _h = PROFILE_TOTALS_HIST.get((group, key))
                if _h is not None:
                    m.histogram(_h, elapsed)
            return name
        return None
