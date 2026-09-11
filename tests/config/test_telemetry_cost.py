"""PI-15 的**确定性**成本守卫（µs 级分辨力）。

为什么要有这个文件：墙钟口径的三档互比（`bench.py telemetry-check`）在本机
A/A 实测只能分辨到 ~0.85%（同一段代码两槽差可达 4%），而 std 档插桩的**真实
成本**是 0.1% 量级——用分辨力不够的量去当硬门禁，只会产生误报（r11 用户裁决
放松了那条门禁）。所以严格性搬到这里：**直接量插桩路径本身的执行时间**，
分辨力比墙钟高约 10⁴ 倍，且不受 GPU 热降/后台负载影响。

口径与真实 run 对齐（实测 3000 帧 std run）：9 个 span 键 / **138 个 span
样本** / 10 个 counter 键 / 8 个 gauge / **4 个 L1 资源边界** + 一次
`snapshot()` + 一次 `build_report()`。断言的是"这样一整套跑完"的成本上限。
"""
from __future__ import annotations

import statistics
import time

import pytest

from video_ocr_engine.domain.metrics import METRICS, Metrics
from video_ocr_engine.pipeline.report import build_report

# 真实 run 的记录量（见模块 docstring 的实测出处）
SPAN_SAMPLES = 138
COUNTER_KEYS = 10
GAUGE_KEYS = 8
CHECKPOINTS = 4
REPLAYS = 300

_SPANS = [n for n in METRICS.names()
          if METRICS.spec(n).kind == "span"][:9]
_COUNTERS = [n for n in METRICS.names()
             if METRICS.spec(n).kind == "counter"][:COUNTER_KEYS]
_GAUGES = [n for n in METRICS.names()
           if METRICS.spec(n).kind == "gauge"][:GAUGE_KEYS]


def _one_run(m: Metrics) -> None:
    """重放一个 3000 帧 std run 的全部遥测动作（含收尾快照）。"""
    for i in range(SPAN_SAMPLES):
        m.record_span(_SPANS[i % len(_SPANS)], 0.001)
    for name in _COUNTERS:
        m.counter(name, 3)
    for i, name in enumerate(_GAUGES):
        m.gauge(name, float(i))
    for ph in ("open", "calibrate", "decode", "ocr"):
        m.checkpoint(ph)
    build_report(m, wall=1.0, config_digest="x" * 8, n_segments=1083,
                 backend="decord/CPU", ocr_backend="tensorrt")


def _median_cost(fn, repeats: int) -> float:
    _warm_env_fingerprint()
    fn(Metrics("std"))                       # 预热（首次含集合/字典建立）
    out = []
    for _ in range(repeats):
        m = Metrics("std")
        t = time.perf_counter()
        fn(m)
        out.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(out)


def _warm_env_fingerprint() -> None:
    """把环境指纹焐热：它首调要 20–43ms（NVML 初始化 / nvidia-smi 子进程）。

    这笔钱只付**一次**、且在报告组装路径上，与"每 run 插桩成本"是两件事，
    分开量（见 `test_environment_fingerprint_is_cached`）。
    """
    build_report(Metrics("std"), wall=0.0)


def test_environment_fingerprint_is_cached():
    """首 run 付一次指纹成本，之后必须接近零（否则每 run +20–43ms）。"""
    from video_ocr_engine.pipeline import report as R
    R._ENV_CACHE = None                       # 模拟新进程
    t = time.perf_counter()
    R.environment()
    first = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(200):
        R.environment()
    per = (time.perf_counter() - t) / 200
    assert per * 1e6 < 50.0, "指纹缓存失效：每次 %.1fµs" % (per * 1e6)
    assert first < 0.30, "指纹首调 %.3fs（应 <300ms：NVML 或 nvidia-smi）" % first


def test_std_recording_path_cost_per_run():
    """整套 std 遥测（138 span + 计数器 + 4 资源边界 + 报告）≤ 2ms/run。

    3000 帧 run 墙钟 ~1.0s → 2ms = 0.2%，比 §13.2 的 +0.1% 设计目标宽 2 倍、
    比墙钟门禁的 0.85% 严 400 倍：真正的插桩膨胀（例如误在 per-frame 路径
    加记录）会在这里亮红灯，而机器噪声不会。
    """
    ms = _median_cost(_one_run, REPLAYS)
    assert ms <= 2.0, "std 档遥测路径成本 %.3fms/run，超预算 2ms" % ms


def test_off_tier_records_nothing():
    """off 档一行关闭：NULL_METRICS 全部 no-op，且拿不到探针。"""
    from video_ocr_engine.domain.metrics import NULL_METRICS
    assert NULL_METRICS.tier == "off" and not NULL_METRICS.enabled
    for name in _SPANS:
        NULL_METRICS.record_span(name, 0.1)
        NULL_METRICS.counter(name, 5)
        NULL_METRICS.gauge(name, 1.0)
        NULL_METRICS.checkpoint("decode")
    assert NULL_METRICS.snapshot() == {}
    assert NULL_METRICS.resource_report() is None
    assert build_report(NULL_METRICS, wall=1.0) == {}


def test_l1_checkpoint_is_microsecond_scale():
    """资源边界单点必须停留在 µs 级（实测 ~10µs）。

    这条是 `domain/resources.py` 里"psutil threads() 43ms 地雷"的量化守卫：
    阈值放到 200µs，仍比 0.1% 预算（1.0s run 的 1ms）宽 5 倍——也就是说，
    真把昂贵调用塞进边界会立刻被这里抓到。
    """
    m = Metrics("std")
    for i in range(32):                     # 先把探针建出来（惰性）
        m.checkpoint("warm%d" % i)
    probes = m._resources
    t = time.perf_counter()
    n = 200
    for i in range(n):
        probes._counters.sample()
    us = (time.perf_counter() - t) / n * 1e6
    assert us < 200.0, "单次资源采样 %.1fµs，远超 µs 级假设" % us


@pytest.mark.parametrize("tier", ["std", "full"])
def test_snapshot_and_report_scale_with_records(tier):
    """收尾快照/报告组装随**记录键数**而非样本数增长（drain 一次合并）。"""
    _warm_env_fingerprint()
    m = Metrics(tier)
    for i in range(SPAN_SAMPLES * 4):       # 4× 真实样本量
        m.record_span(_SPANS[i % len(_SPANS)], 0.001)
    t = time.perf_counter()
    rep = build_report(m, wall=1.0)
    cost_ms = (time.perf_counter() - t) * 1000
    assert rep["report_version"] >= 2
    assert cost_ms < 8.0, "%s 档收尾组装 %.2fms（细档样本翻 4 倍仍须 <8ms）" % (
        tier, cost_ms)
