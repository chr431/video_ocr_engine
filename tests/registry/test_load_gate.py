"""负载门禁（R-perf-quiet）正负测——判据真值表 + 真负载拦截。"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import psutil
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from _measure_lib import (  # noqa: E402
    BackgroundLoadError,
    _judge_quiet,
    assert_machine_quiet,
)


def test_judge_table() -> None:
    """判据核心（注入假数据）：CPU 任一轮超阈 / GPU 超阈 / 全静=通过。"""
    assert _judge_quiet([(3.0, []), (4.0, [])], 0.0, 15.0) is None
    assert "CPU" in _judge_quiet([(3.0, []), (16.0, [])], 0.0, 15.0)
    assert "GPU" in _judge_quiet([(3.0, [])], 42.0, 15.0, gpu_pct_limit=10.0)
    assert _judge_quiet([(3.0, [])], 32.0, 15.0) is None  # 桌面合成常态不算负载


def test_gpu_window_mean_separates_spike_from_sustained(monkeypatch) -> None:
    """滑动平均滤波的判据本体（负测夹具注入读数列）：

    单点爆发（1 高 + 4 低）→ 窗均值低 → 过；持续负载（整窗高）→ 红。
    这是"门禁不误报但能抓到真负载"的机器可判形式；实测背景：nvidia-smi
    瞬时读数 8%~73% 震荡，任务管理器口径稳定 <25%（本轮三次单点
    52/57/59% 假红即由此而来）。
    """
    import _measure_lib as m

    # (a) 单点尖峰：均值必须被压到限下（不误报）
    spike = iter([73.0, 8.0, 12.0, 10.0, 9.0])
    monkeypatch.setattr(m, "_gpu_util", lambda: next(spike))
    assert m._gpu_util_window(samples=5, interval=0.0) < 50.0
    # (b) 持续负载：整窗高 → 均值高（抓得住）
    hi = iter([61.0, 62.0, 60.0, 63.0, 61.0])
    monkeypatch.setattr(m, "_gpu_util", lambda: next(hi))
    assert m._gpu_util_window(samples=5, interval=0.0) >= 50.0
    # (c) 全零 = 不可用（≠ 空闲）：返回负值哨兵
    monkeypatch.setattr(m, "_gpu_util", lambda: 0.0)
    assert m._gpu_util_window(samples=3, interval=0.0) < 0.0


def test_gpu_window_detail_shares_samples_with_verdict(monkeypatch) -> None:
    """判据与错误消息必须同源：_gpu_window_detail 返回的明细就是均值的原始列。"""
    import statistics

    import _measure_lib as m

    seq = iter([10.0, 70.0, 20.0, 30.0, 40.0])
    monkeypatch.setattr(m, "_gpu_util", lambda: next(seq))
    mean, vals = m._gpu_window_detail(samples=5, interval=0.0)
    assert vals == [10.0, 70.0, 20.0, 30.0, 40.0]
    assert mean == statistics.fmean(vals)
    assert "70" in m._format_gpu_window(vals)


def _burn() -> None:
    # 必须零参：mp.Process(target=_burn) 不传 args；带必填参数会让子进程
    # 立即 TypeError 退出，负测于是"没有负载也断言有负载"——本地靠桌面噪声
    # 假通过，安静 CI 上必红（实测两次 CI 红即此因）。
    t0 = time.time()
    while time.time() - t0 < 12:
        sum(i * i for i in range(10000))


def test_gate_catches_real_load() -> None:
    """真负载负测：4 个烧核进程在场时，门禁必须抛 BackgroundLoadError。

    R-guard-negative 的实体版——守卫必须能失败，且失败要发生在浪费
    整轮测量之前。阈值收紧到 5% 使 4 核负载（≈4/32=12.5% 整机）必越线。

    不用固定 sleep 等烧核就位：Windows spawn 在机器拥挤时可慢于 1s，
    只起了 1 个核（≈3%）就会低于阈值而假失败（实测全链跑时出现过）。
    改为有界轮询——负载真正建立后断言必须抛错。
    """
    procs = [mp.Process(target=_burn) for _ in range(4)]
    for q in procs:
        q.start()
    try:
        raised: BackgroundLoadError | None = None
        os_saw_load = False
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                assert_machine_quiet(
                    cpu_pct=5.0,
                    samples=2,
                    interval=0.4,
                    gpu=False,
                    ignore_own_tree=False,
                )  # 烧核进程按外来负载计（负测不依赖桌面噪声）
            except BackgroundLoadError as e:
                raised = e
                break
            # 前提自证：若连系统口径都看不到负载（容器 CPU 计数受限），
            # 那是环境不满足负测前提，应显式 SKIP，而不是冒充"门禁失效"。
            if psutil.cpu_percent(interval=0.2) > 20.0:
                os_saw_load = True
            time.sleep(0.3)
        if raised is None and not os_saw_load:
            pytest.skip("系统口径看不到烧核负载（容器 CPU 计数受限）——负测前提不成立")
        assert raised is not None, "4 核烧核在场而门禁始终判静（负测失效）"
        assert "元凶" in str(raised) or "CPU" in str(raised)
    finally:
        for q in procs:
            q.terminate()
            q.join(timeout=5)


def test_gate_passes_on_quiet_machine() -> None:
    """宽松正测（阈值 90%）：happy path 采样与判据管线本身可用。"""
    assert_machine_quiet(cpu_pct=90.0, samples=1, interval=0.2, gpu=False)
