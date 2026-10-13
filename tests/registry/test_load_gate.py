"""负载门禁（R-perf-quiet）正负测——判据真值表 + 真负载拦截。"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

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


def _burn(_: object) -> None:
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
            time.sleep(0.5)
        assert raised is not None, "4 核烧核在场而门禁始终判静（负测失效）"
        assert "元凶" in str(raised) or "CPU" in str(raised)
    finally:
        for q in procs:
            q.terminate()
            q.join(timeout=5)


def test_gate_passes_on_quiet_machine() -> None:
    """宽松正测（阈值 90%）：happy path 采样与判据管线本身可用。"""
    assert_machine_quiet(cpu_pct=90.0, samples=1, interval=0.2, gpu=False)
