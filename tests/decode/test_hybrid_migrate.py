"""在线移界（_migrate_boundary）纯函数单测。

场景取自 2026-09-05 实测：一次性并发校准的速率偏差使份额错配成为拖尾
（h264 两次运行尾部都 ~2.8s，均衡应为 ~2.05s）。移界 = 在 fast 未认领
前缀内重算 min-max 分界，只动未 started 的片。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from hybrid_decode import _migrate_boundary  # noqa: E402


def _counts(n, per=300):
    return [per] * n


def test_fast_overloaded_moves_boundary():
    """fast 剩余 4 片 @250fps、slow 2 片 @1000fps → 让到 m=3（max 4.8→1.5）。"""
    counts = _counts(8)
    started = [True, True] + [False] * 6   # fast 已认领 [0,2)，k=6
    m = _migrate_boundary(counts, started, k=6,
                          rem_fast=0.0, rem_slow=0.0,
                          r_fast=250.0, r_slow=1000.0)
    assert m == 3


def test_balanced_no_migration():
    """两端按当前分界恰好均衡 → 不移界。"""
    counts = _counts(8)
    started = [True, True] + [False] * 6
    m = _migrate_boundary(counts, started, k=6,
                          rem_fast=0.0, rem_slow=0.0,
                          r_fast=1000.0, r_slow=500.0)
    assert m is None   # base: t_f=4*300/1000=1.2, t_s=2*300/500=1.2


def test_inflight_remainder_respected():
    """fast 在途片剩余帧计入完成时间（p_f=3，#2 在途剩 150 帧）。"""
    counts = _counts(8)
    started = [True, True, True] + [False] * 5   # p_f=3, k=6
    # base: t_f=(150+900)/250=4.2 vs t_s=0.6；m=4 → t_f=450/250=1.8,
    # t_s=1200/1000=1.2 → max 1.8（最优）。
    m = _migrate_boundary(counts, started, k=6,
                          rem_fast=150.0, rem_slow=0.0,
                          r_fast=250.0, r_slow=1000.0)
    assert m == 4


def test_never_below_fast_frontier():
    """m 必须 > p_f（fast 至少保留 1 片未认领，防窃取抖回）。"""
    counts = _counts(8)
    started = [True] * 5 + [False] * 3   # p_f=5, k=6：fast 仅剩 1 片
    m = _migrate_boundary(counts, started, k=6,
                          rem_fast=0.0, rem_slow=0.0,
                          r_fast=1.0, r_slow=10000.0)
    assert m is None


def test_slow_holding_unclaimed_tail():
    """slow 区已有认领（[4,6) 已 started）：slow_own 只计未认领尾 [6,10)。"""
    counts = _counts(10)
    # fast 认领 [0,2)，未认领 [2,4)；slow 认领 [4,6)，未认领 [6,10)。k=4。
    started = [True, True, False, False, True, True, False, False, False, False]
    m = _migrate_boundary(counts, started, k=4,
                          rem_fast=0.0, rem_slow=0.0,
                          r_fast=250.0, r_slow=1000.0)
    # base: t_f=600/250=2.4, t_s=1200/1000=1.2 → max 2.4
    # m=3: t_f=300/250=1.2, t_s=(300+1200)/1000=1.5 → max 1.5 ✓
    assert m == 3


def test_degenerate_inputs():
    counts = _counts(8)
    started = [False] * 8
    assert _migrate_boundary(counts, started, k=0, rem_fast=0, rem_slow=0,
                             r_fast=1, r_slow=1) is None
    assert _migrate_boundary(counts, started, k=8, rem_fast=0, rem_slow=0,
                             r_fast=1, r_slow=1) is None
    assert _migrate_boundary(counts, started, k=4, rem_fast=0, rem_slow=0,
                             r_fast=0, r_slow=1) is None
