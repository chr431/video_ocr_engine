"""hybrid v4 动态分界（_dynamic_split）纯函数测试。

验证：慢端不拖尾约束下给慢端尽量多片；慢端=CPU 时稳态折扣更保守；
两端至少各 1 片；max_share 上限。
"""
from __future__ import annotations

import pytest

from hybrid_decode import _dynamic_split


def test_cpu_slow_hevc_gets_few_chunks():
    """HEVC 8 核：CPU 稳态 ~200fps vs GPU ~1900fps（9x 慢）→ 慢端只 1-2 片。"""
    counts = [286] * 11
    # 校准 48 帧快测：gpu=1900 cpu=800（高估），CPU 折扣 0.45 → 稳态 360
    split = _dynamic_split(counts, 1900, 800, slow_is_cpu=True)
    slow = 11 - split
    assert 1 <= slow <= 2, f"慢端应 1-2 片，实际 {slow}"


def test_cpu_close_to_gpu_gets_more_chunks():
    """h264 8 核：CPU 754 vs GPU 980（1.3x 慢）→ 慢端可分 2+ 片。"""
    counts = [286] * 11
    split = _dynamic_split(counts, 980, 754, slow_is_cpu=True)
    slow = 11 - split
    assert slow >= 2, f"慢端应 ≥2 片，实际 {slow}"


def test_gpu_slow_uses_less_discount():
    """慢端=NVDEC 时折扣 0.85（稳态略降）→ 与 CPU 慢端相比慢端可更多片。"""
    counts = [286] * 11
    s_cpu = _dynamic_split(counts, 1900, 800, slow_is_cpu=True)
    s_gpu = _dynamic_split(counts, 1900, 800, slow_is_cpu=False)
    assert 11 - s_gpu >= 11 - s_cpu


def test_at_least_one_chunk_each_side():
    """极端慢端也保证两端各至少 1 片。"""
    counts = [286] * 11
    split = _dynamic_split(counts, 10000, 10, slow_is_cpu=True)
    assert 1 <= split <= 10


def test_very_small_n():
    assert _dynamic_split([100, 100], 1000, 500, slow_is_cpu=True) == 1
    assert _dynamic_split([100], 1000, 500, slow_is_cpu=True) == 1


def test_max_share_respected():
    """慢端份额不超过 max_share。"""
    counts = [100] * 10
    split = _dynamic_split(counts, 1000, 900, slow_is_cpu=False,
                           max_share=0.3)
    slow = 10 - split
    assert slow * 100 / 1000 <= 0.3 + 1e-9


def test_uneven_chunks_uses_frame_counts():
    """片大小不均匀时按帧数计算（大尾片触发约束提前停止）。"""
    # 最后一片超大（1000 帧），慢端 1 片就拖尾 → 只给 1 片
    counts = [100, 100, 100, 1000]
    split = _dynamic_split(counts, 2000, 400, slow_is_cpu=True)
    slow = 4 - split
    assert slow == 1
