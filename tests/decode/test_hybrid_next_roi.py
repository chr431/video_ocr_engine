"""HybridDecoder.next_roi 采样步长推进测试（防隐性帧号错位回归）。

现役 hybrid 安全门要求 stride==1，但 next_roi 的帧号推进必须与采样网格
一致：硬编码 +1 在放宽安全门（stride>1）后会使校准帧号错位。本测试用
桩 reader 直接驱动 HybridDecoder.next_roi，验证步长推进与交付序。
"""
from __future__ import annotations

import threading

import numpy as np

from hybrid_decode import HybridDecoder


class _StubVR:
    """最小 VideoReader 桩：只提供 HybridDecoder 构造需要触碰的接口。"""

    def __init__(self, n: int):
        self._n = n

    def __len__(self):
        return self._n

    def get_batch(self, frame_list, roi=None):
        return _Batch(np.zeros((len(frame_list), 2, 2), dtype=np.uint8))

    def get_key_indices(self):
        return []

    def get_avg_fps(self):
        return 30.0

    def get_fps(self):
        return 30.0

    def get_color_range(self):
        return 0

    def get_codec(self):
        return "h264"


class _Batch:
    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr


class _StubEx:
    """FieldExtractor 桩：构造 HybridDecoder 所需的最小属性。"""

    def __init__(self, stride: int, roi=(0, 0, 10, 10)):
        self._sample_stride = stride
        self._roi = roi
        self._frame_start = 0
        self._frame_end = None


def _make_decoder(stride: int, n: int = 100):
    ex = _StubEx(stride)
    gpu = _StubVR(n)
    # 只构造对象：不调 hybrid_begin（避免测速/线程），直接操纵分片状态
    dec = HybridDecoder.__new__(HybridDecoder)
    dec._ex = ex
    dec._gpu = gpu
    dec._f0, dec._f1 = 0, n
    dec._starts = [0, 50]
    dec._chunks = [
        {'fis': list(range(0, 50, stride)), 'data': [], 'off': 0,
         'done': True, 'owner': None, 'started': True},
        {'fis': list(range(50, n, stride)), 'data': [], 'off': 0,
         'done': True, 'owner': None, 'started': True},
    ]
    # 预填充数据：每片 fis 对应的占位 crop（next_roi 只消费 _pop_frames，
    # 不真正解码）。片已 done 且数据齐备 → 消费不会阻塞。
    for ch in dec._chunks:
        for fi in ch['fis']:
            ch['data'].append((fi, fi))
    dec._seq_fi = None
    # _pop_frames 依赖的状态（__new__ 绕过 __init__，手动补齐）
    dec._stop = threading.Event()
    dec._err = []
    dec._cv = threading.Condition()
    dec._unconsumed = {}
    return dec


def test_next_roi_stride_one_advances_by_one():
    dec = _make_decoder(stride=1)
    got = [dec.next_roi(0, 0, 1, 1).asnumpy() for _ in range(5)]
    assert got == [0, 1, 2, 3, 4]


def test_next_roi_stride_two_advances_by_two():
    dec = _make_decoder(stride=2)
    got = [dec.next_roi(0, 0, 1, 1).asnumpy() for _ in range(5)]
    assert got == [0, 2, 4, 6, 8]


def test_next_roi_stride_three_advances_by_three():
    dec = _make_decoder(stride=3)
    got = [dec.next_roi(0, 0, 1, 1).asnumpy() for _ in range(5)]
    assert got == [0, 3, 6, 9, 12]


def test_next_roi_starts_at_first_chunk_start():
    dec = _make_decoder(stride=2)
    assert dec.next_roi(0, 0, 1, 1).asnumpy() == 0
