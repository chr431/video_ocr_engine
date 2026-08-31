"""HybridDecoder._producer 片间连续扫掠的步长推进测试（防 per-chunk seek 回归）。

## 背景（stride>1 的隐性性能 bug）
`_producer` 靠"下一片首帧 == 上一片末帧 + 步长"来判定可以续着扫、不 seek
（v3 的核心设计：每生产者 0~1 次 seek）。步长原先硬编码 `+1`，stride==1 时
恒等所以没暴露；stride>1 时下一片首帧是 `fis[-1] + stride`，判定永远失败 →
**每片一次 seek**（GPU 50~190ms/次、CPU 35~65ms/次，16 片 ≈ 1.6s 纯开销）。

实测后果（test5/test.mp4/test6，3000 帧）：hybrid 在 stride=8 下比纯 NVDEC
**慢 38%~59%**，且比任一单端独跑都慢——固定开销之外纯粹是 seek 打爆的。

与 `test_hybrid_next_roi.py` 是同一类缺陷（`+1` 漏 stride），两个方向都要守住：
next_roi 管"顺序交付流"，_producer 管"批量分片流"。

本测试用桩 reader 数 seek 次数：修复后慢端 N 片只应 seek 1 次（首片），
回归后变成 N 次。
"""
from __future__ import annotations

import threading

import numpy as np

from hybrid_decode import HybridDecoder


class _Batch:
    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr


class _CountingVR:
    """记录 seek 次数的 VideoReader 桩。"""

    def __init__(self):
        self.seeks: list[int] = []
        self.batches = 0

    def seek_accurate(self, fi: int):
        self.seeks.append(int(fi))

    def get_batch(self, frame_list, roi=None):
        self.batches += 1
        return _Batch(np.zeros((len(frame_list), 2, 2), dtype=np.uint8))


class _StubEx:
    def __init__(self, stride: int):
        self._sample_stride = stride
        self._roi = (0, 0, 10, 10)
        self._frame_start = 0
        self._frame_end = None


def _make(stride: int, chunk_bounds: list[int]):
    """构造解码器：chunk_bounds 给出每片的首帧（末片隐含到 total）。"""
    total = 100
    dec = HybridDecoder.__new__(HybridDecoder)
    roi = (0, 0, 10, 10)
    dec._ex = _StubEx(stride)
    # _roi 由 __init__ 从 ex._roi 派生（右下角 +1，同 __init__ 语义），
    # __new__ 绕过构造必须补齐，否则 _producer 会在首帧就抛异常——
    # 那样 seek 计数恰好也是 1，会让下面的断言"假通过"。
    dec._roi = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)
    dec._stop = threading.Event()
    dec._err = []
    dec._cv = threading.Condition()
    dec._pname = {}
    dec._f0, dec._f1 = 0, total
    bounds = chunk_bounds
    chunks = []
    for i, a in enumerate(bounds):
        b = bounds[i + 1] if i + 1 < len(bounds) else total
        chunks.append({'fis': list(range(a, b, stride)), 'data': [], 'off': 0,
                       'done': False, 'owner': None, 'started': False})
    dec._chunks = chunks
    dec._starts = list(bounds)
    dec._unconsumed = {'fast': 0, 'slow': 0}
    dec._inflight = 8
    dec._inflight_slow = 8
    # 慢端区从第 1 片起（第 0 片留给快端），让 _producer 连扫多片
    dec._split_idx = 1
    dec._probe = False
    dec._probe_lock = threading.Lock()
    dec._probe_rows = []
    # _producer 用 self._fast_reader 判身份：传别的 reader 即为慢端
    dec._fast_reader = _CountingVR()
    dec._slow_reader = None
    dec._fast_tag = "fast"
    return dec


def _run_producer(dec: HybridDecoder, reader: _CountingVR) -> None:
    """单线程直接跑慢端生产者（不走线程，确定性）。"""
    dec._producer(reader)


def test_producer_stride_one_sweeps_without_extra_seeks():
    dec = _make(stride=1, chunk_bounds=[0, 25, 50, 75])
    r = _CountingVR()
    _run_producer(dec, r)
    # 慢端认领 1..3 共 3 片：首片 seek 一次，后续续扫
    assert len(r.seeks) == 1, f"stride=1 应只 seek 1 次，实际 {r.seeks}"


def test_producer_stride_eight_sweeps_without_extra_seeks():
    """回归点：stride>1 时 prev_end 必须按 stride 推进，否则每片一 seek。"""
    dec = _make(stride=8, chunk_bounds=[0, 24, 48, 72])
    r = _CountingVR()
    _run_producer(dec, r)
    assert len(r.seeks) == 1, (
        f"stride=8 应只 seek 1 次（首片），实际 {len(r.seeks)} 次 {r.seeks} —— "
        "prev_end 未按采样步长推进，片间连续扫掠失效")


def test_producer_stride_three_sweeps_without_extra_seeks():
    dec = _make(stride=3, chunk_bounds=[0, 30, 60, 90])
    r = _CountingVR()
    _run_producer(dec, r)
    assert len(r.seeks) == 1, f"stride=3 应只 seek 1 次，实际 {r.seeks}"


def test_producer_produces_every_frame_of_every_chunk():
    """步长修复不能牺牲交付完整性：每片所有采样帧都要进 ch['data']。"""
    dec = _make(stride=8, chunk_bounds=[0, 24, 48, 72])
    r = _CountingVR()
    _run_producer(dec, r)
    for i, ch in enumerate(dec._chunks):
        if i == 0:      # 快端区，慢端生产者不碰
            continue
        got = [fi for fi, _ in ch['data']]
        assert got == ch['fis'], f"片 {i} 交付帧号与 fis 不一致"
