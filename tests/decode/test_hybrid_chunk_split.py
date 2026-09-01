"""hybrid 分片粒度上限（HYBRID_MAX_CHUNK_FRAMES）纯函数测试。

验证 _split_oversized：拆后每片帧数 ≤ 上限、覆盖完整无缝隙、边界吸附
采样帧、关键帧优先切分。纯单元测试，无需视频/decord/GPU。
"""
from __future__ import annotations

import threading

import hybrid_decode as hd
from hybrid_decode import HybridDecoder
from hybrid_decode import _split_oversized


def _frames(n: int, stride: int = 1) -> list[int]:
    return list(range(0, n * stride, stride))


def _count(specs, frames) -> list[int]:
    """每片实际采样帧数（按 frames 过滤）。"""
    return [sum(1 for f in frames if a <= f < b) for a, b in specs]


def test_max_frames_zero_returns_original():
    specs = [(0, 100), (100, 200)]
    assert _split_oversized(specs, _frames(200), [], 0) == specs


def test_small_chunks_untouched():
    frames = _frames(300)
    specs = [(0, 100), (100, 200), (200, 300)]
    out = _split_oversized(specs, frames, [], 128)
    assert out == specs


def test_split_by_frame_count_when_no_keyframes():
    frames = _frames(1000)
    specs = [(0, 1000)]
    out = _split_oversized(specs, frames, [], 256)
    counts = _count(out, frames)
    assert all(c <= 256 for c in counts)
    assert sum(counts) == 1000
    # 覆盖连续无缝隙
    assert out[0][0] == 0 and out[-1][1] == 1000
    for (_, b), (a2, _) in zip(out[:-1], out[1:]):
        assert b == a2


def test_split_prefers_keyframe_boundaries():
    frames = _frames(1000)
    # 关键帧在 500（采样帧 500）
    out = _split_oversized([(0, 1000)], frames, [500], 256)
    counts = _count(out, frames)
    assert all(c <= 256 for c in counts)
    # 关键帧 500 应是某个切点
    bounds = {a for a, _ in out[1:]} | {b for _, b in out[:-1]}
    assert 500 in bounds


def test_split_boundaries_stay_on_sampled_grid():
    frames = _frames(1000, stride=2)   # 0,2,4,...,1998
    specs = [(0, 2000)]
    out = _split_oversized(specs, frames, [], 256)
    for a, b in out:
        assert a % 2 == 0
        assert b % 2 == 0 or b == 2000


def test_split_multiple_oversized_chunks():
    frames = _frames(3000)
    specs = [(0, 1500), (1500, 3000)]
    out = _split_oversized(specs, frames, [750, 2250], 512)
    counts = _count(out, frames)
    assert all(c <= 512 for c in counts)
    assert sum(counts) == 3000
    # 关键帧应作为切点
    bounds = {a for a, _ in out[1:]} | {b for _, b in out[:-1]}
    assert 750 in bounds and 2250 in bounds


class _ReaderStub:
    def get_key_indices(self):
        return []

    def seek_accurate(self, _fi):
        return None


class _DoneThread:
    def join(self):
        return None


class _ExStub:
    _sample_stride = 1
    _roi = (0, 0, 10, 10)


def test_hybrid_begin_applies_chunk_frame_limit(monkeypatch):
    """入口必须把 HYBRID_MAX_CHUNK_FRAMES 传给模块级拆片函数。"""
    dec = HybridDecoder.__new__(HybridDecoder)
    dec._ex = _ExStub()
    dec._gpu = _ReaderStub()
    dec._cpu = _ReaderStub()
    dec._cpu_thread = _DoneThread()
    dec._cpu_open_err = []
    dec._f0, dec._f1 = 0, 600
    dec._roi = (0, 0, 11, 11)
    dec._max_chunks = 16
    dec._min_gap = 16
    dec._max_chunk_frames = 100
    dec._calib_frames = 40
    dec._chunks = []
    dec._starts = []
    dec._begun = False
    dec._closed = False
    dec._stop = threading.Event()
    dec._err = []
    dec._cv = threading.Condition()
    dec._threads = []
    dec._probe = False
    dec._probe_csv = None
    dec._probe_rows = []
    dec._pname = {}
    dec._probe_lock = threading.Lock()
    dec._fast_tag = "gpu"
    dec._fast_reader = None
    dec._slow_reader = None
    dec._split_idx = 0
    dec._inflight = 2
    dec._inflight_slow = 4
    dec._unconsumed = {"fast": 0, "slow": 0}
    # Producers are unrelated to this call-site test; avoid starting a decode.
    dec._producer = lambda _reader: None
    monkeypatch.setattr(hd, "_measure_rate", lambda *args, **kwargs: 100.0)

    dec.hybrid_begin(_frames(600))

    counts = [len(ch["fis"]) for ch in dec._chunks]
    assert counts == [100] * 6
    assert sum(counts) == 600
