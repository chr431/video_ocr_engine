"""生命周期收尾回归测试。

覆盖 PERFORMACE_REPORT.txt 指出的确定性清理缺陷：
  1. GpuOutputReducer 扩容时旧 _prob_dev 未释放；
  2. GPU helper 的 stream 所有权与释放（owned 销毁 / borrowed 保留）；
  3. 校准异常时 OCR 会话与 reader 的收尾（宿主管线）；
  5. OCR 引擎池总量上限与 LRU 淘汰。

第 1/2 项需要真实 cuda.bindings（monkeypatch 其 runtime 函数为记录器，
不实际分配显存）；第 3~5 项纯逻辑，无需 CUDA/视频/decord。
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from collections import deque

import numpy as np
import pytest


# ═══════════════ 通用假对象 ═══════════════

class _FakeBuffer:
    @staticmethod
    def from_handle(ptr, n):
        return object()


def _make_reducer(stream=1001, owns=True):
    """构造不依赖 cuda.core 的 GpuOutputReducer（绕过 __init__）。"""
    from video_ocr_engine._gpu_kernels import GpuOutputReducer
    red = GpuOutputReducer.__new__(GpuOutputReducer)
    red._dev = None
    red._mod = None
    red._kernel = None
    red._launch_cls = lambda **k: object()      # reduce 先求值 _launch_cls(...)
    red._launch = lambda *a, **k: None
    red._buffer_cls = _FakeBuffer
    red._idx_dev = None
    red._idx_size = 0
    red._prob_dev = None
    red._stream = stream
    red._owns_stream = owns
    return red


@pytest.fixture
def cuda_bindings(monkeypatch):
    """把 cuda.bindings.runtime 的函数替换成记录器；无 cuda.bindings 则跳过。"""
    pytest.importorskip("cuda.bindings")
    from cuda.bindings import runtime as cudart
    rec = {"freed": [], "destroys": [], "syncs": [],
           "malloc_handles": iter(range(1000, 3000))}
    monkeypatch.setattr(cudart, "cudaMalloc",
                        lambda n: (0, next(rec["malloc_handles"])))
    monkeypatch.setattr(cudart, "cudaFree",
                        lambda p: rec["freed"].append(p))
    monkeypatch.setattr(cudart, "cudaStreamSynchronize",
                        lambda s: rec["syncs"].append(s))
    monkeypatch.setattr(cudart, "cudaStreamDestroy",
                        lambda s: rec["destroys"].append(s))
    monkeypatch.setattr(cudart, "cudaMemcpy", lambda *a, **k: None)
    monkeypatch.setattr(cudart, "cudaMemcpyAsync", lambda *a, **k: None)
    return rec


# ═══════════════ 1 & 2. GPU 资源所有权 / stream 释放 ═══════════════

def test_reducer_realloc_frees_old_prob_dev(cuda_bindings):
    """扩容 _idx_dev 时必须显式释放旧 _prob_dev（报告指出的泄漏）。"""
    red = _make_reducer(stream=1001, owns=False)
    red._idx_dev = 999
    red._idx_size = 4
    red._prob_dev = 888
    # 4 < 64 → 触发扩容
    red.reduce(0, (16, 4))
    assert 888 in cuda_bindings["freed"], \
        f"扩容应释放旧 _prob_dev，实际释放序列: {cuda_bindings['freed']}"


def test_reducer_d2h_is_async_on_reducer_stream(cuda_bindings, monkeypatch):
    """S6-c / PI-14：归约 D2H 必须走**异步 + 本流同步**，不得用阻塞 cudaMemcpy。

    阻塞 cudaMemcpy 是隐式**设备级**同步，会连带排干生产者/分析流上在飞的
    工作（decode∥analyze 重叠被破坏）。本测试用假 cudart 记录调用序列锁死
    这一不变式：async 调用 ≥2 次、同流同步恰 1 次、阻塞拷贝 0 次。
    """
    from cuda.bindings import runtime as cudart
    calls = {"async": [], "blocking": []}
    monkeypatch.setattr(cudart, "cudaMemcpyAsync",
                        lambda *a, **k: calls["async"].append(a))
    monkeypatch.setattr(cudart, "cudaMemcpy",
                        lambda *a, **k: calls["blocking"].append(a))
    red = _make_reducer(stream=1001, owns=False)
    red.reduce(0, (16, 4))
    assert not calls["blocking"], "热路径不许出现阻塞 cudaMemcpy（PI-14）"
    assert len(calls["async"]) == 2, "idx/prob 两次 D2H 都应异步"
    assert cuda_bindings["syncs"] == [1001], "应收尾同步本流一次"


def test_reducer_release_destroys_only_owned_stream(cuda_bindings):
    # owned：释放缓冲 + 销毁 stream
    red = _make_reducer(stream=1001, owns=True)
    red._idx_dev = 999
    red._prob_dev = 888
    red.release()
    assert sorted(cuda_bindings["freed"]) == [888, 999]
    assert cuda_bindings["destroys"] == [1001]
    assert red._stream is None and red._owns_stream is False
    red.release()   # 幂等


def test_reducer_release_borrowed_stream_not_destroyed(cuda_bindings):
    red = _make_reducer(stream=2001, owns=False)
    red._idx_dev = 999
    red.release()
    assert cuda_bindings["destroys"] == [], "借用的 stream 不应被销毁"


# ═══════════════ 3. 校准异常收尾（宿主管线） ═══════════════

class _Session:
    """OcrSession 桩（0.11.0 起生产侧为属性接口：q/results/err/wall/raw_ready）。"""

    def __init__(self):
        from queue import Queue
        self.finished = False
        self.q = Queue()
        self.results = {}
        self.err = []
        self.wall = [0.0]
        self.raw_ready = [False]

    def put(self, *a, **k):
        pass

    def finish(self):
        self.finished = True


class _VR:
    def __init__(self):
        self.closed = False
    def __len__(self):
        return 100
    def get_avg_fps(self):
        return 30.0
    def next_roi(self, *a, **k):
        raise RuntimeError("calib boom")
    def close(self):
        self.closed = True


def test_host_calibration_failure_finishes_session_and_closes_reader():
    """校准抛异常时，OCR 会话应被 finish、reader 应被 close。

    S3-3a 起：驱动只吃显式 HostRunSpec——不再需要 `__new__` 绕过
    `__init__` 手工补 12 个私有属性（P1-7"类契约是隐式的"的直接证据）。
    """
    from video_ocr_engine.pipeline.host_backend import (
        HostRunSpec, run_host_pipeline)

    vr = _VR()
    session = _Session()
    spec = HostRunSpec(
        frame_start=0, frame_end=None, sample_stride=1, roi=(0, 0, 10, 5),
        C=5.0, merge_similar=False, keep_crops=False, yuv_output=False,
        segments_similar=lambda a, b: False,
        crop_luma=lambda c: np.zeros((6, 11), dtype=np.uint8),
        batch_luma=lambda c: c, batch_luma_out=lambda c, out: out,
        crop_is_expected=lambda c, h, w: True,
        open_vr=lambda: vr,
        start_ocr_session=lambda _engines=None: session)
    # next_roi 抛错 → 校准抛错 → 清理路径
    with pytest.raises(RuntimeError, match="calib boom"):
        run_host_pipeline(spec)
    assert session.finished is True, "校准异常应 finish OCR 会话"
    assert vr.closed is True, "校准异常应 close reader"



# ═══════════════ 5. OCR 引擎池 LRU 上限 ═══════════════

class _FakeEngine:
    def __init__(self, key):
        self._pool_key = key
        self.released = 0
    def release(self):
        self.released += 1


def _reset_pool():
    import ocr_native
    ocr_native._ENGINE_POOL.clear()
    ocr_native._POOL_IDLE_ORDER.clear()


def test_engine_pool_respects_total_cap():
    """总空闲上限按 LRU 跨 key 生效。

    单 key 上限（4）会先于总上限触发，所以必须用多个 key 才能覆盖到
    _POOL_MAX_TOTAL 这条路径：6 个 key × 4 = 24 > 16。
    """
    import ocr_native
    _reset_pool()
    released = []
    n_keys = 6
    for ki in range(n_keys):
        key = ("v6_small", "tensorrt", 224, ki)
        for _ in range(4):
            e = _FakeEngine(key)
            orig = e.release

            def _rel(e=e):
                released.append(e)
                orig()
            e.release = _rel
            ocr_native.checkin_ocr_engine(e)
    total = 4 * n_keys
    cap = ocr_native._POOL_MAX_TOTAL
    kept = sum(len(v) for v in ocr_native._ENGINE_POOL.values())
    assert kept <= cap, f"空闲总数应受总上限限制: kept={kept} cap={cap}"
    assert len(ocr_native._POOL_IDLE_ORDER) == kept, "LRU 序与池内容不一致"
    assert len(released) == total - kept, \
        f"淘汰数应为 {total - kept}，实际 {len(released)}"
    assert {e._pool_key[3] for e in released} == {0, 1}, \
        "应按最旧优先淘汰（key 索引 0/1 先入池）"
    _reset_pool()


def test_engine_pool_checkout_removes_from_order():
    import ocr_native
    _reset_pool()
    e = _FakeEngine(("v6_small", "tensorrt", 224, 0))
    ocr_native.checkin_ocr_engine(e)
    assert id(e) in ocr_native._POOL_IDLE_ORDER
    e2 = ocr_native.acquire_ocr_engine(
        "v6_small", "tensorrt", fill_width=224, num_threads=0)
    assert id(e2) not in ocr_native._POOL_IDLE_ORDER
    _reset_pool()


def test_engine_pool_release_is_idempotent():
    import ocr_native
    _reset_pool()
    ocr_native.release_ocr_pool()
    ocr_native.release_ocr_pool()   # 幂等
    assert len(ocr_native._ENGINE_POOL) == 0
    assert len(ocr_native._POOL_IDLE_ORDER) == 0
    _reset_pool()
