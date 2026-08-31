"""生命周期收尾回归测试。

覆盖 PERFORMACE_REPORT.txt 指出的确定性清理缺陷：
  1. GpuOutputReducer 扩容时旧 _prob_dev 未释放；
  2. GPU helper 的 stream 所有权与释放（owned 销毁 / borrowed 保留）；
  3. 校准异常时 OCR 会话与 reader 的收尾（宿主管线）；
  4. HybridDecoder.close 幂等、join 后台线程、释放已交付分片数据；
  5. OCR 引擎池总量上限与 LRU 淘汰。

第 1/2 项需要真实 cuda.bindings（monkeypatch 其 runtime 函数为记录器，
不实际分配显存）；第 3~5 项纯逻辑，无需 CUDA/视频/decord。
"""
from __future__ import annotations

import threading
from collections import OrderedDict

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
    def __init__(self):
        self.finished = False
    def __getitem__(self, k):
        if k == "finish":
            return self._finish
        if k == "q":
            from queue import Queue
            return Queue()
        if k == "results":
            return {}
        if k == "err":
            return []
        if k == "wall":
            return [0.0]
        if k == "put":
            return lambda *a, **k: None
        if k == "raw_ready":
            return [False]
        raise KeyError(k)
    def _finish(self):
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
    """校准抛异常时，OCR 会话应被 finish、reader 应被 close。"""
    from video_ocr_engine import FieldExtractor
    from video_ocr_engine._host_pipeline import _host_calibrate

    vr = _VR()
    session = _Session()

    ex = FieldExtractor.__new__(FieldExtractor)
    ex._gpu_pipeline_mode = False
    ex._fps = None
    ex._frame_start = 0
    ex._frame_end = None
    ex._sample_stride = 1
    ex._color_range = 0
    ex._codec = ""
    ex._backend = "decord/CPU"
    ex._degraded = []
    ex._roi = (0, 0, 10, 5)
    ex.timing = {}
    ex._prof_end = lambda *a, **k: None
    ex._open_vr = lambda: vr
    ex._start_ocr_session = lambda _o=None: session
    ex._crop_is_expected = lambda *a, **k: True
    ex._crop_luma = lambda c: np.zeros((6, 11), dtype=np.uint8)
    ex._merge_effective_mode = lambda: ""
    ex._merge_text_sep = ""
    # next_roi 抛错 → _host_calibrate 抛错
    with pytest.raises(RuntimeError):
        FieldExtractor._run_pipelined_host(ex)
    assert session.finished is True, "校准异常应 finish OCR 会话"
    assert vr.closed is True, "校准异常应 close reader"



# ═══════════════ 4. HybridDecoder 生命周期 ═══════════════

class _CountingVR:
    def __init__(self):
        self.seeks = []
        self.closed = False
    def seek_accurate(self, fi): self.seeks.append(fi)
    def close(self): self.closed = True


def _hybrid_stub():
    from hybrid_decode import HybridDecoder
    dec = HybridDecoder.__new__(HybridDecoder)
    dec._closed = False
    dec._stop = threading.Event()
    dec._cv = threading.Condition()
    dec._threads = []
    dec._cpu_thread = None
    dec._cpu = None
    dec._gpu = _CountingVR()
    dec._probe = False
    dec._probe_rows = []
    return dec


def test_hybrid_close_is_idempotent_and_joins_threads():
    dec = _hybrid_stub()

    def worker():
        # 真实后台线程靠 _stop 退出；close 必须先置位再 join。
        dec._stop.wait(10.0)
    t = threading.Thread(target=worker)
    t.start()
    dec._threads = [t]
    dec.close()
    assert not t.is_alive(), "close 应 join 后台线程后再返回"
    assert dec._stop.is_set(), "close 应先置 stop 再 join"
    dec.close()   # 幂等，不抛
    assert dec._closed is True
    assert dec._gpu.closed is True, "close 应关闭底层 GPU reader"


def test_hybrid_pop_frames_releases_delivered_data():
    """已交付的帧应从 ch['data'] 删除，避免 NumPy 数组被钉住。"""
    dec = _hybrid_stub()
    n = 8
    # counted=True：模拟"生产者已完成本片并计数"这一可达时序；done 与
    # counted 由生产者同时置位，done=True+counted=False 在真实流程中不可达。
    ch = {"fis": list(range(n)),
          "data": [(i, np.full((2, 2), i)) for i in range(n)],
          "off": 0, "delivered": 0, "all_delivered": False, "counted": True,
          "consumed": False, "done": True, "owner": "fast", "started": True}
    dec._chunks = [ch]
    dec._starts = [0]
    dec._unconsumed = {"fast": 1, "slow": 0}
    dec._inflight = dec._inflight_slow = 8
    dec._split_idx = 1
    dec._fast_reader = _CountingVR()
    dec._slow_reader = None
    dec._err = []

    got = dec._pop_frames(list(range(n)))
    assert len(got) == n
    assert ch["data"] == [], "已交付数据应被删除"
    assert ch["delivered"] == n
    assert dec._unconsumed["fast"] == 0, "排空后应释放未消费计数"


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
