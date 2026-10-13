"""生命周期与资源收尾回归测试（显存 / 会话 / 池）。

覆盖：
  · GpuOutputReducer 扩容时旧 _prob_dev 未释放；stream 所有权（owned 销毁 /
    borrowed 保留）；
  · GPU 帧池 _YFramePool 归还/复用/并发的显存泄漏钉子；
  · 校准异常时 OCR 会话与 reader 的收尾（宿主管线）；OcrSession 注入路径与
    等待循环出口；
  · OCR 引擎池总量上限、淘汰不自噬、池 key 含 gamma。

reducer 与帧池项需要 cuda.bindings（monkeypatch 其 runtime 函数为记录器，
不实际分配显存）；其余纯逻辑，无需 CUDA/视频/decord。
"""

from __future__ import annotations

import threading

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
    red._launch_cls = lambda **k: object()  # reduce 先求值 _launch_cls(...)
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

    rec = {"freed": [], "destroys": [], "syncs": [], "malloc_handles": iter(range(1000, 3000))}
    monkeypatch.setattr(cudart, "cudaMalloc", lambda n: (0, next(rec["malloc_handles"])))
    # 假件须复刻真实返回形状 (err,)：cuda-python 返回
    # 元组不抛异常，调用方按 `_err, = ...` 解包——返回 None 会误伤。
    monkeypatch.setattr(cudart, "cudaFree", lambda p: (rec["freed"].append(p), (0,))[1])
    monkeypatch.setattr(
        cudart, "cudaStreamSynchronize", lambda s: (rec["syncs"].append(s), (0,))[1]
    )
    monkeypatch.setattr(cudart, "cudaStreamDestroy", lambda s: (rec["destroys"].append(s), (0,))[1])
    monkeypatch.setattr(cudart, "cudaMemcpy", lambda *a, **k: (0,))
    monkeypatch.setattr(cudart, "cudaMemcpyAsync", lambda *a, **k: (0,))
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
    assert 888 in cuda_bindings["freed"], (
        f"扩容应释放旧 _prob_dev，实际释放序列: {cuda_bindings['freed']}"
    )


def test_reducer_d2h_is_async_on_reducer_stream(cuda_bindings, monkeypatch):
    """归约 D2H 必须走**异步 + 本流同步**，不得用阻塞 cudaMemcpy。

    阻塞 cudaMemcpy 是隐式**设备级**同步，会连带排干生产者/分析流上在飞的
    工作（decode∥analyze 重叠被破坏）。本测试用假 cudart 记录调用序列锁死
    这一不变式：async 调用 ≥2 次、同流同步恰 1 次、阻塞拷贝 0 次。
    """
    from cuda.bindings import runtime as cudart

    calls = {"async": [], "blocking": []}
    monkeypatch.setattr(
        cudart, "cudaMemcpyAsync", lambda *a, **k: (calls["async"].append(a), (0,))[1]
    )
    monkeypatch.setattr(cudart, "cudaMemcpy", lambda *a, **k: calls["blocking"].append(a))
    red = _make_reducer(stream=1001, owns=False)
    red.reduce(0, (16, 4))
    assert not calls["blocking"], "热路径不许出现阻塞 cudaMemcpy（设备级同步破坏流重叠）"
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
    red.release()  # 幂等


def test_reducer_release_borrowed_stream_not_destroyed(cuda_bindings):
    red = _make_reducer(stream=2001, owns=False)
    red._idx_dev = 999
    red.release()
    assert cuda_bindings["destroys"] == [], "借用的 stream 不应被销毁"


# ═══════════════ 3. 校准异常收尾（宿主管线） ═══════════════


class _Session:
    """OcrSession 桩（属性接口：q/results/err/wall/raw_ready）。"""

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

    驱动只吃显式 HostRunSpec——不需要 `__new__` 绕过
    `__init__` 手工补私有属性（类契约显式化的证据）。
    """
    from video_ocr_engine.pipeline.host_backend import HostRunSpec, run_host_pipeline

    vr = _VR()
    session = _Session()
    spec = HostRunSpec(
        frame_start=0,
        frame_end=None,
        sample_stride=1,
        roi=(0, 0, 10, 5),
        C=5.0,
        merge_similar=False,
        keep_crops=False,
        yuv_output=False,
        segments_similar=lambda a, b: False,
        crop_luma=lambda c: np.zeros((6, 11), dtype=np.uint8),
        batch_luma=lambda c: c,
        batch_luma_out=lambda c, out: out,
        crop_is_expected=lambda c, h, w: True,
        open_vr=lambda: vr,
        start_ocr_session=lambda _engines=None: session,
    )
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
    import video_ocr_engine.ocr.native as ocr_native

    ocr_native._ENGINE_POOL.clear()
    ocr_native._POOL_IDLE_ORDER.clear()


def test_engine_pool_respects_total_cap():
    """总空闲上限按 LRU 跨 key 生效。

    单 key 上限（4）会先于总上限触发，所以必须用多个 key 才能覆盖到
    _POOL_MAX_TOTAL 这条路径：6 个 key × 4 = 24 > 16。
    """
    import video_ocr_engine.ocr.native as ocr_native

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
    assert len(released) == total - kept, f"淘汰数应为 {total - kept}，实际 {len(released)}"
    assert {e._pool_key[3] for e in released} == {0, 1}, "应按最旧优先淘汰（key 索引 0/1 先入池）"
    _reset_pool()


def test_engine_pool_checkout_removes_from_order():
    import video_ocr_engine.ocr.native as ocr_native

    _reset_pool()
    # 池 key 是 7 元组（variant, engine_type, fill_width, num_threads,
    # pad_floor_env, gamma, gpu_ctc）——假件的 key 必须同形，否则 acquire
    # 恒未命中（走新建分支），本用例会退化成恒真、测不到 checkout。
    e = _FakeEngine(("v6_small", "tensorrt", 224, 0, -1, -1.0, -1))
    ocr_native.checkin_ocr_engine(e)
    assert id(e) in ocr_native._POOL_IDLE_ORDER
    e2 = ocr_native.acquire_ocr_engine("v6_small", "tensorrt", fill_width=224, num_threads=0)
    assert e2 is e, "同参 acquire 必须复用池内引擎（否则用例测不到 checkout）"
    assert id(e2) not in ocr_native._POOL_IDLE_ORDER, "checkout 应从 LRU 序移除"
    _reset_pool()


def test_engine_pool_release_is_idempotent():
    import video_ocr_engine.ocr.native as ocr_native

    _reset_pool()
    ocr_native.release_ocr_pool()
    ocr_native.release_ocr_pool()  # 幂等
    assert len(ocr_native._ENGINE_POOL) == 0
    assert len(ocr_native._POOL_IDLE_ORDER) == 0
    _reset_pool()


# ═══════════════ 6. GPU 帧池 _YFramePool：显存泄漏钉子 ═══════════════


def _fake_pool():
    """绕开 CUDA：用假 ptr 构造 _YFramePool 语义（只测列表管理）。"""
    from video_ocr_engine.gpu import device as dev

    class _P(dev._YFramePool):
        def __init__(self):
            self._fnb = 1024
            self._free = []
            self._lk = threading.Lock()
            self._released = False  # release_all 置位标记
            self.malloc_calls = 0

        def _malloc_frame(self):
            self.malloc_calls += 1
            return dev._YFrame(self, 0x1000 * self.malloc_calls, self._fnb)

    return _P()


def test_frame_pool_recycle_keeps_pool_alive_for_reuse():
    """recycle 后缓冲被 acquire 复用；pool 引用必须保留（GC 兜底依赖它）。

    判据：复用的是**缓冲**（ptr 相同），**不是对象**（`f2 is not f`）——停靠
    对象是被 __del__ 复活过的，CPython 对复活对象的二次死亡不再触发 __del__，
    直接复用对象 = 最终死亡永久泄漏。
    """
    import gc

    pool = _fake_pool()
    f = pool.acquire()
    pool.recycle(f)
    # 关键断言：pool 引用**不得**被置空——置空会切断 __del__ 的 GC 兜底，
    # 只靠 GC 回收的池帧将永久泄漏。
    assert f.pool is pool, "recycle 不得置空 pool（会切断 GC 回收路径）"
    assert f._recycled is True, "须置 _recycled 标志防双入列"
    orig_ptr = f.ptr  # acquire 会把旧壳失能，先存
    f2 = pool.acquire()
    assert f2 is not f, "复用必须换壳（复活对象二次死亡不再触发 __del__）"
    assert f2.ptr == orig_ptr, "缓冲本身应复用（malloc 次数不增）"
    assert f2.pool is pool
    assert f2._recycled is False, "新壳对象 _recycled 初值 False"
    assert f.ptr == 0 and f.pool is None, "旧壳必须失能（不得再持指针/池引用）"
    pool.recycle(f2)
    assert len(pool._free) == 1, "显式 recycle 后必须回到空闲列"
    del f, f2
    gc.collect()


def test_frame_pool_reuse_after_resurrection_frees_on_final_death():
    """泄漏根因钉子：复用帧的最终死亡必须可回收。

    若 acquire 直接复活停靠对象 → 其最终死亡（__del__ 已用过，复活对象二次
    死亡不触发）既不入列也不 cudaFree = +2 MiB/轮永久泄漏。锁定判据：
    复用 → 死亡 → release_all 后该缓冲恰好被释放一次。
    """
    import gc

    pytest.importorskip("cuda")
    from cuda.bindings import runtime as cudart

    pool = _fake_pool()
    f = pool.acquire()
    pool.recycle(f)  # 停靠（对象被 __del__ 或显式路径复活）
    f2 = pool.acquire()  # 复用（换壳）
    assert pool.malloc_calls == 1, "复用不得新增分配"
    freed: list = []
    orig = cudart.cudaFree
    cudart.cudaFree = lambda p: (freed.append(p), (0,))[1]
    try:
        del f, f2
        gc.collect()
        pool.release_all()
    finally:
        cudart.cudaFree = orig
    assert len(freed) == 1, "复用帧最终死亡后 release_all 应释放其缓冲（旧实现此处 0 = 泄漏）"


def test_frame_pool_recycle_reuse_does_not_leak_via_gc():
    """回归钉子：acquire→recycle 循环 N 次不新增 cudaMalloc。

    判据：标志位防双入列 + pool 保留（置空 pool 会切断 __del__ 的 GC 兜底，
    只靠 GC 回收的池帧将永久泄漏）。本用例锁定"循环复用不新增分配"这一
    **最终判据**。
    """
    pool = _fake_pool()
    for _ in range(5):
        f = pool.acquire()
        pool.recycle(f)
    assert pool.malloc_calls == 1, (
        "acquire/recycle 循环应复用同一块（malloc 次数=1），"
        "实际 %d 次 = 每轮泄漏一块显存" % pool.malloc_calls
    )


def test_frame_pool_concurrent_acquire_no_indexerror():
    """无锁 check-then-act 回归钉子：并发 acquire 不得 pop 空列表。"""
    pool = _fake_pool()
    errs: list = []

    def worker():
        try:
            for _ in range(500):
                f = pool.acquire()
                pool.recycle(f)
        except BaseException as e:
            errs.append(e)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, "并发 acquire/recycle 出错: %r" % errs[:2]


def test_frame_pool_overflow_recycle_direct_frees():
    """泄漏钉子：空闲列满时的归还**必须 cudaFree 直释**。

    「列满→告警放弃（CUDA 释放只走显式路径）」的旧语义在轮末死亡波（gc 集中
    析构数百帧、空闲列仅 _MAX）下每块永久泄漏——实测 +2.0 MiB/轮（600 块 ×
    3498B）。锁定判据：列满后多余块走直释。
    """
    pytest.importorskip("cuda")
    from video_ocr_engine.gpu import device as dev
    from cuda.bindings import runtime as cudart

    pool = _fake_pool()
    frames = [pool._malloc_frame() for _ in range(dev._YFramePool._MAX + 2)]
    freed: list = []
    orig = cudart.cudaFree
    cudart.cudaFree = lambda p: (freed.append(p), (0,))[1]
    try:
        for f in frames:
            pool.recycle(f)
    finally:
        cudart.cudaFree = orig
    assert len(pool._free) == dev._YFramePool._MAX, "空闲列应恰好填满"
    assert len(freed) == 2, "溢出的 2 块必须 cudaFree 直释（旧版此处=泄漏）"


def test_frame_pool_released_late_frame_direct_frees():
    """泄漏钉子：release_all 之后迟到帧的归还必须直释。

    把迟到帧回收入**已 release 的孤儿池**（池↔帧互引成环，GC 收环时
    _recycled=True 跳过释放）= 第二条永久泄漏路径。
    """
    pytest.importorskip("cuda")
    from cuda.bindings import runtime as cudart

    pool = _fake_pool()
    pool.release_all()  # 清列 + 置 _released
    assert pool._released is True
    f = pool._malloc_frame()  # 模拟迟到帧（现实中由在飞 payload GC 触发）
    freed: list = []
    orig = cudart.cudaFree
    cudart.cudaFree = lambda p: (freed.append(p), (0,))[1]
    try:
        pool.recycle(f)
    finally:
        cudart.cudaFree = orig
    assert len(pool._free) == 0, "已 release 的池不得再收集空闲帧（孤儿池环）"
    assert len(freed) == 1, "迟到帧必须 cudaFree 直释"


# ═══════════════ 7. OcrSession 注入路径与等待循环出口 ═══════════════


class _FakeOcrEngine:
    backend_name = "fake"

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _FakeSessionSpec:
    buffer_size = 2
    cancel = staticmethod(lambda: None)
    on_backend_used = staticmethod(lambda _n: None)
    on_degraded = staticmethod(lambda _r: None)
    progress = staticmethod(lambda *a, **k: None)
    prof_end = None
    ocr_batch_size = 1
    reorder_window = 1
    metrics = None
    pad_floor_env = None
    gamma = None
    gpu_ctc = None
    fill_width = 224
    num_threads = 0
    variant = "v6_small"
    engine_type = "tensorrt"

    def __call__(self):
        return self


def _mk_session(engines=None):
    """构造最小可用 OcrSession（引擎注入；worker 立即因空队列阻塞）。"""
    from video_ocr_engine.pipeline.ocr_stage import OcrSession

    return OcrSession(_FakeSessionSpec(), _ocr_engines=engines)


def test_session_injected_engines_are_stored_and_used():
    """注入路径回归钉子：_engines 必须存入，否则恒空 → worker IndexError。"""
    eng = _FakeOcrEngine()
    sess = _mk_session([eng])
    assert sess._engines == [eng], "_ocr_engines 必须存入 self._engines"
    assert sess._owns_engines is False
    sess.finish()


def test_session_injected_empty_list_raises_clear_error():
    """空注入列表：显式 ValueError（而非 IndexError）。"""
    sess = _mk_session([])
    # worker 异步跑：轮询等 err 出现（最长 5s）
    for _ in range(100):
        if sess.err:
            break
        sess._thread.join(timeout=0.05)
    assert sess.err, "worker 未记录错误"
    assert isinstance(sess.err[0], ValueError), "应为显式 ValueError，实际 %r" % (sess.err[0],)
    assert "注入引擎路径要求" in str(sess.err[0])


def test_session_put_raises_when_worker_died_without_err():
    """worker 死后 put 不得自旋挂死（BaseException 逃逸场景）。

    构造：直接杀线程不可行，改为等 worker 因空引擎列表退出——此时 err 非空
    （ValueError），故清 err 模拟「线程已死而 err 恒空」这一修复前会挂死的状态。
    """
    sess = _mk_session([])
    for _ in range(100):
        if not sess._thread.is_alive():
            break
        sess._thread.join(timeout=0.05)
    assert not sess._thread.is_alive(), "worker 应已退出"
    sess.err.clear()  # 模拟 BaseException 逃逸：线程死而 err 空
    with pytest.raises(RuntimeError):
        sess.put(object())


# ═══════════════ 8. OCR 引擎池：淘汰不自噬 + key 含 gamma ═══════════════


def test_engine_pool_eviction_prefers_other_key():
    """淘汰优先选其他 key（原实现会刚归还就淘汰=白付重建成本）。"""
    from video_ocr_engine.ocr import native as n

    class _E:
        def __init__(self, key):
            self._pool_key = key

        def release(self):
            pass

    with n._POOL_LOCK:
        n._ENGINE_POOL.clear()
        n._POOL_IDLE_ORDER.clear()
    saved_max = n._POOL_MAX_TOTAL
    try:
        n._POOL_MAX_TOTAL = 2
        k_other, k_new = ("other",), ("new",)
        n.checkin_ocr_engine(_E(k_other))  # 池: other
        n.checkin_ocr_engine(_E(k_other))  # 池: other×2
        e_new = _E(k_new)
        n.checkin_ocr_engine(e_new)  # 超上限 → 淘汰应选 other
        with n._POOL_LOCK:
            still_new = any(cand is e_new for cand in n._ENGINE_POOL.get(k_new, []))
        assert still_new, "刚归还的引擎被自己挤掉了（淘汰自噬）"
    finally:
        n._POOL_MAX_TOTAL = saved_max
        with n._POOL_LOCK:
            n._ENGINE_POOL.clear()
            n._POOL_IDLE_ORDER.clear()


def test_engine_pool_key_includes_gamma():
    """gamma 构造期冻结 → 必须进池 key（否则同进程改 env 后旋钮失效）。"""
    import inspect

    from video_ocr_engine.ocr import native as n

    src = inspect.getsource(n.acquire_ocr_engine)
    assert "round(float(gamma), 6)" in src, "池 key 必须含 gamma"
    key_block = src.split("key = (")[1].split("\n\n")[0]
    assert "gpu_ctc" in key_block, "池 key 必须含 gpu_ctc"
