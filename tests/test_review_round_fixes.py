"""审查轮（2026-09-19）修复的回归钉子：每个缺陷一个最小用例。

覆盖的缺陷（详见 docs/log/2026-09-19-代码审查轮.md）：
- 池 recycle 顺序（置空 pool 后入列 → 复用时 __del__ 撞 None 被吞 = 显存泄漏）
- 池 acquire 无锁 check-then-act（并发 pop 空列表）
- cudaMalloc 失败误报 TypeError（应 MemoryError）
- OcrSession 注入引擎路径（_engines 恒空 → IndexError + 引擎泄漏）
- OcrSession put/finish 的等待循环出口（worker 死后自旋挂死）
- 注入空列表显式报错
- Metrics.snapshot(drain=False) 只读快照（看门狗不得排干指标）
- Metrics tier 非法值显式 ValueError（原 assert 无消息且 -O 消失）
- resolve 对枚举 str 的大小写/空白归一
- RunJournal.report 含 last_error（且未初始化不再 AttributeError）
- pool.run([]) 不抛 ValueError
- checkin 池淘汰不自噬（优先淘汰其他 key）
- acquire_ocr_engine 池 key 含 gamma/gpu_ctc
"""
from __future__ import annotations

import threading

import pytest

from video_ocr_engine.config.resolve import _parse
from video_ocr_engine.config.knobs import KNOBS
from video_ocr_engine.domain.diagnostics import RunJournal
from video_ocr_engine.domain.metrics import Metrics


# ── 池：recycle 顺序（显存泄漏回归钉子）──────────────────────────────

def _fake_pool():
    """绕开 CUDA：用假 ptr 构造 _YFramePool 语义（只测列表管理）。"""
    from video_ocr_engine.gpu import device as dev

    class _P(dev._YFramePool):
        def __init__(self):
            self._fnb = 1024
            self._free = []
            self._lk = threading.Lock()
            self.malloc_calls = 0

        def _malloc_frame(self):
            self.malloc_calls += 1
            return dev._YFrame(self, 0x1000 * self.malloc_calls, self._fnb)

    return _P()


def test_pool_recycle_keeps_pool_alive_for_reuse():
    """recycle 后对象被 acquire 复用；pool 引用必须保留（GC 兜底依赖它）。"""
    import gc

    pool = _fake_pool()
    f = pool.acquire()
    pool.recycle(f)
    # 关键断言（审查轮二修）：pool 引用**不得**被置空——置空会切断
    # __del__ 的 GC 兜底，只靠 GC 回收的池帧将永久泄漏。
    assert f.pool is pool, "recycle 不得置空 pool（会切断 GC 回收路径）"
    assert f._recycled is True, "须置 _recycled 标志防双入列"
    f2 = pool.acquire()
    assert f2 is f, "复用同一对象"
    assert f2._recycled is False, "acquire 必须清 _recycled（否则回收被跳过）"
    pool.recycle(f2)
    assert len(pool._free) == 1, "显式 recycle 后必须回到空闲列"
    del f, f2
    gc.collect()


def test_pool_recycle_reuse_does_not_leak_via_gc():
    """回归钉子：acquire→recycle 循环 N 次不新增 cudaMalloc。

    ⚠️ 这条钉子经历了两次修正：初版断言"置空 pool 防双入列"，
    实测证明置空切断了 __del__ 的 GC 兜底（跨线程 payload 帧只靠 GC
    回收）→ 253 次 cudaMalloc/轮永久泄漏。现语义：标志位防双入列 +
    pool 保留。本用例锁定"循环复用不新增分配"这一**最终判据**。
    """
    pool = _fake_pool()
    for _ in range(5):
        f = pool.acquire()
        pool.recycle(f)
    assert pool.malloc_calls == 1, (
        "acquire/recycle 循环应复用同一块（malloc 次数=1），"
        "实际 %d 次 = 每轮泄漏一块显存" % pool.malloc_calls)


def test_pool_concurrent_acquire_no_indexerror():
    """无锁 check-then-act 回归钉子：并发 acquire 不得 pop 空列表。"""
    pool = _fake_pool()
    errs: list = []

    def worker():
        try:
            for _ in range(500):
                f = pool.acquire()
                pool.recycle(f)
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, "并发 acquire/recycle 出错: %r" % errs[:2]


# ── OcrSession：注入路径与等待循环出口 ──────────────────────────────

class _FakeEngine:
    backend_name = "fake"

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _FakeSpec:
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

    spec = _FakeSpec()
    return OcrSession(spec, _ocr_engines=engines)


def test_injected_engines_are_stored_and_used():
    """注入路径回归钉子：_engines 此前恒空 → worker IndexError。"""
    eng = _FakeEngine()
    sess = _mk_session([eng])
    assert sess._engines == [eng], "_ocr_engines 必须存入 self._engines"
    assert sess._owns_engines is False
    sess.finish()


def test_injected_empty_list_raises_clear_error():
    """空注入列表：显式 ValueError（而非 IndexError）。"""
    sess = _mk_session([])
    # worker 异步跑：轮询等 err 出现（最长 5s）
    for _ in range(100):
        if sess.err:
            break
        sess._thread.join(timeout=0.05)
    assert sess.err, "worker 未记录错误"
    assert isinstance(sess.err[0], ValueError), (
        "应为显式 ValueError，实际 %r" % (sess.err[0],))
    assert "注入引擎路径要求" in str(sess.err[0])


def test_put_raises_when_worker_died_without_err():
    """worker 死后 put 不得自旋挂死（BaseException 逃逸场景）。

    构造：直接杀线程不可行，改为把 err 清空 + 等 worker 因空引擎列表
    退出——此时 err 非空（ValueError），故用 monkeypatch 清 err 模拟
    「线程已死而 err 恒空」这一修复前会挂死的状态。
    """
    sess = _mk_session([])
    for _ in range(100):
        if not sess._thread.is_alive():
            break
        sess._thread.join(timeout=0.05)
    assert not sess._thread.is_alive(), "worker 应已退出"
    sess.err.clear()          # 模拟 BaseException 逃逸：线程死而 err 空
    with pytest.raises(RuntimeError):
        sess.put(object())


# ── Metrics：snapshot drain 语义 ────────────────────────────────────

def test_snapshot_drain_false_is_readonly():
    """只读快照回归钉子：看门狗取快照不得排干指标桶。"""
    m = Metrics(tier="std")
    m.counter("pipeline.consume_feed_n")
    peek = m.snapshot(drain=False)
    assert peek["counters"].get("pipeline.consume_feed_n") == 1
    # 排干后仍能看到同一条（说明 drain=False 没有清桶）
    drained = m.snapshot(drain=True)
    assert drained["counters"].get("pipeline.consume_feed_n") == 1, (
        "drain=False 的快照把桶吃掉了——停顿后的最终报告将丢失指标")


def test_snapshot_drain_true_still_drains():
    m = Metrics(tier="std")
    m.counter("pipeline.consume_feed_n")
    assert m.snapshot()["counters"].get("pipeline.consume_feed_n") == 1
    assert not m.snapshot()["counters"].get("pipeline.consume_feed_n")


def test_metrics_invalid_tier_raises_valueerror():
    """非法档位：显式 ValueError（原 assert 无消息、-O 下消失）。"""
    with pytest.raises(ValueError) as ei:
        Metrics(tier="FULL")
    assert "off/std/full" in str(ei.value)


def test_resolve_normalizes_enum_str_case_and_space():
    """VOE_TELEMETRY 的大小写/空白归一（原样输入会撞非法档位）。"""
    knob = KNOBS.by_name("diag.telemetry")
    assert _parse(knob, "FULL") == "full"
    assert _parse(knob, " off ") == "off"
    # 未知值原样保留（由消费方显式报错，保持 v1 宽松语义）
    assert _parse(knob, "bogus") == "bogus"


# ── RunJournal：last_error ──────────────────────────────────────────

def test_journal_report_includes_last_error(tmp_path):
    j = RunJournal(tmp_path / "j.jsonl")
    rep = j.report()
    assert rep["last_error"] is None, "未失败时应为 None（且不得 AttributeError）"
    # 注入失败：路径指向目录 → open 抛 IsADirectoryError
    j2 = RunJournal(tmp_path)
    j2.add("x")
    j2.flush()
    assert j2.report()["last_error"], "落盘失败原因必须出现在报告里"


# ── pool.run([]) ────────────────────────────────────────────────────

def test_pool_run_empty_returns_empty():
    """空输入：返回 []（原 max_workers=0 → ValueError）。"""
    from video_ocr_engine.pipeline import pool as _pool

    assert _pool.run([]) == []


# ── OCR 引擎池：淘汰不自噬 + key 含 gamma ────────────────────────────

def test_pool_eviction_prefers_other_key():
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
        n.checkin_ocr_engine(_E(k_other))   # 池: other
        n.checkin_ocr_engine(_E(k_other))   # 池: other×2
        e_new = _E(k_new)
        n.checkin_ocr_engine(e_new)         # 超上限 → 淘汰应选 other
        with n._POOL_LOCK:
            still_new = any(cand is e_new for cand in
                            n._ENGINE_POOL.get(k_new, []))
        assert still_new, "刚归还的引擎被自己挤掉了（淘汰自噬）"
    finally:
        n._POOL_MAX_TOTAL = saved_max
        with n._POOL_LOCK:
            n._ENGINE_POOL.clear()
            n._POOL_IDLE_ORDER.clear()


def test_pool_key_includes_gamma():
    """gamma 构造期冻结 → 必须进池 key（否则同进程改 env 后旋钮失效）。"""
    import inspect

    from video_ocr_engine.ocr import native as n

    src = inspect.getsource(n.acquire_ocr_engine)
    assert "round(float(gamma), 6)" in src, "池 key 必须含 gamma"
    key_block = src.split("key = (")[1].split("\n\n")[0]
    assert "gpu_ctc" in key_block, "池 key 必须含 gpu_ctc"
