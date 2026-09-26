"""统一驱动骨架的回退/清理分支单测（P1；无需视频/GPU）。

C10 回退（GPU 校准形状不符 → 宿主）与校准异常清理在 P1 前无任何
测试覆盖（golden/e2e 只走主路径）；本文件用桩 spec/vr/session 直接
驱动 _driver.run_segment_pipeline 的两个分支，锚定资源语义：
fallback 分支 **reader 不 close**（宿主路径复用，fallback_vr 移交）、
会话收尾、设备释放；abort 分支全量清理后原异常上抛。
"""
from __future__ import annotations

import pytest

from video_ocr_engine.domain.metrics import NULL_METRICS
from video_ocr_engine.pipeline._driver import run_segment_pipeline
from video_ocr_engine.pipeline.gpu_backend import GpuRunResult, _GpuLane


class _FakeVR:
    def __init__(self):
        self.closed = False

    def __len__(self):
        return 100

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self):
        self.finished = 0
        self.results = {}
        self.err = []
        self.wall = [0.0]
        self.put = lambda task: None
        self.raw_ready = [False]

    def finish(self):
        self.finished += 1


def _spec(vr, session):
    class _S:
        frame_start = 0
        frame_end = 10
        sample_stride = 1
        C = 12.0
        yuv_output = False        # _GpuLane.__init__ 读取
        color_range = 0
        bin_thresh_ref = [0]      # after_calibrate 读写（F-4 盒）
        on_bin_thresh = None
        fps_box = [None]
        metrics = NULL_METRICS
        prof_end = None
        on_bin_thresh = None
        progress = staticmethod(lambda m, p: None)
        cancel = staticmethod(lambda: None)
        backend_label = staticmethod(lambda: "decord/CPU")
        open_vr = staticmethod(lambda: vr)
        start_ocr_session = staticmethod(lambda engines: session)

    return _S()


class _StubLane(_GpuLane):
    """免设备侧装配的桩 lane：只保留生命周期挂钩的真实实现。"""

    def __init__(self, spec):
        super().__init__(spec)
        self.releases = []
        self._release = lambda ctx: self.releases.append(ctx)
        self._ctx = None

    def after_open(self, vr) -> None:
        pass   # 免 gpu.device 装配（无 GPU 环境可跑）

    def calibrate(self, spec, vr, frames, ocr_session):
        return False, 0    # 强制走 C10 回退分支


class _RaisingLane(_StubLane):
    def calibrate(self, spec, vr, frames, ocr_session):
        raise RuntimeError("形状不符模拟")


def test_fallback_branch_preserves_reader_and_releases_device():
    """C10：校准 False → fell_back_to_host / fallback_vr 移交 / 会话
    finish / 设备释放；reader **不 close**（宿主路径复用）。"""
    vr = _FakeVR()
    session = _FakeSession()
    spec = _spec(vr, session)
    lane = _StubLane(spec)
    res = run_segment_pipeline(spec, GpuRunResult(), lane, None, None)
    assert res.fell_back_to_host is True
    assert res.fallback_vr is vr
    assert vr.closed is False        # reader 移交宿主路径，不 close
    assert session.finished == 1
    assert len(lane.releases) == 1   # 设备侧临时缓冲已释放


def test_calibrate_exception_cleans_up_and_reraises():
    """校准异常 → 会话 finish + 设备释放 + reader close + 原异常上抛。"""
    vr = _FakeVR()
    session = _FakeSession()
    spec = _spec(vr, session)
    lane = _RaisingLane(spec)
    with pytest.raises(RuntimeError, match="形状不符模拟"):
        run_segment_pipeline(spec, GpuRunResult(), lane, None, None)
    assert vr.closed is True
    assert session.finished == 1
    assert len(lane.releases) == 1


# ═════════════ P3：中途失败语义（流中断 / cancel / OCR worker 死）═════════════

class _OKLane(_StubLane):
    """全成功路径桩：校准 (True, th)、帧流可注入行为、emit/similar 空转。"""

    def __init__(self, spec, *, n_items=5, stream_err=None):
        super().__init__(spec)
        self.n_items = n_items
        self.stream_err = stream_err

    def calibrate(self, spec, vr, frames, ocr_session):
        return True, 128

    def after_calibrate(self, th, ocr_session) -> None:
        # 桩不做设备侧装配（y_pool/autocropper）；只留驱动契约要求的接线
        self._session = ocr_session
        self._put_ocr = ocr_session.put

    def after_stream(self) -> None:
        pass   # 桩无生产者线程

    def stop_consume(self) -> None:
        pass

    def frame_items(self, spec, vr, frames, ocr_session):
        for i in range(self.n_items):
            yield i, 1.0, dict(payload=(i, None, 1.0), cluster=0.0)
        if self.stream_err is not None:
            raise self.stream_err

    def emit(self, seg, rep, frac) -> None:
        pass


def test_stream_exception_cleans_up_and_reraises():
    """帧流中途异常 → reader close + 会话 finish + 设备释放 + 异常原样上抛。"""
    vr = _FakeVR()
    session = _FakeSession()
    spec = _spec(vr, session)
    lane = _OKLane(spec, n_items=3, stream_err=RuntimeError("流中断模拟"))
    with pytest.raises(RuntimeError, match="流中断模拟"):
        run_segment_pipeline(spec, GpuRunResult(), lane, None, None)
    assert vr.closed is True
    assert session.finished == 1
    assert len(lane.releases) == 1


def test_cancel_mid_stream_cleans_up():
    """cancel 在 k%100==0 节拍触发 → 清理完整 + 取消异常上抛（不吞）。"""
    vr = _FakeVR()
    session = _FakeSession()
    spec = _spec(vr, session)
    calls = {"n": 0}

    def _cancel():
        calls["n"] += 1
        raise KeyboardInterrupt("用户取消模拟")

    spec.cancel = staticmethod(_cancel)
    lane = _OKLane(spec, n_items=150)   # k=100 处触发 on_cancel
    with pytest.raises(KeyboardInterrupt):
        run_segment_pipeline(spec, GpuRunResult(), lane, None, None)
    assert calls["n"] >= 1
    assert vr.closed is True
    assert session.finished == 1


def test_ocr_worker_failure_raises_with_context():
    """OCR worker 中途死（err 非空）→ 清理完成后 RuntimeError 携原始异常链。"""
    vr = _FakeVR()
    session = _FakeSession()
    boom = RuntimeError("engine boom")
    session.err.append(boom)
    spec = _spec(vr, session)
    lane = _OKLane(spec, n_items=4)
    with pytest.raises(RuntimeError, match="OCR worker 失败") as ei:
        run_segment_pipeline(spec, GpuRunResult(), lane, None, None)
    assert ei.value.__cause__ is boom
    assert vr.closed is True
    assert session.finished == 1
