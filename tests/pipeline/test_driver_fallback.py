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
        fps_box = [None]
        metrics = NULL_METRICS
        prof_end = None
        on_bin_thresh = None
        progress = staticmethod(lambda m, p: None)
        cancel = staticmethod(lambda: None)
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
