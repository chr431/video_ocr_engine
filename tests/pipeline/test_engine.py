"""S3-3d 骨架测试：端口协议 + SegmentEngine 唯一编排 + RunOutcome。"""
from __future__ import annotations

from video_ocr_engine.decode.port import FrameSource
from video_ocr_engine.ocr.port import OcrBackend
from video_ocr_engine.pipeline import RunOutcome, SegmentEngine


class _FakeSource:  # 结构化满足 FrameSource
    def __len__(self):
        return 0
    @property
    def fps(self):
        return 30.0
    @property
    def codec(self):
        return "h264"
    @property
    def color_range(self):
        return 0
    def roi_format(self):
        return "gray"
    def batch(self, frames):
        return None
    def close(self):
        return None


class _FakeOcr:
    backend_name = "fake"
    max_batch = 4
    supports_device_input = False
    def call(self, images):
        return []
    def release(self):
        return None


def test_protocols_are_structural():
    assert isinstance(_FakeSource(), FrameSource)
    assert isinstance(_FakeOcr(), OcrBackend)


def test_runoutcome_tuple_shape_matches_v1():
    o = RunOutcome([1], [[0, 1]], ["t"], [0.5], [0])
    frames, segs, texts, confs, rep = o.as_tuple()
    assert (frames, segs, texts, confs, rep) == ([1], [[0, 1]], ["t"], [0.5], [0])
    assert o.report is None


class _FakeEx:
    """门面桩：记录引擎选择与引擎参数透传。"""

    def __init__(self, gpu_enabled, outcome):
        self._gpu_enabled = gpu_enabled
        self._outcome = outcome
        self.calls = []

    def _gpu_pipeline_enabled(self):
        return self._gpu_enabled

    def _run_pipelined_gpu(self, engines):
        self.calls.append(("gpu", engines))
        return self._outcome

    def _run_pipelined_host(self, engines, vr=None):
        self.calls.append(("host", engines))
        return self._outcome


_OUT = (["f"], [["s"]], ["t"], [0.9], [3])


def test_engine_dispatches_gpu_when_enabled():
    ex = _FakeEx(gpu_enabled=True, outcome=_OUT)
    out = SegmentEngine(ex).run(["E"]).as_tuple()
    assert ex.calls == [("gpu", ["E"])]
    assert out == _OUT


def test_engine_dispatches_host_when_disabled():
    ex = _FakeEx(gpu_enabled=False, outcome=_OUT)
    SegmentEngine(ex).run().as_tuple()
    assert ex.calls == [("host", None)]
