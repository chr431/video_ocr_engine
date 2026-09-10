"""S3-2 骨架测试：端口协议 + SegmentEngine/RunOutcome + 双跑开关。"""
from __future__ import annotations

from video_ocr_engine.decode.port import FrameSource
from video_ocr_engine.ocr.port import OcrBackend
from video_ocr_engine.pipeline import RunOutcome, SegmentEngine
from video_ocr_engine.pipeline.engine import _LegacyBackend


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


def test_legacy_backend_delegates(monkeypatch):
    import video_ocr_engine.extractor as ex_mod
    calls = {}

    class _FakeEx:
        def _run_pipelined_legacy(self, engines):
            calls["engines"] = engines
            return (["f"], [["s"]], ["t"], [0.9], [3])

    out = SegmentEngine(_LegacyBackend(_FakeEx(), ["E"])).run()
    assert calls["engines"] == ["E"]           # 引擎注入透传不丢
    assert out.as_tuple() == (["f"], [["s"]], ["t"], [0.9], [3])


def test_v2_engine_switch_env(monkeypatch):
    monkeypatch.setenv("VOE_V2_ENGINE", "1")
    assert SegmentEngine.legacy_engine_requested()
    monkeypatch.setenv("VOE_V2_ENGINE", "0")
    assert not SegmentEngine.legacy_engine_requested()
    monkeypatch.delenv("VOE_V2_ENGINE")
    assert not SegmentEngine.legacy_engine_requested()
