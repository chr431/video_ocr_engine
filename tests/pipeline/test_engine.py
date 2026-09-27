"""S3-3d/R1 骨架测试：端口协议 + SegmentEngine 注入式编排 + RunOutcome。

R1（2026-09-27 门面拆解轮）起引擎吃 `EngineInputs` 显式契约（不再
duck-type 门面）；分发断言改为 patch engine 模块的两条 lane 入口
（run_gpu_pipeline / run_host_pipeline），语义与旧门面桩一致：
GPU 门控通过 → gpu lane；否则 host lane；回退透传见
tests/api/test_open_vr_degraded.py::test_gpu_shape_fallback_delegates_to_host。
"""
from __future__ import annotations

import video_ocr_engine.pipeline.engine as eng_mod
from video_ocr_engine.decode.port import FrameSource
from video_ocr_engine.ocr.port import OcrBackend
from video_ocr_engine.pipeline import EngineInputs, RunOutcome, SegmentEngine
from video_ocr_engine.pipeline.gpu_backend import GpuRunResult
from video_ocr_engine.pipeline.host_backend import HostRunResult


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


def _inputs(**over):
    """最小合法 EngineInputs（lane 入口被 patch，回调全是轻桩）。"""
    base = dict(
        frame_start=0, frame_end=None, sample_stride=1, roi=(0, 0, 1, 1),
        buffer_size=4, C=3.0, merge_similar=True,
        merge_similar_threshold=3.0, merge_max_changed_pixels=32,
        merge_dense_gate=5, keep_crops=False, yuv_output=False,
        color_range=0, ocr_autocrop=True,
        gpu_pipeline=False,
        segments_similar=lambda a, b: False,
        crop_luma=lambda c: c,
        batch_luma=lambda c: c,
        batch_luma_out=lambda c, o: o,
        crop_is_expected=lambda c, h, w: True,
        content_range_to_crop=lambda f, l, w: None,
        open_vr=lambda: None,
        start_ocr_session=lambda e=None: None,
    )
    base.update(over)
    return EngineInputs(**base)


def test_engine_dispatches_gpu_when_enabled(monkeypatch):
    calls = []

    def fake_gpu(spec, engines):
        calls.append(("gpu", engines))
        return GpuRunResult()

    monkeypatch.setattr(eng_mod, "run_gpu_pipeline", fake_gpu)
    out = SegmentEngine(_inputs(gpu_pipeline=True)).run(["E"])
    assert calls == [("gpu", ["E"])]
    assert out.as_tuple() == ([], [], [], [], [])


def test_engine_dispatches_host_when_disabled(monkeypatch):
    calls = []

    def fake_host(spec, engines, preopened_vr=None):
        calls.append(("host", engines, preopened_vr))
        return HostRunResult()

    monkeypatch.setattr(eng_mod, "run_host_pipeline", fake_host)
    out = SegmentEngine(_inputs(gpu_pipeline=False)).run()
    assert calls == [("host", None, None)]
    assert out.as_tuple() == ([], [], [], [], [])


def test_runoutcome_from_result_maps_lane_fields():
    res = HostRunResult(frames=[7], segs=[[0, 3]], texts=["t"], confs=[0.5],
                        rep_frames=[7], fps=30.0, bin_thresh=128,
                        n_segments=1, fork_stats={"busy": 1})
    o = RunOutcome.from_result(res)
    assert o.as_tuple() == ([7], [[0, 3]], ["t"], [0.5], [7])
    assert (o.fps, o.bin_thresh, o.n_segments, o.fork_stats,
            o.timing, o.crops) == (30.0, 128, 1, {"busy": 1}, {}, {})
