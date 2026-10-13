"""显式预热与 PI-10 池 key 稳定性。

无需视频/GPU 的部分用假引擎验证 key 同源；真机部分（`-m gpu`）验证
"预热后首个 extract 命中热池"这一可观测承诺。
"""

from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor
from video_ocr_engine.pipeline import ocr_stage
from video_ocr_engine.pipeline.ocr_stage import SessionSpec, acquire_engines, warmup_engines


def _spec(**kw) -> SessionSpec:
    # num_threads=2 → 低于 OCR_INSTANCES_MIN_THREADS，走单实例路径（断言 1 个引擎）
    base = dict(
        buffer_size=128,
        model="v6_small",
        fill_width=None,
        force_aspect=0.0,
        reorder_window=64,
        yuv_output=False,
        color_range=0,
        gpu_pipeline_mode=False,
        num_threads_fn=lambda: 2,
        engine_type_fn=lambda: "openvino",
        crop_to_content=lambda c: c,
        crop_after_aspect=lambda p: p,
    )
    base.update(kw)
    return SessionSpec(**base)


def test_acquire_and_warmup_share_one_key(monkeypatch):
    """预热与 extract 必须走同一取引擎出处（否则 PI-10 复用失效）。"""
    seen: list = []

    class _Eng:
        backend_name = "openvino"
        _trt = None

        def release(self):
            seen.append("release")

    def _fake_acquire(variant, engine_type, **kw):
        seen.append((variant, engine_type, kw.get("fill_width"), kw.get("num_threads")))
        return _Eng()

    from video_ocr_engine.ocr import native as _native

    checked: list = []
    monkeypatch.setattr(_native, "acquire_ocr_engine", _fake_acquire)
    monkeypatch.setattr(_native, "checkin_ocr_engine", lambda e: checked.append(e))
    spec = _spec()
    engines, etype = acquire_engines(spec)
    assert etype == "openvino" and len(engines) == 1
    assert warmup_engines(spec) == 1
    assert len(seen) == 2 and seen[0] == seen[1], "预热与取引擎的参数必须逐项相同（池 key 稳定性）"
    assert len(checked) == 1, "预热必须归还引擎（否则占用池位）"


def test_warmup_uses_same_spec_as_run(monkeypatch):
    """FieldExtractor.warmup() 直接复用 _build_session_spec（单一出处）。"""
    ex = FieldExtractor("dummy.mp4", (0, 0, 100, 50), ocr_backend="cpu")
    captured: list = []
    monkeypatch.setattr(ocr_stage, "warmup_engines", lambda spec: captured.append(spec) or 1)
    assert ex.warmup() == 1
    assert len(captured) == 1
    assert captured[0].model == ex._build_session_spec().model


@pytest.mark.gpu
def test_warmup_makes_next_run_hot(monkeypatch):
    """真机：预热后首个 extract 的 engine_init 必须远小于冷启动（PI-10）。"""
    import os

    vid = os.path.join(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"), "test5.mp4")
    if not os.path.exists(vid):
        pytest.skip("无真值视频（RACELOG_VIDEO_DIR）")
    try:
        import cuda.bindings  # noqa: F401  无 CUDA 则跳过
    except ImportError:
        pytest.skip("无 cuda-python（CI 无 GPU 环境）")
    # report 只在 telemetry≠off 时产出（extractor 只挂非空报告）——
    # 默认 off 下本测试恒 skip 且跳因是 KeyError('report')；若被
    # try/except 转成"环境不可用"假 skip，门禁就形同虚设。
    monkeypatch.setenv("VOE_TELEMETRY", "full")
    ex = FieldExtractor(
        vid,
        (843, 993, 948, 1025),
        frame_start=0,
        frame_end=600,
        decode_backend="cpu",
        ocr_backend="tensorrt",
        keep_crops=False,
    )
    try:
        n_warm = ex.warmup()
    except Exception as e:  # noqa: BLE001  仅引擎构建可跳（TRT/引擎文件缺失）
        pytest.skip("TRT/引擎不可用：%r" % (e,))
    assert n_warm >= 1
    res = ex.extract()  # 引擎已建好：此后的失败是真缺陷，不再 skip
    assert "report" in res.meta, "VOE_TELEMETRY=full 必须产出 report"
    rep = res.meta["report"]
    assert rep["counters"].get("ocr.engine_reuse", 0) >= 1, "预热后应命中池复用（PI-10）"
    assert rep["gauges"]["ocr.engine_init"] < 0.1, "热池 engine_init 必须 <0.1s（PI-10 门禁）"
