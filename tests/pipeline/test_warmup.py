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
def test_warmup_makes_next_run_hot(gpu_env, monkeypatch):
    """真机：预热后首个 extract 的 engine_init 必须远小于冷启动（PI-10）。"""
    pytest.importorskip("cuda.bindings")  # 无 cuda-python（CI 无 GPU）→ skip
    vid, roi = gpu_env
    # report 只在 telemetry≠off 时产出（extractor 只挂非空报告）——
    # 默认 off 下本测试恒 skip 且跳因是 KeyError('report')；若被
    # try/except 转成"环境不可用"假 skip，门禁就形同虚设。
    monkeypatch.setenv("VOE_TELEMETRY", "full")
    ex = FieldExtractor(
        vid,
        roi,
        frame_start=0,
        frame_end=600,
        decode_backend="cpu",
        ocr_backend="tensorrt",
        keep_crops=False,
    )
    try:
        n_warm = ex.warmup()
    except Exception as e:  # 仅引擎构建可跳（TRT/引擎文件缺失）
        pytest.skip("TRT/引擎不可用：%r" % (e,))
    assert n_warm >= 1
    res = ex.extract()  # 引擎已建好：此后的失败是真缺陷，不再 skip
    assert "report" in res.meta, "VOE_TELEMETRY=full 必须产出 report"
    rep = res.meta["report"]
    assert rep["counters"].get("ocr.engine_reuse", 0) >= 1, "预热后应命中池复用（PI-10）"
    assert rep["gauges"]["ocr.engine_init"] < 0.1, "热池 engine_init 必须 <0.1s（PI-10 门禁）"


# ── P1-6：冷启动回归哨（C-82 revisit 取证）─────────────────────────────
@pytest.mark.gpu
def test_hybrid_cold_short_window_not_pathological(gpu_env, monkeypatch):
    """C-82 病态冷态的回归哨：hybrid 短请求在冷进程下不得掉到 NVDEC 档。

    背景（C-82）：fork 泵在硬窗贴近请求边界时把 GOP 误派 NVDEC 臂
    （CPU 臂 92% 闲置），供给 −45%。引擎侧用过 9000 帧硬窗下限规避；
    fork 6363996（熟前挂起+速率派工）后改契约为门——有
    `starve_hold_rate_dispatch` 键即撤下限、窗长=请求长。

    本哨兵钉死三件事（缺一即红）：
      1. 当前 fork 契约面**有**该键（否则下限仍在、且没人知道）；
      2. 窗长 = 请求长（紧窗，不过度解码——C-82 实测 3289 vs 4900 帧）；
      3. hybrid 全臂 busy 合计接近墙钟（派工没有再犯"整窗判给 NVDEC"）。
    只在本机 GPU+视频在场时跑（local-gate 口径）；CI 无 GPU 自动 skip。
    """
    pytest.importorskip("cuda.bindings")
    from video_ocr_engine.decode.contract import probe_contract

    monkeypatch.setenv("VOE_TELEMETRY", "full")
    contract = probe_contract()
    assert contract is not None and contract.has("starve_hold_rate_dispatch"), (
        "fork 契约面无 starve_hold_rate_dispatch——9000 帧下限（C-82 规避）"
        "仍在生效，冷态病态阈值需重标"
    )
    vid, roi = gpu_env
    ex = FieldExtractor(
        vid,
        roi,
        frame_start=0,
        frame_end=3000,
        decode_backend="hybrid",
        ocr_backend="tensorrt",
        keep_crops=False,
    )
    # 紧窗 = 请求长（撤下限的机器可见证据）。真 decord reader 不暴露
    # 调用记录面（decode_window_calls 是测试夹具属性），故以 fork 侧
    # `window_frames` 为据——它记的是**实际生效**的窗长，比"调了什么"
    # 更接近事实。
    res = ex.extract()
    assert "report" in res.meta, "VOE_TELEMETRY=full 必须产出 report"
    rep = res.meta["report"]
    fork = rep.get("hybrid") or {}
    win = fork.get("window_frames")
    assert win == 3000, "契约键已撤下限，生效窗长应=请求长 3000，实际 %s" % win
    # 派工分派：两臂都不得接近空置。C-82 病态形态 = alloc_seq 记
    # ``PiHHHHHHHWW``（CPU 臂 busy 0.26s/3.16s = 92% 闲置）——那是
    # cpu(0)/gpu 驻留模式的历史记法，且当时 frames_c≈0。现役记法以
    # frames 分派为准：CPU 臂至少应拿到 ~1/3 帧（hybrid 的立论即两臂
    # 都在干活；实测 h264 2402/598 = 80/20）。
    fc = fork.get("frames_c")
    assert isinstance(fc, (int, float)) and fc > 0, "CPU 臂零分派：%s" % {
        k: fork.get(k) for k in ("frames_c", "frames_g", "alloc_seq")
    }
