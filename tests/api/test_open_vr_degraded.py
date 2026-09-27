"""_open_vr 降级路径表驱动测试（P2，2026-09-20 稳健性轮）。

7 条 ``_degraded`` 路径此前零门面层断言（C10 回退/abort 已由
tests/pipeline/test_driver_fallback.py 覆盖驱动层）。本文件用假 decord
模块（注入 sys.modules，CI 无 decord 也可跑）逐条驱动：

NVDEC 打开失败→CPU / hybrid 打开失败→纯 GPU / codec 探测失败（CPU、
GPU 两分支）/ yuv420 不支持→gray（含标志重置）/ color_range 读取失败
→limited / GPU 形状不符→宿主管线（门面接线）。
"""
from __future__ import annotations

import sys
import types

import pytest

from video_ocr_engine import FieldExtractor

BEHAVIOR = {
    "gpu_open_fail": False,
    "hybrid_open_fail": False,
    "yuv420_fail": False,
    "codec_fail": False,
    "cr_fail": False,
}


class _Ctx:
    def __init__(self, kind: str) -> None:
        self.kind = kind


class _FakeVR:
    """按 BEHAVIOR 行事的假 reader；接口面 = _open_vr 实际触碰的子集。"""

    def __init__(self, path, ctx=_Ctx("cpu"), **kw):
        if ctx.kind == "gpu" and BEHAVIOR["gpu_open_fail"]:
            raise RuntimeError("NVDEC 打开失败模拟")
        if ctx.kind in ("hybrid", "hybrid_gpu") and BEHAVIOR["hybrid_open_fail"]:
            raise RuntimeError("hybrid ctor 失败模拟")
        if kw.get("output_format") == "yuv420" and BEHAVIOR["yuv420_fail"]:
            # 旧 fork DLL 的报错形态（extractor 以 ValueError 识别回退）
            raise ValueError("output_format must be 'rgb', 'gray' or 'yuv420'")
        self.ctx = ctx
        self._n = 50000
        self.decode_window_calls: list[int] = []   # C-57 谓词回归观测用

    def __len__(self):
        return self._n

    def get_codec(self):
        if BEHAVIOR["codec_fail"]:
            raise RuntimeError("codec 探测失败模拟")
        return "h264"

    def get_color_range(self):
        if BEHAVIOR["cr_fail"]:
            raise RuntimeError("color_range 读取失败模拟")
        return 1

    def set_decode_window(self, n: int) -> None:
        self.decode_window_calls.append(n)

    def close(self):
        pass


@pytest.fixture
def fake_decord(monkeypatch):
    """注入假 decord（含 video_reader 子模块：无 _CAPI_VideoReaderSetRoi
    → roi_kw 为空、走无 ROI API 分支）。"""
    for k in BEHAVIOR:
        BEHAVIOR[k] = False
    mod = types.ModuleType("decord")
    mod.cpu = lambda i=0: _Ctx("cpu")
    mod.gpu = lambda i=0: _Ctx("gpu")
    mod.hybrid = lambda i=0: _Ctx("hybrid")
    mod.hybrid_gpu = lambda i=0: _Ctx("hybrid_gpu")
    mod.VideoReader = _FakeVR
    mod.video_reader = types.ModuleType("decord.video_reader")
    # ROI-first API 必须在（无它的 decord 是硬错误不是降级路径——
    # ROI-first 是引擎地基）；_open_vr 只做 hasattr 探测，不真调用
    mod.video_reader._CAPI_VideoReaderSetRoi = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "decord", mod)
    monkeypatch.setitem(sys.modules, "decord.video_reader", mod.video_reader)
    return mod


def _make(**kw):
    d = dict(decode_backend="auto", ocr_backend="cpu", keep_crops=False,
             frame_end=500)
    d.update(kw)
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **d)


def _degraded(ex):
    return " | ".join(ex._degraded)


def test_nvdec_open_fail_falls_back_cpu(fake_decord):
    BEHAVIOR["gpu_open_fail"] = True
    ex = _make(decode_backend="auto")
    vr = ex._open_vr()
    assert ex._backend == "decord/CPU" and vr is not None
    assert "NVDEC 打开失败" in _degraded(ex)


def test_hybrid_backend_with_nvdec_fail_also_falls_cpu(fake_decord):
    # hybrid 分支前置要求 GPU reader 打开成功；NVDEC 失败时与 auto 同路
    BEHAVIOR["gpu_open_fail"] = True
    ex = _make(decode_backend="hybrid")
    ex._open_vr()
    assert ex._backend == "decord/CPU"
    assert "NVDEC 打开失败" in _degraded(ex)


def test_hybrid_open_fail_falls_back_pure_gpu(fake_decord):
    BEHAVIOR["hybrid_open_fail"] = True
    ex = _make(decode_backend="hybrid")
    vr = ex._open_vr()
    assert ex._backend == "decord/GPU" and vr is not None
    assert "hybrid 打开失败，回退纯 GPU" in _degraded(ex)


def test_codec_probe_fail_cpu_and_gpu_branches(fake_decord):
    BEHAVIOR["codec_fail"] = True
    ex_cpu = _make(decode_backend="cpu")
    ex_cpu._open_vr()
    assert ex_cpu._backend == "decord/CPU"
    assert "codec 探测失败" in _degraded(ex_cpu)
    ex_gpu = _make(decode_backend="nvdec")
    ex_gpu._open_vr()
    assert ex_gpu._backend == "decord/GPU"
    assert "codec 探测失败" in _degraded(ex_gpu)


def test_yuv420_unsupported_falls_back_gray(fake_decord):
    BEHAVIOR["yuv420_fail"] = True
    ex = _make(keep_crops=True, rep_crop_format="yuv")
    assert ex._yuv_output is True      # 前置：构造期 yuv 已置位
    vr = ex._open_vr()
    assert ex._yuv_output is False     # 回退后重置
    assert ex._color_range == 0
    assert vr is not None
    assert "不支持 yuv420" in _degraded(ex)


def test_color_range_read_fail_defaults_limited(fake_decord):
    BEHAVIOR["cr_fail"] = True
    ex = _make(keep_crops=True, rep_crop_format="yuv")
    ex._open_vr()
    assert ex._color_range == 0
    assert "color_range 读取失败" in _degraded(ex)


def test_gpu_shape_fallback_delegates_to_host(fake_decord, monkeypatch):
    """C10 引擎接线（R1）：run_gpu_pipeline 报形状不符 → fallback_vr/
    engines 移交宿主 lane，降级透出（patch 面随 R1 从门面方法改到
    pipeline.engine 的 lane 入口；GPU 门控经门面 patch 强制通过）。"""
    import video_ocr_engine.pipeline.engine as eng_mod
    from video_ocr_engine.pipeline.gpu_backend import GpuRunResult
    from video_ocr_engine.pipeline.host_backend import HostRunResult

    class _Res(GpuRunResult):
        pass

    res = _Res(fell_back_to_host=True, fallback_engines=["eng"],
               fallback_vr=object(), fps=30.0)
    monkeypatch.setattr(eng_mod, "run_gpu_pipeline",
                        lambda spec, engines: res)
    captured = {}

    def fake_host(spec, engines, preopened_vr=None):
        captured["engines"] = engines
        captured["vr"] = preopened_vr
        return HostRunResult(fps=30.0)

    monkeypatch.setattr(eng_mod, "run_host_pipeline", fake_host)
    monkeypatch.setattr(FieldExtractor, "_gpu_pipeline_enabled",
                        lambda self: True)
    ex = _make(decode_backend="nvdec", ocr_backend="tensorrt")
    outcome = ex._run_pipelined(None)
    assert captured["engines"] == ["eng"]
    assert captured["vr"] is res.fallback_vr
    assert "GPU 管线形状不符" in _degraded(ex)
    # 门面状态同步（R1 收口到 _run_pipelined 单处）：fps 经 outcome 回写
    assert ex._fps == 30.0 and outcome.fps == 30.0


# ═════════ C-57 谓词防线回归：晚起点 × 硬窗不设窗（fork 缺陷唯一防线）═════════

def test_hard_window_set_when_start_before_window(fake_decord):
    """早起点（start < 窗长）：谓词成立 → set_decode_window(窗长) 被调。"""
    ex = _make(decode_backend="hybrid", frame_start=0, frame_end=3000)
    vr = ex._open_vr()
    assert ex._backend == "decord/hybrid"
    assert vr.decode_window_calls == [3000]


def test_hard_window_skipped_for_late_start(fake_decord):
    """晚起点（start ≥ 窗长）：谓词不成立 → **不设窗**。

    C-57：fork 的硬窗 × seek(start>0) 存在尾帧 EOF 静默替补缺陷（帧数
    守恒但尾 ~20 帧像素错），引擎谓词 `start < 窗长` 是唯一防线——
    本测试锚定该防线不被误删（2026-09-20 审计修复轮 §4）。
    """
    ex = _make(decode_backend="hybrid", frame_start=5000, frame_end=6000)
    vr = ex._open_vr()
    assert ex._backend == "decord/hybrid"
    assert vr.decode_window_calls == []
