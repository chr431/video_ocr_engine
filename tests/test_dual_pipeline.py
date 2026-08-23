"""单实例双完整流水线并行的单元测试（无需视频/GPU）。

只覆盖构造/参数/后端组合/分发逻辑；真实解码与 OCR 由集成冒烟负责。
"""
from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor


def _make(**kwargs):
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **kwargs)


def test_dual_pipeline_default_off():
    ex = _make()
    assert ex._dual_pipeline is False
    assert ex._dual_pipeline_chunks == 0
    assert ex._dual_backends is None


def test_dual_pipeline_env_enabled(monkeypatch):
    monkeypatch.setenv("RVTOL_DUAL_PIPELINE", "1")
    ex = _make()
    assert ex._dual_pipeline is True


def test_dual_pipeline_explicit_override(monkeypatch):
    monkeypatch.setenv("RVTOL_DUAL_PIPELINE", "1")
    ex = _make(dual_pipeline=False)
    assert ex._dual_pipeline is False
    ex2 = _make(dual_pipeline=True, dual_pipeline_chunks=6)
    assert ex2._dual_pipeline is True
    assert ex2._dual_pipeline_chunks == 6


def test_dual_backend_pairs_default_opposite():
    # 互补 OCR 保持 TRT（TRT⊕ONNX 共存推理互相膨胀，实测净负），
    # 仅解码侧互补：auto ∥ cpu。
    ex = _make(decode_backend="auto", ocr_backend="auto")
    assert ex._dual_backend_pairs() == [("auto", "auto"), ("cpu", "auto")]
    ex2 = _make(decode_backend="cpu", ocr_backend="cpu")
    assert ex2._dual_backend_pairs() == [("cpu", "cpu"), ("auto", "auto")]
    ex3 = _make(decode_backend="nvdec", ocr_backend="tensorrt")
    assert ex3._dual_backend_pairs() == [("nvdec", "tensorrt"),
                                         ("cpu", "tensorrt")]


def test_dual_backends_custom_and_duplicate_single():
    ex = _make(dual_backends=[("cpu", "auto"), ("auto", "cpu")])
    assert ex._dual_backend_pairs() == [("cpu", "auto"), ("auto", "cpu")]
    ex2 = _make(dual_backends=[("cpu", "auto")])
    assert ex2._dual_backend_pairs() == [("cpu", "auto"), ("cpu", "auto")]
    ex3 = _make(dual_backends=[("cpu", "auto")], decode_backend="auto")
    assert ex3._dual_backend_pairs() == [("cpu", "auto"), ("cpu", "auto")]


def test_extract_dispatches_to_parallel_when_enabled(monkeypatch):
    ex = _make(dual_pipeline=True)

    def fake_parallel(self):
        self.crops = {1: "crop1", 2: "crop2"}
        return ([0, 1, 2], [[0, 1], [2]], ["a", "b"], [0.9, 0.8], [1, 2])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined_parallel", fake_parallel)
    result = ex.extract()
    assert result.frames == [0, 1, 2]
    assert result.segments[0].frames == (0, 1)
    assert result.segments[0].rep_crop == "crop1"
    assert result.meta["ocr_backend"] == ""


def test_extract_does_not_dispatch_when_disabled(monkeypatch):
    ex = _make(dual_pipeline=False)

    def fake_parallel(self):
        raise AssertionError("不应进入并行路径")

    monkeypatch.setattr(FieldExtractor, "_run_pipelined_parallel", fake_parallel)

    def fake_run(self):
        self.crops = {}
        return ([0], [[0]], ["x"], [0.5], [0])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined", fake_run)
    result = ex.extract()
    assert result.segments[0].text == "x"


# ═══════════════ 慢路径让位（adaptive yield） ═══════════════

def test_dual_should_yield_basic():
    f = FieldExtractor._dual_should_yield
    # 显著落后且队列有剩余 → 让位
    assert f(100.0, 200.0, 0.6, 1) is True
    # 差距不足（≥ ratio）→ 不让位
    assert f(150.0, 200.0, 0.6, 1) is False
    # 快路径领先 → 不让位
    assert f(300.0, 200.0, 0.6, 1) is False
    # 让位后无剩余片 → 不让位（必须有人做完）
    assert f(10.0, 1000.0, 0.6, 0) is False


def test_dual_should_yield_disabled_or_invalid():
    f = FieldExtractor._dual_should_yield
    # ratio=0 禁用；负值视为禁用
    assert f(1.0, 100.0, 0.0, 5) is False
    assert f(1.0, 100.0, -0.5, 5) is False
    # 无效吞吐（尚未完成任何片）
    assert f(0.0, 100.0, 0.6, 5) is False
    assert f(100.0, 0.0, 0.6, 5) is False


# ═══════════════ 双流水线 OCR 线程分核预算 ═══════════════

def test_dual_ocr_threads_trt_fixed_cpu_split():
    import engine_config as config
    from ocr_native import auto_ocr_thread_count
    ex = _make(dual_pipeline=True)
    trt_budget = config.DUAL_PIPELINE_TRT_CPU_THREADS
    # TRT 侧：固定小预算（推理在 GPU，多线程无收益）
    assert ex._dual_ocr_num_threads("tensorrt", 1) == trt_budget
    assert ex._dual_ocr_num_threads("auto", 1) == trt_budget
    # ONNX 侧：(物理核 - TRT 预算) // CPU 侧消费者数，下限 2
    cores = auto_ocr_thread_count()
    assert ex._dual_ocr_num_threads("onnxruntime", 1) == \
        max(2, (cores - trt_budget) // 1)
    assert ex._dual_ocr_num_threads("onnxruntime", 2) == \
        max(2, (cores - trt_budget) // 2)


def test_dual_ocr_threads_onnx_capped_with_trt_peer():
    """混配保护：另一条流水线跑 TRT 时 ONNX 侧封顶（防饥饿 TRT 宿主线程）。"""
    import engine_config as config
    from ocr_native import cpu_physical_cores
    ex = _make(dual_pipeline=True)
    cap = config.DUAL_PIPELINE_ONNX_PEER_THREADS
    got = ex._dual_ocr_num_threads("onnxruntime", 1, has_trt_peer=True)
    assert got == min(max(2, (cpu_physical_cores()
                              - config.DUAL_PIPELINE_TRT_CPU_THREADS)), cap)
    # 无 TRT 对端 → 不封顶
    cores = __import__("ocr_native").auto_ocr_thread_count()
    assert ex._dual_ocr_num_threads("onnxruntime", 1, has_trt_peer=False) \
        == max(2, cores - config.DUAL_PIPELINE_TRT_CPU_THREADS)


def test_dual_ocr_threads_env_override(monkeypatch):
    monkeypatch.setenv("RVTOL_OCR_THREADS", "7")
    ex = _make(dual_pipeline=True)
    assert ex._dual_ocr_num_threads("tensorrt", 1) == 7
    assert ex._dual_ocr_num_threads("onnxruntime", 1) == 7
