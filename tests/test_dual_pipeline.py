"""单实例双完整流水线并行的单元测试（无需视频/GPU）。

只覆盖构造/参数/后端组合/切片（kfe 唯一分片方法）/分发逻辑；
真实解码与 OCR 由集成冒烟负责。
"""
from __future__ import annotations

import pytest

from video_ocr_engine import FieldExtractor


def _make(**kwargs):
    return FieldExtractor("dummy.mp4", (0, 0, 100, 50), **kwargs)


def test_dual_pipeline_default_off():
    ex = _make()
    assert ex._dual_pipeline is False
    assert ex._dual_backends is None


def test_dual_pipeline_env_enabled(monkeypatch):
    monkeypatch.setenv("DUAL_PIPELINE", "1")
    ex = _make()
    assert ex._dual_pipeline is True


def test_dual_pipeline_explicit_override(monkeypatch):
    monkeypatch.setenv("DUAL_PIPELINE", "1")
    ex = _make(dual_pipeline=False)
    assert ex._dual_pipeline is False
    ex2 = _make(dual_pipeline=True)
    assert ex2._dual_pipeline is True


def test_dual_backend_pairs_default_opposite():
    # 互补 OCR 与下游 --dual 一致：TRT ↔ ONNX，解码侧互补 auto ∥ cpu。
    ex = _make(decode_backend="auto", ocr_backend="auto")
    assert ex._dual_backend_pairs() == [("auto", "auto"), ("cpu", "cpu")]
    ex2 = _make(decode_backend="cpu", ocr_backend="cpu")
    assert ex2._dual_backend_pairs() == [("cpu", "cpu"), ("auto", "auto")]
    ex3 = _make(decode_backend="nvdec", ocr_backend="tensorrt")
    assert ex3._dual_backend_pairs() == [("nvdec", "tensorrt"),
                                         ("cpu", "cpu")]


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

    monkeypatch.setattr(
        FieldExtractor, "_run_pipelined_parallel", fake_parallel)
    result = ex.extract()
    assert result.frames == [0, 1, 2]
    assert result.segments[0].frames == (0, 1)
    assert result.segments[0].rep_crop == "crop1"
    assert result.meta["ocr_backend"] == ""


def test_extract_does_not_dispatch_when_disabled(monkeypatch):
    ex = _make(dual_pipeline=False)

    def fake_parallel(self):
        raise AssertionError("不应进入并行路径")

    monkeypatch.setattr(
        FieldExtractor, "_run_pipelined_parallel", fake_parallel)

    def fake_run(self):
        self.crops = {}
        return ([0], [[0]], ["x"], [0.5], [0])

    monkeypatch.setattr(FieldExtractor, "_run_pipelined", fake_run)
    result = ex.extract()
    assert result.segments[0].text == "x"


# ═══════════════ kfe：唯一分片方法（头部试点组 + 关键帧竞争区） ═══════════════

def test_dual_chunk_specs_pilots_plus_keyframe_competition():
    """正常视频：头部 4 试点片 + 关键帧竞争区（kfe），不再等分成固定块。"""
    f = FieldExtractor._dual_chunk_specs
    frames = list(range(0, 1000))
    kf = [200, 400, 600, 800]
    spec, has_pilots = f(frames, kf, last_end=1000, stride=1,
                         min_gap=16, max_chunks=8,
                         unit_div=40, min_chunk=16)
    assert has_pilots is True
    # 头部 4 片 = 试点×2 + 确认×2（unit_div=40 → 每片 25 采样帧）
    assert spec[:4] == [(0, 25), (25, 50), (50, 75), (75, 100)]
    # 竞争区按关键帧边界切分（边界吸附到采样网格）
    assert spec[4:] == [(100, 200), (200, 400), (400, 600),
                        (600, 800), (800, 1000)]
    # 覆盖连续无缝隙
    assert spec[0][0] == 0
    assert all(spec[i][1] == spec[i + 1][0]
               for i in range(len(spec) - 1))
    assert spec[-1][1] == 1000


def test_dual_chunk_specs_no_keyframes_single_big_chunk():
    """无关键帧时竞争区退化为单一大片（kfe 自然退化）。"""
    f = FieldExtractor._dual_chunk_specs
    frames = list(range(0, 1000))
    spec, has_pilots = f(frames, [], last_end=1000, stride=1,
                         min_gap=16, max_chunks=8,
                         unit_div=40, min_chunk=16)
    assert has_pilots is True
    assert spec[:4] == [(0, 25), (25, 50), (50, 75), (75, 100)]
    assert spec[4:] == [(100, 1000)]
    assert spec[-1][1] == 1000


def test_dual_chunk_specs_short_video_no_pilots():
    """短视频放不下头部组：整段进 kfe 竞争区（无关键帧=单一大片）。"""
    f = FieldExtractor._dual_chunk_specs
    frames = list(range(0, 100))
    spec, has_pilots = f(frames, [], last_end=100, stride=1,
                         min_gap=16, max_chunks=8,
                         pilots=4, unit_div=24, min_chunk=30)
    assert has_pilots is False
    assert spec == [(0, 100)]


def test_dual_chunk_specs_short_video_with_keyframes():
    """短视频放不下头部组时仍用 kfe 切竞争区。"""
    f = FieldExtractor._dual_chunk_specs
    frames = list(range(0, 100))
    kf = [30, 60]
    spec, has_pilots = f(frames, kf, last_end=100, stride=1,
                         min_gap=8, max_chunks=8,
                         pilots=4, unit_div=24, min_chunk=30)
    assert has_pilots is False
    assert spec == [(0, 30), (30, 60), (60, 100)]
    assert all(spec[i][1] == spec[i + 1][0]
               for i in range(len(spec) - 1))
    assert spec[-1][1] == 100


def test_dual_chunk_specs_stride_snap():
    """采样步长>1 时竞争区边界吸附到最近采样帧。"""
    f = FieldExtractor._dual_chunk_specs
    frames = list(range(0, 400, 4))
    kf = [110, 210, 310]
    spec, has_pilots = f(frames, kf, last_end=400, stride=4,
                         min_gap=8, max_chunks=8,
                         pilots=4, unit_div=24, min_chunk=16)
    assert has_pilots is True
    fset = set(frames)
    for s, e in spec[1:-1]:
        assert s in fset and e in fset
    assert spec[-1][1] == 400
    assert all(spec[i][1] == spec[i + 1][0]
               for i in range(len(spec) - 1))


def test_dual_chunk_specs_dense_keyframes_capped():
    """关键帧过密时逐步放大间距，竞争区内边界受 max_chunks 上限约束。"""
    f = FieldExtractor._dual_chunk_specs
    frames = list(range(0, 2000))
    kf = [i * 20 for i in range(1, 100)]   # 每 20 帧一个关键帧（密集）
    spec, has_pilots = f(frames, kf, last_end=2000, stride=1,
                         min_gap=8, max_chunks=6,
                         pilots=4, unit_div=24, min_chunk=16)
    assert has_pilots is True
    # 头部 4 试点 + 竞争区内边界（末片不计）≤ max_chunks
    assert len(spec) - 1 - 4 <= 6
    assert spec[0][0] == 0
    assert all(spec[i][1] == spec[i + 1][0]
               for i in range(len(spec) - 1))
    assert spec[-1][1] == 2000


# ═══════════════ 慢路径让位（adaptive yield，kfe 平衡机制，保留） ═══════════════

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
    monkeypatch.setenv("OCR_THREADS", "7")
    ex = _make(dual_pipeline=True)
    assert ex._dual_ocr_num_threads("tensorrt", 1) == 7
    assert ex._dual_ocr_num_threads("onnxruntime", 1) == 7


# ═══════════════ 每关键帧一片的切片生成（kfe，唯一分片方法） ═══════════════

def test_keyframe_every_chunks_basic():
    """基础最小间距下按关键帧切分；边界吸附到采样帧、覆盖无缝隙。"""
    f = FieldExtractor._keyframe_every_chunks
    frames = list(range(0, 1000))          # stride=1 采样帧
    kf = [100, 300, 500, 700]
    chunks = f(frames, kf, rest_start=0, last_end=1000,
               stride=1, min_gap=16, max_chunks=8)
    assert chunks == [(0, 100), (100, 300), (300, 500), (500, 700),
                      (700, 1000)]
    # 覆盖连续无缝隙
    assert chunks[0][0] == 0
    assert all(chunks[i][1] == chunks[i + 1][0]
               for i in range(len(chunks) - 1))
    assert chunks[-1][1] == 1000


def test_keyframe_every_chunks_max_chunks_cap():
    """关键帧过密时逐步放大间距，inner 片数受控在 max_chunks 内。"""
    f = FieldExtractor._keyframe_every_chunks
    frames = list(range(0, 2000))
    kf = [i * 20 for i in range(1, 100)]    # 每 20 帧一个关键帧（密集）
    chunks = f(frames, kf, rest_start=0, last_end=2000,
               stride=1, min_gap=8, max_chunks=6)
    # 只产生 max_chunks 个内部边界（末片不计）
    assert len(chunks) - 1 <= 6
    assert chunks[0][0] == 0
    assert all(chunks[i][1] == chunks[i + 1][0]
               for i in range(len(chunks) - 1))
    assert chunks[-1][1] == 2000


def test_keyframe_every_chunks_stride_snap():
    """采样步长>1 时边界吸附到最近采样帧（保持全帧覆盖、无缝隙）。"""
    f = FieldExtractor._keyframe_every_chunks
    frames = list(range(0, 400, 4))        # stride=4：采样帧 0,4,8,...
    kf = [10, 110, 210, 310]               # 关键帧可能不在采样网格上
    chunks = f(frames, kf, rest_start=0, last_end=400,
               stride=4, min_gap=8, max_chunks=8)
    # 内部边界（非首片起点/末片终点）必须是采样帧列表中的帧号；
    # 末片终点=last_end 允许不在采样网格上（真实代码中 last_end=total）
    fset = set(frames)
    for s, e in chunks[1:-1]:
        assert s in fset and e in fset
    assert chunks[-1][1] == 400
    assert all(chunks[i][1] == chunks[i + 1][0]
               for i in range(len(chunks) - 1))


def test_keyframe_every_chunks_no_keyframes():
    """无关键帧时退化为一整片。"""
    f = FieldExtractor._keyframe_every_chunks
    frames = list(range(0, 500))
    chunks = f(frames, [], rest_start=0, last_end=500,
               stride=1, min_gap=16, max_chunks=8)
    assert chunks == [(0, 500)]


def test_parallel_chunk_seek_required_passthrough(monkeypatch):
    """竞争取片时按“与上一片终点是否相邻”决定 seek_required。"""
    ex = _make(dual_pipeline=True)
    # 直接验证 _run_parallel_chunk 的默认参数存在即可（真实 seek 行为由
    # 集成冒烟覆盖）；此处只锁定签名兼容性。
    import inspect
    sig = inspect.signature(FieldExtractor._run_parallel_chunk)
    assert "seek_required" in sig.parameters
    assert sig.parameters["seek_required"].default is True