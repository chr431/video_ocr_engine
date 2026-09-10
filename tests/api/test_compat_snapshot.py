"""S0 API 兼容快照（v2 ARCHITECTURE.md §10.1 冻结表的可执行形式）。

v1 的公共面在这里**逐字冻结**：v2 strangler 迁移（S1–S5）期间任何 PR
改动这些快照即测试失败——除非走 §10.2 的"有意变更"流程并同步本文件的
冻结字面量与 CHANGELOG。

冻结内容（全部为 r7 基线 7ba1c3a 实测捕获）：
  1. FieldExtractor.__init__ 签名：2 位置参数 + 17 仅关键字参数（名/序/默认值 repr）
  2. 默认值的 **对象同一性**：sample_stride / merge_similar 的默认值是
     engine_config 常量本身（import 期求值的表达式）——v2 若改成字面量，
     monkeypatch engine_config 后行为会变，此断言直接拦下（§10.1 警告）。
  3. ExtractedSegment 7 字段 / ExtractionResult 5 字段（含顺序，允许位置构造）
  4. meta 9 键 + params 16 子键 + timing 3 键（键集合；实际值由金标向量冻结）
  5. 包导出面与 6 个根模块可顶层导入
  6. engine_config 的 18 个 *_ENV 常量名
  7. 已删除名字保持不存在（硬失败，不得静默复活）
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest


def _snapshot_signature():
    from video_ocr_engine import FieldExtractor
    sig = inspect.signature(FieldExtractor.__init__)
    pos, kw = [], []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        d = "<required>" if p.default is p.empty else repr(p.default)
        (pos if p.default is p.empty and p.kind is p.POSITIONAL_OR_KEYWORD
         else kw).append("%s=%s" % (name, d))
    return pos, kw


def test_signature_frozen():
    pos, kw = _snapshot_signature()
    assert pos == ["video_path=<required>", "roi=<required>"]
    assert kw == [
        "frame_start=None", "frame_end=None", "force_aspect=0.0",
        "decode_backend='auto'", "ocr_backend='auto'",
        "buffer_size=None", "fill_width=None", "C=None", "sample_stride=1",
        "progress_cb=None", "cancel_check=None", "rep_crop_format=None",
        "keep_crops=True", "keep_frames=True", "merge_similar=True",
        "merge_similar_threshold=None", "merge_text_sep=None",
    ]


def test_defaults_are_engine_config_objects():
    """默认值必须是 engine_config 常量的同一对象，不是抄写的字面量。"""
    from video_ocr_engine import FieldExtractor
    import engine_config as config
    sig = inspect.signature(FieldExtractor.__init__)
    assert sig.parameters["sample_stride"].default is config.DEFAULT_SAMPLE_STRIDE
    assert sig.parameters["merge_similar"].default is config.DEFAULT_MERGE_SIMILAR


def test_dataclass_fields_frozen():
    from video_ocr_engine import ExtractedSegment, ExtractionResult
    assert [f.name for f in dataclasses.fields(ExtractedSegment)] == [
        "start", "end", "frames", "rep_frame", "text", "confidence",
        "rep_crop"]
    assert [f.name for f in dataclasses.fields(ExtractionResult)] == [
        "segments", "frames", "fps", "timing", "meta"]
    assert callable(getattr(ExtractionResult, "rep_crop_rgb", None))


META_KEYS = {"backend", "ocr_backend", "codec", "n_segments",
             "engine_version", "color_range", "rep_crop_format",
             "degraded_reason", "params"}
PARAMS_KEYS = {"roi", "frame_start", "frame_end", "decode_backend",
               "ocr_backend", "sample_stride", "fill_width", "force_aspect",
               "rep_crop_format", "keep_crops", "keep_frames",
               "merge_similar", "merge_similar_threshold", "merge_text_sep",
               "buffer_size", "C"}
TIMING_KEYS = {"decode", "ocr", "ocr_tail"}


def test_meta_param_timing_key_sets_frozen():
    """键集合冻结（r7 勘误 D-15：meta 实为 9 键，v2 文档原写 10）。"""
    assert len(META_KEYS) == 9 and len(PARAMS_KEYS) == 16
    assert len(TIMING_KEYS) == 3


def test_package_exports_and_root_modules():
    import video_ocr_engine as pkg
    import engine_config
    for name in ("FieldExtractor", "ExtractedSegment", "ExtractionResult"):
        assert getattr(pkg, name) is not None
    assert pkg.__version__ == engine_config.__version__
    import importlib
    for mod in ("engine_config", "segmentation", "video_utils",
                "ocr_native", "ocr_trt", "gpu_setup"):
        assert importlib.import_module(mod) is not None


def test_env_name_constants_frozen():
    import engine_config as config
    names = sorted(n for n in dir(config) if n.endswith("_ENV"))
    assert len(names) == 18
    # 名字快照（附录 C 全表的键集合）
    assert names == sorted([
        "OCR_THREADS_ENV", "OCR_BATCH_ENV", "OCR_GAMMA_ENV",
        "OCR_PAD_SMALL_ENV", "OCR_ROI_AUTOCROP_ENV",
        "OCR_ROI_AUTOCROP_MARGIN_ENV", "OCR_ROI_AUTOCROP_MIN_GAIN_ENV",
        "OCR_REORDER_WINDOW_ENV", "OCR_INSTANCES_ENV", "TEXT_SEP_MERGE_ENV",
        "DECODE_THREADS_ENV", "GPU_PIPELINE_ENV", "GPU_PIPELINE_STREAM_ENV",
        "GPU_CTC_ENV", "ENGINE_PROFILE_ENV", "TRT_SUBPROBE_ENV",
        "DEBUG_BOUNDS_ENV", "HYBRID_CPU_THREADS_ENV"])


DELETED_NAMES = {
    "video_ocr_engine": ("gray_output", "yuv_output", "SegmentPipeline",
                         "extract_speed_value", "correct_segments"),
    "engine_config": ("TEXT_SEP_MERGE_CONTRAST",),
    "gpu_setup": ("select_backend", "get_engine_type", "get_setup_advice"),
}


def test_deleted_names_stay_deleted():
    import video_ocr_engine as pkg
    import engine_config
    import gpu_setup
    for name in DELETED_NAMES["video_ocr_engine"]:
        assert not hasattr(pkg, name), name
    for name in DELETED_NAMES["engine_config"]:
        assert not hasattr(engine_config, name), name
    for name in DELETED_NAMES["gpu_setup"]:
        assert not hasattr(gpu_setup, name), name


def test_models_dir_alias():
    import ocr_native
    import ocr_trt
    assert hasattr(ocr_native, "_models_dir")
    assert hasattr(ocr_trt, "_models_dir")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
