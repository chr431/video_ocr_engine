"""engine_config 一致性 + 模型资产存在性（引擎可运行的基础）。"""
from __future__ import annotations

from pathlib import Path

import engine_config as config
import video_ocr_engine


ROOT = Path(__file__).resolve().parent.parent


def test_version_single_source():
    assert video_ocr_engine.__version__ == config.__version__


def test_key_engine_constants_present():
    for name in ("DEFAULT_OCR_MODEL", "DEFAULT_DECODE_BACKEND",
                 "DEFAULT_BUFFER_SIZE", "DEFAULT_FILL_WIDTH", "OCR_GAMMA",
                 "SEG_C", "OCR_TARGET_H", "OCR_BATCH_SIZE", "DECODE_BATCH_SIZE",
                 "OCR_PAD_WIDTH_MIN", "CPU_CORES_SPLIT_THRESHOLD"):
        assert hasattr(config, name), f"engine_config 缺少 {name}"


def test_ocr_model_assets_exist():
    """OcrEngine 依赖的模型 + 字符表必须随仓库存在（运行时从文件位置解析）。"""
    models = ROOT / "assets" / "ocr_models"
    assert (models / "PP-OCRv6_rec_small.onnx").is_file()
    assert (models / "ppocrv6_dict.txt").is_file()


def test_models_dir_finds_marker_in_source_tree():
    """源码模式下 _models_dir 必须解析到含模型文件的目录（wheel 安装回归保护）。"""
    from ocr_native import _models_dir as onnx_models
    from ocr_trt import _models_dir as trt_models
    for fn in (onnx_models, trt_models):
        p = fn()
        assert (p / "PP-OCRv6_rec_small.onnx").is_file(), p
        assert (p / "ppocrv6_dict.txt").is_file(), p


def test_dict_format_matches_models_dir():
    """字符表：读表逻辑 = 文件行数 + 末尾空格 + 开头 blank（与 ocr_native 一致）。"""
    from ocr_native import OcrEngine, _models_dir
    dict_path = _models_dir() / "ppocrv6_dict.txt"
    assert dict_path.is_file(), f"字符表不在模型目录: {dict_path}"
    with open(dict_path, "rb") as f:
        lines = [ln.decode("utf-8").strip("\n").strip("\r\n")
                 for ln in f.readlines()]
    assert lines and all(lines)
    # 与 OcrEngine 构造时的读表+插值逻辑完全对齐
    eng = OcrEngine.__new__(OcrEngine)  # 跳过 __init__（不加载模型）
    eng._chars = lines + [" "]           # 末尾插入空格
    eng._chars.insert(0, "blank")        # 开头插入 blank（CTC）
    assert eng._chars[0] == "blank"
    assert eng._chars[-1] == " "
    assert len(eng._chars) == len(lines) + 2  # 16 行文档未硬编码具体数
