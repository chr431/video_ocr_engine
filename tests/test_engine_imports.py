"""引擎独立性与 import 纯净性：不依赖应用层（速度/GUI），顶层 import 无泄漏。"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

ENGINE_MODULES = [
    "engine_config", "gpu_setup", "hybrid_decode", "ocr_native",
    "ocr_trt", "segmentation", "video_utils", "video_ocr_engine",
]

# 应用层（RaceVideoToLog）：引擎不得 import / 引入这些模块
APP_MODULES = {
    "config", "segment_flow", "seg_correction", "ocr_text", "ocr_engine",
    "csv_io", "gui", "signals", "monitor", "headless", "constants",
    "export_controller", "analysis",
}


def test_all_engine_modules_importable():
    for name in ENGINE_MODULES:
        importlib.import_module(name)


def test_engine_import_does_not_pull_app_modules():
    # 先清掉可能已加载的引擎模块再重新导入，检查被拉入 sys.modules 的应用层。
    for name in ENGINE_MODULES:
        sys.modules.pop(name, None)
    importlib.import_module("video_ocr_engine")
    leaked = {m for m in APP_MODULES if m in sys.modules}
    assert not leaked, f"引擎 import 引入了应用层模块: {sorted(leaked)}"


def test_extractor_ast_no_app_imports():
    """AST 审计：video_ocr_engine/extractor.py 顶层 import 无应用层/constants。"""
    src = Path(__file__).resolve().parent.parent / "video_ocr_engine" / "extractor.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            imports.add(n.module.split(".")[0])
        elif isinstance(n, ast.Import):
            for a in n.names:
                imports.add(a.name.split(".")[0])
    forbidden = APP_MODULES
    hits = sorted(imports & forbidden)
    assert not hits, f"extractor.py 顶层应用层 import: {hits}"


def test_engine_import_does_not_require_decord():
    """导入引擎不触发 decord（解码是运行时惰性依赖）。"""
    sys.modules.pop("decord", None)
    importlib.import_module("video_ocr_engine")
    assert "decord" not in sys.modules


def test_no_stale_methods_body_in_package():
    """_methods_body.py 是过期生成参考，不应随包发布。"""
    import video_ocr_engine
    stale = Path(video_ocr_engine.__file__).resolve().parent / "_methods_body.py"
    assert not stale.exists()


@pytest.mark.parametrize("name", ENGINE_MODULES)
def test_no_app_attr_leak(name):
    """引擎模块层面不导出应用层符号（轻量冒烟）。"""
    mod = importlib.import_module(name)
    for attr in ("extract_speed_value", "SegmentPipeline", "correct_segments"):
        assert not hasattr(mod, attr), f"{name} 不应有 {attr}"
