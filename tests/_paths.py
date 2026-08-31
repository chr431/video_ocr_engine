"""测试共用的仓库根路径常量。

⛔ **不要写** `Path(__file__).resolve().parent.parent` 这种按**目录深度**猜根的写法。

`tests/` 于 2026-08-31 拆成 decode/ pipeline/ segment/ api/ utils/ 五个子目录后，
测试文件的深度从 2 层变成 3 层，上面那类写法会静默指向 `tests/` 而不是仓库根
——表现为 `FileNotFoundError` 或"资产不存在"断言失败，而不是明显的路径错误。

本模块改为**向上遍历找标记文件**，与调用者所在深度无关，今后再加子目录也不会坏。

用法（配合 tests/conftest.py 把 tests/ 放进 sys.path）：

    from _paths import ROOT, PKG, ASSETS, MODELS
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ROOT", "PKG", "ASSETS", "MODELS"]


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for q in (p, *p.parents):
        if (q / "pyproject.toml").is_file() and (q / "video_ocr_engine").is_dir():
            return q
    raise RuntimeError("找不到仓库根（缺 pyproject.toml 或 video_ocr_engine/）")


ROOT = _find_root()
PKG = ROOT / "video_ocr_engine"
ASSETS = ROOT / "assets"
MODELS = ASSETS / "ocr_models"
