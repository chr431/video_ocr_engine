"""（兼容 shim，0.13.0 起废弃、0.14.0 删除——Q3 裁决）。

实现已迁至 `video_ocr_engine.gpu.context`；本文件仅为旧导入路径保活：模块别名方式保留
全部符号（含下划线名）与模块同一性。请迁移：

    - 旧：import gpu_setup / from gpu_setup import X
    - 新：from video_ocr_engine.gpu import X

迁移表见 docs/MIGRATION.md。
"""
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "根模块 `gpu_setup` 已废弃（实现迁至 video_ocr_engine.gpu.context），0.14.0 删除——"
    "见 docs/MIGRATION.md",
    DeprecationWarning,
    stacklevel=2)

from video_ocr_engine.gpu.context import *  # noqa: F401,F403
import video_ocr_engine.gpu.context as _impl

_sys.modules[__name__] = _impl
