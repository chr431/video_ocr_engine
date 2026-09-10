"""配置脊柱（v2 §5/D5/D6）：注册表 + 唯一 env 读取点 + 不可变 RunConfig。

字段与默认值由 knobs.py 声明；paths 单一实现仍在 engine_config
（S2 迁移），此处仅 re-export，避免两份路径逻辑。
"""
from __future__ import annotations

from engine_config import app_data_dir, app_logs_dir, models_dir  # noqa: F401

from .knobs import KNOBS
from .registry import Knob, Registry
from .resolve import resolve
from .run_config import RunConfig

__all__ = ["KNOBS", "Knob", "Registry", "resolve", "RunConfig",
           "app_data_dir", "app_logs_dir", "models_dir"]
