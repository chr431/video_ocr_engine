"""resolve() —— 全仓库唯一的 env 读取点（D6：构造期一次冻结，r3 裁决）。

优先级（Q5 裁决）：**显式参数 > env > 注册表默认**。
v1 相反（env 盖过构造参数，README 自称排查陷阱）；逃生门 `VOE_ENV_WINS=1`
恢复 v1 语义并发 DeprecationWarning（0.14.0 移除，§10.2/§10.3）。

本模块不读 os.environ 之外的任何环境状态；测试传入显式 env 映射。
解析语义与 engine_config.env_* 逐位一致（由 tests/config/test_resolve.py
的等价性测试锁死，防止两套解析器漂移）。
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from typing import Any, Mapping

from .knobs import KNOBS
from .run_config import RunConfig

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")
ENV_WINS = "VOE_ENV_WINS"


def _parse_bool(raw: str, default: bool) -> bool:
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return default


def _parse(knob, raw: str) -> Any:
    """按 v1 语义解析一个 env 原始值（非法/空回退默认）。"""
    t = knob.type
    s = raw.strip()
    if t == "bool":
        return _parse_bool(raw, bool(knob.default))
    if t == "bool|none":
        # v1 _gpu_pipeline.py:566-570：设置了但非法 → False（显式关），≠未设 None
        return _parse_bool(raw, False)
    if not s:
        return knob.default
    if t == "int":
        try:
            return int(s)
        except ValueError:
            return knob.default
    if t == "float":
        try:
            return float(s)
        except ValueError:
            return knob.default
    return raw  # str：原样（v1 TEXT_SEP_MERGE 是裸 get）


def resolve(*, env: Mapping[str, str] | None = None,
            overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """从 env + 显式参数构造不可变 RunConfig（含 config_digest）。

    overrides：键为旋钮规范名（如 "ocr.pad_small"）；值 None = 未给出，
    不参与优先级竞争（对应 v1 构造参数默认 None 的语义）。
    """
    env = os.environ if env is None else env
    overrides = overrides or {}
    env_wins = env.get(ENV_WINS, "").strip().lower() in _TRUTHY
    if env_wins:
        warnings.warn(
            "VOE_ENV_WINS=1 恢复 v1 的 env>参数 语义（0.14.0 移除）；"
            "v2 默认显式参数 > env > 默认（v2 §10.2）",
            DeprecationWarning, stacklevel=2)

    values: dict[str, Any] = {}
    for k in KNOBS.knobs:
        ov = overrides.get(k.name, None)
        raw = env.get(k.env) if k.env else None
        if raw is not None:
            parsed = _parse(k, raw)
            if env_wins or ov is None:
                values[k.name] = parsed
                continue
        values[k.name] = ov if ov is not None else k.default

    digest_src = json.dumps(values, sort_keys=True, ensure_ascii=True,
                            default=str)
    values["config_digest"] = hashlib.sha256(
        digest_src.encode("utf-8")).hexdigest()[:16]
    return RunConfig.from_values(values)
