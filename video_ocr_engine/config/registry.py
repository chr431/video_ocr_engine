"""旋钮注册表（v2 ARCHITECTURE.md §6.1 / D5）。

一个旋钮 = 一条 Knob 记录：默认值、env 名、类型、生效域、依据锚点同处声明。
resolve() 是全仓库唯一的 env 读取点（D6）；未注册的名字不得读取。

rationale_id 在 S2 之前指向 engine_config 内的依据注释锚点（ec:<起止行>），
制度重构后改为 registry/knobs.yaml 条目 id（事实源仍为 config/knobs.py）——代码只留 id 反查。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Domain = Literal["decode", "segment", "ocr", "pipeline", "diag"]

_REGISTRY_CAP = 64  # §8.6 N-1：注册表上限，防 span/旋钮爆炸


@dataclass(frozen=True)
class Knob:
    name: str  # 规范名，如 "ocr.pad_small"
    py_name: str  # v1 常量名（兼容 alias），如 "OCR_PAD_SMALL_ENV"
    env: str | None  # env 名；None = 无 env 入口
    type: str  # "int" / "float" / "bool" / "str" / "bool|none"
    default: object
    domain: Domain
    applies_to: tuple[str, ...]  # ("cpu", "gpu") —— 取代散落的门控注释
    rationale_id: str  # → 依据锚点（S2 前为 ec:<行>，之后为 knobs.yaml id）
    note: str = ""  # 解析语义备注（三态 / 特殊回退等）
    # 消费者**调用期 env 活读**（子进程 A/B 逐臂切换用），RunConfig 字段
    # 仅供 introspection。2026-10-12 契约收紧：resolve 期 env 实值计入
    # config_digest（排除曾使指纹无法区分 fp16 两臂），显式 override 拒收
    # （resolve 报错指路 env）——字段不再可被构造参数改写。
    env_live_only: bool = False


@dataclass(frozen=True)
class Registry:
    knobs: tuple[Knob, ...] = field(default_factory=tuple)

    def by_name(self, name: str) -> Knob:
        for k in self.knobs:
            if k.name == name:
                return k
        raise KeyError("未注册旋钮: %s" % name)

    def env_names(self) -> tuple[str, ...]:
        return tuple(k.env for k in self.knobs if k.env)

    def __post_init__(self) -> None:
        assert len(self.knobs) <= _REGISTRY_CAP, "旋钮注册表超过 %d 上限" % _REGISTRY_CAP
        names = [k.name for k in self.knobs]
        assert len(set(names)) == len(names), "旋钮名重复"
        envs = [k.env for k in self.knobs if k.env]
        assert len(set(envs)) == len(envs), "env 名重复"
