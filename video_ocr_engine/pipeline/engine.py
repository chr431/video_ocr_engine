"""SegmentEngine —— 唯一编排点（v2 §6.4 / D2）。

S3-2 交付骨架与接缝：端口（FrameSource / SegmentBackend / OcrBackend）、
RunOutcome 载体、以及**旧路径适配器**——engine.run() 目前委托给 v1 的
宿主/GPU 双驱动器（_run_pipelined_host / _run_pipelined_gpu），行为零变化。
S3-3 把驱动器内部逐步迁入 HostSegmentBackend / GpuSegmentBackend 后，
本类成为真正的三段编排（calibrate → backend.run → ocr.drain）。

emit 契约自 S3 起即为**批量签名**（v2 §6.3 r3 定稿）：初期实现可逐段
转发，但接口不再中途变更。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


class SegmentBackend(Protocol):
    """把"一段帧 → 段 + 代表帧"的实现与编排解耦（§6.3）。

    两个实现：HostSegmentBackend（numpy）/ GpuSegmentBackend（设备驻留）。
    语义等价由金标向量逐位校验（§8.5/D4）；插拔的是**执行后端**，
    不是算法（C-32 依然成立）。
    """
    name: str

    def run(self, src, cfg, emit: Callable[[Sequence], None],
            progress, cancel) -> None: ...


@dataclass
class RunOutcome:
    """一次 run 的产物（v2 §6.4；v1 的裸 5 元组至此有名字）。"""
    frames: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    confs: list = field(default_factory=list)
    rep_frames: list = field(default_factory=list)
    report: dict | None = None          # S3-3 起：RunReport（§8.6）

    def as_tuple(self) -> tuple:
        """与 v1 `_run_pipelined` 返回形状逐位兼容（迁移期）。"""
        return (self.frames, self.segments, self.texts, self.confs,
                self.rep_frames)


class _LegacyBackend:
    """S3-2 适配器：完整复用 v1 双驱动器（452+137 行），零行为变化。

    S3-3 的迁移目标就是它的内部；迁移完成前 FieldExtractor 可经
    VOE_V2_ENGINE=1 试运行新编排（当前=同一路径，双跑对账用）。
    """

    name = "legacy"

    def __init__(self, ex, ocr_engines: list | None = None) -> None:
        self._ex = ex
        self._engines = ocr_engines

    def run(self, src=None, cfg=None, emit=None, progress=None,
            cancel=None) -> RunOutcome:
        frames, segs, texts, confs, rep = self._ex._run_pipelined_legacy(
            self._engines)
        return RunOutcome(frames, segs, texts, confs, rep)


class SegmentEngine:
    """编排唯一出处。构造期注入 backend；run() 产出 RunOutcome。"""

    def __init__(self, backend: SegmentBackend) -> None:
        self._backend = backend

    def run(self) -> RunOutcome:
        return self._backend.run()

    @staticmethod
    def legacy_engine_requested() -> bool:
        """VOE_V2_ENGINE=1 时走 SegmentEngine（当前与旧路径同源，双跑对账）。"""
        return os.environ.get("VOE_V2_ENGINE", "").strip().lower() in (
            "1", "true", "yes", "on")
