"""SegmentEngine —— 唯一编排点（v2 §6.4 / D2；S3-3d 起为唯一入口）。

职责：按门面状态选择后端（GPU 全驻留 / 宿主），产出 RunOutcome。
S3-3d 用户裁决后不再保留 VOE_V2_ENGINE 过渡开关与 _LegacyBackend——
引擎即唯一路径（实验钩子不承重）。端口（FrameSource / SegmentBackend /
OcrBackend，decode/port.py 与 ocr/port.py）已定义，S4 起后端以
SegmentBackend 实现类落地并吸收两驱动逐字节相同的 setup/teardown 段。

emit 契约为**批量签名**（v2 §6.3 r3 定稿）。
"""
from __future__ import annotations

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
    report: dict | None = None          # S4 起：RunReport（§8.6）

    def as_tuple(self) -> tuple:
        """与 v1 `_run_pipelined` 返回形状逐位兼容（迁移期）。"""
        return (self.frames, self.segments, self.texts, self.confs,
                self.rep_frames)


class SegmentEngine:
    """编排唯一出处：选择后端（GPU 门控通过 → gpu_backend，否则宿主）。"""

    def __init__(self, ex) -> None:
        self._ex = ex

    def run(self, ocr_engines: list | None = None) -> RunOutcome:
        ex = self._ex
        if ex._gpu_pipeline_enabled():
            outcome = ex._run_pipelined_gpu(ocr_engines)
        else:
            outcome = ex._run_pipelined_host(ocr_engines)
        return RunOutcome(*outcome)
