"""RunConfig —— 一次 run 的不可变配置（frozen，D6）。

S1 阶段承载 resolve() 产出的**旋钮层**（18 个 v1 env 旋钮 + telemetry）；
S3 起扩展为完整 RunConfig（video_path / roi / decode_backend 等每次调用
参数并入对应子域，v2 §6.1）。config_digest 覆盖全部生效值并写进 meta，
供 A/B 对账（v1 的 meta.params 只有 16 字段、env-only 旋钮不在其中）。
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RunConfig:
    # ── decode ──
    decode_num_threads: int = 0            # 0=自动分档
    decode_hybrid_cpu_threads: int = 0     # 0=自动 clamp [8,16]
    # ── ocr ──
    ocr_threads: int = 0                   # 0=全部物理核
    ocr_batch: int = 16
    ocr_gamma: float = 2.0
    ocr_pad_small: int = 0                 # 0=不生效
    ocr_roi_autocrop: bool = True
    ocr_roi_autocrop_margin: int = 10
    ocr_roi_autocrop_min_gain: int = 10
    ocr_reorder_window: int = 64
    ocr_instances: bool = True
    ocr_gpu_ctc: bool = True
    # ── segment ──
    segment_text_sep_merge: str = "binary"
    # ── pipeline ──
    pipeline_gpu: bool | None = None       # None=规则判定
    pipeline_gpu_stream: bool = False
    # ── diag ──
    diag_profile: bool = False
    diag_subprobe: bool = False
    diag_bounds_debug: bool = False
    diag_telemetry: str = "std"            # off/std/full（§8.6 r5）
    # ── 指纹 ──
    config_digest: str = ""

    @classmethod
    def from_values(cls, values: dict) -> "RunConfig":
        return cls(**{k.replace(".", "_"): v for k, v in values.items()})

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name != "config_digest"}

    def get(self, knob_name: str):
        """按旋钮规范名取值（"ocr.gamma" → self.ocr_gamma）。"""
        return getattr(self, knob_name.replace(".", "_"))
