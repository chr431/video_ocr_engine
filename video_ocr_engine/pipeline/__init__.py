"""pipeline —— 编排层（v2 §5/D2：编排唯一一处）。

S3-2 起入住；S3-3 完成双驱动器归一。
"""
from __future__ import annotations

from .engine import RunOutcome, SegmentBackend, SegmentEngine

__all__ = ["RunOutcome", "SegmentBackend", "SegmentEngine"]
