"""双后端公共 setup 段（S4：P1-4"逐字节等价的 setup"至此一处）。

宿主与 GPU 驱动在 v1 里 setup 相位逐字节等价（open 后的 fps/帧区间/
空区间守卫/seek/hybrid_begin）；契约化后字段同名，抽到此共享。
teardown 相位差异实质（宿主无生产者线程/设备缓冲），保留各自实现。
"""
from __future__ import annotations

import logging

from video_ocr_engine.config import constants as config
from .._helpers import _read_fps_from_vr

logger = logging.getLogger(__name__)


def ensure_fps(spec, vr) -> float:
    """B2 语义：同实例同视频缓存（fps_box 单元素盒）。"""
    if spec.fps_box[0] is None:
        _fps = _read_fps_from_vr(vr)
        spec.fps_box[0] = _fps if _fps else config.DEFAULT_FPS_FALLBACK
    return spec.fps_box[0]


def compute_frames(spec, total: int) -> list:
    """帧区间推导 + 超界截断告警（A4：静默截断至少要能被发现）。"""
    if (spec.frame_end or 0) > total:
        logger.warning('frame_end=%s 超出视频总帧数 %d，按片尾截断',
                       spec.frame_end, total)
    end = min(spec.frame_end or total, total)
    return list(range(spec.frame_start, end, spec.sample_stride))


def begin_reading(vr, spec, frames: list) -> None:
    """seek 定位。`hybrid_begin`（项目层 HybridDecoder 壳 v3~v7 的 API，
    2026-09 随壳删除）的化石守卫已清除——fork 读者从未实现该 API，
    hasattr 恒 False，守卫只剩误导（2026-09-20 稳健性轮）。"""
    if spec.frame_start > 0:
        vr.seek_accurate(spec.frame_start)
