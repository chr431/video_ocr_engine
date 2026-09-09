"""宿主流水线（_HostPipelineMixin）：解码校准 / 帧流 / 分段状态机 / OCR 会话。

从 extractor.py 拆出（2026-08）：宿主路径的模块级纯函数（_host_calibrate /
_host_frame_stream / _host_segment_frames）与 FieldExtractor 的 OCR 会话
集中于此，extractor.py 仅保留引擎骨架与流水线分发；
OCR 消费会话（OcrSession）在 _ocr_session.py（0.11.0 拆出，双管线共用）。
模块级函数签名不变，旧导入路径 `from video_ocr_engine.extractor import
_host_calibrate` 等仍可用（extractor 顶部 re-export）。
"""
from __future__ import annotations

import logging
import time

import numpy as np

import engine_config as config
from segmentation import (SegmentStateMachine, _otsu,
                          otsu_median_threshold)
from ._ocr_session import OcrSession
from ._helpers import _ndarray_device_ptr, _decode_progress_pct

logger = logging.getLogger(__name__)


def _host_calibrate(ex, vr, frames, *, with_dev=False):
    """宿主路径 Otsu 校准（单流水线统一入口）。

    ex: FieldExtractor——只调用 _crop_is_expected / _crop_luma /
        _prof_end。
    with_dev: True 时保留 decord GPU 单通道帧的 DLPack 指针（GPU raw OCR
        直通用）；stride==1 时同时捕获 next_roi 的（shape 3D）帧指针。
    stride>1 走 get_batch 等差步长快速路径（校准帧号与后续帧流一致），
    stride==1 走 next_roi 顺序流。
    返回 (calib, th)。calib 元素统一 (fi, crop, gray, sharp, dev_info)，
    dev_info 仅在 with_dev 且帧为 GPU 单通道时非 None。
    """
    x1, y1, x2, y2 = ex._roi
    calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
    calib: list = []
    if ex._sample_stride > 1:
        nds = vr.get_batch(frames[:calib_n], roi=(x1, y1, x2 + 1, y2 + 1))
        crops = nds.asnumpy()
        base, shape = (0, ())
        dev_c = 0
        if with_dev:
            # 与旧单流水线一致：只要请求设备指针就捕获（不先看 shape）——
            # channel 判定由捕获后的 shape 完成（非 GPU 单通道自然 dev_c=0）。
            base, shape = _ndarray_device_ptr(nds)
            dev_c = shape[-1] if len(shape) == 4 else 0
        for k in range(calib_n):
            c = crops[k]
            if not ex._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                c = c[y1:y2 + 1, x1:x2 + 1]
            g = ex._crop_luma(c)
            dev_info = None
            if dev_c == 1 and len(shape) == 4:
                src_h, src_w = shape[1], shape[2]
                dev_info = (nds, base + k * src_h * src_w, src_h, src_w)
            calib.append((frames[k], c, g, float(g.std()), dev_info))
    else:
        for k in range(calib_n):
            nd = vr.next_roi(x1, y1, x2 + 1, y2 + 1)
            c = nd.asnumpy()
            if not ex._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                c = c[y1:y2 + 1, x1:x2 + 1]
            g = ex._crop_luma(c)
            dev_info = None
            if with_dev and len(nd.shape) == 3 and nd.shape[-1] == 1:
                base, shape = _ndarray_device_ptr(nd)
                dev_info = (nd, base, shape[0], shape[1])
            calib.append((frames[k], c, g, float(g.std()), dev_info))
    return calib, otsu_median_threshold(
        [_otsu(g) for _fi, _c, g, _s, _dev in calib])


def _host_frame_stream(ex, frames, vr, calib, th, *, with_dev=False):
    """宿主帧流：先产出校准帧，再批量流式解码剩余帧。

    ex: FieldExtractor——只调用 _batch_luma/_prof_end。
    calib 元素统一 (fi, crop, gray, sharp, dev_info)（可为空列表）。
    with_dev=True 时随帧产出 decord GPU NDArray 设备信息 (owner, ptr, h, w)
    供 GPU raw OCR 直通（仅 gray 单通道输出路径有效）。
    yield (frame_idx, crop, gray, sharp, bin, dev_info)。
    """
    DECODE_BATCH = config.DECODE_BATCH_SIZE
    x1, y1, x2, y2 = ex._roi
    for fi, c, g, s, *dev_rest in calib:
        yield (fi, c, g, s, g > th, dev_rest[0] if dev_rest else None)
    g_buf = None   # 复用批量灰度缓冲（每批形状恒定：B×H×W）
    for bstart in range(len(calib), len(frames), DECODE_BATCH):
        bend = min(bstart + DECODE_BATCH, len(frames))
        _t_d = time.perf_counter()
        nds = vr.get_batch(frames[bstart:bend], roi=(x1, y1, x2 + 1, y2 + 1))
        crops = nds.asnumpy()
        ex._prof_end('producer', 'decode_batch', _t_d)
        _t_g = time.perf_counter()
        if g_buf is None:
            # 灰度 Y 缓冲（每批形状恒定才可跨批复用）：
            #   yuv(NV12)：crops=(B, rows, W) → Y=(B, rows*2//3, W)
            #   gray：crops=(B, H, W[, 1]) → Y=(B, H, W)
            # （旧实现 yuv 把 2//3 乘在宽度上、gray 只取 shape[:2] 少一维，
            #   复用条件因此两种格式下都永假 → 复用从未生效，DESIGN-REVIEW C3。）
            g_buf = np.empty(
                (crops.shape[0], crops.shape[1] * 2 // 3, crops.shape[2])
                if ex._yuv_output else crops.shape[:3],
                dtype=np.uint8)
        if (g_buf.shape[1:] == ((crops.shape[1] * 2 // 3, crops.shape[2])
                                if ex._yuv_output else crops.shape[1:3])
                and len(crops) <= g_buf.shape[0]):
            # 该批可能不是满批（末批 B 更小）：只复用前 B 行，避免形状不匹配
            g = ex._batch_luma_out(crops, g_buf[:len(crops)])
        else:
            g = ex._batch_luma(crops)
        ex._prof_end('producer', 'gray_batch', _t_g)
        g = np.ascontiguousarray(g)
        _t_s = time.perf_counter()
        sharp = g.std(axis=(1, 2))
        ex._prof_end('producer', 'sharp_batch', _t_s)
        _t_b = time.perf_counter()
        bs = g > th
        ex._prof_end('producer', 'bin_batch', _t_b)
        dev_base = 0
        src_h = src_w = 0
        if with_dev and len(nds.shape) == 4 and nds.shape[-1] == 1:
            dev_base, shape = _ndarray_device_ptr(nds)
            src_h, src_w = shape[1], shape[2]
        for k, gi in enumerate(range(bstart, bend)):
            d = None
            if dev_base:
                d = (nds, dev_base + k * src_h * src_w, src_h, src_w)
            yield (frames[gi], crops[k], g[k], float(sharp[k]), bs[k], d)


def _host_segment_frames(ex, frames, stream, *, debug_tag, progress_prefix,
                         emit, segs):
    """宿主分段状态机 —— 编排统一实现在 segmentation.SegmentStateMachine
    （0.11.0 起与 GPU 全驻留管线共用同一状态机；本函数只做宿主侧接线）。

    ex: FieldExtractor。
    stream: (fi, crop, gray, sharp, bin, dev_info) 迭代器（_host_frame_stream）。
    emit(seg, rep_frame, rep_crop, rep_dev, rep_gray, frac)：段闭合投递
        OCR，由调用方闭包实现（入队/全局段号/keys/reps/rep_crops 收敛在
        闭包里；调用方在 emit 前已把 seg 追加进 segs —— 现由状态机持有
        segs，结束时拷回传入列表）。
    debug_tag 非 None 且 DEBUG_BOUNDS 开关开启时打印边界（[HB]=单流水线，
    与 GPU 路径 [GB] 对齐；分值 = win3 聚类分数）。
    返回传入的 segs（同一列表，内容 = 状态机产出的段）。
    """
    machine = SegmentStateMachine(
        frames, C=ex._C,
        on_emit=lambda seg, rep, frac: emit(seg, rep[0], rep[1], rep[3],
                                            rep[2], frac),
        on_similar=lambda a, b: (ex._merge_similar
                                 and ex._segments_similar(a[2], b[2])),
        on_cancel=ex._cancel,
        on_progress=lambda k, frac: ex._progress(
            f'{progress_prefix}: {k}/{len(frames)}',
            _decode_progress_pct(frac)),
        debug_tag=debug_tag)
    for k, (fi, c, g, sharp, b, dev) in enumerate(stream):
        machine.feed(k, fi, sharp, (fi, c, g, dev), bin=b)
    machine.finish()
    segs[:] = machine.segs
    return segs


class _HostPipelineMixin:
    """宿主流水线 mixin：FieldExtractor 组合本类获得 OCR 会话与宿主管线。"""

    # ═══════════════ OCR 输入宽度自适应裁切 ═══════════════
    # 统一实现（含余量/最小收益门槛的实测依据 docstring）在
    # segmentation.content_range_to_crop / crop_to_content / crop_after_aspect；
    # GPU 直通（_autocrop_device）与宿主预处理共用同一余量数学。

    def _content_range_to_crop(self, first: int, last: int, w: int):
        """「有墨迹列范围」→ 裁切区间；实现见 segmentation.content_range_to_crop。"""
        from segmentation import content_range_to_crop
        return content_range_to_crop(
            first, last, w,
            margin_pct=self._ocr_autocrop_margin_pct,
            min_gain=self._ocr_autocrop_min_gain)

    def _crop_to_content(self, crop):
        """按二值图裁掉两侧空白（fa=0 路径）；实现见
        segmentation.crop_to_content（fa>0 走 _crop_after_aspect 顺序⑦）。"""
        from segmentation import crop_to_content
        return crop_to_content(
            crop, self._bin_thresh,
            autocrop=self._ocr_autocrop,
            force_aspect=float(getattr(self, '_force_aspect', 0) or 0.0),
            margin_pct=self._ocr_autocrop_margin_pct,
            min_gain=self._ocr_autocrop_min_gain)

    def _crop_after_aspect(self, img):
        """已定比例图上再按内容列裁（fa>0 路径）；实现见
        segmentation.crop_after_aspect（阈值现算 Otsu，不能用校准阈值）。"""
        from segmentation import crop_after_aspect
        return crop_after_aspect(
            img, autocrop=self._ocr_autocrop,
            margin_pct=self._ocr_autocrop_margin_pct,
            min_gain=self._ocr_autocrop_min_gain)

    def _start_ocr_session(self, _ocr_engines: list | None = None) -> "OcrSession":
        """启动 OCR 消费会话（实现见 _ocr_session.OcrSession；
        一次 extract() 一个实例，宿主/GPU 两管线共用）。"""
        return OcrSession(self, _ocr_engines)

