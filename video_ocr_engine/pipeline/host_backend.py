"""宿主后端（v2 §5/§6.2，S3-3a）。

从 extractor._run_pipelined_host 与 _host_pipeline 的三个模块级函数外提
（2026-09-10）。语义变化：**算法代码只读 HostRunSpec 的显式声明字段**，
不再穿透 FieldExtractor 的私有属性（P0-1 宿主子集至此有契约）；运行态
只写入 HostRunResult 与两个显式可变盒（fps 缓存 / backend 标签），
由门面在调用前后同步。

_segment/_calibrate/_frame_stream 的像素与判定原语经 spec 回调注入
（实现仍唯一出自 segmentation / extractor 薄层）；GPU 路径的
_gpu_fallback_to_host 经 preopened_vr 复用（C10 语义不变）。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from video_ocr_engine.config import constants as config
from video_ocr_engine.domain.segmentation import SegmentStateMachine, _otsu, otsu_median_threshold

logger = logging.getLogger(__name__)


@dataclass
class HostRunSpec:
    """宿主驱动的全量输入契约（取代对 FieldExtractor 私有属性的读取）。"""
    # 帧区间 / ROI / 采样
    frame_start: int
    frame_end: int | None
    sample_stride: int
    roi: tuple                                # (x1, y1, x2, y2)
    # 分段与输出语义
    C: float
    merge_similar: bool
    keep_crops: bool
    yuv_output: bool
    # 判定与像素原语（显式注入；实现唯一出处不变）
    segments_similar: Callable                # (gray_a, gray_b) -> bool
    crop_luma: Callable                       # crop -> gray
    batch_luma: Callable                      # (B,H,W[,C]) -> (B,h,w)
    batch_luma_out: Callable                  # 同上，写入复用缓冲
    crop_is_expected: Callable                # (crop, roi_h, roi_w) -> bool
    # facade 回调
    open_vr: Callable                         # () -> vr（门面负责后端选择/降级记录）
    start_ocr_session: Callable               # (engines|None) -> OcrSession
    backend_label: Callable = lambda: ""      # () -> str（open_vr 后的门面标签）
    progress: Callable = lambda m, p: None
    cancel: Callable = lambda: None
    prof_end: Callable | None = None          # (group, key, t0)
    # 校准阈值即时回写（F-4：_segments_similar 在流式期间活读宿主的
    # _bin_thresh——回写若延迟到 run 结束，合并判定会用旧值，段数漂移）
    on_bin_thresh: Callable | None = None     # (th) -> None
    # 可变盒（门面持有缓存，后端读写）
    fps_box: list = field(default_factory=lambda: [None])     # B2：同实例缓存


@dataclass
class HostRunResult:
    """宿主驱动的全量输出（v1 裸 5 元组的显式化）。"""
    frames: list = field(default_factory=list)
    segs: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    confs: list = field(default_factory=list)
    rep_frames: list = field(default_factory=list)
    fps: float | None = None
    bin_thresh: int = 0
    crops: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    n_segments: int = 0

    def as_tuple(self) -> tuple:
        return (self.frames, self.segs, self.texts, self.confs, self.rep_frames)


def _prof(spec, group: str, key: str, t0: float) -> None:
    if spec.prof_end is not None:
        spec.prof_end(group, key, t0)


def _calibrate(spec: HostRunSpec, vr, frames, *, with_dev: bool):
    """宿主 Otsu 校准（原 _host_calibrate；ex 读取 → spec 字段）。

    stride>1 走 get_batch 等差快速路径；stride==1 走 next_roi 顺序流
    （校准帧号与后续帧流一致）。返回 (calib, th)，calib 元素
    (fi, crop, gray, sharp, dev_info)。
    """
    from .._helpers import _ndarray_device_ptr
    x1, y1, x2, y2 = spec.roi
    calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
    calib: list = []
    if spec.sample_stride > 1:
        nds = vr.get_batch(frames[:calib_n], roi=(x1, y1, x2 + 1, y2 + 1))
        crops = nds.asnumpy()
        base, shape = (0, ())
        dev_c = 0
        if with_dev:
            base, shape = _ndarray_device_ptr(nds)
            dev_c = shape[-1] if len(shape) == 4 else 0
        for k in range(calib_n):
            c = crops[k]
            if not spec.crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                c = c[y1:y2 + 1, x1:x2 + 1]
            g = spec.crop_luma(c)
            dev_info = None
            if dev_c == 1 and len(shape) == 4:
                src_h, src_w = shape[1], shape[2]
                dev_info = (nds, base + k * src_h * src_w, src_h, src_w)
            calib.append((frames[k], c, g, float(g.std()), dev_info))
    else:
        for k in range(calib_n):
            nd = vr.next_roi(x1, y1, x2 + 1, y2 + 1)
            c = nd.asnumpy()
            if not spec.crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                c = c[y1:y2 + 1, x1:x2 + 1]
            g = spec.crop_luma(c)
            dev_info = None
            if with_dev and len(nd.shape) == 3 and nd.shape[-1] == 1:
                base, shape = _ndarray_device_ptr(nd)
                dev_info = (nd, base, shape[0], shape[1])
            calib.append((frames[k], c, g, float(g.std()), dev_info))
    return calib, otsu_median_threshold(
        [_otsu(g) for _fi, _c, g, _s, _dev in calib])


def _frame_stream(spec: HostRunSpec, frames, vr, calib, th, *, with_dev: bool):
    """宿主帧流（原 _host_frame_stream）。

    先产出校准帧，再按 DECODE_BATCH 批量解码。
    yield (frame_idx, crop, gray, sharp, bin, dev_info)。
    """
    from .._helpers import _ndarray_device_ptr
    DECODE_BATCH = config.DECODE_BATCH_SIZE
    x1, y1, x2, y2 = spec.roi
    for fi, c, g, s, *dev_rest in calib:
        yield (fi, c, g, s, g > th, dev_rest[0] if dev_rest else None)
    g_buf = None   # 复用批量灰度缓冲（每批形状恒定：B×H×W）
    for bstart in range(len(calib), len(frames), DECODE_BATCH):
        bend = min(bstart + DECODE_BATCH, len(frames))
        _t_d = time.perf_counter()
        nds = vr.get_batch(frames[bstart:bend], roi=(x1, y1, x2 + 1, y2 + 1))
        crops = nds.asnumpy()
        _prof(spec, 'producer', 'decode_batch', _t_d)
        _t_g = time.perf_counter()
        if g_buf is None:
            # 灰度 Y 缓冲（每批形状恒定才可跨批复用）：
            #   yuv(NV12)：crops=(B, rows, W) → Y=(B, rows*2//3, W)
            #   gray：crops=(B, H, W[, 1]) → Y=(B, H, W)
            g_buf = np.empty(
                (crops.shape[0], crops.shape[1] * 2 // 3, crops.shape[2])
                if spec.yuv_output else crops.shape[:3],
                dtype=np.uint8)
        if (g_buf.shape[1:] == ((crops.shape[1] * 2 // 3, crops.shape[2])
                                if spec.yuv_output else crops.shape[1:3])
                and len(crops) <= g_buf.shape[0]):
            # 末批 B 更小：只复用前 B 行，避免形状不匹配
            g = spec.batch_luma_out(crops, g_buf[:len(crops)])
        else:
            g = spec.batch_luma(crops)
        _prof(spec, 'producer', 'gray_batch', _t_g)
        g = np.ascontiguousarray(g)
        _t_s = time.perf_counter()
        sharp = g.std(axis=(1, 2))
        _prof(spec, 'producer', 'sharp_batch', _t_s)
        _t_b = time.perf_counter()
        bs = g > th
        _prof(spec, 'producer', 'bin_batch', _t_b)
        dev_base = 0
        src_h = src_w = 0
        if with_dev and len(nds.shape) == 4 and nds.shape[-1] == 1:
            dev_base, shape = _ndarray_device_ptr(nds)
            src_h, src_w = shape[1], shape[2]
        for k, gi in enumerate(range(bstart, bend)):
            d = None
            if dev_base:
                d = (nds, dev_base + k * src_h * src_w, src_h, src_w)
            # gray 必须拷贝：payload 灰度会作为代表帧逃逸出批作用域，
            # 而 g_buf 跨批复用——不拷贝则 merge_similar 在后续批读到
            # 被覆写的帧（2026-09-10 D1 调查：test5 host 45/1089 合并
            # 判定失真、段数 1042 vs GPU 正确值 1083；sharp/bs 为标量/
            # 每批新数组不受影响）。拷贝 ~3.5KB/帧，量级可忽略。
            yield (frames[gi], crops[k], g[k].copy(), float(sharp[k]),
                   bs[k], d)


def _segment_frames(spec: HostRunSpec, frames, stream, *, emit, segs):
    """宿主分段状态机接线（原 _host_segment_frames）。

    编排统一实现在 segmentation.SegmentStateMachine（与 GPU 共用）；
    本函数只做宿主侧接线。
    """
    from .._helpers import _decode_progress_pct
    prefix = f'[{spec.backend_label()}] 解码+分段'
    machine = SegmentStateMachine(
        frames, C=spec.C,
        on_emit=lambda seg, rep, frac: emit(seg, rep[0], rep[1], rep[3],
                                            rep[2], frac),
        on_similar=lambda a, b: (spec.merge_similar
                                 and spec.segments_similar(a[2], b[2])),
        on_cancel=spec.cancel,
        on_progress=lambda k, frac: spec.progress(
            f'{prefix}: {k}/{len(frames)}',
            _decode_progress_pct(frac)),
        debug_tag='HB')
    for k, (fi, c, g, sharp, b, dev) in enumerate(stream):
        machine.feed(k, fi, sharp, (fi, c, g, dev), bin=b)
    machine.finish()
    segs[:] = machine.segs
    return segs


def run_host_pipeline(spec: HostRunSpec, ocr_engines=None,
                      preopened_vr=None) -> HostRunResult:
    """宿主驱动（原 extractor._run_pipelined_host 的主体）。

    解码∥分段∥OCR：段一闭合即把代表帧交给 OCR 会话线程；返回显式
    HostRunResult（门面负责同步回实例属性与 meta）。
    """
    res = HostRunResult()
    _t_open = time.perf_counter()
    vr = preopened_vr
    if vr is None:
        vr = spec.open_vr()
    from ._run_common import begin_reading, compute_frames, ensure_fps
    res.fps = ensure_fps(spec, vr)
    total = len(vr)
    frames = compute_frames(spec, total)
    if not frames:
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛（资源由进程回收）
        raise ValueError(
            f"帧区间为空: frame_start={spec.frame_start}, "
            f"frame_end={spec.frame_end}, total={total}")
    hybrid = hasattr(vr, 'hybrid_begin')   # _with_dev 判定用
    try:
        begin_reading(vr, spec, frames)
    except BaseException:
        logger.debug("hybrid_begin 失败进入回退清理", exc_info=True)
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛
        raise
    _prof(spec, 'producer', 'open_and_fps', _t_open)
    # OCR 会话提前到校准前启动：worker 线程内构建引擎，与校准并行重叠；
    # 引擎就绪前 emit 自动走 host 回退，语义不变。
    try:
        ocr_session = spec.start_ocr_session(ocr_engines)
    except BaseException:
        logger.debug("OCR 会话启动失败进入清理", exc_info=True)
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛（资源由进程回收）
        raise
    results = ocr_session.results
    ocr_err = ocr_session.err
    ocr_wall = ocr_session.wall
    _put_ocr = ocr_session.put
    try:
        _t_cal = time.perf_counter()
        # with_dev=True：保留 GPU 单通道帧的 DLPack 指针供 GPU raw OCR 直通。
        # hybrid 交付宿主数组（无 to_dlpack），不存在可直通指针——强采
        # _ndarray_device_ptr 会 AttributeError，必须跳过。
        _with_dev = not hybrid
        calib, th = _calibrate(spec, vr, frames, with_dev=_with_dev)
        res.bin_thresh = th
        if spec.on_bin_thresh is not None:
            spec.on_bin_thresh(th)   # 流式合并判定活读，必须即时回写（F-4）
        _prof(spec, 'producer', 'calib_total', _t_cal)
    except BaseException:
        logger.debug("校准相位异常进入清理", exc_info=True)
        try:
            ocr_session.finish()
        except BaseException:
            logger.debug("ocr_session.finish 清理忽略异常", exc_info=True)
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛（资源由进程回收）
        raise

    segs: list = []
    rep_crops: dict = {}
    seg_idx = 0

    def _emit_ocr(seg, r_frame, r_crop, r_dev, _r_gray, frac) -> None:
        nonlocal seg_idx
        _t_push = time.perf_counter()
        _put_ocr((seg_idx, r_frame, r_crop, r_dev, frac))
        _prof(spec, 'producer', 'q_put_block', _t_push)
        if spec.keep_crops:
            rep_crops[r_frame] = r_crop
        seg_idx += 1

    t0 = time.perf_counter()
    try:
        _segment_frames(
            spec, frames,
            _frame_stream(spec, frames, vr, calib, th, with_dev=_with_dev),
            emit=_emit_ocr, segs=segs)
    finally:
        _t_consume_end = time.perf_counter()
        res.timing['decode'] = _t_consume_end - t0
        _prof(spec, 'producer', 'consumer_total', t0)
        ocr_session.finish()
        res.timing['ocr_tail'] = time.perf_counter() - _t_consume_end
        try:
            vr.close()   # hybrid 探针/资源释放：显式停止生产者线程
        except Exception:
            pass  # 清理路径：close 失败无需上抛（资源由进程回收）
    if ocr_err:
        # C4：补"OCR worker 失败"上下文并保留原始异常链
        raise RuntimeError(f"OCR worker 失败: {ocr_err[0]!r}") from ocr_err[0]
    res.timing['ocr'] = ocr_wall[0]
    res.frames = frames
    res.segs = segs
    res.n_segments = len(segs)
    res.crops = rep_crops
    res.texts = [results[i][0] for i in range(seg_idx)]
    res.confs = [results[i][1] for i in range(seg_idx)]
    res.rep_frames = [results[i][2] for i in range(seg_idx)]
    del vr
    return res
