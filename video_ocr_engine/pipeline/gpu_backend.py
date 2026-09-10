"""GPU 后端（v2 §5/pipeline/gpu_backend，S3-3c 自 _gpu_pipeline.py 迁入）。

452 行驱动的显式契约化（与 host_backend 同构）：算法与编排代码只读
GpuRunSpec 声明字段；判定回调（merge/autocrop/裁切区间）显式注入；
运行态只写 GpuRunResult 与 fps 缓存盒。

B4/B5 随迁修复：autocropper 与 y_pool 从"事后赋值、worker/GC 无锁读"
改为**构造注入**——会话启动前/生产者线程启动前构建完毕（线程可见性由
Thread.start() 的 happens-before 保证）。
B6（__del__ 内 cudaFree）**未随本迁移改动**：池帧归还仍走 __del__ →
池 free-list（溢出才 cudaFree），显式化随 S4 设备侧模块拆分落地。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Callable

import numpy as np

from segmentation import SegmentStateMachine, similar_decision

logger = logging.getLogger(__name__)

_PRODUCER_JOIN_TIMEOUT = 5.0


class _SpecView:
    """设备侧协作函数的 ex 适配视图（迁移期桥接）。

    _gpu_prepare_calibration / _gpu_frame_stream_* 只读 ex 的 4 个属性
    （_color_range / _batch_luma / _bin_thresh / _prof_end）。本视图把它们
    指向 spec 注入的回调——设备函数零改动，S4 拆分时随迁消除。
    """

    def __init__(self, spec: GpuRunSpec, batch_luma: Callable) -> None:
        self._spec = spec
        self._batch_luma = batch_luma

    @property
    def _color_range(self):
        return self._spec.color_range

    @property
    def _bin_thresh(self):
        return self._spec.bin_thresh_ref[0]

    @_bin_thresh.setter
    def _bin_thresh(self, th):
        self._spec.bin_thresh_ref[0] = th
        if self._spec.on_bin_thresh is not None:
            self._spec.on_bin_thresh(th)

    def _batch_luma(self, crops):
        return self._batch_luma(crops)

    def _prof_end(self, group, key, t0):
        if self._spec.prof_end is not None:
            self._spec.prof_end(group, key, t0)


@dataclass
class GpuRunSpec:
    """GPU 驱动的全量输入契约。"""
    # 帧区间 / ROI / 采样 / 输出语义
    frame_start: int
    frame_end: int | None
    sample_stride: int
    roi: tuple
    buffer_size: int
    C: float
    merge_similar: bool
    merge_similar_threshold: float
    merge_max_changed_pixels: int
    keep_crops: bool
    yuv_output: bool
    color_range: int
    ocr_autocrop: bool
    bin_thresh_ref: list            # [th] 单元素盒：校准写、判定回调读
    backend_label: Callable         # () -> str（open_vr 之后读，F-6）
    # 判定/裁切回调（实现唯一出处 segmentation）
    ocr_on_gpu: Callable            # () -> bool（hybrid 引擎是否走 GPU OCR）
    merge_effective_mode: Callable  # () -> str（binary/''）
    content_range_to_crop: Callable  # (first, last, w) -> (x_off, crop_w)|None
    # 门面回调
    open_vr: Callable
    start_ocr_session: Callable
    batch_luma: Callable               # (B,H,W[,C]) -> (B,h,w)（CPU 解码分支）
    progress: Callable = lambda m, p: None
    cancel: Callable = lambda: None
    prof_end: Callable | None = None
    on_bin_thresh: Callable | None = None   # (th) -> None（F-4 同类：即时回写）
    # 可变盒（门面持有缓存）
    fps_box: list = field(default_factory=lambda: [None])


@dataclass
class GpuRunResult:
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
    fell_back_to_host: bool = False
    fallback_engines: list | None = None    # 回退宿主时透传（避免二次 acquire）
    fallback_vr: object = None              # C10：复用已打开的 reader

    def as_tuple(self) -> tuple:
        return (self.frames, self.segs, self.texts, self.confs, self.rep_frames)


def run_gpu_pipeline(spec: GpuRunSpec, ocr_engines=None) -> GpuRunResult:
    """GPU 驱动门入口（原 _GpuPipelineMixin._run_pipelined_gpu 主体）。

    资源阶段：open_vr → 帧区间 → hybrid_begin → 会话启动（构造注入
    autocropper，B4）→ 校准（失败或形状不符可回退宿主）→ 生产者线程
    （解码+GPU analyze 与主线程分段/OCR 重叠）→ 消费循环 → 清理。
    """
    from .._gpu_pipeline import (   # 迁移期：设备侧仍驻留原模块（S4 拆分）
        _GpuRunCtx, _YFramePool, _gpu_frame_stream_cpu,
        _gpu_frame_stream_nvdec, _gpu_prepare_calibration,
        _gpu_release_partial)
    from .._helpers import _decode_progress_pct

    res = GpuRunResult()
    # Cleanup handles are initialized before any calibration/setup can fail.
    ocr_session = None
    producer = None
    producer_stop = threading.Event()
    _t_open = time.perf_counter()
    vr = spec.open_vr()
    # F-6：on_gpu 必须在 open_vr **之后**判定——backend 标签由 open 写入
    # （旧代码同序；门面若在 spec 构造期求值，B1 重置后的空标签会使其
    # 恒为 False → 误走 CPU 解码分支，stride>1 时 rep_frame 帧号漂移）。
    _label = spec.backend_label()
    on_gpu = (_label == 'decord/GPU'
              or (_label == 'decord/hybrid' and spec.ocr_on_gpu()))
    from ._run_common import begin_reading, compute_frames, ensure_fps
    ensure_fps(spec, vr)
    res.fps = spec.fps_box[0]
    x1, y1, x2, y2 = spec.roi
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
    try:
        begin_reading(vr, spec, frames)
    except BaseException:
        logger.debug("hybrid_begin 失败进入回退清理", exc_info=True)
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛
        raise
    if spec.prof_end is not None:
        spec.prof_end('producer', 'open_and_fps', _t_open)
    # OCR 会话提前到校准前启动（引擎构建与校准并行重叠）。
    try:
        ocr_session = spec.start_ocr_session(ocr_engines)
    except BaseException:
        logger.debug("OCR 会话启动失败进入清理", exc_info=True)
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛
        raise
    results = ocr_session.results
    ocr_err = ocr_session.err
    ocr_wall = ocr_session.wall
    _put_ocr = ocr_session.put
    # ── B4/B5：autocropper 与 y_pool 均为构造期装配 ──
    ctx = _GpuRunCtx()
    yuv = spec.yuv_output
    limited = spec.color_range != 1
    # 设备侧协作函数的 ex 适配视图（迁移期桥接，S4 拆分时消除）
    exv = _SpecView(spec, spec.batch_luma)

    class _DeferredAutocropper:
        def __init__(self, ctx_ref, spec_ref, session_ref):
            self._ctx = ctx_ref
            self._spec = spec_ref
            self._session = session_ref

        def process(self, devs):
            """5 元组列表 → 6 元组列表：NVDEC+yuv 先批量提取 Y（池帧），
            再批量 col_ink 得裁切区间。CPU 解码分支设备侧恒为灰度，
            直接进入 col_ink。"""
            an = self._ctx.analyzer
            if an is None:
                return [(d[0], d[1], d[2], d[3], 0, d[3]) for d in devs]
            crop_devs = []
            yfs = []
            if self._ctx.y_pool is not None:
                for d in devs:
                    yf = self._ctx.y_pool.acquire()
                    yfs.append(yf)
                    crop_devs.append(yf.ptr)
                an.luma_into_batch(
                    [d[1] for d in devs], crop_devs,
                    self._ctx.src_h, self._ctx.src_w,
                    self._spec.color_range != 1, stream=an._stream_c)
            else:
                crop_devs = [d[1] for d in devs]
            rows = an.content_range_batch(
                crop_devs, self._ctx.src_h, self._ctx.src_w,
                self._spec.bin_thresh_ref[0], stream=an._stream_c)
            outs = []
            for d, yf, r in zip(devs, yfs or [None] * len(devs), rows):
                owner = yf if yf is not None else d[0]
                ptr = yf.ptr if yf is not None else d[1]
                if int(r[0]) <= int(r[1]):
                    rng = self._spec.content_range_to_crop(
                        int(r[0]), int(r[1]), self._ctx.src_w)
                    xoff, cropw = (rng if rng is not None
                                   else (0, self._ctx.src_w))
                else:
                    xoff, cropw = 0, self._ctx.src_w
                outs.append((owner, ptr, d[2], d[3], xoff, cropw))
            return outs

    def _cleanup_partial():
        nonlocal ocr_session
        if ocr_session is not None:
            try:
                ocr_session.finish()
            except BaseException:
                logger.debug("ocr_session.finish 清理忽略异常", exc_info=True)
            ocr_session = None
        _gpu_release_partial(ctx)

    # 校准（构建 analyzer / 首批直方图 / Otsu 阈值）
    try:
        _calib_ok, _th = _gpu_prepare_calibration(
            exv, ctx, vr, frames, on_gpu=on_gpu, yuv=yuv,
            roi=(x1, y1, x2 + 1, y2 + 1))
    except BaseException:
        logger.debug("GPU 管线异常触发 _cleanup_partial", exc_info=True)
        _cleanup_partial()
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛
        raise
    if not _calib_ok:
        # 形状不符等：回退宿主（C10 语义）——会话收尾（worker 归还引擎）、
        # 释放设备侧临时缓冲，但 **reader 不 close**：宿主路径复用已打开的
        # reader（get_batch 随机访问无消费状态，免二次打开/解码器悬挂）。
        res.fell_back_to_host = True
        res.fallback_engines = ocr_engines
        res.fallback_vr = vr
        res.fps = spec.fps_box[0]
        try:
            ocr_session.finish()   # 空会话收尾：worker 归还引擎
        except BaseException:
            logger.debug("ocr_session.finish 清理忽略异常", exc_info=True)
        _gpu_release_partial(ctx)
        return res
    res.bin_thresh = _th
    spec.bin_thresh_ref[0] = _th
    if spec.on_bin_thresh is not None:
        spec.on_bin_thresh(_th)
    if spec.prof_end is not None:
        spec.prof_end('producer', 'gpu_calib_total', _t_open)
    # B5（真装配点）：y_pool 依赖校准产出的 src_h/src_w；生产者未启动，
    # 此处赋值先行于一切并发读者。
    ctx.y_pool = (_YFramePool(ctx.src_h * ctx.src_w)
                  if (yuv and on_gpu) else None)
    ocr_session.autocropper = _DeferredAutocropper(ctx, spec, ocr_session)

    if on_gpu:
        frame_stream = _gpu_frame_stream_nvdec(
            exv, ctx, vr, frames, yuv=yuv,
            roi=(x1, y1, x2 + 1, y2 + 1), th=_th)
    else:
        frame_stream = _gpu_frame_stream_cpu(
            exv, ctx, vr, frames, yuv=yuv,
            roi=(x1, y1, x2 + 1, y2 + 1), th=_th)

    producer_q: Queue = Queue(maxsize=max(8, spec.buffer_size))
    producer_err: list = []

    def _put_q(item) -> bool:
        while not producer_stop.is_set():
            try:
                producer_q.put(item, timeout=0.2)
                return True
            except Full:
                continue
        return False

    def _producer() -> None:
        try:
            for item in frame_stream:
                if not _put_q(item):
                    return
        except Exception as e:  # noqa: BLE001
            producer_err.append(e)
        finally:
            _put_q(None)

    producer = threading.Thread(target=_producer, daemon=True)
    producer.start()

    rep_crops: dict = {}
    seg_idx = 0
    k = 0
    t0 = time.perf_counter()
    from cuda.bindings import runtime as cudart
    raw_ready_ref = ocr_session.raw_ready

    def _d2h_rep(dev, *, prefer_device=False):
        """代表帧 → 宿主：NVDEC = D2H；CPU 解码默认宿主切片直取。"""
        hc = getattr(dev[0], 'host_crop', None)
        if hc is not None and not prefer_device:
            h = hc()
            if h is not None:
                return np.array(h)
        arr = np.empty((dev[2], dev[3]), dtype=np.uint8)
        # 消费流上异步 D2H + 同步消费流：不走 NULL 流（避免耦合生产者）。
        cudart.cudaMemcpyAsync(
            arr.ctypes.data, int(dev[1]), dev[2] * dev[3],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ctx.analyzer._stream_c)
        cudart.cudaStreamSynchronize(ctx.analyzer._stream_c)
        return arr

    def _autocrop_device(gray_ptr, sharp):
        """GPU 直通裁切：col_ink 判「有墨迹列范围」+ 宿主同一余量规则。"""
        if not spec.ocr_autocrop:
            return None
        if ctx.src_w <= 8 or sharp < 3.0:
            return None
        rng = ctx.analyzer.content_range(int(gray_ptr), ctx.src_h,
                                         ctx.src_w, spec.bin_thresh_ref[0],
                                         stream=ctx.analyzer._stream_c)
        if rng is None:
            return None
        return spec.content_range_to_crop(rng[0], rng[1], ctx.src_w)

    def _similar_device(a_dev, b_dev) -> bool:
        """merge_similar 判定：GPU sim_pair（整数精确）。"""
        if not (spec.merge_similar and a_dev is not None
                and b_dev is not None):
            return False
        use_bin = 1 if spec.merge_effective_mode() == 'binary' else 0
        ya = yb = None
        if yuv and on_gpu:
            ya = ctx.y_pool.acquire()
            yb = ctx.y_pool.acquire()
            ctx.analyzer.luma_into(int(a_dev[1]), int(ya.ptr), ctx.src_h,
                                   ctx.src_w, limited,
                                   stream=ctx.analyzer._stream_c)
            ctx.analyzer.luma_into(int(b_dev[1]), int(yb.ptr), ctx.src_h,
                                   ctx.src_w, limited,
                                   stream=ctx.analyzer._stream_c)
            ap, bp = ya.ptr, yb.ptr
        else:
            ap, bp = int(a_dev[1]), int(b_dev[1])
        try:
            mad, chg = ctx.analyzer.compare_pair(
                ap, bp, ctx.src_h, ctx.src_w, spec.bin_thresh_ref[0],
                use_bin, stream=ctx.analyzer._stream_c)
        finally:
            # S4：合并判定的池帧显式归还（不再依赖 GC 时机；payload 携带的
            # 池帧仍由 __del__ 入列回收——owner 生命周期跨线程，无法在此收口）
            if ya is not None:
                ctx.y_pool.recycle(ya)
            if yb is not None:
                ctx.y_pool.recycle(yb)
        n = ctx.src_h * ctx.src_w
        mean = 255.0 * mad / n if use_bin else mad / n
        return similar_decision(mean, chg,
                                spec.merge_similar_threshold,
                                spec.merge_max_changed_pixels)

    def _emit_ocr(idx, r_frame, r_dev, frac, r_sharp) -> None:
        _t_push = time.perf_counter()
        _raw = raw_ready_ref[0] and r_dev is not None
        crop_h = None
        dev_ocr = None
        if _raw:
            if getattr(ocr_session, 'autocropper', None) is not None:
                # 零拷贝 + 双推迟：emit 只剩构元+入队。
                if (spec.ocr_autocrop and ctx.src_w > 8 and r_sharp >= 3.0):
                    dev_ocr = (r_dev[0], r_dev[1], ctx.src_h,
                               ctx.src_w, r_sharp)
                else:
                    dev_ocr = (r_dev[0], r_dev[1], ctx.src_h,
                               ctx.src_w, 0, ctx.src_w)
            else:
                if yuv and on_gpu:
                    yf = ctx.y_pool.acquire()
                    ctx.analyzer.luma_into(int(r_dev[1]), int(yf.ptr),
                                           ctx.src_h, ctx.src_w, limited,
                                           stream=ctx.analyzer._stream_c)
                    cudart.cudaStreamSynchronize(ctx.analyzer._stream_c)
                    base = (yf, yf.ptr, ctx.src_h, ctx.src_w)
                else:
                    base = r_dev
                xoff, cropw = 0, ctx.src_w
                _t_ac = time.perf_counter()
                rng = _autocrop_device(base[1], r_sharp)
                if rng is not None:
                    xoff, cropw = rng
                if spec.prof_end is not None:
                    spec.prof_end('producer', 'emit_autocrop', _t_ac)
                dev_ocr = (base[0], base[1], ctx.src_h, ctx.src_w,
                           xoff, cropw)
        else:
            crop_h = _d2h_rep(r_dev)
        if spec.keep_crops:
            _t_d2h = time.perf_counter()
            rep_crops[r_frame] = (crop_h if crop_h is not None
                                  else _d2h_rep(
                                      r_dev,
                                      prefer_device=(dev_ocr is not None
                                                     and not yuv)))
            if spec.prof_end is not None:
                spec.prof_end('producer', 'emit_d2h', _t_d2h)
        if dev_ocr is not None and not yuv:
            drop_host = getattr(dev_ocr[0], 'drop_host', None)
            if drop_host is not None:
                drop_host()
        _put_ocr((idx, r_frame, crop_h, dev_ocr, frac))
        if spec.prof_end is not None:
            spec.prof_end('producer', 'emit_put', _t_push)

    def _on_emit(seg, rep, frac):
        nonlocal seg_idx
        _emit_ocr(seg_idx, rep[0], rep[1], frac, rep[2])
        seg_idx += 1

    machine = SegmentStateMachine(
        frames, C=spec.C,
        on_emit=_on_emit,
        on_similar=lambda a, b: _similar_device(a[1], b[1]),
        on_cancel=spec.cancel,
        on_progress=lambda kk, frac: spec.progress(
            f'[{spec.backend_label()}] GPU分段: {kk}/{len(frames)}',
            _decode_progress_pct(frac)),
        debug_tag='GB')

    try:
        while True:
            try:
                item = producer_q.get(timeout=0.2)
            except Empty:
                # C7：解码停滞/等待期间保持取消响应。
                spec.cancel()
                continue
            if item is None:
                break
            if producer_err:
                raise RuntimeError(
                    f"GPU 解码生产者失败: {producer_err[0]!r}"
                ) from producer_err[0]
            fi, dev, sharp, cluster = item
            _t_feed = time.perf_counter()
            machine.feed(k, fi, sharp, (fi, dev, sharp),
                         cluster=float(cluster))
            if spec.prof_end is not None:
                spec.prof_end('producer', 'consume_feed', _t_feed)
            k += 1
        producer.join()
        if producer_err:
            raise RuntimeError(
                f"GPU 解码生产者失败: {producer_err[0]!r}"
            ) from producer_err[0]
        machine.finish()
        segs = machine.segs
    finally:
        producer_stop.set()   # C6：任何退出路径都叫停 producer
        if producer is not None:
            try:
                producer.join(_PRODUCER_JOIN_TIMEOUT)
            except BaseException:
                logger.debug("producer.join 清理忽略异常", exc_info=True)
        _t_consume_end = time.perf_counter()
        res.timing['decode'] = _t_consume_end - t0
        try:
            ocr_session.finish()
        except BaseException:
            logger.debug("ocr_session.finish 清理忽略异常", exc_info=True)
        res.timing['ocr_tail'] = time.perf_counter() - _t_consume_end
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛
        # C5：释放本次 extract 的临时设备缓冲；OCR 引擎缓冲归引擎池。
        _gpu_release_partial(ctx)
    if ocr_err:
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
