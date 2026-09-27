"""GPU 后端（v2 §5/pipeline/gpu_backend，S3-3c 自 _gpu_pipeline.py 迁入；
P1 起为 GPU lane 策略）。

452 行驱动的显式契约化（与 host_backend 同构）：算法与编排代码只读
GpuRunSpec 声明字段；判定回调（merge/autocrop/裁切区间）显式注入；
运行态只写 GpuRunResult 与 fps 缓存盒。

B4/B5：autocropper 与 y_pool 从"事后赋值、worker/GC 无锁读"改为
**构造注入**——会话启动前/生产者线程启动前构建完毕（线程可见性由
Thread.start() 的 happens-before 保证）。
B6（__del__ 内 cudaFree）**未随本迁移改动**：池帧归还仍走 __del__ →
池 free-list（溢出才 cudaFree），显式化随 S4 设备侧模块拆分落地。

P1（宿主/GPU 实现统一轮）：生命周期（open→fps→会话→校准→消费→收尾）
上收到 _driver.run_segment_pipeline 单出处，本模块只剩 GPU 侧策略
（_GpuLane）：设备侧校准、生产者线程+队列帧流、DeviceRef emit、
sim_pair 合并判定，以及形状不符回退宿主（C10）。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Callable

import numpy as np

from video_ocr_engine.domain.segmentation import dense_gate_hit, similar_decision
from ..domain.metrics import NULL_METRICS
from ..gpu.frame_ref import DeviceRef

logger = logging.getLogger(__name__)

_PRODUCER_JOIN_TIMEOUT = 5.0
_KEEP_CROPS_WINDOW = 16   # S6-b：keep_crops 的 D2H 并批窗口（段数计）


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
    merge_dense_gate: int           # 0=关；>0=差异图 win3 阈值（稠密簇门）
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
    # S6-0：注入的指标记录器（§8.6 N-2；off 档为 NullMetrics 单例）
    metrics: object = NULL_METRICS


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
    fork_stats: dict | None = None   # fork 遥测穿透（hybrid 解码器才有）
    fell_back_to_host: bool = False
    fallback_engines: list | None = None    # 回退宿主时透传（避免二次 acquire）
    fallback_vr: object = None              # C10：复用已打开的 reader

    def as_tuple(self) -> tuple:
        return (self.frames, self.segs, self.texts, self.confs, self.rep_frames)


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
            return [d.with_crop(0, d.w) for d in devs]
        crop_devs = []
        yfs = []
        if self._ctx.y_pool is not None:
            for d in devs:
                yf = self._ctx.y_pool.acquire()
                yfs.append(yf)
                crop_devs.append(yf.ptr)
            an.luma_into_batch(
                [d.ptr for d in devs], crop_devs,
                self._ctx.src_h, self._ctx.src_w,
                self._spec.color_range != 1, stream=an._stream_c)
        else:
            crop_devs = [d.ptr for d in devs]
        rows = an.content_range_batch(
            crop_devs, self._ctx.src_h, self._ctx.src_w,
            self._spec.bin_thresh_ref[0], stream=an._stream_c)
        outs = []
        for d, yf, r in zip(devs, yfs or [None] * len(devs), rows):
            owner = yf if yf is not None else d.owner
            ptr = yf.ptr if yf is not None else d.ptr
            if int(r[0]) <= int(r[1]):
                rng = self._spec.content_range_to_crop(
                    int(r[0]), int(r[1]), self._ctx.src_w)
                xoff, cropw = (rng if rng is not None
                               else (0, self._ctx.src_w))
            else:
                xoff, cropw = 0, self._ctx.src_w
            outs.append(DeviceRef(ptr=ptr, h=d.h, w=d.w, owner=owner,
                                  x_off=xoff, crop_w=cropw))
        return outs


class _GpuLane:
    """GPU 全驻留 lane：设备侧校准/帧流/emit/合并判定（_driver 的策略侧）。

    payload 形状 (fi, dev, sharp)——emit 取 [0]/[1]/[2]，
    similar 判定取 [1]（DeviceRef）。
    """

    progress_verb = 'GPU分段'
    debug_tag = 'GB'

    def __init__(self, spec: GpuRunSpec) -> None:
        from ..gpu.streams import _gpu_release_partial   # S9-5 设备侧机制（S4 拆分后规范位置）
        self._spec = spec
        self._release = _gpu_release_partial
        self._ctx = None        # _GpuRunCtx（after_open 前置装配）
        self._hooks = None      # 设备函数显式依赖（gpu/device.DevHooks）
        self._prepare = None    # _gpu_prepare_calibration
        self._stream_fn = None  # (nvdec, cpu) 两帧流
        self._YFramePool = None
        self._on_gpu = False
        self._yuv = spec.yuv_output
        self._limited = spec.color_range != 1
        self._th = 0
        # 生产者线程 / 队列 / 错误槽
        self._producer = None
        self._producer_q: Queue | None = None
        self._stream = None
        self.producer_stop = threading.Event()
        self.producer_err: list = []
        # 消费侧运行态
        self._session = None
        self._put_ocr = None
        self._raw_ready = None
        self._rep_crops: dict = {}
        self._pending_crops: list = []          # [(r_frame, dev, prefer_device)]
        self._seg_idx = 0
        self._merge_hits = 0        # PI-15：逐段计数改局部累加，run 末一次上报

    @property
    def crops(self) -> dict:
        return self._rep_crops

    # ── 生命周期挂钩 ──────────────────────────────────────────────
    def after_open(self, vr) -> None:
        from ..gpu.streams import (DevHooks, _GpuRunCtx,
                                   _gpu_frame_stream_cpu,
                                   _gpu_frame_stream_nvdec,
                                   _gpu_prepare_calibration)
        spec = self._spec
        self._ctx = _GpuRunCtx()
        # 设备侧协作函数的显式依赖（P1 前为 _SpecView ex 视图桥接）
        self._hooks = DevHooks(
            limited=spec.color_range != 1, batch_luma=spec.batch_luma,
            bin_ref=spec.bin_thresh_ref, on_bin=spec.on_bin_thresh,
            prof=spec.prof_end)
        self._prepare = _gpu_prepare_calibration
        self._stream_fn = (_gpu_frame_stream_nvdec, _gpu_frame_stream_cpu)
        # F-6：on_gpu 必须在 open_vr **之后**判定——backend 标签由 open 写入
        # （旧代码同序；门面若在 spec 构造期求值，B1 重置后的空标签会使其
        # 恒为 False → 误走 CPU 解码分支，stride>1 时 rep_frame 帧号漂移）。
        _label = spec.backend_label()
        self._on_gpu = (_label == 'decord/GPU'
                        or (_label == 'decord/hybrid' and spec.ocr_on_gpu()))

    def calibrate(self, spec, vr, frames, ocr_session):
        x1, y1, x2, y2 = spec.roi
        ok, th = self._prepare(
            self._hooks, self._ctx, vr, frames, on_gpu=self._on_gpu,
            yuv=self._yuv, roi=(x1, y1, x2 + 1, y2 + 1))
        self._th = th
        return ok, th

    def after_calibrate(self, th, ocr_session) -> None:
        spec = self._spec
        self._session = ocr_session
        self._put_ocr = ocr_session.put
        spec.bin_thresh_ref[0] = th
        if spec.on_bin_thresh is not None:
            spec.on_bin_thresh(th)
        # B5（真装配点）：y_pool 依赖校准产出的 src_h/src_w；生产者未启动，
        # 此处赋值先行于一切并发读者。
        from ..gpu.pools import _YFramePool
        self._ctx.y_pool = (_YFramePool(self._ctx.src_h * self._ctx.src_w)
                            if (self._yuv and self._on_gpu) else None)
        ocr_session.autocropper = _DeferredAutocropper(self._ctx, spec,
                                                        ocr_session)

    def abort(self, ocr_session) -> None:
        # Cleanup handles are initialized before any calibration/setup can fail.
        try:
            ocr_session.finish()
        except BaseException:
            logger.debug("ocr_session.finish 清理忽略异常", exc_info=True)
        self._release(self._ctx)

    def fallback(self, res, ocr_session, ocr_engines, vr):
        spec = self._spec
        res.fell_back_to_host = True
        res.fallback_engines = ocr_engines
        res.fallback_vr = vr
        res.fps = spec.fps_box[0]
        try:
            ocr_session.finish()   # 空会话收尾：worker 归还引擎
        except BaseException:
            logger.debug("ocr_session.finish 清理忽略异常", exc_info=True)
        self._release(self._ctx)
        return res

    def release(self) -> None:
        # C5：释放本次 extract 的临时设备缓冲；OCR 引擎缓冲归引擎池。
        self._release(self._ctx)

    def report_counters(self) -> None:
        if self._spec.metrics.enabled:
            self._spec.metrics.counter('segment.merges', self._merge_hits)

    # ── 帧流：生产者线程 + 队列，包成生成器交统一驱动消费 ──────────
    def frame_items(self, spec, vr, frames, ocr_session):
        self._raw_ready = ocr_session.raw_ready
        self._producer_q = Queue(maxsize=max(8, spec.buffer_size))
        x1, y1, x2, y2 = spec.roi
        self._stream = self._stream_fn[0 if self._on_gpu else 1](
            self._hooks, self._ctx, vr, frames, yuv=self._yuv,
            roi=(x1, y1, x2 + 1, y2 + 1), th=self._th)
        self._producer = threading.Thread(target=self._producer_loop,
                                          daemon=True)
        self._producer.start()
        while True:
            try:
                item = self._producer_q.get(timeout=0.2)
            except Empty:
                # C7：解码停滞/等待期间保持取消响应。
                spec.cancel()
                continue
            if item is None:
                break
            if self.producer_err:
                raise RuntimeError(
                    f"GPU 解码生产者失败: {self.producer_err[0]!r}"
                ) from self.producer_err[0]
            fi, dev, sharp, cluster = item
            yield fi, sharp, dict(payload=(fi, dev, sharp),
                                  cluster=float(cluster))

    def _put_q(self, item) -> bool:
        # P2c 覆盖：生产者侧背压此前无对应物（宿主路径 q_put_block 一直
        # 有）——FULL 重试等待正是"OCR 消费不动生产者"的直接证据。
        spec = self._spec
        _t_put = time.perf_counter()
        try:
            while not self.producer_stop.is_set():
                try:
                    self._producer_q.put(item, timeout=0.2)
                    return True
                except Full:
                    continue
            return False
        finally:
            if spec.prof_end is not None:
                spec.prof_end('producer', 'q_put_block', _t_put)

    def _producer_loop(self) -> None:
        try:
            for item in self._stream:
                if not self._put_q(item):
                    return
        except BaseException as e:  # noqa: BLE001
            # BaseException 而非 Exception（2026-09-19 审查轮，与
            # gpu/device.py 排空线程对齐）：逃逸的 BaseException 会让
            # producer_err 恒空 → 消费循环见哨兵即正常收尾 → run 报
            # 成功但帧数静默截断。
            self.producer_err.append(e)
        finally:
            self._put_q(None)

    def after_stream(self) -> None:
        self._producer.join()
        if self.producer_err:
            raise RuntimeError(
                f"GPU 解码生产者失败: {self.producer_err[0]!r}"
            ) from self.producer_err[0]

    def stop_consume(self) -> None:
        self.producer_stop.set()   # C6：任何退出路径都叫停 producer
        self._resolve_keep_crops()   # S6-b：尾窗收口（rep_crops 必须完整）
        if self._producer is not None:
            try:
                self._producer.join(_PRODUCER_JOIN_TIMEOUT)
            except BaseException:
                logger.debug("producer.join 清理忽略异常", exc_info=True)

    # ── 代表帧交付 / 合并判定（消费线程内）────────────────────────
    def emit(self, seg, rep, frac) -> None:
        self._emit_ocr(self._seg_idx, rep[0], rep[1], frac, rep[2])
        self._seg_idx += 1

    def _d2h_rep(self, dev, *, prefer_device=False):
        """代表帧 → 宿主：NVDEC = D2H；CPU 解码默认宿主切片直取。"""
        from cuda.bindings import runtime as cudart
        ctx = self._ctx
        hc = getattr(dev.owner, 'host_crop', None)
        if hc is not None and not prefer_device:
            h = hc()
            if h is not None:
                return np.array(h)
        arr = np.empty((dev.h, dev.w), dtype=np.uint8)
        # 消费流上异步 D2H + 同步消费流：不走 NULL 流（避免耦合生产者）。
        cudart.cudaMemcpyAsync(
            arr.ctypes.data, int(dev.ptr), dev.h * dev.w,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ctx.analyzer._stream_c)
        cudart.cudaStreamSynchronize(ctx.analyzer._stream_c)
        _MET = self._spec.metrics
        if _MET.enabled:
            _MET.counter('emit.d2h_calls')
            _MET.counter('emit.d2h_bytes', dev.h * dev.w)
            _MET.counter('emit.syncs')
        return arr

    def _resolve_keep_crops(self) -> None:
        # ── S6-b：keep_crops 的 D2H 并批（v1 每段一次 async D2H + 一次流同步，
        # 3000 帧/1083 段 = 1083 次同步全落在消费线程上）。改为有界窗口收集 +
        # 每窗一次流同步；窗口 = 16 段，避免把过多 decord 批 owner 钉住（PI-5）。
        if not self._pending_crops:
            return
        from cuda.bindings import runtime as cudart
        spec = self._spec
        ctx = self._ctx
        _MET = spec.metrics
        _t_d2h = time.perf_counter()
        arrs: list = []
        for _rf, _dev, _prefer in self._pending_crops:
            _hc = getattr(_dev.owner, 'host_crop', None)
            _h = _hc() if (_hc is not None and not _prefer) else None
            if _h is not None:
                arrs.append(np.array(_h))
                continue
            _arr = np.empty((_dev.h, _dev.w), dtype=np.uint8)
            cudart.cudaMemcpyAsync(
                _arr.ctypes.data, int(_dev.ptr), _dev.h * _dev.w,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                ctx.analyzer._stream_c)
            arrs.append(_arr)
        cudart.cudaStreamSynchronize(ctx.analyzer._stream_c)   # 每窗一次
        for (_rf, _dev, _prefer), _arr in zip(self._pending_crops, arrs):
            self._rep_crops[_rf] = _arr
        if _MET.enabled:
            # PI-15：逐段计数由调用点搬到这里（每窗一次），少 N 次 Python 上报
            _MET.counter('emit.keep_crops_batched', len(self._pending_crops))
            _MET.counter('emit.keep_crops_d2h', len(self._pending_crops))
        if spec.prof_end is not None:
            spec.prof_end('producer', 'emit_d2h', _t_d2h)
        self._pending_crops.clear()

    def _autocrop_device(self, gray_ptr, sharp):
        """GPU 直通裁切：col_ink 判「有墨迹列范围」+ 宿主同一余量规则。"""
        spec = self._spec
        ctx = self._ctx
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

    def similar(self, a, b) -> bool:
        """merge_similar 判定：GPU sim_pair（整数精确）。"""
        return self._similar_device(a[1], b[1])

    def _similar_device(self, a_dev, b_dev) -> bool:
        spec = self._spec
        ctx = self._ctx
        if not (spec.merge_similar and a_dev is not None
                and b_dev is not None):
            return False
        use_bin = 1 if spec.merge_effective_mode() == 'binary' else 0
        ya = yb = None
        if self._yuv and self._on_gpu:
            ya = ctx.y_pool.acquire()
            yb = ctx.y_pool.acquire()
            ctx.analyzer.luma_into(int(a_dev.ptr), int(ya.ptr), ctx.src_h,
                                   ctx.src_w, self._limited,
                                   stream=ctx.analyzer._stream_c)
            ctx.analyzer.luma_into(int(b_dev.ptr), int(yb.ptr), ctx.src_h,
                                   ctx.src_w, self._limited,
                                   stream=ctx.analyzer._stream_c)
            ap, bp = ya.ptr, yb.ptr
        else:
            ap, bp = int(a_dev.ptr), int(b_dev.ptr)
        try:
            _t_cmp = time.perf_counter()
            mad, chg, win3 = ctx.analyzer.compare_pair(
                ap, bp, ctx.src_h, ctx.src_w, spec.bin_thresh_ref[0],
                use_bin, stream=ctx.analyzer._stream_c)
            if spec.prof_end is not None:
                # S6-b：合并判定的提交开销（每段边界一次 grid=1 launch+sync，
                # 消费线程内串行——批量化收益的判据）
                spec.prof_end('producer', 'merge_pair', _t_cmp)
        finally:
            # S4：合并判定的池帧显式归还（不再依赖 GC 时机；payload 携带的
            # 池帧仍由 __del__ 入列回收——owner 生命周期跨线程，无法在此收口）
            if ya is not None:
                ctx.y_pool.recycle(ya)
            if yb is not None:
                ctx.y_pool.recycle(yb)
        n = ctx.src_h * ctx.src_w
        mean = 255.0 * mad / n if use_bin else mad / n
        # 稠密簇门：win3 由 sim_pair 设备侧算出，与宿主 _cluster_win3 逐位
        # 一致（同一 3×3 邻域定义）；两侧共用 dense_gate_hit + similar_decision，
        # 否则 C-32「GPU 为宿主逐位镜像」的契约在阈值处破。
        _dec = similar_decision(mean, chg,
                                spec.merge_similar_threshold,
                                spec.merge_max_changed_pixels,
                                dense=dense_gate_hit(win3,
                                                     spec.merge_dense_gate))
        if _dec:
            self._merge_hits += 1       # PI-15：逐段计数改为局部累加，run 末一次上报
        return _dec

    def _emit_ocr(self, idx, r_frame, r_dev, frac, r_sharp) -> None:
        from cuda.bindings import runtime as cudart
        spec = self._spec
        ctx = self._ctx
        yuv = self._yuv
        on_gpu = self._on_gpu
        _t_push = time.perf_counter()
        _raw = self._raw_ready[0] and r_dev is not None
        crop_h = None
        dev_ocr = None
        if _raw:
            if getattr(self._session, 'autocropper', None) is not None:
                # 零拷贝 + 双推迟：emit 只剩构元+入队。
                if (spec.ocr_autocrop and ctx.src_w > 8 and r_sharp >= 3.0):
                    dev_ocr = DeviceRef(ptr=r_dev.ptr, h=ctx.src_h,
                                        w=ctx.src_w, owner=r_dev.owner,
                                        sharp=r_sharp)      # 延后裁切标记
                else:
                    dev_ocr = DeviceRef(ptr=r_dev.ptr, h=ctx.src_h,
                                        w=ctx.src_w, owner=r_dev.owner,
                                        x_off=0, crop_w=ctx.src_w)
            else:
                if yuv and on_gpu:
                    yf = ctx.y_pool.acquire()
                    ctx.analyzer.luma_into(int(r_dev.ptr), int(yf.ptr),
                                           ctx.src_h, ctx.src_w,
                                           self._limited,
                                           stream=ctx.analyzer._stream_c)
                    cudart.cudaStreamSynchronize(ctx.analyzer._stream_c)
                    base = DeviceRef(ptr=yf.ptr, h=ctx.src_h, w=ctx.src_w,
                                     owner=yf)
                else:
                    base = r_dev
                xoff, cropw = 0, ctx.src_w
                _t_ac = time.perf_counter()
                rng = self._autocrop_device(base.ptr, r_sharp)
                if rng is not None:
                    xoff, cropw = rng
                if spec.prof_end is not None:
                    spec.prof_end('producer', 'emit_autocrop', _t_ac)
                dev_ocr = base.with_crop(xoff, cropw)
        else:
            crop_h = self._d2h_rep(r_dev)
        if spec.keep_crops:
            if crop_h is not None:
                # 该段已在走宿主路径，代表帧就是宿主数组：直接落盘，无需 D2H
                self._rep_crops[r_frame] = crop_h
            else:
                # S6-b：设备代表帧的 D2H 进窗口并批（见 _resolve_keep_crops）。
                # 实测（交错 A/B，4 次独立进程）：冷启动 h264-gpu −2.45%
                # （符号一致），其余配置中性——收益集中在"代表帧留显存"的
                # raw 直通路径。
                self._pending_crops.append(
                    (r_frame, r_dev, dev_ocr is not None and not yuv))
                if len(self._pending_crops) >= _KEEP_CROPS_WINDOW:
                    self._resolve_keep_crops()
        if dev_ocr is not None and not yuv:
            drop_host = getattr(dev_ocr.owner, 'drop_host', None)
            if drop_host is not None:
                drop_host()
        from .ocr_stage import SegmentTask
        self._put_ocr(SegmentTask(idx, r_frame, crop_h, dev_ocr, frac))
        if spec.prof_end is not None:
            spec.prof_end('producer', 'emit_put', _t_push)


def run_gpu_pipeline(spec: GpuRunSpec, ocr_engines=None) -> GpuRunResult:
    """GPU 驱动门入口（原 _GpuPipelineMixin._run_pipelined_gpu 主体）。

    生命周期由 _driver.run_segment_pipeline 承载（P1）；本模块提供
    GPU lane 策略：设备侧校准 → 生产者线程（解码+GPU analyze 与主线程
    分段/OCR 重叠）→ 消费循环 → 清理；形状不符回退宿主（C10）。
    """
    from ._driver import run_segment_pipeline
    return run_segment_pipeline(spec, GpuRunResult(), _GpuLane(spec),
                                ocr_engines)
