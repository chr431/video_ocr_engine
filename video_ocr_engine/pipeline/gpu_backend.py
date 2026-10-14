"""GPU 后端：GPU lane 策略。

452 行驱动的显式契约化（与 host_backend 同构）：算法与编排代码只读
GpuRunSpec 声明字段；判定回调（merge/autocrop/裁切区间）显式注入；
运行态只写 GpuRunResult 与 fps 缓存盒。

autocropper 与 y_pool 为**构造注入**——会话启动前/生产者线程启动前
构建完毕（线程可见性由 Thread.start() 的 happens-before 保证）。池帧
归还走 __del__ → 池 free-list（溢出才 cudaFree）或显式归还路径。

生命周期（open→fps→会话→校准→消费→收尾）上收到
_driver.run_segment_pipeline 单出处，本模块只剩 GPU 侧策略
（_GpuLane）：设备侧校准、内联帧流生成器、DeviceRef emit、
sim_pair 合并判定（投机缓存命中 → 宿主标量），以及形状不符回退宿主。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from video_ocr_engine.domain.segmentation import dense_gate_hit, similar_decision
from ..domain.metrics import NULL_METRICS
from ..gpu.frame_ref import DeviceRef

logger = logging.getLogger(__name__)

_KEEP_CROPS_WINDOW = 16  # keep_crops 的 D2H 并批窗口（段数计）


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
    merge_dense_gate: int  # 0=关；>0=差异图 win3 阈值（稠密簇门）
    keep_crops: bool
    yuv_output: bool
    color_range: int
    ocr_autocrop: bool
    bin_thresh_ref: list  # [th] 单元素盒：校准写、判定回调读
    backend_label: Callable  # () -> str（open_vr 之后读）
    # 判定/裁切回调（实现唯一出处 segmentation）
    ocr_on_gpu: Callable  # () -> bool（hybrid 引擎是否走 GPU OCR）
    merge_effective_mode: Callable  # () -> str（binary/''）
    content_range_to_crop: Callable  # (first, last, w) -> (x_off, crop_w)|None
    # 门面回调
    open_vr: Callable
    start_ocr_session: Callable
    batch_luma: Callable  # (B,H,W[,C]) -> (B,h,w)（CPU 解码分支）
    progress: Callable = lambda m, p: None
    cancel: Callable = lambda: None
    prof_end: Callable | None = None
    on_bin_thresh: Callable | None = None  # (th) -> None（即时回写）
    # 可变盒（门面持有缓存）
    fps_box: list = field(default_factory=lambda: [None])
    # 注入的指标记录器（off 档为 NullMetrics 单例）
    metrics: Any = NULL_METRICS


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
    fork_stats: dict | None = None  # fork 遥测穿透（hybrid 解码器才有）
    fell_back_to_host: bool = False
    fallback_engines: list | None = None  # 回退宿主时透传（避免二次 acquire）
    fallback_vr: object = None  # 复用已打开的 reader

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
                [d.ptr for d in devs],
                crop_devs,
                self._ctx.src_h,
                self._ctx.src_w,
                self._spec.color_range != 1,
                stream=an._stream_c,
            )
        else:
            crop_devs = [d.ptr for d in devs]
        rows = an.content_range_batch(
            crop_devs,
            self._ctx.src_h,
            self._ctx.src_w,
            self._spec.bin_thresh_ref[0],
            stream=an._stream_c,
        )
        outs = []
        for d, yf, r in zip(devs, yfs or [None] * len(devs), rows):
            owner = yf if yf is not None else d.owner
            ptr = yf.ptr if yf is not None else d.ptr
            if int(r[0]) <= int(r[1]):
                rng = self._spec.content_range_to_crop(int(r[0]), int(r[1]), self._ctx.src_w)
                xoff, cropw = rng if rng is not None else (0, self._ctx.src_w)
            else:
                xoff, cropw = 0, self._ctx.src_w
            outs.append(DeviceRef(ptr=ptr, h=d.h, w=d.w, owner=owner, x_off=xoff, crop_w=cropw))
        return outs


class _GpuLane:
    """GPU 全驻留 lane：设备侧校准/帧流/emit/合并判定（_driver 的策略侧）。

    payload 形状 (fi, dev, sharp)——emit 取 [0]/[1]/[2]，
    similar 判定取 [1]（DeviceRef）。
    """

    progress_verb = "GPU分段"
    debug_tag = "GB"

    def __init__(self, spec: GpuRunSpec, warm=None) -> None:
        from ..gpu.streams import _gpu_release_partial  # 设备侧机制（规范位置 gpu/streams.py）

        self._spec = spec
        self._warm = warm  # 预热线句柄（run_gpu_pipeline 入口启动，calibrate 取回）
        self._release = _gpu_release_partial
        # 惰性装配字段（after_open/after_calibrate 前置装配；驱动顺序保证
        # run 期恒非 None——mypy 按 Any 容忍，勿在装配前调用）
        self._ctx: Any = None  # _GpuRunCtx（after_open 前置装配）
        self._hooks: Any = None  # 设备函数显式依赖（gpu/device.DevHooks）
        self._prepare: Any = None  # _gpu_prepare_calibration
        self._stream_fn: Any = None  # (nvdec, cpu) 两帧流
        self._YFramePool: Any = None
        self._on_gpu = False
        self._yuv = spec.yuv_output
        self._limited = spec.color_range != 1
        self._th = 0
        # 帧流（内联生成器；C-84 去生产者线程后无队列/线程态）
        self._stream: Any = None
        # 消费侧运行态
        self._session: Any = None
        self._put_ocr: Any = None
        self._raw_ready: Any = None
        self._rep_crops: dict = {}
        self._pending_crops: list = []  # [(r_frame, dev, prefer_device)]
        self._seg_idx = 0
        self._merge_hits = 0  # 逐段计数改局部累加，run 末一次上报

    @property
    def crops(self) -> dict:
        return self._rep_crops

    # ── 生命周期挂钩 ──────────────────────────────────────────────
    def after_open(self, vr) -> None:
        from ..gpu.streams import (
            DevHooks,
            _GpuRunCtx,
            _gpu_frame_stream_cpu,
            _gpu_frame_stream_nvdec,
            _gpu_prepare_calibration,
        )

        spec = self._spec
        self._ctx = _GpuRunCtx()
        # 设备侧协作函数的显式依赖
        self._hooks = DevHooks(
            limited=spec.color_range != 1,
            batch_luma=spec.batch_luma,
            bin_ref=spec.bin_thresh_ref,
            on_bin=spec.on_bin_thresh,
            prof=spec.prof_end,
        )
        self._prepare = _gpu_prepare_calibration
        self._stream_fn = (_gpu_frame_stream_nvdec, _gpu_frame_stream_cpu)
        # on_gpu 必须在 open_vr **之后**判定——backend 标签由 open 写入
        # （门面若在 spec 构造期求值，后端重置后的空标签会使其
        # 恒为 False → 误走 CPU 解码分支，stride>1 时 rep_frame 帧号漂移）。
        _label = spec.backend_label()
        # 依据 C-76：hybrid 恒走宿主帧流——gpu-out 的解码器内上载中转实测
        # −13% 流交付（3805→3252fps，fork 口径 A/B）；宿主帧流 + OCR 会话
        # 上传 crop 与 cpu 后端同通路（已验证）。NVDEC 后端（decord/GPU）
        # 保持设备直通零拷贝不变。
        self._on_gpu = _label == "decord/GPU"

    def calibrate(self, spec, vr, frames, ocr_session):
        x1, y1, x2, y2 = spec.roi
        ok, th = self._prepare(
            self._hooks,
            self._ctx,
            vr,
            frames,
            on_gpu=self._on_gpu,
            yuv=self._yuv,
            roi=(x1, y1, x2 + 1, y2 + 1),
            warm=self._warm,
        )
        self._th = th
        return ok, th

    def after_calibrate(self, th, ocr_session) -> None:
        spec = self._spec
        self._session = ocr_session
        self._put_ocr = ocr_session.put
        spec.bin_thresh_ref[0] = th
        if spec.on_bin_thresh is not None:
            spec.on_bin_thresh(th)
        # y_pool 依赖校准产出的 src_h/src_w；生产者未启动，此处赋值先行于
        # 一切并发读者。
        from ..gpu.pools import _YFramePool

        self._ctx.y_pool = (
            _YFramePool(self._ctx.src_h * self._ctx.src_w) if (self._yuv and self._on_gpu) else None
        )
        ocr_session.autocropper = _DeferredAutocropper(self._ctx, spec, ocr_session)

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
            ocr_session.finish()  # 空会话收尾：worker 归还引擎
        except BaseException:
            logger.debug("ocr_session.finish 清理忽略异常", exc_info=True)
        self._release(self._ctx)
        return res

    def release(self) -> None:
        # 释放本次 extract 的临时设备缓冲；OCR 引擎缓冲归引擎池。
        self._release(self._ctx)

    def report_counters(self) -> None:
        if self._spec.metrics.enabled:
            self._spec.metrics.counter("segment.merges", self._merge_hits)

    # ── 帧流：生成器内联消费（与宿主 lane 同构）──────────────────────
    def frame_items(self, spec, vr, frames, ocr_session):
        """帧流直通：流生成器在消费线程内联拉动（C-84）。

        曾为「生产者线程 + 队列」结构（解码与消费重叠）；消费端投机化后
        消费者只剩 ~µs/帧（合并判定命中零 GPU 交互，consume_feed 实测
        0.017s/3000 帧），重叠收益已不存在，而独立 Python 线程的 GIL
        争用反而拖慢解码拉动（decode.batch 实测 +240ms/3000 帧，
        nvdec+TRT test5：队列结构 2.98s vs 内联 2.7x s，与宿主 lane
        同水位）。OCR 会话本就有独立 worker/infer 线程，OCR∥解码重叠
        不受影响。流 yield 批列表（streams.py），此处摊平逐帧交付。
        """
        self._raw_ready = ocr_session.raw_ready
        x1, y1, x2, y2 = spec.roi
        self._stream = self._stream_fn[0 if self._on_gpu else 1](
            self._hooks,
            self._ctx,
            vr,
            frames,
            yuv=self._yuv,
            roi=(x1, y1, x2 + 1, y2 + 1),
            th=self._th,
            C=spec.C,
            merge_on=spec.merge_similar,
            use_bin=1 if spec.merge_effective_mode() == "binary" else 0,
        )
        for batch in self._stream:
            for fi, sharp, kw in batch:
                yield fi, sharp, kw

    def after_stream(self) -> None:
        # 帧流内联消费后无后台线程：生成器耗尽即全部完成
        return

    def stop_consume(self) -> None:
        from ..gpu.streams import _gpu_pull_stop

        # 任何退出路径都收口拉流预读线程（消费早停时其 put 循环见停即退；
        # 正常收尾时线程已随哨兵退出，此调用为幂等 no-op）
        _gpu_pull_stop(self._ctx)
        self._resolve_keep_crops()  # 尾窗收口（rep_crops 必须完整）

    # ── 代表帧交付 / 合并判定（消费线程内）────────────────────────
    def emit(self, seg, rep, frac) -> None:
        self._emit_ocr(self._seg_idx, rep[0], rep[1], frac, rep[2])
        self._seg_idx += 1

    def _d2h_rep(self, dev, *, prefer_device=False):
        """代表帧 → 宿主：NVDEC = D2H；CPU 解码默认宿主切片直取。"""
        from cuda.bindings import runtime as cudart

        ctx = self._ctx
        hc = getattr(dev.owner, "host_crop", None)
        if hc is not None and not prefer_device:
            h = hc()
            if h is not None:
                return np.array(h)
        arr = np.empty((dev.h, dev.w), dtype=np.uint8)
        # 消费流上异步 D2H + 同步消费流：不走 NULL 流（避免耦合生产者）。
        cudart.cudaMemcpyAsync(
            arr.ctypes.data,
            int(dev.ptr),
            dev.h * dev.w,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ctx.analyzer._stream_c,
        )
        cudart.cudaStreamSynchronize(ctx.analyzer._stream_c)
        _MET = self._spec.metrics
        if _MET.enabled:
            _MET.counter("emit.d2h_calls")
            _MET.counter("emit.d2h_bytes", dev.h * dev.w)
            _MET.counter("emit.syncs")
        return arr

    def _resolve_keep_crops(self) -> None:
        # ── keep_crops 的 D2H 并批：有界窗口收集 + 每窗一次流同步
        # （否则每段一次 async D2H + 一次流同步，3000 帧/1083 段 = 1083
        # 次同步全落在消费线程上）；窗口 = 16 段，避免钉住过多 decord 批 owner。
        if not self._pending_crops:
            return
        from cuda.bindings import runtime as cudart

        spec = self._spec
        ctx = self._ctx
        _MET = spec.metrics
        _t_d2h = time.perf_counter()
        arrs: list = []
        for _rf, _dev, _prefer in self._pending_crops:
            _hc = getattr(_dev.owner, "host_crop", None)
            _h = _hc() if (_hc is not None and not _prefer) else None
            if _h is not None:
                arrs.append(np.array(_h))
                continue
            _arr = np.empty((_dev.h, _dev.w), dtype=np.uint8)
            cudart.cudaMemcpyAsync(
                _arr.ctypes.data,
                int(_dev.ptr),
                _dev.h * _dev.w,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                ctx.analyzer._stream_c,
            )
            arrs.append(_arr)
        cudart.cudaStreamSynchronize(ctx.analyzer._stream_c)  # 每窗一次
        for (_rf, _dev, _prefer), _arr in zip(self._pending_crops, arrs):
            self._rep_crops[_rf] = _arr
        if _MET.enabled:
            # 逐段计数在此上报（每窗一次），少 N 次 Python 上报
            _MET.counter("emit.keep_crops_batched", len(self._pending_crops))
            _MET.counter("emit.keep_crops_d2h", len(self._pending_crops))
        if spec.prof_end is not None:
            spec.prof_end("producer", "emit_d2h", _t_d2h)
        self._pending_crops.clear()

    def _autocrop_device(self, gray_ptr, sharp):
        """GPU 直通裁切：col_ink 判「有墨迹列范围」+ 宿主同一余量规则。"""
        spec = self._spec
        ctx = self._ctx
        if not spec.ocr_autocrop:
            return None
        if ctx.src_w <= 8 or sharp < 3.0:
            return None
        rng = ctx.analyzer.content_range(
            int(gray_ptr),
            ctx.src_h,
            ctx.src_w,
            spec.bin_thresh_ref[0],
            stream=ctx.analyzer._stream_c,
        )
        if rng is None:
            return None
        return spec.content_range_to_crop(rng[0], rng[1], ctx.src_w)

    def similar(self, a, b) -> bool:
        """merge_similar 判定（整数精确）。

        投机缓存命中（C-84）→ 宿主标量判定，零 GPU 交互：生产者侧已按
        状态机的 merge 无关几何（BoundaryPairPlanner）把「上一段代表帧
        →本段代表帧」对批量预计算（同 sim_pair 内核、同指针，结果逐位
        相同）。未命中（上一边界被合并 → 状态机问的是更旧的
        last_emitted_rep）→ 原同步路径兜底，语义不变。"""
        spec = self._spec
        if not (spec.merge_similar and a[1] is not None and b[1] is not None):
            return False
        _t_cmp = time.perf_counter()
        _MET = spec.metrics
        hit = self._ctx.pair_cache.pop((a[0], b[0]), None)
        if hit is not None:
            mad, chg, win3 = hit
            if _MET.enabled:
                _MET.counter("segment.spec_pair_hit")
        else:
            if _MET.enabled:
                _MET.counter("segment.spec_pair_miss")
            mad, chg, win3 = self._similar_device_scalars(a[1], b[1])
        n = self._ctx.src_h * self._ctx.src_w
        use_bin = spec.merge_effective_mode() == "binary"
        mean = 255.0 * mad / n if use_bin else mad / n
        # 稠密簇门：win3 由 sim_pair 设备侧算出，与宿主 _cluster_win3 逐位
        # 一致（同一 3×3 邻域定义）；两侧共用 dense_gate_hit + similar_decision，
        # 否则 C-32「GPU 为宿主逐位镜像」的契约在阈值处破。
        _dec = similar_decision(
            mean,
            chg,
            spec.merge_similar_threshold,
            spec.merge_max_changed_pixels,
            dense=dense_gate_hit(win3, spec.merge_dense_gate),
        )
        if _dec:
            self._merge_hits += 1  # 逐段计数改为局部累加，run 末一次上报
        if spec.prof_end is not None:
            # 合并判定的全开销（命中=宿主标量；未命中=launch+D2H+同步）
            spec.prof_end("producer", "merge_pair", _t_cmp)
        return _dec

    def _similar_device_scalars(self, a_dev, b_dev) -> "tuple[int, int, int]":
        """兜底路径：GPU sim_pair 的三标量（未投机/yuv/合并链未命中）。"""
        spec = self._spec
        ctx = self._ctx
        use_bin = 1 if spec.merge_effective_mode() == "binary" else 0
        ya = yb = None
        if self._yuv and self._on_gpu:
            ya = ctx.y_pool.acquire()
            yb = ctx.y_pool.acquire()
            ctx.analyzer.luma_into(
                int(a_dev.ptr),
                int(ya.ptr),
                ctx.src_h,
                ctx.src_w,
                self._limited,
                stream=ctx.analyzer._stream_c,
            )
            ctx.analyzer.luma_into(
                int(b_dev.ptr),
                int(yb.ptr),
                ctx.src_h,
                ctx.src_w,
                self._limited,
                stream=ctx.analyzer._stream_c,
            )
            ap, bp = ya.ptr, yb.ptr
        else:
            ap, bp = int(a_dev.ptr), int(b_dev.ptr)
        try:
            return ctx.analyzer.compare_pair(
                ap,
                bp,
                ctx.src_h,
                ctx.src_w,
                spec.bin_thresh_ref[0],
                use_bin,
                stream=ctx.analyzer._stream_c,
            )
        finally:
            # 合并判定的池帧显式归还（不再依赖 GC 时机；payload 携带的池帧
            # 仍由 __del__ 入列回收——owner 生命周期跨线程，无法在此收口）
            if ya is not None:
                ctx.y_pool.recycle(ya)
            if yb is not None:
                ctx.y_pool.recycle(yb)

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
            if getattr(self._session, "autocropper", None) is not None:
                # 零拷贝 + 双推迟：emit 只剩构元+入队。
                if spec.ocr_autocrop and ctx.src_w > 8 and r_sharp >= 3.0:
                    dev_ocr = DeviceRef(
                        ptr=r_dev.ptr, h=ctx.src_h, w=ctx.src_w, owner=r_dev.owner, sharp=r_sharp
                    )  # 延后裁切标记
                else:
                    dev_ocr = DeviceRef(
                        ptr=r_dev.ptr,
                        h=ctx.src_h,
                        w=ctx.src_w,
                        owner=r_dev.owner,
                        x_off=0,
                        crop_w=ctx.src_w,
                    )
            else:
                if yuv and on_gpu:
                    yf = ctx.y_pool.acquire()
                    ctx.analyzer.luma_into(
                        int(r_dev.ptr),
                        int(yf.ptr),
                        ctx.src_h,
                        ctx.src_w,
                        self._limited,
                        stream=ctx.analyzer._stream_c,
                    )
                    cudart.cudaStreamSynchronize(ctx.analyzer._stream_c)
                    base = DeviceRef(ptr=yf.ptr, h=ctx.src_h, w=ctx.src_w, owner=yf)
                else:
                    base = r_dev
                xoff, cropw = 0, ctx.src_w
                _t_ac = time.perf_counter()
                rng = self._autocrop_device(base.ptr, r_sharp)
                if rng is not None:
                    xoff, cropw = rng
                if spec.prof_end is not None:
                    spec.prof_end("producer", "emit_autocrop", _t_ac)
                dev_ocr = base.with_crop(xoff, cropw)
        else:
            crop_h = self._d2h_rep(r_dev)
        if spec.keep_crops:
            if crop_h is not None:
                # 该段已在走宿主路径，代表帧就是宿主数组：直接落盘，无需 D2H
                self._rep_crops[r_frame] = crop_h
            else:
                # 设备代表帧的 D2H 进窗口并批（见 _resolve_keep_crops）。
                # 实测（交错 A/B，4 次独立进程）：冷启动 h264-gpu −2.45%
                # （符号一致），其余配置中性——收益集中在"代表帧留显存"的
                # raw 直通路径。
                self._pending_crops.append((r_frame, r_dev, dev_ocr is not None and not yuv))
                if len(self._pending_crops) >= _KEEP_CROPS_WINDOW:
                    self._resolve_keep_crops()
        if dev_ocr is not None and not yuv:
            drop_host = getattr(dev_ocr.owner, "drop_host", None)
            if drop_host is not None:
                drop_host()
        from .ocr_stage import SegmentTask

        self._put_ocr(SegmentTask(idx, r_frame, crop_h, dev_ocr, frac))
        if spec.prof_end is not None:
            spec.prof_end("producer", "emit_put", _t_push)


def run_gpu_pipeline(spec: GpuRunSpec, ocr_engines=None) -> GpuRunResult:
    """GPU 驱动门入口。

    生命周期由 _driver.run_segment_pipeline 承载；本模块提供
    GPU lane 策略：设备侧校准 → 内联帧流生成器（解码+GPU analyze 与
    OCR worker/infer 线程重叠）→ 消费循环 → 清理；形状不符回退宿主。
    入口先启动 analyzer 预热线（构造 ~0.43s/冷进程，藏进 reader 打开
    背后，见 _gpu_analyzer_prewarm_start）。
    """
    from ._driver import run_segment_pipeline
    from ..gpu.streams import _gpu_analyzer_prewarm_start

    warm = _gpu_analyzer_prewarm_start()
    return run_segment_pipeline(spec, GpuRunResult(), _GpuLane(spec, warm), ocr_engines)
