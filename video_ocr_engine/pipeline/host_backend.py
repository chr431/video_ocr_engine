"""宿主后端（v2 §5/§6.2，S3-3a；P1 起为宿主 lane 策略）。

从 extractor._run_pipelined_host 与 _host_pipeline 的三个模块级函数外提
（2026-09-10）。语义变化：**算法代码只读 HostRunSpec 的显式声明字段**，
不再穿透 FieldExtractor 的私有属性（P0-1 宿主子集至此有契约）；运行态
只写入 HostRunResult 与两个显式可变盒（fps 缓存 / backend 标签），
由门面在调用前后同步。

P1（宿主/GPU 实现统一轮）：生命周期（open→fps→会话→校准→消费→收尾）
上收到 _driver.run_segment_pipeline 单出处，本模块只剩宿主侧策略——
校准（_calibrate）、帧流（_frame_stream）、emit/合并判定（_HostLane）。
像素与判定原语经 spec 回调注入（实现唯一出处 segmentation /
extractor 薄层）；GPU 路径的回退经 preopened_vr 复用（C10 语义不变）。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from video_ocr_engine.config import constants as config
from ..domain.metrics import NULL_METRICS
from ..gpu.frame_ref import DeviceRef
from video_ocr_engine.domain.segmentation import _otsu, otsu_median_threshold

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
    # S6-0：注入的指标记录器（§8.6 N-2；off 档为 NullMetrics 单例）
    metrics: Any = NULL_METRICS


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
    fork_stats: dict | None = None   # fork 遥测穿透（hybrid 解码器才有）
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
                dev_info = DeviceRef(ptr=base + k * src_h * src_w,
                                     h=src_h, w=src_w, owner=nds)
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
                dev_info = DeviceRef(ptr=base, h=shape[0], w=shape[1],
                                     owner=nd)
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
        if spec.metrics.enabled:
            m = spec.metrics
            m.counter('decode.batches')
            m.counter('decode.frames', bend - bstart)
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
                d = DeviceRef(ptr=dev_base + k * src_h * src_w,
                              h=src_h, w=src_w, owner=nds)
            # gray 必须拷贝：payload 灰度会作为代表帧逃逸出批作用域，
            # 而 g_buf 跨批复用——不拷贝则 merge_similar 在后续批读到
            # 被覆写的帧（2026-09-10 D1 调查：test5 host 45/1089 合并
            # 判定失真、段数 1042 vs GPU 正确值 1083；sharp/bs 为标量/
            # 每批新数组不受影响）。拷贝 ~3.5KB/帧，量级可忽略。
            yield (frames[gi], crops[k], g[k].copy(), float(sharp[k]),
                   bs[k], d)


class _HostLane:
    """宿主 lane：校准/帧流/emit/合并判定的宿主实现（_driver 的策略侧）。

    payload 形状 (fi, crop, gray, dev_info)——emit 取 [0]/[1]/[3]，
    similar 判定取 [2]（灰度）。
    """

    progress_verb = '解码+分段'
    debug_tag = 'HB'

    def __init__(self, spec: HostRunSpec) -> None:
        self._spec = spec
        self._with_dev = True
        self._calib: list = []
        self._th = 0
        self._seg_idx = 0
        self._rep_crops: dict = {}
        # calibrate() 前置装配（驱动顺序保证 run 期恒非 None）
        self._put_ocr: Any = None

    @property
    def crops(self) -> dict:
        return self._rep_crops

    def after_open(self, vr) -> None:
        # with_dev 恒 True：保留 GPU 单通道帧的 DLPack 指针供 GPU raw OCR
        # 直通。fork hybrid（CPU-out）读者实测无害：其 get_batch 返回
        # decord NDArray（有 to_dlpack），采到的是 CPU 指针且仅在 GPU raw
        # OCR 直通时消费——而宿主帧 hybrid 只与 CPU OCR 组合（OCR 在 GPU
        # 时 extractor 选 hybrid_gpu 走 gpu_backend），该指针无人读
        # （2026-09-20 起点轮核实；旧 hybrid_begin 化石守卫已删）。
        self._with_dev = True

    def calibrate(self, spec, vr, frames, ocr_session):
        # with_dev=True：保留 GPU 单通道帧的 DLPack 指针供 GPU raw OCR 直通。
        # hybrid 交付宿主数组（无 to_dlpack），不存在可直通指针——强采
        # _ndarray_device_ptr 会 AttributeError，必须跳过。
        self._put_ocr = ocr_session.put
        calib, th = _calibrate(spec, vr, frames, with_dev=self._with_dev)
        self._calib = calib
        self._th = th
        return True, th

    def after_calibrate(self, th, ocr_session) -> None:
        if self._spec.on_bin_thresh is not None:
            self._spec.on_bin_thresh(th)   # 流式合并判定活读，必须即时回写（F-4）

    def frame_items(self, spec, vr, frames, ocr_session):
        for fi, c, g, sharp, b, d in _frame_stream(
                spec, frames, vr, self._calib, self._th,
                with_dev=self._with_dev):
            yield fi, sharp, dict(payload=(fi, c, g, d), bin=b)

    def emit(self, seg, rep, frac) -> None:
        _t_push = time.perf_counter()
        from .ocr_stage import SegmentTask
        self._put_ocr(SegmentTask(self._seg_idx, rep[0], rep[1], rep[3], frac))
        _prof(self._spec, 'producer', 'q_put_block', _t_push)
        if self._spec.keep_crops:
            self._rep_crops[rep[0]] = rep[1]
        self._seg_idx += 1

    def similar(self, a, b) -> bool:
        # P2c 覆盖：merge_pair 此前只有 GPU 路径产出（宿主只能从
        # timing['decode'] 反推合并成本）。merge_similar 关闭时不计
        # （无判定成本可量）。
        if not self._spec.merge_similar:
            return False
        _t = time.perf_counter()
        try:
            return self._spec.segments_similar(a[2], b[2])
        finally:
            _prof(self._spec, 'producer', 'merge_pair', _t)

    # ── 宿主路径的空挂钩（生命周期差异全在 GPU lane 侧）──
    def abort(self, ocr_session) -> None:
        try:
            ocr_session.finish()
        except BaseException:
            logger.debug("ocr_session.finish 清理忽略异常", exc_info=True)

    def fallback(self, res, ocr_session, ocr_engines, vr):
        raise AssertionError("宿主 lane 校准恒成功，不走 fallback 分支")

    def after_stream(self) -> None:
        pass

    def stop_consume(self) -> None:
        pass

    def release(self) -> None:
        pass

    def report_counters(self) -> None:
        pass


def run_host_pipeline(spec: HostRunSpec, ocr_engines=None,
                      preopened_vr=None) -> HostRunResult:
    """宿主驱动入口：解码∥分段∥OCR（段一闭合即把代表帧交给 OCR 会话
    线程）。生命周期由 _driver.run_segment_pipeline 承载（P1），本模块
    只提供宿主 lane 策略。
    """
    from ._driver import run_segment_pipeline
    return run_segment_pipeline(spec, HostRunResult(), _HostLane(spec),
                                ocr_engines, preopened_vr)
