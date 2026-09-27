"""GPU 设备侧协作流（S4 设备侧拆分，2026-09-20 稳健性轮补做）。

自 gpu/device.py 机械搬迁（零逻辑改动）：DevHooks（设备函数显式依赖，
取代 _SpecView 桥接）、_GpuRunCtx（单次运行共享设备状态）、校准
（_gpu_prepare_calibration）、两条帧流（NVDEC 设备直通 / CPU 解码
H2D）与部分释放（_gpu_release_partial）。编排在 pipeline/gpu_backend.py；
池机制在 gpu/pools.py；可用性探测留在 gpu/device.py（§10.4 patch 点）。
"""
import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from video_ocr_engine.config import constants as config
from video_ocr_engine.domain.segmentation import (  # noqa: F401 —— 等价性由金标向量守护（C-32）
    _otsu_from_hist, otsu_median_threshold)
from .frame_ref import DeviceRef
from .pools import _CpuFrameRef, _DevBatchPool
from .._helpers import _ndarray_device_ptr

logger = logging.getLogger(__name__)


@dataclass
class DevHooks:
    """设备侧协作函数的显式依赖（P1 后取代 ex 适配视图 _SpecView 桥接）。

    limited    —— color_range != 1（YUV 的 luma 展开口径）
    batch_luma —— CPU 解码分支的宿主灰度回调 (B,H,W[,C])->(B,h,w)
    bin_ref    —— [th] 单元素盒：校准写入、判定回调读
    on_bin     —— 阈值即时回写（F-4 流式合并活读；可 None）
    prof       —— 计时脊柱 (group, key, t0)（可 None）
    """

    limited: bool
    batch_luma: "Callable | None"
    bin_ref: list
    on_bin: "Callable | None" = None
    prof: "Callable | None" = None

    def set_bin(self, th: int) -> None:
        self.bin_ref[0] = th
        if self.on_bin is not None:
            self.on_bin(th)

    def tick(self, group: str, key: str, t0: float) -> None:
        if self.prof is not None:
            self.prof(group, key, t0)


class _GpuRunCtx:
    """_run_pipelined_gpu 单次运行的共享设备状态（显式替代闭包捕获，0.11.0）。

    字段由 _gpu_prepare_calibration 填充，帧流生成器与消费循环只读/推进；
    _gpu_release_partial 负责统一释放。"""
    __slots__ = ("analyzer", "pool", "y_pool", "calib_owner", "calib_nds",
                 "calib_base", "calib_gray", "src_h", "src_w", "fnb",
                 "prev_holder", "prev_ptr", "calib_n")

    def __init__(self) -> None:
        self.analyzer = None
        self.pool = None
        self.y_pool = None
        self.calib_owner = None
        self.calib_nds = None
        self.calib_base = 0
        self.calib_gray = 0
        self.src_h = 0
        self.src_w = 0
        self.fnb = 0
        self.prev_holder = None
        self.prev_ptr = 0
        self.calib_n = 0


def _gpu_fill_prev(analyzer, prev_buf, base, B, fnb, first_src) -> None:
    """prev 缓冲错位填充（三处调用共用：NVDEC 帧流 / CPU 帧流校准 / 批循环）。

    第 k 行 = (first_src if k==0 else base+(k-1)*fnb) 的 fnb 字节——
    供 analyze_batch 读"上一帧"；D2D 异步到 analyzer._stream。

    2026-09-10 由逐帧 B 次 cudaMemcpyAsync 收拢为 2 次：行 1..B-1 恰好是
    base 的连续区段（base[0..B-2]），一次大拷贝逐位等价（prev_buf 与
    base 恒为不同缓冲，无别名）。每次 cudaMemcpyAsync 有 ~10-20µs 的
    Python+驱动提交开销，B=64 时每批省 ~1ms——hybrid 引擎口径下解码
    批耗时 ~25ms，这笔开销占在引擎内 hybrid 增益流失（2601→2132fps）
    的可归因部分。"""
    from cuda.bindings import runtime as cudart
    _d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    if B > 1:
        cudart.cudaMemcpyAsync(
            prev_buf + fnb, base, (B - 1) * fnb, _d2d, analyzer._stream)
    cudart.cudaMemcpyAsync(prev_buf, first_src, fnb, _d2d, analyzer._stream)


def _gpu_release_partial(ctx: _GpuRunCtx) -> None:
    """释放尚未进入主消费循环的设备资源（池 / 校准缓冲 / 分析器）。"""
    ctx.calib_owner = None
    ctx.calib_nds = None
    ctx.calib_base = 0
    ctx.prev_holder = None
    ctx.prev_ptr = 0
    for _p in (ctx.y_pool, ctx.pool):
        if _p is not None:
            try:
                _p.release_all()
            except BaseException:
                logger.debug("release_all 清理忽略异常", exc_info=True)
                pass
    ctx.y_pool = None
    ctx.pool = None
    if ctx.analyzer is not None:
        try:
            ctx.analyzer.release()
        except Exception:
            pass  # 清理路径：release 失败无需上抛（进程退出兜底）
        ctx.analyzer = None


def _gpu_prepare_calibration(hooks: DevHooks, ctx: "_GpuRunCtx", vr, frames: list, *,
                             on_gpu: bool, yuv: bool,
                             roi: tuple) -> "tuple[bool, int]":
    """GPU 管线校准：解码校准帧 + 逐帧直方图 Otsu（阈值取中位数）。

    填充 ctx（analyzer/pool/calib_owner/calib_nds/calib_base/calib_gray/
    src_h/src_w/fnb/calib_n）并写 hooks 阈值盒。返回 (ok, th)；
    False = 帧形状不符（GPU 分段不支持），调用方回退宿主管线。
    """
    from cuda.bindings import runtime as cudart
    from video_ocr_engine.ocr.trt import GpuFrameAnalyzer
    ctx.analyzer = analyzer = GpuFrameAnalyzer()
    calib_n = ctx.calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
    if on_gpu:
        # ── NVDEC：decord 设备批直通（校准批同样不落 RAM）──
        ctx.calib_nds = vr.get_batch(frames[:calib_n], roi=roi)
        calib_base, calib_shape = _ndarray_device_ptr(ctx.calib_nds)
        ctx.calib_base = calib_base
        if yuv:
            # yuv420（packed NV12）：先 D2D 提取 Y 平面（luma_nv12 与宿主
            # _nv12_luma_full 逐位一致），histogram/analyze 都只消费灰度 Y。
            if len(calib_shape) != 3:
                return False, 0
            ctx.src_h = calib_shape[1] * 2 // 3
            ctx.src_w = calib_shape[2]
            ctx.calib_gray = analyzer.extract_luma(
                calib_base, calib_n, ctx.src_h, ctx.src_w,
                limited=hooks.limited)
        else:
            if len(calib_shape) != 4 or calib_shape[-1] != 1:
                # 灰度帧非 4D 单通道（部分 decord fork 输出 (B,H,W)）：GPU
                # 分段不支持，回退宿主。直接 _run_pipelined_host（防递归）。
                return False, 0
            ctx.src_h, ctx.src_w = calib_shape[1], calib_shape[2]
            ctx.calib_gray = calib_base
        ctx.fnb = ctx.src_h * ctx.src_w
        ctx.prev_holder = ctx.calib_nds      # 保住前一 decord NDArray（防解码池复用）
        ctx.prev_ptr = calib_base            # 灰色模式：上一批/校准末帧 device 指针
    else:
        # ── CPU 解码（P1-3）：校准批 asnumpy → 宿主灰度 → H2D ──
        ctx.calib_nds = vr.get_batch(frames[:calib_n], roi=roi)
        crops = ctx.calib_nds.asnumpy()
        if yuv:
            if crops.ndim != 3:
                return False, 0
            ctx.src_h = crops.shape[1] * 2 // 3
            ctx.src_w = crops.shape[2]
        else:
            if crops.ndim != 4 or crops.shape[-1] != 1:
                return False, 0
            ctx.src_h, ctx.src_w = crops.shape[1], crops.shape[2]
        ctx.fnb = ctx.src_h * ctx.src_w
        g = np.ascontiguousarray(hooks.batch_luma(crops))
        if g.shape != (calib_n, ctx.src_h, ctx.src_w):
            return False, 0
        # 池子必须容得下**校准批**（SEG_CALIB_FRAMES=50 行）：calib_owner 与
        # 后续解码批共用本池，校准 H2D/kernel 都按 calib_n 行访问。此前按
        # DECODE_BATCH 定尺寸，DECODE_BATCH < 50（如调参试过 32）时校准批
        # 向设备缓冲越界写 18 行 —— 越界破坏同 context 内其他分配（TRT 工作
        # 缓冲），异步错误延迟到 TRT enqueue 才爆（invalid argument），并
        # 污染 analyzer 后续结果（段数漂移）。默认 64 ≥ 50 从未触发。
        # （2026-09-09 sweep batch=32 复现后定位，见 docs/log 同日叙事。）
        ctx.pool = _DevBatchPool(
            max(config.GPU_PIPELINE_DECODE_BATCH, calib_n) * ctx.fnb)
        ctx.calib_owner = ctx.pool.acquire(crops)
        cudart.cudaMemcpyAsync(
            ctx.calib_owner.ptr, g.ctypes.data, calib_n * ctx.fnb,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            analyzer._stream)

    # 逐帧直方图校准：与单流水线"前 50 帧 Otsu 取中位数"语义逐位一致
    # （含退化双值帧的阈值行为），D2H 仅 B×1KB 标量表，校准帧不落 RAM。
    # 注意必须用 _otsu_from_hist（输入是直方图行）；_otsu 接收的是
    # 灰度图像并在内部做直方图——传错曾产生"直方图的直方图"垃圾阈值。
    calib_gray_dev = (ctx.calib_gray if on_gpu else ctx.calib_owner.ptr)
    _hist_mat = analyzer.histograms_perframe(
        calib_gray_dev, calib_n, ctx.src_h, ctx.src_w)
    th = otsu_median_threshold(
        [_otsu_from_hist(_hist_mat[k]) for k in range(calib_n)])
    hooks.set_bin(th)
    return True, th



def _gpu_frame_stream_nvdec(hooks: DevHooks, ctx: "_GpuRunCtx", vr, frames: list, *,
                            yuv: bool, roi: tuple, th: int):
    """NVDEC 设备批直通帧流：yield (frame_idx, dev, sharp, cluster)。

    dev 元组 gray=(nds,yptr,H,W) / yuv=(nds,nv12ptr,rows,W)；sharp/cluster
    由 analyze_batch（win3 分数）产出。prev 缓冲经 ctx 跨批衔接
    （prev_holder 保活 decord NDArray，防解码池复用）。
    """
    from cuda.bindings import runtime as cudart
    DECODE_BATCH = config.GPU_PIPELINE_DECODE_BATCH
    _d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    limited = hooks.limited
    analyzer = ctx.analyzer

    def _analyze_batch(gray_base, prev_single, B, H, W):
        fnb = H * W
        prev_buf = analyzer._ensure_prev(
            max(B, DECODE_BATCH) * fnb)
        _gpu_fill_prev(analyzer, prev_buf, gray_base, B, fnb, prev_single)
        return analyzer.analyze_batch(
            gray_base, prev_buf, B, H, W, th), fnb

    # ── 校准帧整批分析（yuv 已在外部提取 Y → calib_gray）──
    B = ctx.calib_n
    sums, fnb = _analyze_batch(ctx.calib_gray, ctx.calib_gray, B,
                               ctx.src_h, ctx.src_w)
    rows = ctx.src_h + (ctx.src_h + 1) // 2
    for k in range(B):
        cur = ctx.calib_base + k * (rows * ctx.src_w if yuv else fnb)
        yield (frames[k],
               DeviceRef(ptr=cur, h=(rows if yuv else ctx.src_h),
                         w=ctx.src_w, owner=ctx.calib_nds),
               float(sums[k, 0]), float(sums[k, 1]))
        ctx.prev_holder = ctx.calib_nds
        ctx.prev_ptr = cur
    # yuv：末帧 Y 存入单帧缓冲（extract_luma 复用主缓冲会覆盖，
    # 下一批开始前必须已有独立副本）。注：_ensure_prev 首次按
    # 64*fnb 分配且批尺寸恒定 → prev_front 指针后续稳定不重分配。
    prev_front = None
    have_prev_front = False
    if yuv:
        prev_front = analyzer._ensure_prev(fnb)
        cudart.cudaMemcpyAsync(
            prev_front, ctx.calib_gray + (ctx.calib_n - 1) * fnb, fnb,
            _d2d, analyzer._stream)
        have_prev_front = True

    # chunk 粒度流水发射（decord get_batch_stream）：后台预取
    # 下一批，解码完成即交付 —— 消费（extract_luma/analyze）与
    # 解码重叠。GPU_PIPELINE_STREAM 默认关（实测零收益，C-10）。
    _use_stream = (config.env_bool(config.GPU_PIPELINE_STREAM_ENV,
                                   default=False)
                   and hasattr(vr, 'get_batch_stream'))

    def _batch_iter():
        if _use_stream:
            for s0, nds in vr.get_batch_stream(
                    frames[ctx.calib_n:], roi=roi, batch=DECODE_BATCH):
                yield s0, nds
            return
        # ── 层4 排空线程（GPU_PIPELINE_DRAINER，默认关，见下）──
        # get_batch 独占一线程预取（深 4 批），生产者的 analyze/sync 不再
        # 阻塞解码排空——decode.batch 引擎税的构成为"生产者分析期间无人
        # 拉动解码，fork 侧银行反压停转"。线程安全性：get_batch 全部调用
        # 恒在此线程（decord Push/pump 单线程不变量保持）；calibrate 的
        # next_roi 在此流启动前已结束。
        # 默认关：实测零收益（引擎税源于线程争用而非排空阻塞，与
        # C-10 流水发射同判）；留旋钮供复评。
        _drainer_on = config.env_bool(config.GPU_PIPELINE_DRAINER_ENV,
                                      default=False)
        bstarts = range(ctx.calib_n, len(frames), DECODE_BATCH)
        if not _drainer_on:
            for bstart in bstarts:
                # roi 不随批传：打开 reader 时 SetRoi 已生效；hybrid 原生
                # 路径每次 get_batch 传 roi 会触发 fork 侧 SetRoi/池深重算
                # （2026-09-10 实测 hybrid av1 2487→2264 fps，-9%；
                # _probe_roi_decode 已证 CPU 路径两种传法等价）。
                _t_dec = time.perf_counter()
                nds = vr.get_batch(frames[bstart:bstart + DECODE_BATCH])
                hooks.tick('producer', 'decode_batch', _t_dec)
                yield bstart, nds
            return
        import queue as _queue
        import threading as _threading
        _q: "_queue.Queue" = _queue.Queue(maxsize=4)
        _err: list = []

        def _drain() -> None:
            try:
                for bstart in bstarts:
                    bend = min(bstart + DECODE_BATCH, len(frames))
                    _t_dec = time.perf_counter()
                    nds = vr.get_batch(frames[bstart:bend])
                    hooks.tick('producer', 'decode_batch', _t_dec)
                    _q.put((bstart, nds))
            except BaseException as e:  # noqa: BLE001
                _err.append(e)
            finally:
                _q.put(None)

        _dt = _threading.Thread(target=_drain, daemon=True,
                                name='voe-decode-drainer')
        _dt.start()
        while True:
            item = _q.get()
            if item is None:
                if _err:
                    raise _err[0]
                return
            yield item

    for bstart, nds in _batch_iter():
        bend = bstart + int(nds.shape[0])
        base, shape = _ndarray_device_ptr(nds)
        B = int(bend - bstart)
        if yuv:
            if len(shape) != 3:
                raise RuntimeError(
                    "GPU yuv 分段仅支持 decord yuv420 输出")
            H = shape[1] * 2 // 3
            W = shape[2]
            rows = H + (H + 1) // 2
            # extract 覆盖 _luma 主缓冲 → 先取上一批末帧副本
            gray_base = analyzer.extract_luma(
                base, B, H, W, limited)
            prev_s = prev_front if have_prev_front else gray_base
        else:
            if len(shape) != 4 or shape[-1] != 1:
                raise RuntimeError(
                    "GPU 分段仅支持 decord gray 输出")
            H, W = shape[1], shape[2]
            rows = H
            gray_base = base
            prev_s = ctx.prev_ptr
        _t_an = time.perf_counter()
        sums, fnb = _analyze_batch(gray_base, prev_s, B, H, W)
        hooks.tick('producer', 'stream_analyze', _t_an)
        if yuv and have_prev_front:
            cudart.cudaMemcpyAsync(
                prev_front, gray_base + (B - 1) * fnb, fnb,
                _d2d, analyzer._stream)
            have_prev_front = True
        # stream 模式 bstart 即首帧号；seq 模式 bstart 是
        # frames 下标 → 统一转帧号
        f0 = (bstart if _use_stream else frames[bstart])
        for k in range(B):
            cur = base + k * (rows * W if yuv else fnb)
            yield (f0 + k, DeviceRef(ptr=cur, h=rows, w=W, owner=nds),
                   float(sums[k, 0]), float(sums[k, 1]))
            ctx.prev_holder = nds
            ctx.prev_ptr = cur
        if yuv:
            have_prev_front = True


def _gpu_frame_stream_cpu(hooks: DevHooks, ctx: "_GpuRunCtx", vr, frames: list, *,
                          yuv: bool, roi: tuple, th: int):
    """CPU 解码（P1-3）帧流：get_batch → 宿主灰度 → H2D → analyze。

    每批缓冲从池取（引用归零归还）；上一批末帧作本批 analyze
    的 prev（fill_prev 读取期间由 prev_owner 保活）。analyze
    同步返回后本批帧指针即可交付（H2D/kernel 均已完成）。
    """
    from cuda.bindings import runtime as cudart
    DECODE_BATCH = config.GPU_PIPELINE_DECODE_BATCH
    _d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    _h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    fnb = ctx.fnb
    analyzer = ctx.analyzer
    calib_n = ctx.calib_n
    src_h, src_w = ctx.src_h, ctx.src_w
    prev_buf = analyzer._ensure_prev(
        max(calib_n, DECODE_BATCH) * fnb)
    # ── 校准帧整批分析（校准批已在外部 H2D → calib_owner）──
    _gpu_fill_prev(analyzer, prev_buf, ctx.calib_owner.ptr,
                   calib_n, fnb, ctx.calib_owner.ptr)
    sums = analyzer.analyze_batch(
        ctx.calib_owner.ptr, prev_buf, calib_n, src_h, src_w, th)
    for k in range(calib_n):
        yield (frames[k],
               DeviceRef(ptr=ctx.calib_owner.ptr + k * fnb, h=src_h,
                         w=src_w, owner=_CpuFrameRef(ctx.calib_owner, k)),
               float(sums[k, 0]), float(sums[k, 1]))
    prev_owner = ctx.calib_owner   # 上一批缓冲（fill_prev 读取期间保活）
    prev_ptr = ctx.calib_owner.ptr + (calib_n - 1) * fnb
    for bstart in range(calib_n, len(frames), DECODE_BATCH):
        bend = min(bstart + DECODE_BATCH, len(frames))
        B = bend - bstart
        # P2c 覆盖缺口：CPU 解码分支此前零打桩（decode.*/stream_analyze
        # 只在 NVDEC 分支产出）。键语义与宿主路径对齐：decode_batch =
        # get_batch + asnumpy（宿主同款）；gray_batch = 灰度转换。
        _t_dec = time.perf_counter()
        nds = vr.get_batch(frames[bstart:bend], roi=roi)
        crops = nds.asnumpy()
        hooks.tick('producer', 'decode_batch', _t_dec)
        if yuv:
            if crops.ndim != 3:
                raise RuntimeError(
                    "GPU yuv 分段仅支持 decord yuv420 输出")
        else:
            if crops.ndim != 4 or crops.shape[-1] != 1:
                raise RuntimeError(
                    "GPU 分段仅支持 decord gray 输出")
        _t_gray = time.perf_counter()
        g = np.ascontiguousarray(hooks.batch_luma(crops))
        hooks.tick('producer', 'gray_batch', _t_gray)
        if g.shape != (B, src_h, src_w):
            raise RuntimeError(
                f"GPU(CPU解码) 灰度形状不符: {g.shape} != "
                f"{(B, src_h, src_w)}")
        owner = ctx.pool.acquire(crops)
        cudart.cudaMemcpyAsync(
            owner.ptr, g.ctypes.data, B * fnb, _h2d,
            analyzer._stream)
        base = owner.ptr
        prev_buf = analyzer._ensure_prev(
            max(B, DECODE_BATCH) * fnb)
        _t_an = time.perf_counter()
        _gpu_fill_prev(analyzer, prev_buf, base, B, fnb, prev_ptr)
        sums = analyzer.analyze_batch(
            base, prev_buf, B, src_h, src_w, th)
        hooks.tick('producer', 'stream_analyze', _t_an)
        for k in range(B):
            yield (frames[bstart + k],
                   DeviceRef(ptr=base + k * fnb, h=src_h, w=src_w,
                             owner=_CpuFrameRef(owner, k)),
                   float(sums[k, 0]), float(sums[k, 1]))
        prev_owner = owner  # noqa: F841  # 保活：下批 fill_prev 读 prev_ptr（旧 owner 设备内存）期间不得回收
        prev_ptr = base + (B - 1) * fnb
