"""GPU 全驻留零拷贝管线（_GpuPipelineMixin）：NVDEC 默认主路径。

从 extractor.py 拆出：_gpu_pipeline_enabled / _run_pipelined_gpu。GPU 预处理/
归约/帧分析内核位于 video_ocr_engine._gpu_kernels（ocr_trt re-export）。
FieldExtractor 组合本 mixin 获得这两个方法。
"""
import logging
import os as _os
import time

import numpy as np

import engine_config as config
from video_utils import nvdec_available, tensorrt_available
from segmentation import (SegmentStateMachine, similar_decision,
                          otsu_median_threshold)
from ._helpers import (_ndarray_device_ptr, _otsu_from_hist,
                       _decode_progress_pct,
                       _read_fps_from_vr)

logger = logging.getLogger(__name__)

# 收尾时等待 producer 退出的上限（秒）。正常路径下 producer 响应
# producer_stop 后立即退出；这里只作防挂死兜底——无超时 join 一旦碰上
# producer 卡在 put，会把"线程泄漏"升级成"整条流水线永久阻塞"。
_PRODUCER_JOIN_TIMEOUT = 5.0


def _cuda_python_available() -> bool:
    """cuda-python（cuda.core / cuda.bindings）是否可导入。

    GPU 分段/校准/CTC kernel 依赖它；缺失时 GPU 管线会初始化失败——
    门控直接判不可用（避免带 NVDEC 但无 cuda-python 的环境崩在
    GpuFrameAnalyzer()）。
    """
    try:
        import importlib.util as _u
        return _u.find_spec('cuda') is not None
    except Exception:
        return False


class _YFrame:
    """池化的单帧设备 Y 缓冲（yuv 代表帧 Y 平面提取用）。

    随队列/闭包传递（作为 dev 元组的 owner），引用归零（GC）时自动
    归还 _YFramePool，不阻塞调用方。
    """

    __slots__ = ("pool", "ptr", "size")

    def __init__(self, pool, ptr, size):
        self.pool = pool
        self.ptr = ptr
        self.size = size

    def __del__(self):
        try:
            self.pool._release(self)
        except Exception:
            pass


class _YFramePool:
    """单帧灰度 (H*W) device 缓冲池：yuv 模式零拷贝的关键件。

    raw OCR（call_gpu_raw）与 GPU 端 merge_similar 判定消费"帧的 Y 平面"。
    yuv 模式下代表帧以 packed NV12 保留在 decord NDArray（owner 保活），
    Y 提取（luma_into，~10KB D2D）按需落到池帧；池帧随队列流入 OCR
    worker，GC 时自动归还（跨线程安全，释放走 CUDA runtime）。
    """

    _MAX = 32   # ≥ OCR 批上限（16）+ 合并判定 2，避免高频 cudaMalloc 抖动

    def __init__(self, fnb: int):
        self._fnb = int(fnb)
        self._free: list = []

    def acquire(self) -> _YFrame:
        if self._free:
            return self._free.pop()
        from cuda.bindings import runtime as cudart
        _err, ptr = cudart.cudaMalloc(self._fnb)
        return _YFrame(self, int(ptr), self._fnb)

    def _release(self, frame: _YFrame) -> None:
        if len(self._free) < self._MAX:
            self._free.append(frame)
            return
        try:
            from cuda.bindings import runtime as cudart
            cudart.cudaFree(frame.ptr)
        except Exception:
            pass

    def release_all(self) -> None:
        """显式释放全部空闲缓冲（extract 结束调用；DESIGN-REVIEW C5：池
        原本只在超 _MAX 时 cudaFree，extract 返回后池随闭包 GC，已入池块
        永不释放 → 长进程显存单调增长）。"""
        while self._free:
            frame = self._free.pop()
            try:
                from cuda.bindings import runtime as cudart
                cudart.cudaFree(frame.ptr)
            except Exception:
                pass


class _DevBatch:
    """CPU 解码批的双缓冲 owner（P1-3）：device 批缓冲（池化）+ 宿主解码数组。

    分段/OCR 消费的 device 帧指针指向本缓冲；引用归零（GC）归还池。
    复用安全性与 _YFramePool 同一契约：raw OCR（call_gpu_raw 返回前同步）
    与 sim_pair（compare_pair 同步）读完才可能归零归还。
    """

    __slots__ = ("pool", "ptr", "size", "host")

    def __init__(self, pool, ptr, size, host):
        self.pool = pool
        self.ptr = ptr
        self.size = size
        self.host = host

    def __del__(self):
        try:
            self.pool._release(self)
        except Exception:
            pass


class _CpuFrameRef:
    """CPU 解码批的单帧引用：保活批缓冲 + 宿主 rep 切片直取（无 D2H）。

    dev 元组的 owner 槽位（dev[0]）；_d2h_rep 探测到 host_crop 属性即走
    宿主切片（拷贝返回，防 rep_crops 的 numpy view 钉住整批解码数组）。
    gray 的尾通道维 squeezed 掉 —— 与 NVDEC 路径 _d2h_rep 的 (H,W) 二维
    对齐（宿主 _segments_similar / OCR 回退预处理只吃二维灰度）。
    """

    __slots__ = ("batch", "k")

    def __init__(self, batch: _DevBatch, k: int):
        self.batch = batch
        self.k = int(k)

    def host_crop(self):
        # drop_host() 之后整批宿主数组已释放：返回 None 让调用方回退到设备
        # D2H，而不是抛 TypeError（None 下标）。当前该分支不可达——
        # raw_ready 单调递增，排空后不会再有同批帧走宿主回退。
        c = self.batch.host
        if c is None:
            return None
        c = c[self.k]
        return c[..., 0] if c.ndim == 3 else c

    def drop_host(self) -> None:
        """释放批量宿主数组，保留设备 owner 供 raw OCR 使用。"""
        self.batch.host = None


class _DevBatchPool:
    """CPU 解码批 device 缓冲池：固定容量 DECODE_BATCH×H×W，引用归零归还。"""

    _MAX = 8    # 2~3 个在途批 + OCR 队列中 rep 引用的批缓冲余量

    def __init__(self, nbytes: int):
        self._nbytes = int(nbytes)
        self._free: list = []

    def acquire(self, host) -> _DevBatch:
        if self._free:
            b = self._free.pop()
            b.host = host
            return b
        from cuda.bindings import runtime as cudart
        _err, ptr = cudart.cudaMalloc(self._nbytes)
        return _DevBatch(self, int(ptr), self._nbytes, host)

    def _release(self, b: _DevBatch) -> None:
        b.host = None
        if len(self._free) < self._MAX:
            self._free.append(b)
            return
        try:
            from cuda.bindings import runtime as cudart
            cudart.cudaFree(b.ptr)
        except Exception:
            pass

    def release_all(self) -> None:
        """显式释放全部空闲缓冲（同 _YFramePool.release_all，C5）。"""
        while self._free:
            b = self._free.pop()
            try:
                from cuda.bindings import runtime as cudart
                cudart.cudaFree(b.ptr)
            except Exception:
                pass


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
                pass
    ctx.y_pool = None
    ctx.pool = None
    if ctx.analyzer is not None:
        try:
            ctx.analyzer.release()
        except Exception:
            pass
        ctx.analyzer = None


def _gpu_prepare_calibration(ex, ctx: "_GpuRunCtx", vr, frames: list, *,
                             on_gpu: bool, yuv: bool,
                             roi: tuple) -> "tuple[bool, int]":
    """GPU 管线校准：解码校准帧 + 逐帧直方图 Otsu（阈值取中位数）。

    填充 ctx（analyzer/pool/calib_owner/calib_nds/calib_base/calib_gray/
    src_h/src_w/fnb/calib_n）并写 ex._bin_thresh。返回 (ok, th)；
    False = 帧形状不符（GPU 分段不支持），调用方回退宿主管线。
    """
    from cuda.bindings import runtime as cudart
    from ocr_trt import GpuFrameAnalyzer
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
                limited=ex._color_range != 1)
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
        g = np.ascontiguousarray(ex._batch_luma(crops))
        if g.shape != (calib_n, ctx.src_h, ctx.src_w):
            return False, 0
        ctx.pool = _DevBatchPool(config.GPU_PIPELINE_DECODE_BATCH * ctx.fnb)
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
    ex._bin_thresh = th
    return True, th


def _gpu_fallback_to_host(ex, ctx: "_GpuRunCtx", vr, ocr_session,
                          _ocr_engines):
    """GPU→宿主回退（C10）：reader 直接复用（get_batch 随机访问、
    无消费状态，免去二次打开/解码器悬挂）。"""
    if ocr_session is not None:
        try:
            ocr_session.finish()
        except BaseException:
            pass
    _gpu_release_partial(ctx)
    ex._degraded.append('GPU 管线形状不符，回退宿主管线')
    return ex._run_pipelined_host(_ocr_engines, vr)


def _gpu_frame_stream_nvdec(ex, ctx: "_GpuRunCtx", vr, frames: list, *,
                            yuv: bool, roi: tuple, th: int):
    """NVDEC 设备批直通帧流：yield (frame_idx, dev, sharp, cluster)。

    dev 元组 gray=(nds,yptr,H,W) / yuv=(nds,nv12ptr,rows,W)；sharp/cluster
    由 analyze_batch（win3 分数）产出。prev 缓冲经 ctx 跨批衔接
    （prev_holder 保活 decord NDArray，防解码池复用）。
    """
    from cuda.bindings import runtime as cudart
    DECODE_BATCH = config.GPU_PIPELINE_DECODE_BATCH
    _d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    limited = ex._color_range != 1
    analyzer = ctx.analyzer

    def _fill_prev(prev_buf, base, B, frame_nbytes, prev_single):
        for k in range(B):
            src = (prev_single if k == 0
                   else base + (k - 1) * frame_nbytes)
            cudart.cudaMemcpyAsync(
                prev_buf + k * frame_nbytes, src, frame_nbytes,
                _d2d, analyzer._stream)

    def _analyze_batch(gray_base, prev_single, B, H, W):
        fnb = H * W
        prev_buf = analyzer._ensure_prev(
            max(B, DECODE_BATCH) * fnb)
        _fill_prev(prev_buf, gray_base, B, fnb, prev_single)
        return analyzer.analyze_batch(
            gray_base, prev_buf, B, H, W, th), fnb

    # ── 校准帧整批分析（yuv 已在外部提取 Y → calib_gray）──
    B = ctx.calib_n
    sums, fnb = _analyze_batch(ctx.calib_gray, ctx.calib_gray, B,
                               ctx.src_h, ctx.src_w)
    rows = ctx.src_h + (ctx.src_h + 1) // 2
    for k in range(B):
        cur = ctx.calib_base + k * (rows * ctx.src_w if yuv else fnb)
        yield (frames[k], (ctx.calib_nds, cur,
                           (rows if yuv else ctx.src_h), ctx.src_w),
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
        else:
            for bstart in range(ctx.calib_n, len(frames), DECODE_BATCH):
                bend = min(bstart + DECODE_BATCH, len(frames))
                yield bstart, vr.get_batch(
                    frames[bstart:bend], roi=roi)

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
        sums, fnb = _analyze_batch(gray_base, prev_s, B, H, W)
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
            yield (f0 + k, (nds, cur, rows, W),
                   float(sums[k, 0]), float(sums[k, 1]))
            ctx.prev_holder = nds
            ctx.prev_ptr = cur
        if yuv:
            have_prev_front = True


def _gpu_frame_stream_cpu(ex, ctx: "_GpuRunCtx", vr, frames: list, *,
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
    for k in range(calib_n):
        src = (ctx.calib_owner.ptr if k == 0
               else ctx.calib_owner.ptr + (k - 1) * fnb)
        cudart.cudaMemcpyAsync(
            prev_buf + k * fnb, src, fnb, _d2d, analyzer._stream)
    sums = analyzer.analyze_batch(
        ctx.calib_owner.ptr, prev_buf, calib_n, src_h, src_w, th)
    for k in range(calib_n):
        yield (frames[k],
               (_CpuFrameRef(ctx.calib_owner, k),
                ctx.calib_owner.ptr + k * fnb, src_h, src_w),
               float(sums[k, 0]), float(sums[k, 1]))
    prev_owner = ctx.calib_owner   # 上一批缓冲（fill_prev 读取期间保活）
    prev_ptr = ctx.calib_owner.ptr + (calib_n - 1) * fnb
    for bstart in range(calib_n, len(frames), DECODE_BATCH):
        bend = min(bstart + DECODE_BATCH, len(frames))
        B = bend - bstart
        nds = vr.get_batch(frames[bstart:bend], roi=roi)
        crops = nds.asnumpy()
        if yuv:
            if crops.ndim != 3:
                raise RuntimeError(
                    "GPU yuv 分段仅支持 decord yuv420 输出")
        else:
            if crops.ndim != 4 or crops.shape[-1] != 1:
                raise RuntimeError(
                    "GPU 分段仅支持 decord gray 输出")
        g = np.ascontiguousarray(ex._batch_luma(crops))
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
        for k in range(B):
            src = prev_ptr if k == 0 else base + (k - 1) * fnb
            cudart.cudaMemcpyAsync(
                prev_buf + k * fnb, src, fnb, _d2d,
                analyzer._stream)
        sums = analyzer.analyze_batch(
            base, prev_buf, B, src_h, src_w, th)
        for k in range(B):
            yield (frames[bstart + k],
                   (_CpuFrameRef(owner, k),
                    base + k * fnb, src_h, src_w),
                   float(sums[k, 0]), float(sums[k, 1]))
        prev_owner = owner
        prev_ptr = base + (B - 1) * fnb


class _GpuPipelineMixin:
    # ═══════════════ GPU 全驻留管线（NVDEC） ═══════════════

    def _gpu_pipeline_enabled(self) -> bool:
        """GPU 全驻留零拷贝管线：NVDEC 直通或 CPU 解码 + H2D（P1-3）。

        默认（GPU_PIPELINE 未设置）启用条件（全部满足）：
        - decode_backend ∈ {auto, nvdec, cpu, hybrid}：auto/nvdec 走 NVDEC
          设备指针直通（NVDEC 打开失败时回退 CPU 解码分支）；cpu 显式
          选择 CPU 软解 + H2D 进 GPU 分段/OCR（P1-3 解耦——CPU 解码的
          墙钟收益与零拷贝 OCR 不再互斥）；hybrid 走 CPU 分支消费
          HybridDecoder 交付的宿主数组（§8.3：双解码收益 + 零拷贝 OCR
          叠加，原互斥门控已移除）。
        - TensorRT 可用且 ocr_backend ≠ cpu —— 全程 raw 才有净收益
          （GPU 分段+ONNX 实测无优势，默认走宿主管线，配置面更简）
        - cuda-python（cuda.core / cuda.bindings）可导入
        force_aspect 已支持（contrast 模式已随 0.9.0 删除，不再门控回退）。

        env GPU_PIPELINE：'0' 显式关闭；'1' 强制尝试（跳过 TRT 要求，
        允许 GPU 分段+ONNX 等实验组合）；不设置 = 上述默认规则。
        """
        _env = _os.environ.get(config.GPU_PIPELINE_ENV)
        if _env is not None:
            if not config.env_bool(config.GPU_PIPELINE_ENV, default=False):
                return False
            forced = True
        else:
            forced = False
        backend = (self._decode_backend or 'auto').lower()
        if backend not in ('auto', 'nvdec', 'cpu', 'hybrid'):
            return False
        if not _cuda_python_available():
            return False
        if not forced:
            if (self._ocr_backend or 'auto').lower() == 'cpu':
                return False
            if not tensorrt_available():
                return False
        if backend == 'cpu':
            # CPU 解码分支不依赖 NVDEC：跳过 nvdec 探测（避免无谓的
            # GPU reader 试开；TRT 可用性已由上方门控确认）。
            return True
        return nvdec_available(str(self._video_path))

    def _run_pipelined_gpu(self, _ocr_engines: list | None = None):
        """GPU 全驻留零拷贝路径：灰度/sharp/聚类/合并判定/OCR 全在 GPU。

        _ocr_engines：内部复用 OCR 引擎时传入（B5）；None 走进程级引擎池。
        过 RAM 的只有：每帧两个标量（sharp/cluster）、校准直方图表、
        merge_similar 两标量、keep_crops 输出（每段一张 D2H，结果必须
        给外部）与 OCR 回退路径（ONNX/无 TRT/引擎未就绪时代表帧 D2H）。
        两种代表帧格式（NVDEC）：
        - gray：代表帧即 decord gray NDArray（owner 保活），raw OCR 直通；
        - yuv：代表帧为 packed NV12（owner 保活），Y 平面按需提取到
          _YFramePool 池帧供 raw OCR / GPU 合并判定（~10KB D2D/次）；
          完整 NV12 仅 keep_crops 时 D2H。
        合并判定（sim_pair kernel）与宿主 _segments_similar 语义对应
        （整数精确；除对比阈值处的 float32 末位舍入外逐位一致）。
        contrast 模式已删除（0.9.0，见 _similar_device）。

        P1-3 解耦（decode=cpu 或 auto/nvdec 的 NVDEC 打开失败回退）：
        每批 asnumpy → 宿主灰度 → H2D → 同一 hist/analyze kernel，rep 帧
        留在显存供 raw OCR —— CPU 解码的墙钟收益（P0-1 高线程）与零拷贝
        OCR 不再互斥。rep 的 keep_crops / OCR 回退走宿主切片直取
        （_d2h_rep 无 D2H），设备侧恒为灰度（yuv 也只上载 Y）。
        返回格式与 _run_pipelined 相同。
        """
        from queue import Empty, Full, Queue
        import threading
        from cuda.bindings import runtime as cudart
        from ocr_trt import GpuFrameAnalyzer
        # Cleanup handles are initialized before any calibration/setup can fail.
        ocr_session = None
        analyzer = None
        pool = None
        _y_pool = None
        calib_owner = None
        producer = None
        producer_stop = threading.Event()
        _t_open = time.perf_counter()
        vr = self._open_vr()
        # decord 原生 hybrid：OCR on GPU 时 reader 是 hybrid_gpu（输出 CUDA
        # 批，设备指针通路可用）；OCR on CPU 时是 hybrid（宿主帧）。
        # （旧项目层 HybridDecoder 已移除 —— 混合解码完全由 decord 承担。）
        on_gpu = (self._backend == 'decord/GPU'
                  or (self._backend == 'decord/hybrid' and self._ocr_on_gpu()))

        ctx = _GpuRunCtx()

        def _cleanup_partial() -> None:
            """收尾尚未进入主消费循环的资源（会话 + 设备侧，见
            _gpu_release_partial）。"""
            nonlocal ocr_session
            if ocr_session is not None:
                try:
                    ocr_session.finish()
                except BaseException:
                    pass
                ocr_session = None
            _gpu_release_partial(ctx)
        if self._fps is None:
            _fps = _read_fps_from_vr(vr)
            self._fps = _fps if _fps else config.DEFAULT_FPS_FALLBACK
        x1, y1, x2, y2 = self._roi
        total = len(vr)
        if (self._frame_end or 0) > total:
            logger.warning('frame_end=%s 超出视频总帧数 %d，按片尾截断',
                           self._frame_end, total)
        end = min(self._frame_end or total, total)
        frames = list(range(self._frame_start, end, self._sample_stride))
        if not frames:
            try:
                vr.close()
            except Exception:
                pass
            raise ValueError(
                f"帧区间为空: frame_start={self._frame_start}, "
                f"frame_end={end}, total={total}")
        hybrid = hasattr(vr, 'hybrid_begin')
        try:
            if self._frame_start > 0 and not hybrid:
                # hybrid 分片定位由生产者在片首完成（其 seek_accurate 已显式
                # 报错，DESIGN-REVIEW B4）——跳过外部 seek。
                vr.seek_accurate(self._frame_start)
            if hybrid:
                vr.hybrid_begin(frames)
        except BaseException:
            try:
                vr.close()
            except Exception:
                pass
            raise
        self._prof_end('producer', 'open_and_fps', _t_open)
        # OCR 会话（引擎初始化/模型加载）提前到校准前启动：worker 线程内
        # 构建引擎，与校准（前 50 帧 hist+Otsu）并行重叠；引擎就绪前
        # _emit_ocr 自动走 host 回退（raw_ready=False），语义不变。
        try:
            ocr_session = self._start_ocr_session(_ocr_engines)
        except BaseException:
            try:
                vr.close()
            except Exception:
                pass
            raise
        results = ocr_session.results
        ocr_err = ocr_session.err
        ocr_wall = ocr_session.wall
        _put_ocr = ocr_session.put

        yuv = self._yuv_output
        yuv = self._yuv_output

        try:
            _calib_ok, _th = _gpu_prepare_calibration(
                self, ctx, vr, frames, on_gpu=on_gpu, yuv=yuv,
                roi=(x1, y1, x2 + 1, y2 + 1))
        except BaseException:
            _cleanup_partial()
            try:
                vr.close()
            except Exception:
                pass
            raise
        if not _calib_ok:
            return _gpu_fallback_to_host(self, ctx, vr, ocr_session,
                                         _ocr_engines)

        self._gpu_pipeline_mode = True
        if on_gpu:
            frame_stream = _gpu_frame_stream_nvdec(
                self, ctx, vr, frames, yuv=yuv,
                roi=(x1, y1, x2 + 1, y2 + 1), th=_th)
        else:
            frame_stream = _gpu_frame_stream_cpu(
                self, ctx, vr, frames, yuv=yuv,
                roi=(x1, y1, x2 + 1, y2 + 1), th=_th)

        # 生产者线程：解码 + GPU analyze 与主线程分段/OCR 重叠
        producer_q: Queue = Queue(maxsize=max(8, self._buffer_size))
        producer_err: list = []
        # C6：消费端任何异常都会跳出消费循环——producer 必须可被叫停，
        # 否则永久阻塞在 put（daemon 线程 + 解码器 + 设备引用泄漏）。

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
                for item in frame_stream:   # 模块生成器（0.11.0 拆分后为对象非函数）
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
        # 零拷贝管线：代表帧只在两处过 RAM —— keep_crops 输出（每段一张
        # D2H）与 OCR 回退路径（ONNX/无 TRT/引擎未就绪）。merge_similar
        # 判定在 GPU（sim_pair 整数精确）；yuv 的 Y 平面按需从保留的 NV12
        # 提取（luma_into → 池帧，~10KB D2D/次）。
        raw_ready_ref = ocr_session.raw_ready
        ctx.y_pool = (_YFramePool(ctx.src_h * ctx.src_w)
                      if (yuv and on_gpu) else None)
        _limited = self._color_range != 1

        def _d2h_rep(dev, *, prefer_device=False):
            """代表帧 → 宿主：NVDEC = D2H；CPU 解码默认宿主切片直取。

            raw GPU OCR 的 CPU 解码代表帧改为从设备单帧 D2H，随后可释放
            整批宿主数组；宿主 OCR 回退仍使用切片拷贝。
            """
            hc = getattr(dev[0], 'host_crop', None)
            if hc is not None and not prefer_device:
                h = hc()
                if h is not None:
                    return np.array(h)
                # 宿主副本已被 drop_host() 释放 → 落到下面的设备 D2H
            from cuda.bindings import runtime as cudart
            arr = np.empty((dev[2], dev[3]), dtype=np.uint8)
            cudart.cudaMemcpy(arr.ctypes.data, int(dev[1]), dev[2] * dev[3],
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            return arr

        def _autocrop_device(gray_ptr, sharp):
            """P0-4 GPU 直通裁切：col_ink 判 rep 帧「有墨迹列范围」+
            宿主同一余量规则（_content_range_to_crop）。

            与宿主 _crop_to_content 同判据（g > th、每列 ≥2 墨迹像素、
            余量与最小收益门槛同 `_content_range_to_crop`）；sharp = rep 帧
            GPU analyze 的 std（对 yuv 已是展开后的 Y，与宿主一致）。返回
            (x_off, crop_w) 或 None（不裁）。"""
            # ⚠️ 这里**不能**再用 `force_aspect > 0` 跳过裁切。
            # 原判据是"宽度被强制，裁切只改缩放不省宽"——那是把裁后区间
            # 拉伸到 force 宽度（顺序 ⑥）才会有的结论，而 ⑥ 实测更差
            # （test5 9 vs 不裁 7）。改成"裁后按整幅同一比例缩放"
            # （顺序 ⑦，GPU 侧在 process_gray_raw 里实现）后，fa>0 下
            # 裁切从"无收益"变成**显著收益**：test5 7→0、test6 17→0。
            #
            # fa=0 的视频由 `_content_range_to_crop` 的最小收益门槛自动
            # 处理：紧凑 ROI 的段裁掉量通常 <10%，低于门槛 → 不裁。
            # （旧注释称"fa=0 时裁切退化 test2 52→80、test 78→127"，那是
            #  实验变体探针的数据，非生产路径；生产门禁实测 fa=0 的
            #  test/test2 在门槛下与不裁持平，见 config 中的实测表。）
            if not self._ocr_autocrop:
                return None
            if ctx.src_w <= 8 or sharp < 3.0:
                return None
            rng = ctx.analyzer.content_range(int(gray_ptr), ctx.src_h,
                                             ctx.src_w, self._bin_thresh)
            if rng is None:
                return None
            return self._content_range_to_crop(rng[0], rng[1], ctx.src_w)

        def _similar_device(a_dev, b_dev) -> bool:
            """merge_similar 判定：GPU sim_pair（整数精确，与宿主 float32
            均值仅差末位舍入）。contrast 模式已随 0.9.0 删除
            （_merge_effective_mode 只会返回 binary/''，未知值映射 binary），
            不再有宿主回退分支（DESIGN-REVIEW B3 死代码清理）。"""
            if not (self._merge_similar and a_dev is not None
                    and b_dev is not None):
                return False
            use_bin = 1 if self._merge_effective_mode() == 'binary' else 0
            ya = yb = None
            if yuv and on_gpu:
                # 仅 NVDEC yuv 需要 Y 提取；CPU 解码分支设备侧恒为灰度。
                ya = ctx.y_pool.acquire()
                yb = ctx.y_pool.acquire()
                ctx.analyzer.luma_into(int(a_dev[1]), int(ya.ptr), ctx.src_h,
                                       ctx.src_w, _limited)
                ctx.analyzer.luma_into(int(b_dev[1]), int(yb.ptr), ctx.src_h,
                                       ctx.src_w, _limited)
                ap, bp = ya.ptr, yb.ptr
            else:
                ap, bp = int(a_dev[1]), int(b_dev[1])
            try:
                mad, chg = ctx.analyzer.compare_pair(
                    ap, bp, ctx.src_h, ctx.src_w, self._bin_thresh, use_bin)
            finally:
                # 池帧引用释放（GC 归还）
                ya = yb = None
            n = ctx.src_h * ctx.src_w
            mean = 255.0 * mad / n if use_bin else mad / n
            return similar_decision(mean, chg,
                                    self._merge_similar_threshold,
                                    self._merge_max_changed_pixels)

        def _emit_ocr(idx, r_frame, r_dev, frac, r_sharp) -> None:
            _t_push = time.perf_counter()
            _raw = raw_ready_ref[0] and r_dev is not None
            crop_h = None
            dev_ocr = None
            if _raw:
                # 零拷贝：gray/CPU 解码直接帧指针；NVDEC yuv 提取 Y 到池帧
                #（owner=池帧，OCR worker 用毕 GC 归还）。
                if yuv and on_gpu:
                    yf = ctx.y_pool.acquire()
                    ctx.analyzer.luma_into(int(r_dev[1]), int(yf.ptr),
                                           ctx.src_h, ctx.src_w, _limited)
                    dev_ocr = (yf, yf.ptr, ctx.src_h, ctx.src_w)
                else:
                    dev_ocr = r_dev
                # P0-4 GPU 直通：rep 帧宽度自适应裁切（col_ink + 宿主同一
                # 余量规则）；未裁切时 (0, src_w) 与旧全宽语义逐位一致。
                xoff, cropw = 0, ctx.src_w
                rng = _autocrop_device(dev_ocr[1], r_sharp)
                if rng is not None:
                    xoff, cropw = rng
                dev_ocr = (dev_ocr[0], dev_ocr[1], ctx.src_h, ctx.src_w,
                           xoff, cropw)
            else:
                # 回退（ONNX/无 TRT/引擎未就绪）：代表帧 D2H → 宿主预处理，
                # crop 与 keep_crops 共用同一副本。
                crop_h = _d2h_rep(r_dev)
            if self._keep_crops:
                # keep_crops 是结果输出（给外部转 RGB），不可避免的传输。
                # CPU 解码 raw 路径从单帧设备缓冲复制，避免钉住整批 host 数组。
                rep_crops[r_frame] = (crop_h if crop_h is not None
                                      else _d2h_rep(
                                          r_dev,
                                          prefer_device=(dev_ocr is not None
                                                         and not yuv)))
            if dev_ocr is not None and not yuv:
                drop_host = getattr(dev_ocr[0], 'drop_host', None)
                if drop_host is not None:
                    drop_host()
            _put_ocr((idx, r_frame, crop_h, dev_ocr, frac))
            self._prof_end('producer', 'q_put_block', _t_push)

        # 分段状态机与宿主管线共用（segmentation.SegmentStateMachine）：
        # GPU 侧数据源是设备侧 kernel 已算好的 win3 分数（cluster）。
        def _on_emit(seg, rep, frac):
            nonlocal seg_idx
            _emit_ocr(seg_idx, rep[0], rep[1], frac, rep[2])
            seg_idx += 1

        machine = SegmentStateMachine(
            frames, C=self._C,
            on_emit=_on_emit,
            on_similar=lambda a, b: _similar_device(a[1], b[1]),
            on_cancel=self._cancel,
            on_progress=lambda kk, frac: self._progress(
                f'[{self._backend}] GPU分段: {kk}/{len(frames)}',
                _decode_progress_pct(frac)),
            debug_tag='GB')

        try:
            while True:
                try:
                    item = producer_q.get(timeout=0.2)
                except Empty:
                    # C7：解码停滞/等待期间保持取消响应（取消由 cancel_check
                    # 抛异常表达，见构造参数注释）
                    self._cancel()
                    continue
                if item is None:
                    break
                if producer_err:
                    raise RuntimeError(
                        f"GPU 解码生产者失败: {producer_err[0]!r}"
                    ) from producer_err[0]
                fi, dev, sharp, cluster = item
                machine.feed(k, fi, sharp, (fi, dev, sharp),
                             cluster=float(cluster))
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
                    pass
            _t_consume_end = time.perf_counter()
            self.timing['decode'] = _t_consume_end - t0
            try:
                ocr_session.finish()
            except BaseException:
                pass
            self.timing['ocr_tail'] = time.perf_counter() - _t_consume_end
            try:
                vr.close()   # HybridDecoder：显式停生产者线程；decord VR 无此方法
            except Exception:
                pass
            # C5：释放本次 extract 的临时设备缓冲（分析器/池）。OCR 引擎
            # 缓冲归进程级引擎池管理（_start_ocr_session），不在此释放。
            _gpu_release_partial(ctx)
        if ocr_err:
            raise RuntimeError(f"OCR worker 失败: {ocr_err[0]!r}") from ocr_err[0]
        self.timing['ocr'] = ocr_wall[0]
        self._n_segments = len(segs)
        self.crops = rep_crops
        del vr
        self._ocr_texts = [results[i][0] for i in range(seg_idx)]
        self._ocr_confs = [results[i][1] for i in range(seg_idx)]
        return (frames, segs, self._ocr_texts, self._ocr_confs,
                [results[i][2] for i in range(seg_idx)])

