"""GPU 设备侧机制（S9-5 自 _gpu_pipeline.py 迁入）：池 / 帧流 / 校准 / 释放。

缓冲区（_YFrame/_YFramePool/_DevBatch/_DevBatchPool/_CpuFrameRef）、
运行上下文（_GpuRunCtx）、校准（_gpu_prepare_calibration）、帧流
（_gpu_frame_stream_nvdec/cpu）、释放（_gpu_release_partial）与
cuda-python 可用性探测。编排在 pipeline/gpu_backend.py；引用类型在
gpu/frame_ref.py。

§10.4 patch 点：nvdec_available / tensorrt_available（经本模块属性解析，
tests/pipeline/test_gpu_pipeline.py patch 面）。
"""
import logging
import time

import numpy as np

from video_ocr_engine.config import constants as config
from video_ocr_engine.domain.video_utils import nvdec_available, tensorrt_available  # noqa: F401 —— §10.4 monkeypatch 点（经 extractor._gpu_pipeline_enabled 使用）
from video_ocr_engine.domain.segmentation import otsu_median_threshold, _otsu_from_hist
from .frame_ref import DeviceRef
from .._helpers import _ndarray_device_ptr

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
        # B6（S4）：析构在任意 GC 上下文/解释器关闭期运行——只做入列回收，
        # 溢出（空闲列满）时告警并放弃该块（泄漏有界：≤ 池上限+在飞数，
        # release_all 于 teardown 释放在列块；CUDA 释放只走显式路径）。
        try:
            self.pool._recycle(self)
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

    def _recycle(self, frame: _YFrame) -> None:
        """入列回收（__del__ 安全路径：无任何 CUDA 调用）。"""
        if len(self._free) < self._MAX:
            self._free.append(frame)
            return
        # B6：溢出不在析构上下文里 cudaFree——告警并放弃（有界泄漏，
        # 由 release_all/进程退出兜底）
        logger.warning("_YFramePool 空闲列满，__del__ 放弃回收一块 %d B 显存"
                       "（B6：CUDA 释放只走显式路径）", self._fnb)

    def recycle(self, frame: "_YFrame") -> None:
        """显式归还（S4）：先置空 pool 引用防 __del__ 双重入列，再入列。"""
        frame.pool = None
        self._recycle(frame)

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
        # B6（S4）：同 _YFrame——只入列，溢出告警不 cudaFree
        try:
            self.pool._recycle(self)
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

    def _recycle(self, b: _DevBatch) -> None:
        """入列回收（__del__ 安全路径：无任何 CUDA 调用；B6/S4）。"""
        b.host = None
        if len(self._free) < self._MAX:
            self._free.append(b)
            return
        logger.warning("_DevBatchPool 空闲列满，__del__ 放弃回收一块 %d B 显存"
                       "（B6：CUDA 释放只走显式路径）", self._nbytes)

    def recycle(self, b: "_DevBatch") -> None:
        """显式归还（S4）：先置空 pool 引用防 __del__ 双重入列，再入列。"""
        b.pool = None
        self._recycle(b)

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
    ex._bin_thresh = th
    return True, th



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
        else:
            for bstart in range(ctx.calib_n, len(frames), DECODE_BATCH):
                bend = min(bstart + DECODE_BATCH, len(frames))
                # roi 不随批传：打开 reader 时 SetRoi 已生效；hybrid 原生
                # 路径每次 get_batch 传 roi 会触发 fork 侧 SetRoi/池深重算
                # （2026-09-10 实测 hybrid av1 2487→2264 fps，-9%；
                # _probe_roi_decode 已证 CPU 路径两种传法等价）。
                _t_dec = time.perf_counter()
                nds = vr.get_batch(frames[bstart:bend])
                # S6-d：解码 vs analyze 的相位划分（双缓冲 A/B 的判据；
                # 两者都在生产者线程内串行，重叠空间 = min(两者)）
                ex._prof_end('producer', 'decode_batch', _t_dec)
                yield bstart, nds

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
        ex._prof_end('producer', 'stream_analyze', _t_an)
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
        _gpu_fill_prev(analyzer, prev_buf, base, B, fnb, prev_ptr)
        sums = analyzer.analyze_batch(
            base, prev_buf, B, src_h, src_w, th)
        for k in range(B):
            yield (frames[bstart + k],
                   DeviceRef(ptr=base + k * fnb, h=src_h, w=src_w,
                             owner=_CpuFrameRef(owner, k)),
                   float(sums[k, 0]), float(sums[k, 1]))
        prev_owner = owner
        prev_ptr = base + (B - 1) * fnb
