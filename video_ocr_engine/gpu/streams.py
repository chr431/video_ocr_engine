"""GPU 设备侧协作流。

DevHooks（设备函数显式依赖）、_GpuRunCtx（单次运行共享设备状态）、校准
（_gpu_prepare_calibration）、两条帧流（NVDEC 设备直通 / CPU 解码
H2D）与部分释放（_gpu_release_partial）。编排在 pipeline/gpu_backend.py；
池机制在 gpu/pools.py；可用性探测留在 gpu/device.py（monkeypatch 点）。
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from video_ocr_engine.config import constants as config
from video_ocr_engine.domain.segmentation import (  # 等价性由金标向量守护（C-32）
    _otsu_from_hist,
    otsu_median_threshold,
)
from .frame_ref import DeviceRef
from .pools import _CpuFrameRef, _DevBatchPool
from .._helpers import _ndarray_device_ptr

if TYPE_CHECKING:  # 仅类型面（运行时零导入成本）
    from video_ocr_engine.ocr.trt import GpuFrameAnalyzer
    from .pools import _DevBatch

logger = logging.getLogger(__name__)


@dataclass
class DevHooks:
    """设备侧协作函数的显式依赖。

    limited    —— color_range != 1（YUV 的 luma 展开口径）
    batch_luma —— CPU 解码分支的宿主灰度回调 (B,H,W[,C])->(B,h,w)
    bin_ref    —— [th] 单元素盒：校准写入、判定回调读
    on_bin     —— 阈值即时回写（流式合并活读；可 None）
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
    """run_gpu_pipeline 单次运行的共享设备状态（显式替代闭包捕获）。

    字段由 _gpu_prepare_calibration 填充，帧流生成器与消费循环只读/推进；
    _gpu_release_partial 负责统一释放。"""

    __slots__ = (
        "analyzer",
        "pool",
        "y_pool",
        "calib_owner",
        "calib_nds",
        "calib_base",
        "calib_gray",
        "src_h",
        "src_w",
        "fnb",
        "prev_holder",
        "prev_ptr",
        "calib_n",
        "pair_cache",
        "pull_q",
        "pull_err",
        "pull_stop",
        "pull_thread",
        "pull_roi",
    )

    def __init__(self) -> None:
        # 惰性协作者：_gpu_prepare_calibration 填充、帧流只读（生命周期
        # 由 _driver 顺序保证）；消费端以 assert 收窄（非 strict 家风）
        self.analyzer: "GpuFrameAnalyzer | None" = None
        self.pool: "_DevBatchPool | None" = None
        self.y_pool: "_DevBatchPool | None" = None
        self.calib_owner: "_DevBatch | None" = None
        # decord NDArray（follow_imports=skip 下不透明）
        self.calib_nds: Any | None = None
        self.calib_base = 0
        self.calib_gray = 0
        self.src_h = 0
        self.src_w = 0
        self.fnb = 0
        self.prev_holder: Any | None = None
        self.prev_ptr = 0
        self.calib_n = 0
        # 边界对投机缓存（C-84）：生产者侧预计算 (fi_a,fi_b)→(mad,chg,win3)，
        # 消费端 similar() 命中直取、未命中走同步兜底；帧流启动前清空重建
        self.pair_cache: dict = {}
        # 拉流预读线程（C-84，_gpu_pull_start 填充/收口）
        self.pull_q: Any = None
        self.pull_err: list = []
        self.pull_stop: Any = None
        self.pull_thread: Any = None
        self.pull_roi: tuple = ()


def _gpu_fill_prev(analyzer, prev_buf, base, B, fnb, first_src) -> None:
    """prev 缓冲错位填充（三处调用共用：NVDEC 帧流 / CPU 帧流校准 / 批循环）。

    第 k 行 = (first_src if k==0 else base+(k-1)*fnb) 的 fnb 字节——
    供 analyze_batch 读"上一帧"；D2D 异步到 analyzer._stream。

    逐帧 B 次 cudaMemcpyAsync 收拢为 2 次：行 1..B-1 恰好是
    base 的连续区段（base[0..B-2]），一次大拷贝逐位等价（prev_buf 与
    base 恒为不同缓冲，无别名）。每次 cudaMemcpyAsync 有 ~10-20µs 的
    Python+驱动提交开销，B=64 时每批省 ~1ms——hybrid 引擎口径下解码
    批耗时 ~25ms，这笔开销占在引擎内 hybrid 增益流失（2601→2132fps）
    的可归因部分。"""
    from cuda.bindings import runtime as cudart

    _d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    if B > 1:
        cudart.cudaMemcpyAsync(prev_buf + fnb, base, (B - 1) * fnb, _d2d, analyzer._stream)
    cudart.cudaMemcpyAsync(prev_buf, first_src, fnb, _d2d, analyzer._stream)


def _gpu_analyzer_prewarm_start():
    """后台预热 GpuFrameAnalyzer，返回 (thread, holder) 句柄。

    构造的固定成本（冷进程实测 ~0.43s：cuda.core 导入 0.233 + 首次
    Device 上下文 0.093 + 模块导入 0.107；cubin 磁盘缓存命中后编译
    ~0）默认全落在校准相位——GPU lane 对宿主 lane 的墙钟差（nvdec
    +TRT，2026-10-14 实测 3.52 vs 3.30s）主项即此（C-84）。从
    run_gpu_pipeline 入口启动本线程，把构造藏进 reader 打开
    （open_and_fps ~0.4-0.6s）背后；失败静默（holder 空）——校准回退
    内联构造，行为不变。线程内构造与 OCR worker 的 TrtEngine 初始化
    并发安全（cuda.core 导入有双检锁；primary ctx retain 与
    cuModuleLoadData 均线程安全）。"""
    import threading

    holder: list = []

    def _build() -> None:
        try:
            from video_ocr_engine.ocr.trt import GpuFrameAnalyzer

            holder.append(GpuFrameAnalyzer())
        except BaseException:  # noqa: S110 — 预热失败回退内联构造（见上）
            pass

    t = threading.Thread(target=_build, daemon=True, name="voe-analyzer-prewarm")
    t.start()
    return (t, holder)


def _gpu_analyzer_prewarm_take(warm) -> "Any | None":
    """取回预热产物（join 等待完成）；未启动/失败返回 None。"""
    if warm is None:
        return None
    t, holder = warm
    t.join()
    return holder[0] if holder else None


def _spec_pairs_flush(analyzer, ctx, pairs: list, th: int, use_bin: int) -> None:
    """投机对批量落账：一次 launch + 一次 D2H + 一次流同步，结果写
    ctx.pair_cache[(fi_a, fi_b)] = (mad, chg, win3)。空表直接返回。

    调用点在批 sums 同步之后（规划器需要 sharp/cluster 值）、批 yield
    之前（消费端见到该批时缓存必须就绪）。"""
    if not pairs:
        return
    res = analyzer.compare_pairs_batch(
        [p[0][1].ptr for p in pairs],
        [p[1][1].ptr for p in pairs],
        ctx.src_h,
        ctx.src_w,
        th,
        bool(use_bin),
        stream=analyzer._stream,
    )
    cache = ctx.pair_cache
    for pair, row in zip(pairs, res):
        cache[(pair[0][0], pair[1][0])] = (int(row[0]), int(row[1]), int(row[2]))


def _gpu_pull_start(ctx: "_GpuRunCtx", hooks: DevHooks, vr, frames: list, calib_n: int) -> None:
    """拉流预读线程（C-84）：校准期即启动的唯一 get_batch 调用者。

    动机：解码/分析/校准的串行链上，analyzer 冷构造等待（~0.2s，
    _gpu_analyzer_prewarm_start 已把大头藏进后台但校准仍要等尾段）与
    解码器冷启动原本全部串行落在墙钟上。本线程从校准相位就开始拉流
    （校准批 → 主批 → 队列），analyzer 等待期间解码器已全速跑批——
    预读深度 GPU_PIPELINE_PULL_DEPTH 批（VRAM ~1.5MB，fork 侧批缓冲
    池化后无 cudaMalloc/Free 抖动）。

    线程纪律：get_batch 全部调用恒在此线程（reader 单线程不变量）；
    主批不传 roi（SetRoi 已在 open 生效，逐批传 roi 触发 fork 池深
    重算，实测 hybrid −9%）；校准批传 roi（既有语义）。校准/消费/
    回退各路径经 ctx.pull_q 交付，_gpu_pull_stop 统一收口。
    """
    import queue as _queue
    import threading as _threading

    from video_ocr_engine.domain.resources import register_thread

    DECODE_BATCH = config.GPU_PIPELINE_DECODE_BATCH
    q: "_queue.Queue" = _queue.Queue(maxsize=config.GPU_PIPELINE_PULL_DEPTH)
    err: list = []
    stop = _threading.Event()

    def _put(item) -> None:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return
            except _queue.Full:
                continue

    def _pull() -> None:
        try:
            _put((0, vr.get_batch(frames[:calib_n], roi=ctx.pull_roi)))
            for bstart in range(calib_n, len(frames), DECODE_BATCH):
                _t_dec = time.perf_counter()
                nds = vr.get_batch(frames[bstart : bstart + DECODE_BATCH])
                hooks.tick("producer", "decode_batch", _t_dec)
                _put((bstart, nds))
        except BaseException as e:
            err.append(e)
        finally:
            _put(None)

    t = _threading.Thread(target=_pull, daemon=True, name="voe-decode-pull")
    t.start()
    register_thread("pull", t)
    ctx.pull_q = q
    ctx.pull_err = err
    ctx.pull_stop = stop
    ctx.pull_thread = t


def _gpu_pull_stop(ctx: "_GpuRunCtx") -> None:
    """拉流线程收口（stop_consume / 回退 / 释放路径统一调用）。

    先置停再排空队列：线程的 put 循环见停即退，在途 get_batch 至多
    一个批的解码时长；join 带超时（卡解码时不无限等，daemon 兜底——
    回退宿主路径复用 reader，超时告警暴露竞态风险而非静默）。
    """
    t = getattr(ctx, "pull_thread", None)
    if t is None:
        return
    ctx.pull_stop.set()
    q = ctx.pull_q
    while q is not None:
        try:
            q.get_nowait()
        except Exception:
            break
    t.join(2.0)
    if t.is_alive():
        logger.warning(
            "拉流预读线程 2s 内未退出（在途 get_batch 卡住？）——回退路径"
            "复用 reader 存在竞态风险，请排查解码停滞"
        )
    ctx.pull_thread = None
    ctx.pull_q = None


def _gpu_release_partial(ctx: _GpuRunCtx) -> None:
    """释放尚未进入主消费循环的设备资源（池 / 校准缓冲 / 分析器）。"""
    # 拉流线程先收口：回退宿主路径会复用 reader，在途拉流必须停止
    _gpu_pull_stop(ctx)
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
        except Exception:  # noqa: S110 清理路径：release 失败无需上抛（进程退出兜底）
            pass  # 清理路径：release 失败无需上抛（进程退出兜底）
        ctx.analyzer = None


def _gpu_prepare_calibration(
    hooks: DevHooks,
    ctx: "_GpuRunCtx",
    vr,
    frames: list,
    *,
    on_gpu: bool,
    yuv: bool,
    roi: tuple,
    warm=None,
) -> "tuple[bool, int]":
    """GPU 管线校准：解码校准帧 + 逐帧直方图 Otsu（阈值取中位数）。

    填充 ctx（analyzer/pool/calib_owner/calib_nds/calib_base/calib_gray/
    src_h/src_w/fnb/calib_n）并写 hooks 阈值盒。返回 (ok, th)；
    False = 帧形状不符（GPU 分段不支持），调用方回退宿主管线。
    warm = 预热线线句柄（run_gpu_pipeline 入口启动，见
    _gpu_analyzer_prewarm_start）：命中即免内联构造。
    """
    from cuda.bindings import runtime as cudart
    from video_ocr_engine.ocr.trt import GpuFrameAnalyzer

    calib_n = ctx.calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
    # ── 拉流预读线程（C-84）：校准期即启动——首个 get_batch 含解码器
    # 冷启动，与 analyzer 预热等待并行；此后主批持续预读（深度
    # GPU_PIPELINE_PULL_DEPTH），消费端的分析/规划器等待全部与解码
    # 重叠。reader 单线程纪律：get_batch 全部调用恒在此线程。
    ctx.pull_roi = roi
    _gpu_pull_start(ctx, hooks, vr, frames, calib_n)
    ctx.analyzer = analyzer = _gpu_analyzer_prewarm_take(warm) or GpuFrameAnalyzer()
    _first = ctx.pull_q.get()  # 校准批（线程首产物）
    if ctx.pull_err:
        _gpu_pull_stop(ctx)
        raise ctx.pull_err[0]
    if _first is None:
        _gpu_pull_stop(ctx)
        raise RuntimeError("拉流预读线程未产出校准批（异常：%r）" % (ctx.pull_err or None,))
    if on_gpu:
        # ── NVDEC：decord 设备批直通（校准批同样不落 RAM）──
        ctx.calib_nds = _first[1]
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
                calib_base, calib_n, ctx.src_h, ctx.src_w, limited=hooks.limited
            )
        else:
            if len(calib_shape) != 4 or calib_shape[-1] != 1:
                # 灰度帧非 4D 单通道（部分 decord fork 输出 (B,H,W)）：GPU
                # 分段不支持，回退宿主。直接 run_host_pipeline（防递归）。
                return False, 0
            ctx.src_h, ctx.src_w = calib_shape[1], calib_shape[2]
            ctx.calib_gray = calib_base
        ctx.fnb = ctx.src_h * ctx.src_w
        ctx.prev_holder = ctx.calib_nds  # 保住前一 decord NDArray（防解码池复用）
        ctx.prev_ptr = calib_base  # 灰色模式：上一批/校准末帧 device 指针
    else:
        # ── CPU 解码：校准批 asnumpy → 宿主灰度 → H2D ──
        ctx.calib_nds = _first[1]
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
        assert hooks.batch_luma is not None  # CPU 解码路径契约（DevHooks）
        g = np.ascontiguousarray(hooks.batch_luma(crops))
        if g.shape != (calib_n, ctx.src_h, ctx.src_w):
            return False, 0
        # 池子必须容得下**校准批**（SEG_CALIB_FRAMES=50 行）：calib_owner 与
        # 后续解码批共用本池，校准 H2D/kernel 都按 calib_n 行访问。若按
        # DECODE_BATCH 定尺寸且 DECODE_BATCH < 50，校准批会向设备缓冲越界
        # 写 18 行 —— 越界破坏同 context 内其他分配（TRT 工作缓冲），异步
        # 错误延迟到 TRT enqueue 才爆（invalid argument），并污染 analyzer
        # 后续结果（段数漂移）。故尺寸取 max(默认 64, calib_n) ≥ 50。
        _pool = _DevBatchPool(max(config.GPU_PIPELINE_DECODE_BATCH, calib_n) * ctx.fnb)
        ctx.pool = _pool
        owner = _pool.acquire(crops)
        ctx.calib_owner = owner
        cudart.cudaMemcpyAsync(
            owner.ptr,
            g.ctypes.data,
            calib_n * ctx.fnb,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            analyzer._stream,
        )

    # 逐帧直方图校准：与单流水线"前 50 帧 Otsu 取中位数"语义逐位一致
    # （含退化双值帧的阈值行为），D2H 仅 B×1KB 标量表，校准帧不落 RAM。
    # 注意必须用 _otsu_from_hist（输入是直方图行）；_otsu 接收的是
    # 灰度图像并在内部做直方图——传错曾产生"直方图的直方图"垃圾阈值。
    if on_gpu:
        calib_gray_dev = ctx.calib_gray
    else:
        assert ctx.calib_owner is not None
        calib_gray_dev = ctx.calib_owner.ptr
    _hist_mat = analyzer.histograms_perframe(calib_gray_dev, calib_n, ctx.src_h, ctx.src_w)
    th = otsu_median_threshold([_otsu_from_hist(_hist_mat[k]) for k in range(calib_n)])
    hooks.set_bin(th)
    return True, th


def _gpu_frame_stream_nvdec(
    hooks: DevHooks,
    ctx: "_GpuRunCtx",
    vr,
    frames: list,
    *,
    yuv: bool,
    roi: tuple,
    th: int,
    C: float = 0.0,
    merge_on: bool = False,
    use_bin: int = 0,
):
    """NVDEC 设备批直通帧流：yield 批列表 [ (fi, sharp, kwargs), ... ]。

    dev 元组 gray=(nds,yptr,H,W) / yuv=(nds,nv12ptr,rows,W)；sharp/cluster
    由 analyze_batch（win3 分数）产出。prev 缓冲经 ctx 跨批衔接
    （prev_holder 保活 decord NDArray，防解码池复用）。

    merge_on 且 gray 时启用边界对投机预计算（C-84）：规划器按状态机的
    merge 无关几何逐帧推进，每批的边界对一次多对内核 + 拼车 D2H（在
    sums 同步后、yield 前的一次独立同步内完成），消费端 similar() 命中
    缓存即零 GPU 交互。yuv 的 gray 驻留在单块 _luma_dev 缓冲（每批
    extract_luma 覆盖），跨批投机读不到稳定数据 → 不投机，保持逐对
    同步路径。"""
    from cuda.bindings import runtime as cudart
    from video_ocr_engine.domain.segmentation import BoundaryPairPlanner

    DECODE_BATCH = config.GPU_PIPELINE_DECODE_BATCH
    _d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    limited = hooks.limited
    analyzer = ctx.analyzer
    assert analyzer is not None  # 生命周期：prepare_calibration 已填
    planner = BoundaryPairPlanner(C) if (merge_on and not yuv) else None

    def _analyze_batch(gray_base, prev_single, B, H, W):
        fnb = H * W
        prev_buf = analyzer._ensure_prev(max(B, DECODE_BATCH) * fnb)
        _gpu_fill_prev(analyzer, prev_buf, gray_base, B, fnb, prev_single)
        return analyzer.analyze_batch(gray_base, prev_buf, B, H, W, th), fnb

    def _walk(k0, base, sums, B, stride, dev_h, dev_w, owner, items, pairs, step=1):
        """批内逐帧：规划器喂料（投机对收集）+ 消费项组装。

        step = 帧号步进：校准批按 frames[k] 编号（step=stride，stride>1
        时 k0+k 是索引而非帧号——金标契约）；主批维持既有 k0+k 口径
        （step=1，历史已录向量）。"""
        for k in range(B):
            fi = k0 + k * step
            sharp = float(sums[k, 0])
            cluster = float(sums[k, 1])
            dev = DeviceRef(ptr=base + k * stride, h=dev_h, w=dev_w, owner=owner)
            if planner is not None:
                pr = planner.feed(fi, sharp, cluster, dev)
                if pr is not None:
                    pairs.append(pr)
            items.append((fi, sharp, {"payload": (fi, dev, sharp), "cluster": cluster}))

    # ── 校准帧整批分析（yuv 已在外部提取 Y → calib_gray）──
    B = ctx.calib_n
    sums, fnb = _analyze_batch(ctx.calib_gray, ctx.calib_gray, B, ctx.src_h, ctx.src_w)
    rows = ctx.src_h + (ctx.src_h + 1) // 2
    items: list = []
    pairs: list = []
    _calib_step = (frames[1] - frames[0]) if len(frames) > 1 else 1
    _walk(
        frames[0],
        ctx.calib_base,
        sums,
        B,
        (rows * ctx.src_w if yuv else fnb),
        (rows if yuv else ctx.src_h),
        ctx.src_w,
        ctx.calib_nds,
        items,
        pairs,
        step=_calib_step,
    )
    _spec_pairs_flush(analyzer, ctx, pairs, th, use_bin)
    ctx.prev_holder = ctx.calib_nds  # 保住前一 decord NDArray（防解码池复用）
    ctx.prev_ptr = ctx.calib_base + (B - 1) * (rows * ctx.src_w if yuv else fnb)
    yield items
    # yuv：末帧 Y 存入单帧缓冲（extract_luma 复用主缓冲会覆盖，
    # 下一批开始前必须已有独立副本）。注：_ensure_prev 首次按
    # 64*fnb 分配且批尺寸恒定 → prev_front 指针后续稳定不重分配。
    prev_front = None
    have_prev_front = False
    if yuv:
        prev_front = analyzer._ensure_prev(fnb)
        cudart.cudaMemcpyAsync(
            prev_front, ctx.calib_gray + (ctx.calib_n - 1) * fnb, fnb, _d2d, analyzer._stream
        )
        have_prev_front = True

    # ── 主批消费：拉流预读队列（C-84；校准期已开跑，解码器领先
    # 消费端至多 GPU_PIPELINE_PULL_DEPTH 批）。逐项 (bstart, nds)，
    # 哨兵 None；错误槽非空即重抛。旧的 GPU_PIPELINE_STREAM /
    # GPU_PIPELINE_DRAINER 环境变体（C-10 零收益判例）被本默认结构
    # 取代移除——预读从校准期启动是其零收益的根因修复。
    def _batch_iter():
        while True:
            item = ctx.pull_q.get()
            if item is None:
                if ctx.pull_err:
                    raise ctx.pull_err[0]
                return
            if ctx.pull_err:
                raise ctx.pull_err[0]
            yield item

    for bstart, nds in _batch_iter():
        bend = bstart + int(nds.shape[0])
        base, shape = _ndarray_device_ptr(nds)
        B = int(bend - bstart)
        if yuv:
            if len(shape) != 3:
                raise RuntimeError("GPU yuv 分段仅支持 decord yuv420 输出")
            H = shape[1] * 2 // 3
            W = shape[2]
            rows = H + (H + 1) // 2
            # extract 覆盖 _luma 主缓冲 → 先取上一批末帧副本
            gray_base = analyzer.extract_luma(base, B, H, W, limited)
            prev_s = prev_front if have_prev_front else gray_base
        else:
            if len(shape) != 4 or shape[-1] != 1:
                raise RuntimeError("GPU 分段仅支持 decord gray 输出")
            H, W = shape[1], shape[2]
            rows = H
            gray_base = base
            prev_s = ctx.prev_ptr
        _t_an = time.perf_counter()
        sums, fnb = _analyze_batch(gray_base, prev_s, B, H, W)
        hooks.tick("producer", "stream_analyze", _t_an)
        if yuv and have_prev_front:
            cudart.cudaMemcpyAsync(
                prev_front, gray_base + (B - 1) * fnb, fnb, _d2d, analyzer._stream
            )
            have_prev_front = True
        # 拉流线程交付的 bstart 是 frames 下标 → 帧号
        f0 = frames[bstart]
        items = []
        pairs = []
        _walk(
            f0,
            base,
            sums,
            B,
            (rows * W if yuv else fnb),
            (rows if yuv else H),
            W,
            nds,
            items,
            pairs,
        )
        _spec_pairs_flush(analyzer, ctx, pairs, th, use_bin)
        ctx.prev_holder = nds
        ctx.prev_ptr = base + (B - 1) * (rows * W if yuv else fnb)
        yield items
        if yuv:
            have_prev_front = True
    if planner is not None:
        # 流末尾段闭合的投机对（消费端 machine.finish() 的提问对象）
        fp = planner.finish()
        if fp is not None:
            _spec_pairs_flush(analyzer, ctx, [fp], th, use_bin)


def _gpu_frame_stream_cpu(
    hooks: "DevHooks",
    ctx: "_GpuRunCtx",
    vr,
    frames: list,
    *,
    yuv: bool,
    roi: tuple,
    th: int,
    C: float = 0.0,
    merge_on: bool = False,
    use_bin: int = 0,
):
    """CPU 解码帧流：get_batch → 宿主灰度 → H2D → analyze。

    每批缓冲从池取（引用归零归还）；上一批末帧作本批 analyze
    的 prev（fill_prev 读取期间由 prev_owner 保活）。analyze
    同步返回后本批帧指针即可交付（H2D/kernel 均已完成）。

    yield 批列表 [ (fi, sharp, kwargs), ... ]。merge_on 时启用边界对
    投机（C-84）——本分支设备内容恒为灰度（yuv 仅指输入着色，灰度已
    在宿主算好 H2D），池缓冲由 payload owner 保活，投机指针稳定。"""
    from cuda.bindings import runtime as cudart
    from video_ocr_engine.domain.segmentation import BoundaryPairPlanner

    DECODE_BATCH = config.GPU_PIPELINE_DECODE_BATCH
    _d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    _h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    fnb = ctx.fnb
    analyzer = ctx.analyzer
    assert analyzer is not None  # 生命周期：prepare_calibration 已填
    assert ctx.pool is not None
    pool = ctx.pool
    assert hooks.batch_luma is not None  # CPU 解码路径契约（DevHooks）
    calib_owner = ctx.calib_owner
    assert calib_owner is not None
    calib_n = ctx.calib_n
    src_h, src_w = ctx.src_h, ctx.src_w
    planner = BoundaryPairPlanner(C) if merge_on else None

    def _walk(k0, base, sums, B, owner, items, pairs, step=1):
        # step 语义同 nvdec 流：校准批 = frames[k]（金标契约），主批 = k0+k
        for k in range(B):
            fi = k0 + k * step
            sharp = float(sums[k, 0])
            cluster = float(sums[k, 1])
            dev = DeviceRef(ptr=base + k * fnb, h=src_h, w=src_w, owner=_CpuFrameRef(owner, k))
            if planner is not None:
                pr = planner.feed(fi, sharp, cluster, dev)
                if pr is not None:
                    pairs.append(pr)
            items.append((fi, sharp, {"payload": (fi, dev, sharp), "cluster": cluster}))

    prev_buf = analyzer._ensure_prev(max(calib_n, DECODE_BATCH) * fnb)
    # ── 校准帧整批分析（校准批已在外部 H2D → calib_owner）──
    _gpu_fill_prev(analyzer, prev_buf, calib_owner.ptr, calib_n, fnb, calib_owner.ptr)
    sums = analyzer.analyze_batch(calib_owner.ptr, prev_buf, calib_n, src_h, src_w, th)
    items: list = []
    pairs: list = []
    _calib_step = (frames[1] - frames[0]) if len(frames) > 1 else 1
    _walk(frames[0], calib_owner.ptr, sums, calib_n, calib_owner, items, pairs, step=_calib_step)
    _spec_pairs_flush(analyzer, ctx, pairs, th, use_bin)
    yield items
    prev_owner = calib_owner  # 上一批缓冲（fill_prev 读取期间保活）
    prev_ptr = calib_owner.ptr + (calib_n - 1) * fnb
    # ── 主批：拉流预读队列（C-84；calibrate 已启动线程，本循环只做
    # 宿主灰度 + H2D + analyze——与拉流线程的解码重叠）──
    while True:
        _item = ctx.pull_q.get()
        if _item is None:
            if ctx.pull_err:
                raise ctx.pull_err[0]
            break
        if ctx.pull_err:
            raise ctx.pull_err[0]
        bstart, nds = _item
        bend = min(bstart + DECODE_BATCH, len(frames))
        B = bend - bstart
        # CPU 解码分支也打桩（decode.*/stream_analyze）。键语义与宿主
        # 路径对齐：decode_batch = get_batch + asnumpy（宿主同款）；
        # gray_batch = 灰度转换。get_batch 时间在拉流线程侧，此处
        # span 只含 asnumpy。
        _t_dec = time.perf_counter()
        crops = nds.asnumpy()
        hooks.tick("producer", "decode_batch", _t_dec)
        if yuv:
            if crops.ndim != 3:
                raise RuntimeError("GPU yuv 分段仅支持 decord yuv420 输出")
        else:
            if crops.ndim != 4 or crops.shape[-1] != 1:
                raise RuntimeError("GPU 分段仅支持 decord gray 输出")
        _t_gray = time.perf_counter()
        g = np.ascontiguousarray(hooks.batch_luma(crops))
        hooks.tick("producer", "gray_batch", _t_gray)
        if g.shape != (B, src_h, src_w):
            raise RuntimeError(f"GPU(CPU解码) 灰度形状不符: {g.shape} != {(B, src_h, src_w)}")
        owner = pool.acquire(crops)
        cudart.cudaMemcpyAsync(owner.ptr, g.ctypes.data, B * fnb, _h2d, analyzer._stream)
        base = owner.ptr
        prev_buf = analyzer._ensure_prev(max(B, DECODE_BATCH) * fnb)
        _t_an = time.perf_counter()
        _gpu_fill_prev(analyzer, prev_buf, base, B, fnb, prev_ptr)
        sums = analyzer.analyze_batch(base, prev_buf, B, src_h, src_w, th)
        hooks.tick("producer", "stream_analyze", _t_an)
        items = []
        pairs = []
        _walk(frames[bstart], base, sums, B, owner, items, pairs)
        _spec_pairs_flush(analyzer, ctx, pairs, th, use_bin)
        yield items
        prev_owner = owner  # noqa: F841  # 保活：下批 fill_prev 读 prev_ptr（旧 owner 设备内存）期间不得回收
        prev_ptr = base + (B - 1) * fnb
    if planner is not None:
        # 流末尾段闭合的投机对（消费端 machine.finish() 的提问对象）
        fp = planner.finish()
        if fp is not None:
            _spec_pairs_flush(analyzer, ctx, [fp], th, use_bin)
