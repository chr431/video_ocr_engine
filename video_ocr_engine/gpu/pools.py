"""GPU 设备缓冲池机制（S4 设备侧拆分，2026-09-20 稳健性轮补做）。

自 gpu/device.py 机械搬迁（零逻辑改动）：_YFrame/_YFramePool（单帧 Y
设备缓冲，yuv 零拷贝关键件）与 _DevBatch/_CpuFrameRef/_DevBatchPool
（CPU 解码批双缓冲 owner）。池契约（复用换壳 C-55 / 放弃路径直释 /
release_all 置 _released）的完整依据见各类 docstring——本文件是池
语义的**唯一出处**，消费方只在 gpu/streams.py 与 pipeline/gpu_backend.py。
"""
import logging
import threading

logger = logging.getLogger(__name__)


class _YFrame:
    """池化的单帧设备 Y 缓冲（yuv 代表帧 Y 平面提取用）。

    随队列/闭包传递（作为 dev 元组的 owner），引用归零（GC）时自动
    归还 _YFramePool，不阻塞调用方。
    """

    __slots__ = ("pool", "ptr", "size", "_recycled")

    def __init__(self, pool, ptr, size):
        self.pool = pool
        self.ptr = ptr
        self.size = size
        self._recycled = False

    def __del__(self):
        # B6（S4）：优先入列回收（无 CUDA 调用，任意 GC 上下文安全）。
        # ⚠️ 2026-09-20 泄漏专项：**放弃路径必须直释**。旧语义「列满→告警
        # 放弃（CUDA 释放只走显式路径）」在两种真实场景下=永久泄漏：
        # ①轮末死亡波（gc.collect 集中析构数百帧，空闲列只有 32）；
        # ②release_all 之后到达的迟到帧（回收入**已 release 的孤儿池**，
        # 池↔帧互引成环，GC 收环时 _recycled=True 跳过释放）。实测
        # +2.0 MiB/轮（600 块 × 3498B，B 层 cudaMalloc 追踪定位）。
        # 因此列满/池已 release 时改走 cudaFree 直释；解释器关闭期 CUDA
        # 已卸载的失败由 except 兜底（单块 fnb，进程退出收）。
        #
        # ⚠️ 2026-09-19 审查轮二修：**pool 引用必须保留**。上一版让
        # recycle() 置空 pool 以"防双入列"，结果破坏了本函数——大量池帧
        # 是随 payload 跨线程传递、**只靠 GC 回收**的（见 gpu_backend
        # _similar_device 注释），pool 一置空，它们的 __del__ 撞 None 被
        # 吞掉 = 永久泄漏（实测 253 次 cudaMalloc/轮未释放、+1 MiB/轮）。
        # 防双入列改用 _recycled 标志位（语义等价，且不切断 GC 兜底）。
        if self._recycled:
            return
        self._recycled = True
        pooled = False
        try:
            pooled = self.pool._recycle(self)
        except Exception:
            logger.debug("Y 池入列失败，转直释", exc_info=True)
        if not pooled:
            try:
                from cuda.bindings import runtime as cudart
                cudart.cudaFree(self.ptr)
            except Exception:
                logger.debug("Y 池放弃路径 cudaFree 失败（进程退出兜底）",
                             exc_info=True)


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
        self._lk = threading.Lock()   # acquire/_recycle/release_all 互斥
                                      # （2026-09-19 审查轮：此前无锁，
                                      #  check-then-act 可 pop 空列表）
        self._released = False

    def acquire(self) -> _YFrame:
        with self._lk:
            if self._free:
                parked = self._free.pop()
                # **复用换壳（2026-09-20 泄漏专项真根因）**：停靠对象是被
                # __del__ 复活过的（_free 列表持有引用）——CPython 实测
                # 复活对象的**二次死亡不再触发 __del__**，若直接把它复用
                # （旧行为），其最终死亡既不入列也不 cudaFree = 每块永久
                # 泄漏（实测 +2 MiB/轮）。换壳：新对象继承 ptr（新对象有
                # 全新的 __del__ 名额），旧对象失能（ptr 清零 + 断池引用，
                # 其 _recycled 恒 True → 死亡早退，不会误释）。
                ptr, size = parked.ptr, parked.size
                parked.ptr = 0
                parked.pool = None
                return _YFrame(self, ptr, size)
        return self._malloc_frame()

    def _malloc_frame(self) -> _YFrame:
        from cuda.bindings import runtime as cudart
        err, ptr = cudart.cudaMalloc(self._fnb)
        if int(err) != 0 or not ptr:
            raise MemoryError(
                "cudaMalloc(%d B) 失败: %s（审查轮：此前丢弃 _err，"
                "int(None) 误报 TypeError）" % (self._fnb, err))
        return _YFrame(self, int(ptr), self._fnb)

    def _recycle(self, frame: _YFrame) -> bool:
        """入列回收（__del__ 安全路径：无任何 CUDA 调用）。

        返回 True=已入列；False=列满或池已 release，调用方须自行
        cudaFree（2026-09-20 泄漏专项：旧版此处放弃=永久泄漏，见
        _YFrame.__del__ 注释）。"""
        with self._lk:
            if not self._released and len(self._free) < self._MAX:
                self._free.append(frame)
                return True
        return False

    def recycle(self, frame: "_YFrame") -> None:
        """显式归还（S4）：入列 + 置 _recycled 标志防 __del__ 双入列。

        ⚠️ **不置空 frame.pool**（2026-09-19 审查轮二修）：置空会切断
        __del__ 的 GC 兜底——大量池帧只靠 GC 回收（跨线程 payload），
        它们会永久泄漏（实测 253 次 cudaMalloc/轮）。防双入列改用
        _recycled 标志位。列满/已 release 时直释（同 __del__ 放弃路径）。"""
        if frame._recycled:
            return
        frame._recycled = True
        if not self._recycle(frame):
            try:
                from cuda.bindings import runtime as cudart
                cudart.cudaFree(frame.ptr)
            except Exception:
                logger.debug("Y 池显式归还列满 cudaFree 失败（进程退出兜底）",
                             exc_info=True)

    def release_all(self) -> None:
        """显式释放全部空闲缓冲（extract 结束调用；DESIGN-REVIEW C5：池
        原本只在超 _MAX 时 cudaFree，extract 返回后池随闭包 GC，已入池块
        永不释放 → 长进程显存单调增长）。置 _released：此后迟到帧的
        __del__ 走直释而非回收入孤儿池（2026-09-20 泄漏专项路径②）。"""
        with self._lk:
            self._released = True
        while True:
            with self._lk:
                if not self._free:
                    break
                frame = self._free.pop()
            try:
                from cuda.bindings import runtime as cudart
                cudart.cudaFree(frame.ptr)
            except Exception:
                logger.debug("cudaFree 失败，忽略（进程退出兜底）",
                             exc_info=True)
                pass  # 释放失败仅余进程退出兜底，无进一步处置手段


class _DevBatch:
    """CPU 解码批的双缓冲 owner（P1-3）：device 批缓冲（池化）+ 宿主解码数组。

    分段/OCR 消费的 device 帧指针指向本缓冲；引用归零（GC）归还池。
    复用安全性与 _YFramePool 同一契约：raw OCR（call_gpu_raw 返回前同步）
    与 sim_pair（compare_pair 同步）读完才可能归零归还。
    """

    __slots__ = ("pool", "ptr", "size", "host", "_recycled")

    def __init__(self, pool, ptr, size, host):
        self.pool = pool
        self.ptr = ptr
        self.size = size
        self.host = host
        self._recycled = False

    def __del__(self):
        # B6（S4）+ 2026-09-20 泄漏专项：同 _YFrame——优先入列，列满/池已
        # release 时直释（放弃=永久泄漏，见 _YFrame.__del__ 注释）。
        if self._recycled:
            return
        self._recycled = True
        pooled = False
        try:
            pooled = self.pool._recycle(self)
        except Exception:
            logger.debug("批池入列失败，转直释", exc_info=True)
        if not pooled:
            try:
                from cuda.bindings import runtime as cudart
                cudart.cudaFree(self.ptr)
            except Exception:
                logger.debug("批池放弃路径 cudaFree 失败（进程退出兜底）",
                             exc_info=True)


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
        self._lk = threading.Lock()   # 同 _YFramePool（审查轮加锁）
        self._released = False

    def acquire(self, host) -> _DevBatch:
        with self._lk:
            if self._free:
                parked = self._free.pop()
                # 复用换壳（同 _YFramePool.acquire：复活对象二次死亡不
                # 再触发 __del__，直接复用 = 最终死亡永久泄漏）。
                ptr, size = parked.ptr, parked.size
                parked.ptr = 0
                parked.pool = None
                return _DevBatch(self, ptr, size, host)
        from cuda.bindings import runtime as cudart
        err, ptr = cudart.cudaMalloc(self._nbytes)
        if int(err) != 0 or not ptr:
            raise MemoryError(
                "cudaMalloc(%d B) 失败: %s（审查轮：此前误报 TypeError）"
                % (self._nbytes, err))
        return _DevBatch(self, int(ptr), self._nbytes, host)

    def _recycle(self, b: _DevBatch) -> bool:
        """入列回收（__del__ 安全路径：无任何 CUDA 调用；B6/S4）。

        返回 True=已入列；False=列满/已 release，调用方须自行 cudaFree
        （2026-09-20 泄漏专项，同 _YFramePool._recycle）。"""
        b.host = None
        with self._lk:
            if not self._released and len(self._free) < self._MAX:
                self._free.append(b)
                return True
        return False

    def recycle(self, b: "_DevBatch") -> None:
        """显式归还（S4）：入列 + 标志位防双入列（同 _YFramePool.recycle
        的二修：不置空 pool，保住 __del__ 的 GC 兜底）。列满/已 release
        时直释（同 __del__ 放弃路径）。"""
        if b._recycled:
            return
        b._recycled = True
        if not self._recycle(b):
            try:
                from cuda.bindings import runtime as cudart
                cudart.cudaFree(b.ptr)
            except Exception:
                logger.debug("批池显式归还列满 cudaFree 失败（进程退出兜底）",
                             exc_info=True)

    def release_all(self) -> None:
        """显式释放全部空闲缓冲（同 _YFramePool.release_all，C5；
        置 _released 同理）。"""
        with self._lk:
            self._released = True
        while True:
            with self._lk:
                if not self._free:
                    break
                b = self._free.pop()
            try:
                from cuda.bindings import runtime as cudart
                cudart.cudaFree(b.ptr)
            except Exception:
                logger.debug("cudaFree 失败，忽略（进程退出兜底）",
                             exc_info=True)
                pass  # 释放失败仅余进程退出兜底，无进一步处置手段
