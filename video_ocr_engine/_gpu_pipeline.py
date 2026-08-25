"""GPU 全驻留零拷贝管线（_GpuPipelineMixin）：NVDEC 默认主路径。

从 extractor.py 拆出：_gpu_pipeline_enabled / _run_pipelined_gpu。GPU 预处理/
归约/帧分析内核位于 video_ocr_engine._gpu_kernels（ocr_trt re-export）。
FieldExtractor 组合本 mixin 获得这两个方法。
"""
import os as _os
import time

import numpy as np

import engine_config as config
from video_utils import nvdec_available
from ._helpers import (_ndarray_device_ptr, _otsu_from_hist,
                       _decode_progress_pct, _otsu_median_threshold,
                       _read_fps_from_vr)


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


class _GpuPipelineMixin:
    # ═══════════════ GPU 全驻留管线（NVDEC） ═══════════════

    def _gpu_pipeline_enabled(self) -> bool:
        """GPU 全驻留管线：NVDEC 场景的默认主路径（内部恒为单通道灰度）。

        默认启用条件（全部满足）：
        - decode_backend ∈ {auto, nvdec} 且 NVDEC 实际可用
        - merge_similar 的分离模式不是 contrast（GPU 路径支持 raw/binary）
        - force_aspect == 0（GPU raw 直通不支持强制宽高比）
        代表帧格式：rep_crop_format="gray" 或 "yuv" 均可 —— yuv 由
        luma_nv12 kernel 在 GPU 提取 Y 平面，OCR 走宿主预处理（代表帧
        D2H 含 UV）；gray 由 raw 直通（单 TRT 引擎时）。OCR 后端任意：
        TRT 时 raw 直通，ONNX/无 TRT 时 D2H 代表帧走宿主 OCR。

        env GPU_PIPELINE：'0' 显式关闭；'1' 强制尝试（条件不满足时
        内部自动回退宿主管线）。不设置 = 按上述默认规则。
        """
        if not config.env_bool(config.GPU_PIPELINE_ENV, default=True):
            return False
        if (self._decode_backend or 'auto').lower() not in ('auto', 'nvdec'):
            return False
        if self._merge_similar and self._merge_effective_mode() == 'contrast':
            return False
        if self._force_aspect > 0:
            # GPU raw 直通（process_gray_raw）按自然宽高比缩放，不支持强制
            # 宽高比；宿主路径支持 → 有 force_aspect 时走宿主，避免与宿主
            # 路径的 OCR 输入不一致（文本结果漂移）。
            return False
        return nvdec_available(str(self._video_path))

    def _run_pipelined_gpu(self):
        """GPU 全驻留零拷贝路径：灰度/sharp/聚类/合并判定/OCR 全在 GPU。

        过 RAM 的只有：每帧两个标量（sharp/cluster）、校准直方图表、
        merge_similar 两标量、keep_crops 输出（每段一张 D2H，结果必须
        给外部）与 OCR 回退路径（ONNX/无 TRT/引擎未就绪时代表帧 D2H）。
        两种代表帧格式：
        - gray：代表帧即 decord gray NDArray（owner 保活），raw OCR 直通；
        - yuv：代表帧为 packed NV12（owner 保活），Y 平面按需提取到
          _YFramePool 池帧供 raw OCR / GPU 合并判定（~10KB D2D/次）；
          完整 NV12 仅 keep_crops 时 D2H。
        合并判定（sim_pair kernel）与宿主 _segments_similar 语义对应
        （整数精确；除对比阈值处的 float32 末位舍入外逐位一致）。
        contrast 模式已被门控排除（走宿主）。
        返回格式与 _run_pipelined 相同。
        """
        from queue import Queue
        import threading
        from ocr_trt import GpuFrameAnalyzer
        _t_open = time.perf_counter()
        vr = self._open_vr()
        if not self._backend.startswith('decord/GPU'):
            # NVDEC 打开失败（nvdec_available 探测仍可能为真）：走宿主流水线。
            # 必须直接调 _run_pipelined_host —— 经 _run_pipelined 会重新判定
            # _gpu_pipeline_enabled() 并再次进入本方法（原实现无限递归）。
            return self._run_pipelined_host(None)
        if self._fps is None:
            _fps = _read_fps_from_vr(vr)
            self._fps = _fps if _fps else config.DEFAULT_FPS_FALLBACK
        x1, y1, x2, y2 = self._roi
        total = len(vr)
        end = min(self._frame_end or total, total)
        if self._frame_start > 0:
            vr.seek_accurate(self._frame_start)
        frames = list(range(self._frame_start, end, self._sample_stride))
        if not frames:
            raise ValueError(
                f"帧区间为空: frame_start={self._frame_start}, "
                f"frame_end={end}, total={total}")
        self._prof_end('producer', 'open_and_fps', _t_open)
        calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
        calib_nds = vr.get_batch(
            frames[:calib_n], roi=(x1, y1, x2 + 1, y2 + 1))
        calib_base, calib_shape = _ndarray_device_ptr(calib_nds)
        analyzer = GpuFrameAnalyzer()
        yuv = self._yuv_output
        if yuv:
            # yuv420（packed NV12）：先 D2D 提取 Y 平面（luma_nv12 与宿主
            # _nv12_luma_full 逐位一致），histogram/analyze 都只消费灰度 Y。
            if len(calib_shape) != 3:
                return self._run_pipelined_host(None)
            src_h = calib_shape[1] * 2 // 3
            src_w = calib_shape[2]
            calib_gray = analyzer.extract_luma(
                calib_base, calib_n, src_h, src_w,
                limited=self._color_range != 1)
        else:
            if len(calib_shape) != 4 or calib_shape[-1] != 1:
                # 灰度帧非 4D 单通道（部分 decord fork 输出 (B,H,W)）：GPU
                # 分段不支持，回退宿主。直接 _run_pipelined_host（防递归）。
                return self._run_pipelined_host(None)
            src_h, src_w = calib_shape[1], calib_shape[2]
            calib_gray = calib_base
        # 逐帧直方图校准：与单流水线"前 50 帧 Otsu 取中位数"语义逐位一致
        # （含退化双值帧的阈值行为），D2H 仅 B×1KB 标量表，校准帧不落 RAM。
        # 注意必须用 _otsu_from_hist（输入是直方图行）；_otsu 接收的是
        # 灰度图像并在内部做直方图——传错曾产生"直方图的直方图"垃圾阈值。
        _hist_mat = analyzer.histograms_perframe(calib_gray, calib_n,
                                                 src_h, src_w)
        ths = [_otsu_from_hist(_hist_mat[k]) for k in range(calib_n)]
        th = _otsu_median_threshold(ths)
        self._bin_thresh = th

        self._gpu_pipeline_mode = True
        ocr_session = self._start_ocr_session(None)
        q = ocr_session["q"]
        results = ocr_session["results"]
        ocr_err = ocr_session["err"]
        ocr_wall = ocr_session["wall"]
        _put_ocr = ocr_session["put"]

        prev_holder = calib_nds      # 保住前一 decord NDArray（防解码池复用）
        prev_ptr = calib_base        # 灰色模式：上一批/校准末帧 device 指针

        def frame_stream():
            nonlocal prev_holder, prev_ptr
            from cuda.bindings import runtime as cudart
            DECODE_BATCH = config.GPU_PIPELINE_DECODE_BATCH
            _d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            limited = self._color_range != 1

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
            B = calib_n
            sums, fnb = _analyze_batch(calib_gray, calib_gray, B,
                                       src_h, src_w)
            rows = src_h + (src_h + 1) // 2
            for k in range(B):
                cur = calib_base + k * (rows * src_w if yuv else fnb)
                yield (frames[k], (calib_nds, cur,
                                   (rows if yuv else src_h), src_w),
                       float(sums[k, 0]), float(sums[k, 1]))
                prev_holder = calib_nds
                prev_ptr = cur
            # yuv：末帧 Y 存入单帧缓冲（extract_luma 复用主缓冲会覆盖，
            # 下一批开始前必须已有独立副本）。注：_ensure_prev 首次按
            # 64*fnb 分配且批尺寸恒定 → prev_front 指针后续稳定不重分配。
            prev_front = None
            have_prev_front = False
            if yuv:
                prev_front = analyzer._ensure_prev(fnb)
                cudart.cudaMemcpyAsync(
                    prev_front, calib_gray + (calib_n - 1) * fnb, fnb,
                    _d2d, analyzer._stream)
                have_prev_front = True

            for bstart in range(calib_n, len(frames), DECODE_BATCH):
                bend = min(bstart + DECODE_BATCH, len(frames))
                nds = vr.get_batch(
                    frames[bstart:bend], roi=(x1, y1, x2 + 1, y2 + 1))
                base, shape = _ndarray_device_ptr(nds)
                B = bend - bstart
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
                    prev_s = prev_ptr
                sums, fnb = _analyze_batch(gray_base, prev_s, B, H, W)
                if yuv and have_prev_front:
                    cudart.cudaMemcpyAsync(
                        prev_front, gray_base + (B - 1) * fnb, fnb,
                        _d2d, analyzer._stream)
                    have_prev_front = True
                for k in range(B):
                    cur = base + k * (rows * W if yuv else fnb)
                    yield (frames[bstart + k], (nds, cur, rows, W),
                           float(sums[k, 0]), float(sums[k, 1]))
                    prev_holder = nds
                    prev_ptr = cur
                if yuv:
                    have_prev_front = True

        # 生产者线程：解码 + GPU analyze 与主线程分段/OCR 重叠
        producer_q: Queue = Queue(maxsize=max(8, self._buffer_size))
        producer_err: list = []

        def _producer() -> None:
            try:
                for item in frame_stream():
                    producer_q.put(item)
            except Exception as e:  # noqa: BLE001
                producer_err.append(e)
            finally:
                producer_q.put(None)

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        segs: list = []
        rep_crops: dict = {}
        seg_idx = 0
        s = 0
        rep_frame = frames[0]
        rep_dev = None           # 代表帧设备元组：gray=(nds,yptr,H,W)；
                                 # yuv=(nds,nv12ptr,rows,W)。owner 保活，
                                 # 全程留显存，不做持续 D2H。
        last_rep_dev = None      # 上一"已发出"段的代表帧设备元组
        rep_sharp = -1.0
        prev_seen = False
        k = 0
        t0 = time.perf_counter()
        # 零拷贝管线：代表帧只在两处过 RAM —— keep_crops 输出（每段一张
        # D2H）与 OCR 回退路径（ONNX/无 TRT/引擎未就绪）。merge_similar
        # 判定在 GPU（sim_pair 整数精确）；yuv 的 Y 平面按需从保留的 NV12
        # 提取（luma_into → 池帧，~10KB D2D/次）。
        raw_ready_ref = ocr_session["raw_ready"]
        _y_pool = _YFramePool(src_h * src_w) if yuv else None
        _limited = self._color_range != 1

        def _similar_device(a_dev, b_dev) -> bool:
            """merge_similar 的 GPU 判定，语义与宿主 _segments_similar 对应：
            平均绝对差（整数精确，与宿主 float32 均值仅差末位舍入）≤ 阈值
            且显著变化像素数 ≤ max_changed。contrast 模式已被门控排除。"""
            if not (self._merge_similar and a_dev is not None
                    and b_dev is not None):
                return False
            use_bin = 1 if self._merge_effective_mode() == 'binary' else 0
            ya = yb = None
            if yuv:
                ya = _y_pool.acquire()
                yb = _y_pool.acquire()
                analyzer.luma_into(int(a_dev[1]), int(ya.ptr), src_h,
                                   src_w, _limited)
                analyzer.luma_into(int(b_dev[1]), int(yb.ptr), src_h,
                                   src_w, _limited)
                ap, bp = ya.ptr, yb.ptr
            else:
                ap, bp = int(a_dev[1]), int(b_dev[1])
            try:
                mad, chg = analyzer.compare_pair(ap, bp, src_h, src_w,
                                                 self._bin_thresh, use_bin)
            finally:
                # 池帧引用释放（GC 归还）
                ya = yb = None
            n = src_h * src_w
            mean = 255.0 * mad / n if use_bin else mad / n
            if mean > self._merge_similar_threshold:
                return False
            return chg <= self._merge_max_changed_pixels

        def _emit_ocr(idx, r_frame, r_dev, frac) -> None:
            _t_push = time.perf_counter()
            _raw = raw_ready_ref[0] and r_dev is not None
            crop_h = None
            dev_ocr = None
            if _raw:
                # 零拷贝：gray 直接 decord NDArray 指针；yuv 提取 Y 到池帧
                #（owner=池帧，OCR worker 用毕 GC 归还）。
                if yuv:
                    yf = _y_pool.acquire()
                    analyzer.luma_into(int(r_dev[1]), int(yf.ptr), src_h,
                                       src_w, _limited)
                    dev_ocr = (yf, yf.ptr, src_h, src_w)
                else:
                    dev_ocr = r_dev
            else:
                # 回退（ONNX/无 TRT/引擎未就绪）：代表帧 D2H → 宿主预处理，
                # crop 与 keep_crops 共用同一副本。
                crop_h = _d2h_rep(r_dev)
            _put_ocr((idx, r_frame, crop_h, dev_ocr, frac))
            if self._keep_crops:
                # keep_crops 是结果输出（给外部转 RGB），不可避免的 D2H
                rep_crops[r_frame] = (crop_h if crop_h is not None
                                      else _d2h_rep(r_dev))
            self._prof_end('producer', 'q_put_block', _t_push)

        def _d2h_rep(dev):
            """代表帧 D2H：gray = (H,W)；yuv = packed NV12 (rows,W) 原样保留。"""
            from cuda.bindings import runtime as cudart
            arr = np.empty((dev[2], dev[3]), dtype=np.uint8)
            cudart.cudaMemcpy(arr.ctypes.data, int(dev[1]), dev[2] * dev[3],
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            return arr

        try:
            while True:
                item = producer_q.get()
                if item is None:
                    break
                if producer_err:
                    raise producer_err[0]
                fi, dev, sharp, cluster = item
                if prev_seen:
                    changed = float(cluster) >= self._C
                    if changed:
                        seg = frames[s:k]
                        if config.env_bool(config.DEBUG_BOUNDS_ENV):
                            print(f'[GB]{fi}:{float(cluster):.0f}',
                                  flush=True)
                        similar = _similar_device(last_rep_dev, rep_dev)
                        if similar:
                            segs[-1].extend(seg)
                        else:
                            segs.append(seg)
                            _emit_ocr(seg_idx, rep_frame, rep_dev,
                                      k / max(len(frames), 1))
                            seg_idx += 1
                            last_rep_dev = rep_dev
                        s = k
                        rep_frame = fi
                        rep_dev = dev
                        rep_sharp = sharp
                    elif sharp > rep_sharp:
                        rep_sharp = sharp
                        rep_frame = fi
                        rep_dev = dev
                else:
                    rep_frame = fi
                    rep_dev = dev
                    rep_sharp = sharp
                    prev_seen = True
                if k % 100 == 0:
                    self._cancel()
                if k % 500 == 0:
                    self._progress(f'[{self._backend}] GPU分段: {k}/{len(frames)}',
                                   _decode_progress_pct(k / max(len(frames), 1)))
                k += 1
            producer.join()
            if producer_err:
                raise producer_err[0]
            seg = frames[s:]
            similar = _similar_device(last_rep_dev, rep_dev)
            if similar:
                segs[-1].extend(seg)
            else:
                segs.append(seg)
                _emit_ocr(seg_idx, rep_frame, rep_dev, 1.0)
                seg_idx += 1
        finally:
            _t_consume_end = time.perf_counter()
            self.timing['decode'] = _t_consume_end - t0
            ocr_session["finish"]()
            self.timing['ocr_tail'] = time.perf_counter() - _t_consume_end
        if ocr_err:
            raise ocr_err[0]
        self.timing['ocr'] = ocr_wall[0]
        self._n_segments = len(segs)
        self.crops = rep_crops
        del vr
        self._ocr_texts = [results[i][0] for i in range(seg_idx)]
        self._ocr_confs = [results[i][1] for i in range(seg_idx)]
        return (frames, segs, self._ocr_texts, self._ocr_confs,
                [results[i][2] for i in range(seg_idx)])

