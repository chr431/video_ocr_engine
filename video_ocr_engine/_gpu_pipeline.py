"""GPU 全驻留管线（_GpuPipelineMixin）：NVDEC+TRT 下的 host 最小化路径。

从 extractor.py 拆出：_gpu_pipeline_enabled / _run_pipelined_gpu。GPU 预处理/
归约/帧分析内核位于 ocr_trt（GpuPreprocessor/GpuOutputReducer/GpuFrameAnalyzer）。
FieldExtractor 组合本 mixin 获得这两个方法。
"""
import os as _os
import time

import numpy as np

import engine_config as config
from video_utils import nvdec_available, tensorrt_available
from ._helpers import (_ndarray_device_ptr, _otsu_from_hist,
                       _decode_progress_pct, _otsu_median_threshold,
                       _read_fps_from_vr)


class _GpuPipelineMixin:
    # ═══════════════ GPU 全驻留管线（NVDEC+TRT） ═══════════════

    def _gpu_pipeline_enabled(self) -> bool:
        """GPU 全驻留管线：NVDEC+TRT 场景的默认主路径（gray 输出）。

        默认启用条件（全部满足）：
        - decode_backend ∈ {auto, nvdec} 且 NVDEC 实际可用
        - ocr_backend ≠ cpu 且 TensorRT 可用
        - gray_output=True 且非 yuv_output（YUV 场景暂走宿主管线）
        - merge_similar 的分离模式不是 contrast（GPU 路径支持 raw/binary）
        - 未开启 dual_pipeline（双流水线优先级更高，保持现状）

        env GPU_PIPELINE：'0' 显式关闭；'1' 强制尝试（条件不满足时
        内部自动回退宿主管线）。不设置 = 按上述默认规则。
        """
        if not config.env_bool(config.GPU_PIPELINE_ENV, default=True):
            return False
        if self._dual_pipeline:
            return False
        if not self._gray_output or self._yuv_output:
            return False
        if (self._decode_backend or 'auto').lower() not in ('auto', 'nvdec'):
            return False
        if (self._ocr_backend or 'auto').lower() == 'cpu':
            return False
        if self._merge_similar and self._merge_effective_mode() == 'contrast':
            return False
        return nvdec_available(str(self._video_path)) and tensorrt_available()

    def _run_pipelined_gpu(self):
        """实验：灰度/sharp/聚类变化分都在 GPU 计算，host 只收标量。

        代表帧保留 GPU device pointer，OCR 走 call_gpu_raw 路径。
        校准阈值仍取前 50 帧 D2H（量小，可接受）。
        返回格式与 _run_pipelined 相同。
        """
        from queue import Queue
        import threading
        from ocr_trt import GpuFrameAnalyzer
        _t_open = time.perf_counter()
        vr = self._open_vr()
        if not self._backend.startswith('decord/GPU'):
            return self._run_pipelined(_force_single=True)
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
        calib_c = calib_shape[-1] if len(calib_shape) == 4 else 0
        if calib_c != 1:
            return self._run_pipelined(_force_single=True)
        src_h, src_w = calib_shape[1], calib_shape[2]
        analyzer = GpuFrameAnalyzer()
        # 逐帧直方图校准：与单流水线"前 50 帧 Otsu 取中位数"语义逐位一致
        # （含退化双值帧的阈值行为），D2H 仅 B×1KB 标量表，校准帧不落 RAM。
        # 注意必须用 _otsu_from_hist（输入是直方图行）；_otsu 接收的是
        # 灰度图像并在内部做直方图——传错曾产生"直方图的直方图"垃圾阈值。
        _hist_mat = analyzer.histograms_perframe(calib_base, calib_n,
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

        prev_holder = calib_nds
        prev_ptr = calib_base

        def frame_stream():
            nonlocal prev_holder, prev_ptr
            from cuda.bindings import runtime as cudart
            DECODE_BATCH = config.GPU_PIPELINE_DECODE_BATCH
            _d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

            def _fill_prev(prev_buf, base, B, frame_nbytes, prev_single):
                for k in range(B):
                    src = (prev_single if k == 0
                           else base + (k - 1) * frame_nbytes)
                    cudart.cudaMemcpyAsync(
                        prev_buf + k * frame_nbytes, src, frame_nbytes,
                        _d2d, analyzer._stream)

            # 校准帧整批分析
            B = calib_n
            frame_nbytes = src_h * src_w
            prev_buf = analyzer._ensure_prev(max(B, DECODE_BATCH) * frame_nbytes)
            _fill_prev(prev_buf, calib_base, B, frame_nbytes, calib_base)
            sums = analyzer.analyze_batch(
                calib_base, prev_buf, B, src_h, src_w, th)
            for k in range(B):
                cur = calib_base + k * frame_nbytes
                yield (frames[k], (calib_nds, cur, src_h, src_w),
                       float(sums[k, 0]), float(sums[k, 1]))
                prev_holder = calib_nds
                prev_ptr = cur

            for bstart in range(calib_n, len(frames), DECODE_BATCH):
                bend = min(bstart + DECODE_BATCH, len(frames))
                nds = vr.get_batch(
                    frames[bstart:bend], roi=(x1, y1, x2 + 1, y2 + 1))
                base, shape = _ndarray_device_ptr(nds)
                if len(shape) != 4 or shape[-1] != 1:
                    raise RuntimeError("GPU 分段仅支持 decord gray 输出")
                H, W = shape[1], shape[2]
                B = bend - bstart
                fnb = H * W
                prev_buf = analyzer._ensure_prev(max(B, DECODE_BATCH) * fnb)
                _fill_prev(prev_buf, base, B, fnb, prev_ptr)
                sums = analyzer.analyze_batch(
                    base, prev_buf, B, H, W, th)
                for k in range(B):
                    cur = base + k * fnb
                    yield (frames[bstart + k], (nds, cur, H, W),
                           float(sums[k, 0]), float(sums[k, 1]))
                    prev_holder = nds
                    prev_ptr = cur

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
        rep_dev = None
        rep_sharp = -1.0
        rep_gray_h = None     # 当前代表帧的宿主副本（D2H，每段一张小 ROI）
        last_rep_gray_h = None  # 上一"已发出"段的代表帧宿主副本
        prev_seen = False
        k = 0
        t0 = time.perf_counter()
        # 代表帧宿主副本：merge_similar 判定直接复用宿主 _segments_similar
        # （逐位一致），且避免每个段边界一次内核启动+同步的开销。每段仅
        # 一张 ROI 灰度（~10KB）过 RAM，整片流量可忽略。

        def _d2h_rep(dev):
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
                        similar = (
                            self._merge_similar and segs
                            and self._segments_similar(last_rep_gray_h,
                                                       rep_gray_h))
                        if similar:
                            segs[-1].extend(seg)
                        else:
                            segs.append(seg)
                            _put_ocr((seg_idx, rep_frame, None, rep_dev,
                                      k / max(len(frames), 1)))
                            if self._keep_crops:
                                rep_crops[rep_frame] = rep_gray_h
                            seg_idx += 1
                            last_rep_gray_h = rep_gray_h
                        s = k
                        rep_frame = fi
                        rep_dev = dev
                        rep_sharp = sharp
                        rep_gray_h = None
                    elif sharp > rep_sharp:
                        rep_sharp = sharp
                        rep_frame = fi
                        rep_dev = dev
                        rep_gray_h = None
                else:
                    rep_frame = fi
                    rep_dev = dev
                    rep_sharp = sharp
                    rep_gray_h = None
                    prev_seen = True
                if rep_gray_h is None and rep_dev is not None:
                    rep_gray_h = _d2h_rep(rep_dev)
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
            similar = (
                self._merge_similar and segs
                and self._segments_similar(last_rep_gray_h, rep_gray_h))
            if similar:
                segs[-1].extend(seg)
            else:
                segs.append(seg)
                _put_ocr((seg_idx, rep_frame, None, rep_dev, 1.0))
                if self._keep_crops:
                    rep_crops[rep_frame] = rep_gray_h
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

