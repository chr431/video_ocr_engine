"""宿主流水线（_HostPipelineMixin）：解码校准 / 帧流 / 分段状态机 / OCR 会话。

从 extractor.py 拆出（2026-08）：宿主路径的模块级纯函数（_host_calibrate /
_host_frame_stream / _host_segment_frames）与 FieldExtractor 的 OCR 会话
（_start_ocr_session）集中于此，extractor.py 仅保留引擎骨架与流水线分发。
模块级函数签名不变，旧导入路径 `from video_ocr_engine.extractor import
_host_calibrate` 等仍可用（extractor 顶部 re-export）。
"""
from __future__ import annotations

import logging
import os as _os
import threading
import time

import numpy as np

import engine_config as config
from segmentation import (
    _cluster_win3, _gray_seg, _gray_seg_batch,
    _gray_seg_yuv, _gray_seg_yuv_batch, _otsu,
)
from video_utils import _nv12_luma_full, _text_sep_gray
from ._helpers import (
    _ocr_batch_size, _ndarray_device_ptr, _otsu_from_hist,
    _decode_progress_pct, _ocr_progress_pct, _otsu_median_threshold,
    _read_fps_from_vr,
)

logger = logging.getLogger(__name__)


def _host_calibrate(ex, vr, frames, *, with_dev=False):
    """宿主路径 Otsu 校准（单流水线统一入口）。

    ex: FieldExtractor——只调用 _crop_is_expected / _crop_luma /
        _prof_end。
    with_dev: True 时保留 decord GPU 单通道帧的 DLPack 指针（GPU raw OCR
        直通用）；stride==1 时同时捕获 next_roi 的（shape 3D）帧指针。
    stride>1 走 get_batch 等差步长快速路径（校准帧号与后续帧流一致），
    stride==1 走 next_roi 顺序流。
    返回 (calib, th)。calib 元素统一 (fi, crop, gray, sharp, dev_info)，
    dev_info 仅在 with_dev 且帧为 GPU 单通道时非 None。
    """
    x1, y1, x2, y2 = ex._roi
    calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
    calib: list = []
    if ex._sample_stride > 1:
        nds = vr.get_batch(frames[:calib_n], roi=(x1, y1, x2 + 1, y2 + 1))
        crops = nds.asnumpy()
        base, shape = (0, ())
        dev_c = 0
        if with_dev:
            # 与旧单流水线一致：只要请求设备指针就捕获（不先看 shape）——
            # channel 判定由捕获后的 shape 完成（非 GPU 单通道自然 dev_c=0）。
            base, shape = _ndarray_device_ptr(nds)
            dev_c = shape[-1] if len(shape) == 4 else 0
        for k in range(calib_n):
            c = crops[k]
            if not ex._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                c = c[y1:y2 + 1, x1:x2 + 1]
            g = ex._crop_luma(c)
            dev_info = None
            if dev_c == 1 and len(shape) == 4:
                src_h, src_w = shape[1], shape[2]
                dev_info = (nds, base + k * src_h * src_w, src_h, src_w)
            calib.append((frames[k], c, g, float(g.std()), dev_info))
    else:
        for k in range(calib_n):
            nd = vr.next_roi(x1, y1, x2 + 1, y2 + 1)
            c = nd.asnumpy()
            if not ex._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                c = c[y1:y2 + 1, x1:x2 + 1]
            g = ex._crop_luma(c)
            dev_info = None
            if with_dev and len(nd.shape) == 3 and nd.shape[-1] == 1:
                base, shape = _ndarray_device_ptr(nd)
                dev_info = (nd, base, shape[0], shape[1])
            calib.append((frames[k], c, g, float(g.std()), dev_info))
    ths = [_otsu(g) for _fi, _c, g, _s, _dev in calib]
    return calib, _otsu_median_threshold(ths)


def _host_frame_stream(ex, frames, vr, calib, th, *, with_dev=False):
    """宿主帧流：先产出校准帧，再批量流式解码剩余帧。

    ex: FieldExtractor——只调用 _batch_luma/_prof_end。
    calib 元素统一 (fi, crop, gray, sharp, dev_info)（可为空列表）。
    with_dev=True 时随帧产出 decord GPU NDArray 设备信息 (owner, ptr, h, w)
    供 GPU raw OCR 直通（仅 gray 单通道输出路径有效）。
    yield (frame_idx, crop, gray, sharp, bin, dev_info)。
    """
    DECODE_BATCH = config.DECODE_BATCH_SIZE
    x1, y1, x2, y2 = ex._roi
    for fi, c, g, s, *dev_rest in calib:
        yield (fi, c, g, s, g > th, dev_rest[0] if dev_rest else None)
    g_buf = None   # 复用批量灰度缓冲（每批形状恒定：B×H×W）
    for bstart in range(len(calib), len(frames), DECODE_BATCH):
        bend = min(bstart + DECODE_BATCH, len(frames))
        _t_d = time.perf_counter()
        nds = vr.get_batch(frames[bstart:bend], roi=(x1, y1, x2 + 1, y2 + 1))
        crops = nds.asnumpy()
        ex._prof_end('producer', 'decode_batch', _t_d)
        _t_g = time.perf_counter()
        if g_buf is None:
            g_buf = np.empty(crops.shape[:2] + (crops.shape[2] * 2 // 3,)
                             if ex._yuv_output else crops.shape[:2],
                             dtype=np.uint8)
        if (g_buf.shape[1:] == (crops.shape[1] * 2 // 3 if ex._yuv_output
                                else crops.shape[1:])
                and len(crops) <= g_buf.shape[0]):
            # 该批可能不是满批（末批 B 更小）：只复用前 B 行，避免形状不匹配
            g = ex._batch_luma_out(crops, g_buf[:len(crops)])
        else:
            g = ex._batch_luma(crops)
        ex._prof_end('producer', 'gray_batch', _t_g)
        g = np.ascontiguousarray(g)
        _t_s = time.perf_counter()
        sharp = g.std(axis=(1, 2))
        ex._prof_end('producer', 'sharp_batch', _t_s)
        _t_b = time.perf_counter()
        bs = g > th
        ex._prof_end('producer', 'bin_batch', _t_b)
        dev_base = 0
        src_h = src_w = 0
        if with_dev and len(nds.shape) == 4 and nds.shape[-1] == 1:
            dev_base, shape = _ndarray_device_ptr(nds)
            src_h, src_w = shape[1], shape[2]
        for k, gi in enumerate(range(bstart, bend)):
            d = None
            if dev_base:
                d = (nds, dev_base + k * src_h * src_w, src_h, src_w)
            yield (frames[gi], crops[k], g[k], float(sharp[k]), bs[k], d)


def _host_segment_frames(ex, frames, stream, *, debug_tag, progress_prefix,
                         emit, segs):
    """宿主分段状态机（单流水线统一入口）。

    ex: FieldExtractor。
    stream: (fi, crop, gray, sharp, bin, dev_info) 迭代器（_host_frame_stream）。
    emit(seg, rep_frame, rep_crop, rep_dev, rep_gray, frac)：段闭合时投递
        OCR，由调用方闭包实现（入队/全局段号/keys/reps/rep_crops 收敛在
        闭包里；调用方在 emit 前已把 seg 追加进 segs）。
    segs 由调用方传入（emit 闭包直写 rep_crops）。
    debug_tag 非 None 且 DEBUG_BOUNDS 开启时打印边界（[HB]=单流水线，与
    GPU 路径 [GB] 对齐）。
    返回 (first_rep_gray, last_rep_gray)：首发射段代表灰度与末发射段代表灰度。
    """
    s = 0
    rep_frame = frames[0]
    rep_crop = None
    rep_dev = None
    rep_sharp = -1.0
    rep_gray = None
    last_rep_gray = None
    first_rep_gray = None
    prev_b = None
    for k, (fi, c, g, sharp, b, dev_info) in enumerate(stream):
        if prev_b is not None:
            d = prev_b != b
            _t_seg = time.perf_counter()
            c3 = _cluster_win3(d)
            changed = c3 >= ex._C
            ex._prof_end('producer', 'segmentation', _t_seg)
            if changed:
                seg = frames[s:k]
                if (debug_tag is not None
                        and config.env_bool(config.DEBUG_BOUNDS_ENV)):
                    print(f'[{debug_tag}]{fi}:{c3:.0f}',
                          flush=True)
                similar = (
                    ex._merge_similar and segs
                    and ex._segments_similar(last_rep_gray, rep_gray))
                if similar:
                    # 同一视觉内容被噪声切成多段：并入前一段，不产生新的
                    # OCR 任务，保留前一段代表帧/文本。
                    segs[-1].extend(seg)
                else:
                    segs.append(seg)
                    emit(seg, rep_frame, rep_crop, rep_dev, rep_gray,
                         k / max(len(frames), 1))
                    if first_rep_gray is None:
                        first_rep_gray = rep_gray
                    last_rep_gray = rep_gray
                s = k
                rep_frame = fi
                rep_crop = c
                rep_dev = dev_info
                rep_sharp = sharp
                rep_gray = g
            elif sharp > rep_sharp:
                rep_sharp = sharp
                rep_frame = fi
                rep_crop = c
                rep_dev = dev_info
                rep_gray = g
        else:
            rep_frame = fi
            rep_crop = c
            rep_dev = dev_info
            rep_sharp = sharp
            rep_gray = g
        prev_b = b
        if k % 100 == 0:
            ex._cancel()
        if k % 500 == 0:
            ex._progress(f'{progress_prefix}: {k}/{len(frames)}',
                         _decode_progress_pct(k / max(len(frames), 1)))
    seg = frames[s:]
    similar = (
        ex._merge_similar and segs
        and ex._segments_similar(last_rep_gray, rep_gray))
    if similar:
        segs[-1].extend(seg)
    else:
        segs.append(seg)
        emit(seg, rep_frame, rep_crop, rep_dev, rep_gray, 1.0)
        if first_rep_gray is None:
            first_rep_gray = rep_gray
    return first_rep_gray, last_rep_gray


class _HostPipelineMixin:
    """宿主流水线 mixin：FieldExtractor 组合本类获得 OCR 会话与宿主管线。"""

    # ═══════════════ OCR 输入宽度自适应裁切 ═══════════════

    def _crop_to_content(self, crop):
        """按二值图的"有墨迹列范围"裁掉两侧空白（宽 ROI 字幕省 OCR 计算）。

        判据与分段完全一致（`g > self._bin_thresh`，墨迹为亮），
        每列墨迹数 ≥ 2 才算有效列（抗孤立噪点）。

        不裁的三类情况（无收益或有风险，一律原样返回）：
          · 关闭 / `force_aspect > 0`（宽度被强制，裁切只改缩放不省宽）
          · 动态范围过小（std < 3，纯黑/纯白帧，Otsu 阈值无意义）
          · 内容已占满 ROI（cols 覆盖全宽）
        余量 `OCR_ROI_AUTOCROP_MARGIN`（占 ROI 宽 %，默认 10）——**不能省**：
        裁太紧会改变 CTC 序列长度，实测会插入多余空格（一致率 98% → 100%）。

        ⚠️ **不要再补"裁后宽度会被 pad 回 OCR_PAD_WIDTH_MIN 就跳过"的守卫。**
        曾有一版守卫，前提是"省不到算力就别冒准确率风险"；用真值复核证明
        前提错了 —— 裁切让输入更贴近模型训练分布（文字填满图像），
        **即使省不到算力也能提准确率**：

        | 视频 | 不裁 | 有守卫（窄 ROI 被全跳过 = 不裁） | 无守卫（裁切生效） |
        |---|---:|---:|---:|
        | test5（h264 7223帧） | 97.951% | 97.951% | **98.768%（+0.82pp）** |
        | test6（AV1 23441帧） | 98.187% | 98.187% | **99.125%（+0.94pp）** |

        逐帧看：文本变化 1.2%，其中**由错变对 69 帧、由对变错 10 帧**
        （`8日→88`、`日1→81` 是纠错；`51→S1`、`115→11S` 是新增错字）。
        墙钟代价 +2.1% / -0.3%（≈噪声）。**净赚约 0.9pp，守卫必须去掉。**
        """
        if not self._ocr_autocrop or getattr(self, '_force_aspect', 0):
            return crop
        import numpy as _np
        g = crop[..., 0] if crop.ndim == 3 else crop
        w = int(g.shape[1])
        if w <= 8 or float(g.std()) < 3.0:
            return crop
        cols = _np.nonzero((g > self._bin_thresh).sum(axis=0) >= 2)[0]
        if len(cols) == 0:
            return crop
        m = max(1, int(round(w * self._ocr_autocrop_margin_pct / 100.0)))
        lo = max(0, int(cols[0]) - m)
        hi = min(w, int(cols[-1]) + 1 + m)
        if lo == 0 and hi == w:
            return crop
        return crop[:, lo:hi]

    def _start_ocr_session(self, _ocr_engines: list | None = None) -> dict:
        """启动一个可跨多个切片持续复用的 OCR 会话。

        返回 dict：q（段任务队列）、results（全局段索引 → text/conf/rep）、
        err、wall、put（投递段任务）、finish（哨兵并 join OCR worker）。
        引擎统一使用单一 OCR 会话；跨切片持续复用，避免每片重建
        OCR worker / infer 线程造成屏障。
        """
        from queue import Full, Queue
        from ocr_native import OcrEngine
        from video_utils import _preprocess_standard

        q: Queue = Queue(maxsize=max(1, self._buffer_size))
        results: dict = {}
        ocr_err: list = []
        ocr_wall = [0.0]
        ocr_ready = [False]   # raw 直通可用：worker 引擎就绪后置位（单 TRT）

        def _put(item) -> None:
            while True:
                if ocr_err:
                    raise ocr_err[0]
                try:
                    q.put(item, timeout=0.2)
                    return
                except Full:
                    continue

        def ocr_worker() -> None:
            t0 = time.perf_counter()
            try:
                if _ocr_engines is not None:
                    engines = list(_ocr_engines)
                    self._ocr_backend_used = (
                        'tensorrt+onnxruntime'
                        if len(engines) == 2 and engines[0].backend_name != engines[1].backend_name
                        else engines[0].backend_name)
                else:
                    _t_eng = time.perf_counter()
                    _engine_progress = lambda msg: self._progress(msg, 2.5)
                    ot = self._ocr_num_threads()
                    ocr_instances = (self._ocr_engine_type() == 'onnxruntime'
                                     and ot >= config.OCR_INSTANCES_MIN_THREADS
                                     and config.env_bool(config.OCR_INSTANCES_ENV,
                                                         default=True))
                    if ocr_instances:
                        half = max(2, ot // 2)
                        engines = [OcrEngine(self._ocr_model, 'onnxruntime', fill_width=self._fill_width, num_threads=half, progress_cb=_engine_progress) for _ in range(2)]
                    else:
                        engines = [OcrEngine(self._ocr_model, self._ocr_engine_type(), fill_width=self._fill_width, num_threads=ot, progress_cb=_engine_progress)]
                    self._ocr_backend_used = engines[0].backend_name
                    self._prof_end('ocr', 'engine_init', _t_eng)
                # 引擎就绪 → 供 GPU 管线 emit 决策（raw 直通需单 TRT 引擎；
                # 置位后该会话内代表帧可全程留显存，仅输出/回退时 D2H）。
                ocr_ready[0] = (len(engines) == 1
                                and getattr(engines[0], '_trt', None)
                                is not None)
                B = _ocr_batch_size()
                infer_q: Queue = Queue(maxsize=config.OCR_INFER_QUEUE_SIZE)
                ocr_progress_frac = [0.0]

                def _put_infer(item) -> bool:
                    while True:
                        if ocr_err:
                            return False
                        try:
                            infer_q.put(item, timeout=0.2)
                            return True
                        except Full:
                            continue

                def _report_ocr_progress(idx: int, frac: float) -> None:
                    if frac - ocr_progress_frac[0] >= 0.01 or frac >= 1.0:
                        ocr_progress_frac[0] = frac
                        self._progress(f'[OCR] 段 {idx + 1}', _ocr_progress_pct(frac))

                def infer_worker(eng) -> None:
                    try:
                        while True:
                            item = infer_q.get()
                            if item is None:
                                return
                            idxs, reps, procs, fracs, raw_infos = item
                            _t_i = time.perf_counter()
                            if raw_infos is not None:
                                res = eng.call_gpu_raw(
                                    raw_infos[1], force_aspect=raw_infos[0])
                            else:
                                res = eng(procs)
                            self._prof_end('ocr', 'infer', _t_i)
                            _t_c = time.perf_counter()
                            for idx, rep, r, frac in zip(idxs, reps, res, fracs):
                                if hasattr(r, 'txts'):
                                    raw_text = str(r.txts[0]) if r.txts and r.txts[0] else None
                                    scores = getattr(r, 'scores', [])
                                    ocr_conf = float(scores[0]) if scores else 0.0
                                else:
                                    raw_text, ocr_conf = (None, 0.0)
                                results[idx] = (raw_text, ocr_conf, rep)
                                _report_ocr_progress(idx, frac)
                            self._prof_end('ocr', 'ctc_decode', _t_c)
                    except Exception as e:
                        ocr_err.append(e)

                infer_threads = [
                    threading.Thread(target=infer_worker, args=(eng,), daemon=True)
                    for eng in engines]
                for t in infer_threads:
                    t.start()
                b_idx, b_reps, b_crops, b_devs, b_fracs = ([], [], [], [], [])

                def flush() -> None:
                    if not b_idx:
                        return
                    # 分流：带 dev 的项走 raw 直通（单 TRT 引擎时），带 crop 的
                    # 项走宿主预处理。两类可能并存于同一批（引擎就绪切换仅有
                    # 一批；ONNX/回退引擎全程 crop）→ 拆批投递，不混流。
                    # raw 代表帧：gray = decord gray NDArray 指针；yuv =
                    # _YFramePool 池帧提取的 Y 平面（由 GPU 管线保证）。
                    raw_sel = [
                        i for i in range(len(b_devs))
                        if b_devs[i] is not None
                        and len(engines) == 1
                        and getattr(engines[0], '_trt', None) is not None
                        and getattr(self, '_gpu_pipeline_mode', False)]
                    if raw_sel:
                        # 把 raw 任务交给 infer 线程异步执行，避免 OCR worker
                        # 被 GPU 预处理 + TRT 同步阻塞。载荷 = (force_aspect,
                        # infos)——raw 内核需按 force_aspect 决定 content 宽。
                        infos = [(d[1], d[2], d[3], d[0])
                                 for d in (b_devs[i] for i in raw_sel)]
                        if not _put_infer((
                                [b_idx[i] for i in raw_sel],
                                [b_reps[i] for i in raw_sel], None,
                                [b_fracs[i] for i in raw_sel],
                                (float(self._force_aspect), infos))):
                            return
                    host_sel = [
                        i for i in range(len(b_crops))
                        if b_crops[i] is not None]
                    if host_sel:
                        _t_p = time.perf_counter()
                        # 内容宽度自适应裁切（宽 ROI 字幕省卷积；见
                        # _crop_to_content 的实测与约束说明）
                        prepped = []
                        for i in host_sel:
                            c = b_crops[i]
                            if self._yuv_output:
                                c = _nv12_luma_full(
                                    c, self._color_range)[..., None]
                            prepped.append((i, _preprocess_standard(
                                self._crop_to_content(c),
                                force_aspect=self._force_aspect)))
                        self._prof_end('ocr', 'preprocess', _t_p)
                        # 跨批按宽度分组：pad 宽 = 批内最大宽，顺序分批时
                        # 每批都被满宽成员顶上去（实测收益 0%），只有把宽度
                        # 相近的段分到同一批才真的降下来（-23.9%）。
                        # OcrEngine.__call__ 内部虽已按宽度排序，但那只优化
                        # host resize 顺序，不改变 pad 宽度。
                        if self._ocr_reorder_window > 1:
                            prepped.sort(key=lambda t: t[1].shape[1])
                        for s in range(0, len(prepped), B):
                            chk = prepped[s:s + B]
                            if not _put_infer((
                                    [b_idx[t[0]] for t in chk],
                                    [b_reps[t[0]] for t in chk],
                                    [t[1] for t in chk],
                                    [b_fracs[t[0]] for t in chk], None)):
                                return
                    b_idx.clear()
                    b_reps.clear()
                    b_crops.clear()
                    b_devs.clear()
                    b_fracs.clear()

                while True:
                    _t_w = time.perf_counter()
                    item = q.get()
                    self._prof_end('ocr', 'q_get_wait', _t_w)
                    if item is None:
                        break
                    if ocr_err:
                        break
                    idx, rep, crop, dev, frac = item
                    b_idx.append(idx)
                    b_reps.append(rep)
                    b_crops.append(crop)
                    b_devs.append(dev)
                    b_fracs.append(frac)
                    # 攒够"重排窗口"再 flush：窗口 = 1 批时与旧行为一致。
                    # 窗口更大 → 可选出宽度更接近的同批组合，pad 更省；
                    # 代价只是 OCR 起步稍晚（吞吐不变，尾批由收尾 flush 兜住）。
                    if len(b_idx) >= max(B, self._ocr_reorder_window):
                        flush()
                flush()
                for _ in infer_threads:
                    while True:
                        try:
                            infer_q.put(None, timeout=0.2)
                            break
                        except Full:
                            if not any(t.is_alive() for t in infer_threads):
                                break
                for t in infer_threads:
                    t.join()
            except Exception as e:
                ocr_err.append(e)
            finally:
                ocr_wall[0] = time.perf_counter() - t0

        ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
        ocr_thread.start()

        def _finish() -> None:
            while True:
                try:
                    q.put(None, timeout=0.2)
                    break
                except Full:
                    if not ocr_thread.is_alive():
                        break
            ocr_thread.join()

        return {
            "q": q,
            "results": results,
            "err": ocr_err,
            "wall": ocr_wall,
            "thread": ocr_thread,
            "put": _put,
            "finish": _finish,
            "seg_idx": 0,
            "raw_ready": ocr_ready,
        }
