"""OCR 消费会话（OcrSession）——宿主管线与 GPU 全驻留管线共用的 OCR 端。

从 _host_pipeline._start_ocr_session 类化拆出（0.11.0）：段任务队列 →
预处理(raw 直通判定) → 宽度重排分批 → 推理 worker（进程级引擎池取还）。
一次 extract() 一个实例；GPU 路径把设备指针任务放进同一队列，raw 直通由
raw_ready 判定分流。
"""
from __future__ import annotations

import logging
import sys
import threading

import engine_config as config
from segmentation import preprocess_standard
from ._helpers import _ocr_batch_size, _ocr_progress_pct

logger = logging.getLogger(__name__)


class OcrSession:
    """OCR 消费会话：段任务队列 → 预处理(raw 直通判定) → 宽度重排分批 →
    推理 worker（进程级引擎池取还）→ 结果收敛。

    属性：
      q          — 段任务队列（生产者 put，OCR worker 消费）
      results    — 全局段索引 → (text, conf, rep_frame)
      err        — worker 异常（首个）；非空时生产者应中止并上抛
      wall       — [ocr 总耗时]（单元素列表，worker 退出时写入）
      raw_ready  — [bool] 单 TRT 引擎就绪 → 代表帧可全程留显存走 raw 直通
    方法：put(item)（C7 取消响应式入队）、finish()（哨兵 + join）。
    """

    def __init__(self, ex, _ocr_engines: list | None = None) -> None:
        from queue import Queue

        self._ex = ex
        self._owns_engines = _ocr_engines is None
        self.q: "Queue" = Queue(maxsize=max(1, ex._buffer_size))
        self.results: dict = {}
        self.err: list = []
        self.wall = [0.0]
        self.raw_ready = [False]   # raw 直通可用：worker 引擎就绪后置位（单 TRT）
        # 批量 autocrop 挂点（GPU 管线注入，2026-09-10）：autocropper 存在时，
        # raw 项 emit 为 5 元组 (owner,ptr,h,w,sharp)，autocrop 推迟到 flush
        # 批量执行（一次 kernel 处理整批段，消费端每段的 launch+sync 开销
        # 在 hybrid 引擎口径下曾吃掉 ~14% 墙钟）。None = emit 内逐段裁切。
        self.autocropper = None
        self._engines: list = []
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── 生产者接口 ──────────────────────────────────────────

    def put(self, item) -> None:
        """入队一个段任务；队列满时保持取消响应（C7）。"""
        from queue import Full
        while True:
            if self.err:
                raise self.err[0]
            try:
                self.q.put(item, timeout=0.2)
                return
            except Full:
                self._ex._cancel()
                continue

    def finish(self) -> None:
        """投递哨兵并等待 OCR worker 退出。"""
        from queue import Full
        while True:
            try:
                self.q.put(None, timeout=0.2)
                break
            except Full:
                if not self._thread.is_alive():
                    break
        self._thread.join()

    # ── OCR worker（引擎取还 / 预处理分流 / 推理）────────────

    @staticmethod
    def _bump_priority() -> None:
        """Windows：本线程提到 ABOVE_NORMAL（D2，2026-09-10）。

        hybrid 的 CPU 软解线程（libavcodec 池，NORMAL）吃满物理核时会
        饥饿本线程的 Python 调度：infer 相位随 HYBRID_CPU_THREADS 单调
        膨胀（nvdec 1.01s → hybrid nt16 1.36s / 3000 帧，OCR 走 GPU 不
        缺 CPU）。OCR 是临界路径，升一档让它在与解码线程争核时先行；
        线程退出即失效，无跨会话影响。非 Windows 与失败路径静默。
        """
        if sys.platform != 'win32':
            return
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            if not k32.SetThreadPriority(k32.GetCurrentThread(), 1):
                pass  # 设置失败（句柄/权限）：维持 NORMAL，无正确性影响
        except Exception:  # noqa: BLE001
            pass  # ctypes 缺失/调用失败：维持 NORMAL，无正确性影响（见上 docstring）

    def _worker(self) -> None:
        import time
        from queue import Full, Queue
        from ocr_native import acquire_ocr_engine, checkin_ocr_engine
        self._bump_priority()
        ex = self._ex
        t0 = time.perf_counter()
        engines: list = []
        failed = False
        try:
            if not self._owns_engines:
                engines = list(self._engines)
                ex._ocr_backend_used = (
                    'tensorrt+onnxruntime'
                    if len(engines) == 2 and engines[0].backend_name != engines[1].backend_name
                    else engines[0].backend_name)
            else:
                _t_eng = time.perf_counter()
                ot = ex._ocr_num_threads()
                engine_type = ex._ocr_engine_type()
                ocr_instances = (engine_type == 'onnxruntime'
                                 and ot >= config.OCR_INSTANCES_MIN_THREADS
                                 and config.env_bool(config.OCR_INSTANCES_ENV,
                                                     default=True))
                try:
                    # 进程级引擎池（B5/C5）：跨视频复用 TRT 上下文/设备
                    # 缓冲，免去每个 extract 重付反序列化+分配。
                    if ocr_instances:
                        half = max(2, ot // 2)
                        engines = [
                            acquire_ocr_engine(
                                ex._ocr_model, 'onnxruntime',
                                fill_width=ex._fill_width,
                                num_threads=half)
                            for _ in range(2)]
                    else:
                        engines = [acquire_ocr_engine(
                            ex._ocr_model, engine_type,
                            fill_width=ex._fill_width, num_threads=ot)]
                except BaseException:
                    logger.debug("引擎半建状态回收", exc_info=True)
                    # 半建状态（如第二实例构建失败）：已取到的引擎归池
                    while engines:
                        try:
                            checkin_ocr_engine(engines.pop())
                        except Exception:
                            pass  # 归池失败无更好的回收路径（原 _host_pipeline 存量语义）
                    raise
                ex._ocr_backend_used = engines[0].backend_name
                if (engine_type == 'tensorrt'
                        and engines[0].backend_name != 'tensorrt'):
                    # D3：请求 TRT 但引擎回退 ONNX → 降级原因透出 meta
                    ex._degraded.append('TRT 引擎不可用，回退 ONNX')
                ex._prof_end('ocr', 'engine_init', _t_eng)
            # 引擎就绪 → 供 GPU 管线 emit 决策（raw 直通需单 TRT 引擎；
            # 置位后该会话内代表帧可全程留显存，仅输出/回退时 D2H）。
            self.raw_ready[0] = (len(engines) == 1
                                 and getattr(engines[0], '_trt', None)
                                 is not None)
            B = _ocr_batch_size()
            # TRT 批对齐 max_batch（§8.1）：TRT 按 profile max_batch 切
            # 子批，末批 batch 维变化前必须 cudaStreamSynchronize（TRT
            # 不允许 in-flight 改 context 形状）——OCR_BATCH=16 在
            # max_batch=6 下切成 6+6+4，**每个 OCR 批一次全流水同步**
            # （实测 test5 全片 CPU+TRT：批 16→18 墙钟 -10.3%、
            # ocr_infer -13%）。单 TRT 引擎时分块对齐到 max_batch 整倍数
            # 即零 sync；ONNX 路径保持 B（18=16+2 切伤 ONNX_CHUNK=16
            # 尾批，实测 +3.7%），双实例同理。
            _trt_max = (engines[0]._trt.max_batch
                        if (len(engines) == 1 and
                            getattr(engines[0], '_trt', None) is not None)
                        else 0)
            chunk = (max(B, -(-B // _trt_max) * _trt_max)
                     if _trt_max else B)
            infer_q: Queue = Queue(maxsize=config.OCR_INFER_QUEUE_SIZE)
            ocr_progress_frac = [0.0]

            def _put_infer(item) -> bool:
                while True:
                    if self.err:
                        return False
                    try:
                        infer_q.put(item, timeout=0.2)
                        return True
                    except Full:
                        continue

            def _report_ocr_progress(idx: int, frac: float) -> None:
                if frac - ocr_progress_frac[0] >= 0.01 or frac >= 1.0:
                    ocr_progress_frac[0] = frac
                    ex._progress(f'[OCR] 段 {idx + 1}',
                                 _ocr_progress_pct(frac))

            def infer_worker(eng) -> None:
                self._bump_priority()
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
                        ex._prof_end('ocr', 'infer', _t_i)
                        _t_c = time.perf_counter()
                        for idx, rep, r, frac in zip(idxs, reps, res, fracs):
                            if hasattr(r, 'txts'):
                                raw_text = str(r.txts[0]) if r.txts and r.txts[0] else None
                                scores = getattr(r, 'scores', [])
                                ocr_conf = float(scores[0]) if scores else 0.0
                            else:
                                raw_text, ocr_conf = (None, 0.0)
                            self.results[idx] = (raw_text, ocr_conf, rep)
                            _report_ocr_progress(idx, frac)
                        ex._prof_end('ocr', 'ctc_decode', _t_c)
                except Exception as e:
                    self.err.append(e)

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
                    and getattr(ex, '_gpu_pipeline_mode', False)]
                if raw_sel:
                    # 批量预处理（emit 时推迟的 5 元组项）：yuv 批量 luma
                    # 提取 + 一次 col_ink_batch 裁切区间，整批一次 sync，
                    # 再按「裁后内容宽」分组。单帧判定内核与宿主
                    # _crop_to_content 同判据，逐位一致。
                    deferred = [i for i in raw_sel if len(b_devs[i]) == 5]
                    if deferred and self.autocropper is not None:
                        for i, dev6 in zip(
                                deferred,
                                self.autocropper.process(
                                    [b_devs[i] for i in deferred])):
                            b_devs[i] = dev6
                    elif deferred:
                        # autocropper 缺席兜底：全宽（与未裁语义一致）
                        for i in deferred:
                            o, p, h, w, _sharp = b_devs[i]
                            b_devs[i] = (o, p, h, w, 0, w)
                    # 跨批按「裁后内容宽」分组（与宿主裁切路径同一策略）：
                    # 顺序分批时每批几乎必有满宽成员 → pad 宽被顶回全宽，
                    # 裁切收益归零；把宽度相近的段分到同一批才真的降下来。
                    # 6 元组 dev = (owner, ptr, h, w, x_off, crop_w)。
                    if ex._ocr_reorder_window > 1:
                        raw_sel.sort(
                            key=lambda i: (b_devs[i][5]
                                           if len(b_devs[i]) >= 6
                                           else b_devs[i][3]))
                    # 把 raw 任务交给 infer 线程异步执行，避免 OCR worker
                    # 被 GPU 预处理 + TRT 同步阻塞。载荷 = (force_aspect,
                    # infos)——raw 内核按 force_aspect / 逐项区间决定内容宽。
                    # 按批大小拆子批投递：pad 宽 = 子批内最大内容宽
                    # （与宿主裁切路径的子批结构一致）。
                    for s in range(0, len(raw_sel), chunk):
                        chk = raw_sel[s:s + chunk]
                        infos = []
                        for i in chk:
                            d = b_devs[i]
                            if len(d) >= 6:
                                infos.append((d[1], d[2], d[3], d[0],
                                              int(d[4]), int(d[5])))
                            else:
                                infos.append((d[1], d[2], d[3], d[0]))
                        if not _put_infer((
                                [b_idx[i] for i in chk],
                                [b_reps[i] for i in chk], None,
                                [b_fracs[i] for i in chk],
                                (float(ex._force_aspect), infos))):
                            return
                host_sel = [
                    i for i in range(len(b_crops))
                    if b_crops[i] is not None]
                if host_sel:
                    _t_p = time.perf_counter()
                    # 内容宽度自适应裁切（宽 ROI 字幕省卷积；统一实现见
                    # segmentation.crop_to_content / crop_after_aspect 与
                    # 其 docstring 的实测与约束说明）。
                    prepped = []
                    for i in host_sel:
                        c = b_crops[i]
                        if ex._yuv_output:
                            from video_utils import _nv12_luma_full
                            c = _nv12_luma_full(
                                c, ex._color_range)[..., None]
                        if ex._force_aspect and ex._force_aspect > 0:
                            # force_aspect>0：**先定比例、后裁**（顺序 ⑦）。
                            # 反序（先裁再定比例）会因内容宽高比被改变而
                            # 引入畸变，实测更差（test5 9 vs 0）。
                            p = preprocess_standard(
                                c, force_aspect=ex._force_aspect)
                            p = ex._crop_after_aspect(p)
                        else:
                            p = preprocess_standard(
                                ex._crop_to_content(c),
                                force_aspect=ex._force_aspect)
                        prepped.append((i, p))
                    ex._prof_end('ocr', 'preprocess', _t_p)
                    # 跨批按宽度分组：pad 宽 = 批内最大宽，顺序分批时
                    # 每批都被满宽成员顶上去（实测收益 0%），只有把宽度
                    # 相近的段分到同一批才真的降下来（-23.9%）。
                    # OcrEngine.__call__ 内部虽已按宽度排序，但那只优化
                    # host resize 顺序，不改变 pad 宽度。
                    if ex._ocr_reorder_window > 1:
                        prepped.sort(key=lambda t: t[1].shape[1])
                    for s in range(0, len(prepped), chunk):
                        chk = prepped[s:s + chunk]
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
                item = self.q.get()
                ex._prof_end('ocr', 'q_get_wait', _t_w)
                if item is None:
                    break
                if self.err:
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
                if len(b_idx) >= max(chunk, ex._ocr_reorder_window):
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
            failed = True
            self.err.append(e)
        finally:
            self.wall[0] = time.perf_counter() - t0
            if self._owns_engines and engines:
                # C5/B5：池来源的引擎归还复用；失败退出 → release
                # （引擎状态可能被污染，不回池）。
                for eng in engines:
                    try:
                        if failed:
                            eng.release()
                        else:
                            checkin_ocr_engine(eng)
                    except Exception:
                        pass  # 归还/释放失败只能忽略：worker 退出路径（原存量语义）
