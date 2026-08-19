"""引擎识别链方法体（由 _gen_engine_extractor.py 生成，勿手改）。"""

@property
def frames(self) -> list:
    """全部采样帧号（run 后有效）。"""
    return self._frames

@frames.setter
def frames(self, v: list) -> None:
    self._frames = v

@property
def segment_frames(self) -> list:
    """每段的帧号序列（[[start..end], ...]）。"""
    return self._segs

@segment_frames.setter
def segment_frames(self, v: list) -> None:
    self._segs = v

@property
def ocr_values(self) -> list:
    """每段 OCR 原始读数（None=该段未读出）。"""
    return self._ocr_vals

@ocr_values.setter
def ocr_values(self, v: list) -> None:
    self._ocr_vals = v

@property
def ocr_texts(self) -> list:
    """每段 OCR 原始文本（识别层原始输出；速度解析前的源，None=未读出）。"""
    return self._ocr_texts

@ocr_texts.setter
def ocr_texts(self, v: list) -> None:
    self._ocr_texts = v

@property
def ocr_confidences(self) -> list:
    """每段 OCR 置信度（0-1，0.0=不可用）。"""
    return self._ocr_confs

@ocr_confidences.setter
def ocr_confidences(self, v: list) -> None:
    self._ocr_confs = v

@property
def corrected_values(self) -> list:
    """每段纠正后读数（DP/尖峰第二遍后；finalize 可重设）。"""
    return self._corr_vals

@corrected_values.setter
def corrected_values(self, v: list) -> None:
    self._corr_vals = v

@property
def confidence_values(self) -> list:
    """每段置信度（_dense_correct 前）。"""
    return self._conf_vals

@confidence_values.setter
def confidence_values(self, v: list) -> None:
    self._conf_vals = v

@property
def n_segments(self) -> int:
    """段总数（run 后有效；无段时 0）。"""
    return getattr(self, '_n_segments', 0)

@n_segments.setter
def n_segments(self, v: int) -> None:
    self._n_segments = v

@property
def n_corrected(self) -> int:
    """纠正段数（DP + 第二遍尖峰）。"""
    return getattr(self, '_n_corr', 0)

@n_corrected.setter
def n_corrected(self, v: int) -> None:
    self._n_corr = v

def _prof_end(self, group: str, key: str, t0: float) -> None:
    """累加一段耗时到 profile（线程安全；关闭时仅一次属性判断）。"""
    if not self._profile_enabled:
        return
    elapsed = time.perf_counter() - t0
    with self._prof_lock:
        d = self.profile.setdefault(group, {})
        d[key] = d.get(key, 0.0) + elapsed

def _open_vr(self):
    """按 decode_backend 打开 decord 解码器（auto/cpu/nvdec）。

        auto: 尝试 GPU (NVDEC) 失败回退 CPU。cpu: 强制 CPU。
        nvdec: 强制 GPU（失败回退 CPU 并警告）。替代旧 DECORD_FORCE_CPU env。
        混合（显式 cpu+nvdec 或 HYBRID_DECODE_ENV 开启）：走
        _open_hybrid_vrs（双解码器并行），不经过本方法。

        ROI-first（decord ≥0.7.5）：构造时传入固定 ROI（半开区间）——
        解码器只输出该矩形（CPU filter 先 crop 再转换 / GPU 转换 kernel
        只算 ROI 窗口 + 输出池 ROI 尺寸），免全帧转换与逐帧裁剪。
        """
    from decord import cpu as _cpu
    try:
        import decord.video_reader as _vr_mod
        _has_roi_api = hasattr(_vr_mod, '_CAPI_VideoReaderSetRoi')
    except ImportError:
        _has_roi_api = False
    roi = (self._roi[0], self._roi[1], self._roi[2] + 1, self._roi[3] + 1)
    roi_kw = {'roi': roi} if _has_roi_api else {}
    backend = (self._decode_backend or 'auto').lower()
    vr = None
    label = 'CPU'
    if backend in ('auto', 'nvdec'):
        try:
            from decord import gpu as _g
            vr = self._open_decord_reader(_g(0), roi_kw)
            label = 'GPU'
        except Exception:
            vr = None
            if backend == 'nvdec':
                logger.warning('NVDEC 解码不可用，回退 CPU')
    if vr is None:
        vr = self._open_decord_reader(_cpu(0), roi_kw, num_threads=self._decode_num_threads())
        label = 'CPU'
    self._backend = f'decord/{label}'
    if label == 'CPU':
        try:
            self._codec = str(vr.get_codec() or '').lower()
        except Exception:
            self._codec = ''
        if self._codec == 'av1':
            nt = self._decode_num_threads(codec='av1')
            if nt != self._decode_num_threads():
                vr = self._open_decord_reader(_cpu(0), roi_kw, num_threads=nt)
    else:
        try:
            self._codec = str(vr.get_codec() or '').lower()
        except Exception:
            self._codec = ''
    self._remember_color_range(vr)
    return vr

def _hybrid_env_enabled(self) -> bool:
    """实验开关 config.HYBRID_DECODE_ENV（RVTOL_HYBRID_DECODE）。

        1/true/yes/on（大小写不敏感）为开启，默认关闭。开启后 GPU 模式
        （auto / nvdec）内部改走 CPU+NVDEC 双解码器并行；不暴露给 GUI/CLI。
        """
    _v = _os.environ.get(config.HYBRID_DECODE_ENV, '').strip().lower()
    return _v in ('1', 'true', 'yes', 'on')

def _is_hybrid(self) -> bool:
    """是否启用 CPU+NVDEC 混合并行解码。

        显式传 decode_backend='cpu+nvdec'/'hybrid'（旧版程序化用法）恒为
        混合；否则需 HYBRID_DECODE_ENV 开启 且 后端为 GPU 系（auto /
        nvdec）——即"混合是 GPU 模式的实验变体"，cpu 不受影响。
        """
    _b = (self._decode_backend or 'auto').lower()
    if _b in HYBRID_BACKEND_ALIASES:
        return True
    return self._hybrid_env_enabled() and _b in ('auto', 'nvdec')

def _hybrid_split(self) -> float:
    """混合解码的 CPU 段帧数比例（env RVTOL_HYBRID_SPLIT 优先）。

        保守分法（默认 config.HYBRID_CPU_SPLIT）：只把 CPU 软解当"增量"。
        AV1 特判：CPU 软解 AV1 极耗核且与 GPU 段并发竞争反而拖慢 GPU 吞吐
        → 返回 0（CPU 段空，等效纯 GPU；_open_hybrid_vrs 已按纯 GPU 分支走，
        此返回为其他路径的防御性兜底）。
        """
    if getattr(self, '_hybrid_codec', '') == 'av1':
        return 0.0
    _env = _os.environ.get('RVTOL_HYBRID_SPLIT')
    if _env:
        try:
            v = float(_env)
            if 0.0 < v < 1.0:
                return v
        except ValueError:
            pass
    return float(config.HYBRID_CPU_SPLIT)

def _decord_format(self) -> str:
    """当前管线请求的 decord output_format。"""
    if self._yuv_output:
        return 'yuv420'
    return 'gray' if self._gray_output else 'rgb'

def _decode_num_threads(self, codec: str | None=None) -> int | None:
    """CPU 软解的 decord FFmpeg 帧线程数（少核/AV1 分核）。

        物理核 ≤ CPU_CORES_SPLIT_THRESHOLD（8）时返回 cores//2：FFmpeg
        fork 默认 2 帧线程只用 2 核，少核下解码成瓶颈，且 OCR 全核会与
        解码过订阅；实测（test5，affinity 模拟）4 核 28.0 vs 33.1s、
        8 核 17.8 vs 20.7s。核数多时（16）分核反而更差（12.0 vs 9.5s）
        → 返回 None（decord 默认，FFmpeg 帧线程落在 SMT 份额上）。
        codec='av1'：AV1 软解吞吐极低（~270fps vs h264 ~1247fps），解码
        是绝对瓶颈 → 解码分 max(2, min(cores*3//4, cores-2)) 核、OCR 保
        至少 2 线程。实测（test6）：16 核 dcd=12/ocrT=4 → 78.8s vs 现状
        87.4s（-10%）、8 核 dcd=6/ocrT=2 → 81.7s vs 101.2s（-19%）、
        4 核 dcd=2/ocrT=2 持平（ocrT=1 是灾难，ONNX 单线程追不上段率）。
        GPU(NVDEC) 不调用本方法。
        """
    from ocr_native import auto_ocr_thread_count
    cores = auto_ocr_thread_count()
    if codec == 'av1':
        return max(2, cores // 2)
    if cores <= config.CPU_CORES_SPLIT_THRESHOLD:
        return max(2, cores // 2)
    return None

def _open_decord_reader(self, ctx, roi_kw: dict, num_threads=None):
    """按当前输出格式打开 decord reader。

        yuv420 仅在 fork ≥0.7.10 可用：旧 DLL 会抛 ValueError，此时
        回退 gray（分段/OCR 不变，仅代表帧预览退化灰度）并重置标志。
        num_threads：CPU 软解的 FFmpeg 帧线程数（少核分核，None=decord
        默认；GPU/NVDEC 不传）。
        """
    from decord import VideoReader
    fmt = self._decord_format()
    nt_kw = {'num_threads': num_threads} if num_threads else {}
    try:
        return VideoReader(str(self._video_path), ctx=ctx, output_format=fmt, **nt_kw, **roi_kw)
    except ValueError:
        if not self._yuv_output:
            raise
        logger.warning('当前 decord 不支持 yuv420 输出，回退 gray （代表帧预览将为灰度）')
        self._yuv_output = False
        self._color_range = 0
        return VideoReader(str(self._video_path), ctx=ctx, output_format='gray', **nt_kw, **roi_kw)

def _remember_color_range(self, vr) -> None:
    """YUV 模式下从 decoder 读取流 color_range（0=limited/tv）。"""
    if not self._yuv_output:
        return
    try:
        self._color_range = int(vr.get_color_range() or 0)
    except Exception:
        self._color_range = 0

def _crop_luma(self, crop: np.ndarray) -> np.ndarray:
    """crop → 分段/OCR 灰度：YUV 时取 Y 并按 range 展开，否则 _gray_seg。"""
    if self._yuv_output:
        return _gray_seg_yuv(crop, self._color_range)
    return _gray_seg(crop)

def _batch_luma(self, crops: np.ndarray) -> np.ndarray:
    if self._yuv_output:
        return _gray_seg_yuv_batch(crops, self._color_range)
    return _gray_seg_batch(crops)

def _crop_is_expected(self, c: np.ndarray, roi_h: int, roi_w: int) -> bool:
    """ROI-first 输出尺寸是否符合当前输出格式（旧路径全帧则 False）。"""
    if self._yuv_output:
        return c.ndim == 2 and c.shape[0] == roi_h + (roi_h + 1) // 2 and (c.shape[1] == roi_w)
    return c.shape[0] == roi_h and c.shape[1] == roi_w

def _open_hybrid_vrs(self):
    """CPU+NVDEC 混合解码：打开一对 ROI-first 解码器（CPU 前段 + GPU 后段）。

        与 _open_vr 相同 ROI 语义（闭合框 → 半开 +1）。两个 reader 使用
        同一输出格式：gray（≥0.7.9 直出 Y）或 yuv420（≥0.7.10，Y 平面
        跨后端一致）。GPU 不可用 → 回退单 CPU reader（vr_gpu=None，
        调用方按纯 CPU 走）。
        AV1 特判：CPU 软解 AV1 极耗核（~330fps）且与 GPU 段并发竞争拖慢
        GPU 吞吐 → 不再打开 CPU reader，直接返回 (vr_gpu, vr_gpu)；调用方
        见 vr_gpu is vr → 置 hybrid=False 走纯 GPU 分支（无队列/线程开销，
        与纯 GPU 完全一致）。_hybrid_split 同步返回 0（防御性，其他路径兜底）。
        返回 (vr_cpu, vr_gpu)。
        """
    from decord import cpu as _cpu
    try:
        import decord.video_reader as _vr_mod
        _has_roi_api = hasattr(_vr_mod, '_CAPI_VideoReaderSetRoi')
    except ImportError:
        _has_roi_api = False
    roi = (self._roi[0], self._roi[1], self._roi[2] + 1, self._roi[3] + 1)
    roi_kw = {'roi': roi} if _has_roi_api else {}
    try:
        from decord import gpu as _g
        vr_gpu = self._open_decord_reader(_g(0), roi_kw)
        self._remember_color_range(vr_gpu)
        self._backend = 'decord/CPU+NVDEC'
    except Exception:
        logger.warning('NVDEC 解码不可用，CPU+NVDEC 回退纯 CPU')
        self._backend = 'decord/CPU'
        vr = self._open_decord_reader(_cpu(0), roi_kw, num_threads=self._decode_num_threads())
        self._remember_color_range(vr)
        return (vr, None)
    try:
        self._hybrid_codec = str(vr_gpu.get_codec() or '').lower()
    except Exception:
        self._hybrid_codec = ''
    if self._hybrid_codec == 'av1':
        logger.warning('AV1 视频：CPU 软解与 GPU 并发竞争反而拖慢解码，CPU+NVDEC 按纯 GPU 解码（不打开 CPU reader）')
        self._backend = 'decord/GPU'
        return (vr_gpu, vr_gpu)
    vr = self._open_decord_reader(_cpu(0), roi_kw, num_threads=self._decode_num_threads())
    self._remember_color_range(vr)
    return (vr, vr_gpu)

def _ocr_engine_type(self) -> str:
    """OCR 推理后端：auto/tensorrt → tensorrt（OcrEngine 失败回退 onnx），cpu → onnxruntime。"""
    return 'onnxruntime' if (self._ocr_backend or 'auto').lower() == 'cpu' else 'tensorrt'

def _ocr_num_threads(self) -> int:
    """OCR 推理线程预算：RVTOL_OCR_THREADS env 钩子优先，否则全物理核；
        CPU 软解且物理核 ≤ 8 时与解码显式分核（cores//2，防过订阅）。

        解码（NVDEC 全卸载 / CPU 下 FFmpeg 帧线程 2 + filter auto 只占
        SMT 份额）不抢物理核，OCR 吃满全部物理核；CPU 软解在少核机上
        FFmpeg 帧线程与 OCR 争抢（实测 4 核 ocrT=2 28.0s vs 全核 33.1s、
        8 核 ocrT=4 17.8s vs 20.7s），分核更优；核数多时（16）分核反而
        差 → 保持全核。显式参数传入引擎，不污染全局 env。
        """
    from ocr_native import auto_ocr_thread_count
    _env = _os.environ.get('RVTOL_OCR_THREADS')
    if _env:
        return max(1, int(_env))
    cores = auto_ocr_thread_count()
    if getattr(self, '_codec', '') == 'av1' and getattr(self, '_backend', '').startswith('decord/CPU'):
        return max(2, cores // 2)
    if getattr(self, '_backend', '').startswith('decord/CPU') and cores <= config.CPU_CORES_SPLIT_THRESHOLD:
        return max(2, cores // 2)
    return cores

def _decode_all(self):
    hybrid = self._is_hybrid()
    vr_gpu = None
    if hybrid:
        vr, vr_gpu = self._open_hybrid_vrs()
        if vr_gpu is None:
            hybrid = False
        elif vr_gpu is vr:
            hybrid = False
    else:
        vr = self._open_vr()
    if self._fps is None:
        for m in ('get_avg_fps', 'get_fps'):
            fn = getattr(vr, m, None)
            if fn is None:
                continue
            try:
                self._fps = float(fn())
                break
            except Exception:
                self._fps = None
        if not self._fps or self._fps <= 0:
            self._fps = config.DEFAULT_FPS_FALLBACK
    x1, y1, x2, y2 = self._roi
    total = len(vr)
    end = min(self._frame_end or total, total)
    if self._frame_start > 0:
        vr.seek_accurate(self._frame_start)
    frames = list(range(self._frame_start, end))
    DECODE_BATCH = config.DECODE_BATCH_SIZE
    crops = {}
    grays = {}
    sharp = {}
    t0 = time.perf_counter()
    if hybrid:
        import threading
        from queue import Queue
        cpu_fis, gpu_fis = _hybrid_ranges(frames, 0, self._hybrid_split())
        q_cpu: Queue = Queue(maxsize=config.HYBRID_QUEUE_SIZE)
        q_gpu = Queue(maxsize=config.HYBRID_QUEUE_SIZE) if gpu_fis else None
        err: list = []
        threads: list = []
        roi_half = (x1, y1, x2 + 1, y2 + 1)
        if cpu_fis:
            t = threading.Thread(target=_decode_range_worker, args=(vr, cpu_fis, q_cpu, roi_half, None, err, DECODE_BATCH, self._yuv_output, self._color_range), daemon=True)
            t.start()
            threads.append(t)
        else:
            q_cpu.put(None)
        if q_gpu is not None:
            try:
                vr_gpu.seek_accurate(gpu_fis[0])
            except Exception as e:
                err.append(e)
                q_gpu.put(None)
            else:
                t = threading.Thread(target=_decode_range_worker, args=(vr_gpu, gpu_fis, q_gpu, roi_half, None, err, DECODE_BATCH, self._yuv_output, self._color_range), daemon=True)
                t.start()
                threads.append(t)
        for q in (q_cpu, q_gpu):
            if q is None:
                continue
            for fi, c, g, s, _b in _drain_queue(q):
                if not self._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                    c = c[y1:y2 + 1, x1:x2 + 1]
                    g = self._crop_luma(c)
                    s = float(g.std())
                crops[fi] = c
                grays[fi] = g
                sharp[fi] = s
        for t in threads:
            t.join()
        if err:
            raise err[0]
    else:
        for k, fi in enumerate(frames):
            c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
            if not self._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                c = c[y1:y2 + 1, x1:x2 + 1]
            crops[fi] = c
            g = self._crop_luma(c)
            grays[fi] = g
            sharp[fi] = float(g.std())
            if k % 500 == 0:
                self._progress(f'[{self._backend}] 解码: {k}/{len(frames)}', 3 + k / max(len(frames), 1) * 70)
            if k % 100 == 0:
                self._cancel()
    self.timing['decode'] = time.perf_counter() - t0
    del vr, vr_gpu
    return (frames, crops, grays, sharp)

def _segment(self, frames, grays):
    t0 = time.perf_counter()
    ths = []
    step = max(1, len(frames) // config.SEG_CALIB_FRAMES)
    for fi in frames[::step][:config.SEG_CALIB_FRAMES]:
        ths.append(_otsu(grays[fi]))
    th = int(np.median(ths)) if ths else config.OTSU_FALLBACK_THRESH
    self._bin_thresh = th
    prev_b = grays[frames[0]] > th
    edges = []
    for fi in frames[1:]:
        b = grays[fi] > th
        d = prev_b != b
        edges.append(_cluster_win3(d) < self._C)
        prev_b = b
    segs = []
    s = 0
    for i in range(len(frames) - 1):
        if not edges[i]:
            segs.append(frames[s:i + 1])
            s = i + 1
    segs.append(frames[s:])
    self.timing['segment'] = time.perf_counter() - t0
    return segs

def _ocr_segments(self, segs, crops, sharp):
    from ocr_native import OcrEngine
    from video_utils import _preprocess_standard
    eng = OcrEngine(self._ocr_model, self._ocr_engine_type(), fill_width=self._fill_width, num_threads=self._ocr_num_threads(), progress_cb=lambda msg: self._progress(msg, 2.5))
    self._ocr_backend_used = eng.backend_name
    seg_vals = []
    rep_frames = []
    texts = []
    confs = []
    t0 = time.perf_counter()
    B = _ocr_batch_size()
    reps = [max(seg, key=lambda fi: sharp[fi]) for seg in segs]
    for k in range(0, len(segs), B):
        chunk = segs[k:k + B]
        procs = [_preprocess_standard(_nv12_luma_full(crops[rep], self._color_range)[..., None] if self._yuv_output else crops[rep], force_aspect=self._force_aspect) for rep in reps[k:k + B]]
        results = eng(procs)
        for rep, res in zip(reps[k:k + B], results):
            sv, _rt, _c = extract_speed_value(res)
            seg_vals.append(int(sv) if sv is not None and sv >= 0 else None)
            rep_frames.append(rep)
            if hasattr(res, 'txts'):
                texts.append(str(res.txts[0]) if res.txts and res.txts[0] else None)
                scores = getattr(res, 'scores', [])
                confs.append(float(scores[0]) if scores else 0.0)
            else:
                texts.append(None)
                confs.append(0.0)
        done = min(k + B, len(segs))
        self._progress(f'[OCR] 段: {done}/{len(segs)}', 73 + done / max(len(segs), 1) * 15)
    self.timing['ocr'] = time.perf_counter() - t0
    self._n_segments = len(segs)
    self._ocr_texts = texts
    self._ocr_confs = confs
    return (seg_vals, rep_frames)

def _run_pipelined(self):
    """流水线：解码线程增量分段，OCR 线程批处理已闭合段的代表帧。

        解码是 I/O 瓶颈（CPU 占用低），段边界（win3）在解码循环内增量计算，
        段一闭合就把代表帧（最清晰）交给 OCR 工作线程 —— 解码∥OCR 重叠摊薄
        总墙钟。代表帧选择与串行 _segment/_ocr_segments 完全一致（每段 max
        灰度 std），OCR 批 _ocr_batch_size()。cpu+nvdec 时两个解码线程
        （CPU 前段 + GPU 后段）并行填有界队列，消费者按序合并，帧序与单解码器一致。

        返回 (frames, segs, seg_vals, rep_frames)；self.crops = {rep_frame:
        crop}（仅代表帧，供 review 预览，比存全帧省内存）。
        """
    from queue import Queue
    import threading
    from ocr_native import OcrEngine
    from video_utils import _preprocess_standard
    from ocr_engine import extract_speed_value
    _t_open = time.perf_counter()
    hybrid = self._is_hybrid()
    vr_gpu = None
    if hybrid:
        vr, vr_gpu = self._open_hybrid_vrs()
        if vr_gpu is None:
            hybrid = False
    else:
        vr = self._open_vr()
    if self._fps is None:
        for m in ('get_avg_fps', 'get_fps'):
            fn = getattr(vr, m, None)
            if fn is None:
                continue
            try:
                self._fps = float(fn())
                break
            except Exception:
                self._fps = None
        if not self._fps or self._fps <= 0:
            self._fps = config.DEFAULT_FPS_FALLBACK
    x1, y1, x2, y2 = self._roi
    total = len(vr)
    end = min(self._frame_end or total, total)
    if self._frame_start > 0:
        vr.seek_accurate(self._frame_start)
    frames = list(range(self._frame_start, end))
    self._prof_end('producer', 'open_and_fps', _t_open)
    calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
    calib: list = []
    _t_cal = time.perf_counter()
    for k in range(calib_n):
        _t_p = time.perf_counter()
        c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
        self._prof_end('producer', 'calib_decode', _t_p)
        if not self._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
            c = c[y1:y2 + 1, x1:x2 + 1]
        _t_p = time.perf_counter()
        g = self._crop_luma(c)
        self._prof_end('producer', 'calib_gray', _t_p)
        calib.append((frames[k], c, g, float(g.std())))
    ths = [_otsu(g) for _fi, _c, g, _s in calib]
    th = int(np.median(ths)) if ths else config.OTSU_FALLBACK_THRESH
    self._bin_thresh = th
    self._prof_end('producer', 'calib_total', _t_cal)
    q: Queue = Queue(maxsize=max(1, self._buffer_size))
    results: dict = {}
    ocr_err: list = []
    ocr_wall = [0.0]

    def ocr_worker() -> None:
        t0 = time.perf_counter()
        try:
            _hybrid_ocr = _os.environ.get(config.HYBRID_OCR_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on')
            _t_eng = time.perf_counter()
            _engine_progress = lambda msg: self._progress(msg, 2.5)
            if _hybrid_ocr:
                engines = [OcrEngine(self._ocr_model, 'tensorrt', fill_width=self._fill_width, num_threads=self._ocr_num_threads(), progress_cb=_engine_progress), OcrEngine(self._ocr_model, 'onnxruntime', fill_width=self._fill_width, num_threads=self._ocr_num_threads(), progress_cb=_engine_progress)]
            else:
                ot = self._ocr_num_threads()
                dual_onnx = self._ocr_engine_type() == 'onnxruntime' and ot >= 8 and (_os.environ.get('RVTOL_DUAL_ONNX', '1') != '0')
                if dual_onnx:
                    half = max(2, ot // 2)
                    engines = [OcrEngine(self._ocr_model, 'onnxruntime', fill_width=self._fill_width, num_threads=half, progress_cb=_engine_progress) for _ in range(2)]
                else:
                    engines = [OcrEngine(self._ocr_model, self._ocr_engine_type(), fill_width=self._fill_width, num_threads=ot, progress_cb=_engine_progress)]
            self._ocr_backend_used = 'tensorrt+onnxruntime' if len(engines) == 2 and engines[0].backend_name != engines[1].backend_name else engines[0].backend_name
            self._prof_end('ocr', 'engine_init', _t_eng)
            B = _ocr_batch_size()
            infer_q: Queue = Queue(maxsize=config.OCR_INFER_QUEUE_SIZE)
            ocr_progress_frac = [0.0]

            def _report_ocr_progress(idx: int, frac: float) -> None:
                if frac - ocr_progress_frac[0] >= 0.01 or frac >= 1.0:
                    ocr_progress_frac[0] = frac
                    self._progress(f'[OCR] 段 {idx + 1}', 58.0 + frac * 28.0)

            def infer_worker(eng) -> None:
                while True:
                    item = infer_q.get()
                    if item is None:
                        return
                    idxs, reps, procs, fracs = item
                    _t_i = time.perf_counter()
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
                        sv, _rt, _c = extract_speed_value(r)
                        results[idx] = (int(sv) if sv is not None and sv >= 0 else None, raw_text, ocr_conf, rep)
                        _report_ocr_progress(idx, frac)
                    self._prof_end('ocr', 'ctc_decode', _t_c)
            infer_threads = [threading.Thread(target=infer_worker, args=(eng,), daemon=True) for eng in engines]
            for t in infer_threads:
                t.start()
            b_idx, b_reps, b_crops, b_fracs = ([], [], [], [])

            def flush() -> None:
                if not b_idx:
                    return
                _t_p = time.perf_counter()
                procs = [_preprocess_standard(_nv12_luma_full(c, self._color_range)[..., None] if self._yuv_output else c, force_aspect=self._force_aspect) for c in b_crops]
                self._prof_end('ocr', 'preprocess', _t_p)
                infer_q.put((list(b_idx), list(b_reps), procs, list(b_fracs)))
                b_idx.clear()
                b_reps.clear()
                b_crops.clear()
                b_fracs.clear()
            while True:
                _t_w = time.perf_counter()
                item = q.get()
                self._prof_end('ocr', 'q_get_wait', _t_w)
                if item is None:
                    break
                idx, rep, crop, frac = item
                b_idx.append(idx)
                b_reps.append(rep)
                b_crops.append(crop)
                b_fracs.append(frac)
                if len(b_idx) >= B:
                    flush()
            flush()
            for _ in infer_threads:
                infer_q.put(None)
            for t in infer_threads:
                t.join()
        except Exception as e:
            ocr_err.append(e)
        finally:
            ocr_wall[0] = time.perf_counter() - t0
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()
    DECODE_BATCH = config.DECODE_BATCH_SIZE
    dec_threads: list = []
    dec_err: list = []
    if hybrid:
        from queue import Queue as _Queue
        cpu_fis, gpu_fis = _hybrid_ranges(frames, calib_n, self._hybrid_split())
        cpu_q: _Queue = _Queue(maxsize=config.HYBRID_QUEUE_SIZE)
        gpu_q = _Queue(maxsize=config.HYBRID_QUEUE_SIZE) if gpu_fis else None
        roi_half = (x1, y1, x2 + 1, y2 + 1)
        if cpu_fis:
            t = threading.Thread(target=_decode_range_worker, args=(vr, cpu_fis, cpu_q, roi_half, th, dec_err, DECODE_BATCH, self._yuv_output, self._color_range), daemon=True)
            t.start()
            dec_threads.append(t)
        else:
            cpu_q.put(None)
        if gpu_q is not None:
            try:
                vr_gpu.seek_accurate(gpu_fis[0])
            except Exception as e:
                dec_err.append(e)
                gpu_q.put(None)
            else:
                t = threading.Thread(target=_decode_range_worker, args=(vr_gpu, gpu_fis, gpu_q, roi_half, th, dec_err, DECODE_BATCH, self._yuv_output, self._color_range), daemon=True)
                t.start()
                dec_threads.append(t)

        def frame_stream():
            """先产出校准帧（CPU reader），再按序消费 CPU 段队列
                与 GPU 段队列 —— 帧序与单解码器完全一致。"""
            for fi, c, g, s in calib:
                yield (fi, c, g, s, g > th)
            yield from _drain_queue(cpu_q)
            if gpu_q is not None:
                yield from _drain_queue(gpu_q)
    else:

        def frame_stream():
            """先产出校准帧，再批量流式解码剩余帧。

                yield (fi, crop, gray, sharp, bin) —— bin 为预计算的二值化。
                """
            for fi, c, g, s in calib:
                yield (fi, c, g, s, g > th)
            for bstart in range(calib_n, len(frames), DECODE_BATCH):
                bend = min(bstart + DECODE_BATCH, len(frames))
                _t_d = time.perf_counter()
                crops = vr.get_batch(frames[bstart:bend], roi=(x1, y1, x2 + 1, y2 + 1)).asnumpy()
                self._prof_end('producer', 'decode_batch', _t_d)
                _t_g = time.perf_counter()
                g = self._batch_luma(crops)
                self._prof_end('producer', 'gray_batch', _t_g)
                _t_s = time.perf_counter()
                sharp = g.std(axis=(1, 2))
                self._prof_end('producer', 'sharp_batch', _t_s)
                _t_b = time.perf_counter()
                bs = g > th
                self._prof_end('producer', 'bin_batch', _t_b)
                for k, gi in enumerate(range(bstart, bend)):
                    yield (frames[gi], crops[k], g[k], float(sharp[k]), bs[k])
    segs: list = []
    rep_crops: dict = {}
    seg_idx = 0
    s = 0
    rep_frame = frames[0]
    rep_crop = None
    rep_sharp = -1.0
    prev_b = None
    t0 = time.perf_counter()
    consumer_ok = [False]
    try:
        for k, (fi, c, g, sharp, b) in enumerate(frame_stream()):
            if prev_b is not None:
                d = prev_b != b
                _t_seg = time.perf_counter()
                changed = _cluster_win3(d) >= self._C
                self._prof_end('producer', 'segmentation', _t_seg)
                if changed:
                    seg = frames[s:k]
                    segs.append(seg)
                    _t_push = time.perf_counter()
                    q.put((seg_idx, rep_frame, rep_crop, k / max(len(frames), 1)))
                    self._prof_end('producer', 'q_put_block', _t_push)
                    rep_crops[rep_frame] = rep_crop
                    seg_idx += 1
                    s = k
                    rep_frame = fi
                    rep_crop = c
                    rep_sharp = sharp
                elif sharp > rep_sharp:
                    rep_sharp = sharp
                    rep_frame = fi
                    rep_crop = c
            else:
                rep_frame = fi
                rep_crop = c
                rep_sharp = sharp
            prev_b = b
            if k % 100 == 0:
                self._cancel()
            if k % 500 == 0:
                self._progress(f'[{self._backend}] 解码+分段: {k}/{len(frames)}', 3 + k / max(len(frames), 1) * 55)
        seg = frames[s:]
        segs.append(seg)
        _t_push = time.perf_counter()
        q.put((seg_idx, rep_frame, rep_crop, 1.0))
        self._prof_end('producer', 'q_put_block', _t_push)
        rep_crops[rep_frame] = rep_crop
        seg_idx += 1
        consumer_ok[0] = True
    finally:
        _t_consume_end = time.perf_counter()
        self.timing['decode'] = _t_consume_end - t0
        self._prof_end('producer', 'consumer_total', t0)
        if consumer_ok[0]:
            for t in dec_threads:
                t.join()
        q.put(None)
        ocr_thread.join()
        self.timing['ocr_tail'] = time.perf_counter() - _t_consume_end
    if dec_err:
        raise dec_err[0]
    if ocr_err:
        raise ocr_err[0]
    self.timing['ocr'] = ocr_wall[0]
    self._n_segments = len(segs)
    self.crops = rep_crops
    del vr, vr_gpu
    self._ocr_texts = [results[i][1] for i in range(seg_idx)]
    self._ocr_confs = [results[i][2] for i in range(seg_idx)]
    return (frames, segs, [results[i][0] for i in range(seg_idx)], [results[i][3] for i in range(seg_idx)])

def prepare_review_rgb(self) -> None:
    """最终检查前：把全部代表帧 packed YUV420 就地转成 RGB。

        只转换代表帧（每段一张，不转换全片帧）：test5 ~2.5k 段、
        test6 ~8.1k 段均为毫秒~亚秒级 numpy 操作。转换后释放
        self.crops 的 YUV 引用（segments 内已换成 RGB，finalize 不需要）。
        """
    if not self._yuv_output:
        return
    for seg in self.segments:
        crop = seg.get('rep_crop')
        if crop is not None and crop.ndim == 2:
            seg['rep_crop'] = nv12_to_rgb(crop)
    self.crops.clear()

def timing_flat(self) -> dict:
    """展平 timing dict（丢弃嵌套值），兼容 headless/gui_export 调用。"""
    return {k: v for k, v in self.timing.items() if isinstance(v, (int, float))}