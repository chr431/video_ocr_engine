"""单实例双完整流水线（_DualPipelineMixin）：kfe 唯一分片 + 竞争闸门 + 端到端让位。

从 extractor.py 拆出：双流水线全套方法（互补后端 / 试点组 / kfe 切片 / 双
OcrEngine 线程预算 / 慢路径让位 / 并行取片合并）。FieldExtractor 组合本 mixin
获得这些方法。让位判定与 INFLIGHT 竞争闸门为 kfe 的平衡机制，保持不变。
"""
import logging
import os as _os
import time

import engine_config as config
from ocr_native import auto_ocr_thread_count
from segmentation import _otsu
from video_utils import nvdec_available, tensorrt_available
from ._helpers import (_ocr_batch_size, _otsu_median_threshold,
                       _read_fps_from_vr)

logger = logging.getLogger("video_ocr_engine.extractor")


class _DualPipelineMixin:
    # ═══════════════ 单实例双完整流水线并行（kfe 唯一分片） ═══════════════

    @staticmethod
    def _opposite_decode(backend: str) -> str:
        """互补解码后端：CPU 软解 ↔ auto（NVDEC 优先）。"""
        return "auto" if str(backend or "").strip().lower() == "cpu" else "cpu"

    @staticmethod
    def _opposite_ocr(backend: str) -> str:
        """互补 OCR 后端：TRT ↔ ONNX，两条流水线分别用 GPU/CPU 硬件。

        与下游 video_subtitle_extractor --dual 的互补策略一致：主后端为
        GPU/TRT 时副线程用 CPU+ONNX，主后端为 CPU 时副线程换回 TRT。
        早期“混配必互相拖慢”的结论被后续定位修正：真正的瓶颈是全局阈值
        路径缺少 seek_accurate 到片首导致 CPU 解码随机访问减半，以及混配下
        让位把并行对端交给慢路径；修正后显式混配已能接近双 TRT 并显著
        优于单 TRT（见 docs/PERFORMANCE.md 4.5 节）。
        """
        _b = str(backend or "").strip().lower()
        return "auto" if _b in ("cpu", "onnxruntime") else "cpu"

    def _dual_backend_pairs(self) -> list[tuple[str, str]]:
        """返回两条流水线的 (decode, ocr) 后端组合。

        默认：主后端 + 互补后端（CPU ↔ GPU/TRT）。调用方可显式传
        dual_backends=[('cpu','auto'), ('cpu','auto')] 等自定义组合；
        少于两条时复制第一条补足两条。
        """
        if self._dual_backends:
            pairs = [tuple(p) for p in self._dual_backends]
            if len(pairs) == 1:
                pairs = pairs * 2
            return pairs[:2]
        main = (self._decode_backend or "auto", self._ocr_backend or "auto")
        opp = (self._opposite_decode(main[0]), self._opposite_ocr(main[1]))
        pairs = [main]
        if opp != main:
            pairs.append(opp)
        return pairs

    @staticmethod
    def _nearest_keyframe_sample(target: int, key_frames: list[int],
                                 frames: list[int]) -> int:
        """返回离 target 最近的关键帧，再吸附到最近的采样帧号（保持采样网格）。"""
        import bisect
        if not key_frames or not frames:
            return target
        idx = bisect.bisect_left(key_frames, target)
        cand = [idx - 1, idx]
        cand = [i for i in cand if 0 <= i < len(key_frames)]
        if not cand:
            return target
        key = min((key_frames[i] for i in cand),
                  key=lambda k: (abs(k - target), k))
        sidx = bisect.bisect_left(frames, key)
        sc = [sidx - 1, sidx]
        sc = [i for i in sc if 0 <= i < len(frames)]
        if not sc:
            return target
        return min((frames[i] for i in sc),
                   key=lambda f: (abs(f - key), f))

    @classmethod
    def _keyframe_every_chunks(cls, frames: list[int],
                               key_frames: list[int], rest_start: int,
                               last_end: int, stride: int, min_gap: int,
                               max_chunks: int) -> list[tuple[int, int]]:
        """每关键帧一片（kfe）——双流水线唯一分片方法：竞争区切片生成。

        按基础最小片间距切；若关键帧过密（mkv 重编码 ~每 30-140 源帧一个
        关键帧）导致片数超过上限，逐步放大间距合并，片数受控在 max_chunks
        以内。边界吸附到最近采样帧（保持全帧覆盖、无缝隙/无重叠；吸附帧离
        关键帧 ≤ stride/2，seek_accurate 仍便宜）。返回覆盖 [rest_start,
        last_end) 的连续切片列表（首片起点=rest_start，末片终点=last_end）。
        """
        _key_list = [k for k in key_frames if rest_start < k < last_end]
        if not _key_list:
            return [(rest_start, last_end)]
        _mg = max(1, int(min_gap))
        _mx = max(1, int(max_chunks))
        _s = max(1, int(stride))
        _big: list[tuple[int, int]] = [(rest_start, last_end)]
        for _iter in range(80):
            _cand: list[tuple[int, int]] = []
            _prev2 = rest_start
            for _k in _key_list:
                _b = cls._nearest_keyframe_sample(_k, key_frames, frames)
                if (_b - _prev2) // _s >= _mg and _b < last_end:
                    _cand.append((_prev2, _b))
                    _prev2 = _b
            _cand.append((_prev2, last_end))
            _big = _cand
            if len(_cand) - 1 <= _mx:
                break
            _mg = max(_mg + 1, int(_mg * 1.5))
        return _big

    @classmethod
    def _dual_chunk_specs(cls, frames: list[int], key_frames: list[int], *,
                          last_end: int, stride: int,
                          min_gap: int | None = None,
                          max_chunks: int | None = None,
                          unit_div: int | None = None,
                          min_chunk: int | None = None,
                          pilots: int = 4) -> tuple[list[tuple[int, int]], bool]:
        """双流水线切片（kfe 唯一分片方法）：头部试点小片组 + 关键帧竞争区。

        头部 pilots 个小片（各约 1/unit_div 视频长，至少 min_chunk 采样帧）
        保留给两条流水线各领一组（试点→确认），用于启动竞态消解与端到端让位
        取样；试点之外的大竞争区交给 _keyframe_every_chunks 按关键帧每片一切
        ——不再等分成固定块（dual-2 / DUAL_PROPORTIONAL 已移除）。无关键帧或
        视频过短时自然退化为单大竞争片，全帧覆盖无缝。

        返回 (chunk_specs, has_pilots)。has_pilots 表示是否预留了头部试点组
        （false 时整段都进竞争区，末尾按流水线数预留单片）。
        """
        _unit_div = max(2, int(unit_div if unit_div is not None
                              else config.DUAL_PIPELINE_PILOT_DIV))
        _min_chunk = max(1, int(min_chunk if min_chunk is not None
                               else config.DUAL_PIPELINE_MIN_CHUNK_FRAMES))
        _unit_n = max(_min_chunk, len(frames) // _unit_div)
        _has_pilots = pilots * _unit_n < len(frames)
        _chunks: list[tuple[int, int]] = []
        if _has_pilots:
            for i in range(pilots):
                _a = i * _unit_n
                _b = (i + 1) * _unit_n
                _chunks.append((frames[_a], frames[_b]))
            _rest = frames[pilots * _unit_n]
        else:
            _rest = frames[0]
        _mg = max(1, int(min_gap if min_gap is not None
                        else config.DUAL_KEYFRAME_EVERY_MIN_GAP))
        _mx = max(1, int(max_chunks if max_chunks is not None
                        else config.DUAL_KEYFRAME_EVERY_MAX_CHUNKS))
        _chunks.extend(cls._keyframe_every_chunks(
            frames, key_frames, _rest, last_end, max(1, int(stride)),
            _mg, _mx))
        return _chunks, _has_pilots

    def _new_worker(self, decode_backend: str, ocr_backend: str,
                    progress_cb=None, cancel_check=None) -> "FieldExtractor":
        """创建一条子流水线实例（关闭 dual，避免递归）。"""
        # 延迟导入避免循环依赖：extractor 顶层 import 本 mixin，
        # 方法被调用时两个模块都已加载完成，再取类名是安全的。
        from video_ocr_engine.extractor import FieldExtractor  # noqa: F401
        return FieldExtractor(
            str(self._video_path), self._roi,
            frame_start=self._frame_start,
            frame_end=self._frame_end,
            force_aspect=self._force_aspect,
            decode_backend=decode_backend,
            ocr_backend=ocr_backend,
            buffer_size=self._buffer_size,
            fill_width=self._fill_width,
            C=self._C,
            fps=self._fps,
            sample_stride=self._sample_stride,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
            gray_output=self._gray_output,
            yuv_output=self._yuv_output,
            keep_crops=self._keep_crops,
            keep_frames=self._keep_frames,
            merge_similar=self._merge_similar,
            merge_similar_threshold=self._merge_similar_threshold,
            dual_pipeline=False)

    def _dual_ocr_num_threads(self, ocr_backend: str = "",
                              n_cpu_peers: int = 1,
                              has_trt_peer: bool = False) -> int:
        """双流水线的 OCR 线程预算（按消费者后端分核，消除满核×2 过订阅）。

        env OCR_THREADS 优先（实验钩子，显式即全量生效）。否则：
        - TRT（auto/tensorrt）侧：DUAL_PIPELINE_TRT_CPU_THREADS（默认 2）——
          推理在 GPU、预处理是 worker 单线程 numpy，多线程无收益；
        - ONNX（cpu）侧：(物理核 - TRT 预算) // CPU 侧消费者数，下限 2——
          独占剩余物理核，同时给 FFmpeg 软解/系统留出余量；
        - 混配保护（has_trt_peer）：另一条流水线在跑 TRT 时，ONNX 侧进一步
          封顶 DUAL_PIPELINE_ONNX_PEER_THREADS——ONNX 满核计算会饥饿 TRT
          宿主提交线程（实测 TRT 2.57→4.47ms/段，限 6 线程恢复到 3.39）。
        """
        _env = _os.environ.get(config.OCR_THREADS_ENV)
        if _env and _env.isdigit():
            return max(1, int(_env))
        kind = (ocr_backend or 'auto').strip().lower()
        trt_budget = max(1, int(config.DUAL_PIPELINE_TRT_CPU_THREADS))
        if kind in ('auto', 'tensorrt'):
            return trt_budget
        n = max(2, (auto_ocr_thread_count() - trt_budget)
                // max(1, int(n_cpu_peers)))
        if has_trt_peer:
            n = min(n, max(2, int(config.DUAL_PIPELINE_ONNX_PEER_THREADS)))
        return n

    @staticmethod
    def _dual_should_yield(my_fps: float, other_fps: float,
                           ratio: float, remaining_after: int) -> bool:
        """慢路径让位判定：滚动吞吐显著落后且快路径仍有余量可接手。

        my_fps/other_fps 为两条流水线的滚动帧率；ratio 越小越保守。
        remaining_after 保证让位后队列至少还剩 1 片给快路径，避免误伤。
        """
        if ratio <= 0.0 or my_fps <= 0.0 or other_fps <= 0.0:
            return False
        return my_fps < ratio * other_fps and remaining_after >= 1

    def _run_pipelined_parallel(self):
        """单实例双完整流水线并行：同一视频切多片，两流水线动态取片。

        与旧“混合解码/混合 OCR”只在一个阶段内并行的方案不同：
        这里是两条完整“解码→分段→OCR”流水线各自带互补后端（如 GPU+TRT 与
        CPU+ONNX），从共享队列取连续小片，谁快谁多干，避免机械对半切导致
        快流水线闲置。最后按片序合并段文本/置信度/代表帧。

        需要 NVDEC 与 TensorRT 均可用；不满足则回退单流水线（复用探测解码
        器，不重复打开）。

        相对初版的改进（2026-08）：
        - 探测/全局校准/移交一体：主线程用 ROI-first reader 读元数据并做
          全局 Otsu 校准一次（消除每片阈值漂移），再把 reader 移交给同
          后端的第一条流水线复用（省一次 GPU reader 打开）；
        - 切片 = kfe（唯一分片方法，dual-2 等分/比例分配已移除）：头部
          试点×2 + 确认×2 小片预留（各约 1/24 视频长，消解启动竞态并给
          让位判定取样），试点之外的大竞争区按剩余区域内的关键帧边界每片
          一切（过密时按 MIN_GAP/MAX_CHUNKS 放大间距合并，片数受控），
          无关键帧/视频过短时自然退化为单个大竞争片；
        - 竞争取片闸门（INFLIGHT 上限）+ 端到端让位：慢路径按含 OCR 排空
          的端到端速率显著落后时停止取片，剩余片由快路径完成，避免尾部
          等待（AV1 等编码下 CPU+ONNX 慢路径不再拖垮整体）；
        - 跨片边界 merge_similar 缝合：相邻片尾/首段代表帧相似则合并，
          OCR 结果沿用前段（丢弃被并入段的重复识别），与单流水线行为对齐。
        """
        from queue import Empty, Queue
        import threading
        from ocr_native import OcrEngine

        # ── 1. 探测解码器（ROI-first）：元数据 + 全局校准 + 移交一体 ──
        _t_probe = time.perf_counter()
        try:
            _vr = self._open_vr()
        except Exception as e:  # noqa: BLE001
            logger.warning(f'双流水线探测解码器打开失败，回退单流水线: {e}')
            return self._run_pipelined(_force_single=True)
        self._prof_end('parallel', 'probe_open', _t_probe)
        # 双流水线需要 TensorRT；NVDEC 探测在主后端为 CPU 系时补一次
        # （主后端为 GPU 系时 probe 打开成功即已证明）。不满足回退单流水线，
        # 复用探测解码器（不重复打开）。
        if not tensorrt_available() or (
                not self._backend.startswith('decord/GPU')
                and not nvdec_available(str(self._video_path))):
            logger.warning(
                '单实例双流水线需要 NVDEC 和 TensorRT 均可用，回退单流水线')
            return self._run_pipelined(_force_single=True, _external_vr=_vr)
        _fps = _read_fps_from_vr(_vr) or config.DEFAULT_FPS_FALLBACK
        total = len(_vr)
        if self._fps is None:
            self._fps = _fps
        end = min(self._frame_end or total, total)
        frames = list(range(self._frame_start, end, self._sample_stride))
        if not frames:
            raise ValueError(
                f"帧区间为空: frame_start={self._frame_start}, "
                f"frame_end={end}, total={total}")
        if len(frames) < 2:
            return self._run_pipelined(_force_single=True, _external_vr=_vr)
        # 最小帧数门控：短窗口摊不平双流水线固定开销（探测/校准、第二套
        # OCR 引擎初始化、跨片边界），实测反而变慢，直接回退单流水线。
        if len(frames) < max(2, int(config.DUAL_PIPELINE_MIN_FRAMES)):
            logger.info(
                '采样帧数 %d < %d，双流水线固定开销不划算，回退单流水线',
                len(frames), config.DUAL_PIPELINE_MIN_FRAMES)
            return self._run_pipelined(_force_single=True, _external_vr=_vr)

        pairs = self._dual_backend_pairs()
        # 编码回退（校准前，零额外开销）：默认互补组合下，已知 CPU 软解
        # 净负的编码直接回退单流水线；显式 dual_backends 视为用户知情选择，
        # 不回退。env DUAL_NO_CODEC_FALLBACK=1 关闭。
        _codec_fb = tuple(getattr(config, 'DUAL_PIPELINE_CODEC_FALLBACK', ()))
        if (_codec_fb and self._dual_backends is None and self._codec
                and self._codec.lower() in _codec_fb
                and not config.env_bool(config.DUAL_PIPELINE_NO_CODEC_FALLBACK_ENV)):
            logger.info(
                '编码 %s 下互补 CPU 流水线已知净负，双流水线回退单流水线'
                '（DUAL_NO_CODEC_FALLBACK=1 可关闭）', self._codec)
            return self._run_pipelined(_force_single=True, _external_vr=_vr)

        # 全局 Otsu 校准只做一次（前 SEG_CALIB_FRAMES 个采样帧，与单流水线
        # 语义一致）：各片共享同一二值化阈值，消除跨片阈值漂移导致的分段
        # 边界不一致，也省去每片重复的 50 帧校准解码。
        x1p, y1p, x2p, y2p = self._roi
        calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
        ths: list = []
        _t_cal0 = time.perf_counter()
        _cal_nds = _vr.get_batch(frames[:calib_n],
                                 roi=(x1p, y1p, x2p + 1, y2p + 1))
        _cal_crops = _cal_nds.asnumpy()
        del _cal_nds
        self._prof_end('parallel', 'calib_decode', _t_cal0)
        _t_cal1 = time.perf_counter()
        for k in range(calib_n):
            c = _cal_crops[k]
            if not self._crop_is_expected(c, y2p - y1p + 1, x2p - x1p + 1):
                c = c[y1p:y2p + 1, x1p:x2p + 1]
            ths.append(_otsu(self._crop_luma(c)))
        del _cal_crops
        th = _otsu_median_threshold(ths)
        self._bin_thresh = th
        self._prof_end('parallel', 'calib_otsu', _t_cal1)

        # ── 2. 切片：头部小片（试点×2 + 确认×2）+ 关键帧竞争区 ──
        # kfe 是双流水线唯一分片方法：头部 4 个小片（各约 1/DIV 视频长）由
        # 两条流水线各领一个试点、再各取一个确认片二次取样（启动竞态消解与
        # 端到端让位取样都依赖它）；试点之外的大竞争区不再等分成固定块
        # （dual-2 / DUAL_PROPORTIONAL 等过时方法已移除），而是按剩余区域
        # 内每个关键帧边界切一片交给共享队列竞争。边界落在关键帧上、seek
        # 便宜（相邻片连续扫掠免精确 seek）；关键帧过密时按
        # DUAL_KEYFRAME_EVERY_MIN_GAP / _MAX_CHUNKS 放大间距合并，片数受控。
        # 无关键帧/视频过短时自然退化为单个大竞争片。
        last_end = end if self._frame_end not in (None, 0) else total
        try:
            _key_frames = [int(v) for v in _vr.get_key_indices()]
        except Exception:
            _key_frames = []
        _min_gap_env = _os.environ.get(
            config.DUAL_KEYFRAME_EVERY_MIN_GAP_ENV, '').strip()
        _min_gap = (max(1, int(_min_gap_env))
                    if _min_gap_env and _min_gap_env.isdigit()
                    else config.DUAL_KEYFRAME_EVERY_MIN_GAP)
        _max_chunks_env = _os.environ.get(
            config.DUAL_KEYFRAME_EVERY_MAX_CHUNKS_ENV, '').strip()
        _max_chunks = (max(1, int(_max_chunks_env))
                       if _max_chunks_env and _max_chunks_env.isdigit()
                       else config.DUAL_KEYFRAME_EVERY_MAX_CHUNKS)
        chunk_specs, has_pilots = self._dual_chunk_specs(
            frames, _key_frames, last_end=last_end,
            stride=self._sample_stride, min_gap=_min_gap,
            max_chunks=_max_chunks)
        n_specs = len(chunk_specs)
        # 探测解码器移交给后端方向相同的第一条流水线复用（省一次 GPU
        # reader 打开）。全局校准已把移交方解码器推进到帧头附近（帧 50），
        # 故把靠后的头部片组（idx 2/3）分给移交方（沿帧序单调前进、免向后
        # 精确 seek）；新开解码器的一方领 idx 0/1（从当前位置自然起步）。
        probe_is_gpu = self._backend.startswith('decord/GPU')
        handoff_ci = None
        for ci, (dec, _ob) in enumerate(pairs):
            gpu_intent = (dec or 'auto').strip().lower() in ('auto', 'nvdec')
            if gpu_intent == probe_is_gpu:
                handoff_ci = ci
                break
        # 预留：每条流水线固定领走一组头部片（试点+确认，共 4 片中的 2 片，
        # init 完成即按序处理），剩余大片进队列竞争。消除启动竞态——TRT
        # 反序列化与 ONNX 加载耗时不同，若全部切片先入共享队列，先就绪者
        # 可能抢光切片使另一条空转退化。头部组同时充当让位判定的两次取样，
        # 失衡时慢路径最多"浪费"自己那组小片（约 2×1/DIV 视频长）。
        n_reserve = min(len(pairs), n_specs)
        if has_pilots and n_reserve == 2:
            groups = [(0, 1), (2, 3)]
            if handoff_ci == 0:
                reserved_idx = {0: groups[0], 1: groups[1]}
            elif handoff_ci == 1:
                reserved_idx = {0: groups[1], 1: groups[0]}
            else:
                reserved_idx = {ci: groups[ci] for ci in range(n_reserve)}
            compete_range = range(4, n_specs)
        else:
            # 无头部组的短视频回退：尾部预留单片。
            reserved_idx = {ci: (n_specs - n_reserve + ci,)
                            for ci in range(n_reserve)}
            compete_range = range(n_specs - n_reserve)
        n_compete = len(list(compete_range))
        item_q: Queue = Queue()
        for idx in compete_range:
            spec = chunk_specs[idx]
            item_q.put((idx, spec[0], spec[1]))
        remaining = [len(compete_range)]  # 竞争队列剩余片数（让位判定用）

        # ── 3. 两个消费者线程：每条完整流水线 + 持久 OCR 引擎 ──
        result_lock = threading.Lock()
        errors: list = []
        cancel_event = threading.Event()
        chunk_results: dict = {}
        worker_stats: dict = {}
        ready_t: dict = {}
        e2e_speed: dict = {}    # tag -> 端到端吞吐（帧/片起点到该片 OCR 排空的墙钟）
        prog_lock = threading.Lock()
        prog_last = [-1.0]
        slow_ratio = float(config.DUAL_PIPELINE_SLOW_RATIO)
        _env_ratio = _os.environ.get(config.DUAL_PIPELINE_SLOW_RATIO_ENV)
        if _env_ratio:
            try:
                slow_ratio = max(0.0, float(_env_ratio))
            except ValueError:
                pass
        # 混配（TRT ⊕ ONNX）时默认让位阈值用独立常量：两条路径分属 GPU/CPU，
        # 阈值 0.5 在 h264 对比路径不触发、在 AV1 极端失衡时让快路径接管；
        # 早期“直接禁用让位”在 AV1 关闭回退下无法止损，而 0.8 在 h264 会误让。
        if _env_ratio is None:
            _ocr_kinds = {
                'onnxruntime'
                if (ob or '').strip().lower() in ('cpu', 'onnxruntime')
                else 'tensorrt'
                for _dec, ob in pairs}
            if len(_ocr_kinds) > 1:
                slow_ratio = float(config.DUAL_PIPELINE_MIXED_SLOW_RATIO)

        def _chunk_progress(idx: int, n: int):
            def cb(msg: str, pct: float) -> None:
                overall = ((idx + min(max(float(pct), 0.0), 100.0) / 100.0)
                           / n * 100.0)
                with prog_lock:
                    if overall <= prog_last[0]:
                        return
                    prog_last[0] = overall
                self._progress(f'[并行 {idx + 1}/{n}] {msg}', overall)
            return cb

        def _consumer(decode_backend: str, ocr_backend: str, tag: str,
                      ci: int, handoff_vr) -> None:
            worker = self._new_worker(
                decode_backend, ocr_backend,
                progress_cb=None, cancel_check=self._cancel)
            try:
                if handoff_vr is not None:
                    # 复用主线程探测解码器（同后端方向）；探测阶段写入的
                    # 实例字段同步给子 worker（统计/颜色域/阈值语义一致）。
                    worker_vr = handoff_vr
                    worker._backend = self._backend
                    worker._codec = self._codec
                    worker._color_range = self._color_range
                else:
                    worker_vr = worker._open_vr()
            except Exception as e:  # noqa: BLE001
                with result_lock:
                    errors.append(e)
                cancel_event.set()
                return
            with result_lock:
                ready_t[tag] = time.perf_counter()
            n_cpu_peers = max(1, sum(
                1 for _d, ob in pairs
                if (ob or '').strip().lower() == 'cpu'))
            has_trt_peer = any(
                (ob or 'auto').strip().lower() in ('auto', 'tensorrt')
                for _d, ob in pairs)
            try:
                eng = OcrEngine(
                    self._ocr_model,
                    worker._ocr_engine_type(),
                    fill_width=self._fill_width,
                    num_threads=self._dual_ocr_num_threads(
                        worker._ocr_engine_type(), n_cpu_peers,
                        has_trt_peer),
                    progress_cb=lambda m: self._progress(
                        f'[{tag}] {m}', 2.5))
            except Exception as e:  # noqa: BLE001
                with result_lock:
                    errors.append(e)
                cancel_event.set()
                return
            # 一个 worker 只开一个持久 OCR 会话：所有切片共用它的队列和
            # infer 线程。切片之间不再 join，后一片解码可与前一片 OCR 重叠。
            session = worker._start_ocr_session([eng])
            chunk_meta: dict = {}
            chunks_done = 0
            wall = 0.0
            yielded = [False]
            timeline: list = []   # (idx, t_start, t_end, n_frames) 剖面用
            _prev_end = [None]    # 上一片结束时刻（拉片空隙统计）
            # 竞争取片闸门用：pending = 本流水线已取但 OCR 尚未排空的片
            # （存每片 push 完成后的全局段计数）；prev_end_abs = 上一片终点
            # 绝对帧号——下一片起点与之相邻时判定“连续扫掠”，免 seek_accurate
            # （实测连续 ~1ms vs 乱序跳跃 40-70ms）。
            pending: list = []
            prev_end_abs = [None]
            inf_cap = config.DUAL_PIPELINE_INFLIGHT
            _env_inf = _os.environ.get(
                config.DUAL_PIPELINE_INFLIGHT_ENV, '').strip()
            if _env_inf and _env_inf.isdigit():
                inf_cap = max(1, int(_env_inf))
            # 端到端吞吐跟踪：e2e_last = (片起点时刻, 片帧数)。竞争闸门保证
            # 最近一片已排空（含半批容忍）后，(now - 片起点) 即该片真正端到端
            # 墙钟（含 OCR 排空尾），用它做竞争/让位决策——免疫“解码快、OCR
            # 慢”路径（如 CPU+ONNX 宽 ROI 字幕）生产者速率虚高导致的误判。
            e2e_last = [None]

            def _do_chunk(idx: int, start: int, end_f: int,
                          seek_required: bool = True) -> None:
                nonlocal chunks_done, wall
                worker._frame_start = int(start)
                worker._frame_end = int(end_f)
                worker._progress = _chunk_progress(idx, n_specs)
                _t_chunk = time.perf_counter()
                gap = (_t_chunk - _prev_end[0]
                       if _prev_end[0] is not None else 0.0)
                try:
                    (segs, keys, reps, crops_chunk, dec_elapsed,
                     g_first, g_last, prod_elapsed) = self._run_parallel_chunk(
                        worker, worker_vr, session, idx,
                        start, end_f, n_specs, th,
                        seek_required=seek_required)
                except Exception as e:  # noqa: BLE001
                    with result_lock:
                        errors.append(e)
                    cancel_event.set()
                    raise
                chunk_time = time.perf_counter() - _t_chunk
                wall += chunk_time
                chunks_done += 1
                pending.append(int(session["seg_idx"]))
                prev_end_abs[0] = int(end_f)
                _prev_end[0] = time.perf_counter()
                timeline.append((idx, round(_t_chunk, 3),
                                 round(_prev_end[0], 3),
                                 len(range(int(start), min(int(end_f), total),
                                           self._sample_stride)),
                                 round(gap, 3)))
                chunk_meta[idx] = (segs, keys, reps, crops_chunk,
                                   dec_elapsed, g_first, g_last)
                # 端到端速率由竞争闸门排空后按 (此刻 - 片起点) 记录（_e2e_fps），
                # 此处只保留片帧数给 e2e_last；生产者净速率口径已随 proportional/
                # priority 一并移除。
                n_fr = len(range(int(start), min(int(end_f), total),
                                 self._sample_stride))
                e2e_last[0] = (_t_chunk, n_fr)

            def _e2e_fps() -> float:
                """最近一片的端到端速率：帧数 / (此刻 - 片起点)。

                配合竞争闸门（片在取下一片前排空，含半批容忍），
                此刻-片起点 已包含该片 OCR 排空尾，即真实端到端墙钟。
                未处理过任何片时返回 0（让位判定视为尚未有依据）。
                """
                if e2e_last[0] is None:
                    return 0.0
                _t0, _fr = e2e_last[0]
                _dt = time.perf_counter() - _t0
                return _fr / _dt if _dt > 0 else 0.0

            def _other_best_e2e() -> float:
                """对方流水线的端到端速率最大值（e2e_speed，最近一次在竞争
                闸门排空后记录的口径）。"""
                best = 0.0
                with result_lock:
                    for t2, v in e2e_speed.items():
                        if t2 != tag and v > best:
                            best = v
                return best

            try:
                # 预留头部片组：init 完成即按序处理（试点→确认），把本流水线
                # 的竞争闸门/端到端让位依据尽快推进到位。
                for ridx in reserved_idx.get(ci, ()):
                    if cancel_event.is_set():
                        break
                    try:
                        _spec = chunk_specs[ridx]
                        _seek = (prev_end_abs[0] is None
                                 or _spec[0] != prev_end_abs[0])
                        _do_chunk(ridx, _spec[0], _spec[1],
                                  seek_required=_seek)
                    except Exception:  # noqa: BLE001 — 已入 errors
                        break
                while not cancel_event.is_set():
                    # 让位判定（分级）：端到端速率比极端悬殊时单个试点片即可判定
                    # （阈值 0.35，容忍首次解码 warm-up 噪声）；一般悬殊需
                    # 两次取样确认——试点片含 warm-up，单次比值噪声大
                    # （实测 test3 GPU 试点被 warm-up 拖低而误判让位）。
                    min_samples = 2 if has_pilots else 1
                    # 竞争片密（关键帧切片等细粒度，n_compete>=6）时单片/双片
                    # 测速噪声大（试点头片含解码 warm-up），让位需更多取样才
                    # 确认，防把实际更快的 GPU 路径误判为慢路径而误让位。
                    if n_compete >= 6:
                        min_samples = 4
                    if slow_ratio > 0 and chunks_done >= 1:
                        with result_lock:
                            rem = remaining[0]
                        # 让位判定用端到端速率（含 OCR 排空）：生产者净速率
                        # 会把“解码快、OCR 慢”路径（CPU+ONNX 宽 ROI）误判为快，
                        # 让位方向反了（慢路径抢片、快路径误让）。端到端口径
                        # 来自竞争闸门排空后记录的 e2e_speed，双方都至少取过
                        # 一片竞争片后才可能触发，天然规避试点头片 warm-up 噪声。
                        my_fps = e2e_speed.get(tag, 0.0)
                        other_fps = _other_best_e2e()
                        confirmed = self._dual_should_yield(
                            my_fps, other_fps,
                            slow_ratio if chunks_done >= min_samples else 0.0,
                            rem)
                        extreme = (
                            rem >= 1 and my_fps > 0.0
                            and other_fps > 0.0
                            and chunks_done >= min_samples
                            and my_fps < config.DUAL_PIPELINE_EXTREME_SLOW_RATIO * other_fps)
                        if confirmed or extreme:
                            yielded[0] = True
                            break
                    # 竞争取片闸门（in-flight 片数上限）：本流水线“已取但 OCR
                    # 尚未排空”的片数达到上限时暂停取片，等自己的 OCR 追上来，
                    # 让对方取——防“解码快、OCR 慢”路径在自由竞争中跑得太前
                    # （抢占过多切片却因 OCR 瓶颈拖慢整体墙钟）。片数口径与
                    # 内容无关，免疫“分段稀疏时段做不了多少 OCR 工作”的偏差。
                    # 排空判定带半批容忍（≤B-1 段）：OCR worker 会把不足一
                    # 批（16）的尾部段先攒在 b_idx，等下一片补齐才 flush——
                    # 若 producer 在此精确等待 len(results) ≥ pu 而队列空，
                    # 双方互等死锁（producer 等排空、OCR worker 等下一片补齐
                    # 批次）。容忍最后一半批未 flush 后，producer 先取下一片，
                    # 下一片头部段补齐批次即可恢复前进。
                    if inf_cap > 0:
                        _grace = _ocr_batch_size() - 1
                        while (not cancel_event.is_set()
                               and not item_q.empty()):
                            _rlen = len(session["results"])
                            _inflight = sum(
                                1 for pu in pending if pu > _rlen + _grace)
                            if _inflight < inf_cap:
                                break
                            time.sleep(0.02)
                    # 竞争闸门已保证最近一片排空（含半批容忍）→ 此刻
                    # (now - 片起点) 为该片真正端到端耗时，记录为端到端
                    # 速率（竞争/让位的准确口径；双方都取过至少一片后才有值）。
                    with result_lock:
                        e2e_speed[tag] = _e2e_fps()
                    try:
                        item = item_q.get_nowait()
                    except Empty:
                        break
                    with result_lock:
                        remaining[0] -= 1
                    idx, start, end_f = item
                    _seek = (prev_end_abs[0] is None
                             or start != prev_end_abs[0])
                    try:
                        _do_chunk(idx, start, end_f,
                                  seek_required=_seek)
                    except Exception:  # noqa: BLE001 — 已入 errors
                        break
            finally:
                _t_drain0 = time.perf_counter()
                try:
                    session["finish"]()
                except Exception as e:  # noqa: BLE001
                    with result_lock:
                        if not errors:
                            errors.append(e)
                    cancel_event.set()
                drain_s = time.perf_counter() - _t_drain0
                if session["err"]:
                    with result_lock:
                        if not errors:
                            errors.append(session["err"][0])
                # OCR 会话结束后按 chunk 内全局段索引组装结果
                for idx in sorted(chunk_meta):
                    (segs, keys, reps, crops_chunk, dec_elapsed,
                     g_first, g_last) = chunk_meta[idx]
                    texts: list = []
                    confs: list = []
                    reps_out: list = []
                    for k, rep in zip(keys, reps):
                        item = session["results"].get(k)
                        if item is not None:
                            texts.append(item[0])
                            confs.append(item[1])
                            reps_out.append(item[2])
                        else:
                            texts.append(None)
                            confs.append(0.0)
                            reps_out.append(rep)
                    with result_lock:
                        chunk_results[idx] = {
                            "segs": segs, "texts": texts, "confs": confs,
                            "reps": reps_out, "crops": crops_chunk,
                            "decode": dec_elapsed, "g_first": g_first,
                            "g_last": g_last,
                            "ocr_backend": worker._ocr_backend_used,
                            "backend": worker._backend}
                with result_lock:
                    worker_stats[tag] = {
                        "chunks": chunks_done, "wall": wall + drain_s,
                        "busy_wall": wall, "drain": drain_s,
                        "yielded": yielded[0],
                        "timeline": timeline,
                        "profile": (worker.profile
                                    if worker._profile_enabled else {}),
                        "backend": worker._backend,
                        "ocr": worker._ocr_backend_used,
                        "ocr_wall": session["wall"][0]}

        threads = [
            threading.Thread(
                target=_consumer,
                args=(dec, ocr, f'pipe{i + 1}', i,
                      _vr if i == handoff_ci else None),
                daemon=True)
            for i, (dec, ocr) in enumerate(pairs)
        ]
        if handoff_ci is None:
            del _vr  # 无后端方向匹配的消费者（罕见）：探测 reader 就地释放
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        if len(chunk_results) != n_specs:
            raise RuntimeError(
                f"双流水线切片结果不完整: {len(chunk_results)}/{n_specs}")

        # ── 4. 按片序合并（帧序全局单调）+ 跨片边界 merge_similar 缝合 ──
        # 每片只保留首/末段代表帧灰度：相邻片的末段/首段在片界被硬切开，
        # 相似（同一视觉内容）则并入前段，OCR 文本/置信度沿用前段，被并入
        # 段的识别结果直接丢弃——与单流水线的连续 merge_similar 行为对齐。
        rows: list = []   # [seg, text, conf, rep, boundary_gray]
        all_crops: dict = {}
        timing_sum: dict = {}
        backend_names: list = []
        ocr_backend_names: list = []
        stitched = 0
        prev_boundary_gray = None
        for i in sorted(chunk_results):
            cr = chunk_results[i]
            segs = cr["segs"]
            n_seg = len(segs)
            for j, (seg, tx, cf, rep) in enumerate(
                    zip(segs, cr["texts"], cr["confs"], cr["reps"])):
                if j == 0:
                    gray_here = cr["g_first"]
                elif j == n_seg - 1:
                    gray_here = cr["g_last"]
                else:
                    gray_here = None
                if (self._merge_similar and rows and j == 0
                        and prev_boundary_gray is not None
                        and gray_here is not None
                        and self._segments_similar(prev_boundary_gray,
                                                   gray_here)):
                    rows[-1][0].extend(seg)
                    if self._keep_crops:
                        all_crops.pop(rep, None)
                    stitched += 1
                    continue
                rows.append([seg, tx, cf, rep,
                             gray_here if j == n_seg - 1 else None])
            prev_boundary_gray = cr["g_last"]
            all_crops.update(cr["crops"])
            timing_sum['decode'] = timing_sum.get('decode', 0.0) + float(
                cr['decode'])
            ocr_backend_names.append(cr["ocr_backend"] or "")
            backend_names.append(cr["backend"] or "")
        all_segs = [r[0] for r in rows]
        all_texts = [r[1] for r in rows]
        all_confs = [r[2] for r in rows]
        all_reps = [r[3] for r in rows]
        self._frames = frames
        self._segs = all_segs
        self.crops = all_crops
        self._ocr_texts = all_texts
        self._ocr_confs = all_confs
        self._n_segments = len(all_segs)
        self._backend = "dual:" + "+".join(backend_names)
        self._ocr_backend_used = "+".join(ocr_backend_names)
        self.timing = timing_sum
        self.timing['parallel_probe'] = time.perf_counter() - _t_probe
        if stitched:
            self.timing['parallel_stitched'] = stitched
        if ready_t and len(ready_t) > 1:
            self.timing['parallel_reserve_skew'] = (
                max(ready_t.values()) - min(ready_t.values()))
        self.timing['parallel_yield_ratio'] = slow_ratio
        if self._profile_enabled:
            # 剖面聚合：各流水线 worker 的 producer/ocr 分相耗时按 tag 汇入
            # self.profile（group 键加 pipe 前缀），并记录分片时间线
            # (idx, t0, t1, frames, gap)——用于定位双流水线的串行化/空隙。
            for tag in sorted(worker_stats):
                st = worker_stats[tag]
                self.timing[f'parallel_{tag}_timeline'] = st["timeline"]
                for grp, phases in st.get("profile", {}).items():
                    dst = self.profile.setdefault(f'{grp}:{tag}', {})
                    for k, v in phases.items():
                        dst[k] = dst.get(k, 0.0) + float(v)
        # 每条流水线完成的片数/墙钟（含 OCR 会话排水；诊断 GPU/CPU 是否闲置）
        ocr_walls: list = []
        for tag in sorted(worker_stats):
            st = worker_stats[tag]
            self.timing[f'parallel_{tag}_chunks'] = st["chunks"]
            self.timing[f'parallel_{tag}_s'] = st["wall"]
            self.timing[f'parallel_{tag}_drain'] = st["drain"]
            self.timing[f'parallel_{tag}_yield'] = int(st["yielded"])
            self.timing[f'parallel_{tag}_backend'] = st["backend"]
            self.timing[f'parallel_{tag}_ocr'] = st["ocr"]
            ocr_walls.append(st["ocr_wall"])
        if ocr_walls:
            self.timing['ocr'] = max(ocr_walls)
        self._progress("并行双流水线完成", 100.0)
        return (frames, all_segs, all_texts, all_confs, all_reps)
