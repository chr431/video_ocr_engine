"""FieldExtractor — 通用视频文本提取引擎（识别链：解码∥像素分段∥OCR 文本）。

引擎只输出每段原始文本与置信度；速度解析/纠错/CSV 等领域后处理由上层
应用完成（引擎保持通用性，不携带任何下游领域后处理）。

方法体最初由既有视频项目的历史 tools/archive 生成脚本从 segment_flow.py
抽取；独立成仓后随引擎维护，不再依赖任何下游仓库。

模块划分（2026-08 七轮修正后按逻辑拆分）：
  extractor.py      — 引擎骨架：构造/参数校验/解码器打开/流水线分发/结果组装
  _host_pipeline.py — 宿主流水线：校准/帧流/分段状态机/OCR 会话
                      （_host_calibrate / _host_frame_stream /
                      _host_segment_frames / _HostPipelineMixin）
  _helpers.py       — 无类依赖的独立工具函数
  _result_types.py  — ExtractedSegment / ExtractionResult
  _gpu_pipeline.py  — _GpuPipelineMixin（GPU 全驻留管线）
双流水线并行已被移除（2026-08 清理）；CPU+NVDEC 双解码（decode_backend=
"hybrid"，与 auto/cpu/nvdec 并列）由顶层 hybrid_decode.py 承担。
"""
import logging
import os as _os
import threading
import time
from pathlib import Path

import numpy as np

import engine_config as config
from segmentation import (
    _gray_seg, _gray_seg_batch,
    _gray_seg_yuv, _gray_seg_yuv_batch,
)
from video_utils import _text_sep_gray
# 下列 re-export 为引擎内部结构（_helpers/_result_types/_host_pipeline 均
# 属下划线私有命名，从 extractor 再导出仅为旧导入路径兼容，勿直接 import；
# 公共入口是 video_ocr_engine.__init__ 的三件套）。
from ._result_types import (  # noqa: F401
    ExtractedSegment, ExtractionResult,
)
from ._helpers import (  # noqa: F401
    _ocr_batch_size, _ndarray_device_ptr, _otsu_from_hist, _gray_mean_abs_diff,
    _decode_progress_pct, _ocr_progress_pct,
    _otsu_median_threshold, _read_fps_from_vr,
)
from ._host_pipeline import (  # noqa: F401
    _host_calibrate, _host_frame_stream, _host_segment_frames,
    _HostPipelineMixin,
)
from ._gpu_pipeline import _GpuPipelineMixin

logger = logging.getLogger(__name__)


class FieldExtractor(_GpuPipelineMixin, _HostPipelineMixin):
    """从视频固定区域提取文本的通用引擎（识别链：解码∥分段∥OCR）。

    构造参数：
      常用 —— video_path / roi / frame_start / frame_end / force_aspect /
      decode_backend(auto|cpu|nvdec|hybrid) / ocr_backend(auto|cpu|tensorrt) /
      sample_stride / rep_crop_format(yuv|gray) / keep_crops / keep_frames /
      merge_similar / merge_text_sep / progress_cb / cancel_check。
      高级（默认即最优，改动前读 docs/PERFORMANCE.md）—— buffer_size /
      fill_width / C（分段聚类阈值，默认取 engine_config.SEG_C）/
      merge_similar_threshold。
    已废弃别名（勿再使用）：gray_output / yuv_output —— rep_crop_format
    的旧名（yuv_output=True≡"yuv"；gray_output=True≡"gray"）。
    内部链恒为单通道灰度（yuv420 时取 Y 平面、否则 decord gray），不再输出
    RGB 帧；代表帧像素格式由 rep_crop_format 决定（"yuv"=packed NV12 默认 /
    "gray"），外部用 nv12_to_rgb 转 RGB。
    sample_stride：分频采样步长（默认 1 = 逐帧处理）。>1 时只解码/分段每个
    第 N 帧（字幕等慢更新内容可显著降低处理压力；需要 decord fork ≥0.7.12
    的等差步长快速路径，否则退化为逐索引 seek）。
    """

    def __init__(self, video_path: str, roi: tuple, *, frame_start=None,
                 frame_end=None, force_aspect: float = 0.0,
                 decode_backend: str = "auto", ocr_backend: str = "auto",
                 buffer_size: int | None = None, fill_width: int | None = None,
                 C: float | None = None,
                 sample_stride: int = config.DEFAULT_SAMPLE_STRIDE,
                 progress_cb=None, cancel_check=None, gray_output: bool = False,
                 yuv_output: bool = False,
                 rep_crop_format: str | None = None,
                 keep_crops: bool = True,
                 keep_frames: bool = True,
                 merge_similar: bool = config.DEFAULT_MERGE_SIMILAR,
                 merge_similar_threshold: float | None = None,
                 merge_text_sep: str | None = None):
        self._video_path = Path(video_path)
        self._roi = tuple(roi)
        # fps 强制自测：open decoder 后从 get_avg_fps/get_fps 读（truth 头的
        # fps 可能与视频实际帧率偏离；自测无额外解码开销，只在打开时读一次）。
        self._fps = None
        self._frame_start = frame_start or 0
        self._frame_end = frame_end
        self._force_aspect = force_aspect
        self._decode_backend = decode_backend
        self._ocr_backend = ocr_backend
        self._ocr_model = config.DEFAULT_OCR_MODEL
        self._ocr_backend_used = ""    # run 后填实际引擎（供 CSV 头输出）
        self._buffer_size = (buffer_size if buffer_size is not None
                             else config.DEFAULT_BUFFER_SIZE)
        self._fill_width = (fill_width if fill_width is not None
                            else config.DEFAULT_FILL_WIDTH)
        self._C = (C if C is not None else config.SEG_C)  # 分段聚类阈值
        self._sample_stride = max(1, int(sample_stride))
        self._keep_crops = bool(keep_crops)
        # rep_crop_format：代表帧 keep_crops 的像素格式。内部链恒为单通道
        # 灰度（不产 RGB 帧——RGB→灰度在解码侧/fork 内完成）：
        #   "yuv"  —— packed NV12（默认；内部只取 Y 平面，外部 nv12_to_rgb 转 RGB）
        #   "gray" —— 灰度
        # 旧参数 gray_output / yuv_output 保留为 deprecated 别名：
        #   yuv_output=True  → "yuv"；gray_output=True → "gray"；
        #   两者均 False（旧默认 = RGB）→ 新默认 "yuv"（行为变更，见 README）。
        fmt = (rep_crop_format or '').strip().lower()
        if not fmt:
            fmt = 'yuv' if yuv_output else ('gray' if gray_output else
                                            config.DEFAULT_REP_CROP_FORMAT)
        if fmt not in ('yuv', 'gray'):
            raise ValueError(
                f"rep_crop_format 必须为 'yuv' 或 'gray'，收到 {fmt!r}")
        self._rep_crop_format = fmt
        # 实际 decord 输出格式：keep_crops=False 时无 UV 平面需求 → 直接 gray
        #（省 0.5B/px 解码转换/传输）；yuv420 仅在 keep YUV 时启用。
        self._yuv_output = bool(self._keep_crops and fmt == 'yuv')
        self._gray_output = not self._yuv_output   # deprecated 兼容别名
        self._keep_frames = bool(keep_frames)
        self._merge_similar = bool(merge_similar)
        self._merge_similar_threshold = (
            float(merge_similar_threshold)
            if merge_similar_threshold is not None
            else float(config.SEG_MERGE_SIMILAR_THRESHOLD))
        self._merge_text_sep = (
            merge_text_sep if merge_text_sep is not None
            else config.DEFAULT_MERGE_TEXT_SEP)
        self._color_range = 0            # run 时从 decoder get_color_range 读取
        self._codec = ""                 # run 时从 decoder get_codec 探测
        self._backend = ""
        self._bin_thresh = 0
        self._progress = progress_cb or (lambda m, p: None)
        self._cancel = cancel_check or (lambda: None)
        self.timing: dict = {}
        self.crops: dict = {}
        self._frames: list = []
        self._ocr_texts: list = []
        self._ocr_confs: list = []
        self._n_segments = 0
        self._profile_enabled = config.env_bool(config.ENGINE_PROFILE_ENV)
        self.profile: dict = {}
        self._prof_lock = None
        if self._profile_enabled:
            self._prof_lock = threading.Lock()
        self._validate_params()
        roi_w = max(1, self._roi[2] - self._roi[0] + 1)
        roi_h = max(1, self._roi[3] - self._roi[1] + 1)
        self._merge_max_changed_pixels = max(
            32, int(roi_w * roi_h * config.SEG_MERGE_MAX_CHANGED_RATIO))
        # 后处理参数由子类（SegmentPipeline）在构造时设置；引擎识别链不读。

    def _validate_params(self) -> None:
        """构造期静态参数校验（帧范围相对视频总长在打开解码器后校验）。"""
        if len(self._roi) != 4:
            raise ValueError(
                f"roi 必须为 (x1, y1, x2, y2) 四元组，收到 {len(self._roi)} 个元素")
        x1, y1, x2, y2 = self._roi
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            raise ValueError(f"roi 坐标不能为负: {self._roi}")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"roi 必须满足 x2 > x1 且 y2 > y1: {self._roi}")
        if self._frame_start < 0:
            raise ValueError(f"frame_start 不能为负: {self._frame_start}")
        if (self._frame_end is not None and self._frame_end != 0
                and self._frame_end <= self._frame_start):
            raise ValueError(
                f"frame_end 必须大于 frame_start（或为 0/None 表示到末尾）: "
                f"start={self._frame_start}, end={self._frame_end}")

    def _merge_effective_mode(self) -> str:
        """merge_similar 使用的分离模式（env 钩子优先级与 _segments_similar
        一致）：'contrast' | 'binary' | ''（原始灰度比较）。"""
        _m = _os.environ.get(
            config.TEXT_SEP_MERGE_ENV, self._merge_text_sep or ''
        ).strip().lower()
        if _m in ('1', 'contrast'):
            return 'contrast'
        if _m in ('2', 'binary'):
            return 'binary'
        return ''

    def _segments_similar(self, a, b) -> bool:
        """相似段判定：平均绝对差小 且 显著变化像素占比也小。

        只用平均绝对差会把宽 ROI 中的单字短字幕（如“在”“不”）误判为噪声：
        大部分区域未变，均值被稀释。因此额外限制 abs(diff)>10 的像素数。
        分离模式由 _merge_effective_mode 决定（binary 为引擎默认）。
        """
        _text_mode = self._merge_effective_mode()
        if _text_mode == 'contrast':
            a = _text_sep_gray(a, 'contrast', th=self._bin_thresh)
            b = _text_sep_gray(b, 'contrast', th=self._bin_thresh)
        elif _text_mode == 'binary':
            a = _text_sep_gray(a, 'binary', th=self._bin_thresh)
            b = _text_sep_gray(b, 'binary', th=self._bin_thresh)
        if a is None or b is None or a.shape != b.shape:
            return False
        # int16 精确差：避免 float32 双全帧临时数组（a/b 为 uint8 灰度或
        # float32 分离图，255.0/0.0 转 int16 精确）。与 GPU sim_pair 的
        # 整数精确累加一致（阈值处仅 float32 末位舍入差异，文档已承认）。
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        if float(diff.mean()) > self._merge_similar_threshold:
            return False
        changed = int(np.sum(diff > 10))
        return changed <= self._merge_max_changed_pixels

    def extract(self):
        """通用文本提取：解码∥分段∥OCR → 结构化结果（每段原始文本+置信度）。

        引擎的正式通用入口（无任何领域语义）。返回 ExtractionResult：
          - segments: list[ExtractedSegment]（start/end/rep_frame/text/confidence/
            rep_crop）
          - frames / fps / timing / meta
        识别层不解析文本含义（速度/数值由上层应用处理）。fps 强制自测。
        """
        frames, segs, texts, confs, rep_frames = self._run_pipelined()
        self._frames = frames
        segments = [
            ExtractedSegment(
                start=seg[0], end=seg[-1],
                frames=tuple(seg) if self._keep_frames else (),
                rep_frame=rep_frames[i],
                text=texts[i] if i < len(texts) else None,
                confidence=confs[i] if i < len(confs) else 0.0,
                rep_crop=(self.crops.get(rep_frames[i])
                          if self._keep_crops else None))
            for i, seg in enumerate(segs)
        ]
        return ExtractionResult(
            segments=segments,
            frames=frames if self._keep_frames else [],
            fps=self._fps or 0.0,
            timing=dict(self.timing),
            meta={"backend": self._backend,
                  "ocr_backend": self._ocr_backend_used,
                  "codec": self._codec,
                  "n_segments": len(segments)})

    @property
    def frames(self) -> list:
        """全部采样帧号（run 后有效）。"""
        return self._frames

    @frames.setter
    def frames(self, v: list) -> None:
        self._frames = v

    def _prof_end(self, group: str, key: str, t0: float) -> None:
        """累加一段耗时到 profile（线程安全；关闭时仅一次属性判断）。"""
        if not self._profile_enabled:
            return
        elapsed = time.perf_counter() - t0
        with self._prof_lock:
            d = self.profile.setdefault(group, {})
            d[key] = d.get(key, 0.0) + elapsed

    def _open_vr(self):
        """按 decode_backend 打开解码器（auto/cpu/nvdec/hybrid）。

            auto: 尝试 GPU (NVDEC) 失败回退 CPU。cpu: 强制 CPU。
            nvdec: 强制 GPU（失败回退 CPU 并警告）。
            旧 DECORD_FORCE_CPU env 在 backend=auto 时仍作为兼容钩子生效
            （decode_backend 参数是现行首选，下游仍设钩子的场景保底）。
            hybrid: CPU+NVDEC 混合解码（HybridDecoder，kfe 分片双生产者竞争）；
                NVDEC 不可用时回退 CPU 并警告；激活安全门见下方条件。

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
        if (backend == 'auto'
                and config.env_bool(config.DECORD_FORCE_CPU_ENV)):
            backend = 'cpu'
        vr = None
        label = 'CPU'
        if backend in ('auto', 'nvdec', 'hybrid'):
            try:
                from decord import gpu as _g
                vr = self._open_decord_reader(_g(0), roi_kw)
                label = 'GPU'
            except Exception:
                vr = None
                if backend in ('nvdec', 'hybrid'):
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
        # CPU+NVDEC 混合解码（decode_backend="hybrid" 显式选择，与 auto/cpu/nvdec
        # 并列）：速率比例分界 + 两端连续扫掠（HybridDecoder v3）。
        # 安全门仅保留架构/接口限制：stride==1（next_roi 顺序交付语义）、
        # 未开 GPU 全驻留管线（互斥）。编码（含 AV1）不再回退——v3 的
        # 速率比例分界已实测：CPU 慢于 NVDEC 的 HEVC/AV1 场景与纯 NVDEC
        # 持平不退化，CPU 快于 NVDEC 的 h264 场景显著更快；尊重用户显式
        # 选择。NVDEC 不可用时上面已回退 CPU 并警告；初始化失败回退纯
        # GPU 不致命。
        if (backend == 'hybrid' and label == 'GPU'
                and self._sample_stride == 1
                and not self._gpu_pipeline_enabled()):
            try:
                from hybrid_decode import HybridDecoder
                _mc = int(_os.environ.get(
                    config.HYBRID_MAX_CHUNKS_ENV, '16') or 16)
                _ct = int(_os.environ.get(
                    config.HYBRID_CPU_THREADS_ENV, '0') or 0)
                _mcf = int(_os.environ.get(
                    config.HYBRID_MAX_CHUNK_FRAMES_ENV, '0') or 0)
                vr = HybridDecoder(self, vr, max_chunks=_mc,
                                   cpu_threads=_ct, max_chunk_frames=_mcf)
                self._backend = 'decord/GPU+CPU-hybrid'
                logger.info('混合解码开启(速率分界): codec=%s chunks<=%d cpuT=%d mcf=%d',
                            self._codec, _mc, _ct, _mcf)
            except Exception as e:  # noqa: BLE001
                logger.warning('混合解码初始化失败，回退纯 GPU: %s', e)
        return vr

    def _decord_format(self) -> str:
        """当前管线请求的 decord output_format。

        内部链永远只消费单通道（Y 平面 / decord gray，不再输出 RGB）：
        - keep_crops 需要 YUV 代表帧 → 'yuv420'（packed NV12；内部取 Y 平面，
          等价灰度，另保留 UV 供外部 nv12_to_rgb）
        - 否则 'gray'
        """
        return 'yuv420' if self._yuv_output else 'gray'

    def _decode_num_threads(self, codec: str | None=None) -> int | None:
        """CPU 软解的 decord FFmpeg 帧线程数（少核/AV1 分核）。

            物理核 ≤ CPU_CORES_SPLIT_THRESHOLD（8）时返回 max(2, cores//2)：
            FFmpeg fork 默认 2 帧线程只用 2 核，少核下解码成瓶颈，且 OCR 全核
            会与解码过订阅；实测（test5，affinity 模拟）4 核 28.0 vs 33.1s、
            8 核 17.8 vs 20.7s。核数多时（16）分核反而更差（12.0 vs 9.5s）
            → 返回 None（decord 默认，FFmpeg 帧线程落在 SMT 份额上）。
            codec='av1'：AV1 软解吞吐极低（~270fps vs h264 ~1247fps），解码
            是绝对瓶颈 → 同样返回 max(2, cores//2)，OCR 由 _ocr_num_threads
            保至少 2 线程。
            注：旧实现曾对 AV1 返回 max(2, min(cores*3//4, cores-2))，
            旧 docstring 的实测数字（16 核 dcd=12/ocrT=4 → 78.8s、8 核
            dcd=6/ocrT=2 → 81.7s）属于该旧公式；现实现已统一收敛为
            cores//2，勿按旧数值调参。
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

    def _batch_luma_out(self, crops: np.ndarray,
                        out: np.ndarray) -> np.ndarray:
        """批量灰度写入预分配 out（省每批临时数组分配；形状恒定才可复用）。"""
        if self._yuv_output:
            from segmentation import _nv12_batch_luma_full_out
            return _nv12_batch_luma_full_out(crops, self._color_range, out)
        from segmentation import _gray_batch_out
        return _gray_batch_out(crops, out)

    def _crop_is_expected(self, c: np.ndarray, roi_h: int, roi_w: int) -> bool:
        """ROI-first 输出尺寸是否符合当前输出格式（旧路径全帧则 False）。"""
        if self._yuv_output:
            return c.ndim == 2 and c.shape[0] == roi_h + (roi_h + 1) // 2 and (c.shape[1] == roi_w)
        return c.shape[0] == roi_h and c.shape[1] == roi_w

    def _ocr_engine_type(self) -> str:
        """OCR 推理后端：auto/tensorrt → tensorrt（OcrEngine 失败回退 onnx），cpu → onnxruntime。"""
        return 'onnxruntime' if (self._ocr_backend or 'auto').lower() == 'cpu' else 'tensorrt'

    def _ocr_num_threads(self) -> int:
        """OCR 推理线程预算：OCR_THREADS env 钩子优先，否则全物理核；
            CPU 软解且物理核 ≤ 8 时与解码显式分核（cores//2，防过订阅）。

            解码（NVDEC 全卸载 / CPU 下 FFmpeg 帧线程 2 + filter auto 只占
            SMT 份额）不抢物理核，OCR 吃满全部物理核；CPU 软解在少核机上
            FFmpeg 帧线程与 OCR 争抢（实测 4 核 ocrT=2 28.0s vs 全核 33.1s、
            8 核 ocrT=4 17.8s vs 20.7s），分核更优；核数多时（16）分核反而
            差 → 保持全核。显式参数传入引擎，不污染全局 env。
            """
        from ocr_native import auto_ocr_thread_count
        _env = _os.environ.get(config.OCR_THREADS_ENV)
        if _env:
            return max(1, int(_env))
        cores = auto_ocr_thread_count()
        if getattr(self, '_codec', '') == 'av1' and getattr(self, '_backend', '').startswith('decord/CPU'):
            return max(2, cores // 2)
        if getattr(self, '_backend', '').startswith('decord/CPU') and cores <= config.CPU_CORES_SPLIT_THRESHOLD:
            return max(2, cores // 2)
        return cores

    def _run_pipelined(self, _ocr_engines: list | None = None):
        """入口分发：GPU 全驻留管线（_run_pipelined_gpu）或宿主管线。"""
        if self._gpu_pipeline_enabled():
            return self._run_pipelined_gpu()
        return self._run_pipelined_host(_ocr_engines)

    def _run_pipelined_host(self, _ocr_engines: list | None = None):
        """宿主流水线：解码线程增量分段，OCR 线程批处理已闭合段的代表帧。
        _ocr_engines：内部复用 OCR 引擎时传入；None 走常规创建。

            解码是 I/O 瓶颈（CPU 占用低），段边界（win3）在解码循环内增量计算，
            段一闭合就把代表帧（最清晰）交给 OCR 工作线程 —— 解码∥OCR 重叠摊薄
            总墙钟。代表帧选择为段内灰度 std 最大帧，OCR 批 _ocr_batch_size()。

            返回 (frames, segs, ocr_texts, ocr_confs, rep_frames)；
            self.crops = {rep_frame: crop}（仅代表帧，供 review 预览，
            比存全帧省内存）。分段/代表帧选择语义由模块级共享状态机
            _host_segment_frames 承担。
            """
        self._gpu_pipeline_mode = False
        _t_open = time.perf_counter()
        vr = self._open_vr()
        if self._fps is None:
            _fps = _read_fps_from_vr(vr)
            self._fps = _fps if _fps else config.DEFAULT_FPS_FALLBACK
        total = len(vr)
        end = min(self._frame_end or total, total)
        if self._frame_start > 0:
            vr.seek_accurate(self._frame_start)
        frames = list(range(self._frame_start, end, self._sample_stride))
        if not frames:
            raise ValueError(
                f"帧区间为空: frame_start={self._frame_start}, "
                f"frame_end={end}, total={total}")
        # 混合解码（hybrid_begin）：采样帧序列就绪后才生成关键帧分片并
        # 启动双解码生产者竞争。
        hybrid = hasattr(vr, 'hybrid_begin')
        if hybrid:
            vr.hybrid_begin(frames)
        self._prof_end('producer', 'open_and_fps', _t_open)
        # OCR 会话（引擎初始化/模型加载）提前到校准前启动：worker 线程内
        # 构建引擎，与校准（_host_calibrate，前 50 帧解码+Otsu）并行重叠，
        # 引擎就绪前 _emit_ocr 自动走 host 回退，语义不变。
        ocr_session = self._start_ocr_session(_ocr_engines)
        q = ocr_session["q"]
        results = ocr_session["results"]
        ocr_err = ocr_session["err"]
        ocr_wall = ocr_session["wall"]
        _put_ocr = ocr_session["put"]
        _t_cal = time.perf_counter()
        # 宿主校准统一走 _host_calibrate（stride>1 用 get_batch 等差快速路径、
        # stride==1 用 next_roi 顺序流——校准帧号与后续流水线帧号一致）。
        # with_dev=True：保留 GPU 单通道帧的 DLPack 指针供 GPU raw OCR 直通。
        # 混合解码（HybridDecoder）交付的是 asnumpy() 宿主数组（_Batch 无
        # to_dlpack），不存在可为 raw OCR 直通的 GPU 指针 —— 4D 单通道
        # gray 时强采 _ndarray_device_ptr 会 AttributeError 崩溃，必须跳过。
        _with_dev = not hybrid
        calib, th = _host_calibrate(self, vr, frames, with_dev=_with_dev)
        self._bin_thresh = th
        self._prof_end('producer', 'calib_total', _t_cal)

        segs: list = []
        rep_crops: dict = {}
        seg_idx = 0

        def _emit_ocr(seg, r_frame, r_crop, r_dev, _r_gray, frac) -> None:
            nonlocal seg_idx
            _t_push = time.perf_counter()
            _put_ocr((seg_idx, r_frame, r_crop, r_dev, frac))
            self._prof_end('producer', 'q_put_block', _t_push)
            if self._keep_crops:
                rep_crops[r_frame] = r_crop
            seg_idx += 1

        t0 = time.perf_counter()
        try:
            _host_segment_frames(
                self, frames,
                _host_frame_stream(self, frames, vr, calib, th,
                                   with_dev=_with_dev),
                debug_tag='HB',
                progress_prefix=f'[{self._backend}] 解码+分段',
                emit=_emit_ocr, segs=segs)
        finally:
            _t_consume_end = time.perf_counter()
            self.timing['decode'] = _t_consume_end - t0
            self._prof_end('producer', 'consumer_total', t0)
            ocr_session["finish"]()
            self.timing['ocr_tail'] = time.perf_counter() - _t_consume_end
            try:
                vr.close()   # hybrid 探针/资源释放：显式停止生产者线程
            except Exception:
                pass
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
