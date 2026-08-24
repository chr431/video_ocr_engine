"""FieldExtractor — 通用视频文本提取引擎（识别链：解码∥像素分段∥OCR 文本）。

引擎只输出每段原始文本与置信度；速度解析/纠错/CSV 等领域后处理由上层
应用完成（RaceVideoToLog 的 SegmentPipeline 继承本类并叠加后处理）。

方法体最初由 RaceVideoToLog 的 tools/archive/_gen_engine_extractor.py 从
segment_flow.py 抽取；独立成仓后随引擎维护，不再依赖 RaceVideoToLog。
"""
import csv
import logging
import os as _os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

import engine_config as config  # 识别链只用引擎域常量
from segmentation import (
    _cluster_win3, _gray, _gray_batch, _gray_seg,
    _gray_seg_batch, _gray_seg_yuv, _gray_seg_yuv_batch, _otsu,
)
from ocr_native import OcrEngine, auto_ocr_thread_count
from video_utils import (_nv12_luma_full, _preprocess_standard,
                         _text_sep_gray,
                         nv12_to_rgb, nvdec_available,
                         tensorrt_available)  # 识别链 YUV/preprocess/RGB 预览

logger = logging.getLogger(__name__)


def _ocr_batch_size() -> int:
    _env = _os.environ.get("OCR_BATCH")
    if _env and _env.isdigit():
        return max(1, int(_env))
    return config.OCR_BATCH_SIZE


def _ndarray_device_ptr(nd):
    """从 decord GPU NDArray DLPack 解析 device 数据基址。

    返回 (base_ptr:int, shape:tuple[int,...])。调用方必须保持 nd 存活。
    """
    import ctypes
    cap = nd.to_dlpack()
    _get = ctypes.pythonapi.PyCapsule_GetPointer
    _get.restype = ctypes.c_void_p
    _get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = _get(cap, b"dltensor")

    class _DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int32),
                    ("device_id", ctypes.c_int32)]

    class _DLDataType(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                    ("lanes", ctypes.c_uint16)]

    class _DLTensor(ctypes.Structure):
        _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice),
                    ("ndim", ctypes.c_int32), ("dtype", _DLDataType),
                    ("shape", ctypes.POINTER(ctypes.c_int64)),
                    ("strides", ctypes.POINTER(ctypes.c_int64)),
                    ("byte_offset", ctypes.c_uint64)]

    t = ctypes.cast(ptr, ctypes.POINTER(_DLTensor)).contents
    shape = tuple(int(t.shape[i]) for i in range(t.ndim))
    return int(t.data), shape


def _otsu_from_hist(hist) -> int:
    """从 256-bin 直方图算 Otsu 阈值（与 segmentation._otsu 等价）。"""
    hist = np.asarray(hist, dtype=np.int64)
    total = int(hist.sum())
    if total <= 0:
        return config.OTSU_FALLBACK_THRESH
    st = float((np.arange(256) * hist).sum())
    sb = 0.0
    wb = 0
    best = config.OTSU_FALLBACK_THRESH
    vmax = -1.0
    for t in range(256):
        wb += int(hist[t])
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += t * int(hist[t])
        mb = sb / wb
        mf = (st - sb) / wf
        vb = wb * wf * (mb - mf) ** 2
        if vb > vmax:
            vmax = vb
            best = t
    return best


def _gray_mean_abs_diff(a, b) -> float:
    """两帧分段灰度 ROI 的平均绝对差；形状不一致时视为不相似。"""
    if a is None or b is None:
        return float("inf")
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


@dataclass
class ExtractedSegment:
    """引擎输出的单个文本字段段（原有字段区间 + 代表帧 + 原始文本）。"""

    start: int                      # 段首帧号
    end: int                        # 段末帧号
    frames: tuple = ()              # 段内帧号序列
    rep_frame: int = -1             # 代表帧号（段内最清晰帧）
    text: Optional[str] = None      # OCR 原始文本（None=未读出）
    confidence: float = 0.0         # OCR 置信度 0-1
    rep_crop: Any = None            # 代表帧 ROI 图像（YUV420 或 RGB）


@dataclass
class ExtractionResult:
    """引擎通用提取结果（无领域语义）。"""

    segments: list = field(default_factory=list)  # list[ExtractedSegment]
    frames: list = field(default_factory=list)     # 全部采样帧号
    fps: float = 0.0                               # 自测帧率
    timing: dict = field(default_factory=dict)     # 各阶段耗时
    meta: dict = field(default_factory=dict)       # backend/codec/引擎版本等


class FieldExtractor:
    """从视频固定区域提取文本的通用引擎（识别链：解码∥分段∥OCR）。

    构造参数（识别链用）：video_path / roi / frame_start / frame_end /
    force_aspect / decode_backend / ocr_backend / buffer_size / fill_width /
    sample_stride / progress_cb / cancel_check / gray_output / yuv_output /
    keep_crops / keep_frames / merge_similar / merge_similar_threshold。
    分段阈值 C 取自 engine_config.SEG_C；引擎不含速度后处理参数。
    sample_stride：分频采样步长（默认 1 = 逐帧处理，与 RaceVideoToLog 完全
    兼容）。>1 时只解码/分段每个第 N 帧（字幕等慢更新内容可显著降低处理压力；
    需要 decord fork ≥0.7.12 的等差步长快速路径，否则退化为逐索引 seek）。
    """

    def __init__(self, video_path: str, roi: tuple, *, frame_start=None,
                 frame_end=None, force_aspect: float = 0.0,
                 decode_backend: str = "auto", ocr_backend: str = "auto",
                 buffer_size: int | None = None, fill_width: int | None = None,
                 C: float | None = None, fps: float | None = None,
                 sample_stride: int = config.DEFAULT_SAMPLE_STRIDE,
                 progress_cb=None, cancel_check=None, gray_output: bool = False,
                 yuv_output: bool = False, keep_crops: bool = True,
                 keep_frames: bool = True,
                 merge_similar: bool = config.DEFAULT_MERGE_SIMILAR,
                 merge_similar_threshold: float | None = None,
                 merge_text_sep: str | None = None,
                 dual_pipeline: bool | None = None,
                 dual_pipeline_chunks: int = 0,
                 dual_backends: list | None = None):
        self._video_path = Path(video_path)
        self._roi = tuple(roi)
        # fps 强制自测：open decoder 后从 get_avg_fps/get_fps 读，忽略外部
        # 传入（truth 头的 fps 可能与视频实际帧率偏离；自测无额外解码开销，
        # 只在打开时读一次元数据）。fps 参数保留仅为 API 兼容（已废弃）。
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
        self._gray_output = gray_output
        self._yuv_output = yuv_output
        self._keep_crops = bool(keep_crops)
        self._keep_frames = bool(keep_frames)
        self._merge_similar = bool(merge_similar)
        self._merge_similar_threshold = (
            float(merge_similar_threshold)
            if merge_similar_threshold is not None
            else float(config.SEG_MERGE_SIMILAR_THRESHOLD))
        self._merge_text_sep = (
            merge_text_sep if merge_text_sep is not None
            else config.DEFAULT_MERGE_TEXT_SEP)
        if dual_pipeline is None:
            _env_dual = _os.environ.get(
                config.DUAL_PIPELINE_ENV, '').strip().lower()
            dual_pipeline = _env_dual in ('1', 'true', 'yes', 'on')
        self._dual_pipeline = bool(dual_pipeline)
        self._dual_pipeline_chunks = max(0, int(dual_pipeline_chunks or 0))
        self._dual_backends = (
            [tuple(map(str, p)) for p in dual_backends]
            if dual_backends else None)
        self._color_range = 0            # run 时从 decoder get_color_range 读取
        self._codec = ""                 # run 时从 decoder get_codec 探测
        self._backend = ""
        self._bin_thresh = 0
        self._progress = progress_cb or (lambda m, p: None)
        self._cancel = cancel_check or (lambda: None)
        self.rows: list = []
        self.timing: dict = {}
        self.segments: list[dict] = []
        self.crops: dict = {}
        self._segs: list = []
        self._frames: list = []
        self._ocr_vals: list = []
        self._ocr_texts: list = []
        self._ocr_confs: list = []
        self._corr_vals: list = []
        self._conf_vals: list = []
        self._pinned: set = set()
        self._n_segments = 0
        self._n_corr = 0
        self._profile_enabled = _os.environ.get(
            "ENGINE_PROFILE", "").strip().lower() in ("1", "true", "yes", "on")
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
            'TEXT_SEP_MERGE', self._merge_text_sep or ''
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
        diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
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
        识别层不解析文本含义（速度/数值由上层应用处理）。ffis 强制自测。
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
        _env = _os.environ.get('OCR_THREADS')
        if _env:
            return max(1, int(_env))
        cores = auto_ocr_thread_count()
        if getattr(self, '_codec', '') == 'av1' and getattr(self, '_backend', '').startswith('decord/CPU'):
            return max(2, cores // 2)
        if getattr(self, '_backend', '').startswith('decord/CPU') and cores <= config.CPU_CORES_SPLIT_THRESHOLD:
            return max(2, cores // 2)
        return cores

    def _decode_all(self):
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
        frames = list(range(self._frame_start, end, self._sample_stride))
        if not frames:
            raise ValueError(
                f"帧区间为空: frame_start={self._frame_start}, "
                f"frame_end={end}, total={total}")
        DECODE_BATCH = config.DECODE_BATCH_SIZE
        crops = {}
        grays = {}
        sharp = {}
        t0 = time.perf_counter()
        for k, fi in enumerate(frames):
            if self._sample_stride > 1:
                # 分频采样（串行参考路径）：单帧 get_batch（等差数列长度为
                # 1 走通用 seek 路径），保证 crops[fi] 对应真实采样帧号。
                c = vr.get_batch([fi], roi=(x1, y1, x2 + 1, y2 + 1)).asnumpy()[0]
            else:
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
        del vr
        return (frames, crops, grays, sharp)

    def _segment(self, frames, grays):
        if not frames:
            raise ValueError("分段帧列表为空")
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

    def _start_ocr_session(self, _ocr_engines: list | None = None) -> dict:
        """启动一个可跨多个切片持续复用的 OCR 会话。

        返回 dict：q（段任务队列）、results（全局段索引 → text/conf/rep）、
        err、wall、put（投递段任务）、finish（哨兵并 join OCR worker）。
        单流水线仍用该会话；双流水线多条切片共用同一会话，避免每片重建
        OCR worker / infer 线程造成屏障。
        """
        from queue import Full, Queue
        import threading
        from ocr_native import OcrEngine
        from video_utils import _preprocess_standard

        q: Queue = Queue(maxsize=max(1, self._buffer_size))
        results: dict = {}
        ocr_err: list = []
        ocr_wall = [0.0]

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
                    dual_onnx = self._ocr_engine_type() == 'onnxruntime' and ot >= 8 and (_os.environ.get('DUAL_ONNX', '1') != '0')
                    if dual_onnx:
                        half = max(2, ot // 2)
                        engines = [OcrEngine(self._ocr_model, 'onnxruntime', fill_width=self._fill_width, num_threads=half, progress_cb=_engine_progress) for _ in range(2)]
                    else:
                        engines = [OcrEngine(self._ocr_model, self._ocr_engine_type(), fill_width=self._fill_width, num_threads=ot, progress_cb=_engine_progress)]
                    self._ocr_backend_used = engines[0].backend_name
                    self._prof_end('ocr', 'engine_init', _t_eng)
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
                        self._progress(f'[OCR] 段 {idx + 1}', 58.0 + frac * 28.0)

                def infer_worker(eng) -> None:
                    try:
                        while True:
                            item = infer_q.get()
                            if item is None:
                                return
                            idxs, reps, procs, fracs, raw_infos = item
                            _t_i = time.perf_counter()
                            if raw_infos is not None:
                                res = eng.call_gpu_raw(raw_infos)
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

                def _store_result(idx, rep, r, frac) -> None:
                    if hasattr(r, 'txts'):
                        raw_text = (str(r.txts[0])
                                    if r.txts and r.txts[0] else None)
                        scores = getattr(r, 'scores', [])
                        ocr_conf = float(scores[0]) if scores else 0.0
                    else:
                        raw_text, ocr_conf = (None, 0.0)
                    results[idx] = (raw_text, ocr_conf, rep)
                    _report_ocr_progress(idx, frac)

                def flush() -> None:
                    if not b_idx:
                        return
                    # GPU 直通：只有单 TRT 引擎且代表帧全部带 GPU 指针时走
                    raw_ok = (
                        len(engines) == 1
                        and getattr(engines[0], '_trt', None) is not None
                        and b_devs and all(d is not None for d in b_devs)
                        and getattr(self, '_gpu_pipeline_mode', False))
                    if raw_ok:
                        # 把 raw 任务交给 infer 线程异步执行，避免 OCR worker
                        # 被 GPU 预处理 + TRT 同步阻塞。
                        infos = [(d[1], d[2], d[3], d[0]) for d in b_devs]
                        if not _put_infer((
                                list(b_idx), list(b_reps), None,
                                list(b_fracs), infos)):
                            return
                        b_idx.clear(); b_reps.clear(); b_crops.clear()
                        b_devs.clear(); b_fracs.clear()
                        return
                    _t_p = time.perf_counter()
                    procs = [_preprocess_standard(
                        _nv12_luma_full(c, self._color_range)[..., None]
                        if self._yuv_output else c,
                        force_aspect=self._force_aspect) for c in b_crops]
                    self._prof_end('ocr', 'preprocess', _t_p)
                    if not _put_infer((list(b_idx), list(b_reps), procs, list(b_fracs), None)):
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
                    if len(b_idx) >= B:
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
        }

    def _run_parallel_chunk(self, worker, vr, session, chunk_idx: int,
                            start: int, end_f: int, n_chunks: int,
                            th: int | None = None,
                            seek_required: bool = True):
        """双流水线 worker 处理单个切片的解码/分段，送入共享 OCR 会话。

        不等待 OCR 完成、不新建 OCR 线程——只把段任务塞进 session["q"]，
        这样下一个切片可以立即开始解码，真正跨片重叠。
        th：全局分段 Otsu 阈值（主线程校准一次后传入；None=片内自行校准，
        仅保留给非并行调用方）。传 th 时跳过每片 50 帧校准解码，且各片
        二值化阈值与单流水线一致（消除跨片阈值漂移）。
        返回 (segs, keys, reps, rep_crops, decode_elapsed,
              first_rep_gray, last_rep_gray)；首/末代表帧灰度供跨片边界
        merge_similar 缝合使用。
        """
        x1, y1, x2, y2 = worker._roi
        total = len(vr)
        end = min(worker._frame_end or total, total)
        frames = list(range(start, end, worker._sample_stride))
        if not frames:
            raise ValueError(
                f"帧区间为空: frame_start={start}, "
                f"frame_end={end}, total={total}")
        calib_n = 0
        calib: list = []
        if th is None:
            # 兼容路径：无全局阈值时按旧逻辑片内校准（前 50 帧 Otsu）。
            # 仅此路径使用顺序 next_roi（stride==1 时依赖解码器当前位置）
            # ——需要先精确定位；全局 th 路径也需要 seek_accurate 到片首，
            # 否则 get_batch 从当前/文件头随机跳到目标帧，CPU 解码吞吐约慢一倍。
            vr.seek_accurate(start)
            calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
            if worker._sample_stride > 1:
                _calib_crops = vr.get_batch(
                    frames[:calib_n], roi=(x1, y1, x2 + 1, y2 + 1)).asnumpy()
                for k in range(calib_n):
                    c = _calib_crops[k]
                    if not worker._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                        c = c[y1:y2 + 1, x1:x2 + 1]
                    g = worker._crop_luma(c)
                    calib.append((frames[k], c, g, float(g.std())))
            else:
                for k in range(calib_n):
                    c = vr.next_roi(x1, y1, x2 + 1, y2 + 1).asnumpy()
                    if not worker._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                        c = c[y1:y2 + 1, x1:x2 + 1]
                    g = worker._crop_luma(c)
                    calib.append((frames[k], c, g, float(g.std())))
            ths = [_otsu(g) for _fi, _c, g, _s in calib]
            th = int(np.median(ths)) if ths else config.OTSU_FALLBACK_THRESH
        worker._bin_thresh = th
        # 生产者净耗时（seek+解码+灰度/sharp/二分/分段 分相累加）：之前 seek
        # 未计入吞吐信号，导致试点测速远高于真实整片速度（AV1 实测真实比 2.2:1
        # 试点却算出 6.4:1）。把 seek 计入后，测速才能反映每片实际成本。
        prod_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # 全局阈值路径仍需精确定位到本片起点：只依赖 get_batch 随机访问会
        # 让 decord 每次从当前/文件头跳到目标帧（实测 CPU 解码吞吐约慢一倍），
        # 而先 seek_accurate 一次后 get_batch 可走连续等差步长快速路径。
        # 单次精确 seek 成本 ~150-300ms，远小于随机访问带来的额外解码开销。
        # 例外：本片起点与上一片终点相邻（同一流水线沿帧序连续扫掠）时解码
        # 器已停在正确位置附近，seek_accurate 实测 ~1ms（vs 乱序跳跃 40-70ms），
        # 直接跳过——分片越密（每关键帧一片）节省越明显。
        # 另：NVDEC 硬解路径跳过显式 seek——get_batch 内部随机定位实测比显式
        # seek+get_batch 更便宜（本机 h264 GPU ~46ms vs ~69ms，且硬解随机访问
        # 不衰减）；CPU 软解仍保留显式 seek（跳过会让随机访问约慢一倍，字幕
        # 宽 ROI 实测 +3.5s）。DUAL_PIPELINE_SEEK=1 强制全部显式；=0 全部跳过。
        _seek_env = _os.environ.get(
            config.DUAL_PIPELINE_SEEK_ENV, '').strip().lower()
        _seek_all_off = _seek_env in ('0', 'false', 'no', 'off')
        _seek_gpu_skip = (
            _seek_env not in ('1', 'true', 'yes', 'on')
            and worker._backend.startswith('decord/GPU'))
        if (th is not None and seek_required
                and not _seek_all_off and not _seek_gpu_skip):
            _t_seek = time.perf_counter()
            vr.seek_accurate(start)
            prod_acc[0] += time.perf_counter() - _t_seek
            worker._prof_end('producer', 'seek_accurate', _t_seek)
        DECODE_BATCH = config.DECODE_BATCH_SIZE

        def frame_stream():
            for fi, c, g, s in calib:
                yield (fi, c, g, s, g > th)
            for bstart in range(calib_n, len(frames), DECODE_BATCH):
                bend = min(bstart + DECODE_BATCH, len(frames))
                _t_d = time.perf_counter()
                crops = vr.get_batch(
                    frames[bstart:bend],
                    roi=(x1, y1, x2 + 1, y2 + 1)).asnumpy()
                prod_acc[1] += time.perf_counter() - _t_d
                worker._prof_end('producer', 'decode_batch', _t_d)
                _t_g = time.perf_counter()
                g = worker._batch_luma(crops)
                prod_acc[2] += time.perf_counter() - _t_g
                worker._prof_end('producer', 'gray_batch', _t_g)
                _t_s = time.perf_counter()
                sharp = g.std(axis=(1, 2))
                prod_acc[3] += time.perf_counter() - _t_s
                worker._prof_end('producer', 'sharp_batch', _t_s)
                _t_b = time.perf_counter()
                bs = g > th
                prod_acc[4] += time.perf_counter() - _t_b
                worker._prof_end('producer', 'bin_batch', _t_b)
                for k, gi in enumerate(range(bstart, bend)):
                    yield (frames[gi], crops[k], g[k], float(sharp[k]), bs[k])

        segs: list = []
        keys: list = []
        reps: list = []
        rep_crops: dict = {}
        seg_idx = int(session.get("seg_idx", 0))
        s = 0
        rep_frame = frames[0]
        rep_crop = None
        rep_sharp = -1.0
        rep_gray = None
        last_rep_gray = None
        prev_b = None
        t0 = time.perf_counter()

        def emit(seg, r_frame, r_crop, r_gray, frac) -> None:
            nonlocal seg_idx
            segs.append(seg)
            keys.append(seg_idx)
            reps.append(r_frame)
            session["put"]((seg_idx, r_frame, r_crop, None, frac))
            if worker._keep_crops:
                rep_crops[r_frame] = r_crop
            if first_emit_gray[0] is None:
                first_emit_gray[0] = r_gray
            seg_idx += 1

        first_emit_gray = [None]  # 首个 emit 段的代表帧灰度（跨片缝合用）

        for k, (fi, c, g, sharp, b) in enumerate(frame_stream()):
            if prev_b is not None:
                d = prev_b != b
                _t_seg = time.perf_counter()
                changed = _cluster_win3(d) >= worker._C
                prod_acc[5] += time.perf_counter() - _t_seg
                worker._prof_end('producer', 'segmentation', _t_seg)
                if changed:
                    seg = frames[s:k]
                    similar = (
                        worker._merge_similar and segs
                        and worker._segments_similar(last_rep_gray, rep_gray))
                    if similar:
                        segs[-1].extend(seg)
                    else:
                        emit(seg, rep_frame, rep_crop, rep_gray,
                             k / max(len(frames), 1))
                        last_rep_gray = rep_gray
                    s = k
                    rep_frame = fi
                    rep_crop = c
                    rep_sharp = sharp
                    rep_gray = g
                elif sharp > rep_sharp:
                    rep_sharp = sharp
                    rep_frame = fi
                    rep_crop = c
                    rep_gray = g
            else:
                rep_frame = fi
                rep_crop = c
                rep_sharp = sharp
                rep_gray = g
            prev_b = b
            if k % 100 == 0:
                worker._cancel()
            if k % 500 == 0:
                worker._progress(
                    f'[{worker._backend}] 并行解码+分段: '
                    f'{k}/{len(frames)}',
                    3 + k / max(len(frames), 1) * 55)
        seg = frames[s:]
        similar = (
            worker._merge_similar and segs
            and worker._segments_similar(last_rep_gray, rep_gray))
        if similar:
            segs[-1].extend(seg)
        else:
            emit(seg, rep_frame, rep_crop, rep_gray, 1.0)
        session["seg_idx"] = seg_idx
        return (segs, keys, reps, rep_crops, time.perf_counter() - t0,
                first_emit_gray[0], last_rep_gray, sum(prod_acc))

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
        env = _os.environ.get('GPU_PIPELINE', '').strip().lower()
        if env in ('0', 'false', 'no', 'off'):
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
        th = int(np.median(ths)) if ths else config.OTSU_FALLBACK_THRESH
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
            DECODE_BATCH = 64  # GPU 分段实验：更大批减少 kernel/同步次数
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
                        if _os.environ.get('DEBUG_BOUNDS'):
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
                                   3 + k / max(len(frames), 1) * 55)
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

    def _run_pipelined(self, _ocr_engines: list | None = None,
                       _force_single: bool = False,
                       _external_vr=None):
        """流水线：解码线程增量分段，OCR 线程批处理已闭合段的代表帧。
        _ocr_engines：内部并行流水线复用 OCR 引擎时传入；None 走常规创建。
        _force_single：内部回退单实例时传入，绕过 dual_pipeline 分发。
        _external_vr：内部并行流水线复用已打开解码器时传入（不重新 open）。

            解码是 I/O 瓶颈（CPU 占用低），段边界（win3）在解码循环内增量计算，
            段一闭合就把代表帧（最清晰）交给 OCR 工作线程 —— 解码∥OCR 重叠摊薄
            总墙钟。代表帧选择与串行 _segment/_ocr_segments 完全一致（每段 max
            灰度 std），OCR 批 _ocr_batch_size()。

            返回 (frames, segs, seg_vals, rep_frames)；self.crops = {rep_frame:
            crop}（仅代表帧，供 review 预览，比存全帧省内存）。
            """
        if self._dual_pipeline and _ocr_engines is None and not _force_single:
            return self._run_pipelined_parallel()
        if self._gpu_pipeline_enabled():
            return self._run_pipelined_gpu()
        _t_open = time.perf_counter()
        if _external_vr is not None:
            vr = _external_vr
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
        frames = list(range(self._frame_start, end, self._sample_stride))
        if not frames:
            raise ValueError(
                f"帧区间为空: frame_start={self._frame_start}, "
                f"frame_end={end}, total={total}")
        self._prof_end('producer', 'open_and_fps', _t_open)
        calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
        calib: list = []
        _t_cal = time.perf_counter()
        if self._sample_stride > 1:
            # 分频采样：校准帧也按 stride 抽取（真实帧号 = frames[k]）——用
            # get_batch 的等差步长快速路径（decord fork ≥0.7.12）顺序流式取，
            # 校准帧与后续流水线帧号一致，避免 next_roi（逐帧）与采样不匹配。
            _calib_nds = vr.get_batch(frames[:calib_n],
                                      roi=(x1, y1, x2 + 1, y2 + 1))
            _calib_crops = _calib_nds.asnumpy()
            _calib_base, _calib_shape = _ndarray_device_ptr(_calib_nds)
            _calib_c = _calib_shape[-1] if _calib_shape else 1
            for k in range(calib_n):
                c = _calib_crops[k]
                if not self._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                    c = c[y1:y2 + 1, x1:x2 + 1]
                g = self._crop_luma(c)
                dev_info = None
                if _calib_c == 1 and len(_calib_shape) == 4:
                    src_h, src_w = _calib_shape[1], _calib_shape[2]
                    dev_info = (_calib_nds,
                                _calib_base + k * src_h * src_w,
                                src_h, src_w)
                calib.append((frames[k], c, g, float(g.std()), dev_info))
        else:
            for k in range(calib_n):
                _t_p = time.perf_counter()
                _nd = vr.next_roi(x1, y1, x2 + 1, y2 + 1)
                c = _nd.asnumpy()
                self._prof_end('producer', 'calib_decode', _t_p)
                if not self._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                    c = c[y1:y2 + 1, x1:x2 + 1]
                _t_p = time.perf_counter()
                g = self._crop_luma(c)
                self._prof_end('producer', 'calib_gray', _t_p)
                dev_info = None
                if _nd.shape[-1] == 1 and len(_nd.shape) == 3:
                    _base, _shape = _ndarray_device_ptr(_nd)
                    dev_info = (_nd, _base, _shape[0], _shape[1])
                calib.append((frames[k], c, g, float(g.std()), dev_info))
        ths = [_otsu(g) for _fi, _c, g, _s, _dev in calib]
        th = int(np.median(ths)) if ths else config.OTSU_FALLBACK_THRESH
        self._bin_thresh = th
        self._prof_end('producer', 'calib_total', _t_cal)
        ocr_session = self._start_ocr_session(_ocr_engines)
        q = ocr_session["q"]
        results = ocr_session["results"]
        ocr_err = ocr_session["err"]
        ocr_wall = ocr_session["wall"]
        _put_ocr = ocr_session["put"]

        DECODE_BATCH = config.DECODE_BATCH_SIZE

        def frame_stream():
            """先产出校准帧，再批量流式解码剩余帧。

                yield (fi, crop, gray, sharp, bin, dev_info) —— bin 为
                预计算的二值化；dev_info 仅在 gray 输出时保留 GPU 指针。
                """
            for fi, c, g, s, dev in calib:
                yield (fi, c, g, s, g > th, dev)
            for bstart in range(calib_n, len(frames), DECODE_BATCH):
                bend = min(bstart + DECODE_BATCH, len(frames))
                _t_d = time.perf_counter()
                nds = vr.get_batch(
                    frames[bstart:bend],
                    roi=(x1, y1, x2 + 1, y2 + 1))
                crops = nds.asnumpy()
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
                dev_info = None
                if nds.shape[-1] == 1 and len(nds.shape) == 4:
                    _base, _shape = _ndarray_device_ptr(nds)
                    src_h, src_w = _shape[1], _shape[2]
                else:
                    _base = 0
                    src_h = src_w = 0
                for k, gi in enumerate(range(bstart, bend)):
                    d = None
                    if _base:
                        d = (nds, _base + k * src_h * src_w,
                             src_h, src_w)
                    yield (frames[gi], crops[k], g[k],
                           float(sharp[k]), bs[k], d)
        segs: list = []
        rep_crops: dict = {}
        seg_idx = 0
        s = 0
        rep_frame = frames[0]
        rep_crop = None
        rep_dev = None
        rep_sharp = -1.0
        rep_gray = None
        last_rep_gray = None
        prev_b = None
        t0 = time.perf_counter()
        try:
            for k, (fi, c, g, sharp, b, dev_info) in enumerate(frame_stream()):
                if prev_b is not None:
                    d = prev_b != b
                    _t_seg = time.perf_counter()
                    changed = _cluster_win3(d) >= self._C
                    self._prof_end('producer', 'segmentation', _t_seg)
                    if changed:
                        seg = frames[s:k]
                        if _os.environ.get('DEBUG_BOUNDS'):
                            print(f'[HB]{fi}:{_cluster_win3(d):.0f}',
                                  flush=True)
                        similar = (
                            self._merge_similar and segs
                            and self._segments_similar(last_rep_gray, rep_gray))
                        if similar:
                            # 同一视觉内容被噪声切成多段：并入前一段，
                            # 不产生新的 OCR 任务，保留前一段代表帧/文本。
                            segs[-1].extend(seg)
                        else:
                            segs.append(seg)
                            _t_push = time.perf_counter()
                            _put_ocr((seg_idx, rep_frame, rep_crop,
                                      rep_dev, k / max(len(frames), 1)))
                            self._prof_end('producer', 'q_put_block', _t_push)
                            if self._keep_crops:
                                rep_crops[rep_frame] = rep_crop
                            seg_idx += 1
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
                    self._cancel()
                if k % 500 == 0:
                    self._progress(f'[{self._backend}] 解码+分段: {k}/{len(frames)}', 3 + k / max(len(frames), 1) * 55)
            seg = frames[s:]
            similar = (
                self._merge_similar and segs
                and self._segments_similar(last_rep_gray, rep_gray))
            if similar:
                segs[-1].extend(seg)
            else:
                segs.append(seg)
                _t_push = time.perf_counter()
                _put_ocr((seg_idx, rep_frame, rep_crop, rep_dev, 1.0))
                self._prof_end('producer', 'q_put_block', _t_push)
                if self._keep_crops:
                    rep_crops[rep_frame] = rep_crop
                seg_idx += 1
                last_rep_gray = rep_gray
        finally:
            _t_consume_end = time.perf_counter()
            self.timing['decode'] = _t_consume_end - t0
            self._prof_end('producer', 'consumer_total', t0)
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

    # ═══════════════ 单实例双完整流水线并行（实验） ═══════════════

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

    def _dual_pipeline_available(self) -> bool:
        """单实例双流水线需要 NVDEC 和 TensorRT 均可用，构成 CPU/GPU 互补。"""
        return nvdec_available(str(self._video_path)) and tensorrt_available()

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
    def _snap_keyframe_chunks(cls, chunk_specs: list[tuple[int, int]],
                              frames: list[int], key_frames: list[int],
                              snap_from_idx: int) -> list[tuple[int, int]]:
        """把 snap_from_idx 起的大竞争片内部边界吸附到最近关键帧。

        只吸附内部边界，不移动第一大片起点和最后一大片终点，保证全帧覆盖；
        如果吸附后会越界/重叠，则保留原边界。
        """
        if snap_from_idx >= len(chunk_specs) - 1 or len(key_frames) <= 1:
            return chunk_specs
        last_end = chunk_specs[-1][1]
        starts = [chunk_specs[snap_from_idx][0]]
        for i in range(snap_from_idx + 1, len(chunk_specs)):
            orig = chunk_specs[i][0]
            cand = cls._nearest_keyframe_sample(orig, key_frames, frames)
            prev = starts[-1]
            if cand <= prev or cand >= last_end:
                cand = orig
            starts.append(cand)
        ends = starts[1:] + [last_end]
        new = list(chunk_specs)
        for k, (s, e) in enumerate(zip(starts, ends)):
            new[snap_from_idx + k] = (s, e)
        return new

    @classmethod
    def _keyframe_every_chunks(cls, frames: list[int],
                               key_frames: list[int], rest_start: int,
                               last_end: int, stride: int, min_gap: int,
                               max_chunks: int) -> list[tuple[int, int]]:
        """每关键帧一片（实验）的竞争区切片生成。

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

    def _new_worker(self, decode_backend: str, ocr_backend: str,
                    progress_cb=None, cancel_check=None) -> "FieldExtractor":
        """创建一条子流水线实例（关闭 dual，避免递归）。"""
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
        _env = _os.environ.get('OCR_THREADS')
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
        - 切片预留 + 动态竞争：每条流水线预留 1 片（init 完成即领走），
          其余进共享队列抢占——消除启动竞态把某条流水线饿死成串行；
        - 慢路径让位：滚动吞吐显著落后时停止取片，剩余片由快路径完成，
          避免尾部等待（AV1 等编码下 CPU+ONNX 慢路径不再拖垮整体）；
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
        _fps = None
        for _m in ('get_avg_fps', 'get_fps'):
            _fn = getattr(_vr, _m, None)
            if _fn is None:
                continue
            try:
                _fps = float(_fn())
                break
            except Exception:
                _fps = None
        if not _fps or _fps <= 0:
            _fps = config.DEFAULT_FPS_FALLBACK
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
                and _os.environ.get('DUAL_NO_CODEC_FALLBACK', '')
                .strip().lower() not in ('1', 'true', 'yes', 'on')):
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
        th = int(np.median(ths)) if ths else config.OTSU_FALLBACK_THRESH
        self._bin_thresh = th
        self._prof_end('parallel', 'calib_otsu', _t_cal1)

        # ── 2. 切片：头部小片（试点×2 + 确认×2）+ 大竞争片队列 ──
        # 头部 4 个小片（各约 1/DIV 视频长）：两条流水线先各领一个试点片测
        # 吞吐，再各取一个确认片二次取样；分级让位在头部结束即可判定，失衡
        # 时慢路径最多浪费约 2×1/DIV。剩余帧切成 n_chunks 个大竞争片——实测
        # 升序连续扫掠无边界代价（decord 单调前进不重寻址），乱序跳跃才有
        # ~150ms/次的精确 seek，故大片按帧序排队、赢家沿队列升序扫掠。
        n_chunks = (self._dual_pipeline_chunks
                    if self._dual_pipeline_chunks > 0
                    else config.DUAL_PIPELINE_CHUNKS)
        min_chunk = config.DUAL_PIPELINE_MIN_CHUNK_FRAMES
        unit_div = max(2, int(config.DUAL_PIPELINE_PILOT_DIV))
        unit_n = max(min_chunk, len(frames) // unit_div)
        last_end = end if self._frame_end not in (None, 0) else total
        chunk_specs: list[tuple[int, int]] = []
        has_pilots = 4 * unit_n < len(frames)
        if has_pilots:
            for i in range(4):
                a = i * unit_n
                b = (i + 1) * unit_n
                chunk_specs.append((frames[a], frames[b]))
            rest_a = 4 * unit_n
            for i in range(max(1, n_chunks)):
                a = rest_a + i * (len(frames) - rest_a) // max(1, n_chunks)
                b = rest_a + (i + 1) * (len(frames) - rest_a) // max(
                    1, n_chunks)
                chunk_specs.append((frames[a], last_end
                                    if i == n_chunks - 1 else frames[b]))
        else:
            # 视频太短放不下头部小片组：退化为等分 + 尾部预留。
            n_chunks = max(2, min(n_chunks,
                                  max(2, len(frames) // min_chunk)))
            n_chunks = min(n_chunks, len(frames))
            for i in range(n_chunks):
                a = i * len(frames) // n_chunks
                b = (i + 1) * len(frames) // n_chunks
                chunk_specs.append((frames[a], last_end
                                    if i == n_chunks - 1 else frames[b]))
        # 关键帧分片实验：只吸附大竞争片的内部边界，保留试点/确认片与首尾覆盖。
        _kf_env = _os.environ.get(
            config.DUAL_PIPELINE_KEYFRAME_ENV, '1').strip().lower()
        if _kf_env not in ('0', 'false', 'no', 'off'):
            try:
                _key_frames = [int(v) for v in _vr.get_key_indices()]
            except Exception:
                _key_frames = []
            if _key_frames:
                _snap_from = 4 if has_pilots else 0
                chunk_specs = self._snap_keyframe_chunks(
                    chunk_specs, frames, _key_frames, _snap_from)
# 每个关键帧切大片（实验）：DUAL_KEYFRAME_EVERY=1 时，试点之外的大
        # 竞争区不再等分，而是按剩余区域内的每个关键帧边界切出一片，交给共享
        # 队列自由竞争。片越密，快慢路径越容易自动配平，前提是每片落到关键帧
        # 上、seek 开销足够低。
        _kfe_env = _os.environ.get(
            config.DUAL_PIPELINE_KEYFRAME_EVERY_ENV, '').strip().lower()
        if _kfe_env in ('1', 'true', 'yes', 'on') and has_pilots and \
                len(frames) > 4 * unit_n:
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
            _big = self._keyframe_every_chunks(
                frames, _key_frames, frames[4 * unit_n],
                chunk_specs[-1][1], self._sample_stride,
                _min_gap, _max_chunks)
            chunk_specs = chunk_specs[:4] + _big
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

        # 按试点测速比例分配实验：仅在有头部小组、且大竞争片恰好 2 片时启用，
        # 两个路径各拿一个按速度比例切出的连续大区间。
        _prop_env = _os.environ.get(
            config.DUAL_PIPELINE_PROPORTIONAL_ENV, '').strip().lower()
        proportional = (
            _prop_env in ('1', 'true', 'yes', 'on')
            and has_pilots and len(pairs) == 2
            and len(list(compete_range)) == 2)
        owned_queues = ([Queue() for _ in range(len(pairs))]
                        if proportional else None)
        allocation_ready = threading.Event()
        allocation_lock = threading.Lock()
        allocation_done = [False]
        _prio_env = _os.environ.get(
            config.DUAL_PIPELINE_PRIORITY_ENV, '').strip().lower()
        priority_mode = (
            _prio_env in ('1', 'true', 'yes', 'on')
            and not proportional)
        pick_cond = threading.Condition()
        waiting_flags = [False, False]
        cur_speed: dict = {}

        # ── 3. 两个消费者线程：每条完整流水线 + 持久 OCR 引擎 ──
        result_lock = threading.Lock()
        errors: list = []
        cancel_event = threading.Event()
        chunk_results: dict = {}
        worker_stats: dict = {}
        ready_t: dict = {}
        throughput: dict = {}   # tag -> (chunks_done, steady_fps)
        e2e_speed: dict = {}    # tag -> 端到端吞吐（帧/片起点到该片 OCR 排空的墙钟）
        prog_lock = threading.Lock()
        prog_last = [-1.0]
        slow_ratio = float(config.DUAL_PIPELINE_SLOW_RATIO)
        _env_ratio = _os.environ.get('DUAL_SLOW_RATIO')
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
            fps_samples: list = []
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
                n_fr = len(range(int(start), min(int(end_f), total),
                                 self._sample_stride))
                # 吞吐口径用生产者净耗时（解码+分段 host 开销），免疫 OCR
                # 背压：片墙钟会因另一条流水线 OCR 拥塞而虚高，直接比较会
                # 误判让位（实测 GPU 路径因此被误判为慢路径）。
                base_t = prod_elapsed if prod_elapsed > 0 else chunk_time
                fps = n_fr / base_t if base_t > 0 else 0.0
                fps_samples.append(fps)
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

            def _steady_fps() -> float:
                """稳态吞吐：排除首个片（解码器/引擎 warm-up）后的近段均值。

                试点片必然含首次解码 warm-up，直接进均值会让吞吐比失真
                （实测字幕场景 GPU/CPU 稳态比 ~0.7 被 warm-up 抬到 ~0.9，
                让位判定失效）。无头部片的短视频回退为全样本。
                """
                if not fps_samples:
                    return 0.0
                steady = fps_samples[1:] if (
                    has_pilots and len(fps_samples) >= 2) else fps_samples
                recent = steady[-3:]
                return sum(recent) / len(recent)

            def _other_best() -> float:
                best = 0.0
                with result_lock:
                    for t2, v in throughput.items():
                        if t2 != tag and v[1] > best:
                            best = v[1]
                return best

            def _other_best_e2e() -> float:
                """对方流水线的端到端速率最大值（e2e_speed，最近一次在竞争
                闸门排空后记录的口径）。"""
                best = 0.0
                with result_lock:
                    for t2, v in e2e_speed.items():
                        if t2 != tag and v > best:
                            best = v
                return best

            def _priority_get():
                """在线优先取片：双方同时等待时，速度更快的一方先拿。"""
                other = 1 - ci
                other_tag = f'pipe{other + 1}'
                with pick_cond:
                    waiting_flags[ci] = True
                    try:
                        while True:
                            if cancel_event.is_set():
                                return None
                            my = cur_speed.get(tag, 0.0)
                            other_speed = cur_speed.get(other_tag, 0.0)
                            if (waiting_flags[other] and other_speed > my
                                    and not item_q.empty()):
                                pick_cond.wait(timeout=0.05)
                                continue
                            try:
                                return item_q.get_nowait()
                            except Empty:
                                return None
                    finally:
                        waiting_flags[ci] = False

            try:
                # 预留头部片组：init 完成即按序处理（试点→确认）。每片完成
                # 都立即上报吞吐，让对方的分级让位判定尽早拿到依据。
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
                    else:
                        _fps_now = _steady_fps()
                        with result_lock:
                            throughput[tag] = (chunks_done, _fps_now)
                        cur_speed[tag] = _fps_now
                if proportional:
                    # 等两条流水线都完成试点/确认片后，按稳态吞吐比例分配剩余区间。
                    with result_lock:
                        if all(t in throughput for t in ("pipe1", "pipe2")):
                            allocation_ready.set()
                    while (not allocation_ready.is_set()
                           and not cancel_event.is_set()):
                        allocation_ready.wait(0.05)
                    if cancel_event.is_set():
                        pass
                    else:
                        if ci == 0:
                            with allocation_lock:
                                if not allocation_done[0]:
                                    big_idx = list(compete_range)
                                    total_rem = len(frames) - 4 * unit_n
                                    with result_lock:
                                        v1 = throughput.get(
                                            "pipe1", (0, 0.0))[1]
                                        v2 = throughput.get(
                                            "pipe2", (0, 0.0))[1]
                                    if v1 <= 0 and v2 <= 0:
                                        v1 = v2 = 1.0
                                    n1 = total_rem
                                    if v2 > 0:
                                        n1 = int(
                                            total_rem * v1 / (v1 + v2))
                                        n1 = max(
                                            1, min(total_rem - 1, n1))
                                    elif v1 <= 0:
                                        n1 = 0
                                    split_idx = 4 * unit_n + n1
                                    end_abs = chunk_specs[-1][1]
                                    start_abs = frames[4 * unit_n]
                                    split_abs = (end_abs if split_idx >= len(frames)
                                                 else frames[split_idx])
                                    owned_queues[0].put(
                                        (big_idx[0], start_abs, split_abs))
                                    owned_queues[1].put(
                                        (big_idx[1], split_abs, end_abs))
                                    allocation_done[0] = True
                        while True:
                            try:
                                item = owned_queues[ci].get_nowait()
                            except Empty:
                                break
                            idx, start, end_f = item
                            try:
                                _do_chunk(idx, start, end_f)
                            except Exception:  # noqa: BLE001 — 已入 errors
                                break
                            with result_lock:
                                throughput[tag] = (chunks_done, _steady_fps())
                else:
                    while not cancel_event.is_set():
                        # 让位判定（分级）：吞吐比极端悬殊时单个试点片即可判定
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
                                and my_fps < 0.35 * other_fps)
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
                        if priority_mode:
                            item = _priority_get()
                            if item is None:
                                break
                        else:
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
                        _fps_now = _steady_fps()
                        with result_lock:
                            throughput[tag] = (chunks_done, _fps_now)
                        cur_speed[tag] = _fps_now
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

