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
    _apply_gamma, _cluster_win3, _gray, _gray_batch, _gray_seg,
    _gray_seg_batch, _gray_seg_yuv, _gray_seg_yuv_batch, _otsu, _seg_gamma,
)
from hybrid_decode import (
    HYBRID_BACKEND_ALIASES, _decode_range_worker, _drain_queue, _hybrid_ranges,
)
from ocr_native import OcrEngine, auto_ocr_thread_count
from video_utils import (_nv12_luma_full, _preprocess_standard,
                         nv12_to_rgb, nvdec_available,
                         tensorrt_available)  # 识别链 YUV/preprocess/RGB 预览

logger = logging.getLogger(__name__)


def _ocr_batch_size() -> int:
    _env = _os.environ.get("RVTOL_OCR_BATCH")
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
                 keep_frames: bool = True, merge_similar: bool = False,
                 merge_similar_threshold: float | None = None,
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
        self._hybrid_codec = ""
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
            "RVTOL_PROFILE", "").strip().lower() in ("1", "true", "yes", "on")
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

    def _segments_similar(self, a, b) -> bool:
        """相似段判定：平均绝对差小 且 显著变化像素占比也小。

        只用平均绝对差会把宽 ROI 中的单字短字幕（如“在”“不”）误判为噪声：
        大部分区域未变，均值被稀释。因此额外限制 abs(diff)>10 的像素数。
        """
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
        del vr, vr_gpu
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
                        and (_os.environ.get(
                            'RVTOL_GPU_RAW', '').strip().lower()
                            in ('1', 'true', 'yes', 'on')
                            or getattr(self, '_gpu_pipeline_mode', False)))
                    if raw_ok:
                        infos = [(d[1], d[2], d[3], d[0]) for d in b_devs]
                        raw_res = engines[0].call_gpu_raw(infos)
                        for idx, rep, r, frac in zip(
                                b_idx, b_reps, raw_res, b_fracs):
                            _store_result(idx, rep, r, frac)
                        b_idx.clear(); b_reps.clear(); b_crops.clear()
                        b_devs.clear(); b_fracs.clear()
                        return
                    _t_p = time.perf_counter()
                    procs = [_preprocess_standard(
                        _nv12_luma_full(c, self._color_range)[..., None]
                        if self._yuv_output else c,
                        force_aspect=self._force_aspect) for c in b_crops]
                    self._prof_end('ocr', 'preprocess', _t_p)
                    if not _put_infer((list(b_idx), list(b_reps), procs, list(b_fracs))):
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
                            start: int, end_f: int, n_chunks: int):
        """双流水线 worker 处理单个切片的解码/分段，送入共享 OCR 会话。

        不等待 OCR 完成、不新建 OCR 线程——只把段任务塞进 session["q"]，
        这样下一个切片可以立即开始解码，真正跨片重叠。
        返回 (segs, keys, reps, rep_crops, decode_elapsed)。
        """
        x1, y1, x2, y2 = worker._roi
        total = len(vr)
        end = min(worker._frame_end or total, total)
        if worker._frame_start > 0:
            vr.seek_accurate(worker._frame_start)
        frames = list(range(worker._frame_start, end, worker._sample_stride))
        if not frames:
            raise ValueError(
                f"帧区间为空: frame_start={worker._frame_start}, "
                f"frame_end={end}, total={total}")
        calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
        calib: list = []
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
        DECODE_BATCH = config.DECODE_BATCH_SIZE

        def frame_stream():
            for fi, c, g, s in calib:
                yield (fi, c, g, s, g > th)
            for bstart in range(calib_n, len(frames), DECODE_BATCH):
                bend = min(bstart + DECODE_BATCH, len(frames))
                crops = vr.get_batch(
                    frames[bstart:bend],
                    roi=(x1, y1, x2 + 1, y2 + 1)).asnumpy()
                g = worker._batch_luma(crops)
                sharp = g.std(axis=(1, 2))
                bs = g > th
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

        def emit(seg, rep_frame, rep_crop, frac) -> None:
            nonlocal seg_idx
            segs.append(seg)
            keys.append(seg_idx)
            reps.append(rep_frame)
            session["put"]((seg_idx, rep_frame, rep_crop, None, frac))
            if worker._keep_crops:
                rep_crops[rep_frame] = rep_crop
            seg_idx += 1

        for k, (fi, c, g, sharp, b) in enumerate(frame_stream()):
            if prev_b is not None:
                d = prev_b != b
                changed = _cluster_win3(d) >= worker._C
                if changed:
                    seg = frames[s:k]
                    similar = (
                        worker._merge_similar and segs
                        and worker._segments_similar(last_rep_gray, rep_gray))
                    if similar:
                        segs[-1].extend(seg)
                    else:
                        emit(seg, rep_frame, rep_crop,
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
            emit(seg, rep_frame, rep_crop, 1.0)
        session["seg_idx"] = seg_idx
        return segs, keys, reps, rep_crops, time.perf_counter() - t0

    def _gpu_pipeline_enabled(self) -> bool:
        """实验 GPU 全流水线开关：RVTOL_GPU_PIPELINE=1 + gray + GPU/TRT 可用。"""
        _env = _os.environ.get('RVTOL_GPU_PIPELINE', '').strip().lower()
        if _env not in ('1', 'true', 'yes', 'on'):
            return False
        if not self._gray_output or self._yuv_output:
            return False
        return nvdec_available(str(self._video_path)) and tensorrt_available()

    def _run_pipelined_gpu(self):
        """实验：灰度/sharp/聚类变化分都在 GPU 计算，host 只收标量。

        代表帧保留 GPU device pointer，OCR 走 RVTOL_GPU_RAW 自动开启的
        call_gpu_raw 路径。校准阈值仍取前 50 帧 D2H（量小，可接受）。
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
        calib_crops = calib_nds.asnumpy()
        calib_th = []
        for k in range(calib_n):
            c = calib_crops[k]
            if not self._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                c = c[y1:y2 + 1, x1:x2 + 1]
            calib_th.append(_otsu(self._crop_luma(c)))
        th = int(np.median(calib_th)) if calib_th else config.OTSU_FALLBACK_THRESH
        self._bin_thresh = th
        calib_base, calib_shape = _ndarray_device_ptr(calib_nds)
        calib_c = calib_shape[-1] if len(calib_shape) == 4 else 0
        if calib_c != 1:
            return self._run_pipelined(_force_single=True)

        self._gpu_pipeline_mode = True
        analyzer = GpuFrameAnalyzer()
        ocr_session = self._start_ocr_session(None)
        q = ocr_session["q"]
        results = ocr_session["results"]
        ocr_err = ocr_session["err"]
        ocr_wall = ocr_session["wall"]
        _put_ocr = ocr_session["put"]

        src_h, src_w = calib_shape[1], calib_shape[2]
        prev_holder = calib_nds
        prev_ptr = calib_base

        def frame_stream():
            nonlocal prev_holder, prev_ptr
            from cuda.bindings import runtime as cudart
            DECODE_BATCH = config.DECODE_BATCH_SIZE
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

        segs: list = []
        rep_crops: dict = {}
        seg_idx = 0
        s = 0
        rep_frame = frames[0]
        rep_dev = None
        rep_sharp = -1.0
        prev_seen = False
        t0 = time.perf_counter()
        try:
            for k, (fi, dev, sharp, cluster) in enumerate(frame_stream()):
                if prev_seen:
                    changed = float(cluster) >= self._C
                    if changed:
                        seg = frames[s:k]
                        segs.append(seg)
                        _put_ocr((seg_idx, rep_frame, None, rep_dev,
                                  k / max(len(frames), 1)))
                        seg_idx += 1
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
                                   3 + k / max(len(frames), 1) * 55)
            seg = frames[s:]
            segs.append(seg)
            _put_ocr((seg_idx, rep_frame, None, rep_dev, 1.0))
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
            灰度 std），OCR 批 _ocr_batch_size()。cpu+nvdec 时两个解码线程
            （CPU 前段 + GPU 后段）并行填有界队列，消费者按序合并，帧序与单解码器一致。

            返回 (frames, segs, seg_vals, rep_frames)；self.crops = {rep_frame:
            crop}（仅代表帧，供 review 预览，比存全帧省内存）。
            """
        if self._dual_pipeline and _ocr_engines is None and not _force_single:
            return self._run_pipelined_parallel()
        if self._gpu_pipeline_enabled():
            return self._run_pipelined_gpu()
        from queue import Full, Queue
        import threading
        from ocr_native import OcrEngine
        from video_utils import _preprocess_standard
        _t_open = time.perf_counter()
        hybrid = self._is_hybrid()
        vr_gpu = None
        if hybrid:
            vr, vr_gpu = self._open_hybrid_vrs()
            if vr_gpu is None:
                hybrid = False
        else:
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
        ocr_thread = ocr_session["thread"]
        _put_ocr = ocr_session["put"]

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
                for fi, c, g, s, dev in calib:
                    yield (fi, c, g, s, g > th, dev)
                for item in _drain_queue(cpu_q):
                    yield (*item, None)
                if gpu_q is not None:
                    for item in _drain_queue(gpu_q):
                        yield (*item, None)
        else:

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
        consumer_ok = [False]
        try:
            for k, (fi, c, g, sharp, b, dev_info) in enumerate(frame_stream()):
                if prev_b is not None:
                    d = prev_b != b
                    _t_seg = time.perf_counter()
                    changed = _cluster_win3(d) >= self._C
                    self._prof_end('producer', 'segmentation', _t_seg)
                    if changed:
                        seg = frames[s:k]
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
            consumer_ok[0] = True
        finally:
            _t_consume_end = time.perf_counter()
            self.timing['decode'] = _t_consume_end - t0
            self._prof_end('producer', 'consumer_total', t0)
            if consumer_ok[0]:
                for t in dec_threads:
                    t.join()
            ocr_session["finish"]()
            self.timing['ocr_tail'] = time.perf_counter() - _t_consume_end
        if dec_err:
            raise dec_err[0]
        if ocr_err:
            raise ocr_err[0]
        self.timing['ocr'] = ocr_wall[0]
        self._n_segments = len(segs)
        self.crops = rep_crops
        del vr, vr_gpu
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
        """互补 OCR 后端：CPU ONNX ↔ auto（TensorRT 优先）。"""
        return "auto" if str(backend or "").strip().lower() == "cpu" else "cpu"

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

    def _dual_ocr_num_threads(self) -> int:
        """双流水线的 OCR 线程预算：env 钩子优先，否则全物理核。

        双流水线设计默认是“CPU+ONNX 与 GPU+TRT”互补，CPU 侧独占软解+ONNX，
        与 GPU 侧不抢核；少核/AV1 的精细分核暂不在此实验路径展开。
        """
        _env = _os.environ.get('RVTOL_OCR_THREADS')
        if _env and _env.isdigit():
            return max(1, int(_env))
        return auto_ocr_thread_count()

    def _run_pipelined_parallel(self):
        """单实例双完整流水线并行：同一视频切多片，两流水线动态取片。

        与旧“混合解码/混合 OCR”只在一个阶段内并行的方案不同：
        这里是两条完整“解码→分段→OCR”流水线各自带互补后端（如 GPU+TRT 与
        CPU+ONNX），从共享队列取连续小片，谁快谁多干，避免机械对半切导致
        快流水线闲置。最后按片序合并段文本/置信度/代表帧。

        需要 NVDEC 与 TensorRT 均可用；不满足则由 _run_pipelined 分发到此处
        前先探测，回退单流水线（_force_single=True）。
        """
        from queue import Empty, Queue
        import threading
        from ocr_native import OcrEngine

        if not self._dual_pipeline_available():
            logger.warning(
                '单实例双流水线需要 NVDEC 和 TensorRT 均可用，回退单流水线')
            return self._run_pipelined(_force_single=True)

        # ── 1. 探测视频总长 / fps，构造全局采样帧列表 ──
        from video_utils import open_decord_vr
        _t_probe = time.perf_counter()
        _vr, _label = open_decord_vr(str(self._video_path))
        try:
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
        finally:
            del _vr
        if self._fps is None:
            self._fps = _fps
        end = min(self._frame_end or total, total)
        frames = list(range(self._frame_start, end, self._sample_stride))
        if not frames:
            raise ValueError(
                f"帧区间为空: frame_start={self._frame_start}, "
                f"frame_end={end}, total={total}")
        if len(frames) < 2:
            return self._run_pipelined(_force_single=True)

        # ── 2. 切连续小片；每个 worker 从队列取片（动态负载均衡）──
        n_chunks = (self._dual_pipeline_chunks
                    if self._dual_pipeline_chunks > 0
                    else config.DUAL_PIPELINE_CHUNKS)
        min_chunk = config.DUAL_PIPELINE_MIN_CHUNK_FRAMES
        n_chunks = max(2, min(n_chunks, max(2, len(frames) // min_chunk)))
        n_chunks = min(n_chunks, len(frames))
        chunk_specs: list[tuple[int, int]] = []
        for i in range(n_chunks):
            a = i * len(frames) // n_chunks
            b = (i + 1) * len(frames) // n_chunks
            start = frames[a]
            if i == n_chunks - 1:
                end_f = end if self._frame_end not in (None, 0) else total
            else:
                end_f = frames[b]
            chunk_specs.append((start, end_f))
        item_q: Queue = Queue()
        for idx, spec in enumerate(chunk_specs):
            item_q.put((idx, spec[0], spec[1]))

        # ── 3. 两个消费者线程：每条完整流水线 + 持久 OCR 引擎 ──
        result_lock = threading.Lock()
        errors: list = []
        cancel_event = threading.Event()
        chunk_results: dict = {}
        worker_stats: dict = {}
        prog_lock = threading.Lock()
        prog_last = [-1.0]

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

        def _consumer(decode_backend: str, ocr_backend: str, tag: str) -> None:
            worker = self._new_worker(
                decode_backend, ocr_backend,
                progress_cb=None, cancel_check=self._cancel)
            try:
                worker_vr = worker._open_vr()
            except Exception as e:  # noqa: BLE001
                with result_lock:
                    errors.append(e)
                cancel_event.set()
                return
            try:
                eng = OcrEngine(
                    self._ocr_model,
                    worker._ocr_engine_type(),
                    fill_width=self._fill_width,
                    num_threads=self._dual_ocr_num_threads(),
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
            try:
                while not cancel_event.is_set():
                    try:
                        item = item_q.get_nowait()
                    except Empty:
                        break
                    idx, start, end_f = item
                    worker._frame_start = int(start)
                    worker._frame_end = int(end_f)
                    worker._progress = _chunk_progress(idx, n_chunks)
                    _t_chunk = time.perf_counter()
                    try:
                        segs, keys, reps, crops_chunk, dec_elapsed = (
                            self._run_parallel_chunk(
                                worker, worker_vr, session, idx,
                                start, end_f, n_chunks))
                    except Exception as e:  # noqa: BLE001
                        with result_lock:
                            errors.append(e)
                        cancel_event.set()
                        break
                    wall += time.perf_counter() - _t_chunk
                    chunks_done += 1
                    chunk_meta[idx] = (
                        segs, keys, reps, crops_chunk, dec_elapsed)
            finally:
                try:
                    session["finish"]()
                except Exception as e:  # noqa: BLE001
                    with result_lock:
                        if not errors:
                            errors.append(e)
                    cancel_event.set()
                if session["err"]:
                    with result_lock:
                        if not errors:
                            errors.append(session["err"][0])
                # OCR 会话结束后按 chunk 内全局段索引组装结果
                for idx in sorted(chunk_meta):
                    segs, keys, reps, crops_chunk, dec_elapsed = chunk_meta[idx]
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
                        chunk_results[idx] = (
                            None, segs, texts, confs, reps_out,
                            crops_chunk, {"decode": dec_elapsed},
                            worker._ocr_backend_used, worker._backend)
                with result_lock:
                    worker_stats[tag] = (
                        chunks_done, wall, worker._backend,
                        worker._ocr_backend_used, session["wall"][0])

        pairs = self._dual_backend_pairs()
        threads = [
            threading.Thread(
                target=_consumer, args=(dec, ocr, f'pipe{i + 1}'),
                daemon=True)
            for i, (dec, ocr) in enumerate(pairs)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        if len(chunk_results) != n_chunks:
            raise RuntimeError(
                f"双流水线切片结果不完整: {len(chunk_results)}/{n_chunks}")

        # ── 4. 按片序合并（帧序全局单调）──
        all_segs: list = []
        all_texts: list = []
        all_confs: list = []
        all_reps: list = []
        all_crops: dict = {}
        timing_sum: dict = {}
        backend_names: list = []
        ocr_backend_names: list = []
        for i in sorted(chunk_results):
            _fr, segs, texts, confs, reps, crops_chunk, tim, ob, bk = (
                chunk_results[i])
            all_segs.extend(segs)
            all_texts.extend(texts)
            all_confs.extend(confs)
            all_reps.extend(reps)
            all_crops.update(crops_chunk)
            for k, v in tim.items():
                if isinstance(v, (int, float)):
                    timing_sum[k] = timing_sum.get(k, 0.0) + float(v)
            ocr_backend_names.append(ob or "")
            backend_names.append(bk or "")
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
        # 每条流水线完成的片数/墙钟（诊断 GPU/CPU 路径是否闲置）
        ocr_walls: list = []
        for tag in sorted(worker_stats):
            chunks_done, wall, w_backend, w_ocr, w_ocr_wall = worker_stats[tag]
            self.timing[f'parallel_{tag}_chunks'] = chunks_done
            self.timing[f'parallel_{tag}_s'] = wall
            self.timing[f'parallel_{tag}_backend'] = w_backend
            self.timing[f'parallel_{tag}_ocr'] = w_ocr
            ocr_walls.append(w_ocr_wall)
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

