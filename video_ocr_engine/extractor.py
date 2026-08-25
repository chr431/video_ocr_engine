"""FieldExtractor — 通用视频文本提取引擎（识别链：解码∥像素分段∥OCR 文本）。

引擎只输出每段原始文本与置信度；速度解析/纠错/CSV 等领域后处理由上层
应用完成（RaceVideoToLog 的 SegmentPipeline 继承本类并叠加后处理）。

方法体最初由 RaceVideoToLog 的 tools/archive/_gen_engine_extractor.py 从
segment_flow.py 抽取；独立成仓后随引擎维护，不再依赖 RaceVideoToLog。

模块划分（2026-08 七轮修正后按逻辑拆分）：
  extractor.py      — 引擎核心：解码/校准/分段/OCR 会话/流水线分发/结果组装
  _helpers.py       — 无类依赖的独立工具函数
  _result_types.py  — ExtractedSegment / ExtractionResult
  _gpu_pipeline.py  — _GpuPipelineMixin（GPU 全驻留管线）
  _dual_pipeline.py — _DualPipelineMixin（kfe 唯一分片的双流水线并行）
"""
import logging
import os as _os
import threading
import time
from pathlib import Path

import numpy as np

import engine_config as config
from segmentation import (
    _cluster_win3, _gray_seg, _gray_seg_batch,
    _gray_seg_yuv, _gray_seg_yuv_batch, _otsu,
)
from video_utils import (_nv12_luma_full,
                         _text_sep_gray, nv12_to_rgb)
# 下列 re-export 保持外部兼容（__init__ / 上层应用仍从 extractor 取）。
from ._result_types import (  # noqa: F401
    ExtractedSegment, ExtractionResult,
)
from ._helpers import (  # noqa: F401
    _ocr_batch_size, _ndarray_device_ptr, _otsu_from_hist, _gray_mean_abs_diff,
    _decode_progress_pct, _ocr_progress_pct,
    _otsu_median_threshold, _read_fps_from_vr,
)
from ._gpu_pipeline import _GpuPipelineMixin
from ._dual_pipeline import _DualPipelineMixin

logger = logging.getLogger(__name__)


def _host_calibrate(ex, vr, frames, *, with_dev=False, profile=True,
                    seek_first=None):
    """宿主路径 Otsu 校准（单流水线与并行片共用，收敛两份逐行副本）。

    ex: FieldExtractor（单流水线传 self、并行片传 worker，二者同形）——
        只调用 _crop_is_expected / _crop_luma / （profile 时）_prof_end。
    with_dev: True 时保留 decord GPU 单通道帧的 DLPack 指针（GPU raw OCR
        直通用）；stride==1 时同时捕获 next_roi 的（shape 3D）帧指针。
    profile: False 时跳过 calib_decode/calib_gray 分相记时（并行片旧实现
        不记时，保持语义一致）。
    seek_first: 非 None 时先 seek_accurate（并行片 th=None 兼容路径用；
        单流水线在调用前已按 frame_start 定位，不重复 seek）。
    stride>1 走 get_batch 等差步长快速路径（校准帧号与后续帧流一致），
    stride==1 走 next_roi 顺序流。
    返回 (calib, th)。calib 元素统一 (fi, crop, gray, sharp, dev_info)，
    dev_info 仅在 with_dev 且帧为 GPU 单通道时非 None。
    """
    x1, y1, x2, y2 = ex._roi
    calib_n = min(config.SEG_CALIB_FRAMES, len(frames))
    if seek_first is not None:
        vr.seek_accurate(seek_first)
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
            _t_p = time.perf_counter()
            nd = vr.next_roi(x1, y1, x2 + 1, y2 + 1)
            c = nd.asnumpy()
            if profile:
                ex._prof_end('producer', 'calib_decode', _t_p)
            if not ex._crop_is_expected(c, y2 - y1 + 1, x2 - x1 + 1):
                c = c[y1:y2 + 1, x1:x2 + 1]
            _t_p = time.perf_counter()
            g = ex._crop_luma(c)
            if profile:
                ex._prof_end('producer', 'calib_gray', _t_p)
            dev_info = None
            if with_dev and len(nd.shape) == 3 and nd.shape[-1] == 1:
                base, shape = _ndarray_device_ptr(nd)
                dev_info = (nd, base, shape[0], shape[1])
            calib.append((frames[k], c, g, float(g.std()), dev_info))
    ths = [_otsu(g) for _fi, _c, g, _s, _dev in calib]
    return calib, _otsu_median_threshold(ths)


def _host_frame_stream(ex, frames, vr, calib, th, *, with_dev=False,
                       phase_times=None):
    """宿主帧流：先产出校准帧，再批量流式解码剩余帧（两条宿主路径共用）。

    ex: FieldExtractor（self 或并行 worker）——只调用 _batch_luma/_prof_end。
    calib 元素统一 (fi, crop, gray, sharp, dev_info)（可为空列表）。
    phase_times 非 None 时把 decode/gray/sharp/bin 分相累加到 [1..4]
    （并行片生产者净耗时统计；单流水线不统计）。
    with_dev=True 时随帧产出 decord GPU NDArray 设备信息 (owner, ptr, h, w)
    供 GPU raw OCR 直通（仅 gray 单通道输出路径有效）。
    yield (frame_idx, crop, gray, sharp, bin, dev_info)。
    """
    DECODE_BATCH = config.DECODE_BATCH_SIZE
    x1, y1, x2, y2 = ex._roi
    for fi, c, g, s, *dev_rest in calib:
        yield (fi, c, g, s, g > th, dev_rest[0] if dev_rest else None)
    for bstart in range(len(calib), len(frames), DECODE_BATCH):
        bend = min(bstart + DECODE_BATCH, len(frames))
        _t_d = time.perf_counter()
        nds = vr.get_batch(frames[bstart:bend], roi=(x1, y1, x2 + 1, y2 + 1))
        crops = nds.asnumpy()
        if phase_times is not None:
            phase_times[1] += time.perf_counter() - _t_d
        ex._prof_end('producer', 'decode_batch', _t_d)
        _t_g = time.perf_counter()
        g = ex._batch_luma(crops)
        if phase_times is not None:
            phase_times[2] += time.perf_counter() - _t_g
        ex._prof_end('producer', 'gray_batch', _t_g)
        _t_s = time.perf_counter()
        sharp = g.std(axis=(1, 2))
        if phase_times is not None:
            phase_times[3] += time.perf_counter() - _t_s
        ex._prof_end('producer', 'sharp_batch', _t_s)
        _t_b = time.perf_counter()
        bs = g > th
        if phase_times is not None:
            phase_times[4] += time.perf_counter() - _t_b
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
                         emit, segs, rep_crops, phase_times=None):
    """宿主分段状态机（单流水线与并行片共用，消除两份逐行副本）。

    ex: FieldExtractor（self 或并行 worker，二者同形）。
    stream: (fi, crop, gray, sharp, bin, dev_info) 迭代器（_host_frame_stream）。
    emit(seg, rep_frame, rep_crop, rep_dev, rep_gray, frac)：段闭合时投递
        OCR，由调用方闭包实现（两路径的入队/全局段号/keys/reps/rep_crops
        差异收敛在闭包里；本机在调用前已把 seg 追加进 segs）。
    segs/rep_crops 由调用方传入（emit 闭包直写 rep_crops）。
    debug_tag 非 None 且 DEBUG_BOUNDS 开启时打印边界（[PB]=并行片 /
    [HB]=单流水线，与 GPU 路径 [GB] 对齐；并行片此前缺此诊断，属补齐）。
    phase_times[5] 非 None 时累加分段判定净耗时（并行片生产者净耗时口径）。
    返回 (first_rep_gray, last_rep_gray)：首发射段代表灰度与末发射段代表
    灰度（跨片缝合用，仅并行路径读取）。与旧实现逐位一致：末发射段的灰度
    不因尾部段 emit 而更新（并行片缝合依赖此语义）。
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
            changed = _cluster_win3(d) >= ex._C
            if phase_times is not None:
                phase_times[5] += time.perf_counter() - _t_seg
            ex._prof_end('producer', 'segmentation', _t_seg)
            if changed:
                seg = frames[s:k]
                if (debug_tag is not None
                        and config.env_bool(config.DEBUG_BOUNDS_ENV)):
                    print(f'[{debug_tag}]{fi}:{_cluster_win3(d):.0f}',
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


class FieldExtractor(_GpuPipelineMixin, _DualPipelineMixin):
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
        # CPU+NVDEC 混合解码（闲置 CPU 帮解码）：仅 auto/nvdec、非 AV1、
        # stride==1、未开 dual/GPU 管线时生效；失败回退纯 GPU 不致命。
        if (label == 'GPU' and backend == 'auto'
                and self._sample_stride == 1
                and not self._dual_pipeline
                and not getattr(self, '_in_dual_worker', False)
                and not self._gpu_pipeline_enabled()
                and self._codec not in ('', 'av1')
                and config.env_bool(config.HYBRID_DECODE_ENV)):
            try:
                from hybrid_decode import HybridDecoder
                _mc = int(_os.environ.get(
                    config.HYBRID_MAX_CHUNKS_ENV, '16') or 16)
                _ct = int(_os.environ.get(
                    config.HYBRID_CPU_THREADS_ENV, '0') or 0)
                vr = HybridDecoder(self, vr, max_chunks=_mc,
                                   cpu_threads=_ct)
                self._backend = 'decord/GPU+CPU-hybrid'
                logger.info('混合解码开启(kfe竞争): codec=%s chunks<=%d cpuT=%d',
                            self._codec, _mc, _ct)
            except Exception as e:  # noqa: BLE001
                logger.warning('混合解码初始化失败，回退纯 GPU: %s', e)
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
        _env = _os.environ.get(config.OCR_THREADS_ENV)
        if _env:
            return max(1, int(_env))
        cores = auto_ocr_thread_count()
        if getattr(self, '_codec', '') == 'av1' and getattr(self, '_backend', '').startswith('decord/CPU'):
            return max(2, cores // 2)
        if getattr(self, '_backend', '').startswith('decord/CPU') and cores <= config.CPU_CORES_SPLIT_THRESHOLD:
            return max(2, cores // 2)
        return cores

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
                    dual_onnx = (self._ocr_engine_type() == 'onnxruntime'
                                 and ot >= config.DUAL_ONNX_MIN_THREADS
                                 and config.env_bool(config.DUAL_PIPELINE_ONNX_ENV,
                                                     default=True))
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
        返回 (segs, keys, reps, rep_crops, decode_elapsed, first_rep_gray,
              last_rep_gray, producer_elapsed)；首/末代表帧灰度与生产者净耗时
        供跨片边界 merge_similar 缝合与让位判定使用。
        """
        total = len(vr)
        end = min(worker._frame_end or total, total)
        frames = list(range(start, end, worker._sample_stride))
        if not frames:
            raise ValueError(
                f"帧区间为空: frame_start={start}, "
                f"frame_end={end}, total={total}")
        calib: list = []
        if th is None:
            # 兼容路径：无全局阈值时片内自行校准（前 SEG_CALIB_FRAMES 帧 Otsu）。
            # 共用宿主校准助手 _host_calibrate：stride>1 走 get_batch 等差快速路径、
            # stride==1 走 next_roi 顺序流；先精确 seek 到片首（该路径依赖解码器
            # 当前位置语义，与旧实现一致）。
            calib, th = _host_calibrate(worker, vr, frames, with_dev=False,
                                        profile=False, seek_first=start)
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
        # 帧流（校准帧 + 批量解码→灰度→sharp→二值化）与分段状态机由共享助手
        # _host_frame_stream / _host_segment_frames 承担（见模块顶部），本路径
        # 仅传入 phase_times=prod_acc 做生产者净耗时统计（含 OCR 背压免疫口径）。

        segs: list = []
        keys: list = []
        reps: list = []
        rep_crops: dict = {}
        seg_idx = int(session.get("seg_idx", 0))
        t0 = time.perf_counter()

        def emit(seg, r_frame, r_crop, r_dev, _r_gray, frac) -> None:
            nonlocal seg_idx
            keys.append(seg_idx)
            reps.append(r_frame)
            session["put"]((seg_idx, r_frame, r_crop, r_dev, frac))
            if worker._keep_crops:
                rep_crops[r_frame] = r_crop
            seg_idx += 1

        # 分段状态机（共享 _host_segment_frames，见模块顶部）：段闭合/相似
        # 合并/代表帧选择/首末代表灰度/取消与进度节奏全部收敛于此；并行片
        # 的产物净耗时统计（prod_acc[5]）经 phase_times 传入。
        first_gray, last_gray = _host_segment_frames(
            worker, frames,
            _host_frame_stream(worker, frames, vr, calib, th,
                               phase_times=prod_acc),
            debug_tag='PB',
            progress_prefix=f'[{worker._backend}] 并行解码+分段',
            emit=emit, segs=segs, rep_crops=rep_crops,
            phase_times=prod_acc)
        session["seg_idx"] = seg_idx
        return (segs, keys, reps, rep_crops, time.perf_counter() - t0,
                first_gray, last_gray, sum(prod_acc))

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

            返回 (frames, segs, ocr_texts, ocr_confs, rep_frames)；
            self.crops = {rep_frame: crop}（仅代表帧，供 review 预览，
            比存全帧省内存）。分段/代表帧选择语义由模块级共享状态机
            _host_segment_frames 承担（本方法与并行片路径共用）。
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
        if hasattr(vr, 'hybrid_begin'):
            vr.hybrid_begin(frames)
        self._prof_end('producer', 'open_and_fps', _t_open)
        _t_cal = time.perf_counter()
        # 宿主校准统一走 _host_calibrate（stride>1 用 get_batch 等差快速路径、
        # stride==1 用 next_roi 顺序流——校准帧号与后续流水线帧号一致）。
        # with_dev=True：保留 GPU 单通道帧的 DLPack 指针供 GPU raw OCR 直通。
        calib, th = _host_calibrate(self, vr, frames, with_dev=True)
        self._bin_thresh = th
        self._prof_end('producer', 'calib_total', _t_cal)
        ocr_session = self._start_ocr_session(_ocr_engines)
        q = ocr_session["q"]
        results = ocr_session["results"]
        ocr_err = ocr_session["err"]
        ocr_wall = ocr_session["wall"]
        _put_ocr = ocr_session["put"]

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
                                   with_dev=True),
                debug_tag='HB',
                progress_prefix=f'[{self._backend}] 解码+分段',
                emit=_emit_ocr, segs=segs, rep_crops=rep_crops)
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
