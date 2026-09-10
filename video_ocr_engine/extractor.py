"""FieldExtractor — 通用视频文本提取引擎（识别链：解码∥像素分段∥OCR 文本）。

引擎只输出每段原始文本与置信度；速度解析/纠错/CSV 等领域后处理由上层
应用完成（引擎保持通用性，不携带任何下游领域后处理）。

方法体最初由既有视频项目的历史 tools/archive 生成脚本从 segment_flow.py
抽取；独立成仓后随引擎维护，不再依赖任何下游仓库。

模块划分（2026-08 七轮修正后按逻辑拆分）：
  extractor.py      — 引擎骨架：构造/参数校验/解码器打开/流水线分发/结果组装
  _host_pipeline.py — 宿主管线：校准/帧流/分段状态机/OCR 会话
                      （_host_calibrate / _host_frame_stream /
                      _host_segment_frames / _HostPipelineMixin）
  _helpers.py       — 无类依赖的独立工具函数
  _result_types.py  — ExtractedSegment / ExtractionResult
  _gpu_pipeline.py  — _GpuPipelineMixin（GPU 全驻留管线）
双流水线并行已被移除（2026-08 清理）；CPU+NVDEC 双解码（decode_backend=
"hybrid"）由 decord fork 原生实现（≥v0.7.15 的 hybrid/hybrid_gpu ctx），
引擎只透传解码参数；项目层 hybrid_decode.py 已删除，勿再引用。
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
    _text_sep_binary,
)
# 下列 re-export 为引擎内部结构（_helpers/_result_types/_host_pipeline 均
# 属下划线私有命名，从 extractor 再导出仅为旧导入路径兼容，勿直接 import；
# 公共入口是 video_ocr_engine.__init__ 的三件套）。
from ._result_types import (  # noqa: F401
    ExtractedSegment, ExtractionResult,
)
from ._helpers import (  # noqa: F401
    _ocr_batch_size, _ndarray_device_ptr,
    _decode_progress_pct, _ocr_progress_pct,
    _read_fps_from_vr,
)
from ._host_pipeline import (  # noqa: F401
    _HostPipelineMixin,
)
from ._gpu_pipeline import _GpuPipelineMixin
from .pipeline.engine import SegmentEngine, _LegacyBackend
from .pipeline.gpu_backend import GpuRunSpec, run_gpu_pipeline
from .pipeline.host_backend import HostRunSpec, run_host_pipeline

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
                 progress_cb=None, cancel_check=None,
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
        fmt = (rep_crop_format or '').strip().lower() or config.DEFAULT_REP_CROP_FORMAT
        if fmt not in ('yuv', 'gray'):
            raise ValueError(
                f"rep_crop_format 必须为 'yuv' 或 'gray'，收到 {fmt!r}")
        self._rep_crop_format = fmt
        # 实际 decord 输出格式：keep_crops=False 时无 UV 平面需求 → 直接 gray
        #（省 0.5B/px 解码转换/传输）；yuv420 仅在 keep YUV 时启用。
        self._yuv_output = bool(self._keep_crops and fmt == 'yuv')
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
        self._degraded: list = []        # 本次提取的降级/回退原因（D3，meta 透出）
        # OCR 输入宽度自适应裁切（宽 ROI 字幕省卷积）+ 跨批按宽度分组。
        # 详见 _host_pipeline._crop_to_content 与 config 中的实测注释。
        # 四个 autocrop/重排旋钮为 property 调用期读 env（A6：与
        # OCR_PAD_SMALL/OCR_GAMMA 等同时机，构造后改 env 即生效）。
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
        self._ensure_roi_capable_decoder()
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
        # B3（Q8 裁决，S3）：未知后端构造期硬失败。v1 语义是静默兜底——
        # decode_backend 未知值静默走 CPU、ocr_backend 未知值当 tensorrt
        # （P0-3：排查陷阱）；合法集与 _open_vr/_ocr_engine_type 的判定一致
        _dec = (self._decode_backend or 'auto').lower()
        if _dec not in ('auto', 'cpu', 'nvdec', 'hybrid'):
            raise ValueError(
                f"decode_backend 必须为 auto/cpu/nvdec/hybrid，"
                f"收到 {self._decode_backend!r}")
        _ocr = (self._ocr_backend or 'auto').lower()
        if _ocr not in ('auto', 'cpu', 'tensorrt'):
            raise ValueError(
                f"ocr_backend 必须为 auto/cpu/tensorrt，"
                f"收到 {self._ocr_backend!r}")


    # ── env 旋钮统一调用期读取（A6：与 OCR_PAD_SMALL/OCR_GAMMA 等同时机，
    #    构造后改 env 即生效；此前 autocrop 四项在构造期烘焙，语义不一致）──
    @property
    def _ocr_autocrop(self) -> bool:
        return config.env_bool(config.OCR_ROI_AUTOCROP_ENV,
                               default=config.OCR_ROI_AUTOCROP_DEFAULT)

    @property
    def _ocr_autocrop_margin_pct(self) -> int:
        return config.env_int(config.OCR_ROI_AUTOCROP_MARGIN_ENV,
                              config.OCR_ROI_AUTOCROP_MARGIN_PCT)

    @property
    def _ocr_autocrop_min_gain(self) -> float:
        # 最小收益门槛：裁掉比例低于此值就整段不裁（紧凑 ROI 自动不裁，
        # 避免在几乎没留白的段上承担切笔画的风险）。见 config 中的实测表。
        return max(0, config.env_int(config.OCR_ROI_AUTOCROP_MIN_GAIN_ENV,
                                     config.OCR_ROI_AUTOCROP_MIN_GAIN_PCT)) / 100.0

    @property
    def _ocr_reorder_window(self) -> int:
        return max(1, config.env_int(config.OCR_REORDER_WINDOW_ENV,
                                     config.OCR_REORDER_WINDOW_DEFAULT))

    def _ensure_roi_capable_decoder(self) -> None:
        """构造期校验解码器支持 ROI-first 输出（DESIGN-REVIEW C8）。

        无 `_CAPI_VideoReaderSetRoi` 的 decord（如 PyPI 版）会让引擎静默按
        **整帧**处理（roi 参数被忽略、校准阈值与分段数据尺寸错位、merge 因
        形状不齐恒 False）——静默错误比报错更危险，故直接拒绝。
        decord 未安装时不在此拦截（保持"构造不依赖 decord"的测试约定），
        由 extract() 打开解码器时自然报错。
        """
        try:
            import decord.video_reader as _vr_mod
        except ImportError:
            return
        if not hasattr(_vr_mod, '_CAPI_VideoReaderSetRoi'):
            raise ValueError(
                "当前 decord 不支持 ROI-first 解码（缺 _CAPI_VideoReaderSetRoi）。"
                "引擎需要 chr431/decord fork（见 README「解码后端」）；"
                "拒绝在整帧模式下静默忽略 roi 参数。")

    def _merge_effective_mode(self) -> str:
        """merge_similar 使用的分离模式（env 钩子优先级与 _segments_similar
        一致）：'binary' | ''（原始灰度比较）。contrast 模式已移除
        （实验证实无净收益，0.9.0 清理；历史见 docs/PERFORMANCE.md）。"""
        _m = _os.environ.get(
            config.TEXT_SEP_MERGE_ENV, self._merge_text_sep or ''
        ).strip().lower()
        if _m in ('2', 'binary'):
            return 'binary'
        if _m in ('off', ''):
            return ''
        return 'binary'   # contrast/未知值 → 引擎默认 binary

    def _set_bin_thresh(self, th: int) -> None:
        """校准阈值即时回写（F-4：合并判定在流式期间活读本值，
        回写延迟到 run 结束会使宿主路径段数漂移 1042 vs 1083 类复发）。"""
        self._bin_thresh = th

    def _segments_similar(self, a, b) -> bool:
        """相似段判定：平均绝对差小 且 显著变化像素占比也小。

        只用平均绝对差会把宽 ROI 中的单字短字幕（如“在”“不”）误判为噪声：
        大部分区域未变，均值被稀释。因此额外限制 abs(diff)>10 的像素数。
        分离模式由 _merge_effective_mode 决定（binary 为引擎默认）。
        """
        _text_mode = self._merge_effective_mode()
        if _text_mode == 'binary':
            # S2：直连实现（原 _text_sep_gray 转发壳已删；mode 恒 binary，
            # th 由调用方显式给出——壳的历史默认分支无活调用点）
            a = _text_sep_binary(a, self._bin_thresh)
            b = _text_sep_binary(b, self._bin_thresh)
        if a is None or b is None or a.shape != b.shape:
            return False
        # int16 精确差：避免 float32 双全帧临时数组（a/b 为 uint8 灰度或
        # float32 分离图，255.0/0.0 转 int16 精确）。与 GPU sim_pair 的
        # 整数精确累加一致（阈值处仅 float32 末位舍入差异，文档已承认）。
        # 两阈值判定统一走 segmentation.similar_decision（GPU 共用）。
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        from segmentation import similar_decision
        return similar_decision(float(diff.mean()), int(np.sum(diff > 10)),
                                self._merge_similar_threshold,
                                self._merge_max_changed_pixels)

    def extract(self):
        """通用文本提取：解码∥分段∥OCR → 结构化结果（每段原始文本+置信度）。

        引擎的正式通用入口（无任何领域语义）。返回 ExtractionResult：
          - segments: list[ExtractedSegment]（start/end/rep_frame/text/confidence/
            rep_crop）
          - frames / fps / timing / meta
        识别层不解析文本含义（速度/数值由上层应用处理）。fps 强制自测。
        """
        # B1（Q8 裁决，S3）：每次 extract 全量重置运行态——v1 只在 __init__
        # 赋值，同实例第二次 extract 会带上一次的降级原因/计时/剖面
        # （README 却声称"每次全量重跑并覆盖实例状态"）。
        # 不重置：_fps（B2：同实例同视频，文档化缓存）。
        self._degraded = []
        self.timing = {}
        self.crops = {}
        self.profile = {}
        self._frames = []
        self._ocr_texts = []
        self._ocr_confs = []
        self._n_segments = 0
        self._backend = ""
        self._ocr_backend_used = ""
        self._bin_thresh = 0
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
                  "n_segments": len(segments),
                  "engine_version": config.__version__,
                  # D4：rep_crop_rgb helper 依赖这两个字段还原预览
                  "color_range": self._color_range,
                  "rep_crop_format": ("yuv" if self._yuv_output
                                      else "gray"),
                  "degraded_reason": (self._degraded or None),   # D3
                  "params": {                                     # D3
                      "roi": tuple(self._roi),
                      "frame_start": self._frame_start,
                      "frame_end": self._frame_end,
                      "decode_backend": self._decode_backend,
                      "ocr_backend": self._ocr_backend,
                      "sample_stride": self._sample_stride,
                      "fill_width": self._fill_width,
                      "force_aspect": self._force_aspect,
                      "rep_crop_format": self._rep_crop_format,
                      "keep_crops": self._keep_crops,
                      "keep_frames": self._keep_frames,
                      "merge_similar": self._merge_similar,
                      "merge_similar_threshold": self._merge_similar_threshold,
                      "merge_text_sep": self._merge_effective_mode(),
                      "buffer_size": self._buffer_size,
                      "C": self._C}})

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
        vr = None
        label = 'CPU'
        # ⚠️ auto 恒为 NVDEC 优先是**刻意决策**（2026-09-10 重申，勿再改）：
        # 本机强多核下实测 h264 CPU 软解确比 NVDEC 快 1.7~2.8×，但 auto 不
        # 据此分流——弱 CPU 上 h264 软解可能慢于 NVDEC，且 CPU 解码必然
        # 引入资源争用与整机功耗上升，NVDEC 稳妥优先。峰值吞吐差异留给
        # 用户显式 decode_backend="cpu"。见 CONCLUSIONS C-07/C-08/C-33。
        if backend in ('auto', 'nvdec', 'hybrid'):
            try:
                from decord import gpu as _g
                vr = self._open_decord_reader(_g(0), roi_kw)
                label = 'GPU'
            except Exception:
                vr = None
                self._degraded.append('NVDEC 打开失败，回退 CPU')
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
            # codec 感知线程档位（2026-09-10 实测表，见 _decode_num_threads）：
            # hevc/av1 在 FFmpeg9 下的帧线程扩展性与 h264 分化（hevc
            # stride=1 到 32 线程仍在涨、av1 stride=8 最优 48），通用档位
            # 按最常见 h264 设定，打开后读 codec、档位不同则重开一次
            # （实测重开 ~20-30ms，hevc stride1 墙钟 -27%）。
            nt = self._decode_num_threads()
            nt_codec = self._decode_num_threads(codec=self._codec or None)
            if nt_codec != nt:
                vr = self._open_decord_reader(_cpu(0), roi_kw,
                                              num_threads=nt_codec)
        else:
            try:
                self._codec = str(vr.get_codec() or '').lower()
            except Exception:
                self._codec = ''
        self._remember_color_range(vr)
        # CPU+NVDEC 混合解码（decode_backend="hybrid" 显式选择，与 auto/cpu/nvdec
        # 并列）：速率比例分界 + 两端连续扫掠（HybridDecoder v3/v4）。
        # NVDEC 可用即包装（GPU 全驻留管线开启时由其 CPU 分支消费宿主数组，
        # §8.3 合并；关闭时走宿主管线，行为不变）。
        # **stride>1 已解禁**（原门控要求 stride==1，理由是 next_roi 的
        # 顺序交付语义）：next_roi 现在按 _sample_stride 推进（见
        # hybrid_decode），且 stride>1 时宿主校准与主循环都走 get_batch
        # 等差快速路径、不碰 next_roi。解禁的意义是 hybrid 只在"解码占
        # 墙钟大头"时才可能赢，而 stride>1 恰恰把解码占比推到最高。
        # 编码（含 AV1）不再回退——v3 的速率比例分界已实测：CPU 慢于
        # NVDEC 的 HEVC/AV1 场景与纯 NVDEC 持平不退化，CPU 快于 NVDEC 的
        # h264 场景显著更快；尊重用户显式选择。NVDEC 不可用时上面已回退
        # CPU 并警告；初始化失败回退纯 GPU 不致命。
        if backend == 'hybrid' and label == 'GPU':
            # ── decord 原生混合解码（fork ≥ v0.7.15，2026-09）──────────
            # 单 demux 流在解码器内部按关键帧 chunk 路由 CPU 软解 + NVDEC，
            # 引擎只需把 ctx 换成 hybrid/hybrid_gpu，参数面（output_format /
            # roi / num_threads）与 cpu/gpu 完全一致 —— 零适配透传：
            #   hybrid_gpu：输出帧驻留显存（GPU chunk 零拷贝、CPU chunk 解码
            #     器内部 H2D 上载），get_batch 返回 CUDA 批 —— gpu_pipeline 的
            #     设备指针通路（_ndarray_device_ptr）直接可用，OCR on GPU（TRT）
            #     时选它；
            #   hybrid：输出宿主帧（与 cpu() 同布局），OCR on CPU / 宿主管线
            #     时选它。
            # 旧 decord（无 hybrid ctx）回退项目层 HybridDecoder 壳（v3~v7）。
            try:
                from decord import hybrid as _hy, hybrid_gpu as _hyg
                _ct = config.env_int(config.HYBRID_CPU_THREADS_ENV, 0)
                if _ct <= 0:
                    # 延续项目层 hybrid 的 CPU 线程分档（见下方 v5 注释的历史
                    # 依据）：核数 3/8，钳 [MIN, MAX]。av1 例外：fork 0.8.1
                    # (FFmpeg9) 的 dav1d 帧线程扩展到 ~逻辑核 3/4 才饱和
                    # （NT16 797fps → NT24 1164fps → NT32 1178fps），沿用手持
                    # 策略会让 hybrid 的 CPU 分片喂不满（见 _decode_num_threads
                    # av1 分支的实测）。
                    if self._codec == 'av1':
                        _ct = self._decode_num_threads(codec='av1') or _ct
                    elif self._decord_has_hybrid_preroute():
                        # fork ≥0.8.3（chunk 预路由）下 decode-only nt16 比
                        # nt12 高：hevc 2907 vs 2771、h264 2692 vs 2412
                        # （2026-09-10 路线图轮，`_probe_roadmap_decode`）。
                        # 0.8.2 同 nt 反向（hevc 2344→2192 回退），按版本门控。
                        _ct = max(config.HYBRID_CPU_THREADS_AUTO_MIN,
                                  min((_os.cpu_count() or 8) // 2,
                                      config.HYBRID_CPU_THREADS_AUTO_MAX))
                    else:
                        _ct = max(config.HYBRID_CPU_THREADS_AUTO_MIN,
                                  min((_os.cpu_count() or 8) * 3 // 8,
                                      config.HYBRID_CPU_THREADS_AUTO_MAX))
                _hctx = _hyg(0) if self._ocr_on_gpu() else _hy(0)
                vr = self._open_decord_reader(_hctx, roi_kw, num_threads=_ct)
                self._backend = 'decord/hybrid'
                logger.info('混合解码开启(原生): codec=%s ctx=%s cpuT=%d',
                            self._codec,
                            'hybrid_gpu' if self._ocr_on_gpu() else 'hybrid',
                            _ct)
            except Exception as e:  # noqa: BLE001
                self._degraded.append(f'hybrid 打开失败，回退纯 GPU: {e}')
                logger.warning('原生混合解码打开失败，回退纯 GPU: %s', e)
        return vr

    def _decord_has_hybrid_preroute(self) -> bool:
        """decord ≥0.8.3（hybrid chunk 预路由 + 池深修复）判定。

        按 `decord.__version__` 门控（注意 DECORD_LIBRARY_PATH 开发态
        换 dll 不改 Python 包版本号——此时判 False 走旧档位，可用
        HYBRID_CPU_THREADS 显式覆盖）。
        """
        try:
            import decord as _d
            parts = []
            for p in str(getattr(_d, '__version__', '')).split('.'):
                digits = ''.join(ch for ch in p if ch.isdigit())
                if not digits:
                    break
                parts.append(int(digits))
                if len(parts) == 3:
                    break
            return tuple(parts) >= (0, 8, 3)
        except Exception:  # noqa: BLE001 版本不可知 → 保守走旧档位
            return False

    def _decord_format(self) -> str:
        """当前管线请求的 decord output_format。

        内部链永远只消费单通道（Y 平面 / decord gray，不再输出 RGB）：
        - keep_crops 需要 YUV 代表帧 → 'yuv420'（packed NV12；内部取 Y 平面，
          等价灰度，另保留 UV 供外部 nv12_to_rgb）
        - 否则 'gray'
        """
        return 'yuv420' if self._yuv_output else 'gray'

    def _ocr_on_gpu(self) -> bool:
        """OCR 推理是否卸载到 GPU（TensorRT）。

            为 True 时 host CPU 在解码阶段基本空闲（TRT 只占少量提交线程），
            解码可以放宽线程预算（见 _decode_num_threads）。
            仅按配置判断，不表示 TRT 一定可用（不可用时 OcrEngine 内部回退
            ONNX，此时解码线程偏多只是轻微过订阅，实测不劣化）。
            """
        return (self._ocr_backend or 'auto').lower() != 'cpu'

    def _run_pipelined_gpu(self, _ocr_engines: list | None = None):
        """GPU 全驻留管线门面（S3-3c）：构建显式 GpuRunSpec → 调用
        pipeline.gpu_backend.run_gpu_pipeline → 同步结果回实例。

        驱动主体已迁入 gpu_backend（B4 autocropper / B5 y_pool 构造注入）；
        形状不符回退经 fallback_to_host 回调复用已打开的 reader（C10）。"""
        self._gpu_pipeline_mode = True   # 会话启动前置位（F-5）
        spec = GpuRunSpec(
            frame_start=self._frame_start, frame_end=self._frame_end,
            sample_stride=self._sample_stride, roi=tuple(self._roi),
            buffer_size=self._buffer_size,
            C=self._C, merge_similar=self._merge_similar,
            merge_similar_threshold=self._merge_similar_threshold,
            merge_max_changed_pixels=self._merge_max_changed_pixels,
            keep_crops=self._keep_crops, yuv_output=self._yuv_output,
            color_range=self._color_range, ocr_autocrop=self._ocr_autocrop,
            bin_thresh_ref=[self._bin_thresh],
            backend_label=lambda: self._backend,
            ocr_on_gpu=self._ocr_on_gpu,
            merge_effective_mode=self._merge_effective_mode,
            content_range_to_crop=self._content_range_to_crop,
            open_vr=self._open_vr,
            start_ocr_session=self._start_ocr_session,
            batch_luma=self._batch_luma,
            fallback_to_host=lambda engines, vr: self._run_pipelined_host(
                engines, vr),
            on_degraded=self._degraded.append,
            progress=self._progress, cancel=self._cancel,
            prof_end=self._prof_end,
            on_bin_thresh=self._set_bin_thresh,
            fps_box=[self._fps])
        res = run_gpu_pipeline(spec, _ocr_engines)
        if res.fell_back_to_host:
            self._degraded.append('GPU 管线形状不符，回退宿主管线')
            return self._run_pipelined_host(res.fallback_engines)
        self._fps = spec.fps_box[0]
        self._bin_thresh = res.bin_thresh
        self.timing.update(res.timing)
        self._n_segments = res.n_segments
        self.crops = res.crops
        self._ocr_texts = res.texts
        self._ocr_confs = res.confs
        return res.as_tuple()

    def _decode_num_threads(self, codec: str | None=None) -> int | None:
        """CPU 软解的 decord FFmpeg 帧线程数（按 OCR 是否在 GPU 分档）。

            ── OCR 在 GPU（TRT，现役默认）────────────────────────────
            host CPU 空闲 → 解码吃满逻辑核（上下限见
            config.DECODE_THREADS_GPU_OCR_MIN/MAX）。
            背景：fork 的默认线程上限（DECORD_FFMPEG_THREAD_COUNT，随版本
            变化）是"ONNX 占满物理核"时代定的；TRT 成默认后 host 在解码
            阶段基本空闲，旧上限成为瓶颈。实测（7945HX 16C32T + RTX 4060，
            test5 1080p h264 全片
            7223 帧，TRT）：8 线程 6.452s → 16 线程 4.875s（-24%）→ 32 线程
            5.085s；新三国01 标清整集 73430 源帧 stride8：8 线程 15.897s →
            32 线程 10.812s（-32%）。相对现役默认（NVDEC+TRT）为 -45%/-50%。
            绑核 8 逻辑核模拟弱 CPU 时不劣化（1080p -6%、标清 -35%）。

            ── OCR 在 CPU（ONNX，无 NVIDIA 显卡场景）───────────────
            解码与 ORT 真正抢核，但**不是"越少越好"**——取决于解码与 OCR
            谁占墙钟，而段密度决定这一点：
              · 物理核 ≤ CPU_CORES_SPLIT_THRESHOLD（8）：max(2, cores//2)
                （4 核 28.0 vs 33.1s、8 核 17.8 vs 20.7s）
              · 多核 + stride>1（解码受限）：逻辑核 3/4，钳 [8, 24]
              · 多核 + stride==1（OCR 受限）：逻辑核 1/3，钳 [8, 12]
            判据：stride>1 时采样帧数 ÷ stride 而解码帧数不变 → 解码占比
            必然上升；stride==1 时段数可接近采样帧数 → OCR 占比上升。
            实测（16C32T，test5 3000 帧，decode=cpu ocr=cpu）：
              stride=8（339 段）  dcd 8 → 2.841s，24 → 2.026s（-27.8%）
              stride=1（1083 段） dcd 8 → 3.746s，10 → 3.617s，16 → 3.811s
            （旧实现的"多核返回 None"让解码一直跑 fork 默认 8 线程，
            低段密度场景白丢 ~28%；且它引用的"加线程变慢"实测来自 OCR
            受限的高段密度场景，被错误地当成了普适结论。）

            codec='av1'：fork 0.8.1 (FFmpeg9) 起 dav1d 帧线程扩展性改善
            （FFmpeg8 时代"不随 FFmpeg 帧线程数扩展"的旧实测是被 OCR 墙钟
            掩盖的口径）→ 逻辑核 3/4 钳 [8, 24]，不分 OCR 位置。
            GPU(NVDEC) 不调用本方法。

            DECODE_THREADS env 覆盖（>0 时直接返回，与 OCR_THREADS 对齐）：
            调参与 A/B 用；不设置时行为与上述分档一致。
            """
        _ovr = config.env_int(config.DECODE_THREADS_ENV, 0)
        if _ovr > 0:
            return _ovr
        from ocr_native import auto_ocr_thread_count
        cores = auto_ocr_thread_count()
        logical = _os.cpu_count() or cores
        if codec == 'av1':
            # fork 0.8.1 (FFmpeg9) 复测：dav1d 帧线程扩展性大幅改善，
            # 旧结论（8/16/24/32 线程全 5.8~5.9s，FFmpeg8 口径且被 OCR
            # 墙钟掩盖）已过时。顺序解码（stride=1）16T 797fps →
            # 24T 1164fps → 32T 1178fps（饱和）→ ONNX 墙钟 24T 最优
            # （3.958s，32T 持平）；stride=8 等差快速路径扩展到 48T
            # （2.536s vs 24T 2.880s，48T 后饱和；2026-09-10 实测表）。
            if self._sample_stride > 1:
                return max(8, min(48, logical))
            return max(8, min(24, logical * 3 // 4))
        if codec == 'hevc':
            # FFmpeg9 hevc 软解扩展性同样大幅改善（2026-09-10 ONNX 墙钟
            # 实测表，test.mp4 3000 帧窗口）：stride=1 8T 4.748 → 16T
            # 3.763 → 24T 3.520 → 32T 3.110（48T 3.240 回落）；stride=8
            # 16T 2.757 → 32T 2.289 → 48T 2.219（渐近）。旧通用档位
            # （10/24）分别慢 27%/12%。
            if self._sample_stride > 1:
                return max(8, min(48, logical))
            return max(8, min(32, logical))
        if self._ocr_on_gpu():
            return max(config.DECODE_THREADS_GPU_OCR_MIN,
                       min(config.DECODE_THREADS_GPU_OCR_MAX, logical))
        if cores <= config.CPU_CORES_SPLIT_THRESHOLD:
            return max(2, cores // 2)
        if self._sample_stride > 1:
            return max(8, min(config.DECODE_THREADS_CPU_OCR_MAX,
                              logical * 3 // 4))
        return max(8, min(config.DECODE_THREADS_CPU_OCR_STRIDE1_MAX,
                          logical // 3))

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
            self._degraded.append('decord 不支持 yuv420 输出，代表帧退化灰度')
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
        _env = config.env_int(config.OCR_THREADS_ENV, 0)
        if _env:
            return max(1, _env)
        cores = auto_ocr_thread_count()
        if getattr(self, '_codec', '') == 'av1' and getattr(self, '_backend', '').startswith('decord/CPU'):
            return max(2, cores // 2)
        if getattr(self, '_backend', '').startswith('decord/CPU') and cores <= config.CPU_CORES_SPLIT_THRESHOLD:
            return max(2, cores // 2)
        return cores

    def _run_pipelined(self, _ocr_engines: list | None = None):
        """入口分发：GPU 全驻留管线（_run_pipelined_gpu）或宿主管线。

        _ocr_engines 两条路径都透传（B5：GPU 路径此前丢弃该参数）；
        None = 从进程级 OCR 引擎池取（ocr_native.acquire_ocr_engine）。

        S3-2：可经 VOE_V2_ENGINE=1 走 SegmentEngine 编排（当前引擎内
        部仍委托本类的 legacy 双驱动器，双跑对账用；S3-3 迁移内部）。"""
        if SegmentEngine.legacy_engine_requested():
            return SegmentEngine(
                _LegacyBackend(self, _ocr_engines)).run().as_tuple()
        return self._run_pipelined_legacy(_ocr_engines)

    def _run_pipelined_legacy(self, _ocr_engines: list | None = None):
        """v1 双驱动器分派（S3-3 的迁移对象）。"""
        if self._gpu_pipeline_enabled():
            return self._run_pipelined_gpu(_ocr_engines)
        return self._run_pipelined_host(_ocr_engines)

    def _run_pipelined_host(self, _ocr_engines: list | None = None,
                            _preopened_vr=None):
        """宿主管线门面（S3-3a）：构建显式 HostRunSpec → 调用
        pipeline.host_backend.run_host_pipeline → 同步结果回实例。

        驱动主体与三个协作函数已迁入 host_backend（算法代码只读 spec
        声明字段，P0-1 的宿主侧私有属性穿透至此消除）。
        """
        self._gpu_pipeline_mode = False
        spec = HostRunSpec(
            frame_start=self._frame_start, frame_end=self._frame_end,
            sample_stride=self._sample_stride, roi=tuple(self._roi),
            C=self._C, merge_similar=self._merge_similar,
            keep_crops=self._keep_crops, yuv_output=self._yuv_output,
            segments_similar=self._segments_similar,
            crop_luma=self._crop_luma, batch_luma=self._batch_luma,
            batch_luma_out=self._batch_luma_out,
            crop_is_expected=self._crop_is_expected,
            open_vr=self._open_vr,
            start_ocr_session=self._start_ocr_session,
            backend_label=lambda: self._backend,
            progress=self._progress, cancel=self._cancel,
            prof_end=self._prof_end,
            on_bin_thresh=self._set_bin_thresh,
            fps_box=[self._fps])
        if _preopened_vr is not None:
            res = run_host_pipeline(spec, _ocr_engines,
                                    preopened_vr=_preopened_vr)
        else:
            res = run_host_pipeline(spec, _ocr_engines)
        # 同步回实例（B2：fps 缓存经 box 读写；其余为本次 run 的输出）
        self._fps = spec.fps_box[0]
        self._bin_thresh = res.bin_thresh
        self.timing.update(res.timing)
        self._n_segments = res.n_segments
        self.crops = res.crops
        self._ocr_texts = res.texts
        self._ocr_confs = res.confs
        return res.as_tuple()
