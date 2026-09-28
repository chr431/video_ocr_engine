"""FieldExtractor — 通用视频文本提取引擎（识别链：解码∥像素分段∥OCR 文本）。

引擎只输出每段原始文本与置信度；速度解析/纠错/CSV 等领域后处理由上层
应用完成（引擎保持通用性，不携带任何下游领域后处理）。

方法体最初由既有视频项目的历史 tools/archive 生成脚本从 segment_flow.py
抽取；独立成仓后随引擎维护，不再依赖任何下游仓库。

模块划分（S9 模块化 + R1 门面拆解后的现行布局；旧扁平面
`_host_pipeline.py` / `_gpu_pipeline.py` / `_ocr_session.py` 均已删除）：
  extractor.py  — 薄门面：构造/参数校验/run 态重置/结果组装；
                  实现一律下放（本文件只保留委托与状态同步）
  decode/       — port=FrameSource 协议；decord_source=打开决策树 +
                  输出格式适配（R1 落地的 decord 适配器）
  pipeline/     — engine=SegmentEngine 唯一编排（EngineInputs 显式
                  run 契约，注入式，R1 起不再反向调门面 _run_*）；
                  host_backend / gpu_backend 两个执行后端
                  （HostRunSpec / GpuRunSpec 显式契约）；policy=后端
                  选择纯函数；ocr_stage=OcrSession（吃 SessionSpec）；
                  report=RunReport 组装；_driver=生命周期单出处
  domain/       — segmentation=分段/状态机/裁切/预处理的**唯一实现处**；
                  prof=计时脊柱（ProfSpine）；video_utils=像素转换；
                  metrics=指标注册表；resources=资源层
  ocr/          — native=OCR 调度+引擎池；trt=TRT 执行；port=后端协议
  gpu/          — context=DLL 注册 / device=设备池+帧流+校准 / frame_ref=DeviceRef
  config/       — constants + 旋钮注册表/解析（env 读取唯一入口）
  _helpers.py / _result_types.py / _gpu_kernels.py — 工具 / 轨迹类型 / 设备侧核
双流水线并行已被移除（2026-08 清理）；CPU+NVDEC 双解码（decode_backend=
"hybrid"）由 decord fork 原生实现（≥v0.7.15 的 hybrid/hybrid_gpu ctx），
引擎只透传解码参数；项目层 hybrid_decode.py 已删除，勿再引用。
"""
import logging
import os as _os
import time
from pathlib import Path

import numpy as np

from video_ocr_engine.config import constants as config
from video_ocr_engine.domain.segmentation import _text_sep_binary
# 下列 re-export 为引擎内部结构（_helpers/_result_types 均属下划线私有
# 命名，从 extractor 再导出仅为旧导入路径兼容，勿直接 import；公共入口
# 是 video_ocr_engine.__init__ 的三件套）。
from ._result_types import (  # noqa: F401
    ExtractedSegment, ExtractionResult,
)
from ._helpers import (  # noqa: F401
    _ocr_batch_size, _ndarray_device_ptr,
    _decode_progress_pct, _ocr_progress_pct,
    _read_fps_from_vr,
)
from .config import resolve
from .decode.decord_source import DecordFrameSource, ensure_roi_capable_decoder
from .domain.metrics import NULL_METRICS, make_metrics
from .domain.prof import ProfSpine
from .pipeline.engine import EngineInputs, SegmentEngine
from .pipeline.policy import (
    gpu_pipeline_enabled, merge_effective_mode,
    ocr_engine_type, ocr_num_threads, ocr_on_gpu,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 仅为 `_start_ocr_session` 的返回注解服务（运行期懒导入，避免
    # extractor → ocr_stage 的导入环；此前注解裸引 "OcrSession" 是
    # F821，lint 基线轮修正）
    from .pipeline.ocr_stage import OcrSession
from .domain.diagnostics import NULL_DIAG, open_diagnostics
from .pipeline.report import finalized_report, write_report_file

logger = logging.getLogger(__name__)


class FieldExtractor:
    """从视频固定区域提取文本的通用引擎（识别链：解码∥分段∥OCR）。

    构造参数：
      常用 —— video_path / roi / frame_start / frame_end / force_aspect /
      decode_backend(auto|cpu|nvdec|hybrid) /
      ocr_backend(auto|cpu|tensorrt|hybrid=双车道 TRT+OpenVINO) /
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
                 frame_end=None,
                 force_aspect: float = config.DEFAULT_FORCE_ASPECT,
                 decode_backend: str = config.DEFAULT_DECODE_BACKEND,
                 ocr_backend: str = config.DEFAULT_OCR_BACKEND,
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
        # S9-2（D6/Q5）：配置构造期一次解析并冻结（resolve 为唯一 env
        # 读取点；构造后改 env 不再生效——v1 README 曾承诺"仍生效"，
        # 属有意行为变更，见 docs/MIGRATION.md）。优先级：显式参数 >
        # env > 默认（Q5）；v1 逃生门 VOE_ENV_WINS 已于 0.16.0 删除。
        self._rc = resolve(env=_os.environ)
        _pad_env = self._rc.ocr_pad_small
        if fill_width is not None:
            self._fill_width = fill_width          # Q5：显式参数锁定
            self._pad_floor_env = 0
        else:
            self._fill_width = config.DEFAULT_FILL_WIDTH
            self._pad_floor_env = _pad_env          # 参数缺省 → env 抬升下限
        self._merge_text_sep_resolved = (merge_text_sep if merge_text_sep is not None
                                         else self._rc.segment_text_sep_merge)
        self._merge_text_sep = self._merge_text_sep_resolved
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
        # S9-2(D6)：merge_text_sep 构造期解析（参数 > env > 默认）；
        # 早前 :120 的赋值是初值，此处为最终冻结值
        self._merge_text_sep = self._merge_text_sep_resolved
        self._color_range = 0            # run 时从 decoder get_color_range 读取
        self._codec = ""                 # run 时从 decoder get_codec 探测
        self._backend = ""
        self._bin_thresh = 0
        self._degraded: list = []        # 本次提取的降级/回退原因（D3，meta 透出）
        # OCR 输入宽度自适应裁切（宽 ROI 字幕省卷积）+ 跨批按宽度分组。
        # 详见 segmentation.content_range_to_crop 与 config 中的实测注释。
        # 四个 autocrop/重排旋钮为 property 调用期读 env（A6：与
        # OCR_PAD_SMALL/OCR_GAMMA 等同时机，构造后改 env 即生效）。
        self._progress = progress_cb or (lambda m, p: None)
        self._cancel = cancel_check or (lambda: None)
        # R2（0.16.0）：run 态全部私有（ex.timing/crops/frames 兼容副产物
        # 已删除，读面 = ExtractionResult：timing / frames / segments[*].rep_crop）。
        # profile 例外保留公开：ENGINE_PROFILE=1 的唯一读面（result 不携带）。
        self._timing: dict = {}
        self._crops: dict = {}
        self._frames: list = []
        self._ocr_texts: list = []
        self._ocr_confs: list = []
        self._n_segments = 0
        self._profile_enabled = self._rc.diag_profile
        # R1：计时脊柱（domain/prof.ProfSpine）——extract() 每 run 重建；
        # 此处的初版仅供 extract 之前的 _prof_end 偶发调用（NULL_METRICS）。
        # profile 读面 = 只读 property（ENGINE_PROFILE 唯一出口）。
        self._spine = ProfSpine(self._profile_enabled, NULL_METRICS)
        # F-5：GPU lane 前置位单元素盒（引擎 lane 启动写、SessionSpec 与
        # 报告组装读；兼容读面见 _gpu_pipeline_mode property）
        self._gpu_mode_box = [False]
        # R1：解码源适配器（open 前无资源；_open_vr 每次重建并同步状态）
        self._source = self._make_source()
        # S6-0（§8.6 N-2/N-3）：每次 run 新建 Metrics 与报告（B1 同类重置）。
        # telemetry=off → NULL_METRICS 单例，全链路空调用、不组装报告。
        self._metrics = NULL_METRICS
        self._diag = NULL_DIAG
        self._report: dict = {}
        self._trace = None
        self._validate_params()
        ensure_roi_capable_decoder()
        roi_w = max(1, self._roi[2] - self._roi[0] + 1)
        roi_h = max(1, self._roi[3] - self._roi[1] + 1)
        self._merge_max_changed_pixels = max(
            32, int(roi_w * roi_h * config.SEG_MERGE_MAX_CHANGED_RATIO))
        # 稠密簇门（segment.merge_dense_gate）：0=关；>0=差异图 win3 阈值。
        # 构造期一次冻结（D6）；与断段判据 SEG_C 共用同一"内容变了"定义。
        self._merge_dense_gate = max(0, int(self._rc.segment_merge_dense_gate))
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
        if _ocr not in ('auto', 'cpu', 'tensorrt', 'hybrid'):
            raise ValueError(
                f"ocr_backend 必须为 auto/cpu/tensorrt/hybrid，"
                f"收到 {self._ocr_backend!r}")


    # ── env 旋钮统一调用期读取（A6：与 OCR_PAD_SMALL/OCR_GAMMA 等同时机，
    #    构造后改 env 即生效；此前 autocrop 四项在构造期烘焙，语义不一致）──
    @property
    def _ocr_autocrop(self) -> bool:
        return self._rc.ocr_roi_autocrop

    @property
    def _ocr_autocrop_margin_pct(self) -> int:
        return self._rc.ocr_roi_autocrop_margin

    @property
    def _ocr_autocrop_min_gain(self) -> float:
        # 最小收益门槛：裁掉比例低于此值就整段不裁（紧凑 ROI 自动不裁，
        # 避免在几乎没留白的段上承担切笔画的风险）。见 config 中的实测表。
        return max(0, self._rc.ocr_roi_autocrop_min_gain) / 100.0

    @property
    def _ocr_reorder_window(self) -> int:
        return max(1, self._rc.ocr_reorder_window)

    def _merge_effective_mode(self) -> str:
        """分离模式归一（实现：pipeline/policy.merge_effective_mode，
        env 钩子优先级与 _segments_similar 一致）：'binary' | ''。"""
        return merge_effective_mode(self._merge_text_sep)

    # ═══════════════ OCR 输入宽度自适应裁切 ═══════════════
    # 统一实现（含余量/最小收益门槛的实测依据 docstring）在
    # segmentation.content_range_to_crop / crop_to_content / crop_after_aspect；
    # GPU 直通（_autocrop_device）与宿主预处理共用同一余量数学。

    def _content_range_to_crop(self, first: int, last: int, w: int):
        """「有墨迹列范围」→ 裁切区间；实现见 segmentation.content_range_to_crop。"""
        from video_ocr_engine.domain.segmentation import content_range_to_crop
        return content_range_to_crop(
            first, last, w,
            margin_pct=self._ocr_autocrop_margin_pct,
            min_gain=self._ocr_autocrop_min_gain)

    def _crop_to_content(self, crop):
        """按二值图裁掉两侧空白（fa=0 路径）；实现见
        segmentation.crop_to_content（fa>0 走 _crop_after_aspect 顺序⑦）。"""
        from video_ocr_engine.domain.segmentation import crop_to_content
        return crop_to_content(
            crop, self._bin_thresh,
            autocrop=self._ocr_autocrop,
            force_aspect=float(getattr(self, '_force_aspect', 0) or 0.0),
            margin_pct=self._ocr_autocrop_margin_pct,
            min_gain=self._ocr_autocrop_min_gain)

    def _crop_after_aspect(self, img):
        """已定比例图上再按内容列裁（fa>0 路径）；实现见
        segmentation.crop_after_aspect（阈值现算 Otsu，不能用校准阈值）。"""
        from video_ocr_engine.domain.segmentation import crop_after_aspect
        return crop_after_aspect(
            img, autocrop=self._ocr_autocrop,
            margin_pct=self._ocr_autocrop_margin_pct,
            min_gain=self._ocr_autocrop_min_gain)

    def _start_ocr_session(self, _ocr_engines: list | None = None) -> "OcrSession":
        """启动 OCR 消费会话（S3-3b：构建显式 SessionSpec）。

        会话不再读本实例的私有属性；跨线程回写经输出挂钩（可见性由
        finish() join 建立，§9 契约）。一次 extract() 一个实例，
        宿主/GPU 两管线共用。
        """
        from .pipeline.ocr_stage import OcrSession
        return OcrSession(self._build_session_spec(), _ocr_engines)

    def _build_session_spec(self):
        """纯构建（无线程/无引擎）——S9-2 供会话与测试共用。"""
        s = self
        from .pipeline.ocr_stage import SessionSpec, effective_reorder_window
        from .ocr.native import ocr_pad_floor
        _x1, _y1, _x2, _y2 = s._roi
        _roi_w = max(1, _x2 - _x1 + 1)
        _roi_h = max(1, _y2 - _y1 + 1)
        _force_aspect = float(getattr(s, '_force_aspect', 0) or 0.0)
        # S6-f：pad 下限支配时按宽分组不可能有收益 → 窗口收敛到 1（唯一出处
        # 见 ocr_stage.effective_reorder_window 的判据说明）
        _floor = ocr_pad_floor(s._ocr_model, s._fill_width, s._pad_floor_env)
        _window = effective_reorder_window(
            _roi_w, _roi_h, _floor, _force_aspect, s._ocr_reorder_window)
        return SessionSpec(
            buffer_size=s._buffer_size,
            model=s._ocr_model,
            fill_width=s._fill_width,
            force_aspect=_force_aspect,
            reorder_window=_window,
            yuv_output=s._yuv_output,
            color_range=s._color_range,
            gpu_pipeline_mode=s._gpu_mode_box[0],
            num_threads_fn=s._ocr_num_threads,
            engine_type_fn=s._ocr_engine_type,
            crop_to_content=s._crop_to_content,
            crop_after_aspect=s._crop_after_aspect,
            prof_end=s._prof_end,
            progress=s._progress,
            cancel=s._cancel,
            on_backend_used=lambda v: setattr(s, '_ocr_backend_used', v),
            on_degraded=s._degraded.append,
            gamma=s._rc.ocr_gamma,
            ocr_batch=s._rc.ocr_batch,
            ocr_instances=s._rc.ocr_instances,
            gpu_ctc=s._rc.ocr_gpu_ctc,
            pad_floor_env=s._pad_floor_env,
            metrics=s._metrics,
        )

    def _set_bin_thresh(self, th: int) -> None:
        """校准阈值即时回写（F-4：合并判定在流式期间活读本值，
        回写延迟到 run 结束会使宿主路径段数漂移 1042 vs 1083 类复发）。"""
        self._bin_thresh = th

    def _segments_similar(self, a, b) -> bool:
        """相似段判定：平均绝对差小 且 显著变化像素占比也小。

        只用平均绝对差会把宽 ROI 中的单字短字幕（如“在”“不”）误判为噪声：
        大部分区域未变，均值被稀释。因此额外限制 abs(diff)>10 的像素数。
        分离模式由 _merge_effective_mode 决定（binary 为引擎默认）。
        （R1 留在门面：探针对本方法有类级 patch 面；判定原语唯一出处
        仍是 segmentation.py，C-32。）
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
        from video_ocr_engine.domain.segmentation import (dense_gate_hit,
                                                          similar_decision,
                                                          _cluster_win3)
        # 稠密簇门：diff>10 与 changed_px 同一谓词（binary 模式下 diff∈{0,255}
        # 与阈值穿越逐位等价）。win3 ≥ 分段阈值 C 的稠密簇 = 笔画级变化，
        # 与断段判据自洽（同一 win3 定义“内容变了”），恒不合并。
        _dense = dense_gate_hit(_cluster_win3(diff > 10),
                                self._merge_dense_gate)
        return similar_decision(float(diff.mean()), int(np.sum(diff > 10)),
                                self._merge_similar_threshold,
                                self._merge_max_changed_pixels,
                                dense=_dense)

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
        self._timing = {}
        self._crops = {}
        self._frames = []
        self._ocr_texts = []
        self._ocr_confs = []
        self._n_segments = 0
        self._backend = ""
        self._ocr_backend_used = ""
        self._bin_thresh = 0
        # S6-0：RunReport 的一次 run 生命周期（§8.6 N-3）
        self._metrics = make_metrics(self._rc.diag_telemetry)
        # 自诊断：挂在既有 VOE_REPORT_FILE opt-in 上（不新增旋钮）。
        # 未 opt-in → NULL_DIAG，下面 tick 是一次属性判断即返回。
        self._diag = open_diagnostics(self._rc.diag_report_file,
                                      metrics=self._metrics)
        self._report = {}
        self._hardware = None
        self._trace = None
        self._fork_stats = None
        _t_run = time.perf_counter()
        # L2 设备峰值采样：**仅 full 档**建采样线程（B6——std/off 这里是一次
        # 属性比较即返回），run 结束立刻停并丢弃半帧。
        self._metrics.start_hardware()
        # P4 实验性全事件时间线（VOE_TRACE_FILE，默认关）：off 档 + trace
        # 同设 → 警告并忽略（trace 事件源就是遥测脊柱，off 档脊柱不产名）。
        _tf = (self._rc.diag_trace_file or "").strip()
        if _tf:
            if self._metrics.enabled:
                from .domain.trace import TraceRecorder
                self._trace = TraceRecorder()
                self._trace.start_hardware()
            else:
                logger.warning("VOE_TRACE_FILE 需 VOE_TELEMETRY != off，已忽略")
        # R1：计时脊柱（原 _profile_enabled/_prof_lock/_metric_totals/
        # _metric_max 的 B1 重置至此一并完成；profile 读面 = 派生 property）
        self._spine = ProfSpine(self._profile_enabled, self._metrics,
                                self._trace, self._diag)
        try:
            _outcome = self._run_pipelined()   # S9-6：RunOutcome（裸 5 元组已退场）
        finally:
            self._hardware = self._metrics.hardware_report()
            # 诊断收尾（2026-09-19 审查轮）：此前 _diag.stop() 只在成功
            # 路径的 _assemble_report 里调用——失败路径看门狗线程泄漏
            # （每 30s 写 stall JSON 并继续排干指标桶）。stop 幂等，
            # 成功路径再调一次返回 {}。
            if self._diag.armed:
                try:
                    self._diag.stop()
                except Exception:  # noqa: BLE001
                    logger.debug("诊断收尾失败（run 失败路径）",
                                 exc_info=True)
        _wall = time.perf_counter() - _t_run
        frames, segs, texts, confs, rep_frames = _outcome.as_tuple()
        self._frames = frames
        segments = [
            ExtractedSegment(
                start=seg[0], end=seg[-1],
                frames=tuple(seg) if self._keep_frames else (),
                rep_frame=rep_frames[i],
                text=texts[i] if i < len(texts) else None,
                confidence=confs[i] if i < len(confs) else 0.0,
                rep_crop=(self._crops.get(rep_frames[i])
                          if self._keep_crops else None))
            for i, seg in enumerate(segs)
        ]
        meta = self._build_meta(segments, _wall)
        if self._trace is not None:
            # 收尾：顶层 run 事件（真实锚点）+ 落盘（含 NVML 点列）
            self._trace.record_run(_t_run, _wall)
            _info = self._trace.dump(
                self._rc.diag_trace_file, meta=meta, wall=_wall)
            logger.debug("trace 时间线已落盘：%s", _info)
        return ExtractionResult(
            segments=segments,
            frames=frames if self._keep_frames else [],
            fps=self._fps or 0.0,
            timing=dict(self._timing),
            meta=meta)

    def _build_meta(self, segments: list, wall: float) -> dict:
        """meta 组装（§10.1 的 9 键逐字不变 + S6-0 增补 report，只增不改）。"""
        meta = {"backend": self._backend,
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
                    "merge_dense_gate": self._merge_dense_gate,
                    "merge_text_sep": self._merge_effective_mode(),
                    "buffer_size": self._buffer_size,
                    "C": self._C}}
        report = self._assemble_report(wall, len(segments))
        if report:
            meta["report"] = report                     # §8.6 N-3：只增不改
            _rf = (self._rc.diag_report_file or "").strip()
            if _rf:
                try:
                    write_report_file(report, _rf)
                except OSError:
                    # 写 sidecar 是显式 opt-in 的副作用，失败不该毁掉 run
                    logger.warning("RunReport sidecar 写入失败: %s", _rf,
                                   exc_info=True)
        return meta

    def _assemble_report(self, wall: float, n_segments: int) -> dict:
        """把单次 run 的观测收敛成 RunReport（实现：pipeline/report.
        finalized_report；off 档返回 {}）。"""
        rep = finalized_report(
            self._metrics, self._spine, self._timing, self._trace,
            self._diag, self._degraded,
            wall=wall, n_segments=n_segments,
            backend=self._backend, ocr_backend=self._ocr_backend_used,
            hardware=self._hardware, fork_stats=self._fork_stats,
            gpu_mode=self._gpu_mode_box[0],
            config_digest=self._rc.config_digest)
        self._report = rep
        return rep

    @property
    def profile(self) -> dict:
        """ENGINE_PROFILE=1 的 13 相位原始字典（run 后有效）。

        R2（0.16.0）：这是 ENGINE_PROFILE 的**唯一读面**（result 不携带）；
        遥测口径（std/full 档）请读 result.report（RunReport 类型化视图）。
        """
        return self._spine.profile

    def warmup(self) -> int:
        """显式预热 OCR 引擎池（v2 §7.5 P-b / S6-e）。

        把冷启动的一次性成本（TRT 反序列化 + 建上下文 **0.39–0.44s**，实测
        `ocr.engine_init`）从"首个视频的关键路径"挪到调用方显式选择的时刻——
        批量循环、UI 首帧、需要稳定首段延迟的场景。

        与 `extract()` 用**完全相同**的池 key（同源 `_build_session_spec`，
        PI-10 key 稳定性），因此随后的 extract 命中热池（实测首个 extract
        的 `ocr.engine_init` 从 0.39s 降到 ~0.0001s）。

        注意 C-24 的边界：**不改变任何 run 的总吞吐**（预热只是把成本提前）；
        本方法不改变 `extract()` 内部行为，也不自动触发。返回预热的引擎数
        （0 = 需要时再建，例如非 TRT/ONNX 双实例路径）。
        """
        spec = self._build_session_spec()
        from .pipeline.ocr_stage import warmup_engines
        t0 = time.perf_counter()
        n = warmup_engines(spec)
        logger.debug("OCR 引擎池预热完成：%d 个，耗时 %.3fs",
                     n, time.perf_counter() - t0)
        return n

    def _prof_end(self, group: str, key: str, t0: float) -> None:
        """单一计时脊柱（§8.6 N-2）的薄委托（实现：domain/prof.ProfSpine）。

        同一 t0 同时喂 profile 与指标；两档都关时只做两次属性判断——
        PI-15 的 off 档"一行关闭"由 spine 兑现。
        """
        self._spine.end(group, key, t0)

    # ── 解码器打开（决策树实现：decode/decord_source.DecordFrameSource）──
    def _make_source(self) -> DecordFrameSource:
        """构建解码源适配器（open 前无资源；每 run 由 _open_vr 重建）。"""
        return DecordFrameSource(
            self._video_path, self._roi, self._decode_backend,
            yuv_output=self._yuv_output, degraded=self._degraded,
            frame_start=self._frame_start, frame_end=self._frame_end,
            decode_threads_fn=self._decode_num_threads,
            hybrid_cpu_threads=self._rc.decode_hybrid_cpu_threads,
            ocr_on_gpu_fn=self._ocr_on_gpu)

    def _open_vr(self):
        """按 decode_backend 打开解码器（auto/cpu/nvdec/hybrid）。

        决策树与降级链在 DecordFrameSource.open（decode/decord_source）；
        返回 reader 本体（driver 消费面）。open 后同步四个门面读点
        （_backend/_codec/_color_range/_yuv_output——yuv 不可用回退 gray
        时源内翻转）。
        """
        src = self._make_source()
        self._source = src
        vr = src.open()
        self._backend = src.backend_label
        self._codec = src.codec
        self._color_range = src.color_range
        self._yuv_output = src.yuv_output
        return vr

    def _ocr_on_gpu(self) -> bool:
        """OCR 推理是否卸载到 GPU（实现：pipeline/policy.ocr_on_gpu）。"""
        return ocr_on_gpu(self._ocr_backend)

    @property
    def _gpu_pipeline_mode(self) -> bool:
        """F-5 前置位的兼容读面（真值在 _gpu_mode_box；写经引擎 lane）。"""
        return self._gpu_mode_box[0]

    def _gpu_pipeline_enabled(self) -> bool:
        """GPU 全驻留管线门控（实现：pipeline/policy.gpu_pipeline_enabled）。

        判定依据与三态语义的完整 docstring 见 policy 模块；门控对
        `gpu.device` 模块属性的 patch 点在 policy 内函数级 import 保持。
        """
        return gpu_pipeline_enabled(self._rc, self._decode_backend,
                                    self._ocr_backend,
                                    str(self._video_path))

    def _decode_num_threads(self, codec: str | None=None) -> int | None:
        """CPU 软解的 decord FFmpeg 帧线程数（按 codec/stride/OCR 位置分档）。

            完整分档依据与实测表已迁 ``config.decode_caliber.decode_num_threads``
            （2026-09-18 口径轮：单一事实源，探针与引擎同源派生）。
            """
        from video_ocr_engine.config.decode_caliber import decode_num_threads
        return decode_num_threads(
            codec, sample_stride=self._sample_stride,
            ocr_on_gpu=self._ocr_on_gpu(),
            override=self._rc.decode_num_threads)

    # ── 输出格式适配（实现：DecordFrameSource；活读 yuv/color_range）──
    def _crop_luma(self, crop: np.ndarray) -> np.ndarray:
        """crop → 分段/OCR 灰度（YUV 取 Y 按 range 展开，否则 _gray_seg）。"""
        return self._source.crop_luma(crop)

    def _batch_luma(self, crops: np.ndarray) -> np.ndarray:
        return self._source.batch_luma(crops)

    def _batch_luma_out(self, crops: np.ndarray,
                        out: np.ndarray) -> np.ndarray:
        """批量灰度写入预分配 out（省每批临时数组分配；形状恒定才可复用）。"""
        return self._source.batch_luma_out(crops, out)

    def _crop_is_expected(self, c: np.ndarray, roi_h: int, roi_w: int) -> bool:
        """ROI-first 输出尺寸是否符合当前输出格式（旧路径全帧则 False）。"""
        return self._source.crop_is_expected(c, roi_h, roi_w)

    def _ocr_engine_type(self) -> str:
        """OCR 推理后端类型（实现：pipeline/policy.ocr_engine_type）。"""
        return ocr_engine_type(self._ocr_backend)

    def _ocr_num_threads(self) -> int:
        """OCR 推理线程预算（实现：pipeline/policy.ocr_num_threads；
        codec/backend_label 为 open 后的活读值）。"""
        return ocr_num_threads(self._rc, getattr(self, '_codec', ''),
                               getattr(self, '_backend', ''))

    def _run_pipelined(self, _ocr_engines: list | None = None):
        """入口分发（S3-3d → R1 注入式）：构建 EngineInputs → SegmentEngine。

        _ocr_engines 两条 lane 都透传（B5）；None = 从进程级 OCR 引擎池取
        （ocr_native.acquire_ocr_engine）。GPU 门控在此求值一次（输入全为
        run 内常量）；GPU→宿主回退（C10）在引擎单处处理。lane 结果的状态
        同步（B2 fps 缓存经盒读写；其余为本次 run 的输出）收口在此——
        伪造 `_run_pipelined` 的测试（返回 RunOutcome）不受同步影响。
        """
        inp = EngineInputs(
            frame_start=self._frame_start, frame_end=self._frame_end,
            sample_stride=self._sample_stride, roi=tuple(self._roi),
            buffer_size=self._buffer_size,
            C=self._C, merge_similar=self._merge_similar,
            merge_similar_threshold=self._merge_similar_threshold,
            merge_max_changed_pixels=self._merge_max_changed_pixels,
            merge_dense_gate=self._merge_dense_gate,
            keep_crops=self._keep_crops, yuv_output=self._yuv_output,
            color_range=self._color_range, ocr_autocrop=self._ocr_autocrop,
            gpu_pipeline=self._gpu_pipeline_enabled(),
            segments_similar=self._segments_similar,
            crop_luma=self._crop_luma, batch_luma=self._batch_luma,
            batch_luma_out=self._batch_luma_out,
            crop_is_expected=self._crop_is_expected,
            content_range_to_crop=self._content_range_to_crop,
            open_vr=self._open_vr,
            start_ocr_session=self._start_ocr_session,
            backend_label=lambda: self._backend,
            ocr_on_gpu=self._ocr_on_gpu,
            merge_effective_mode=self._merge_effective_mode,
            progress=self._progress, cancel=self._cancel,
            prof_end=self._prof_end,
            on_bin_thresh=self._set_bin_thresh,
            bin_thresh_ref=[self._bin_thresh],
            fps_box=[self._fps],
            gpu_mode_box=self._gpu_mode_box,
            degraded=self._degraded,
            metrics=self._metrics)
        outcome = SegmentEngine(inp).run(_ocr_engines)
        # 状态同步（原 _run_pipelined_gpu/_host 尾部的双份收口至此单处）
        self._fps = outcome.fps
        self._bin_thresh = outcome.bin_thresh
        self._timing.update(outcome.timing)
        self._n_segments = outcome.n_segments
        self._crops = outcome.crops
        self._fork_stats = outcome.fork_stats
        self._ocr_texts = outcome.texts
        self._ocr_confs = outcome.confs
        return outcome
