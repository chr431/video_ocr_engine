"""SegmentEngine —— 唯一编排点（v2 §6.4 / D2；S3-3d 起为唯一入口）。

R1（2026-09-27 门面拆解轮）：编排从「反向调门面 `_run_pipelined_gpu/
_host`」改为**注入式**——门面在 `_run_pipelined` 里构建一次
`EngineInputs`（一次 run 的显式输入契约：帧区间/采样/语义快照 + 策略与
源注入的回调），引擎据此自建 HostRunSpec / GpuRunSpec 并分发到两后端
（host_backend / gpu_backend，lane 策略与生命周期仍在 _driver 单出处）。
GPU→宿主回退（C10）随之收进引擎单处（此前挂在门面 `_run_pipelined_gpu`
的尾部）。GPU 门控由门面在构建 inputs 时求值一次（其输入——rc 冻结值
/后端名/视频路径——均为 run 内常量，与逐次调用等价）。

时序约束逐条保持（均有历史修复背书，勿扰动）：
- F-4：`on_bin_thresh` 回写即时性（宿主合并判定活读阈值）；
- F-5：`gpu_mode_box` 单元素盒替代门面属性 `_gpu_pipeline_mode`——
  lane 启动（会话启动**前**）写盒、SessionSpec 与报告组装读盒，
  读写时序与旧属性一致；
- F-6：`backend_label` 经回调在 open **之后**才读（backend 标签由
  open 写入；构造期求值会拿到 B1 重置后的空标签）；
- `yuv_output`/`color_range` 按**open 前快照**入 GPU spec（首跑 0 /
  复用实例带上一跑值——与旧门面构建时点逐位一致）。

端口（FrameSource / SegmentBackend / OcrBackend）已定义；R1 起
decode 侧以 DecordFrameSource 适配器落地打开与格式适配，driver 消费
面仍为裸 vr（端口收口属后续轮）。emit 契约为**批量签名**
（v2 §6.3 r3 定稿）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from ..domain.metrics import NULL_METRICS
from .gpu_backend import GpuRunSpec, run_gpu_pipeline
from .host_backend import HostRunSpec, run_host_pipeline


class SegmentBackend(Protocol):
    """把"一段帧 → 段 + 代表帧"的实现与编排解耦（§6.3）。

    两个实现：HostSegmentBackend（numpy）/ GpuSegmentBackend（设备驻留）。
    语义等价由金标向量逐位校验（§8.5/D4）；插拔的是**执行后端**，
    不是算法（C-32 依然成立）。
    """
    name: str

    def run(self, src, cfg, emit: Callable[[Sequence], None],
            progress, cancel) -> None: ...


@dataclass
class EngineInputs:
    """一次 run 的显式输入契约（门面 → 引擎；字段与两 RunSpec 同源）。

    语义快照字段在门面构建本对象时取值（等价于旧 `_run_pipelined_*`
    的 spec 构建时点）；回调由门面注入（策略纯函数 / FrameSource
    适配器 / ProfSpine 计时脊柱 / 会话启动）。
    """
    # 帧区间 / ROI / 采样 / 语义快照
    frame_start: int
    frame_end: int | None
    sample_stride: int
    roi: tuple                                # (x1, y1, x2, y2)
    buffer_size: int
    C: float
    merge_similar: bool
    merge_similar_threshold: float
    merge_max_changed_pixels: int
    merge_dense_gate: int                     # 0=关；>0=差异图 win3 阈值
    keep_crops: bool
    yuv_output: bool                          # open 前快照（见模块 docstring）
    color_range: int                          # 同上
    ocr_autocrop: bool
    # 门控（门面求值一次；输入全为 run 内常量）
    gpu_pipeline: bool
    # 判定与像素原语（显式注入；实现唯一出处 segmentation / source）
    segments_similar: Callable                # (gray_a, gray_b) -> bool
    crop_luma: Callable                       # crop -> gray
    batch_luma: Callable                      # (B,H,W[,C]) -> (B,h,w)
    batch_luma_out: Callable                  # 同上，写入复用缓冲
    crop_is_expected: Callable                # (crop, roi_h, roi_w) -> bool
    content_range_to_crop: Callable           # (first, last, w) -> 区间|None
    # 门面回调
    open_vr: Callable                         # () -> vr（打开决策树在 source）
    start_ocr_session: Callable               # (engines|None) -> OcrSession
    backend_label: Callable = lambda: ""      # () -> str（open_vr 后读，F-6）
    ocr_on_gpu: Callable = lambda: False      # () -> bool（hybrid 引擎走 GPU OCR）
    merge_effective_mode: Callable = lambda: "binary"   # () -> str（binary/''）
    progress: Callable = lambda m, p: None
    cancel: Callable = lambda: None
    prof_end: Callable | None = None          # (group, key, t0)
    on_bin_thresh: Callable | None = None     # (th) -> None（F-4 即时回写）
    # 可变盒 / 共享状态（门面持有，引擎与后端读写）
    bin_thresh_ref: list = field(default_factory=lambda: [0])   # [th]（GPU lane）
    fps_box: list = field(default_factory=lambda: [None])       # B2：同实例缓存
    gpu_mode_box: list = field(default_factory=lambda: [False])  # F-5 前置位
    degraded: list = field(default_factory=list)                # D3 降级原因
    # S6-0：注入的指标记录器（§8.6 N-2；off 档为 NullMetrics 单例）
    metrics: Any = NULL_METRICS


@dataclass
class RunOutcome:
    """一次 run 的产物（v2 §6.4；v1 的裸 5 元组至此有名字）。

    R1 起携带 lane 结果同步面（timing/crops/fps/bin_thresh 等），
    门面状态同步从两处 lane 尾巴收口到 `_run_pipelined` 单处。
    """
    frames: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    confs: list = field(default_factory=list)
    rep_frames: list = field(default_factory=list)
    # lane 结果同步面
    timing: dict = field(default_factory=dict)
    crops: dict = field(default_factory=dict)
    fps: float | None = None
    bin_thresh: int = 0
    n_segments: int = 0
    fork_stats: dict | None = None   # fork 遥测穿透（hybrid 解码器才有）
    report: dict | None = None       # S4 起：RunReport（§8.6）

    @classmethod
    def from_result(cls, res) -> "RunOutcome":
        """从 Host/GpuRunResult 收敛（两结果对象字段同名）。"""
        return cls(
            frames=res.frames, segments=res.segs,
            texts=res.texts, confs=res.confs, rep_frames=res.rep_frames,
            timing=res.timing, crops=res.crops, fps=res.fps,
            bin_thresh=res.bin_thresh, n_segments=res.n_segments,
            fork_stats=getattr(res, "fork_stats", None))

    def as_tuple(self) -> tuple:
        """与 v1 `_run_pipelined` 返回形状逐位兼容（迁移期）。"""
        return (self.frames, self.segments, self.texts, self.confs,
                self.rep_frames)


class SegmentEngine:
    """编排唯一出处：EngineInputs 显式契约 → 选后端 → RunOutcome。

    GPU 门控通过 → gpu_backend（形状不符回退宿主，C10）；否则宿主
    host_backend。生命周期单出处 `_driver.run_segment_pipeline`。
    """

    def __init__(self, inputs: EngineInputs) -> None:
        self._inputs = inputs

    def run(self, ocr_engines: list | None = None) -> RunOutcome:
        inp = self._inputs
        # S6-0（§8.6 N-2）：编排顶层 span——唯一编排点正是"量一次、
        # 处处可读"成立的前提（v1 结构上做不到）。
        metrics = inp.metrics
        if metrics is None or not metrics.enabled:
            return self._run_backend(ocr_engines)
        with metrics.span("pipeline.run"):
            return self._run_backend(ocr_engines)

    def _run_backend(self, ocr_engines: list | None) -> RunOutcome:
        inp = self._inputs
        if inp.gpu_pipeline:
            return self._run_gpu(ocr_engines)
        return self._run_host(ocr_engines)

    def _run_gpu(self, ocr_engines: list | None) -> RunOutcome:
        inp = self._inputs
        inp.gpu_mode_box[0] = True    # F-5：会话启动前置位（读点在 SessionSpec）
        spec = GpuRunSpec(
            frame_start=inp.frame_start, frame_end=inp.frame_end,
            sample_stride=inp.sample_stride, roi=inp.roi,
            buffer_size=inp.buffer_size,
            C=inp.C, merge_similar=inp.merge_similar,
            merge_similar_threshold=inp.merge_similar_threshold,
            merge_max_changed_pixels=inp.merge_max_changed_pixels,
            merge_dense_gate=inp.merge_dense_gate,
            keep_crops=inp.keep_crops, yuv_output=inp.yuv_output,
            color_range=inp.color_range, ocr_autocrop=inp.ocr_autocrop,
            bin_thresh_ref=inp.bin_thresh_ref,
            backend_label=inp.backend_label,
            ocr_on_gpu=inp.ocr_on_gpu,
            merge_effective_mode=inp.merge_effective_mode,
            content_range_to_crop=inp.content_range_to_crop,
            open_vr=inp.open_vr,
            start_ocr_session=inp.start_ocr_session,
            batch_luma=inp.batch_luma,
            progress=inp.progress, cancel=inp.cancel,
            prof_end=inp.prof_end,
            on_bin_thresh=inp.on_bin_thresh,
            metrics=inp.metrics,
            fps_box=inp.fps_box)
        res = run_gpu_pipeline(spec, ocr_engines)
        if res.fell_back_to_host:
            # C10 接线：形状不符 → fallback_vr/engines 移交宿主 lane
            #（reader 不 close：宿主路径复用已打开的 reader），降级透出。
            inp.degraded.append('GPU 管线形状不符，回退宿主管线')
            return self._run_host(res.fallback_engines, res.fallback_vr)
        return RunOutcome.from_result(res)

    def _run_host(self, ocr_engines: list | None,
                  preopened_vr=None) -> RunOutcome:
        inp = self._inputs
        inp.gpu_mode_box[0] = False   # F-5 复位（宿主 lane 恒走宿主会话）
        spec = HostRunSpec(
            frame_start=inp.frame_start, frame_end=inp.frame_end,
            sample_stride=inp.sample_stride, roi=inp.roi,
            C=inp.C, merge_similar=inp.merge_similar,
            keep_crops=inp.keep_crops, yuv_output=inp.yuv_output,
            segments_similar=inp.segments_similar,
            crop_luma=inp.crop_luma, batch_luma=inp.batch_luma,
            batch_luma_out=inp.batch_luma_out,
            crop_is_expected=inp.crop_is_expected,
            open_vr=inp.open_vr,
            start_ocr_session=inp.start_ocr_session,
            backend_label=inp.backend_label,
            progress=inp.progress, cancel=inp.cancel,
            prof_end=inp.prof_end,
            on_bin_thresh=inp.on_bin_thresh,
            metrics=inp.metrics,
            fps_box=inp.fps_box)
        if preopened_vr is not None:
            res = run_host_pipeline(spec, ocr_engines,
                                    preopened_vr=preopened_vr)
        else:
            res = run_host_pipeline(spec, ocr_engines)
        return RunOutcome.from_result(res)
