"""宿主管线 mixin（_HostPipelineMixin）：OCR 会话启动 + 裁切薄层。

S3-3a：_host_calibrate / _host_frame_stream / _host_segment_frames 已迁至
pipeline.host_backend（显式 HostRunSpec 契约，不再读 FieldExtractor 私有
属性）；本模块只保留 OcrSession 组装与 autocrop 薄层（OcrSession 经
ex._crop_to_content 等调用，S3-3b 随会话契约化一并迁移）。
"""
from __future__ import annotations

from ._ocr_session import OcrSession

import logging

logger = logging.getLogger(__name__)


class _HostPipelineMixin:
    """宿主流水线 mixin：FieldExtractor 组合本类获得 OCR 会话与宿主管线。"""

    # ═══════════════ OCR 输入宽度自适应裁切 ═══════════════
    # 统一实现（含余量/最小收益门槛的实测依据 docstring）在
    # segmentation.content_range_to_crop / crop_to_content / crop_after_aspect；
    # GPU 直通（_autocrop_device）与宿主预处理共用同一余量数学。

    def _content_range_to_crop(self, first: int, last: int, w: int):
        """「有墨迹列范围」→ 裁切区间；实现见 segmentation.content_range_to_crop。"""
        from segmentation import content_range_to_crop
        return content_range_to_crop(
            first, last, w,
            margin_pct=self._ocr_autocrop_margin_pct,
            min_gain=self._ocr_autocrop_min_gain)

    def _crop_to_content(self, crop):
        """按二值图裁掉两侧空白（fa=0 路径）；实现见
        segmentation.crop_to_content（fa>0 走 _crop_after_aspect 顺序⑦）。"""
        from segmentation import crop_to_content
        return crop_to_content(
            crop, self._bin_thresh,
            autocrop=self._ocr_autocrop,
            force_aspect=float(getattr(self, '_force_aspect', 0) or 0.0),
            margin_pct=self._ocr_autocrop_margin_pct,
            min_gain=self._ocr_autocrop_min_gain)

    def _crop_after_aspect(self, img):
        """已定比例图上再按内容列裁（fa>0 路径）；实现见
        segmentation.crop_after_aspect（阈值现算 Otsu，不能用校准阈值）。"""
        from segmentation import crop_after_aspect
        return crop_after_aspect(
            img, autocrop=self._ocr_autocrop,
            margin_pct=self._ocr_autocrop_margin_pct,
            min_gain=self._ocr_autocrop_min_gain)

    def _start_ocr_session(self, _ocr_engines: list | None = None) -> "OcrSession":
        """启动 OCR 消费会话（实现见 _ocr_session.OcrSession；
        一次 extract() 一个实例，宿主/GPU 两管线共用）。"""
        return OcrSession(self, _ocr_engines)

