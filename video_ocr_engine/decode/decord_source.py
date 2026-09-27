"""DecordFrameSource —— decord 解码后端的打开与格式适配（FrameSource 端
口的 decord 适配器，R1 2026-09-27 自 extractor 门面下放）。

S3-2 定义 FrameSource 端口时预留的 `decode/decord_source.py` 至此真正
落地：本模块持有"按 decode_backend 打开解码器"的完整决策树
（auto/nvdec/cpu/hybrid + 降级链 + codec 探测重开 + hybrid 硬窗谓词）
与输出格式适配（yuv420/gray、luma 提取四件套、color_range 记忆）。
语义要点来自 S0 冻结的 decoder_contract.yaml（DC-01..10）——尤其
roi_format（ROI-first 是引擎性能地基）与 color_range（决定 Y 平面展开）。

门面 `_open_vr` = `source.open()` + 四字段状态同步（backend/codec/
color_range/yuv_output 回写 `ex._*`，保住既有读点与探针/测试 patch 面）；
driver 消费面本轮仍为裸 vr（`open()` 返回 reader 本体，零驱动侧改动），
FrameSource 协议面（len/fps/codec/batch/close）以薄委托形式先行接线，
端口收口（driver 直接消费 source）属后续轮次。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from video_ocr_engine.config.decode_caliber import decode_num_threads
from video_ocr_engine.domain.segmentation import (
    _gray_seg, _gray_seg_batch,
    _gray_seg_yuv, _gray_seg_yuv_batch,
    _nv12_batch_luma_full_out, _gray_batch_out,
)

logger = logging.getLogger(__name__)


def ensure_roi_capable_decoder() -> None:
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


class DecordFrameSource:
    """一段视频的解码访问面（一次 run 绑定一个实例；open 前无资源）。

    状态字段（open 后有效，门面在 `_open_vr` 返回后同步）：
      backend_label / codec / color_range / yuv_output（yuv 不可用时原地
      翻 False——代表帧退化灰度，降级原因进 degraded 列表）。
    """

    def __init__(self, video_path, roi: tuple, decode_backend: str,
                 yuv_output: bool, degraded: list, *,
                 frame_start: int = 0, frame_end=None,
                 decode_threads_fn=None, hybrid_cpu_threads: int = 0,
                 ocr_on_gpu_fn=None) -> None:
        self._video_path = Path(video_path)
        self._roi = tuple(roi)
        self._decode_backend = decode_backend
        self.yuv_output = yuv_output
        self._degraded = degraded          # 与门面共享的降级原因列表（D3）
        self._frame_start = frame_start
        self._frame_end = frame_end
        # (codec|None) -> int|None：codec 感知 CPU 软解线程档位（事实源
        # config.decode_caliber；门面注入以共享同一 rc override）
        self._decode_threads_fn = decode_threads_fn
        self._hybrid_cpu_threads = hybrid_cpu_threads   # rc 冻结值；<=0 = auto
        self._ocr_on_gpu_fn = ocr_on_gpu_fn
        self._vr = None
        self.backend_label = ""
        self.codec = ""
        self.color_range = 0

    # ── FrameSource 协议面（薄委托；driver 本轮仍消费裸 vr）─────────
    def __len__(self) -> int:
        return len(self._require_open())

    @property
    def fps(self) -> float:
        from video_ocr_engine._helpers import _read_fps_from_vr
        return _read_fps_from_vr(self._require_open()) or 0.0

    def roi_format(self) -> str:
        return 'yuv420' if self.yuv_output else 'gray'

    def batch(self, frames):
        return self._require_open().get_batch(frames)

    def close(self) -> None:
        if self._vr is not None:
            self._vr.close()

    def _require_open(self):
        if self._vr is None:
            raise RuntimeError("DecordFrameSource.open() 尚未调用")
        return self._vr

    # ── 打开决策树（原 extractor._open_vr，行为逐位保持）─────────────
    def open(self):
        """按 decode_backend 打开解码器（auto/cpu/nvdec/hybrid）。

            auto: 尝试 GPU (NVDEC) 失败回退 CPU。cpu: 强制 CPU。
            nvdec: 强制 GPU（失败回退 CPU 并警告）。
            hybrid: CPU+NVDEC 混合解码（fork 原生 hybrid/hybrid_gpu ctx）；
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
        from video_ocr_engine.config.decode_caliber import roi_for_decord
        roi = roi_for_decord(self._roi)
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
                vr = self._open_reader(_g(0), roi_kw)
                label = 'GPU'
            except Exception as e:  # noqa: BLE001
                # 保留异常文本（2026-09-19 审查轮）：此前吞掉原始异常，
                # 「驱动/权限不可用」与「代码 bug 导致打开失败」在
                # auto 路径下完全无法区分，性能回归静默发生。
                vr = None
                self._degraded.append('NVDEC 打开失败，回退 CPU: %r' % (e,))
                logger.warning('NVDEC 解码不可用，回退 CPU: %r', e,
                               exc_info=True)
        if vr is None:
            vr = self._open_reader(_cpu(0), roi_kw,
                                   num_threads=self._decode_threads())
            label = 'CPU'
        self.backend_label = f'decord/{label}'
        if label == 'CPU':
            try:
                self.codec = str(vr.get_codec() or '').lower()
            except Exception:  # noqa: BLE001
                # 探测失败=线程档位按默认 h264 走（hevc/av1 实测差
                # 4~13%）——记降级原因（2026-09-19 审查轮）。
                self.codec = ''
                self._degraded.append('codec 探测失败，解码线程档位按默认')
                logger.debug('codec 探测失败', exc_info=True)
            # codec 感知线程档位（2026-09-10 实测表，见 decode_num_threads）：
            # hevc/av1 在 FFmpeg9 下的帧线程扩展性与 h264 分化（hevc
            # stride=1 到 32 线程仍在涨、av1 stride=8 最优 48），通用档位
            # 按最常见 h264 设定，打开后读 codec、档位不同则重开一次
            # （实测重开 ~20-30ms，hevc stride1 墙钟 -27%）。
            nt = self._decode_threads()
            nt_codec = self._decode_threads(codec=self.codec or None)
            if nt_codec != nt:
                vr = self._open_reader(_cpu(0), roi_kw,
                                      num_threads=nt_codec)
        else:
            try:
                self.codec = str(vr.get_codec() or '').lower()
            except Exception:  # noqa: BLE001
                # 探测失败=线程档位按默认 h264 走（hevc/av1 实测差
                # 4~13%）——记降级原因（2026-09-19 审查轮）。
                self.codec = ''
                self._degraded.append('codec 探测失败，解码线程档位按默认')
                logger.debug('codec 探测失败', exc_info=True)
        self._remember_color_range(vr)
        # CPU+NVDEC 混合解码（decode_backend="hybrid" 显式选择，与 auto/cpu/nvdec
        # 并列）：速率比例分界 + 两端连续扫掠（fork 原生 hybrid ctx）。
        # NVDEC 可用即包装（GPU 全驻留管线开启时由其 CPU 分支消费宿主数组，
        # §8.3 合并；关闭时走宿主管线，行为不变）。
        # **stride>1 已解禁**（原门控要求 stride==1，理由是 next_roi 的
        # 顺序交付语义）：next_roi 现按 sample_stride 推进，且 stride>1 时
        # 宿主校准与主循环都走 get_batch 等差快速路径、不碰 next_roi。
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
            try:
                from decord import hybrid as _hy, hybrid_gpu as _hyg
                _ct = self._hybrid_cpu_threads
                if _ct <= 0:
                    # §14 定档（2026-09-12，引擎全片 e2e 配对 A/B，三码文本
                    # 集逐位一致）：hybrid CPU 臂与 cpu 后端共用同一 codec
                    # 感知策略（decode_num_threads）——本机 h264+TRT=32 /
                    # hevc=32 / av1=24，恰为各码实测最优：h264lg 16→24
                    # −5.9%（3/3）、24→32 再 −4.8%（3/3）；hevc 16→24
                    # −4.4%（3/3）、24→32 持平；av1 16→24 −12.8%（3/3）但
                    # 32 反向 +5.1%（dav1d 过订阅）。旧"逻辑核//2 钳 [8,16]"
                    # 是项目层 hybrid 时代残留（S6 续只测到 16 为止），三码
                    # 全劣 4~13%。decode-only 口径 24T 已饱和、32T 无增益，
                    # 但引擎消费节奏（OCR/分段线程与解码共存）顶点更高——
                    # 档位以引擎口径为准。复评触发：decord 再换代 / 核数
                    # 格局变化。
                    _ct = (self._decode_threads(codec=self.codec or None)
                           or 16)
                _on_gpu = self._ocr_on_gpu_fn is not None and self._ocr_on_gpu_fn()
                _hctx = _hyg(0) if _on_gpu else _hy(0)
                vr = self._open_reader(_hctx, roi_kw, num_threads=_ct)
                # 硬窗界（2026-09-17 越窗修复）：短窗时声明消费上限，fork
                # 的 demux 与 GOP 派工在窗缘硬停——窗口外一个包都不读
                # （实测 w3000 曾把全片 7761 包喂进两臂）。仅当窗口 < 全长
                # 且 **start < 窗长** 才设：后者不是任意保守——fork 实测
                # （2026-09-20 起点轮）晚起点（start=5000/win=1000）+
                # seek_accurate + 硬窗交付的帧与 cpu/nvdec 基线**位级不
                # 一致**（同请求无窗时三者一致）→ 晚起点硬窗存在 fork 级
                # 缺陷，此谓词是唯一防线，修复前不得删。全片运行不设 =
                # fork 深库存行为不变。需 fork ≥ 遥测穿透版（stock decord
                # 无此方法，getattr 容忍）。
                _sdw = getattr(vr, 'set_decode_window', None)
                if (_sdw is not None and self._frame_end is not None
                        and self._frame_start < (self._frame_end - self._frame_start)
                        and self._frame_end - self._frame_start < len(vr)):
                    _sdw(self._frame_end - self._frame_start)
                self.backend_label = 'decord/hybrid'
                logger.info('混合解码开启(原生): codec=%s ctx=%s cpuT=%d',
                            self.codec,
                            'hybrid_gpu' if _on_gpu else 'hybrid',
                            _ct)
            except Exception as e:  # noqa: BLE001
                self._degraded.append(f'hybrid 打开失败，回退纯 GPU: {e}')
                logger.warning('原生混合解码打开失败，回退纯 GPU: %s', e)
        self._vr = vr
        return vr

    def _decode_threads(self, codec: str | None = None) -> int | None:
        """CPU 软解线程档位（门面注入的 decode_threads_fn，或本地回退）。"""
        if self._decode_threads_fn is not None:
            return self._decode_threads_fn(codec)
        return decode_num_threads(codec, sample_stride=1, ocr_on_gpu=True)

    def _open_reader(self, ctx, roi_kw: dict, num_threads=None):
        """按当前输出格式打开 decord reader（原 extractor._open_decord_reader）。

            yuv420 仅在 fork ≥0.7.10 可用：旧 DLL 会抛 ValueError，此时
            回退 gray（分段/OCR 不变，仅代表帧预览退化灰度）并重置标志。
            num_threads：CPU 软解的 FFmpeg 帧线程数（少核分核，None=decord
            默认；GPU/NVDEC 不传）。
            """
        from decord import VideoReader
        fmt = 'yuv420' if self.yuv_output else 'gray'
        nt_kw = {'num_threads': num_threads} if num_threads else {}
        try:
            return VideoReader(str(self._video_path), ctx=ctx, output_format=fmt, **nt_kw, **roi_kw)
        except ValueError:
            if not self.yuv_output:
                raise
            logger.warning('当前 decord 不支持 yuv420 输出，回退 gray （代表帧预览将为灰度）')
            self._degraded.append('decord 不支持 yuv420 输出，代表帧退化灰度')
            self.yuv_output = False
            self.color_range = 0
            return VideoReader(str(self._video_path), ctx=ctx, output_format='gray', **nt_kw, **roi_kw)

    def _remember_color_range(self, vr) -> None:
        """YUV 模式下从 decoder 读取流 color_range（0=limited/tv）。"""
        if not self.yuv_output:
            return
        try:
            self.color_range = int(vr.get_color_range() or 0)
        except Exception:  # noqa: BLE001
            # 失败按 limited 处理但**记录**（2026-09-19 审查轮）：full
            # range 流被当 limited 会错误拉伸 Y → 分段阈值/OCR 像素静默
            # 改变，此前 meta['color_range']=0 与"真的是 limited"不可分。
            self.color_range = 0
            self._degraded.append('color_range 读取失败，按 limited 处理')

    # ── 输出格式适配（luma 四件套，活读 yuv/color_range 状态）────────
    def decord_format(self) -> str:
        """当前管线请求的 decord output_format。

        内部链永远只消费单通道（Y 平面 / decord gray，不再输出 RGB）：
        - keep_crops 需要 YUV 代表帧 → 'yuv420'（packed NV12；内部取 Y 平面，
          等价灰度，另保留 UV 供外部 nv12_to_rgb）
        - 否则 'gray'
        """
        return 'yuv420' if self.yuv_output else 'gray'

    def crop_luma(self, crop: np.ndarray) -> np.ndarray:
        """crop → 分段/OCR 灰度：YUV 时取 Y 并按 range 展开，否则 _gray_seg。"""
        if self.yuv_output:
            return _gray_seg_yuv(crop, self.color_range)
        return _gray_seg(crop)

    def batch_luma(self, crops: np.ndarray) -> np.ndarray:
        if self.yuv_output:
            return _gray_seg_yuv_batch(crops, self.color_range)
        return _gray_seg_batch(crops)

    def batch_luma_out(self, crops: np.ndarray,
                       out: np.ndarray) -> np.ndarray:
        """批量灰度写入预分配 out（省每批临时数组分配；形状恒定才可复用）。"""
        if self.yuv_output:
            return _nv12_batch_luma_full_out(crops, self.color_range, out)
        return _gray_batch_out(crops, out)

    def crop_is_expected(self, c: np.ndarray, roi_h: int, roi_w: int) -> bool:
        """ROI-first 输出尺寸是否符合当前输出格式（旧路径全帧则 False）。"""
        if self.yuv_output:
            return c.ndim == 2 and c.shape[0] == roi_h + (roi_h + 1) // 2 and (c.shape[1] == roi_w)
        return c.shape[0] == roi_h and c.shape[1] == roi_w
