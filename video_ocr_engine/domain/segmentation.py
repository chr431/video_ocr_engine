"""分段 + OCR 预处理统一实现（宿主管线与 GPU 全驻留管线共用，0.11.0）。

为什么是这个文件：分段判定（校准/二值/断段/相似合并）、裁切（内容列范围 →
余量 → 最小收益门槛）与预处理（resize + gamma）的算法实现此前散在
segmentation 原语 / extractor / _host_pipeline / _gpu_pipeline 四处，宿主与
GPU 各有一份同构逻辑。本文件是**唯一实现地**：

  - 宿主管线直接调用这些纯函数与 SegmentStateMachine；
  - GPU 全驻留管线的 CUDA kernel（_gpu_kernels.py）是本文件算法的设备侧
    逐位镜像（像素级操作无法跨 numpy/CUDA 共享），但其判定阈值、裁切余量
    数学、相似合并判据统一引用这里 —— 两侧语义只有一个出处。

章节：§1 像素原语 §2 校准与判定 §3 分段状态机 §4 裁切与预处理
"""
from __future__ import annotations

import numpy as np

from video_ocr_engine.config import constants as config
from video_ocr_engine.domain.video_utils import (_gray, _GRAY_W, _nv12_luma_full,
                         _nv12_batch_luma_full, _np_resize)


# ═══════════════════ §1 像素原语 ═══════════════════

def _gray_batch(crops: np.ndarray) -> np.ndarray:
    """批量灰度：(B,H,W,3) → (B,H,W)；decord gray 输出 (B,H,W,1) 直接取通道。

    gray 输出模式（CPU/GPU 解码，decord ≥0.7.9）crops 已是 1 通道，跳过 matmul；
    yuv420 模式（≥0.7.10）请用 _gray_seg_yuv/_gray_seg_yuv_batch。
    """
    if crops.shape[-1] == 1:
        return crops[..., 0]
    return (crops.astype(np.float32) @ _GRAY_W).astype(np.uint8)


def _gray_batch_out(crops: np.ndarray, out: np.ndarray) -> np.ndarray:
    """批量灰度写入预分配 out（形状必须与 crops 匹配）。

    省去每批临时 float32 全帧数组（_gray_batch 的 astype 临时）与结果分配；
    数值路径与 _gray_batch 逐位一致（同一 float32 权重乘法 + uint8 截断）。
    """
    if crops.shape[-1] == 1:
        out[...] = crops[..., 0]
        return out
    tmp = crops.astype(np.float32, copy=False) @ _GRAY_W
    out[...] = np.clip(tmp, 0, 255)
    return out


def _gray_seg(crop: np.ndarray) -> np.ndarray:
    """分段/代表帧选择用灰度（raw，已锁定基线）。"""
    return _gray(crop)


def _gray_seg_batch(crops: np.ndarray) -> np.ndarray:
    """批量分段灰度。"""
    return _gray_batch(crops)


def _gray_seg_yuv(crop: np.ndarray, color_range: int = 0) -> np.ndarray:
    """decord yuv420 crop → 分段灰度：取 Y 平面 + range 展开。"""
    return _nv12_luma_full(crop, color_range)


def _gray_seg_yuv_batch(crops: np.ndarray, color_range: int = 0) -> np.ndarray:
    """decord yuv420 批量 crops → 批量分段灰度。"""
    return _nv12_batch_luma_full(crops, color_range)


def _nv12_batch_luma_full_out(crops: np.ndarray, color_range: int,
                              out: np.ndarray) -> np.ndarray:
    """批量 Y 平面 + range 展开，写入预分配 out（形状必须匹配）。

    数值路径与 _nv12_batch_luma_full 逐位一致（S6-f′ 起两者共用同一张
    256 项 LUT：Y 是 8-bit，展开是单字节纯函数）；省每批 float32 临时数组
    与结果分配（主流水线每批形状恒定，缓冲可跨批复用）。
    """
    from ..domain.video_utils import _Y_LIMITED_LUT
    h = crops.shape[1] * 2 // 3
    w = crops.shape[2]
    out = out[:crops.shape[0], :h, :w]   # 末批可能不满 B；按实际 Y 高/宽取切片
    if color_range == 1:
        out[...] = crops[:, :h]
        return out
    out[...] = _Y_LIMITED_LUT[crops[:, :h]]
    return out





def _otsu_from_hist(hist) -> int:
    """从 256-bin 直方图算 Otsu 阈值（与 _otsu 等价；GPU 校准直方图行用）。"""
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


def _cluster_win3(diff: np.ndarray) -> float:
    """最大 3×3 窗口变化像素和 —— 聚类判别的廉价代理（纯 numpy，无 scipy）。

    原 scipy.ndimage.label 连通分量对 test6 23k 边贡献 ~2.3s；且 scipy 非
    pyproject 依赖，PyInstaller 打包会连带整个 scipy 增肥 exe。本实现用
    6 次切片错位累加求最大 3×3 窗口和（越界按 0），数值与
    uniform_filter 逐位一致（含边界，500 随机掩码最大差 0）。
    语义：真实数字变化必然产生 ≥5 像素连成 3×3 的密集簇（实测变帧恒=9）；
    噪声孤立像素的最大窗口和 < 5。C=5 下 test/test5/test6 0 漏检且段数更少。

    **2026-08-30 改用 uint8**（原 int32）：窗口和的数学上界 = 9（3×3 全 1），
    uint8 足够且不会溢出；又因 `np.bool_` 与 `np.uint8` 同为 1 字节，
    `diff.view(np.uint8)` 是**零拷贝**（连类型转换都省掉），切片加法的内存
    带宽降到 1/4。实测（`tools/_probe_cluster_dtype.py`，63 组随机掩码 +
    结构化图案逐位等价）：

    | ROI | 面积 | int32 | uint8 | 加速 |
    |---|---:|---:|---:|---:|
    | 106×33 | 3.5k px | 15.42 µs | 13.69 µs | 1.13× |
    | 800×52 | 41.6k px | 95.33 µs | **72.48 µs** | **1.32×** |
    | 800×200 | 160k px | 554.91 µs | **271.85 µs** | **2.04×** |
    | 1600×600 | 960k px | 3451.00 µs | 1797.27 µs | 1.92× |

    只对**宿主路径**有意义——现役默认（NVDEC+TRT）走 GPU 全驻留管线，
    分段在 GPU 上做（CUDA kernel 为本函数的设备侧逐位镜像），不调用本函数。
    """
    if not diff.any():
        return 0.0
    # bool 与 uint8 同为 1 字节 → view 零拷贝；非连续时退回 astype
    s = diff.view(np.uint8) if diff.flags.c_contiguous else diff.astype(np.uint8)
    # 行向 3 列和（左右越界 0）
    c3 = s.copy()
    c3[:, 1:] += s[:, :-1]
    c3[:, :-1] += s[:, 1:]
    # 列向 3 行和（上下越界 0）
    w3 = c3.copy()
    w3[1:, :] += c3[:-1, :]
    w3[:-1, :] += c3[1:, :]
    return float(w3.max())


# ═══════════════════ §2 校准与判定 ═══════════════════
def _otsu(g: np.ndarray) -> int:
    """灰度图 Otsu 阈值：做直方图后委托 _otsu_from_hist（算法单一实现）。"""
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    return _otsu_from_hist(hist)




def otsu_median_threshold(ths) -> int:
    """校准阈值：Otsu 列表取中位数；空列表回退 OTSU_FALLBACK_THRESH。"""
    if not ths:
        return config.OTSU_FALLBACK_THRESH
    return int(np.median(ths))


def similar_decision(mean: float, changed_px: int, threshold: float,
                     max_changed: int) -> bool:
    """相似段合并的两阈值判定（宿主 numpy 统计与 GPU sim_pair 标量共用）。

    只看平均绝对差会把宽 ROI 中的单字短字幕（如"在""不"）误判为噪声：
    大部分区域未变，均值被稀释 —— 因此附加"显著变化像素数"上限。
    """
    if mean > threshold:
        return False
    return changed_px <= max_changed


def _text_sep_binary(gray: np.ndarray, th: int) -> np.ndarray:
    """从背景中分离字幕文字的灰度图（merge_similar 判定用）。

    用阈值把文字变白、背景变黑（float32 255/0）。
    """
    g = gray.astype(np.float32)
    return np.where(g > th, 255.0, 0.0).astype(np.float32)


# ═══════════════════ §3 分段状态机（双管线共用编排） ═══════════════════

class SegmentStateMachine:
    """断段 / 代表帧选择 / 相似合并的统一编排（宿主与 GPU 管线共用）。

    两侧差异只在数据来源，状态机以双入口吸收：
      - 宿主：feed(bin=二值图) —— 帧间 diff 的 win3 由本机计算；
      - GPU：feed(cluster=分数) —— win3 已由设备 kernel 算好。
    代表帧选择 = 段内 sharp（灰度 std）最大帧；断段判定 = win3 ≥ C；
    合并 = on_similar 回调（宿主传代表灰度，GPU 传设备指针）。

    payload：随帧携带的任意对象，成为代表帧时原样传给 on_emit
      （宿主 = (frame_idx, crop, gray, dev_info)；GPU = (frame_idx, dev, sharp)）。
    """

    def __init__(self, frames: list, *, C: float,
                 on_emit, on_similar,
                 on_cancel=None, on_progress=None,
                 debug_tag: str | None = None,
                 debug_bounds: "bool | None" = None):
        self._debug_bounds = debug_bounds   # S9-2(D6):None=feed 期读 env(直连用户兼容)
        """on_emit(seg, rep_payload, frac) — 段闭合投递（frac = k/total）。
        on_similar(a_payload, b_payload) -> bool — 相邻发射段代表帧相似判定；
            仅在已有发射段时被调用（空段列表不调用，等价宿主原 `segs and` 短路）。
        on_cancel / on_progress(k, frac) — 取消与进度钩子（100/500 帧节拍，
            与两条管线历史节奏一致）。debug_tag 供 DEBUG_BOUNDS 边界行。"""
        self._frames = frames
        self._total = max(len(frames), 1)
        self._C = C
        self._on_emit = on_emit
        self._on_similar = on_similar
        self._on_cancel = on_cancel
        self._on_progress = on_progress
        self._debug_tag = debug_tag
        self.segs: list = []
        self._s = 0
        self._started = False
        self._prev_bin = None
        self._rep_payload = None
        self._last_rep_payload = None
        self._rep_sharp = -1.0

    def _set_rep(self, fi, sharp, payload) -> None:
        self._rep_frame_fi = fi
        self._rep_sharp = sharp
        self._rep_payload = payload

    def feed(self, k: int, fi: int, sharp: float, payload,
             *, bin: "np.ndarray | None" = None,
             cluster: "float | None" = None) -> None:
        """消费一帧。宿主管线传 bin（二值图），GPU 管线传 cluster（win3 分数）。"""
        if self._started:
            if bin is not None:
                d = self._prev_bin != bin
                score = _cluster_win3(d)
                changed = score >= self._C
            else:
                score = float(cluster)
                changed = score >= self._C
            if changed:
                seg = self._frames[self._s:k]
                if (self._debug_tag is not None
                        and (self._debug_bounds if self._debug_bounds is not None
                             else config.env_bool(config.DEBUG_BOUNDS_ENV))):
                    print(f'[{self._debug_tag}]{fi}:{score:.0f}', flush=True)
                if self.segs and self._on_similar(self._last_rep_payload,
                                                  self._rep_payload):
                    # 同一视觉内容被噪声切成多段：并入前一段，不产生新的
                    # OCR 任务，保留前一段代表帧/文本。
                    self.segs[-1].extend(seg)
                else:
                    self.segs.append(seg)
                    self._on_emit(seg, self._rep_payload,
                                  k / self._total)
                    self._last_rep_payload = self._rep_payload
                self._s = k
                self._set_rep(fi, sharp, payload)
            elif sharp > self._rep_sharp:
                self._set_rep(fi, sharp, payload)
        else:
            self._set_rep(fi, sharp, payload)
            self._started = True
        self._prev_bin = bin
        if k % 100 == 0 and self._on_cancel is not None:
            self._on_cancel()
        if k % 500 == 0 and self._on_progress is not None:
            self._on_progress(k, k / self._total)

    def finish(self) -> None:
        """流结束：处理尾段（与 feed 内闭合逻辑同判据）。"""
        seg = self._frames[self._s:]
        if self.segs and self._on_similar(self._last_rep_payload,
                                          self._rep_payload):
            self.segs[-1].extend(seg)
        else:
            self.segs.append(seg)
            self._on_emit(seg, self._rep_payload, 1.0)


# ═══════════════════ §4 裁切与预处理 ═══════════════════

def content_range_to_crop(first: int, last: int, w: int, *, margin_pct: int,
                          min_gain: float):
    """「有墨迹列范围」→ 裁切区间 (x_off, crop_w)；满宽/收益不足返回 None。

    宿主 `_crop_to_content` 与 GPU 直通（`_autocrop_device`）共用的同一余量
    数学。余量 `OCR_ROI_AUTOCROP_MARGIN`（占 ROI 宽 %）。

    **最小收益门槛** `OCR_ROI_AUTOCROP_MIN_GAIN`（占 ROI 宽 %，默认 10）：
    裁掉宽度占 ROI 宽的比例低于门槛时返回 None（整段不裁）。

    这一条比"加大余量"更根本。余量是全局的，为规避紧凑 ROI 的误裁而
    调大，会连宽 ROI 的收益一起削掉（test5 裁掉量中位数：余量 10 时
    13.2% → 余量 20 时只剩 3.8%）。而误裁**只发生在"微裁"段** ——
    test 在余量 10 下 72% 的段被裁、裁掉量中位数却只有 1.2%，61 段
    误裁全在这些段里；test5/test6 裁掉 13% 且零误裁。
    有了门槛后，紧凑 ROI 自动几乎不裁、宽 ROI 照裁，两者不再需要折中。
    """
    m = max(1, int(round(w * margin_pct / 100.0)))
    lo = max(0, int(first) - m)
    hi = min(w, int(last) + 1 + m)
    if lo == 0 and hi == w:
        return None
    if min_gain > 0.0:
        if (w - (hi - lo)) / w < min_gain:
            return None          # 收益太小：不值得承担切笔画的风险
    return lo, hi - lo


def crop_to_content(crop: np.ndarray, bin_thresh: int, *, autocrop: bool = True,
                    force_aspect: float = 0.0, margin_pct: int = 10,
                    min_gain: float = 0.1) -> np.ndarray:
    """按二值图的"有墨迹列范围"裁掉两侧空白（宽 ROI 字幕省 OCR 计算）。

    判据与分段完全一致（`g > bin_thresh`，墨迹为亮），
    每列墨迹数 ≥ 2 才算有效列（抗孤立噪点）。

    不裁的三类情况（无收益或有风险，一律原样返回）：
      · 关闭 / `force_aspect > 0`（宽度被强制 → 走 crop_after_aspect 顺序⑦）
      · 动态范围过小（std < 3，纯黑/纯白帧，Otsu 阈值无意义）
      · 内容已占满 ROI（cols 覆盖全宽）
    余量与门槛数学见 `content_range_to_crop`；**不能**靠加大余量规避误裁
    （会把宽 ROI 收益一起削掉），也不能补"裁后会被 pad 回下限就跳过"的
    守卫 —— 裁切让输入更贴近模型训练分布，即使省不到算力也能提准确率
    （实测 test5 +0.82pp / test6 +0.94pp，见 _probe_roi_whitespace 系列）。
    """
    if not autocrop or force_aspect > 0:
        return crop
    g = crop[..., 0] if crop.ndim == 3 else crop
    w = int(g.shape[1])
    if w <= 8 or float(g.std()) < 3.0:
        return crop
    cols = np.nonzero((g > bin_thresh).sum(axis=0) >= 2)[0]
    if len(cols) == 0:
        return crop
    rng = content_range_to_crop(int(cols[0]), int(cols[-1]), w,
                                margin_pct=margin_pct, min_gain=min_gain)
    if rng is None:
        return crop
    lo, cw = rng
    return crop[:, lo:lo + cw]


def crop_after_aspect(img: np.ndarray, *, autocrop: bool = True,
                      margin_pct: int = 10, min_gain: float = 0.1) -> np.ndarray:
    """`force_aspect > 0` 时，在**已定比例**的图上再按内容列裁。

    ## 顺序很关键：必须"先定比例、后裁切"（⑦），不能"先裁再定比例"（⑥）
    实测（生产口径：段代表帧 + 数值 tol=1 误读数）：

    | 顺序 | test5 | test6 |
    |---|---:|---:|
    | ① 不裁（原行为） | 7 | 17 |
    | ⑥ 先裁再定比例 | 9 | 5 |
    | **⑦ 先定比例再裁** | **0** | **0** |

    先裁会改变内容的宽高比，再拉到 force 宽度就引入畸变；先定比例则
    force 宽度作用于完整 ROI，裁掉的只是定比例后残留的空白边。
    （fa=0 时调本函数反而更差：test2 52→80、test 78→127 —— 调用方只在
    force_aspect > 0 时使用本函数，fa=0 走 `crop_to_content`。）

    ## 阈值必须现算
    不能用校准二值阈值——那是**原始灰度**的阈值，而这里输入已过
    force_aspect 缩放 + gamma，数值分布完全不同 → 对当前图现算 Otsu。
    余量数学复用 `content_range_to_crop`。
    """
    if not autocrop:
        return img
    g = img[..., 0] if img.ndim == 3 else img
    w = int(g.shape[1])
    if w <= 8 or float(g.std()) < 3.0:
        return img
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    cols = np.nonzero((g > _otsu(g8)).sum(axis=0) >= 2)[0]
    if len(cols) == 0:
        return img
    rng = content_range_to_crop(int(cols[0]), int(cols[-1]), w,
                                margin_pct=margin_pct, min_gain=min_gain)
    if rng is None:
        return img
    lo, cw = rng
    return img[:, lo:lo + cw]


def preprocess_standard(crop: np.ndarray, force_aspect: float = 0.0,
                        gamma: "float | None" = None) -> np.ndarray:
    """标准预处理：resize 到 OCR_TARGET_H 高 + 可选强制宽高比 + 灰度 gamma。

    force_aspect > 0 时强制横向宽度 = OCR_TARGET_H × force_aspect（px，
    宽高比固定；可能放大或缩小——"force" 语义，非上限）。0 = 按原宽高比
    resize。输出 float32（与 cv2 路径数值差 <= 1e-5）。

    gamma：灰度对比度增强指数（255*(gray/255)^g）。None = 用 env
    OCR_GAMMA，都没有则 config.OCR_GAMMA（正式默认 2.0）。
    白字黄底等背景色块场景放大高段分离，平滑无裁剪不侵蚀笔画。
    gamma <= 0 跳过灰度变换（保留 RGB，回退旧行为）；
    灰度权重与 segment 灰度共用 config.GRAY_RGB_WEIGHTS。

    宽度 pad（fill_width）在 OCR 引擎 _resize_norm 层处理（替换固定 224），
    此处不 pad。
    """
    target_h = config.OCR_TARGET_H
    h, w = crop.shape[:2]
    new_w = max(1, int(w * target_h / h)) if h > 0 else w
    if force_aspect > 0:
        new_w = max(1, int(round(target_h * force_aspect)))
    if new_w == w and abs(target_h - h) <= config.OCR_RESIZE_TOL * target_h:
        # 目标尺寸已一致（或高差在容差内）→ 跳过无谓 resize；宽高任一需变
        # 都必须走 _np_resize（force_aspect 改宽时不能只比高度）
        resized = crop.astype(np.float32)
    else:
        resized = _np_resize(crop, new_w, target_h)
    if gamma is None:
        gamma = config.env_float(config.OCR_GAMMA_ENV, float(config.OCR_GAMMA))
    if gamma > 0:
        # 灰度 + gamma（正式预处理）：RGB 逐通道 gamma 视觉差异小、回归多
        # （tools/_gamma_misread_montage 对比），灰度版视觉更清晰、回归少。
        if resized.ndim == 2:
            gray = resized                                # 2D (H,W) 灰度输入
        elif resized.shape[-1] == 1:
            gray = resized[..., 0]                        # decord gray 输出
        else:
            gray = resized @ _GRAY_W                      # (h, w) float32
        resized = 255.0 * np.power(gray / 255.0, gamma)
        resized = np.stack([resized] * 3, axis=-1)
    return resized
