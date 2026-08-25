"""视频/数据通用工具。"""
from __future__ import annotations
import os as _os
from dataclasses import dataclass
from pathlib import Path

from functools import lru_cache

import numpy as np

import engine_config as config
# OCR/分段预处理灰度权重（Rec.601，与 config.GRAY_RGB_WEIGHTS 单一事实源一致）。
_GRAY_W = np.asarray(config.GRAY_RGB_WEIGHTS, dtype=np.float32)


def _gray(crop: np.ndarray) -> np.ndarray:
    """RGB → 灰度（uint8）。权重与 _GRAY_W 一致（分段与 OCR 预处理共用）。

    decord gray 输出 (H,W,1) 直接取通道（跳过 matmul）。
    """
    if crop.shape[-1] == 1:
        return crop[..., 0]
    return (crop.astype(np.float32) @ _GRAY_W).astype(np.uint8)


# ── decord yuv420（packed NV12）转换 ───────────────────────────────────
# 布局：前 h 行 = 原始 Y，后 ceil(h/2) 行 = interleaved U/V（原始 4:2:0）。
# get_color_range() 给出 0=limited/tv、1=full/pc；按它展开 Y 后与 decord
# gray 输出逐位一致。
def _nv12_luma(crop: "np.ndarray") -> "np.ndarray":
    """取 packed NV12 的原始 Y 平面（(h+ceil(h/2), w) → (h, w)）。"""
    h = crop.shape[0] * 2 // 3
    return crop[:h]


def _nv12_batch_luma(crops: "np.ndarray") -> "np.ndarray":
    """批量取 packed NV12 的原始 Y 平面（(B, h+ceil(h/2), w) → (B, h, w)）。"""
    h = crops.shape[1] * 2 // 3
    return crops[:, :h]


def _nv12_luma_full(crop: "np.ndarray", color_range: int = 0) -> "np.ndarray":
    """packed NV12 的 Y 平面按流 color_range 展开（复刻 decord gray 输出）。

    limited/tv(0)：floor((raw-16)*255/219 + 0.5) 后 clip 0..255 —— 与
    decord CPU swscale GRAY8 / GPU gray kernel 逐位一致；full/pc(1) 原样。
    """
    y = _nv12_luma(crop)
    if color_range == 1:
        return y
    v = (y.astype(np.float32) - 16.0) * (255.0 / 219.0)
    return np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)


def _nv12_batch_luma_full(crops: "np.ndarray", color_range: int = 0) -> "np.ndarray":
    y = _nv12_batch_luma(crops)
    if color_range == 1:
        return y
    v = (y.astype(np.float32) - 16.0) * (255.0 / 219.0)
    return np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)


def nv12_to_rgb(crop: "np.ndarray") -> "np.ndarray":
    """packed NV12 → RGB（uint8，BT.601 limited 色度矩阵）。

    Y 已由 decoder 按流 range 展开（与 gray 输出一致），UV 为原始
    4:2:0：chroma 按 2x2 块 nearest 上采样（decord RGB 路径的 MPEG-2
    siting 语义），再做与 decord improc 相同的 BT.601 矩阵转换。
    """
    if crop.ndim != 2:
        return crop[..., :3] if crop.ndim == 3 else crop
    rows, w = crop.shape
    h = rows * 2 // 3
    # decord yuv420 的 Y/U/V 均为原始 8-bit。RGB 转换复刻 decord improc
    # 的 BT.601 矩阵语义（系数 1.164/1.596/2.017，0..1 输入、偏移 16/128）
    y = (crop[:h].astype(np.float32) - 16.0) / 255.0
    uv_rows = (h + 1) // 2
    uv = crop[h:h + uv_rows, :w // 2 * 2]
    u = uv[:, 0::2].astype(np.float32)
    v = uv[:, 1::2].astype(np.float32)
    # nearest 上采样到 luma 分辨率（每个 2x2 块共用同一 chroma 样本；
    # 奇数宽/高时末行末列补中性色度 128）
    if u.shape[0] * 2 < h:
        u = np.pad(u, ((0, h - u.shape[0] * 2), (0, 0)),
                   mode='constant', constant_values=128)
    if v.shape[0] * 2 < h:
        v = np.pad(v, ((0, h - v.shape[0] * 2), (0, 0)),
                   mode='constant', constant_values=128)
    u = np.repeat(u, 2, axis=0)[:h]
    v = np.repeat(v, 2, axis=0)[:h]
    if u.shape[1] * 2 < w:
        u = np.pad(u, ((0, 0), (0, w - u.shape[1] * 2)),
                   mode='constant', constant_values=128)
    if v.shape[1] * 2 < w:
        v = np.pad(v, ((0, 0), (0, w - v.shape[1] * 2)),
                   mode='constant', constant_values=128)
    u = np.repeat(u, 2, axis=1)[:, :w]
    v = np.repeat(v, 2, axis=1)[:, :w]
    un = (u - 128.0) / 255.0
    vn = (v - 128.0) / 255.0
    r = 1.164383 * y + 1.596027 * vn
    g = 1.164383 * y - 0.391762 * un - 0.812968 * vn
    b = 1.164383 * y + 2.017232 * un
    rgb = np.stack([r, g, b], axis=-1)
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return (rgb * 255.0).astype(np.uint8)


@dataclass
class VideoMetadata:
    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    codec: str
    frame_count: int


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


@lru_cache(maxsize=64)
def _resize_map(src_w: int, src_h: int, new_w: int, new_h: int):
    """双线性坐标映射（缓存）：只依赖输入/输出尺寸，与像素无关。

    主流水线每帧同一 ROI 调 _np_resize（目标尺寸恒定）→ 映射缓存
    后每帧省去 arange/clip/cast 等 ~60% 的 numpy 工作量。
    """
    scale_x = src_w / new_w
    scale_y = src_h / new_h
    src_x = np.clip((np.arange(new_w) + 0.5) * scale_x - 0.5, 0, src_w - 1)
    src_y = np.clip((np.arange(new_h) + 0.5) * scale_y - 0.5, 0, src_h - 1)
    x0 = src_x.astype(np.int32)
    y0 = src_y.astype(np.int32)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    wx = (src_x - x0).astype(np.float32)
    wy = (src_y - y0).astype(np.float32)
    return x0, x1, y0, y1, wx, wy


def _np_resize(img: "np.ndarray", new_w: int, new_h: int) -> "np.ndarray":
    """双线性 resize（float32），与 cv2.resize INTER_LINEAR 像素对齐一致。

    坐标映射复刻 OpenCV：src = (dst + 0.5) * scale - 0.5（像素中心对齐）。
    与 cv2 的数值差 <= 1e-5（浮点累加顺序），无实际影响；输出 float32。
    移除 cv2 依赖后的轻量替代（EXE -83MB）。
    """
    src_h, src_w = img.shape[:2]
    if new_w == src_w and new_h == src_h:
        return img.astype(np.float32)
    x0, x1, y0, y1, wx, wy = _resize_map(src_w, src_h, new_w, new_h)
    f = img.astype(np.float32)
    wx3 = wx[None, :, None]
    wy3 = wy[:, None, None]
    return ((1 - wx3) * (1 - wy3) * f[y0[:, None], x0[None, :]] +
            wx3 * (1 - wy3) * f[y0[:, None], x1[None, :]] +
            (1 - wx3) * wy3 * f[y1[:, None], x0[None, :]] +
            wx3 * wy3 * f[y1[:, None], x1[None, :]])


def _preprocess_standard(crop: "np.ndarray", force_aspect: float = 0.0,
                         gamma: "float | None" = None) -> "np.ndarray":
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
        _env = _os.environ.get(config.OCR_GAMMA_ENV)
        gamma = float(_env) if _env else float(config.OCR_GAMMA)
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


def _box_blur(img: "np.ndarray", k: int) -> "np.ndarray":
    """纯 numpy 盒式模糊（用 sliding_window_view，无需 cv2/scipy）。"""
    if k <= 1:
        return img.astype(np.float32).copy()
    pad = k // 2
    p = np.pad(img, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(p, (k, k))
    return win.mean(axis=(2, 3)).astype(np.float32)


def _text_sep_gray(gray: "np.ndarray", mode: str = "contrast",
                   th: "int | None" = None) -> "np.ndarray":
    """从背景中分离字幕文字的灰度图（实验）。

    mode:
      - "contrast"：局部背景估计 + 绝对差分，突出文字笔画/边缘；
      - "binary"：用阈值把文字变白、背景变黑。
    """
    g = gray.astype(np.float32)
    if mode == "binary":
        if th is None:
            th = int(np.mean(g))
        return np.where(g > th, 255.0, 0.0).astype(np.float32)
    k = max(9, min(31, int(round(gray.shape[0] * 0.6 / 2) * 2 + 1)))
    bg = _box_blur(g, k)
    sep = np.abs(g - bg)
    p95 = float(np.percentile(sep, 95))
    if p95 > 1.0:
        sep = np.clip(sep / p95 * 255.0, 0.0, 255.0)
    return sep.astype(np.float32)


def open_decord_vr(video_path, force_cpu: bool = False):
    """Open video with decord — GPU (NVDEC) preferred, CPU fallback.

    Returns (VideoReader, label) where label is ``'GPU'`` or ``'CPU'``.
    Set ``DECORD_FORCE_CPU=1`` in the environment or pass *force_cpu=True*
    to skip GPU even when available.
    """
    from decord import VideoReader as _VR

    _vr = None
    _label = "CPU"
    _force = force_cpu or _os.environ.get(
        config.DECORD_FORCE_CPU_ENV, "").strip() == "1"

    if not _force:
        try:
            from decord import gpu as _decord_gpu
            _vr = _VR(str(video_path), ctx=_decord_gpu(0))
            _label = "GPU"
        except Exception:
            pass

    if _vr is None:
        try:
            from decord import cpu as _decord_cpu
            _vr = _VR(str(video_path), ctx=_decord_cpu(0))
        except ModuleNotFoundError:
            raise RuntimeError(
                "decord 未安装（需要自建 fork，PyPI 版不支持）。"
                "请运行 setup_venv.bat 或从 chr431/decord 获取发布产物到 _decord_build\\")
        except Exception as _e:
            raise RuntimeError(f"decord 无法打开视频: {_e}")

    return _vr, _label


def nvdec_available(video_path=None) -> bool:
    """轻量探测 NVDEC 解码是否可用（尝试用 GPU reader 打开视频）。

    video_path 为 None 时只探测 decord GPU 模块/上下文是否可用，不打开文件。
    """
    try:
        from decord import VideoReader, gpu
        if video_path is None:
            from decord import gpu as _g
            _g(0)
            return True
        vr = VideoReader(str(video_path), ctx=gpu(0))
        del vr
        return True
    except Exception:
        return False


def tensorrt_available() -> bool:
    """轻量探测 TensorRT 是否可用（存在 nvinfer DLL 且绑定可导入）。"""
    try:
        import tensorrt  # noqa: F401 — shim / binding 导入
        import tensorrt_bindings
        pkg = Path(tensorrt_bindings.__file__).resolve().parent
        candidates = list(pkg.glob("nvinfer*.dll"))
        libs = pkg.parent / "tensorrt_libs"
        if libs.is_dir():
            candidates.extend(libs.glob("nvinfer*.dll"))
        for entry in _os.environ.get("PATH", "").split(_os.pathsep):
            if entry:
                d = Path(entry)
                if d.is_dir():
                    candidates.extend(d.glob("nvinfer*.dll"))
        return bool(candidates)
    except Exception:
        return False


def rss_mb() -> float:
    """当前进程 RSS（MB）。psutil 缺失返回 -1。"""
    try:
        import psutil
        return psutil.Process(_os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def sum_nbytes(seq) -> int:
    """序列中 ndarray/bytes 元素的总字节数（兼容 (frame, bytes) 二元组）。"""
    s = 0
    for x in seq:
        if hasattr(x, "nbytes"):
            s += x.nbytes
        elif hasattr(x, "__len__") and len(x) == 2:
            if hasattr(x[1], "nbytes"):
                s += x[1].nbytes
    return s
