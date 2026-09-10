"""视频/数据通用工具。"""
from __future__ import annotations
import os as _os
from pathlib import Path

from functools import lru_cache

import numpy as np

from video_ocr_engine.config import constants as config
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


#: limited→full 展开的 **256 项 LUT**（S6-f′）：Y 是 8-bit，展开是**单字节的
#: 纯函数**，所以 LUT 与逐元素浮点式**逐位一致**——用同一串 float32 运算
#: 构表（模块导入期一次）：floor((y-16)*(255/219)+0.5) 后 clip 0..255。
#: 收益（宿主路径 test5 3000 帧 yuv 实测）：`decode.luma_batch` 0.090s →
#: 0.020s（见 docs/log/2026-09-10-S6性能轮.md §8），且不再每批分配
#: B×h×w×4 的 float32 临时数组。
_Y_LIMITED_LUT = np.clip(
    np.floor((np.arange(256, dtype=np.float32) - np.float32(16.0))
             * np.float32(255.0 / 219.0) + np.float32(0.5)),
    0.0, 255.0).astype(np.uint8)


def _nv12_luma_full(crop: "np.ndarray", color_range: int = 0) -> "np.ndarray":
    """packed NV12 的 Y 平面按流 color_range 展开（复刻 decord gray 输出）。

    limited/tv(0)：floor((raw-16)*255/219 + 0.5) 后 clip 0..255 —— 与
    decord CPU swscale GRAY8 / GPU gray kernel 逐位一致；full/pc(1) 原样。
    """
    y = _nv12_luma(crop)
    if color_range == 1:
        return y
    return _Y_LIMITED_LUT[y]


def _nv12_batch_luma_full(crops: "np.ndarray", color_range: int = 0) -> "np.ndarray":
    y = _nv12_batch_luma(crops)
    if color_range == 1:
        return y
    return _Y_LIMITED_LUT[y]


def _nv12_batch_luma(crops: "np.ndarray") -> "np.ndarray":
    """批量取 packed NV12 的原始 Y 平面（(B, h+ceil(h/2), w) → (B, h, w)）。"""
    h = crops.shape[1] * 2 // 3
    return crops[:, :h]



def nv12_to_rgb(crop: "np.ndarray", color_range: int = 0) -> "np.ndarray":
    """packed NV12 → RGB（uint8，BT.601 矩阵）。

    Y/U/V 均为原始 8-bit，range 展开在本函数完成（DESIGN-REVIEW C9：旧
    docstring 一度称"Y 已由 decoder 展开"，与同模块 `_nv12_luma_full` 的
    原始 Y 语义矛盾）：
    - color_range=0（limited/tv，默认）：(y-16)/219 展开 + 1.164/1.596/
      2.017 矩阵——与 decord improc 系数一致（历史行为，默认不变）；
    - color_range=1（full/pc）：Y 原样 + 1.402/0.344/0.714/1.772 矩阵
      （旧版缺失该分支，full-range 流预览会偏色）。
    UV 为原始 4:2:0：chroma 按 2x2 块 nearest 上采样（decord RGB 路径的
    MPEG-2 siting 语义）。
    """
    if crop.ndim != 2:
        return crop[..., :3] if crop.ndim == 3 else crop
    rows, w = crop.shape
    h = rows * 2 // 3
    if color_range == 1:
        y = crop[:h].astype(np.float32) / 255.0
    else:
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
    if color_range == 1:
        r = y + 1.402 * vn
        g = y - 0.344136 * un - 0.714136 * vn
        b = y + 1.772 * un
    else:
        r = 1.164383 * y + 1.596027 * vn
        g = 1.164383 * y - 0.391762 * un - 0.812968 * vn
        b = 1.164383 * y + 2.017232 * un
    rgb = np.stack([r, g, b], axis=-1)
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return (rgb * 255.0).astype(np.uint8)


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
    支持 2D 灰度（(H,W)、3D 单通道（(H,W,1)）与 RGB（(H,W,3)）；
    2D/单通道输入输出保持原通道语义（2D 输出 2D）。
    移除 cv2 依赖后的轻量替代（EXE -83MB）。
    """
    src_h, src_w = img.shape[:2]
    if new_w == src_w and new_h == src_h:
        return img.astype(np.float32)
    x0, x1, y0, y1, wx, wy = _resize_map(src_w, src_h, new_w, new_h)
    f = img.astype(np.float32)
    one_ch = f.ndim == 2
    if one_ch:
        f = f[..., None]
    wx3 = wx[None, :, None]
    wy3 = wy[:, None, None]
    out = ((1 - wx3) * (1 - wy3) * f[y0[:, None], x0[None, :]] +
           wx3 * (1 - wy3) * f[y0[:, None], x1[None, :]] +
           (1 - wx3) * wy3 * f[y1[:, None], x0[None, :]] +
           wx3 * wy3 * f[y1[:, None], x1[None, :]])
    return out[..., 0] if one_ch else out


def nvdec_available(video_path=None) -> bool:
    """轻量探测 NVDEC 解码是否可用（尝试用 GPU reader 打开视频）。

    video_path 为 None 时只探测 decord GPU 模块/上下文是否可用，不打开文件。
    **只缓存成功结果**：失败（含瞬态失败——驱动忙/显存压力/并发探测）不缓存，
    下次调用重新探测；成功按参数缓存（批量多集调用时避免每次重新打开视频；
    GPU 上下文在进程生命周期内稳定，成功结果缓存安全）。
    """
    try:
        return _nvdec_probe_success(video_path)
    except Exception:
        return False


@lru_cache(maxsize=64)
def _nvdec_probe_success(video_path=None) -> bool:
    """nvdec_available 的成功路径（失败抛异常 → 不进缓存）。"""
    from decord import VideoReader, gpu
    if video_path is None:
        from decord import gpu as _g
        _g(0)
        return True
    vr = VideoReader(str(video_path), ctx=gpu(0))
    del vr
    return True


@lru_cache(maxsize=1)
def tensorrt_available() -> bool:
    """轻量探测 TensorRT 是否可用（存在 nvinfer DLL 且绑定可导入）。

    结果缓存：DLL 搜索路径在进程内稳定，批量调用避免重复 glob 扫描。
    """
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
