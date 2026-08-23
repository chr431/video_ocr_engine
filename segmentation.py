"""分段用灰度 / Otsu / 聚类判别（生产与串行参考路径共用）。"""
from __future__ import annotations

import numpy as np

import engine_config as config
from video_utils import (_gray, _GRAY_W, _nv12_luma_full,
                         _nv12_batch_luma_full)


def _gray_batch(crops: np.ndarray) -> np.ndarray:
    """批量灰度：(B,H,W,3) → (B,H,W)；decord gray 输出 (B,H,W,1) 直接取通道。

    gray 输出模式（CPU/GPU 解码，decord ≥0.7.9）crops 已是 1 通道，跳过 matmul；
    yuv420 模式（≥0.7.10）请用 _gray_seg_yuv/_gray_seg_yuv_batch。
    """
    if crops.shape[-1] == 1:
        return crops[..., 0]
    return (crops.astype(np.float32) @ _GRAY_W).astype(np.uint8)


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


def _otsu(g: np.ndarray) -> int:
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    total = int(g.size)
    st = float((np.arange(256) * hist).sum())
    sb = 0.0
    wb = 0
    best = config.OTSU_FALLBACK_THRESH
    vmax = -1.0
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += t * hist[t]
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
    6 次切片错位累加求最大 3×3 窗口和（越界按 0），16µs/边，数值与
    uniform_filter 逐位一致（含边界，500 随机掩码最大差 0）。
    语义：真实数字变化必然产生 ≥5 像素连成 3×3 的密集簇（实测变帧恒=9）；
    噪声孤立像素的最大窗口和 < 5。C=5 下 test/test5/test6 0 漏检且段数更少。
    """
    if not diff.any():
        return 0.0
    s = diff.astype(np.int32)
    # 行向 3 列和（左右越界 0）
    c3 = s.copy()
    c3[:, 1:] += s[:, :-1]
    c3[:, :-1] += s[:, 1:]
    # 列向 3 行和（上下越界 0）
    w3 = c3.copy()
    w3[1:, :] += c3[:-1, :]
    w3[:-1, :] += c3[1:, :]
    return float(w3.max())
