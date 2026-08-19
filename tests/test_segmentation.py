"""分段纯函数测试（灰度 / Otsu / 3x3 聚类判别）。"""
from __future__ import annotations

import numpy as np
import pytest

from segmentation import (
    _apply_gamma, _cluster_win3, _gray_batch, _gray_seg_yuv,
    _gray_seg_yuv_batch, _otsu,
)
from video_utils import _nv12_luma_full


def test_gray_batch_single_and_rgb():
    rgb = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    g = _gray_batch(rgb)
    # Rec.601: R=255 -> 0.299*255 ≈ 76（四舍五入到 uint8）
    assert g.shape == (2, 4, 4)
    assert int(g[0, 0, 0]) == 76
    # 1 通道输入直接透传
    one = np.zeros((2, 4, 4, 1), dtype=np.uint8)
    one[..., 0] = 7
    assert np.array_equal(_gray_batch(one), one[..., 0])


def test_gray_seg_yuv_matches_nv12_luma_full(monkeypatch):
    """decord yuv420 → 取 Y + range 展开（SEG_GAMMA=0 时即 _nv12_luma_full）。"""
    monkeypatch.setenv("RVTOL_SEG_GAMMA", "0")  # 锁定分段 gamma，防环境干扰
    h, w = 16, 20
    y = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    crop = np.vstack([y, np.full((h // 2, w), 128, dtype=np.uint8)])
    assert np.array_equal(_gray_seg_yuv(crop, 0), _nv12_luma_full(crop, 0))
    assert np.array_equal(_gray_seg_yuv(crop, 1), _nv12_luma_full(crop, 1))


def test_seg_gray_yuv_batch_shape():
    h, w = 8, 12
    y = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    crop = np.vstack([y, np.full(((h + 1) // 2, w), 128, dtype=np.uint8)])
    crops = np.stack([crop, crop + 1])
    g = _gray_seg_yuv_batch(crops, 0)
    assert g.shape == (2, h, w)


def test_otsu_bimodal():
    g = np.zeros((64, 64), dtype=np.uint8)
    g[:32] = 70
    g[32:] = 180
    th = _otsu(g)
    # 干净双峰：Otsu 取第一个能完美分离的阈值（=较低峰位）
    assert 40 <= th <= 140, th


def test_otsu_fallback_flat():
    g = np.full((8, 8), 100, dtype=np.uint8)
    assert _otsu(g) == 127  # 无法分割 → 兜底阈值


def test_cluster_win3_dense_cluster_counts_9():
    """真实数字变化：≥5 像素连成 3x3 密集簇 → 窗口和 9。"""
    diff = np.zeros((10, 10), dtype=bool)
    diff[3:6, 3:6] = True
    assert _cluster_win3(diff) == 9.0


def test_cluster_win3_isolated_noise_below_5():
    """噪声孤立像素：最大 3x3 窗口和 < 5 → 视为未变。"""
    rng = np.random.default_rng(0)
    diff = rng.random((20, 20)) < 0.03
    assert _cluster_win3(diff) < 5.0


def test_cluster_win3_empty():
    assert _cluster_win3(np.zeros((5, 5), dtype=bool)) == 0.0


@pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0])
def test_apply_gamma_range(gamma):
    g = np.arange(256, dtype=np.uint8)
    out = _apply_gamma(g, gamma).astype(np.float64)
    assert out.min() >= 0.0 and out.max() <= 255.0
