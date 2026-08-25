"""video_utils 测试：NV12 Y 展开 / RGB 转换 / resize / 标准预处理。"""
from __future__ import annotations

import numpy as np
import pytest

from video_utils import (
    _gray, _np_resize, _nv12_luma, _nv12_luma_full, _preprocess_standard,
    nv12_to_rgb,
)


def _gray_expected(raw_y: np.ndarray, color_range: int) -> np.ndarray:
    if color_range == 1:
        return raw_y
    v = (raw_y.astype(np.float32) - 16.0) * (255.0 / 219.0)
    return np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)


def test_luma_full_limited_and_full():
    y = np.arange(256, dtype=np.uint8).reshape(16, 16)
    crop = np.zeros((16 + 8, 16), dtype=np.uint8)
    crop[:16] = y
    assert _nv12_luma(crop).shape == (16, 16)
    assert np.array_equal(_nv12_luma_full(crop, 1), y)
    assert np.array_equal(_nv12_luma_full(crop, 0), _gray_expected(y, 0))


def test_nv12_to_rgb_shape_even_and_odd():
    # 灰度 YUV（U=V=128）：R=G=B，形状为 (h, w, 3)
    for h, w in ((4, 6), (5, 7), (1, 1)):
        rows = h + (h + 1) // 2
        crop = np.zeros((rows, w), dtype=np.uint8)
        crop[:h] = 180
        crop[h:] = 128
        rgb = nv12_to_rgb(crop)
        assert rgb.shape == (h, w, 3)
        assert rgb.dtype == np.uint8
        # decord BT.601 矩阵语义：gray = clip(1.164383*(Y-16))
        expect = np.clip(np.rint(1.164383 * (180 - 16)), 0, 255)
        assert int(np.abs(rgb.astype(int) - int(expect)).max()) <= 1
        assert np.all(rgb[..., 0] == rgb[..., 1])
        assert np.all(rgb[..., 1] == rgb[..., 2])


def test_nv12_to_rgb_passthrough_rgb_array():
    arr = np.zeros((2, 3, 3), dtype=np.uint8)
    arr[..., 0] = 1
    assert np.array_equal(nv12_to_rgb(arr), arr)


def test_np_resize_identity_and_shape():
    img = np.random.default_rng(0).integers(0, 256, (48, 90, 3), dtype=np.uint8)
    out = _np_resize(img, 90, 48)
    assert out.shape == (48, 90, 3)
    assert out.dtype == np.float32


def test_preprocess_standard_target_height_and_gamma(monkeypatch):
    monkeypatch.delenv("OCR_GAMMA", raising=False)
    crop = np.random.default_rng(1).integers(0, 256, (60, 200, 3), dtype=np.uint8)
    out = _preprocess_standard(crop)
    assert out.shape[0] == 48  # OCR_TARGET_H
    assert out.dtype == np.float32


def test_preprocess_standard_force_aspect():
    out = _preprocess_standard(np.zeros((60, 90, 3), dtype=np.uint8), force_aspect=2.0)
    assert out.shape[0] == 48
    assert out.shape[1] == 96  # 48 × 2.0


def test_gray_weights_rec601():
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    assert int(_gray(rgb)[0, 0]) == 76


def test_preprocess_standard_2d_gray_input(monkeypatch):
    # 2D 灰度输入（GPU 管线/代表帧 D2H 等场景）：直接 gamma，不按 RGB matmul
    monkeypatch.delenv("OCR_GAMMA", raising=False)
    crop = np.random.default_rng(2).integers(0, 256, (48, 90), dtype=np.uint8)
    out = _preprocess_standard(crop)
    assert out.shape == (48, 90, 3)
    assert out.dtype == np.float32
