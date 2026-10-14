"""分段纯函数测试（灰度 / Otsu / 3x3 聚类判别）。"""

from __future__ import annotations

import numpy as np

from video_ocr_engine.domain.segmentation import (
    _cluster_win3,
    _gray_batch,
    _gray_seg_yuv,
    _gray_seg_yuv_batch,
    _otsu,
)
from video_ocr_engine.domain.video_utils import _nv12_luma_full


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


def test_gray_seg_yuv_matches_nv12_luma_full():
    """decord yuv420 → 取 Y + range 展开，与 _nv12_luma_full 一致。"""
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


# ── 边界对规划器（GPU lane 投机预计算，C-84）────────────────────────
def test_boundary_pair_planner_matches_state_machine_all_emit():
    """全 emit 模拟：规划器产出的对 == 状态机实际提问的对（逐对相等）。

    这是投机命中的充分条件：上一边界发射 ⇒ last_emitted_rep == 上一段
    代表帧 == 规划器产出的 a 侧。"""
    import random

    from video_ocr_engine.domain.segmentation import BoundaryPairPlanner, SegmentStateMachine

    rng = random.Random(42)
    for _trial in range(50):
        n = rng.randint(2, 400)
        C = 5.0
        sharps = [rng.random() * 100.0 for _ in range(n)]
        # 首帧分数被两侧共同忽略（started 分支）；其余取阈值邻域值
        clusters = [0.0] + [float(rng.choice([0, 1, 4, 5, 6, 9])) for _ in range(n - 1)]
        frames = list(range(1000, 1000 + n))
        asked: list = []

        def on_similar(a, b):
            asked.append((a[0], b[0]))
            return False  # 全 emit

        sm = SegmentStateMachine(
            frames, C=C, on_emit=lambda seg, rep, frac: None, on_similar=on_similar
        )
        planner = BoundaryPairPlanner(C)
        planned: list = []
        for k, fi in enumerate(frames):
            pr = planner.feed(fi, sharps[k], clusters[k], (fi, None, sharps[k]))
            if pr is not None:
                planned.append((pr[0][0], pr[1][0]))
            sm.feed(k, fi, sharps[k], (fi, None, sharps[k]), cluster=clusters[k])
        fp = planner.finish()
        if fp is not None:
            planned.append((fp[0][0], fp[1][0]))
        sm.finish()
        assert planned == asked, f"trial={_trial} planned={planned[:8]} asked={asked[:8]}"


def test_boundary_pair_planner_reps_match_under_merges():
    """合并随机发生：闭合段代表帧序列仍逐位一致（merge 无关几何）。

    合并链下规划器的 a 侧（上一段代表帧）会偏离状态机的提问
    （last_emitted_rep 更旧）→ 消费端未命中走同步兜底；但 b 侧
    （闭合段代表帧）恒一致——这是「投机只改何时算、不改算出什么」
    的结构前提。"""
    import random

    from video_ocr_engine.domain.segmentation import BoundaryPairPlanner, SegmentStateMachine

    rng = random.Random(7)
    for _trial in range(50):
        n = rng.randint(2, 300)
        C = 5.0
        sharps = [rng.random() * 100.0 for _ in range(n)]
        clusters = [0.0] + [float(rng.choice([0, 0, 2, 5, 7])) for _ in range(n - 1)]
        frames = list(range(0, n))
        asked: list = []

        def on_similar(a, b):
            asked.append((a[0], b[0]))
            return rng.random() < 0.3  # 随机合并链

        sm = SegmentStateMachine(
            frames, C=C, on_emit=lambda seg, rep, frac: None, on_similar=on_similar
        )
        planner = BoundaryPairPlanner(C)
        planned: list = []
        for k, fi in enumerate(frames):
            pr = planner.feed(fi, sharps[k], clusters[k], (fi, None, sharps[k]))
            if pr is not None:
                planned.append(pr[1][0])
            sm.feed(k, fi, sharps[k], (fi, None, sharps[k]), cluster=clusters[k])
        fp = planner.finish()
        if fp is not None:
            planned.append(fp[1][0])
        sm.finish()
        # 状态机提问的 b 侧 = 闭合段代表帧；首段短路不提问（两侧同步缺席）
        assert planned == [b for _a, b in asked]
