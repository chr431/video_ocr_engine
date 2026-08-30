"""临时探针 I：量化宿主路径"Python 层每帧开销"（C 重写决策输入）。

用合成帧（真实 ROI 尺寸）跑与 _host_frame_stream / _host_segment_frames
完全相同的逐帧逻辑，剥离解码与 OCR，得到纯 Python+numpy 的每帧成本。
再对照各解码速率，算出"Python 成为瓶颈"的临界点。
"""
from __future__ import annotations

import time

import numpy as np

import engine_config as config
from segmentation import _cluster_win3
from video_ocr_engine._host_pipeline import _host_segment_frames


class FakeEx:
    """最小 FieldExtractor 替身（只提供分段状态机需要的钩子）。"""
    _C = config.SEG_C
    _merge_similar = True
    _merge_similar_threshold = config.SEG_MERGE_SIMILAR_THRESHOLD
    _merge_text_sep = config.DEFAULT_MERGE_TEXT_SEP
    _bin_thresh = 86
    _merge_max_changed_pixels = 64
    _profile_enabled = False
    n_seg_checks = 0

    def _prof_end(self, *a):
        pass

    def _cancel(self):
        pass

    def _progress(self, *a):
        pass

    def _merge_effective_mode(self):
        return 'binary'

    def _segments_similar(self, a, b):
        from video_utils import _text_sep_gray
        a = _text_sep_gray(a, 'binary', th=self._bin_thresh)
        b = _text_sep_gray(b, 'binary', th=self._bin_thresh)
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        if float(d.mean()) > self._merge_similar_threshold:
            return False
        return int(np.sum(d > 10)) <= self._merge_max_changed_pixels


def bench(h, w, n, seg_rate=0.15):
    """seg_rate：切段频率（模拟真实：test5 1083/3000≈36%，
    新三国01 1151/9178≈12%）。

    生成方式：内容每 1/seg_rate 帧变一次（而非逐帧翻转），使边界数与
    seg_rate 匹配——否则每帧都是边界，会把 _segments_similar 的调用
    次数放大到最坏情况，高估成本。
    """
    rng = np.random.default_rng(0)
    frames = list(range(n))
    grays = rng.integers(0, 256, size=(n, h, w), dtype=np.uint8)
    period = max(2, int(round(1.0 / seg_rate)))
    # 每 period 帧切换一次内容模板 → 边界数 ≈ n/period
    tmpl = [grays[0], grays[0] ^ 0xFF]
    out = np.empty_like(grays)
    for i in range(n):
        out[i] = tmpl[(i // period) % 2]

    def stream():
        for i in range(n):
            g = out[i]
            b = g > 86
            yield (frames[i], None, g, float(g.std()), b, None)

    ex = FakeEx()
    segs = []
    emitted = []

    def emit(seg, rf, rc, rd, rg, frac):
        emitted.append(len(seg))

    t0 = time.perf_counter()
    _host_segment_frames(ex, frames, stream(), debug_tag=None,
                         progress_prefix='', emit=emit, segs=segs)
    dt = time.perf_counter() - t0
    per = dt / n * 1e6
    print(f"  ROI {w}x{h:2d}  {n:6d} 帧: {dt:7.3f}s  "
          f"{per:7.2f} µs/帧   段数={len(segs)}")
    return per


def main():
    print("=== 宿主路径 Python 逐帧成本（解码/OCR 已剥离）===")
    print("  -- 真实 ROI 尺寸 --")
    p_narrow = bench(33, 106, 7223, 0.35)      # test5 速度数字（2492/7223≈34%）
    p_wide = bench(25, 407, 9000, 0.13)         # 字幕条（1151/9178≈12%）
    print("  -- 宽 ROI 真实场景 --")
    # text_test（video_subtitle_extractor）：1920×1080，ROI 560,996,1360,1048
    # = 800×52 = 41.6k px，6000 帧 233 段（段率 ≈3.9%，字幕保持型，取 0.05）
    p_text = bench(52, 800, 6000, 0.05)
    print("  -- 大 ROI（ROI 变大时 numpy 开销随面积增长）--")
    p_big = bench(200, 800, 2000, 0.30)
    p_huge = bench(600, 1600, 500, 0.30)

    print("\n=== 瓶颈临界点（Python 逐帧成本 vs 解码逐帧成本）===")
    print("  解码速率      每帧预算    窄ROI   字幕条   text_test(41.6k px)")
    for rate in (1000, 2000, 4000, 8000, 12000, 20000):
        budget = 1e6 / rate
        f1 = "是" if p_narrow > budget * 0.5 else "否"
        f2 = "是" if p_wide > budget * 0.5 else "否"
        f3 = "是" if p_text > budget * 0.5 else "否"
        print(f"  {rate:6d} fps  {budget:8.1f} µs   "
              f"占用 {p_narrow/budget*100:5.1f}% ({f1})   "
              f"占用 {p_wide/budget*100:5.1f}% ({f2})   "
              f"占用 {p_text/budget*100:5.1f}% ({f3})")

    print("\n=== 单项：_cluster_win3（每帧一次）===")
    for (h, w) in ((33, 106), (25, 407), (200, 800), (600, 1600)):
        d = np.zeros((h, w), dtype=bool)
        d[::7, ::5] = True
        t0 = time.perf_counter()
        for _ in range(2000):
            _cluster_win3(d)
        dt = (time.perf_counter() - t0) / 2000 * 1e6
        print(f"  ROI {w}x{h:2d}: {dt:6.2f} µs/次")


if __name__ == "__main__":
    main()
