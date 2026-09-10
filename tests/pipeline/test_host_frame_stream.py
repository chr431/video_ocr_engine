"""宿主帧流批量灰度缓冲复用（DESIGN-REVIEW C3；S3-3a 起改为 Spec 版）。

yuv(NV12) 的 g_buf 初值/复用条件曾双双写错（2//3 乘在宽度上、两元比一元
永假）→ yuv 模式从未复用、每批重新分配。本测试统计 batch_luma（重新分配
路径）与 batch_luma_out（复用路径）回调次数：复用条件成立时前者不该被调。
"""
import numpy as np

from video_ocr_engine.pipeline.host_backend import HostRunSpec, _frame_stream


class _FakeNds:
    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape


class _FakeVr:
    """恒定形状的假 reader：每次 get_batch 返回同形满批数组。"""

    def __init__(self, arr):
        self._arr = arr

    def get_batch(self, frames, roi=None):
        return _FakeNds(self._arr)


class _Counters:
    def __init__(self, yuv):
        self.yuv = yuv
        self.luma_calls = 0        # 重新分配路径（复用失效时会走）
        self.out_calls = 0         # 复用路径
        self.out_shapes = []


def _spec(ct: _Counters, batch_marker=False) -> HostRunSpec:
    def batch_luma(crops):
        ct.luma_calls += 1
        h = (crops.shape[1] * 2 // 3 if ct.yuv else crops.shape[1])
        return np.zeros((len(crops), h, crops.shape[2]), dtype=np.uint8)

    def batch_luma_out(crops, out):
        ct.out_calls += 1
        ct.out_shapes.append(out.shape)
        if batch_marker:
            ct.batch += 1
            out[...] = ct.batch
        else:
            out[...] = 0
        return out

    return HostRunSpec(
        frame_start=0, frame_end=None, sample_stride=1, roi=(0, 0, 7, 9),
        C=5.0, merge_similar=True, keep_crops=False, yuv_output=ct.yuv,
        segments_similar=lambda a, b: False,
        crop_luma=lambda c: c[..., 0] if c.ndim == 3 else c,
        batch_luma=batch_luma, batch_luma_out=batch_luma_out,
        crop_is_expected=lambda c, h, w: False,
        open_vr=lambda: None, start_ocr_session=lambda e: None)


def _run(ct, crop_shape, n_frames=40, batch_marker=False):
    # DECODE_BATCH_SIZE=16 → 40 帧 = 3 批（16/16/8）
    crops = np.zeros(crop_shape, dtype=np.uint8)
    vr = _FakeVr(crops)
    out = list(_frame_stream(_spec(ct, batch_marker),
                             list(range(n_frames)), vr, [], 100,
                             with_dev=False))
    assert len(out) == n_frames
    return ct


def test_yuv_gray_buffer_reused():
    """yuv 模式：luma 输出形状应为 (B, rows*2//3, W) 且全程复用（C3 回归）。"""
    rows, w = 15, 8          # rows*2//3 = 10
    ct = _run(_Counters(yuv=True), (16, rows, w))
    assert ct.luma_calls == 0, "yuv 复用失效：走了重新分配路径"
    assert ct.out_calls == 3
    assert all(s == (16, 10, w) for s in ct.out_shapes)


def test_gray_gray_buffer_reused():
    """gray 模式：复用行为保持不变（回归守卫）。"""
    ct = _run(_Counters(yuv=False), (16, 12, 8))
    assert ct.luma_calls == 0
    assert ct.out_calls == 3
    assert all(s == (16, 12, 8) for s in ct.out_shapes)


def test_yielded_grays_survive_batch_advance():
    """yield 的灰度跨批保持稳定（D1 修复回归，2026-09-10）。

    复用缓冲只服务批内计算；逃逸进 payload 的代表帧灰度必须是独立拷贝。
    修复前 host 侧 45/1089 合并判定失真（段数 1042 vs GPU 正确值 1083）。
    """
    ct = _Counters(yuv=False)
    ct.batch = -1
    crops = np.zeros((16, 12, 8), dtype=np.uint8)
    out = list(_frame_stream(_spec(ct, batch_marker=True),
                             list(range(40)), _FakeVr(crops), [], 100,
                             with_dev=False))
    # 3 批标记 0/1/2；帧 0（批 0）在流耗尽后必须仍是 0，而不是被
    # 批 2 覆写成的 2
    assert out[0][2][0, 0] == 0, "帧 0 灰度被后续批覆写（别名未修复）"
    assert out[15][2][0, 0] == 0
    assert out[16][2][0, 0] == 1
    assert out[39][2][0, 0] == 2
