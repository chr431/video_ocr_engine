"""_host_frame_stream 批量灰度缓冲复用（DESIGN-REVIEW C3）。

yuv(NV12) 的 g_buf 初值/复用条件曾双双写错（2//3 乘在宽度上、两元比一元
永假）→ yuv 模式从未复用、每批重新分配。本测试用桩 ex 统计
_batch_luma（重新分配路径）与 _batch_luma_out（复用路径）的调用次数：
只要复用条件成立，_batch_luma 就不该被调用。
"""
import numpy as np

from video_ocr_engine._host_pipeline import _host_frame_stream


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


class _StubEx:
    def __init__(self, yuv):
        self._yuv_output = yuv
        self._roi = (0, 0, 7, 9)
        self.luma_calls = 0        # 重新分配路径（复用失效时会走）
        self.out_calls = 0         # 复用路径
        self.out_shapes = []

    def _prof_end(self, *args, **kwargs):
        pass

    def _batch_luma(self, crops):
        self.luma_calls += 1
        h = (crops.shape[1] * 2 // 3 if self._yuv_output
             else crops.shape[1])
        return np.zeros((len(crops), h, crops.shape[2]), dtype=np.uint8)

    def _batch_luma_out(self, crops, out):
        self.out_calls += 1
        self.out_shapes.append(out.shape)
        h = (crops.shape[1] * 2 // 3 if self._yuv_output
             else crops.shape[1])
        out[...] = 0
        return out


def _run(ex, crop_shape, n_frames=40):
    # DECODE_BATCH_SIZE=16 → 40 帧 = 3 批（16/16/8）
    crops = np.zeros(crop_shape, dtype=np.uint8)
    vr = _FakeVr(crops)
    out = list(_host_frame_stream(ex, list(range(n_frames)), vr, [], 100))
    assert len(out) == n_frames
    return ex


def test_yuv_gray_buffer_reused():
    """yuv 模式：luma 输出形状应为 (B, rows*2//3, W) 且全程复用（C3 回归）。"""
    rows, w = 15, 8          # rows*2//3 = 10
    ex = _run(_StubEx(yuv=True), (16, rows, w))
    assert ex.luma_calls == 0, "yuv 复用失效：走了重新分配路径"
    assert ex.out_calls == 3
    assert all(s == (16, 10, w) for s in ex.out_shapes)


def test_gray_gray_buffer_reused():
    """gray 模式：复用行为保持不变（回归守卫）。"""
    ex = _run(_StubEx(yuv=False), (16, 12, 8))
    assert ex.luma_calls == 0
    assert ex.out_calls == 3
    assert all(s == (16, 12, 8) for s in ex.out_shapes)
