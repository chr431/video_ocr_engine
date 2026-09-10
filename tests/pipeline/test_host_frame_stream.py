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


class _BatchMarkedEx(_StubEx):
    """_batch_luma_out 按批号写标记值：批 i 的灰度全为 i。

    用于验证 yield 出去的灰度是拷贝——若返回 g_buf 视图，后续批的
    覆写会改写先前 yield 的帧（D1：merge_similar 拿到被覆写的帧）。"""


def test_yielded_grays_survive_batch_advance():
    """yield 的灰度跨批保持稳定（D1 修复回归，2026-09-10）。

    复用缓冲只服务批内计算；逃逸进 payload 的代表帧灰度必须是
    独立拷贝。修复前 host 侧 45/1089 合并判定失真（段数 1042 vs
    GPU 正确值 1083）。
    """
    from video_ocr_engine import _host_pipeline as hp

    class _MarkedEx(_StubEx):
        def __init__(self):
            super().__init__(yuv=False)
            self.batch = -1

        def _batch_luma_out(self, crops, out):
            self.batch += 1
            out[...] = self.batch
            return out

    crops = np.zeros((16, 12, 8), dtype=np.uint8)
    ex = _MarkedEx()
    out = list(_host_frame_stream(ex, list(range(40)), _FakeVr(crops), [], 100))
    # 3 批标记 0/1/2；帧 0（批 0）在流耗尽后必须仍是 0，而不是被
    # 批 2 覆写成的 2
    assert out[0][2][0, 0] == 0, "帧 0 灰度被后续批覆写（别名未修复）"
    assert out[15][2][0, 0] == 0
    assert out[16][2][0, 0] == 1
    assert out[39][2][0, 0] == 2
