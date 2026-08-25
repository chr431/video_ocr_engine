"""CPU+NVDEC 混合解码读取器 v2：kfe 分片 + 双解码生产者竞争。

思路来源（2026-08-25 用户提案）：固定 split 无法跨机器自适应——CPU 慢时
NVDEC 被饿、CPU 快时反之。改为 dual_pipeline kfe 的同款思路：

  - 采样帧序列按关键帧边界切成分片（复用 _keyframe_every_chunks）；
  - CPU 与 NVDEC 两个解码器线程作为生产者，从共享分片队列动态取片，
    谁快谁多拿（机器自适应，无饿死）；
  - 解出的 ROI 帧按全局帧序交付给唯一消费者（宿主校准/分段/OCR 零改动，
    单 TRT 后端不变）；
  - in-flight 分片数上限约束内存（默认 2 片）；分片边界落关键帧使两个
    生产者的跳片 seek 都落在关键帧上（便宜）；相邻片连续时免 seek。

对外仍是 VideoReader 同形替身：len / get_batch / next_roi / seek_accurate
/ get_*；正确性依赖 v0.7.8+ 双后端 YUV420 逐位一致。
"""
from __future__ import annotations

import bisect
import threading

import numpy as np


class _Batch:
    """最小 decord NDArray 兼容壳：asnumpy() / shape。"""

    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape


class HybridDecoder:
    """双解码生产者竞争分片的混合读取器（对下游透明）。"""

    def __init__(self, ex, gpu_vr, *, max_chunks: int = 16,
                 cpu_threads: int = 0, inflight: int = 2,
                 min_gap: int = 16):
        self._gpu = gpu_vr
        self._ex = ex
        self._roi = (ex._roi[0], ex._roi[1], ex._roi[2] + 1, ex._roi[3] + 1)
        total = len(gpu_vr)
        f0 = int(ex._frame_start or 0)
        f1 = min(int(ex._frame_end or total), total)
        if f1 - f0 < 4:
            raise ValueError('window too short for hybrid decode')
        self._f0, self._f1 = f0, f1
        self._max_chunks = max(2, int(max_chunks))
        self._min_gap = max(1, int(min_gap))
        self._inflight = max(1, int(inflight))
        nt_kw = {}
        if cpu_threads and cpu_threads > 0:
            nt_kw['num_threads'] = cpu_threads
        from decord import cpu as _cpu
        self._cpu = ex._open_decord_reader(_cpu(0), {}, **nt_kw)

        self._stop = threading.Event()
        self._err = []
        self._cv = threading.Condition()
        self._chunks = []      # {'fis','data','off','done'}
        self._starts = []      # 每片首帧（bisect 用）
        self._pending = []     # 待领取的分片下标（FIFO）
        self._tokens = 0       # 可新开分片的容量令牌
        self._begun = False
        self._seq_fi = None
        self._threads = []

    # ─────────────── 分片生成与启动（frames 就绪后调用） ───────────────

    def hybrid_begin(self, frames) -> None:
        if self._begun:
            return
        self._begun = True
        fr = [f for f in frames if self._f0 <= f < self._f1]
        if len(fr) < 4:
            raise ValueError('hybrid: sampled frames too few')
        try:
            keys = [int(k) for k in self._gpu.get_key_indices()]
        except Exception:
            keys = []
        try:
            specs = type(self._ex)._keyframe_every_chunks(
                fr, keys, fr[0], fr[-1] + 1,
                max(1, int(getattr(self._ex, '_sample_stride', 1))),
                self._min_gap, self._max_chunks)
        except Exception:
            specs = []
        if len(specs) < 2:
            n = min(self._max_chunks, max(2, len(fr) // 256))
            step = (len(fr) + n - 1) // n
            specs = [(fr[i], fr[min(i + step, len(fr)) - 1] + 1)
                     for i in range(0, len(fr), step)]
        for a, b in specs:
            fis = [f for f in fr if a <= f < b]
            if not fis:
                continue
            self._chunks.append({'fis': fis, 'data': [], 'off': 0,
                                 'done': False})
            self._starts.append(fis[0])
        n = len(self._chunks)
        self._pending = list(range(n))
        with self._cv:
            self._tokens = min(self._inflight, n)
            self._cv.notify_all()
        for reader in (self._gpu, self._cpu):
            t = threading.Thread(target=self._producer, args=(reader,),
                                 daemon=True)
            t.start()
            self._threads.append(t)

    # ─────────────── 生产者 ───────────────

    def _take_chunk(self):
        """领一个分片下标；无容量令牌或队列空时等待。返回 -1 = 终止。"""
        deadline_wait = 0.2
        while not self._stop.is_set():
            with self._cv:
                if self._pending and self._tokens > 0:
                    idx = self._pending.pop(0)
                    self._tokens -= 1
                    return idx
                self._cv.wait(deadline_wait)
        return -1

    def _producer(self, reader):
        prev_end = None
        while not self._stop.is_set():
            idx = self._take_chunk()
            if idx < 0:
                return
            ch = self._chunks[idx]
            fis = ch['fis']
            try:
                if prev_end is None or fis[0] != prev_end:
                    reader.seek_accurate(fis[0])
                i = 0
                batch = 64
                while i < len(fis) and not self._stop.is_set():
                    be = min(i + batch, len(fis))
                    chunk = fis[i:be]
                    arr = reader.get_batch(chunk, roi=self._roi).asnumpy()
                    with self._cv:
                        for k, fi in enumerate(chunk):
                            ch['data'].append((fi, arr[k]))
                        self._cv.notify_all()
                    i = be
                with self._cv:
                    ch['done'] = True
                    self._cv.notify_all()
                prev_end = fis[-1] + 1
            except Exception as e:  # noqa: BLE001
                self._err.append(e)
                with self._cv:
                    ch['done'] = True
                    self._cv.notify_all()
                return

    # ─────────────── 消费者（主线程） ───────────────

    def _chunk_index(self, fi: int) -> int:
        return max(bisect.bisect_right(self._starts, fi) - 1, 0)

    def _pop_frame(self, fi: int):
        ci = self._chunk_index(fi)
        ch = self._chunks[ci]
        stalled = 0
        while True:
            with self._cv:
                if ch['off'] < len(ch['data']):
                    got, crop = ch['data'][ch['off']]
                    ch['off'] += 1
                    if got != fi:
                        raise RuntimeError(
                            'hybrid 序错位: want=%d got=%d' % (fi, got))
                    if ch['off'] == len(ch['fis']):
                        #该片交付完毕：归还容量令牌，生产者可开新片
                        self._tokens += 1
                        self._cv.notify_all()
                    return crop
                if self._err:
                    raise RuntimeError('hybrid 解码失败: %r'
                                       % self._err[:1])
            if self._stop.is_set():
                raise RuntimeError('hybrid 解码被取消')
            stalled += 1
            if stalled > 6000:   # ~20min 无进展防御
                raise RuntimeError('hybrid 解码停滞')
            with self._cv:
                self._cv.wait(0.05)

    # ─────────────── VideoReader 兼容接口 ───────────────

    def __len__(self):
        return len(self._gpu)

    def get_batch(self, frame_list, roi=None):
        arrs = [self._pop_frame(fi) for fi in frame_list]
        return _Batch(np.stack(arrs))

    def next_roi(self, x1, y1, x2, y2):
        """stride==1 校准顺序流：与 get_batch 共享同一交付序。"""
        if self._seq_fi is None:
            self._seq_fi = self._starts[0] if self._starts else self._f0
        crop = self._pop_frame(self._seq_fi)
        fi = self._seq_fi
        self._seq_fi = fi + 1
        return _Batch(crop)

    def seek_accurate(self, fi: int):
        # 吞掉：分片定位由生产者在片首完成，外部 seek 会与预取竞态。
        return

    def get_avg_fps(self):
        return self._gpu.get_avg_fps()

    def get_fps(self):
        try:
            return self._gpu.get_fps()
        except Exception:
            return self.get_avg_fps()

    def get_color_range(self):
        try:
            return self._gpu.get_color_range()
        except Exception:
            return 0

    def get_codec(self):
        try:
            return self._gpu.get_codec()
        except Exception:
            return ''

    def get_key_indices(self):
        try:
            return self._gpu.get_key_indices()
        except Exception:
            return []

    def close(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
