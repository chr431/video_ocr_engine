"""CPU+NVDEC 混合解码辅助（区间切分 / 解码 worker / 队列消费）。"""
from __future__ import annotations

import engine_config as config
HYBRID_BACKEND_ALIASES: tuple[str, ...] = ("cpu+nvdec", "hybrid")


def _hybrid_ranges(frames: list, calib_n: int,
                   split_ratio: float) -> tuple[list, list]:
    """CPU+NVDEC 混合解码区间切分：(cpu_fis, gpu_fis)，无重叠全覆盖。

    split_pos = max(calib_n, int(len(frames)*split_ratio))：CPU 解
    frames[calib_n:split_pos]（前段，接在校准帧之后），GPU 解
    frames[split_pos:]（后段）。跨后端相邻帧对仅接缝一处，重叠 0 帧。
    """
    split_pos = min(len(frames),
                    max(calib_n, int(len(frames) * split_ratio)))
    return frames[calib_n:split_pos], frames[split_pos:]


def _decode_range_worker(vr, fis, q, roi, th, err,
                         batch: int = config.DECODE_BATCH_SIZE,
                         yuv: bool = False, color_range: int = 0):
    """批量解码帧区间 [fis] → (fi, crop, gray, sharp, bin) 入队。

    CPU+NVDEC 混合解码的解码线程体：_run_pipelined 传 th（分段阈值，
    bin 参与增量分段）；_decode_all 传 th=None（bin=None 不用，参考
    路径）。批量特征（gray/std/二值化）在解码线程内向量化完成，与
    消费者分段线程重叠。yuv=True 时 crops 为 packed YUV420，取 Y 平面
    并按 color_range 展开后再算特征。异常记入 err 并放哨兵。
    roi 为半开区间。
    """
    from segmentation import _gray_seg_batch, _gray_seg_yuv_batch
    try:
        for bstart in range(0, len(fis), batch):
            bend = min(bstart + batch, len(fis))
            crops = vr.get_batch(fis[bstart:bend], roi=roi).asnumpy()
            g = (_gray_seg_yuv_batch(crops, color_range) if yuv
                 else _gray_seg_batch(crops))
            sharp = g.std(axis=(1, 2))
            bs = (g > th) if th is not None else None
            for k, fi in enumerate(fis[bstart:bend]):
                b = None if bs is None else bs[k]
                q.put((fi, crops[k], g[k], float(sharp[k]), b))
    except Exception as e:  # noqa: BLE001 — 经 err 回传主线程 raise
        err.append(e)
    finally:
        q.put(None)


def _drain_queue(q):
    """按序消费解码队列直到哨兵（None）。"""
    while True:
        item = q.get()
        if item is None:
            return
        yield item
