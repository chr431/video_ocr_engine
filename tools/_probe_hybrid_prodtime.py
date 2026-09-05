"""临时探针：hybrid 生产者线程的时间分解（C 解码 / asnumpy / Python 记账）。

对 decord VideoReader 实例的 get_batch 打计时补丁（返回 shim 包装 asnumpy），
并开 HYBRID_PROBE 拿逐片 produce 总和：
  in_get_batch   — ctypes 调用期间（含 C++ 解码 + 内部等待，GIL 已放开）
  in_asnumpy     — asnumpy 拷贝
  other          — 生产者 produce 总和 − 前两者（Python 记账 / cv / GIL 等待）

用法：
  python tools/_probe_hybrid_prodtime.py --video test5.mp4 [--envs GPU_PIPELINE=0]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _BatchShim:
    """decord NDArray 有 __slots__，用 shim 包装以计时 asnumpy。"""

    __slots__ = ("_arr", "_stats", "_lock", "shape")

    def __init__(self, arr, stats, lock):
        self._arr = arr
        self._stats = stats
        self._lock = lock
        self.shape = arr.shape

    def asnumpy(self):
        t0 = time.perf_counter()
        out = self._arr.asnumpy()
        with self._lock:
            self._stats["in_asnumpy"] += time.perf_counter() - t0
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test5.mp4")
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--envs", default="GPU_PIPELINE=0")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("HYBRID_PROBE", "1")
    for kv in args.envs.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[k] = v

    from video_ocr_engine import FieldExtractor
    import hybrid_decode

    stats = {}
    lock = threading.Lock()

    def _wrap_reader(reader, tag):
        orig_gb = reader.get_batch

        def get_batch(frame_list, roi=None):
            t0 = time.perf_counter()
            batch = orig_gb(frame_list, roi=roi)
            with lock:
                st = stats.setdefault(tag, {"in_get_batch": 0.0,
                                            "in_asnumpy": 0.0, "n": 0})
                st["in_get_batch"] += time.perf_counter() - t0
                st["n"] += 1
            return _BatchShim(batch, st, lock)
        reader.get_batch = get_batch

    orig_producer = hybrid_decode.HybridDecoder._producer

    def _producer_patched(self, reader):
        tag = self._pname.get(id(reader),
                              "fast" if reader is self._fast_reader else "slow")
        _wrap_reader(reader, tag)
        return orig_producer(self, reader)

    hybrid_decode.HybridDecoder._producer = _producer_patched

    roi = tuple(int(v) for v in args.roi.split(","))
    ex = FieldExtractor(args.video, roi, frame_end=args.frames,
                        sample_stride=1, decode_backend="hybrid",
                        ocr_backend="auto", keep_frames=True)
    t0 = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t0
    print(f"\n=== wall={wall:.3f}s timing={ {k: round(v, 3) for k, v in ex.timing.items()} }")
    print("=== 生产者时间分解 ===")
    for tag, st in stats.items():
        print(f"  [{tag}] get_batch={st['in_get_batch']:.3f}s "
              f"({st['n']}次, GIL 放开) asnumpy={st['in_asnumpy']:.3f}s")


if __name__ == "__main__":
    main()
