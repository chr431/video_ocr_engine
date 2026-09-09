"""复刻 probe_decode_rates 的 8 片场景，逐片分解 seek / 批量读耗时。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decord import VideoReader, cpu  # noqa: E402

VIDEO = r"D:\Videos\racelog_test\test6.mp4"
ROI = (841, 994, 950, 1027)
N, BATCH, STEP = 4000, 64, 500

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 4
vr = VideoReader(VIDEO, ctx=cpu(0), num_threads=NT, output_format="gray")
keys = list(vr.get_key_indices())

tot_seek = tot_read = 0.0
print("chunk  seek(ms)  read64(ms)  d=s%300  seg_since_last")
pos = 0
for c in range(8):
    s = min(c * STEP, N - 1)
    t0 = time.perf_counter()
    vr.seek_accurate(s)
    t1 = time.perf_counter()
    vr.get_batch(list(range(s, min(s + BATCH, N))), roi=ROI).asnumpy()
    t2 = time.perf_counter()
    tot_seek += t1 - t0
    tot_read += t2 - t1
    print(f"{c}      {(t1-t0)*1000:8.1f}  {(t2-t1)*1000:9.1f}   {s % 300:5d}    {s - pos}")
    pos = s + BATCH
print(f"合计 seek={tot_seek:.3f}s read={tot_read:.3f}s")
