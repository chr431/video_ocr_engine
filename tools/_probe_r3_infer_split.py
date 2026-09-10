"""R3 调查：hybrid vs nvdec 的 OCR worker infer 差异分解。

两配置各跑一次 GPU 管线 e2e（fork dll），对比：
  - ENGINE_PROFILE 分相（infer / q_get_wait / consume_feed / emit_put）
  - TRT SUBPROBE（htod / enqueue / dtoh / sync 累计）
  - call_gpu_raw vs eng(procs) 两条路径的批数分布（raw 直通是否生效）

用法：
  DECORD_LIBRARY_PATH=D:\\Repo\\decord\\build-081fix \
  python tools/_probe_r3_infer_split.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VID = r"D:\Videos\racelog_test\test6.mp4"
ROI = (841, 994, 949, 1026)
FRAMES = 3000


def run_once(backend: str) -> None:
    os.environ["ENGINE_PROFILE"] = "1"
    os.environ["TRT_SUBPROBE"] = "1"
    import ocr_trt
    ocr_trt.SUBPROBE.update(htod=0.0, enqueue=0.0, dtoh=0.0, sync=0.0)
    from video_ocr_engine import FieldExtractor
    ex = FieldExtractor(VID, ROI, frame_end=FRAMES, sample_stride=1,
                        decode_backend=backend, ocr_backend="auto",
                        keep_frames=True)
    t0 = time.perf_counter()
    res = ex.extract()
    wall = time.perf_counter() - t0
    print(f"\n===== {backend} wall={wall:.3f}s segs={len(res.segments)} "
          f"timing={ {k: round(v, 3) for k, v in res.timing.items()} }",
          flush=True)
    for group, d in sorted(ex.profile.items()):
        for k, v in sorted(d.items()):
            print(f"  {group}.{k} = {v:.3f}", flush=True)
    sp = dict(ocr_trt.SUBPROBE)
    print(f"  SUBPROBE = { {k: round(v, 3) for k, v in sp.items()} }",
          flush=True)


def main() -> None:
    for bk in ("nvdec", "hybrid"):
        run_once(bk)


if __name__ == "__main__":
    main()
