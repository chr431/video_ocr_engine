"""单视频引擎 e2e + 管线模式取证：打印 GPU 管线是否启用与 hybrid-stats 汇总。

用法（DECORD_* env 由外层设置，如 DECORD_HYBRID_STATS=1 看规划/上传账）：
  python tools/_probe_e2e_mode.py test6_hevc.mp4 hybrid
  python tools/_probe_e2e_mode.py test6.mp4 hybrid cpu   # argv[3]=ocr_backend
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from video_ocr_engine.extractor import FieldExtractor  # noqa: E402

vid = os.path.join(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"),
                   sys.argv[1])
backend = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
ocr = sys.argv[3] if len(sys.argv) > 3 else "tensorrt"
# 与 _probe_stress_harness 同一 ROI 约定（hevc/test6 与 h264/test5 两种）
roi = ((841, 994, 950, 1027) if ("hevc" in vid or "test6" in vid)
       else (843, 993, 949, 1026))

ex = FieldExtractor(video_path=vid, roi=roi, decode_backend=backend,
                    ocr_backend=ocr, keep_crops=False)
t = time.perf_counter()
res = ex.extract()
wall = time.perf_counter() - t
mode = getattr(ex, "_gpu_pipeline_mode", None)
print("E2E %s %s wall=%.3fs segs=%d gpu_pipeline_mode=%s"
      % (backend, os.path.basename(vid), wall,
         len(res.segments), mode))
