"""_probe_ov_version_ab.py —— OpenVINO 换代 A/B（2026.3.1 vs 2026.4.x）。

口径：模型级（生产模型/形状/线程数，与 _probe_ov_cpu_ab 的 OV 侧逐字
同参）+ 引擎级（FieldExtractor ocr=cpu，test5 2000 帧，段数/文本集 sha
作准确度门）。跨版本无法同进程交错（同一 site-packages 只有一个版本），
故为顺序口径：基线版 → pip 升级 → 复测，两次均带机器状态语境列。
"""
from __future__ import annotations

import hashlib
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

import openvino as ov  # noqa: E402
from video_ocr_engine.config import constants as config  # noqa: E402
from video_ocr_engine.ocr.native import OcrEngine  # noqa: E402


def bench(fn, iters: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def main() -> int:
    print("openvino version:", ov.__version__)
    models = Path(config.MODELS_DIR) if hasattr(config, "MODELS_DIR") else None
    if models is None:
        from video_ocr_engine.ocr.native import _models_dir  # type: ignore
        models = _models_dir()
    model_path = Path(models) / "PP-OCRv6_rec_small.onnx"
    core = ov.Core()
    n_threads = max(2, __import__("os").cpu_count() or 8)
    from video_ocr_engine.config.decode_caliber import cpu_physical_cores
    n_threads = cpu_physical_cores()
    comp = core.compile_model(str(model_path), "CPU",
                              {"INFERENCE_NUM_THREADS": str(n_threads)})
    out = comp.outputs[0]
    rng = np.random.default_rng(0)
    rows = []
    for B, W in ((8, 320), (32, 320), (32, 960), (1, 320)):
        x = rng.standard_normal((B, 3, config.OCR_TARGET_H, W)).astype(np.float32)
        res = comp([x])[out]
        dt = bench(lambda: comp([x])[out])
        rows.append((B, W, dt))
        print("B=%3d W=%4d  %.2fms/batch  %.0f crops/s" % (B, W, dt * 1e3, B / dt))
    # 引擎级
    from _measure_lib import ROI, VIDS
    from video_ocr_engine import FieldExtractor
    segs_all, texts_all = [], []
    walls = []
    for rep in range(2):
        ex = FieldExtractor(VIDS["test5"], ROI["test5"], frame_start=0, frame_end=2000,
                            decode_backend="nvdec", ocr_backend="cpu")
        t0 = time.perf_counter()
        r = ex.extract()
        walls.append(time.perf_counter() - t0)
        if rep == 0:
            segs_all = [(s.start, s.end, s.rep_frame) for s in r.segments]
            texts_all = [s.text for s in r.segments]
    sha = hashlib.sha256(repr(texts_all).encode()).hexdigest()[:16]
    print("engine OV e2e: %.2fs (%.0f fps) nseg=%d texts_sha=%s"
          % (min(walls), 2000 / min(walls), len(segs_all), sha))
    return 0


if __name__ == "__main__":
    sys.exit(main())
