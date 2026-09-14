"""_probe_ov_cpu_ab.py —— OCR CPU 推理后端 A/B：onnxruntime vs OpenVINO（模型级）。

命题
----
ONNX 宿主路径（无 NVIDIA GPU / ocr=cpu 部署）在引擎级是 CPU-OCR-bound；
静态量化轮判死 ORT 量化（准确率 0.58）时留档"OpenVINO EP 为 CPU 提速
正路（需立项）"。本探针先做**模型级**裁决：同模型（PP-OCRv6_rec_small
onnx）、同线程（= 生产 intra_op 物理核）、同形状（生产 B/W 组合），
ORT（生产配置）vs OpenVINO CPU 插件，量每批延迟 + 数值一致性。
引擎集成（若立项）另需逐帧真值准确率门禁。

口径
----
- ORT：CPUExecutionProvider、intra=auto_ocr_thread_count()（物理核）、
  inter=2——与 native._init_onnx 生产路径逐字同参。
- OpenVINO：compile_model(onnx, "CPU")，INFERENCE_NUM_THREADS=同值；
  另跑一档 OpenVINO 默认线程（不设属性）作参照。
- 计时：warmup 5 + 30 次取中位（批间无间隔；微基准无争用口径）。
- 数值：同一随机输入下 max|Δpreds| + (B,S) argmax 行一致率。

用法
----
    python tools/_probe_ov_cpu_ab.py [--threads N] [--iters 30]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


def bench(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3   # ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0, help="0=物理核")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    from video_ocr_engine.config import constants as config
    from video_ocr_engine.ocr.native import auto_ocr_thread_count
    n_threads = args.threads or auto_ocr_thread_count()
    model = config.models_dir() / "PP-OCRv6_rec_small.onnx"
    print("model=%s  threads=%d" % (model.name, n_threads))

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = n_threads
    so.inter_op_num_threads = 2
    sess = ort.InferenceSession(str(model), sess_options=so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    import openvino as ov
    core = ov.Core()
    ov_16 = core.compile_model(str(model), "CPU",
                               {"INFERENCE_NUM_THREADS": str(n_threads)})
    ov_def = core.compile_model(str(model), "CPU")

    rng = np.random.default_rng(0)
    print("\n%-18s %10s %10s %10s %10s %8s" % (
        "shape(B,3,48,W)", "ORT ms", "OV ms", "OVdef ms", "OV/ORT", "argmax一致"))
    for B, W in ((18, 224), (16, 320), (1, 224)):
        x = rng.standard_normal((B, 3, 48, W)).astype(np.float32) * 0.2
        t_ort = bench(lambda: sess.run(None, {in_name: x})[0], args.iters)
        t_ov = bench(lambda: ov_16(x)[ov_16.outputs[0]], args.iters)
        t_ovd = bench(lambda: ov_def(x)[ov_def.outputs[0]], args.iters)
        p_ort = sess.run(None, {in_name: x})[0]
        p_ov = np.array(ov_16(x)[ov_16.outputs[0]])
        diff = float(np.abs(p_ort - p_ov).max())
        am = float((p_ort.argmax(2) == p_ov.argmax(2)).mean())
        print("%-18s %10.2f %10.2f %10.2f %10.2fx %7.2f%%" % (
            "(%d,·,·,%d)" % (B, W), t_ort, t_ov, t_ovd, t_ov / t_ort, am * 100))
        print("%22s max|Δpreds|=%.4g" % ("", diff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
