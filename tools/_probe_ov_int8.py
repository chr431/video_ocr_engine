"""_probe_ov_int8.py —— OV INT8（NNCF PTQ）模型级裁决。

命题
----
TRT 侧静态量化判死（argmax 格一致 0.58，Quant 轮），但 OV/NNCF 校准管线
未测过且 Zen4 有 AVX512-VNNI。本探针做**模型级**裁决：NNCF PTQ INT8
（真实预处理数据校准）vs fp32——速度（B18W224/B16W320）+ 数值一致
（真实批 argmax 逐行一致率）。真值门禁另跑（若速度收益显著）。

用法
----
    python tools/_probe_ov_int8.py [--calib-frames 300]
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


def collect_real_batches(frames: int) -> list[np.ndarray]:
    """从 test5 收集真实预处理输入批（fa=1.5 生产口径）。"""
    import decord
    from video_ocr_engine.domain.segmentation import preprocess_standard
    from video_ocr_engine.ocr.native import OcrEngine
    vr = decord.VideoReader(r"D:\Videos\racelog_test\test5.mp4", ctx=decord.cpu(0),
                            num_threads=4)
    eng = OcrEngine("v6_small", "onnxruntime", fill_width=224, num_threads=4)
    batches = []
    for s in range(0, frames - 18, 90):
        imgs = []
        for i in range(s, s + 18):
            fr = vr[i]
            p = preprocess_standard(fr.asnumpy(), force_aspect=1.5)
            imgs.append(p)
        batch_np = np.stack([eng._resize_norm(im, 224 / 48, 48) for im in imgs])
        batches.append(batch_np)
    return batches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-frames", type=int, default=300)
    ap.add_argument("--model", default="PP-OCRv6_rec_small.onnx")
    args = ap.parse_args()
    import openvino as ov
    import nncf

    core = ov.Core()
    model_path = str(ROOT / "assets/ocr_models") + "/" + args.model         if not args.model.startswith("D:") else args.model
    print("收集真实校准数据…", flush=True)
    calib_batches = collect_real_batches(args.calib_frames)
    print("校准批数:", len(calib_batches))

    calib_data = [{"x": b} for b in calib_batches]
    model = core.read_model(model_path)
    print("NNCF PTQ 量化中…", flush=True)
    t0 = time.perf_counter()
    q_model = nncf.quantize(model, nncf.Dataset(calib_data))
    print("量化完成 %.1fs" % (time.perf_counter() - t0), flush=True)

    props = {"INFERENCE_NUM_THREADS": "8"}
    fp32 = core.compile_model(model_path, "CPU", props)
    int8 = core.compile_model(q_model, "CPU", props)
    out8, out32 = int8.outputs[0], fp32.outputs[0]

    rng = np.random.default_rng(0)
    print("\n%-16s %9s %9s %8s %10s %10s" % (
        "input", "fp32 ms", "int8 ms", "x", "argmax一致", "max|Δ|"))
    for tag, gen in [
        ("B18W224 真实", lambda: calib_batches[0]),
        ("B16W320 真实", lambda: np.concatenate(
            [calib_batches[0][:16, :, :, :320]], axis=0)),
        ("B18W224 随机", lambda: (rng.standard_normal((18, 3, 48, 224)) * 0.2)
         .astype(np.float32)),
    ]:
        def bench(m, o, xin):
            for _ in range(5):
                m(xin)
            ts = []
            for _ in range(20):
                t = time.perf_counter()
                m(xin)
                ts.append(time.perf_counter() - t)
            pa = np.array(m(xin)[o])
            pb = np.array((fp32 if m is int8 else int8)(xin)[out32 if m is int8 else out8])
            return statistics.median(ts) * 1e3, pa, pb
        xin = gen()
        t32, pa, pb = bench(fp32, out32, xin)
        t8, pb2, pa2 = bench(int8, out8, xin)
        am = (pa.argmax(2) == pb2.argmax(2)).mean()
        print("%-16s %9.1f %9.1f %7.2fx %9.2f%% %10.4g"
              % (tag, t32, t8, t8 / t32, am * 100,
                 float(np.abs(pa - pb2).max())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
