"""路线图轮 M2：OCR rec 推理层微基准（ONNX-CPU / TRT）× 批大小扫描。

问题：TRT 引擎 profile 的 max_batch=6（ocr_trt.py 注释），OCR 批被切成
6+6+... 子批；ONNX 走动态 batch。本探针用引擎真实输入形态
（N × 1 × 48 × 224，pad 宽下限 224 主导的常态形态）测：

  1. ONNX(ORT_CPU_EP) fps vs batch ∈ sweep
  2. TRT fps vs batch ∈ sweep（含 >6 的子批切分开销口径）
  3. 宿主 _resize_norm 预处理单帧成本（ONNX 路径的 Python 侧税）

内容用随机噪声——rec CNN 的卷积代价由输入形状决定，与内容无关
（计时用途，不用于准确率结论）。

用法：python tools/_probe_roadmap_ocr.py [--bs 1,2,4,6,8,12,16,24,32,48]
       [--n 512] [--engine onnx|trt|both]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def bench_onnx(bs_list, n, threads=None):
    import numpy as np
    from ocr_native import OcrEngine
    ot = OcrEngine(variant="v6_small", engine_type="onnxruntime",
                   num_threads=threads)
    x = np.random.rand(n, 3, 48, 224).astype(np.float32)
    out = {}
    sess = ot._session
    # 预热
    sess.run(None, {"x": x[:8]})
    for bs in bs_list:
        reps = max(1, n // max(bs, 1))
        t0 = time.perf_counter()
        for i in range(reps):
            sess.run(None, {"x": x[i * bs:(i + 1) * bs]})
        dt = time.perf_counter() - t0
        fps = reps * bs / dt
        out[bs] = fps
        print(f"  onnx bs={bs:3d}: {fps:7.0f} fps  ({dt/reps*1000:6.2f} ms/批)", flush=True)
    return out


def bench_trt(bs_list, n):
    import numpy as np
    from ocr_native import OcrEngine
    eng = OcrEngine(variant="v6_small", engine_type="tensorrt")
    trt = eng._trt
    print(f"  TRT max_batch={trt.max_batch}", flush=True)
    x = np.random.rand(n, 3, 48, 224).astype(np.float32)
    out = {}
    # 走引擎真实入口（含子批切分逻辑），喂设备 buffer
    for bs in bs_list:
        reps = max(1, n // max(bs, 1))
        t0 = time.perf_counter()
        for i in range(reps):
            eng._infer(x[i * bs:(i + 1) * bs])
        dt = time.perf_counter() - t0
        fps = reps * bs / dt
        out[bs] = fps
        print(f"  trt  bs={bs:3d}: {fps:7.0f} fps  ({dt/reps*1000:6.2f} ms/批)", flush=True)
    return out


def build_test_engine(max_b: int, fp16: bool, out_path: Path) -> None:
    """构建实验 TRT 引擎（FP32/FP16 × 任意 profile batch），不动生产缓存。

    复刻 ocr_trt.TrtEngine._build 的 profile 配置（宽 32-2048/opt 320），
    仅改 batch 上限与精度 flag。TRT 11：显式 batch 为默认。
    """
    import tensorrt as trt
    import engine_config as config
    models = Path(__file__).resolve().parent.parent / "assets" / "ocr_models"
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    try:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        flags = 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(models / "PP-OCRv6_rec_small.onnx", "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX 解析失败")
    bc = builder.create_builder_config()
    bc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                             config.TRT_WORKSPACE_BYTES)
    if fp16:
        bc.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    h = config.OCR_TARGET_H
    profile.set_shape(network.get_input(0).name,
                      min=(1, 3, h, config.TRT_PROFILE_MIN_W),
                      opt=(max_b, 3, h, config.TRT_PROFILE_OPT_W),
                      max=(max_b, 3, h, config.TRT_PROFILE_MAX_W))
    bc.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, bc)
    if serialized is None:
        raise RuntimeError("TRT 引擎构建失败")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(serialized)
    print(f"  [build] {out_path.name} fp16={fp16} max_b={max_b} "
          f"({out_path.stat().st_size // 1024} KB)", flush=True)


def bench_engine_file(engine_path: Path, bs_list, n) -> None:
    """把 TrtEngine 指到自定义 engine 文件跑批扫描（monkeypatch 候选路径）。"""
    import numpy as np
    import ocr_trt
    from ocr_native import OcrEngine
    orig = ocr_trt.TrtEngine._engine_candidates

    @staticmethod
    def _cand(size, _orig=orig, _p=engine_path):
        return [_p]
    ocr_trt.TrtEngine._engine_candidates = _cand
    try:
        eng = OcrEngine(variant="v6_small", engine_type="tensorrt")
    finally:
        ocr_trt.TrtEngine._engine_candidates = orig
    if eng._trt is None:
        raise RuntimeError("实验引擎加载失败（回退了 ONNX）")
    print(f"  [load] {engine_path.name} max_batch={eng._trt.max_batch}", flush=True)
    x = np.random.rand(n, 3, 48, 224).astype(np.float32)
    for bs in bs_list:
        reps = max(1, n // max(bs, 1))
        t0 = time.perf_counter()
        for i in range(reps):
            eng._infer(x[i * bs:(i + 1) * bs])
        dt = time.perf_counter() - t0
        print(f"  {engine_path.stem} bs={bs:3d}: {reps * bs / dt:7.0f} fps  "
              f"({dt / reps * 1000:6.2f} ms/批)", flush=True)


def bench_preprocess(n):
    import numpy as np
    from ocr_native import OcrEngine
    img = (np.random.rand(33, 130, 1) * 255).astype(np.uint8)
    t0 = time.perf_counter()
    reps = 2000
    for _ in range(reps):
        OcrEngine._resize_norm(img, 224 / 48, 33)
    dt = time.perf_counter() - t0
    per_us = dt / reps * 1e6
    print(f"  _resize_norm: {per_us:.0f} us/帧 ({1e6/per_us:.0f} fps 单线程)", flush=True)
    return per_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", default="1,2,4,6,8,12,16,24,32,48")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--engine", default="both")
    ap.add_argument("--onnx-threads", type=int, default=16,
                    help="ONNX intra_op 线程数（默认 16=全物理核，引擎口径）")
    ap.add_argument("--fp16-experiment", action="store_true",
                    help="构建 FP32/FP16 × max_b 实验引擎并跑批扫描")
    ap.add_argument("--exp-max-b", default="6,24")
    args = ap.parse_args()
    models = Path(__file__).resolve().parent.parent / "assets" / "ocr_models"
    bs_list = [int(b) for b in args.bs.split(",")]
    print(f"[OCR rec 微基准] n={args.n} 形态=(N,3,48,224) "
          f"onnx_threads={args.onnx_threads}", flush=True)
    if args.engine in ("onnx", "both"):
        bench_onnx(bs_list, args.n, threads=args.onnx_threads)
    if args.engine in ("trt", "both"):
        bench_trt(bs_list, args.n)
    if args.fp16_experiment:
        exp_dir = Path(__file__).resolve().parent / "_roadmap_20260910"
        for max_b in [int(x) for x in args.exp_max_b.split(",")]:
            for fp16 in (False, True):
                name = f"exp_{('fp16' if fp16 else 'fp32')}_b{max_b}.engine"
                p = exp_dir / name
                if not p.exists():
                    build_test_engine(max_b, fp16, p)
                bench_engine_file(p, bs_list, args.n)
    bench_preprocess(args.n)


if __name__ == "__main__":
    main()
