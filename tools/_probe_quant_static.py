# -*- coding: utf-8 -*-
"""静态 INT8 QDQ 量化构建 + 精度/速度评测（2026-09-14 压缩轮探针，已判死）。

历史（ARCHIVE）：动态 INT8=20× 劣化+输出尽毁（ORT 1.27），静态 QDQ
未测——本探针补测。结论（见 docs/log/2026-09-14-Graph整段与静态量化
定稿.md）：全图/仅 Conv 两变体均 argmax 格一致 0.58、CTC 序列一致
0、速度 +30~81%——静态与动态同命，量化方向在 ORT 1.29/Win-x64 关闭。
产物与校准集留在 bench/quant/（gitignore）；复评（OpenVINO EP 立项）
时直接可跑。

用法：
  python tools/_probe_quant_static.py            # 构建 QDQ（QUANT_VARIANT 选）
  python tools/_probe_quant_static.py --eval     # 精度（argmax/CTC）+ 速度
前置：bench/quant/calib3.npy + eval.npy（本脚本 --calib 可从视频重建）。
"""
import argparse
import os
import sys
import time

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader,
                                      CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "bench", "quant", "PP-OCRv6_rec_small_op17.onnx")  # opset 11→17（QDQ 的 axis 属性需 ≥13）
OUT = os.path.join(ROOT, "bench", "quant", "PP-OCRv6_rec_small_int8_qdq.onnx")
CALIB = os.path.join(ROOT, "bench", "quant", "calib3.npy")


class Reader(CalibrationDataReader):
    def __init__(self, path, batch=16):
        x = np.load(path)                      # (N,3,48,W)
        n = (len(x) // batch) * batch          # 只出整批（尾批形状不一致，
        self.batches = [x[i:i+batch]           #   校准器 np.asarray 会炸）
                        for i in range(0, n, batch)]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.batches):
            return None
        b = self.batches[self.i]
        self.i += 1
        return {"x": np.ascontiguousarray(b, dtype=np.float32)}

    def rewind(self):
        self.i = 0


def build():
    print("building QDQ int8 (perchannel, entropy)...")
    import os
    mode = os.environ.get("QUANT_VARIANT", "convonly")
    kw = {}
    if mode == "convonly":
        # 只量化 Conv（激活 int8 对称 + 权重 per-channel）：CRNN/LayerNorm/
        # Erf 链保持 fp32——OCR 识别头对激活量化极敏感（全图版格一致 0.58）
        kw = dict(
            op_types_to_quantize=["Conv"],
            per_channel=True,
            reduce_range=False,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": True,
                           "WeightSymmetric": True},
        )
    elif mode == "percentile_sym":
        kw = dict(
            per_channel=True, reduce_range=False,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.Percentile,
            extra_options={"ActivationSymmetric": True,
                           "WeightSymmetric": True},
        )
    else:  # full_asym（默认原方案）
        kw = dict(
            per_channel=True, reduce_range=False,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.Percentile,
            extra_options={"ActivationSymmetric": False,
                           "WeightSymmetric": True},
        )
    quantize_static(SRC, OUT, Reader(CALIB), **kw)
    print("saved", OUT, os.path.getsize(OUT))


def evaluate():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    s32 = ort.InferenceSession(SRC, so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(OUT, so, providers=["CPUExecutionProvider"])

    # 精度：eval.npy 真实帧（64 张，未见过的段）+ 校准宽度扩到批
    ev = np.load(os.path.join(ROOT, "bench", "quant", "eval.npy"))  # (64,3,48,W)
    # 按 W=224/320 生产主形状 pad 后对比
    for W in (224, 320):
        batch = np.zeros((64, 3, 48, W), dtype=np.float32)
        ww = min(W, ev.shape[3])
        batch[:, :, :, :ww] = ev[:, :, :, :ww]
        o32 = s32.run(None, {"x": batch})[0]
        o8 = s8.run(None, {"x": batch})[0]
        a32 = o32.argmax(-1)
        a8 = o8.argmax(-1)
        cell = (a32 == a8).mean()
        row = (a32 == a8).all(-1).mean()
        # 文本级：CTC 折叠后是否同文
        def ctc(x):
            idx = x.argmax(-1)
            out = []
            prev = -1
            for r in idx:
                line = []
                pp = -1
                for v in r:
                    if v != pp and v != 0:
                        line.append(int(v))
                    pp = v
                out.append(tuple(line))
            return out
        same_text = sum(1 for x, y in zip(ctc(o32), ctc(o8)) if x == y) / 64
        print(f"W={W}: argmax 格一致 {cell:.4f} | 行级全同 {row:.4f} | "
              f"CTC 序列一致 {same_text:.4f}")

    # 速度：生产形状 B16/18 W224/320
    for B, W in ((16, 224), (18, 224), (16, 320)):
        x = np.random.randint(0, 255, size=(B, 3, 48, W)).astype(np.float32)
        x = (x / 255.0 - 0.5) / 0.5
        for name, sess in (("fp32", s32), ("int8", s8)):
            for _ in range(3):
                sess.run(None, {"x": x})
            t0 = time.perf_counter()
            N = 12
            for _ in range(N):
                sess.run(None, {"x": x})
            dt = (time.perf_counter() - t0) / N * 1000
            print(f"B{B} W{W} {name}: {dt:.1f}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.eval:
        evaluate()
    else:
        build()
