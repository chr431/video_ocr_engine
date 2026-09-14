"""_probe_ov_prep_fusion.py —— gamma/resize 换序 + OV preproc 融合的赌局测量。

⚠️ 终审判死（2026-09-14，log 同日预处理融合赌局轮）：生产批为逐帧
异宽裁切（内容裁切），融合语义不可表达；本探针保留模型级取证口径，
其"引擎级胜利"曾被形状臆造 (32,105) vs 实际 (33,154) 的静默回落
污染——教训：对照法必须断言分支真的走到。

命题
----
gamma 位于 resize 之后（不可与线性插值换序 ⇒ resize 无法入 OV 图）。
**换序**（gamma 提前到原分辨率）后 resize 成为最后一步几何操作 →
可整体融合进 OV PrePostProcessor（resize linear + 仿射归一化），
host 只剩灰度+gamma（小面积 pow）。代价 = 数值语义变更
（gamma(interp(x)) ≠ interp(gamma(x))），需真值门禁裁决。

口径
----
- 对照 A（现役）：host `_np_resize`(105×32→224×48) + gamma + 归一化
  → feed (1,3,48,224)。
- 赌注 B（融合）：host 灰度+gamma(105×32) + 通道复制 → feed
  (1,32,105,3) f32 → OV preproc: resize(48,224 linear) + scale/mean。
- 本探针测：B 相对 A 的单流延迟差、输入张量 max|Δ|、以及纯 numpy
  换序（不入图）的微小收益。真值门禁另跑（PROBE_* 透传 + 换序开关）。

用法
----
    python tools/_probe_ov_prep_fusion.py [--iters 200]
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

from video_ocr_engine.config import constants as config
from video_ocr_engine.domain.segmentation import preprocess_standard, _GRAY_W
from video_ocr_engine.domain.video_utils import _np_resize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    import openvino as ov

    rng = np.random.default_rng(0)
    crop = (rng.random((32, 105, 3)) * 255).astype(np.float32)
    GAMMA = 2.0
    H, W = 48, 224

    # ── A 现役 host 链 ──
    def prep_current():
        g = crop @ _GRAY_W
        rs = _np_resize(g, W, H)
        gm = 255.0 * np.power(rs / 255.0, GAMMA)
        x = gm[None, ..., None] if False else np.repeat(
            gm[None, :, :, None], 3, axis=3)
        # _resize_norm 等价：transpose + (x/255-0.5)/0.5
        x = x.transpose(0, 3, 1, 2)
        return ((x / 255.0 - 0.5) / 0.5).astype(np.float32)

    # ── B 换序 host 链（gamma 在原分辨率）──
    def prep_swapped():
        g = crop @ _GRAY_W
        gm = 255.0 * np.power(g / 255.0, GAMMA)
        return np.repeat(gm[None, :, :, None], 3, axis=3).astype(np.float32)

    # ── 融合模型：NHWC f32 → resize → scale/mean ──
    core = ov.Core()
    m = core.read_model(str(config.models_dir() / "PP-OCRv6_rec_small.onnx"))
    ppp = ov.preprocess.PrePostProcessor(m)
    inp = ppp.input()
    inp.tensor().set_element_type(ov.Type.f32).set_layout("NHWC") \
        .set_spatial_dynamic_shape()
    inp.preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR,
                            H, W).mean(127.5).scale(127.5)
    # 归一化 (x/255-0.5)/0.5 = x/127.5 - 1 = (x-127.5)/127.5：
    # OV 语义 = (input - mean) / scale（scale 为除数，作用于原始像素值）
    inp.model().set_layout("NCHW")
    fused = ppp.build()
    fused = core.compile_model(fused, "CPU", {"INFERENCE_NUM_THREADS": "8"})
    base = core.compile_model(
        str(config.models_dir() / "PP-OCRv6_rec_small.onnx"), "CPU",
        {"INFERENCE_NUM_THREADS": "8"})
    base_in, base_out = base.input(0), base.outputs[0]
    f_in, f_out = fused.input(0), fused.outputs[0]

    xa = prep_current()
    xb = prep_swapped()

    # 归一化语义自检：融合模型的 scale/mean 顺序是否复刻 (x/255-0.5)/0.5
    # 用跳过 resize 的对照太麻烦，直接看端到端 max|Δ|（换序本身会引入差，
    # 报告出来供真值门禁参照）。
    out_a = np.array(base(xa)[base_out])
    out_b = np.array(fused(xb)[f_out])
    am = (out_a.argmax(2) == out_b.argmax(2)).mean()
    print("输入批 max|Δpreds|=%.4g  argmax 行一致率=%.4f%%"
          % (float(np.abs(out_a - out_b).max()), am * 100))

    def bench(fn, n):
        fn()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return statistics.median(ts) * 1e6

    # A：host 全链 + 推理
    t_a = bench(lambda: base(prep_current())[base_out], args.iters)
    # B：host 灰度+gamma + 融合推理
    t_b = bench(lambda: fused(prep_swapped())[f_out], args.iters)
    # 纯 host 链耗时（不含推理）
    h_a = bench(prep_current, args.iters)
    h_b = bench(prep_swapped, args.iters)
    print("A host 链 %.1fµs + 推理 = %.1fµs 总" % (h_a, t_a))
    print("B host 链 %.1fµs + 融合推理 = %.1fµs 总" % (h_b, t_b))
    print("B−A 总差 = %+.1fµs（%+.1f%%）" % (t_b - t_a, (t_b / t_a - 1) * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
