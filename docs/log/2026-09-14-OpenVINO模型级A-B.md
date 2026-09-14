# 2026-09-14 OpenVINO vs onnxruntime：OCR CPU 后端模型级 A/B（立项裁决前置）

> 命题（用户）：CPU 推理从 ONNX 改为 OpenVINO 是否可能性能提升？
> 静态量化判死轮（log 2026-09-14-Graph整段与静态量化定稿）留档过
> "OpenVINO EP 为 CPU 提速正路（需立项）"。本轮先做**模型级**裁决
> （未做引擎集成）。

## 1. 模型级 A/B（`tools/_probe_ov_cpu_ab.py`）

口径：同模型 `PP-OCRv6_rec_small.onnx`、同线程（16=物理核，
与 `_init_onnx` 生产配置逐字同参）、同随机输入；ORT
CPUExecutionProvider（intra 16/inter 2）vs OpenVINO 2026.3.1 CPU
插件（INFERENCE_NUM_THREADS=16；另设默认线程档参照）。warmup5 +
中位 of 30：

| 形状 (B,3,48,W) | ORT ms | OV ms | OVdef ms | OV/ORT | argmax 一致 |
|---|---:|---:|---:|---:|---:|
| (18,·,·,224) 生产 | 47.79 | 22.95 | 21.79 | **0.48×** | 100% |
| (16,·,·,320) | 58.40 | 27.03 | 27.26 | **0.46×** | 100% |
| (1,·,·,224) | 4.63 | 2.26 | 2.16 | **0.49×** | 100% |

max|Δpreds|（logits）0.05~0.09（不同内核实现的浮点差），(B,S)
argmax 逐行一致。动态形状（B/W 均动）单次 compile 直跑，OV 无需
形状特化即拿到 2.1×。

## 2. 引擎级外推（ONNX 路径现状，h264-cpu 全片，ENGINE_PROFILE）

wall 8.74s：`ocr.infer` 双实例合计 **16.31s**（两 OCR 线程 ~93% 饱和，
OCR 严格 bound）、producer `decode_batch` 4.14s（47%）、
`preproc_resize` 2.06s（CPU 预处理，下一瓶颈）。

⇒ infer 0.48× 后 OCR 侧功 ≈3.9s/线程 < 解码 4.1s，wall 翻为解码/
预处理共绑，预计 **−35~45%**（无 NVIDIA GPU 部署 / ocr=cpu 场景）。

## 3. 立项要件（未做，供决策）

- 集成面：`ocr_backend` 扩 "openvino"（OcrEngine._init_ov + extract
  校验白名单 + run_config/knob + DEPENDENCIES（openvino 2026.3.1，
  wheel ~76MB））。CTC/预处理零改动（同 preds 输出）。
- 门禁：逐帧真值准确率（`_probe_acc_ab` 需支持后端注入）+ 段数 +
  bench ab 交错（h264-cpu / hevc-cpu）。随机输入 argmax 一致≠真值
  准确率（铁律 3，fp16 轮同款门禁）。
- 风险：OV 数值与 ORT 有 ≤0.09 的 logits 差（真值裕度内预期安全，
  以门禁为准）；新依赖体积；Zen4 上 oneDNN 已快 2.1×，Intel 端
  预期不差（OV 主场）。
