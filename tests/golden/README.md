# tests/golden —— S0 冻结契约（v2 ARCHITECTURE.md §8.5 / §11 S0）

金标向量 = v1 全链提取的**阶段输出摘要**（段结构 / 文本 / 置信度全精度 /
校准阈值 / 降级原因 / crop sha）。不存像素。它是 v2 strangler 迁移
（S3/S4/S5）逐位对账的参照系：**任一阶段改动后重跑 `--verify`，漂移即回退**
（分歧处理：硬阻塞 + 定性白名单，见 v2 §8.5）。

## 文件

| 文件 | 作用 |
|---|---|
| `record.py` | 录制 / 复核（`--verify` 为 S0 可复现门禁，28/28 一致，2026-09-10） |
| `manifest.yaml` | 录制基准（commit / GPU / driver / decord / 视频指纹）+ 每用例期望哈希 |
| `case-*/stage-{calib,ocr}.json` | 28 用例的摘要向量（timing 仅供参考，不参与比对） |
| `decoder_contract.yaml` | 引擎对 decord fork 的 10 项隐式假设（S5 验收表） |
| `probe_triage.yaml` | 69 探针三分法底账（§8.6 根治验收依据） |
| `bench_baseline.json` | 四配置 × 3 轮性能基线 + 噪声底（D10 阈值校准依据） |
| `FINDINGS.md` | 录制期发现的 v1 行为缺陷（F-1 force_aspect 越界报错路径崩溃等） |

## 覆盖矩阵（28 用例，窗口 [0,3000)）

- **A（12）**：test5 h264 × decode{cpu,nvdec,hybrid} × pipeline{host,GPU} × OCR{onnx,TRT}
  —— 全部 1083 段：D1 修复后 host=GPU 跨配置等价的基线成立
- **B（7）**：rep_crop_format{gray,yuv} × keep_crops{T,F}、stride 8、force_aspect 4、
  关闭 merge_similar（1090 段）
- **C（9）**：test6 同内容三编码（av1/h264/hevc）× decode{cpu,nvdec,hybrid}
  —— 全部 1109 段：跨编码等价成立

## 门禁

- 本机（须 GPU + 真值视频）：`python tests/golden/record.py --verify`
- CI（无 GPU）：`tests/golden/test_golden_manifest.py` 校验 manifest 哈希与盘上一致
- 性能：`bench_baseline.json`（热轮中位；噪声底 1.3% → D10 PR 硬失败 5% 初值成立）
