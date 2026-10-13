# tests/golden —— 金标冻结契约

金标向量 = 全链提取的**阶段输出摘要**（段结构 / 文本 / 置信度全精度 /
校准阈值 / 降级原因 / crop sha）。不存像素。它是逐位对账的参照系：
**任一阶段改动后重跑 `--verify`，漂移即回退**（分歧处理：硬阻塞 + 定性白名单）。

## 文件

| 文件 | 作用 |
|---|---|
| `record.py` | 录制 / 复核（`--verify` 为可复现门禁；用例数以 test_golden_manifest 为准） |
| `manifest.yaml` | 录制基准（commit / GPU / driver / decord / 视频指纹）+ 每用例期望哈希 |
| `case-*/stage-{calib,ocr}.json` | 逐用例摘要向量（timing 仅供参考，不参与比对） |
| `decoder_contract.yaml` | 引擎对 decord fork 的 10 项隐式假设（验收表） |
| `bench_baseline.json` | 四配置 × 3 轮性能基线 + 噪声底（阈值校准依据） |
| `FINDINGS.md` | 金标门禁的已知失败模式 |

## 覆盖矩阵（33 用例；A/B/C 窗口 [0,3000)，D 组晚起点窗）

- **A（12）**：test5 h264 × decode{cpu,nvdec,hybrid} × pipeline{host,GPU} × OCR{cpu,TRT}
  —— 全部 1083 段：host 与 GPU 跨配置等价成立
- **B（7）**：rep_crop_format{gray,yuv} × keep_crops{T,F}、stride 8、force_aspect 4、
  关闭 merge_similar（1090 段）
- **C（9）**：test6 同内容三编码（av1/h264/hevc）× decode{cpu,nvdec,hybrid}
  —— 全部 1109 段：跨编码等价成立
- **D（5）**：三码 × hybrid × 晚起点 seek(5000) × 窗（test5 尾格 2761 顺带覆盖
  片尾窗；h264-w1000 因置信度运行间翻动撤出）。hevc/av1 段数 337/1155
  跨编码相等。

## 门禁

- 本机（须 GPU + 真值视频）：`python tests/golden/record.py --verify`
  （`--tier smoke` = 3 用例 ≈30s 迭代用；verify 为复现式判定——差异须
  复跑复现才算回归，一次性差异按翻动放行并计数）
- CI（无 GPU）：`tests/golden/test_golden_manifest.py` 校验 manifest 哈希与盘上一致
- 性能：`bench_baseline.json`（热轮中位；噪声底 1.3% → PR 硬失败 5%）
