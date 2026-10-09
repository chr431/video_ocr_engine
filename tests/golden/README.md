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

## 覆盖矩阵（33 用例；A/B/C 窗口 [0,3000)，D 组晚起点窗）

- **A（12）**：test5 h264 × decode{cpu,nvdec,hybrid} × pipeline{host,GPU} × OCR{onnx,TRT}
  —— 全部 1083 段：D1 修复后 host=GPU 跨配置等价的基线成立
- **B（7）**：rep_crop_format{gray,yuv} × keep_crops{T,F}、stride 8、force_aspect 4、
  关闭 merge_similar（1090 段）
- **C（9）**：test6 同内容三编码（av1/h264/hevc）× decode{cpu,nvdec,hybrid}
  —— 全部 1109 段：跨编码等价成立
- **D（5，2026-10-08 窗口架构重做补录）**：三码 × hybrid × 晚起点
  seek(5000) × 窗（test5 尾格 2761 顺带覆盖片尾窗；h264-w1000 因
  F-12 置信度翻动撤出）——关闭
  09-19 窗口缺陷穿透的金标盲区（彼时金标只有早起点窗）；hevc/av1
  段数 337/1155 跨编码相等。录制口径 = fork 0.9.0 dev（窗路径）；
  0.8.5 wheel（无 window_seek_safe 键→引擎谓词不设窗）输出应逐位同

## 门禁

- 本机（须 GPU + 真值视频）：`python tests/golden/record.py --verify`
  （`--tier smoke` = 3 用例 ≈30s 迭代用；verify 为复现式判定——差异须
  复跑复现才算回归，一次性差异按 F-12 翻动放行并计数）
- CI（无 GPU）：`tests/golden/test_golden_manifest.py` 校验 manifest 哈希与盘上一致
- 性能：`bench_baseline.json`（热轮中位；噪声底 1.3% → D10 PR 硬失败 5% 初值成立）
