# 性能调优记录（video_ocr_engine）

本文档从 RaceVideoToLog 拆仓时迁移，集中保留与**引擎识别链**（解码 → 像素分段 →
代表帧 → OCR）相关的性能基线、实验结论、已锁定参数和已验证死路。RaceVideoToLog
中的速度纠错/DP 等应用层结论不在此展开，但涉及 `engine_config.py` 的共用参数会保留。

> 所有“实测”数据默认来自 RaceVideoToLog 开发机：
> **7945HX（16 物理核 / 32 线程）+ RTX 4060 Laptop（8GB）**，Windows。
> 测试视频为 `D:\Videos\racelog_test`（test/test2/test3/test5/test6）与
> `D:\Videos\batch_test`（标清字幕剧集）。不同机器/片源数值会变，但相对结论有效。

---

## 1. 测量方法论（重要）

- **A/B 必须单跑、串行**。多个 profile/bench 并行会互抢 CPU/GPU，产生 ±2s 级假象
  （例如 auto 曾并行测得 5.4s，单跑实为 2.8s）。
- 引擎已支持 `RVTOL_PROFILE=1` 细粒度剖面：
  - producer：`open_and_fps / calib / decode_batch / gray / sharp / bin /
    segmentation / q_put_block / consumer_total`
  - OCR：`engine_init / q_get_wait / preprocess / infer / ctc_decode`
- `decode` 计时只计生产者消费流结束，OCR 收尾单列 `ocr_tail`（实测仅 ~0.1s）。
- 快速迭代优先用短窗口（如 3000 帧），提交前再跑全量。

---

## 2. 性能基线（RaceVideoToLog 测试集，2026-08）

### 端到端（test5 7223 帧，decord v0.7.8 + onnxruntime 1.29）

| 组合 | 耗时 |
|---|---:|
| CPU + CPU | 9.0s |
| GPU + CPU | 8.6s |
| CPU + TRT | 6.8s |
| GPU + TRT（auto 默认） | 8.1s |
| CPU+NVDEC 混合 + TRT | 7.0s |
| CPU+NVDEC 混合 + CPU | 8.1s |

### 后端矩阵（test5 h264 / test6 AV1，16 核，2026-08-16）

| 组合 | test5 h264 | test6 AV1 |
|---|---:|---:|
| GPU + TRT（auto，生产默认） | 7.8s | 18.0s |
| GPU + ONNX | 8.5s | 27.4s |
| CPU + TRT | 6.9s | 75.5s（AV1 CPU 灾难） |
| CPU + ONNX | 9.6s | 87.4s |

结论：
- **h264**：CPU 软解在 16 核机器上可能比 NVDEC 更快，但不是所有机器都如此。
- **HEVC/AV1**：CPU 软解明显更慢，GPU（NVDEC）是必须的。
- **auto = GPU + TRT 是跨编码最稳的默认策略**；CPU+TRT 只在 h264/标清场景可能更优。

### 下游场景：video_subtitle_extractor（标清宽 ROI + 跳帧，2026-08）

测试：`D:\Videos\batch_test\新三国01.mkv`，696×424，h264，
`stride=8`，`ROI=144,398,551,423`（约 407×25 宽 ROI）。

| 方案 | 耗时 |
|---|---:|
| 原默认：auto 解码 + auto OCR（GPU+TRT） | ~20.8s |
| CPU 解码 + CPU OCR（RGB） | ~19.3s |
| CPU 解码 + TensorRT OCR（RGB） | ~15.6s |
| **CPU 解码 + TensorRT OCR + gray 输出** | **~15.1s** |
| CPU 解码 + CPU OCR + gray 输出 | ~17.8s |

结论：
- 标清 h264 + 宽 ROI + 跳帧场景，在本机（16 物理核）上 **CPU 软解比 NVDEC 更快**。
- **机制**：NVDEC 的 h264 解码器有约 **2Gp/s 上限**；FFmpeg CPU 解码器最多可利用
  约 **13 个核心**。因此 16 核 CPU 上 h264 软解能显著超过 NVDEC。
- **但这个结论不可泛化**：
  - 用户 CPU 较弱时，NVDEC 依然更好；
  - HEVC / AV1 即使在本机上也是 NVDEC 更好（CPU 软解只有 NVDEC 的 1/3~1/5）。
- `gray_output=True` 能再省约 6-8%：解码直接出单通道，省掉 RGB→灰度转换与数据量。
- 高段数视频（如新三国03，约 6500 段）TensorRT OCR 优势巨大：
  CPU+CPU gray ~60s，CPU+TRT gray ~28s。
- 因此 `video_subtitle_extractor` **默认仍保持 `decode_backend="auto"`**：
  auto 逻辑 = 优先 NVDEC，不可用则回退 CPU。`gray_output=True` 保留为默认优化；
  强多核 CPU + h264 用户可手动选择 `cpu` 获得更好性能。

### 相似段合并（subtitle 场景的大幅加速，2026-08）

高分段/高噪声字幕视频中，同一条字幕常被噪声切成大量短段，导致 OCR 重复执行。
`FieldExtractor(merge_similar=True, merge_similar_threshold=3.0)` 会在 OCR 前比较
相邻段代表帧，满足以下两个条件时合并为同一段，只 OCR 一次：
1. 灰度平均绝对差 ≤ 阈值；
2. `abs(diff)>10` 的显著变化像素占比 ≤ 1%（下限 32 像素）。

条件 2 是为了防止宽 ROI 中“大部分区域未变、只有单个短字幕变化”被均值稀释后
误判为噪声（例如“在”“不”这类单字字幕）。

实测（新三国03，stride=8，CPU+TRT gray）：
- 原始段数 6506 → 合并后约 1165 段（OCR 次数 -82%）
- CPU+TRT gray：约 29.4s → **16.1s**
- CPU+CPU gray：约 60s → **20.5s**
- 输出字幕条数与合并前对比：仅新三国03 少 4 条，且经核对全部是同一条字幕在
  相邻秒的重复/空格变体；新三国04/05 的单字短字幕（“在”“不”）不再丢失。

该功能默认关闭，避免影响速度数字等需要逐段精确 OCR 的场景；
`video_subtitle_extractor` 默认开启，并可用 `--no-merge-similar` 关闭。

---

## 3. 线程预算与分核规则

- **OCR 线程 = 全部物理核**（16C32T → 16）。实测 OCR 8→16 线程：
  GPU 解码 11.3→9.0s、CPU 解码 12.8→9.5s；超物理核不再提升。
- `RVTOL_OCR_THREADS` env 钩子优先（实验用）。
- **少核 CPU 软解分核**（`CPU_CORES_SPLIT_THRESHOLD=8`）：
  - 物理核 ≤8 且 CPU 软解时，OCR 与 decord FFmpeg 帧线程各分 `cores//2`。
  - 4 核 CPU+ONNX：28.0s vs 33.1s（-15%）
  - 8 核 CPU+ONNX：17.8s vs 20.7s（-14%）
  - 16 核：分核反而差，保持 OCR 全核、FFmpeg 默认 2 帧线程落 SMT。
- **AV1 CPU 软解**：dav1d 帧并行上限约 6.6 核；AV1+CPU 解码任何核数用
  `dcd=ocrT=cores//2` 最稳（16 核 45.7s vs 12/4 的 58.5s）。
- **双 ONNX 实例 OCR**（`RVTOL_DUAL_ONNX=0` 关闭）：
  单实例 intra-op 线程池扩展亚线性（16 线程仅 4.2×）；两个独立实例各 `ocrT//2`
  线程并发取批，纯吞吐 313→355 段/s（+15-18%），RSS +~200MB。显式 OCR=cpu 且
  核数≥8 时默认启用。
- **decode batch / FFmpeg 线程进一步扫描**（2026-08 补充）：
  `DECODE_BATCH_SIZE` 32/64、FFmpeg 解码线程 4 在标清宽 ROI 跳帧场景无收益，
  维持现状。

---

## 4. 混合解码 / 混合 OCR（实验开关，默认关闭）

### CPU+NVDEC 混合解码（`RVTOL_HYBRID_DECODE=1`）

- CPU 解前 10%（`HYBRID_CPU_SPLIT=0.10`，calib 后）、GPU 解后 90%，双 worker
  并行填有界队列，消费者按序合并。
- 实测（decode 阶段，venv+TRT）：HEVC 2.5 vs 2.6s、h264 3.1 vs 3.3s / 7.2 vs
  7.7s、AV1 14.4 vs 14.1s —— 三种编码均不弱于纯 GPU。
- **AV1 特判**：CPU 软解 AV1 极耗核且与 GPU 段并发竞争反而拖慢 GPU 吞吐
  （混合 19.1s vs 纯 GPU 14.4s）→ 不打开 CPU reader，按纯 GPU 分支走。
- 收益不确定且增加复杂度，默认关闭。

### TRT+ONNX 混合 OCR（`RVTOL_HYBRID_OCR=1`）

- TRT（GPU）+ onnxruntime（CPU）双引擎并发处理段批；OCR 无状态约束，按段索引
  聚合，实现简单。
- test6 实测：纯 ONNX 22.9s → 混合 15.3s（≈ 纯 TRT 14.7s）。
- **结论：TRT 可用时 auto 已最优**；混合只对“TRT 可用但强制 OCR=cpu”有意义，
  默认关闭。

---

## 5. 已锁定参数（勿随意改动）

| 参数 | 值 | 结论 |
|---|---|---|
| `OCR_GAMMA` | 2.0 | OCR 预处理灰度 gamma；全量最优，固定 |
| `SEG_GAMMA` | 0.0 | 分段/代表帧选择用 raw 灰度；`RVTOL_SEG_GAMMA=2.0` 对照实验净负，不做 |
| `OCR_PAD_WIDTH_MIN` | 224 | 速度数字窄图最优；降宽省推理但 +19 误读 |
| `DEFAULT_BUFFER_SIZE` | 128 | 64/128/256 无显著差异；128 兼顾突发背压 |
| `OCR_BATCH_SIZE` | 16 | B=16 最优；B=8 退化，B=32/64 更大等待/内存成本抵消 |
| `DECODE_BATCH_SIZE` | 16 | 固定成本摊薄；32/64 在当前场景无额外收益 |
| `TRT_PROFILE_BATCH` | 6 | pb8/12/16 单调变差（更大 TRT kernel 与 NVDEC 抢 GPU） |
| `TRT_WORKSPACE_BYTES` | 1GB | 构建用；FP16/INT8 无收益，保持 FP32 |
| `CPU_CORES_SPLIT_THRESHOLD` | 8 | 少核 CPU 软解分核阈值 |
| `OCR_ONNX_CHUNK` | 16 | ONNX 推理单批上限；16 比 64 内存峰值更低且更快 |
| `OCR_CTC_CHUNK` | 64 | CTC 后处理分块，控制内存峰值 |

---

## 6. 已验证死路（不要重新投入）

- **异步批量解码**：fork `experiment/async-batch-decode` 完整实现后实测生产路径
  0% 收益（test6 灰 ROI 1734 vs 1741fps）。全部测试视频均为 NVDEC 硬件解码上限，
  软件层无法超过硬件解码器。
- **用编码码流判断 ROI 是否变化**：`pkt_size/pict_type` 在移动背景中分离度仅
  0.126-0.417，静态背景召回仅 0.269；要拿到可用信号必须熵解码，成本≈解码本身。
  现有 `sample_stride` 分频采样已拿到主体收益。
- **Tesseract OCR**：对定制数字 ROI 正确率 33-95%，吞吐 ~15 段/s vs ONNX 双实例
  ~400 段/s（慢约 25 倍）。
- **FP16 / INT8 TRT 引擎**：tiny/small 非算力受限，实测无收益；FP16 构建还慢 2.2 倍。
- **多预处理自动选择 / 窗口重 OCR 自动化 / scipy 连通域**：均已被现有方案覆盖或净负。
- **onnxruntime 1.29 新增参数**（`ORT_INTRA/INTER_OP_NUM_THREADS`、parallel
  执行、spin off）：全部无收益，保持现状。
- **stride>1 跳过 B 帧（AVDISCARD_NONREF）加速解码（2026-08 尝试，已封板）**：
  在 decord `GetBatch` 等差步长快速路径中，对非采样帧设置
  `AVDISCARD_NONREF` 跳过 B 帧，并按 packet 计数推进帧号。初步速度提升明显
  （3000 帧 stride=8 从 ~0.63s 降到 ~0.12s），但正确性失败：
  - 与 `seek_accurate` 真值对比大量采样帧 `maxdiff=255`；
  - FFmpeg h264 报 `missing picture in access unit`；
  - 即使只在“目标是参考帧”时启用、并对跳过参考帧做显式排空，仍然错帧。
  结论：当前 decord 的 FFmpeg 多线程解码架构下，动态切换 `AVDISCARD_NONREF`
  不可靠。
- **stride>1 跳过 B 帧（packet 级过滤，2026-08 二次尝试，已封板）**：
  不依赖 `skip_frame`，改为在 `PushNextFiltered()` 中按 `pict_type` 直接把 B 帧
  packet 丢弃、只把参考帧 packet 推给 decoder。结果仍然：
  - 大量 `missing picture in access unit`；
  - 与 `seek_accurate` 真值对比仍有多帧 `maxdiff=255`。
  根因：H.264 High profile 中部分 B 帧本身是参考帧，仅按 `pict_type==B` 丢弃会
  丢掉解码依赖；要正确过滤必须解析 slice header 的 `reference` 标记，复杂度接近
  重写一个跳帧解码器。
  结论：在现有 decord/FFmpeg 架构内，安全跳过 B 帧加速 stride>1 解码的成本和风险
  都过高，暂不继续。
- **三级流水线（OCR 专职 preprocess 线程 ×2）**：严格单跑 A/B 无净收益，回滚；
  当前两段式（主循环 flush 内 preprocess 与 infer 线程并行）已是最优近似。

---

## 7. 依赖性能笔记（摘要）

- **onnxruntime 1.29.0**：PyPI 版升级安全、性能持平（test5 h264 -2%，test6 AV1
  +1% 波动内）；逐帧读数与 1.28 完全一致。
- **decord fork**：必须使用自建 `chr431/decord`，PyPI 版不支持 ROI-first /
  GPU gray / YUV420 / 等差步长快速路径。`sample_stride>1` 建议 fork ≥v0.7.12。
- **TensorRT**：只装 thin binding（`tensorrt_*_bindings`），运行 DLL 从 PATH 加载；
  首次构建 FP32 引擎约 1 分钟；FP16 不推荐。
- 详细依赖版本与已知问题见 `docs/DEPENDENCIES.md`。

---

## 8. 性能演进摘要（拆仓前 RaceVideoToLog 历史）

- v2.14：CPU 解码 + CPU 推理 +22%（12.3s → 9.6s）；批量流水线替代逐帧 next_roi；
  ONNX 分片 64→16（峰值 920MB→300MB）。
- v2.15：CPU+NVDEC 混合解码、TRT+ONNX 混合 OCR、GPU gray 输出、YUV420 输出；
  双解码器背压修复内存 11GB→500MB。
- v2.15.2：AV1 帧并行修复（dcd=12 326→648fps）、双 ONNX 实例、少核分核。
- v2.16 拆仓：识别链独立为 `video_ocr_engine`，本文档随拆仓迁移。

---

> 维护约定：新增性能实验后，把结论（尤其“死路”）追加到本文档，避免重复投入。
