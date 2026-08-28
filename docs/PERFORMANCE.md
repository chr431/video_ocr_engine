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
- 引擎已支持 `ENGINE_PROFILE=1` 细粒度剖面：
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

> **2026-08 四轮更新：gray+NVDEC+TRT 场景的默认主路径已切换为显存全驻留
> GPU 管线**（`GPU_PIPELINE=0` 可回退宿主）。正确性与宿主逐位一致、
> 窗口 clean 约 -10%、内存争抢下墙钟更稳定；整集 stride8 双方同受 NVDEC
> 跳帧解码供给率限制（速度持平）。详见引擎仓 CLAUDE.md"GPU 管线转正"小节。
> 以下为宿主管线的历史基线数据，仍适用于 YUV 输出 / OCR=cpu 场景。

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
- `OCR_THREADS` env 钩子优先（实验用）。
- **少核 CPU 软解分核**（`CPU_CORES_SPLIT_THRESHOLD=8`）：
  - 物理核 ≤8 且 CPU 软解时，OCR 与 decord FFmpeg 帧线程各分 `cores//2`。
  - 4 核 CPU+ONNX：28.0s vs 33.1s（-15%）
  - 8 核 CPU+ONNX：17.8s vs 20.7s（-14%）
  - 16 核：分核反而差，保持 OCR 全核、FFmpeg 默认 2 帧线程落 SMT。
- **AV1 CPU 软解**：dav1d 帧并行上限约 6.6 核；AV1+CPU 解码任何核数用
  `dcd=ocrT=cores//2` 最稳（16 核 45.7s vs 12/4 的 58.5s）。
- **双 ONNX 实例 OCR**（`OCR_INSTANCES=0` 关闭）：
  单实例 intra-op 线程池扩展亚线性（16 线程仅 4.2×）；两个独立实例各 `ocrT//2`
  线程并发取批，纯吞吐 313→355 段/s（+15-18%），RSS +~200MB。显式 OCR=cpu 且
  核数≥8 时默认启用。
- **decode batch / FFmpeg 线程进一步扫描**（2026-08 补充）：
  `DECODE_BATCH_SIZE` 32/64、FFmpeg 解码线程 4 在标清宽 ROI 跳帧场景无收益，
  维持现状。

---

## 4. 已删除的混合解码 / 混合 OCR（历史结论，档案见 docs/ARCHIVE.md §A）

> **状态（2026-08 归档）**：v1 混合解码 / TRT+ONNX 混合 OCR 的实验记录已
> 整体迁移至 `docs/ARCHIVE.md`（§A），本节只保留状态摘要。
> 现役：CPU+NVDEC 混合解码为 **v3/v4（速率比例分界 + 两端连续扫掠，
> `hybrid_decode.py`）**，激活条件 = 显式 `decode_backend="hybrid"` +
> NVDEC 可用 + stride==1 + 未开 GPU 全驻留管线；编码门控（AV1 回退）已
> 移除。TRT+ONNX 混合 OCR 保持删除。

### 4.4b CPU+NVDEC 混合解码 v2 退化根因与 v3 重写（2026-08，探针实测；v3 现役）

v2（kfe 共享队列竞争）在 CPU 慢于 NVDEC 时退化（用户报告"总体被 CPU 拖累、
甚至不如单独 NVDEC"）。用 HYBRID_PROBE=1 逐片时序探针定位（HEVC test.mp4，
CPU 465fps vs GPU 2132fps）：

1. **交替领取**：FIFO 竞争 + 共享 in-flight 令牌使分片在 GPU/CPU 间严格交替
   （GPU 拿 #0,2,4,6,8,9；CPU 拿 #1,3,5,7,10）。消费者按全局帧序取帧 →
   慢生产者的每一片都是关键路径串行等待，快生产者被令牌限制无法超前；
2. **seek 爆炸**：交替领取使"连续扫掠免 seek"失效——每个生产者除首片外
   几乎每片 seek（GPU ~50-190ms/次、CPU ~35-65ms/次）；
3. **结果**：HEVC hybrid decode 2.4-2.8s 反比纯 NVDEC 2.0s 慢 20-40%。

v3 重写（速率比例分界 + 两端连续扫掠 + 对称接管）：
- `hybrid_begin` 并行实测两后端顺序速率（256 帧 + 16 帧 warmup 丢弃），
  按速率比例把分片切成两段：快端从头连续扫掠（0 次 seek），慢端 seek 一次
  到分界片首后连续扫掠（1 次 seek）；慢端份额夹 [15%, 45%]，速率比 >1.8x
  只给 1 片试探；
- 快端扫完自己区后逐片接管慢端未开始片（一次 seek 连续扫掠）——校准误差自愈；
  慢端只做自己区、区空即退出（不反向接管，避免破坏快端连续扫掠）；
- 每生产者"已产出未消费"片数 ≤ inflight（默认 2）防字幕宽 ROI 内存暴涨；
- **编码门控（AV1 回退）移除**：v3 实测 AV1 不退化，尊重用户显式选择。

本机实测（7945HX + RTX 4060 Laptop，A/B 单跑；decode 阶段耗时，越小越好）：

| 视频 | 编码 | NVDEC | CPU | hybrid v3 | vs NVDEC |
|---|---|---|---|---|---|
| test5 6000帧 | h264 | 5.99s | 5.17s | **4.37s** | **-27%** |
| test3 3000帧 | h264 | 2.91s | 2.84s | **2.44s** | **-16%** |
| test.mp4 3000帧 | hevc | 1.97-2.28s | 4.47s | 2.05-2.22s | 持平 |
| test2 3000帧 | hevc | 2.10s | 4.41s | **1.77s** | **-16%** |
| test6 3000帧 | av1 | 1.86-2.22s | 6.19s | 1.80-1.98s | **-10~19%** |

文本一致性：所有场景唯一文本集与单路径 100% 一致（段数/代表帧一致）。
端到端墙钟：h264 场景混合显著更快；HEVC/AV1 场景与纯 NVDEC 持平
（ocr_tail 略大——解码更快导致 OCR 积压排空，属正常流水线行为）。

### 4.4c CPU+NVDEC 混合解码 v4（2026-08，动态分界 + 稳态折扣 + 短校准；v4 现役）

**目标**：CPU 解码明显慢于 NVDEC（弱 CPU，8 核亲和模拟）时 hybrid 仍提供
decode 提升。背景：v3 在 rf>rs*1.8 时只给慢端 1 片试探，弱 CPU 下 hybrid
decode 反而比纯 NVDEC 慢（h264 8 核 +20%、HEVC 8 核 +7%）。

**探针定位**（并行争抢探针 + 分相 profile，勿再猜）：
1. NVDEC 与 CPU 软解互不拖慢（并行解码 GPU 仅降 9-16%）；真正瓶颈是
   **慢端拖尾 + OCR 尾批 + 校准固定开销**；
2. **短校准高估 CPU 稳态速率**：HEVC 软解有缓冲衰减，48 帧测 495fps、
   384 帧测 205fps（快测高估 2.2 倍）→ 按速率比例给慢端多片时慢端
   拖尾、decode 反被拖慢；
3. **OCR 尾批堆积**：hybrid decode 结束更早，OCR 尾批来不及排空 →
   ocr_tail 增大（+0.1-0.2s），墙钟被 OCR 吃掉；
4. **校准固定开销**：256 帧校准在弱 CPU 下 ~0.4s，吃掉 decode 收益。

**v4 设计**（`hybrid_decode.py`）：
- 短校准（默认 40 帧 + 8 warmup，`HYBRID_CALIB_FRAMES` 可调）；
- 稳态折扣（慢端=CPU ×0.45、=NVDEC ×0.85，`HYBRID_SLOW_DISCOUNT` 可调）：
  修正短校准对 CPU 软解稳态速率的高估；
- 动态分界（`_dynamic_split` 纯函数，单测覆盖）：慢端片数从 1 递增，只要
  "慢端生产时间 ≤ 快端生产时间×0.95"就继续——慢端贡献最大化且不拖尾；
- 慢端预取（`HYBRID_SLOW_INFLIGHT` 默认 4）：尾段提前就绪，减少 OCR 尾批；
- 其余（连续扫掠/对称接管/inflight/接口/激活条件）同 v3。

**实测**（TRT venv，进程亲和 8 逻辑核 = 弱 CPU 模拟，交错 A/B 3 轮中位）：

| 场景 | 编码 | NVDEC decode | hybrid v4 decode | Δ decode | Δ 墙钟 |
|---|---|---|---|---|---|
| test5 3000帧 | h264（CPU 慢 23%） | 2.956s | 2.420s | **-18.1%** | **-2.0%** |
| test.mp4 3000帧 | hevc（CPU 慢 4.6×） | 1.345s | 1.300s | **-3.3%** | +11.4% |

16 核无亲和回归：test5 h264 decode -24.5%、wall -12.7%（与 v3 持平）。
文本一致性：全部 100%。结论：CPU 明显慢于 NVDEC 时 hybrid decode 确实
提升（h264 -18%、HEVC -3%）；h264 墙钟转正（-2%）；HEVC 墙钟仍受 OCR
尾批/争抢影响（+11%）——CPU 慢 4.6× 时慢端最多 1-2 片，贡献上限 ~5%，
decode 收益不足以覆盖 OCR 固定开销属物理限制。基准工具 `bench_hybrid.py`
新增 `--affinity N`（进程绑定前 N 个逻辑核）复现弱 CPU 场景。

---

## 4.5 单实例双完整流水线并行（历史，已归档 — docs/ARCHIVE.md §B）

> **⚠️ 本节全部内容已迁移至 `docs/ARCHIVE.md`（§B）**，包括：双流水线
> 设计/实测、跨编码对照、二轮探针归因（内存子系统争抢）、五轮修正
> （seek/让位）、关键帧分片实验、kfe 死路与复活、AV1 对照、
> DUAL_PROPORTIONAL / DUAL_PRIORITY / 让位阈值 0.5 等全部历史条目。
> 代码中不存在 `_dual_pipeline.py` / `tests/test_dual_pipeline.py`，
> **也没有任何 `DUAL_*` 环境变量或构造参数**（基准提交 e8b2637）。
> 现役并行维度只有一个：`decode_backend="hybrid"` 的 CPU+NVDEC 双解码
> 生产者竞争（`hybrid_decode.py`）。勿按 `DUAL_*` 参数调优。

（完整档案见 docs/ARCHIVE.md §B）

---

## 5. 已锁定参数（勿随意改动）

| 参数 | 值 | 结论 |
|---|---|---|
| `OCR_GAMMA` | 2.0 | OCR 预处理灰度 gamma；全量最优，固定 |
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

---

## 9. GPU 分段 + ONNX OCR 无净收益（默认门控调整，2026 实测）

背景：GPU 管线与 OCR 后端解耦后，"GPU 分段 + ONNX 宿主 OCR"（`gpu_onnx`）
作为可用配置存在。受控 A/B 验证其是否值得成为默认：

**方法**：同一 NVDEC 解码 + 同一 ONNX CPU OCR + 相同采样（stride=8），唯一
差异 = 分段/校准/合并/预处理在 GPU（cuda-python kernel）还是宿主（numpy）；
各 3 次取中位（丢弃首轮 warmup），段数两边完全一致（562 / 77）。

| 场景（stride=8） | 管线 | 墙钟 | 解码相 | OCR 相 | 段数 |
|---|---:|---:|---:|---:|---:|
| test5（h264，ROI 33×106，5000 帧） | GPU 分段 + ONNX | 5.56s | 4.65s | 4.79s | 562 |
| | 宿主 + NVDEC + ONNX | **5.31s** | 4.75s | 4.81s | 562 |
| 新三国01（h264，字幕 ROI 26×676，6000 帧） | GPU 分段 + ONNX | 2.70s | 1.56s | 1.95s | 77 |
| | 宿主 + NVDEC + ONNX | **2.52s** | 1.67s | 2.02s | 77 |

**结论（勿再重复投入）**：
- **GPU 管线 + ONNX 相对宿主管线无优势，实测反慢 4~7%**。省的是每帧
  asnumpy（3.5~26KB D2H）+ 微小 ROI 上的 numpy 分段（微秒级）；加的是每批
  kernel 启动 + `analyze_batch`/`histograms_perframe`/`compare_pair`/实时
  D2H 的同步调用（串行化 producer，吃掉解码重叠余量）+ luma/prev/池帧 D2D。
- 两侧瓶颈相同：NVDEC 解码供给率（h264 ~1000+ 源帧 fps 等效）与 ONNX OCR，
  GPU 管线只优化分段侧——分段侧在 ≤676px 宽 ROI 上不构成瓶颈。
- 只有在 **ROI 极大（≥10 万像素）且分段成为墙钟主项** 时 GPU 分段才可能
  有净收益（且受 NVDEC 供给率上限约束）——无实测先例，未立项。
- **门控调整**：GPU 零拷贝管线默认仅在 **NVDEC+TRT** 组合启用（全程 raw 才
  有量级收益）；无 TRT / ocr_backend="cpu" → 宿主管线（配置面更简）。
  `GPU_PIPELINE=1` 保留为强制开关（允许 GPU 分段+ONNX 实验组合）。

---

## 10. 2026-08 一轮性能试验（引擎现役代码，7945HX + RTX 4060 Laptop）

本轮按清单逐项 A/B（单跑串行，test5 窗口 3000 帧 stride=1 为基准，
新三国01 窗口 3000 源帧 stride=8 为高段数场景）。**基线**：

| 配置 | 墙钟 | decode | ocr | 段数 |
|---|---:|---:|---:|---:|
| GPU 管线默认（gpu_yuv） | 3.238s | 2.975s | 3.10s | 1083 |
| hybrid（GPU_PIPELINE=0） | 3.127s | 2.31s | 2.72s | 1083 |
| 新三国01 窗口（GPU 管线） | 8.64~9.5s | — | — | 503 |

### 10.1 实验②：OCR 会话提前到校准前（有效，-19%）

改动：`_start_ocr_session`（引擎加载/模型构建，实测 TRT 加载 0.392s）从
"校准后"提前到"校准前"启动（worker 线程），与校准 + decode 开头并行。
宿主路径（extractor）与 GPU 路径（_gpu_pipeline）同步接线；引擎就绪前
`_emit_ocr` 自动走 host 回退（raw_ready=False），语义不变。

实测（test5 窗口）：模拟原顺序（串行引擎加载 + 管线）4.002s vs 新顺序
3.239s —— **引擎加载 0.4s 从墙钟消失（decode 2.98s 期间引擎已就绪）**。
段数/唯一文本与基线逐位一致（1083/265）。

结论：**decode 占墙钟 92% 时，任何"启动阶段"成本（引擎加载/校准）都应
与 decode 重叠**；同类机会（如首次 TRT 构建 ~1-2min）同理。

### 10.2 实验③：探测缓存（微小收益，零风险保留）

`nvdec_available`/`tensorrt_available` 加 `lru_cache`（按参数/无参）。
实测：冷探测 nvdec 0.033s + trt 0.079s（进程首轮含 CUDA 初始化 0.16s），
热探测本就 <0.03s；批量 3 次 extract 有缓存 5.538s vs 无缓存 5.589s
（**省 0.05s，<1%**）。

结论：探测不是瓶颈（decode 主导），但缓存零成本、避免"每次 extract
重新打开视频探测"，保留。**勿再投入跨 extract 复用 OCR 引擎**——引擎
加载已被实验②完全隐藏，复用无额外收益。

### 10.3 实验④：_segments_similar int16 比较（无收益，数值等价保留）

`np.abs(a.astype(float32) - b.astype(float32))`（2 个 float32 全帧临时
数组）→ `np.abs(a.astype(np.int16) - b.astype(np.int16))`（1 个 int16）。
实测新三国01 窗口（503 段边界）：int16 9.499s vs float32 9.416s
（**±1% 波动内持平**），段数/文本逐位一致（503/262）。

结论：**段边界判定不是墙钟项**（数千次调用仅 ~µs 级）；int16 版省内存
且与 GPU `sim_pair` 整数精确语义一致，保留为一致性改进。勿再优化此函数。

### 10.4 实验①：hybrid 分片粒度上限（防御性，测试集不可触发）

`HYBRID_MAX_CHUNK_FRAMES`（默认 0）：>0 时把超过该采样帧数的 hybrid
分片继续拆小（`_split_oversized`，优先关键帧边界/否则等分吸附采样帧
网格），内存上界 = inflight × 上限。实测：**所有测试视频关键帧间隔
~286-300 帧（标准 h264 GOP），kfe 单片 ~300 帧，拆片不触发**；
test5 全片 hybrid 峰值 RSS 1022MB（大头不在 ch['data']，而在 decord
缓冲池 + OCR 批 + numpy 临时）。

结论：**现有测试集无法 A/B**（无长 GOP 视频）；保留为防御性开关
（真实世界超长 GOP >256 帧/片时生效），单测覆盖拆片正确性。
`HYBRID_MAX_CHUNK_FRAMES` 默认 0 不改变现行为。

### 10.5 实验⑤：hybrid 批量交付减锁 + 多轮校准（均无净收益，保留开关）

- `_pop_frames` 批量交付（get_batch 一次锁取同片连续帧）：3.203s vs
  基线 3.127s（**+2.5% 波动内持平**）——消费者不是瓶颈（decode 2.3s
  主导），锁开销可忽略；
- `HYBRID_CALIB_ROUNDS=3` 多轮取中位数：3.879s vs 单轮 3.203s
  （**-21% 净负**）——3 轮额外测速 ~0.68s 成本 > 分界精度收益
  （test5 GPU 871→973fps 更准，但 CPU 稳定快端，分界 [0,7)→[0,6)
  不改变墙钟）。默认保持 1 轮。

结论：**hybrid 的墙钟瓶颈是解码供给率（NVDEC/CPU 顺序吞吐），不是
队列/锁/校准**；批量交付已作为 `_pop_frames` 实现落地（无净收益但零
风险），多轮校准保留为 `HYBRID_CALIB_ROUNDS` 实验开关（对速率比接近 1、
分界易翻转的视频可能有用，但需接受 0.68s 成本），默认 1 轮不启用。

### 10.6 本轮总结

- **有效**：实验②（OCR 会话提前，-19%，引擎加载隐藏到 decode 后）。
- **防御/一致性**：实验①（长 GOP 内存上限）、③（探测缓存）、
  ④（int16 比较）——测试集无净收益但零风险。
- **死路（勿重复投入）**：实验⑤批量交付减锁（消费者非瓶颈）、
  多轮校准（成本>收益）、跨 extract 复用 OCR 引擎（已被实验②覆盖）。
- **测试集限制**：所有测试视频 GOP ~300 帧 → 无法验证长 GOP 内存
  场景；如后续有长 GOP 片源应补测实验①。

---

## 11. 三条主力生产管线性能实验（2026-08，7945HX + RTX 4060 Laptop）

本机环境：test5.mp4（h264，7761 帧，59.8fps，ROI 843,993,948,1025 ≈
33×106 窄 ROI），新三国01.mkv（h264 标清，ROI 144,398,551,423 ≈
407×25 宽 ROI 字幕条）。三条主力管线：

- **CPU+ONNX**（decode=cpu, ocr=cpu）
- **NVDEC+ONNX**（decode=auto, ocr=cpu；默认门控 → 宿主管线）
- **NVDEC+TRT**（decode=auto, ocr=auto；默认 GPU 全驻留零拷贝管线）

### 11.1 基线（优化前，单跑串行中位）

| 管线 | test5 3000帧 stride1 | 新三国01 6000帧 stride8 |
|---|---:|---:|
| CPU+ONNX | 3.65s（decode 3.27 / ocr 3.56） | 2.27s（decode 1.51 / ocr 1.92） |
| NVDEC+ONNX | 3.39~3.65s（decode 2.97 / ocr 3.32） | 2.18s（decode 1.48 / ocr 1.87） |
| NVDEC+TRT | 3.18s（decode 3.03 / ocr 3.17） | 2.03s（decode 1.54 / ocr 1.71） |

段数/文本三条管线完全一致（test5：1083 段/265 唯一文本；新三国01：
77 段/40 唯一文本），为空文本比例与 OCR 后端无关的分段一致性提供基准。

### 11.2 探针定位（分相 profile + 微基准，勿再猜）

- **decode 是全部管线的绝对主项**（墙钟 92~98%）：NVDEC ~1000fps、
  CPU 软解 ~810fps（test5 h264，3000 帧 stride1）。
- **ONNX OCR infer = 第二主项**：单实例 16 线程 ~385 段/s、双实例
  8+8 ~458 段/s（批 16，宽 ROI 预处理输入）；`infer` 相位在
  CPU+ONNX 5.2s / NVDEC+ONNX 4.4~4.7s（含 OCR worker 排队重叠）。
- **TRT OCR infer 相位仅 ~1.0~1.3s**：批 16 全路径（pre+3×6 子批+
  DtoH+同步）微基准 ~10ms/批 → 1600 段/s，接近硬件上限；批 4 小批
  1255 段/s、批 6 1637 段/s、批 12 1654 段/s——**批 16 拆 6+6+4 子批
  是生产最优近似**。
- **拆批固定损耗 ~2ms/批**（ORT 单次 run 16 38ms vs 6 16ms 是线性
  计算量，不是拆批损耗；真正的拆批损耗是『同形状 3×6 50ms vs
  6+6+4 变形状 45ms vs 单次 16 38ms』中的 ~2ms/批固定调度开销）。
- **GPU 管线 producer 无 gray/sharp/bin/seg 分相**（_producer 线程
  内 profile 未接线）；decode 3.0s = NVDEC 供给率上限，GPU 分段
  kernel/同步/拷贝均非瓶颈（raw 聚批 16 帧仅 0.044ms，量级可忽略）。

### 11.3 落地优化（低风险）

1. **host 帧流 batch luma 预分配复用**（extractor._host_frame_stream
   + segmentation._gray_batch_out / _nv12_batch_luma_full_out）：
   复用每批灰度缓冲，避免每批临时数组分配。微基准 yuv 批量转换
   0.159→0.086ms/批（-46%）；端到端在测量波动内（decode 相位
   2.61→2.61s 持平，wall 3.65→3.66s 持平）。**净收益 <1%**，作为
   一致性改进保留（消除每批分配，数值逐位一致，76 单测 + e2e 全过）。
2. **TRT 输出 host 缓冲复用**（ocr_trt.execute_async /
   execute_device_async）：无 out_host 调用时复用『最大尺寸』
   np.float32 连续缓冲，避免每批重新分配。微基准 execute_device(6)
   4.16ms 持平；**生产路径（有 out_host）不受影响**（本就预分配整批
   输出），保持零风险。

### 11.4 已验证死路（勿再投入）

- **ONNX 分片粒度调整**：16→8/6/4 全部更慢（串行 16 38.3ms vs
  8 41.2ms vs 6 45.2ms vs 4 47.6ms/批16）；尾批 12+4 拆批损耗 ~4.5ms
  ——批 16 已是吞吐最优，`OCR_ONNX_CHUNK=16` 保持。
- **ORT 图内动态 batch 分片**：手动模拟 split（同形状/变形状）
  全部 ≥ 单批 16；ORT 1.27 下批 16 单次 38ms 是纯计算量线性
  （n=4 11.5ms → n=16 38.6ms），拆批只会加固定开销。
- **GpuPreprocessor 小批 D2D 聚批改批量接口**：raw 聚批 16/32/128
  帧全部 ~0.04ms，量级可忽略；聚批不是瓶颈，改动无收益。
- **pinned host 缓冲**：本机 DtoH 12.6MB/批与 enqueue 重叠，pinned
  只省 host 侧分配；复用普通缓冲已足够（见 11.3-2），不引入 pinned
  复杂度。
- **GPU 管线 producer 线程内补 profile 分相**：decode 是 NVDEC
  硬件上限，补分相无收益（保留为诊断空窗）。

### 11.5 优化后终测（交错 3 轮中位，与 11.1 同口径）

| 管线 | test5 3000帧 stride1 | 新三国01 6000帧 stride8 |
|---|---:|---:|
| CPU+ONNX | 3.66s（-0%） | 2.27s（-0%） |
| NVDEC+ONNX | 3.59s（-0%） | 2.37s（-0%） |
| NVDEC+TRT | 3.18s（-0%） | 2.02s（-0%） |

结论：**三条主力管线的墙钟瓶颈均为解码供给率（NVDEC/CPU 顺序吞吐），
OCR 侧（ONNX/TRT）已接近各自硬件上限；低风险优化只能做到零风险零
退化（batch luma 复用、TRT 输出缓冲复用），量级收益需来自解码侧
（hybrid 双解码 / NVDEC 供给率），非本次改动范围。**

### 11.6 代码结构拆分（2026-08，extractor.py 969 → 568 行）

`extractor.py` 过长（969 行）且宿主流水线逻辑与引擎骨架混杂，按已有
`_gpu_pipeline.py` 的 mixin 模式同构拆分：

- **`video_ocr_engine/_host_pipeline.py`（新，441 行）**：宿主路径
  模块级函数 `_host_calibrate` / `_host_frame_stream` /
  `_host_segment_frames`（原样迁移，含 batch luma 复用优化）+ 新
  `_HostPipelineMixin`（`_start_ocr_session` OCR 会话原样迁移）；
- **`extractor.py`（568 行）**：保留 FieldExtractor 骨架（构造/参数
  校验/`_open_vr` 解码器/`_run_pipelined` 分发/结果组装），类基类
  改为 `(_GpuPipelineMixin, _HostPipelineMixin)`；
- **兼容**：extractor 顶部 re-export 全部模块级函数与
  `_HostPipelineMixin`，旧导入路径
  `from video_ocr_engine.extractor import _host_calibrate` 等不变；
  公共 API 与行为零变化（76 单测全过，三管线 e2e 段数/文本逐位一致）。
