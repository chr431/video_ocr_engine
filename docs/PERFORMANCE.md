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

## 4. 混合解码 / 混合 OCR（队列实测无提升，维持默认关闭）

> **2026-08 队列实测结论**：在 `D:\Videos\batch_test` 5 个视频、stride=8、
> 默认 auto+auto+gray+merge 条件下：
>
> | 方案 | 总耗时 |
> |---|---:|
> | 默认（不启用混合） | 109.6s |
> | `RVTOL_HYBRID_DECODE=1` | 108.8s（-0.7%，波动内） |
> | `RVTOL_HYBRID_OCR=1` | 111.6s（+1.8%） |
> | 两者同时启用 | 111.9s（+2.1%） |
>
> 结论：混合方案在 5 视频队列上**没有实际性能提升**（解码混合基本持平，OCR 混合
> 甚至略慢），因此维持默认关闭，不建议启用。代码和 env 钩子保留，不主动删除。

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

## 4.5 单实例双完整流水线并行（2026-08 新实验，默认关闭）

与 4 节的“混合解码/混合 OCR”不同：双完整流水线把同一视频切成多个连续小片，
两条完整“解码→分段→OCR”流水线（各自独立解码器 + OCR 引擎）作为消费者从队列
动态取片，最后按帧序合并。

测试：`D:\Videos\batch_test\新三国01.mkv`，h264 标清，
`frame_end=24000`、`stride=4`（6000 个采样帧），auto/gray/merge_similar：

| 方案 | 耗时 | 相对单 auto |
|---|---:|---:|
| 单流水线 auto+auto（GPU+TRT） | 7.32s | 1.00× |
| 双流水线默认（auto+auto ∥ cpu+cpu，2 片） | 4.97s | **0.68×（-32%）** |
| 双流水线默认（4 片） | 5.27s | **0.72×（-28%）** |
| 双流水线默认（8 片） | 5.70s | **0.78×（-22%）** |

补充观察：

- 默认双流水线（主+互补）在本场景有实际收益；显式两条 `("cpu","auto")`
  仍可能略快，但这依赖“CPU 软解 + TensorRT”是当前机器最优单流水线组合。
- Plan B（持久 OCR 会话跨片流式）已消除“每片 join OCR”屏障；8 片仍比
  2~4 片略慢，剩余成本主要是跨片 seek 与边界处理。
- 双流水线输出的唯一非空字幕集合与单流水线一致（抽查无缺失），但重复/
  噪声分段更少（如 516 段 → 447 段）。

### Race 跨编码补充（2026-08，1500 帧窗口）

| 视频 | 编码 | 单 auto | 默认双 2 片 | 默认双 8 片 |
|---|---|---|---:|---:|
| test | HEVC | 2.09s | 2.19s | 1.99s |
| test2 | h264 | 1.54s | 1.81s | 2.01s |
| test3 | h264 | 1.75s | **1.45s** | 1.93s |
| test5 | h264 | 1.84s | **1.71s** | 1.93s |
| test6 | AV1 | **1.61s** | 2.29s | 2.70s |

结论：引擎级双流水线在 Race 速度数字/多编码场景**不适合作为通用默认**。
动态切片能让 GPU+TRT 快路径拿到更多片，但 CPU 软解+ONNX 慢路径（尤其 AV1）
仍可能成为最终瓶颈；建议仅在用户显式开启且已知 h264/CPU 较强时使用。

### 双流水线重构（2026-08 二轮：探针定位 + 机制修正）

初版双流水线收益不稳定的根因，通过新增剖面探针（`RVTOL_PROFILE=1` 下
`producer:pipeN` / `ocr:pipeN` 分相聚合 + `parallel_pipeN_timeline`
分片时间线）逐项定位：

1. **TRT⊕ONNX 混配的单边饥饿（引擎级微基准定位）**。用剥离解码/生产者的
   引擎级速率探针（真实宽 ROI 输入、批 16、独跑 vs 同进程双线程 vs 跨进程）：
   - 干净速率：TRT **2.57** ms/段；ONNX(16T/14T) **8.44/8.28** ms/段。
     早先"ONNX 单跑 16.5ms"是被其所在流水线自身解码+生产者争抢污染的读数
     ——混合流水线里 ONNX 8.54ms 即满速，**从未被拖慢**；
   - 真正单边受损的是 TRT：∥ONNX14T 时 2.57→4.47 ms（同进程），跨进程
     仍 4.05（→非 GIL）；`RVTOL_ORT_SPIN=0` 无效（→非自旋等待）；在真实
     双流水线里叠加解码竞争后恶化到 8.28。机制 = ONNX 的真实矩阵计算占满
     物理核，TRT 宿主线程（HtoD 提交/stream 同步轮询）被调度饥饿；
   - 解除路径量化：ONNX 削到 6T/4T 可把 TRT 恢复到 3.39/3.22（+32%/+25%），
     但任何混配合计吞吐（≤373 段/s）都不及纯 TRT 单跑（389）——
     **混合架构没有赢面，正确解是双流水线都用 TRT**（trt∥trt 单引擎虽互退
     +40%，合计 ~556/s 仍线性扩展）；
   - 已固化 `DUAL_PIPELINE_ONNX_PEER_THREADS=6`：显式 `dual_backends` 混配
     时自动给 ONNX 侧限流，保护对端 TRT 宿主线程。
2. **NVDEC 是固定功能资源**：两个 NVDEC 解码会话合计吞吐反而比单个低
   ~15%（TRT+TRT 对照实测）；FFmpeg 软解才是解码侧的真实增量。
3. **旧方案 -25~-35% 的相当部分来自阈值漂移的"意外减负"**：每片各自校准
   Otsu → 各片二值化不一致 → merge_similar 在部分片上更激进（832 段 vs
   单流水线 1889 段，OCR 次数近乎减半）。唯一文本集一致（211=211）属侥幸
   （字幕重复性强），对逐段精确 OCR 场景是正确性风险。
4. **让位判定曾被背压噪声污染**：片墙钟含 OCR 队列 q.put 阻塞，GPU 路径
   因对方 OCR 拥塞被误判为慢路径而让位（timeline 显示 GPU 只干了头部片）。
   改用生产者净耗时（decode+gray+sharp+bin+segment 分相和）作吞吐口径。

对应机制修正（`_run_pipelined_parallel` 重写）：

- 默认互补对改为 `(auto,auto) ∥ (cpu,auto)`：两条都用 TRT，仅解码互补；
- 头部"试点+确认"小片组（各约 1/24 视频长，4 片预留）实测两侧生产者净速率，
  分级让位（稳态比 <0.8 确认 / <0.35 单片即可），慢路径最多浪费自己那组小片；
- 全局 Otsu 校准一次（前 50 采样帧），各片阈值与单流水线一致（消除 #3 漂移，
  段数与单流水线完全一致）；探测 reader 移交复用省一次打开；
- 跨片边界 merge_similar 缝合：相邻片尾/首段代表帧相似则并入前段；
- 采样帧数 < `DUAL_PIPELINE_MIN_FRAMES`(3000) 或 AV1 编码时回退单流水线。

最终实测（7945HX + RTX 4060 Laptop，A/B 单跑串行）：

| 场景 | 单流水线 | 双流水线 | Δ |
|---|---:|---:|---:|
| 新三国01 窗口（24000 帧 stride=4，6000 采样帧） | 9.46s | 6.71s | **-29%** |
| 字幕批量 5 集全片（stride=8，含后处理） | 115.0s | 84.0s | **-27%** |
| test5 全片（h264，7223 帧） | 7.83s | 4.51s | **-42%** |
| test3 全片（h264） | 3.27s | 2.40s | **-27%** |

输出一致性：字幕批量 CSV 与单流水线仅差 1 行（跨片缝合吸收的重复行）；
Race 段数 ±1；唯一文本集逐一核对一致。短窗口（<3000 采样帧）回退单流水线，
无回归。

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
