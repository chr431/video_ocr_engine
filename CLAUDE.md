# CLAUDE.md — 开发记录与约定

本文件用于记录 `video_ocr_engine` 的开发决策、实验结论与维护约定。
**README 只写用户向 API/使用说明**；开发过程结论（尤其“为什么这样做/为什么不做”）
写在这里和 `docs/PERFORMANCE.md`，避免污染用户文档。

> ⚠️ **当前状态（重要，先读这里）**：本文大量章节是 2026-08 双流水线实验的
> **历史记录**，其中"单实例双完整流水线并行"已完成两轮删除（最终提交
> e8b2637 "彻底移除 dual_pipeline"）。**现役代码没有 dual pipeline、没有
> `DUAL_*` 环境变量、没有 `_dual_pipeline.py` / `tests/test_dual_pipeline.py`**
> （历史提及均为旧档案）。现役并行维度只有一个：
> `decode_backend="hybrid"` 的 CPU+NVDEC 双解码生产者竞争（`hybrid_decode.py`，
> kfe 分片纯函数 `_keyframe_every_chunks` 也在此），OCR/分段仍是单路径。
> 下方其余章节（GPU 管线、TRT 拷贝、hybrid、分核、性能结论）为现行事实。
> 历史数值仅用于理解"为什么这样设计"，勿按旧 `DUAL_*` 参数调优。

## 通用约定

- 引擎是通用文本提取库，不携带速度/字幕领域后处理；领域语义由上层应用完成。
- 默认行为必须保持向后兼容：新功能默认关闭，除非明确作为新默认。
- 性能实验结论（尤其失败/无收益/最优参数）追加到 `docs/PERFORMANCE.md`，
  避免重复投入。

### 公共 API 清理（0.7.0，破坏性）

为收敛易用性，0.7.0 删除了一批遗留（下游两项目已独立维护、风险接受）：

- 构造参数 `fps`（此前即被静默忽略）；`gray_output`/`yuv_output` 降级为
  "已废弃别名"（仍接受，勿再使用；主参数 `rep_crop_format`）；
- 实例旧属性轨：`segments`(dict)/`rows`/`segment_frames`/`ocr_values`/
  `n_segments`/`ocr_texts`/`ocr_confidences` 属性（新 API 一律走
  `extract() → ExtractionResult`）；
- 方法 `prepare_review_rgb()`（新 API 下为 no-op，且会清 `crops`）、
  `timing_flat()`；
- `video_utils.open_decord_vr/rss_mb/sum_nbytes/VideoMetadata/format_duration`
  （仓库内零调用）；
- 保持：`FieldExtractor.frames` 属性（单测依赖）；`self.crops`（内部存储）。

后续新增遗留面一律先标 deprecated、两个版本后删除。

## 单实例双完整流水线并行（历史，已删除 — e8b2637）

> **本节全部为已删除功能的历史记录**，代码中不存在（基准：e8b2637 提交）。
> 保留价值：解释当时实验的损耗来源结论，避免后人重复探测；不要据本节
> 调参（`DUAL_*` 环境变量均已移除）。

### 设计

- 一个 `FieldExtractor` 实例内，把同一视频的采样帧序列切片；两条完整
  “解码→分段→OCR”流水线作为消费者动态取片，最后按帧序合并。
- **默认互补对 = `(auto,auto) ∥ (cpu,cpu)`**（修正后）：与下游
  `video_subtitle_extractor --dual` 一致，一条 GPU+NVDEC+TRT、一条
  CPU+ONNX，两条流水线分别利用 GPU 与 CPU 硬件。
  早期“混配无赢面”的结论被后续实际测试修正：当时双流水线的 CPU 路径缺少
  `seek_accurate` 到片首，解码随机访问让 CPU 侧吞吐约慢一倍；且混配下让位
  会把并行对端交给慢路径。修正后显式混配/默认混配实测与双 TRT 基本持平，
  相对单 TRT 约 -33%。
- 切片结构 = 头部小片组（试点×2 + 确认×2，各约 1/24 视频长）+ 大竞争片
  （默认 2 片）。头部组预留（每条流水线一组）兼作让位判定取样。
- 让位：按**生产者净耗时**（decode/gray/sharp/bin/segment 分相和，免疫 OCR
  背压噪声）比较稳态速率；稳态比 <0.8 确认让位、<0.35 单试点即可；
  慢路径让位后剩余片由快路径升序扫掠（升序连续 get_batch 无 seek 代价，
  实测乱序跳跃才有 ~150ms/次精确 seek）。
- 全局 Otsu 校准一次（探测 reader 上，前 50 采样帧），各片阈值与单流水线
  一致；探测 reader 移交给同后端方向的第一条流水线复用（省一次打开）。
- 跨片边界 merge_similar 缝合：相邻片尾/首段代表帧相似则并入前段
  （OCR 结果沿用前段），段数与单流水线一致。
- 回退：NVDEC/TRT 缺失、采样帧数 <`DUAL_PIPELINE_MIN_FRAMES`(3000)、
  AV1 编码（CPU 软解已知净负，`DUAL_NO_CODEC_FALLBACK=1` 可关）。
- 默认关闭（`dual_pipeline=False`），环境变量 `DUAL_PIPELINE=1` 可开启。

### 探针定位的损耗来源（2026-08 二轮，勿再猜测）

1. **混配退化的真因 = 内存子系统争抢（三轮探针定位，勿再用"核饱和/
   调度饥饿/GIL/对称收敛"表述）**。判别链：CPU 占用仅 ~46%（未饱和）；
   跨进程仍退化（非 GIL）；ORT 自旋参数无效（非自旋）；双侧绑核
   隔离+进程优先级组合无效（非调度/迁移）；GPU 时钟恒定 2490MHz、温度
   正常（非降频）；**单线程重 SIMD 计算（L2 驻留、零 DRAM 流量）完全不伤
   TRT；而 8 进程纯内存流拷贝（~100GB/s）让 TRT 10.26ms/段、enqueue 子相
   位膨胀 2.2→14.8ms/批（灾难级）**。机制 = 对端多线程宽矩阵乘的聚合访存
   流量占满 DRAM/Infinity Fabric，TRT 宿主提交路径（ioctl、页表遍历、
   pageable 拷贝 staging）全部变慢，GPU 反过来饿等提交（util 63%→42%）。
   另：早先"ONNX 单跑 16.5ms"是被其流水线自身解码/生产者污染的读数，
   干净速率 8.3-8.4 ms/段。
   解除路径量化：ONNX 削到 6T/4T 把 TRT 恢复到 3.39/3.22 ms/段，但任何
   混配合计吞吐 ≤ 纯 TRT 单跑的 96% → 默认对双 TRT 不混推理后端；
   显式混配时 `DUAL_PIPELINE_ONNX_PEER_THREADS=6` 自动限流（本质是限
   聚合访存流量）。trt∥trt 的互退（各 +40%，合计仍线性扩展）则是 GPU
   双上下文时间片切换，与内存无关。
   补充（四轮-b，已删除）：曾尝试削减 ONNX 侧可裁剪的输出流量——图级追加
   ArgMax+ReduceMax（需 onnx 包一次性构建缓存，
   数值与旧路径 100% 一致）。结果：共存时 TRT 退化仅 +75%→+67%、ONNX
   自身 -6%，单引擎反而 +10%（ORT 归约核不如 numpy 成块 SIMD）——证明
   混配干扰的主体是 ONNX 计算内部的激活/权重访存，而非可裁剪的 I/O
   张量；默认关闭。进一步缓解只剩模型量化路线（未立项）。

   > 注意：以上是引擎级剥离探针的历史结论，曾据此把默认互补对改成双 TRT。
   > 后续端到端双流水线实测推翻了“混配无赢面”的生产结论——真正把混配
   > 拖垮的是全局阈值路径缺少 `seek_accurate` 到片首（CPU 解码随机访问
   > 约慢一倍）以及混配下让位误把并行对端交给慢路径；修正后默认互补对
   > 已改回下游 `--dual` 的 CPU+ONNX ∥ GPU+TRT（见下文“五轮修正”）。
2. NVDEC 固定功能上限：两个 NVDEC 会话合计吞吐比单个低 ~15%，
   FFmpeg 软解才是解码增量（h264 标清 CPU 解码 ~1240fps vs 单 NVDEC ~810fps）。
3. 初版 -25~-35% 里相当部分来自每片阈值漂移导致的意外激进合并
   （832 段 vs 1889 段，OCR 近半被跳过）；全局校准修正后段数与单流水线一致。
4. 片墙钟含 OCR 背压等待，直接当吞吐信号会让 GPU 路径被误判让位
   （timeline 剖面可见 GPU 只干了头部片）→ 净耗时口径修复。
5. 升序连续扫掠无边界代价；乱序跳跃 ~150ms/次（decord 内部精确 seek）。

### 最终实测（A/B 单跑串行，7945HX + RTX 4060 Laptop）

| 场景 | 单流水线 | 双流水线 | Δ |
|---|---:|---:|---:|
| 新三国01 窗口（6000 采样帧，gray+merge） | 9.46s | 6.71s | -29% |
| 字幕批量 5 集全片（stride=8，含后处理） | 115.0s | 84.0s | -27% |
| test5 全片（h264 速度数字，7223 帧） | 7.83s | 4.51s | -42% |
| test3 全片（h264） | 3.27s | 2.40s | -27% |

输出一致性：字幕批量 CSV 与单流水线仅差 1 行（缝合吸收的重复行）；
唯一文本集一致；Race 段数 ±1。短窗口回退单流水线无回归。

### 五轮修正：混配双线程真正并行（2026-08，本机实测）

背景：引擎内双流水线显式混配（GPU+TRT ∥ CPU+ONNX）时，CPU 路径比下游
`--dual` 的独立实例慢很多。用“同一个视频切两半、两个独立 FieldExtractor
实例并发”作对照，发现独立实例 4.26s 即可完成，而引擎内双流水线要 6.6s+。

定位到两个可修复点：

1. **全局阈值路径漏了 `seek_accurate(start)`**：`_run_parallel_chunk` 在
   `th is not None` 时直接 `get_batch`，但 CPU 解码器还停在文件头/上次位置，
   decord 等价于每次随机跳到片首，CPU 解码吞吐约慢一倍（单一半 CPU 流水线
   2.52s → 引擎内同片 5.07s）。补一次精确 seek 后无试点混配从 6.57s 降到
   4.71s。
2. **混配下让位方向错误**：让位逻辑把本可并行的 TRT 对端也停掉，全部交给
   ONNX 路径，从 ~4.8s 退化到 ~6.8s。混配两条流水线分属 GPU/CPU 硬件，
   默认让位阈值取 `DUAL_PIPELINE_MIXED_SLOW_RATIO=0.5`（`DUAL_SLOW_RATIO` 仍可显式覆盖）；默认互补对同步改回
   下游 `--dual` 的 `(auto,auto) ∥ (cpu,cpu)`。

修正后本机实测（新三国01，3000 采样帧，stride=8，host 双流水线）：

| 方案 | 墙钟 | 相对单 TRT |
|---|---:|---:|
| 单 GPU+TRT | 7.28s | 1.00× |
| 双流水线（修正后，默认混配） | 4.84s | **0.66×（-34%）** |
| 双流水线（修正后，显式双 TRT） | 4.85s | 0.67× |
| 两个独立实例同视频两半（对照） | 4.26s | 0.59× |

结论：混配不再靠“让位止损”，而是真正同时利用 CPU+ONNX 与 GPU+TRT；
引擎内双流水线与下游双实例的差距从 ~2.4s 缩小到 ~0.6s。

### 每关键帧切一片的探针定位与死路（2026-08，勿再投入）

`DUAL_KEYFRAME_EVERY=1` 把大竞争区按每个关键帧边界切一片交给共享队列
自由竞争。用 ENGINE_PROFILE 分片时间线 + producer 分相探针定位：

- **片间死区 gap ≈ 0**：队列/唤醒/跨片缝合机制没有额外开销，分片本身
  不是瓶颈来源；
- **唯一随片数增长的开销是 `seek_accurate`**（h264 GPU ~50ms/次、CPU
  ~15-20ms/次，AV1 GPU ~20ms/次）。每关键帧一片使片序在两条流水线间
  交错，“升序连续扫掠免 seek”的假设失效，seek 总耗时随片数线性增长；
- **自由竞争会向“生产者快、OCR 慢”的路径倾斜**：h264 下 CPU 软解吞吐
  高于 NVDEC，小片竞争让 CPU+ONNX 抢到更多片，但 ONNX infer 是真正的
  墙钟瓶颈，整体反而变慢（test5 3.11s vs 基线 2.44s）。现有让位阈值按
  生产者净耗时判定，对这种“解码快、OCR 慢”的路径不触发（2 片中与 GPU
  基本均分，正好是墙钟最优）；
- AV1 下 CPU 生产者过慢，0.5 让位把剩余片交给 GPU，每关键帧一片与普通
  2/4 片基线基本持平（AV1 关键帧 seek 便宜、片多不亏也不赚）。

结论：默认保持“2 大竞争片 + 关键帧吸附 + 混配 0.5 让位”；
`DUAL_KEYFRAME_EVERY` 保留为实验开关但不默认启用。详细数值见
docs/PERFORMANCE.md“每关键帧切一片实验”。

### 六轮修正：关键帧切片复活——连续扫掠免 seek + 竞争闸门 + 端到端让位（2026-08，本机实测）

上小节曾把“每关键帧一片自由竞争”判为死路。本轮把两个障碍分别修复后，
该方案在失衡情境（字幕宽 ROI）取代 2 片成为双流水线最优，均衡情境
（Race h264 速度数字）与 2 片基本持平、仍明显优于单流水线。

**障碍 1 — seek 总耗时随片数线性增长**：

- 实测：h264 GPU 精确 seek ~40-70ms/次、CPU ~30-40ms/次；而“连续扫掠”
  （下一片起点 == 上一片终点，解码器已停在相邻位置）仅 ~1ms；
- 修复：`_run_parallel_chunk` 增加 `seek_required` 参数——仅在下一片与
  上一片不连续时才 `seek_accurate`。分片时间线里连续片的 seek 归零。

**障碍 2 — “解码快、OCR 慢”路径在自由竞争中抢片**：

- CPU+ONNX（尤其宽 ROI 字幕）解码比 NVDEC 快，但 ONNX 是真正的墙钟瓶颈；
  按生产者净耗时判定时它被误判为快路径、抢走多数小片，整体反而变慢
  （字幕 kfe 曾 20s vs 单 9.4s）；按生产者净速率让位还会反向误伤（
  max_chunks=16 时 GPU 被误让、CPU 独跑 12 片 16s）；
- 两层修复：
  a. **竞争取片闸门**（`DUAL_PIPELINE_INFLIGHT`，实测 1 最优）：in-flight
     片数（已取但 OCR 未排空）达上限即暂停取片等自己 OCR 追上来，让对方取；
     片数口径与内容无关，免疫“分段稀疏时段做不了多少 OCR 工作”的偏差；
  b. **端到端速率让位**：让位判定改用“片起点 → 该片 OCR 排空”的墙钟
     （e2e_speed，竞争闸门排空后记录）替代生产者净速率；双方都至少取过
     一片竞争片后才可能触发，天然规避试点头片 warm-up 噪声。

**实现附带修复**：

- OCR worker 尾批死锁：OCR worker 把不足 16 的尾批段先攒在 b_idx、等下一
  片补齐才 flush；竞争闸门若精确等 `len(results) ≥ pu` 而队列已空 → producer
  等排空、OCR worker 等下一片补齐批次，互等死锁（INFLIGHT=1 实测必现）。
  排空判定带半批容忍（≤OCR_BATCH-1 段），producer 先取下一片、头部段补齐
  批次即可恢复前进；
- 关键帧切片粒度：`DUAL_KEYFRAME_EVERY_MIN_GAP`（默认 16 采样帧）+
  `DUAL_KEYFRAME_EVERY_MAX_CHUNKS`（默认 8）——mkv 重编码关键帧过密（每
  30-140 源帧一个）时逐步放大间距合并，片数受控且边界仍落关键帧（seek 便宜）。

**本机实测**（A/B 单跑串行取最优，7945HX + RTX 4060 Laptop）：

| 场景 | 单流水线 | 双流水线最优 | Δ |
|---|---:|---:|---:|
| 新三国01 窗口（6000 采样帧） | 8.75s | kfe+INFLIGHT=1 8.52s | -3% |
| 新三国01 窗口 双 2 片（默认） | 8.75s | 13.36s | +53%（失衡下 2 片必回退） |
| test5 窗口（3000 帧） | 3.37s | 双 2 片 2.54s | -25% |
| test5 窗口 kfe | 3.37s | 2.87s | -15% |
| test3 窗口（3000 帧） | 3.22s | kfe 2.76s | -14% |
| test5 全片（7223 帧） | 7.66s | 双 2 片 5.71s | -25% |
| test3 全片（3190 帧） | 3.39s | 双 2 片 3.00s | -12% |
| test6 AV1（默认回退） | 单流水线 | 回退单（不回归） | 0 |
| 字幕整集 stride=8（新三国01 全片） | 21.38s | 双 2 片 11.68s | **-45%** |

结论：每关键帧一片不再靠“自由竞争”，而是靠“竞争闸门限速（按 OCR 排空）
+ 端到端让位方向正确（按含 OCR 的墙钟）+ 连续扫掠免 seek（相邻片零成本）”。
作为实验增强保留（`DUAL_KEYFRAME_EVERY=1`），默认仍是 2 片 + 关键帧吸附 +
混配 0.5 让位；字幕类失衡场景建议显式 `DUAL_KEYFRAME_EVERY=1` +
`DUAL_PIPELINE_INFLIGHT=1`（或把 chunks 提到 8）。详细数值见
docs/PERFORMANCE.md“关键帧切片复活”。

### 实现注意

- 每条 worker 使用“持久 OCR 会话”（`_start_ocr_session`）：一个 OCR worker +
  infer 队列跨所有切片复用，切片之间不做 join；后一片解码可与前一片 OCR 重叠。
- `ENGINE_PROFILE=1` 时各流水线 profile 按 `producer:pipeN / ocr:pipeN`
  聚合进 `self.profile`，分片时间线在 `timing['parallel_pipeN_timeline']`
  （(idx, t0, t1, frames, gap)），用于诊断串行化/空隙。
- 当前仅支持 2 条流水线；显式 `dual_backends` 不受编码回退影响。

### TRT 拷贝路径（2026-08）

- `TrtEngine.execute()` 已移除 `host_in` 固定 staging 数组：直接以当前批
  numpy 连续内存作为 `cudaMemcpy` 源，Host→Device 只保留一次拷贝。
- `_infer_locked()` 对 TRT 路径预分配整批输出数组，每个子批直接 DtoH 进
  对应切片，免去逐批 `host_out` 分配和 `np.concatenate` 拷贝。
- 使用专用 CUDA Stream + `execute_async_v3`：所有子批一次性异步 enqueue，
  最后只 `cudaStreamSynchronize` 一次，消除每子批一次 host-GPU 往返同步。
- 微基准（test5 3000 帧 / 16 图 batch）：
  - 基线 OCR micro：21.94ms/batch（729 img/s）
  - 异步 stream 后：18.58ms/batch（861 img/s，约 +18%）
  - GPU 预处理后：18.58ms/batch（861 img/s，micro 持平）
  - E2E：基线 3.457s → 3.252s（-6%，decode 仍为主瓶颈）
- pinned host staging 微测反而更慢（额外 CPU 拷贝），未采纳。
- GPU 预处理已落地为 `GpuPreprocessor`：
  - 对已 48 高的 float32 HWC 图在 GPU 完成 transpose+normalize+pad；
  - 直接生成显存模型输入，`TrtEngine.execute_device_async` 跳过 HtoD；
  - 与旧 CPU `_resize_norm` 输出逐项一致（随机批对比 same=True）。
- decord GPU 帧直通是 GPU 主管线内部 raw 路径（不再提供独立实验开关）：
  - 从 decord gray NDArray DLPack 解析 device ptr，代表帧 D2D 聚批后
    `prep_gray_raw` kernel 在 GPU 完成 resize+gamma+normalize+pad；
  - 独立开启曾实测反而慢 20~30%（当前仍做每帧 asnumpy 供分段，raw 只省
    代表帧，却增加 GPU kernel 与 D2D 竞争）；在显存全驻留主管线中该路径
    仍作为内部实现保留。
  - 结论：要真正收益必须把灰度/sharp/分段也留在 GPU。
- GPU 灰度/sharp/聚类分段已实现实验路径（`GPU_PIPELINE=1`）：
  - `GpuFrameAnalyzer` 一次 kernel 分析整批帧，只回传 (sharp, cluster) 标量；
  - 避免整帧 ROI D2H；校准阈值仍取前 50 帧 D2H。
  - 默认关闭。
  - 已继续优化：GpuPreprocessor 与 TrtEngine 共享 CUDA stream、
    raw 推理异步、GPU 分段批加大到 64、
    GPU 直方图 Otsu 校准、生产者线程让 analyze 与分段/OCR 重叠。
  - test5 1500 帧 A/B：开启仍比 host 慢（2.50 vs 1.74），raw 单独开启也慢
    （2.34~2.39 vs 1.71）。micro 上 raw 与 host 接近，E2E 慢主要来自
    GPU 路径与 decode/TRT 之间仍缺少真正统一的异步流水线。
  - 结论：当前实验路径已完整实现但无净收益；不建议启用。

### 显存全驻立路径补全与争抢韧性验证（2026-08 三轮，GPU_CTC）

在"内存子系统争抢"结论（见双流水线小节）之后，把 GPU+TRT 路径最后两个
RAM 大触点补掉，形成显存全驻留闭环：

- **逐帧直方图校准**（`GpuFrameAnalyzer.histograms_perframe`）：每帧 256-bin
  直方图在 GPU 统计，D2H 仅 B×1KB 标量表，宿主复刻"前 50 帧 Otsu 取中位
  数"语义——校准阈值行为与单流水线逐位一致（此前全局池化直方图阈值不同，
  段数差 4×）；
- **TRT 输出 GPU argmax 归约**（`GpuOutputReducer` + `TrtEngine.
  execute_device_argmax`，env `GPU_CTC=1`）：(B,S,C) float32 在 GPU
  上沿 vocab 维归约成 (B,S) 索引+概率，DtoH 从 ~16MB/批降到 ~12KB
  （~1300×）；并列取首个与 numpy.argmax 一致，宿主 `_ctc_from_idxprob`
  与原批解码语义一致。注意按 profile max_batch 分子批循环。

争抢韧性实测（新三国01 窗口 6000 采样帧，对端 8 进程内存流拷贝 ~100GB/s）：

| 路径 | 干净 | 争抢下 | 退化 |
|---|---:|---:|---:|
| 宿主路径 | 9.49s | 16.12s | ×1.70 |
| 显存全驻留 | 11.51s | 15.65s | **×1.36** |

唯一文本集 host vs gpu = 211/211 完全一致。

### GPU 管线转正（2026-08 四轮：merge_similar 补齐 + 默认启用）

四轮改动后，显存全驻留路径在 gray 输出场景正式取代宿主管线成为默认：

- **merge_similar 补齐**：代表帧灰度在 emit 时 D2H 成小数组（每段一张
  ~10KB，整片流量可忽略），段边界判定直接复用宿主 `_segments_similar`
  ——按构造逐位一致，且避免每边界一次内核启动+同步的开销（该方案曾使
  数千边界吃掉 1-3s）。keep_crops 时 rep_crops 复用同一副本。
- **analyze_gray 并行化**：旧实现每帧单线程串行扫 H*W*9，改为 block=一帧
  256 线程分片 + shared 归约（cluster 整数计数逐位不变；sharp 浮点和仅
  用于段内比较，微小差异无影响）。
- **默认启用门控**（`_gpu_pipeline_enabled` 重写）：gray_output=True 且
  decode∈{auto,nvdec} 且 ocr≠cpu 且 NVDEC/TRT 可用且 merge 分离模式非
  contrast 时自动启用（双流水线已移除，无并行互斥条件）；`GPU_PIPELINE=0`
  显式关闭，'1' 强制尝试。yuv_output 场景暂走宿主管线。
- 单测：门控矩阵 + 合并模式解析（tests/test_gpu_pipeline.py，9 例）。

实测（新三国01，7945HX + RTX 4060 Laptop，多次运行）：

| 场景 | 宿主 | GPU 全驻留 | 备注 |
|---|---|---|---|
| 窗口 6000 采样帧（stride4）clean | 9.2~10.3s | **8.7~8.8s（约-10%）** | |
| 同上 + 对端 ~100GB/s 内存流拷贝 | 12.6~17.0s（波动大） | **12.4~13.5s（更稳定）** | 负载相等后退化幅度相当，VRAM 路径墙钟可预测性更好 |
| 整集 stride8 全片 | 20.9s | 21.5s（±3%） | 双方都卡 NVDEC 跳帧解码供给率，OCR 已完全重叠 |
| 批量 5 集（stride8） | 114~115s | 113.3s | 同上 |

正确性：修复校准取值 bug 后，窗口与整集的段数/代表帧/文本与宿主路径
**逐位一致**（1889/1328/211 与 1151/573；rep 帧差异 0）。批量 CSV 仅剩
1 行残差（新三国03 章节卡的空格变体重复行，GPU 少合并一次，内容相同）。
结论：作为 gray+NVDEC+TRT 的默认路径成立；整集 stride8 受限于 NVDEC
跳帧解码供给率，两路径同瓶颈。

教训记录：GPU 校准曾把直方图行误传给 `_otsu`（它接收灰度图并在内部做
直方图），产生"直方图的直方图"垃圾阈值 th=23（正确为 86），导致段数
3888 vs 1889 与批量时间戳 ±1s 偏移——逐帧直方图必须配 `_otsu_from_hist`。
sharp 用 int64 精确累加 + summary float64 直传，保证近平局选帧与宿主
"严格大于保先者"语义对齐。
- 当前仍存在 1 次原始 ROI D2H（decord asnumpy） + 1 次 DtoH（TRT 输出）。

### 内部恒为单通道灰度 + rep_crop_format（yuv/gray）

- **引擎内部取消 RGB 链路**：decord 输出只有 `'yuv420'`（keep_crops 且
  `rep_crop_format="yuv"`，默认）或 `'gray'`；RGB→灰度转换移到解码侧/fork 内
  （调研结论：内部灰度性能更好且不影响准确度）。`keep_crops=False` 时自动
  退化为 `gray` 输出（省 0.5B/px UV 传输）。
- `rep_crop_format` 语义（取代旧 `gray_output`/`yuv_output` 组合；旧参数
  保留为 deprecated 别名）：`"yuv"`=packed NV12（内部只取 Y 平面，外部
  `nv12_to_rgb` 转 RGB——预览只对代表帧调用，毫秒级）；`"gray"`=灰度。
- **GPU 管线 = 零拷贝闭环（默认仅 NVDEC+TRT）**：分段/校准（hist/analyze
  kernel）、merge_similar 判定（`sim_pair` kernel，整数精确；contrast 模式
  在边界时 D2H 两帧 → 宿主 `_segments_similar`，kernel 化无净收益）、代表帧
  保活（gray=decord NDArray 指针 / yuv=NV12 NDArray 保留，Y 平面按需经
  `luma_into` 提取到 `_YFramePool` 池帧，~10KB D2D/次）、raw OCR（single
  TRT，`force_aspect` 已支持：content_w = round(48*aspect)，与宿主
  `_preprocess_standard` 语义一致）：过 RAM 的只有每帧两标量、校准直方图表、
  merge 标量（contrast 时两帧）、CTC 归约结果与 keep_crops 输出（每段一张
  D2H，结果必须给外部）。ONNX/无 TRT/引擎未就绪 → 代表帧 D2H + 宿主 OCR（
  仅经 `GPU_PIPELINE=1` 强制时才会出现 GPU+ONNX 组合——实测无净收益，见
  docs/PERFORMANCE.md §9，默认门控只放行 NVDEC+TRT）。
- **门控与 OCR 后端**：`_gpu_pipeline_enabled` 默认要求 NVDEC + TRT +
  cuda-python 且 `ocr_backend≠cpu`（`GPU_PIPELINE=0` 关 / `=1` 强制跳过 TRT
  要求）；raw 可用性由 OCR 会话 `raw_ready` 标志（worker 引擎就绪后置位，
  单 TRT）决定，flush 按 item 分流（不混批）。
- **待实测**：零拷贝路径的端到端/争抢数字尚未在本机重测（需真实 NVDEC+
  TRT 环境），补测后追加 docs/PERFORMANCE.md。
- 未做：hybrid + GPU 管线合并（Phase 4，建议开关保护）；`gray_output`
  旧默认（RGB）已移除——上游应用需迁移为 `rep_crop_format` / `nv12_to_rgb`。

### 相似段合并的分离模式（生产默认 binary）

- `merge_similar` 的代表帧比较在分离图上进行，默认 binary（黑底白字）。
- `TEXT_SEP_MERGE=contrast|binary|off` 可覆盖；contrast 作为实验入口无净收益。
- OCR 输入保持 gray+gamma 2.0；原先“用分离图直接作 OCR 输入”的实验已删除。

### Race 跨编码实测（2026-08 一轮，1500 帧窗口，旧 CPU+ONNX 互补对）

> 以下为一轮（CPU+ONNX 互补对）的历史结论；二轮重构后短窗口由
> `DUAL_PIPELINE_MIN_FRAMES` 门控回退，全片长见上文最终实测（-27~-42%）。

- h264（test3/test5）：默认双流水线 2 片有 7~17% 收益；两条 CPU+TRT 略快。
- HEVC（test）：只有 8 片时接近持平；AV1（test6）：双流水线全部明显变慢
  （+42~87%）——CPU 软解/ONNX 路径成为瓶颈。
- 动态切片会让 GPU+TRT 路径拿到更多片（如 AV1 8 片时 GPU 5 片 / CPU 3 片），
  但“快路径做完后等待慢路径”仍存在，分片数只能缓解不能根治。
- **一轮生产结论（已被二轮取代）：保持默认关闭。**

### 七轮修正：kfe 转正为唯一分片方法（2026-08）

按“kfe 成为双流水线唯一分片方法”的决定，移除全部过时的 dual-2 家族：

- **删除**：`DUAL_PIPELINE_CHUNKS`（默认 2 大竞争片）、构造参数
  `dual_pipeline_chunks`、等分大竞争片分支；`DUAL_KEYFRAME_SLICING`
  （2 片关键帧吸附，`_snap_keyframe_chunks` 一并删除——kfe 自身就把边界
  落在关键帧上）；`DUAL_PROPORTIONAL`（试点测速比例分配，只适用于恰 2 片
  竞争区）；`DUAL_PRIORITY`（在线优先取片）。
- **转正**：`_keyframe_every_chunks` 成为唯一竞争区分片方法（移除
  `DUAL_KEYFRAME_EVERY` 实验开关，始终启用）；`DUAL_KEYFRAME_EVERY_MIN_GAP`
  / `DUAL_KEYFRAME_EVERY_MAX_CHUNKS` 保留为粒度旋钮。切片 = 头部试点×2 +
  确认×2 预留 + 关键帧竞争区；无关键帧/短视频自然退化为单大竞争片。
- **保留**：INFLIGHT 竞争取片闸门、端到端让位（e2e 口径）、
  `DUAL_PIPELINE_INFLIGHT`、`DUAL_PIPELINE_SEEK`、`DUAL_SLOW_RATIO` 覆盖。
- 实现：切片生成收敛进 `FieldExtractor._dual_chunk_specs`（纯函数，可单测）；
  producer 稳态吞吐 / 比例分配 / 优先取片相关状态全部移除，让位判定只依赖
  `e2e_speed`。
- 测试：`tests/test_dual_pipeline.py` 新增 `_dual_chunk_specs` 用例（试点组 +
  关键帧竞争区、无关键帧退化、短视频、stride 吸附、过密合并上限）。
- 结构（2026-08 同轮，按逻辑拆分过长的单文件）：
  - `extractor.py`（2356 行）→ `extractor.py`（引擎核心）+ `_helpers.py`
    （独立工具函数）+ `_result_types.py`（结果 dataclass）+ `_gpu_pipeline.py`
    （`_GpuPipelineMixin`）+ `_dual_pipeline.py`（`_DualPipelineMixin`，
    kfe 唯一分片 + 竞争闸门 + 端到端让位）；
  - `ocr_trt.py`（945 行）→ `ocr_trt.py`（TrtEngine + 构建缓存）+
    `video_ocr_engine/_gpu_kernels.py`（GpuPreprocessor / GpuOutputReducer /
    GpuFrameAnalyzer），`ocr_trt` 顶部 re-export 保持旧导入路径兼容；
  - 公共 API 不变（`FieldExtractor` / `ExtractedSegment` / `ExtractionResult`
    及全部方法名），79 个单测保持通过。
- 死代码清理（同轮）：pyproject `py-modules` 移除已不存在的 `hybrid_decode`
  引用；删除无调用方方法（`_decode_all` / `_segment` / `_dual_pipeline_available`
  / `_start_ocr_session._store_result`）与 extractor 顶层未使用导入（csv /
  `_gray` / `_gray_batch` / `_preprocess_standard`）；删除过时生成产物 build/。

### CPU+NVDEC 混合解码 v3（2026-08，速率比例分界 + 两端连续扫掠）

v1 曾因无净收益被删除（见 docs/PERFORMANCE.md §4）；v2（kfe 共享队列竞争）
2026-08-25 复活但实测退化（见下）；v3 以探针定位 v2 根因后重写为现役实现。

**v2 退化根因（探针实测，HEVC CPU 慢 4.5× 场景）**：
1. FIFO 竞争 + in-flight 令牌使分片在 GPU/CPU 间严格交替领取；消费者按
   全局帧序取帧 → 慢生产者每一片都是关键路径串行等待，快生产者被令牌
   限制无法超前；
2. 交替领取使"连续扫掠免 seek"失效：每生产者除首片外几乎每片 seek
   （GPU ~50-190ms/次、CPU ~35-65ms/次）；
3. 结果：HEVC hybrid decode 2.4-2.8s 反比纯 NVDEC 2.0s 慢 20-40%。

**v3 设计**（`hybrid_decode.py`，2026-08 重写）：
- 采样帧序列仍按关键帧边界切分片（kfe，边界 seek 便宜）；
- `hybrid_begin` 时并行实测两后端顺序解码速率（256 帧 + 16 帧 warmup
  丢弃，双线程），按速率比例把分片切成两段：快端从头连续扫掠前半
  （0 次 seek），慢端 seek 一次到分界片首后连续扫掠后半（1 次 seek）；
  慢端份额夹在 [15%, 45%]，速率比 >1.8x 时只给 1 片试探；
- **对称接管**：快端扫完自己区后从慢端区第一个未开始片逐片接管（一次
  seek 连续扫掠）——校准误差自愈；慢端只做自己区、区空即退出（不反向
  接管快端区，避免破坏快端连续扫掠）；
- **内存上界**：每生产者"已产出未消费"片数 ≤ inflight（默认 2），
  消费者按序排空后才继续产下一片（字幕宽 ROI 防内存暴涨）；
- **对外接口不变**：VideoReader 同形替身（`len` / `get_batch` /
  `next_roi` / `seek_accurate` / `get_*`），正确性依赖 decord fork
  v0.7.8+ 双后端 YUV420 逐位一致。
- **激活条件**（`extractor.py` open 路径，全部满足才生效）：显式
  `decode_backend="hybrid"` 且 NVDEC 实际可用（否则回退 CPU 并警告）、
  `_sample_stride==1`（next_roi 顺序交付语义）、未开 GPU 全驻留管线
  （互斥）。**编码门控已移除**（含 AV1——v3 实测 AV1 不再退化，尊重用户
  显式选择）；`HYBRID_CPU_THREADS`（0=核数//2）、`HYBRID_MAX_CHUNKS`
  （默认 16）可调。初始化失败 try/except 回退纯 GPU 不致命。
- **流程钩子**：采样帧序列就绪后 producer 调 `vr.hybrid_begin(frames)`
  才生成关键帧分片、测速并启动双生产者（先校准后建片）。
- **实测**（7945HX + RTX 4060 Laptop，A/B 单跑；decode 阶段耗时）：

  | 视频 | 编码 | NVDEC | CPU | hybrid v3 | vs NVDEC |
  |---|---|---|---|---|---|
  | test5 6000帧 | h264 | 5.99s | 5.17s | **4.37s** | -27% |
  | test3 3000帧 | h264 | 2.91s | 2.84s | **2.44s** | -16% |
  | test.mp4 3000帧 | hevc | 1.97-2.28s | 4.47s | 2.05-2.22s | 持平 |
  | test2 3000帧 | hevc | 2.10s | 4.41s | **1.77s** | -16% |
  | test6 3000帧 | av1 | 1.86-2.22s | 6.19s | 1.80-1.98s | -10~19% |

  文本一致性：全部 100%（唯一文本集与单路径一致）。诊断开关
  `HYBRID_PROBE=1`（逐片时序）保留。

### CPU+NVDEC 混合解码 v4（2026-08，动态分界 + 稳态折扣 + 短校准）

v3 的短板：**CPU 明显慢于 NVDEC（8 核亲和模拟弱 CPU）时 hybrid 无收益**。
本机实测（7945HX + RTX 4060 Laptop，进程亲和 8 逻辑核，GPU_PIPELINE=0）：
- h264 test5：CPU 754fps vs NVDEC 980fps（CPU 慢 23%）→ v3 hybrid decode
  反而慢（+20%）；HEVC test.mp4：CPU 464fps vs NVDEC 2121fps（CPU 慢 4.6×）
  → v3 hybrid decode 慢（+7%）。

探针定位（并行争抢探针 + 分相 profile，勿再猜）：
1. **NVDEC 与 CPU 软解本身互不拖慢**（并行解码 GPU 仅降 9-16%）；
2. **慢端拖尾**：v3 在 rf>rs*1.8 时只给慢端 1 片试探，收益极小；而按速率
   比例给慢端多片时，短校准高估 CPU 稳态速率（HEVC 软解缓冲衰减：48 帧
   测 495fps、384 帧测 205fps，快测高估 2.2 倍）→ 慢端分到过多片 →
   慢端拖尾、decode 反被拖慢（比例 25% → 慢端 3 片 1.36s > 快端 1.16s）；
3. **OCR 尾批堆积**：hybrid decode 结束更早，OCR（TRT/ONNX）尾批来不及
   在 decode 阶段排空 → ocr_tail 增大（+0.1-0.2s），墙钟被 OCR 吃掉；
4. **校准固定开销**：256 帧校准在弱 CPU 下 ~0.4s（CPU 侧 256/631≈0.4s），
   完全吃掉 decode 收益。

**v4 设计**（`hybrid_decode.py`，2026-08）：
- **短校准**：默认 40 帧 + 8 帧 warmup（`HYBRID_CALIB_FRAMES` 可调），
  弱 CPU 下校准 ~0.1s；
- **稳态折扣**：慢端稳态速率 = 校准速率 × 折扣（慢端=CPU 软解 ×0.45 修正
  缓冲衰减高估、=NVDEC ×0.85；`HYBRID_SLOW_DISCOUNT` 可覆盖）；
- **动态分界**（`_dynamic_split` 纯函数，可单测）：慢端片数从 1 递增，
  只要"慢端生产时间 ≤ 快端生产时间 × 0.95"就继续，超过即停——慢端
  贡献最大化且永不拖尾；两端各至少 1 片；
- **慢端预取**（`HYBRID_SLOW_INFLIGHT`，默认 4 片）：慢端可提前产 4 片，
  消费者到尾段时连续消费，减少 OCR 尾批堆积；
- 其余（连续扫掠 / 对称接管 / inflight / 对外接口 / 激活条件）同 v3。

**实测**（TRT venv，进程亲和 8 逻辑核 = 模拟弱 CPU，交错 A/B 3 轮中位）：

| 场景 | 编码 | NVDEC decode | hybrid v4 decode | Δ | 墙钟 Δ |
|---|---|---|---|---|---|
| test5 3000帧 | h264（CPU 慢 23%） | 2.956s | 2.420s | **-18.1%** | **-2.0%** |
| test.mp4 3000帧 | hevc（CPU 慢 4.6×） | 1.345s | 1.300s | **-3.3%** | +11.4% |

16 核无亲和回归：test5 h264 decode -24.5%、wall -12.7%（与 v3 持平）。
文本一致性：全部 100%。结论：**CPU 明显慢于 NVDEC 时 hybrid 的 decode
确实提升（h264 -18%、HEVC -3%）；h264 墙钟也转正（-2%）；HEVC 墙钟
仍受 OCR 尾批/争抢影响（+11%）——decode 收益 < OCR 固定开销时属物理
限制（CPU 慢 4.6× 时慢端最多 1-2 片，贡献上限 ~5%）。

### 底层重构轮（2026-08，维护性收尾：env 收敛 + next_roi 步长修复 + GPU 异步开关 + 文档归档）

按"维护性修复为主 + 性能收尾"结论落地的零风险改动（A/B 语义不变）：

1. **`HybridDecoder.next_roi` 步长修复**：`_seq_fi = fi + 1` 硬编码漏
   `sample_stride`（现役 hybrid 安全门要求 stride==1 故未触发；放宽安全门
   后校准帧号会错位）。已改为按 `ex._sample_stride` 推进，新增
   `tests/test_hybrid_next_roi.py` 防回归。
2. **env 解析收敛**：`engine_config` 新增 `env_int` / `env_float` 统一解析
   （缺省/空/非法 → default），全部 HYBRID_*/OCR_* 数值解析从调用点迁移：
   `HYBRID_CALIB_FRAMES` / `HYBRID_SLOW_INFLIGHT` / `HYBRID_SLOW_DISCOUNT`
   / `HYBRID_CALIB_ROUNDS` / `HYBRID_MAX_CHUNKS` / `HYBRID_CPU_THREADS` /
   `HYBRID_MAX_CHUNK_FRAMES` / `OCR_THREADS` / `OCR_BATCH` / `OCR_PAD_SMALL`
   / `OCR_GAMMA`。魔法值收敛为 `HYBRID_*_DEFAULT` 常量（0.45/0.85/40/4/1）。
3. **GPU 分段 kernel 同步点可选异步**：`GpuFrameAnalyzer.analyze_batch` /
   `histograms_perframe` / `compare_pair` 增加 `async_mode` 参数（默认
   False = 历史同步语义逐位不变）；`GPU_PIPELINE_ASYNC=1` 开启异步 D2H +
   立即同步的实验路径（kernel 启动与 D2H 重叠）。**默认关闭，纯实验开关**。
   **真机 A/B（test5 3000帧 stride8，3 轮中位）：3.278s（默认）vs 3.281s
   （ASYNC）——严格持平（±0.1%），无净收益；decode 仍为绝对瓶颈（占墙钟
   ~83%）。保持实验态、不转正**（避免后人按 §9 的"同步点串行化 producer"
   推断重复投入；该推断对 GPU+ONNX 路径成立，对 NVDEC+TRT 主路径不成立——
   分段 kernel 时间被 decode 完全掩盖）。
4. **文档归档**：`docs/PERFORMANCE.md` 中 dual_pipeline 全部历史档案迁移到
   `docs/ARCHIVE.md`（§A 混合 OCR/解码旧档案、§B 双流水线全史），正文只留
   指针与现役结论；新增 `tools/trim_perf_doc.py` 维护工具（后续归档章节可
   一键剪切）。
5. **新增单测**：`tests/test_hybrid_next_roi.py`（stride 1/2/3 推进、起始帧
   取 starts[0]）。

> 真机验证已完成（test5 3000帧 stride8，3 轮中位）：`GPU_PIPELINE_ASYNC=1`
> 与默认严格持平（3.281 vs 3.278s，±0.1%）——无净收益，保持实验态不转正
> （见第 3 条）。hybrid 转正的工程化收尾仍待做（现役为显式
> `decode_backend="hybrid"` + 7 个 HYBRID_* env 旋钮，需决策默认值与
> 长视频回归后再考虑默认启用）。
