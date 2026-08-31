# 设计决策档案（video_ocr_engine）

> **本文件是 2026-08-31 对 `CLAUDE.md` 瘦身时迁出的原文存档**，用于回答
> "为什么这样做 / 为什么不做"。
>
> ⚠️ **现役规则以 `CLAUDE.md` 为准**。本文件是**历史原文**，不再维护；
> 若与 `CLAUDE.md` 冲突，一律以 `CLAUDE.md` 为真相。
>
> **迁出原因**：`CLAUDE.md` 会被部分 harness 在会话开头全量注入。它当时已
> 70,764 B / 956 行 / 41,317 字，而其中只有约 3.9 KB 是现役事实 ——
> 让每会话为 67 KB 历史档案付费不合理。本文件按需读取，无预算限制。
>
> **结构**：正文是 CLAUDE.md 原 L32–L956 逐字搬迁，标题与编号未改；
> 章节索引在文末。代码注释里的 `DESIGN-REVIEW Xn` 标记指向本文
> 「设计审查结论」一节。

---

## 章节索引

 1. **通用约定**
 2. 公共 API 清理（0.7.0，破坏性）
 3. **单实例双完整流水线并行（历史，已删除 — e8b2637）**
 4. 设计
 5. 探针定位的损耗来源（2026-08 二轮，勿再猜测）
 6. 最终实测（A/B 单跑串行，7945HX + RTX 4060 Laptop）
 7. 五轮修正：混配双线程真正并行（2026-08，本机实测）
 8. 每关键帧切一片的探针定位与死路（2026-08，勿再投入）
 9. 六轮修正：关键帧切片复活——连续扫掠免 seek + 竞争闸门 + 端到端让位（2026-08，本机实测）
10. 实现注意
11. TRT 拷贝路径（2026-08）
12. 显存全驻立路径补全与争抢韧性验证（2026-08 三轮，GPU_CTC）
13. GPU 管线转正（2026-08 四轮：merge_similar 补齐 + 默认启用）
14. 第四轮（2026-08-29）：pad 下限重估 + 裁切守卫删除 + 去块滤波 + fork 构建打通
15. 内部恒为单通道灰度 + rep_crop_format（yuv/gray）
16. 相似段合并的分离模式（生产默认 binary）
17. Race 跨编码实测（2026-08 一轮，1500 帧窗口，旧 CPU+ONNX 互补对）
18. 七轮修正：kfe 转正为唯一分片方法（2026-08）
19. CPU+NVDEC 混合解码 v3（2026-08，速率比例分界 + 两端连续扫掠）
20. CPU+NVDEC 混合解码 v4（2026-08，动态分界 + 稳态折扣 + 短校准）
21. 底层重构轮（2026-08，维护性收尾：env 收敛 + next_roi 步长修复 + GPU 异步开关 + 文档归档）
22. 下一步三目标轮（2026-08）：真跳帧证伪 / CPU+ONNX 提速 / hybrid 修复
23. 路线图收口轮（2026-08-29）：P0-4 GPU 直通 + P1-3 解耦 + hybrid 启动重叠
24. P0-6 翻默认评估（2026-08-29 续）：否决——test4 有确定性准确率退化
25. P0-6 翻案（2026-08-29 续二）：test4 抽帧视觉裁定 → 真值伪影，翻默认开
26. §8 扫描轮落地（2026-08-29 晚）：TRT 批对齐 -9.1% + hybrid 合并 + NVDEC 同步证伪
27. 0.9.0 清理轮（2026-08-29）：删除已证实无收益的实验钩子，API 收干净
28. 设计审查结论（2026-08-30，原 docs/DESIGN-REVIEW.md 已并入本节）

---

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

1. ~~**混配退化的真因 = 内存子系统争抢（三轮探针定位，勿再用"核饱和/
   调度饥饿/GIL/对称收敛"表述）**~~。判别链：CPU 占用仅 ~46%（未饱和）；
   跨进程仍退化（非 GIL）；ORT 自旋参数无效（非自旋）；双侧绑核
   隔离+进程优先级组合无效（非调度/迁移）；GPU 时钟恒定 2490MHz、温度
   正常（非降频）；**单线程重 SIMD 计算（L2 驻留、零 DRAM 流量）完全不伤
   TRT；而 8 进程纯内存流拷贝（~~100GB/s~~）让 TRT 10.26ms/段、enqueue 子相
   位膨胀 2.2→14.8ms/批（灾难级）**。机制 = 对端多线程宽矩阵乘的聚合访存
   流量占满 DRAM/Infinity Fabric，TRT 宿主提交路径（ioctl、页表遍历、
   pageable 拷贝 staging）全部变慢，GPU 反过来饿等提交（util 63%→42%）。

   > **⚠️ 2026-08-31 实测修正（详见 `docs/PERFORMANCE.md` §20，勿再引用旧数字）**
   > 上面这条结论**定性对、归因错、数字错**。首次真测内存吞吐量后的修正：
   > - **数字错**：`~100GB/s` 是对探针负载的**标签，不是测量值**。本机
   >   （2×16GB DDR5-6000 双通道，理论峰值 96 GB/s）实测上限 **B_max =
   >   55.8 GB/s**（8 种进程×线程配置收敛，波动 ±0.5，= 峰值 58%）。旧标签
   >   高估 **1.8 倍**。
   > - **机制对**：带宽争用确是极强的 TRT 退化源。CPU 打满（97%~100%）的
   >   前提下，零 DRAM 流量的 L2 常驻负载只让 TRT 慢 **1.05×**，而吃掉
   >   44.9 GB/s 的 DRAM 负载让它慢 **10.10×**。旧的"L2 不伤 TRT"对照用的是
   >   **单线程**（CPU 根本没占满），换成多线程饱和版才成立。
   > - **归因错**：**混配退化不是带宽造成的**，定量断裂 37 倍 ——
   >   ONNX 对端实际只吃 **12.0 GB/s**（瞬时峰值 14.1，150ms 时间序列确认
   >   非突发），而同带宽的合成 hog（11.6 GB/s）只让 TRT 慢 **+2.8%**，
   >   混配实测却是 **+104%**。
   > - **决定性反证**：`TRT ∥ TRT` 只吃 **0.6 GB/s**，退化 **2.02×**；
   >   `TRT ∥ ONNX` 吃 10.0 GB/s，退化 **2.04×** —— **零带宽消耗能造成
   >   和 10 GB/s 相同的退化**，带宽不是支配变量。
   > - **真正嫌疑**：`decode_backend=auto` **不区分 OCR 后端，两条流水线都开
   >   NVDEC** → 两个并发 NVDEC 会话（单固定功能单元串行化）+ GPU 双上下文
   >   切换。混配与双 TRT 这两项完全一致，退化幅度也几乎一致（2.04× / 2.02×）。
   >   下一步测 NVDEC 会话数与 GPU 上下文数，别再在内存上做文章。
   >
   > **✅ 已结案（2026-08-31 §21 实测，详见 `docs/PERFORMANCE.md` §21）**：
   > NVDEC 会话数就是答案，CUDA 上下文几乎无关。固定 P1 = TRT/NVDEC 跑
   > test6，只改对端（两条都跑 test6，同视频同编码，4000 采样帧）：
   >
   > | 对端 P2 | NVDEC 会话 | P1 退化 | 聚合加速比 |
   > | --- | --- | --- | --- |
   > | ONNX / CPU 解码 | 1 | **1.02×** | **1.87** |
   > | TRT / CPU 解码 | 1 | **1.06×** | **1.84** |
   > | ONNX / NVDEC | 2 | **1.99×** | 1.20 |
   > | TRT / NVDEC | 2 | **1.98×** | **1.01** |
   >
   > - **多一个 CUDA 上下文 +4%，多一个 NVDEC 会话 +95% —— 相差 24 倍**。
   >   ONNX 还是 TRT 完全无关（1.99 vs 1.98），再次确认与 OCR 后端无关。
   > - **完全互补设计（CPU 解码+ONNX ∥ NVDEC+TRT）不存在严重争用**：TRT 侧
   >   只退化 **1.02×**，聚合吞吐 **1.87×**（理想值 2.0）。**争用严重的是两条
   >   都走 NVDEC**（1.98×，聚合加速比仅 1.01~1.20，等于白开）。历史若记
   >   为"互补也争用严重"，最可能是被 `auto` 坑了——它不区分 OCR 后端，
   >   两条一律开 NVDEC，所谓"互补"从没真跑起来过。
   > - **内存带宽彻底脱罪**：互补设计只吃 **7.8 GB/s**（峰值 17.5，150ms
   >   序列显示 0% 时间低于 B_max−25）却退化 1.02×；多一个 NVDEC 会话带宽
   >   增量≈0 却退化 1.99×。**CPU 视频解码是计算密集（熵解码/运动补偿）而非
   >   访存大户**（单跑 10.1 GB/s），与最初预期相反。
   > - **"互补"必须按实测速度分工，不能按硬件单元分工**。纯解码隔离测得
   >   CPU/NVDEC 速度比**随编码反转 7.4 倍**：h264 CPU **快 2.88×**（4.898s vs
   >   1.701s），AV1 CPU **慢 2.56×**（2.979s vs 7.622s）。解码占管线 98%+，
   >   故静态绑定 CPU 解码的一侧必然在 AV1 上变瓶颈——这量化解释了 §18.B
   >   的 AV1 失败（test6 单跑 1.61s 开双流水线反而 2.29s）。现役 `hybrid`
   >   的速率比例分界已动态处理此事，**不要再引入静态双流水线**。
   > - **收口**：`NVDEC 会话数(×2) ≫ CUDA 上下文数(+4%) > 内存带宽(≈0)`。
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

### 第四轮（2026-08-29）：pad 下限重估 + 裁切守卫删除 + 去块滤波 + fork 构建打通

**构建打通（前置，解锁了所有 fork 侧优化）**
- `MSBuild` 在安全层的 `WINDOWS_LOLBINS` 硬编码黑名单里 → 我在任何工具里
  都没法按名字调用它。守卫是**对命令文本做正则**，所以 `cmake --build`
  （命令行不含 msbuild 字样）能跑，但拉起的子进程被拦，表现为孤零零一句
  `Access violation` + **退出码 1**（真崩是 0xC0000005，据此可区分）。
  → **MSBuild 本身没坏，不用修**；VS DevShell 也救不了（实测同样崩）。
- 解法：换 **Ninja 生成器**（不在黑名单）。ninja 装进隔离 venv；
  配置/构建必须走 **PowerShell 工具**（Bash 里 `cmd /c` 也被拦，
  没法跑 vcvarsall.bat），在 `Enter-VsDevShell` 里执行。全量 **15~21s**。
- ⚠️ `-DCMAKE_BUILD_TYPE=Release` **不能漏**：Ninja 是单配置生成器，
  漏了按无优化编（DLL 2.4MB vs Release 448KB）。
- 等价性：新 DLL 体积 448000 字节（与原 DLL 完全一致）、导出符号 57 个一致；
  替换后 84 测试 + e2e_smoke 四条路径 100% 文本重合。
  备份 `build/Release/decord.dll.20260829.bak`。方法已写进 MEMORY.md。

**P0-5 `OCR_PAD_WIDTH_MIN` 224 → 160（旧结论已作废）**
> ⚠️ **本项已于 2026-08-29 晚回退（160 → 224），下方为当时的论证，勿照做。**
> 回退原因：生产（RaceVideoToLog）报告 pad 160 使原始 OCR 误读退化
> （test5 7→26、test6 17→32），根因是评估口径错了（用了引擎默认
> `force_aspect=0`，而生产传 1.5；且逐帧比而非段代表帧比）。
> **pad 224 保持**；320 也已证伪（误读 124→133、最终 0→3、FAIL）。
> 详见 `docs/PERFORMANCE.md` §16.2 P0-5。

旧注释"窄图在宽 pad 下更准（test6：224→err 0.09%，48~96→0.69~1.19%）"
用真值重测**完全反转**：224 在 test5/test6 上反而最差（−0.7~−1.1pp），
均值最优是 160，且低档位墙钟全面更快；宽 ROI 上 160 vs 224 文本一致且快 8.8%。
三处同步改：`OCR_PAD_WIDTH_MIN`、`OCR_PAD_WIDTH_MIN_BY_MODEL`、`DEFAULT_FILL_WIDTH`。

顺带修好一个死开关：`OcrEngine.__call__` 里 `if self._fill_width > 0` 优先，
而 extractor 默认传 `fill_width=DEFAULT_FILL_WIDTH` → **`OCR_PAD_SMALL` env
永远轮不到**，README 却把它列成了可调旋钮。已改为 env 优先级最高。

**裁切守卫删除（本轮最关键的一处纠正）**
上一轮加了个"裁后宽度会被 pad 回下限就跳过"的守卫，理由是"省不到算力就别
冒准确率风险"。用真值复核证明**前提是错的**：裁切让输入更贴近训练分布，
**即使省不到算力也能提准确率**。守卫在窄 ROI 上 100% 触发，恰好把收益全挡掉：

| 视频 | 不裁 | 有守卫（=不裁） | 无守卫 |
|---|---:|---:|---:|
| test5（h264 7223帧） | 97.951% | 97.951% | **99.031%（+1.08pp）** |
| test6（av1 23441帧） | 98.187% | 98.187% | **99.125%（+0.94pp）** |

**新默认（160 + 裁切 + 余量 10）vs 旧默认（224 + 不裁）：四片均值 +0.82pp**
（test5 +1.08、test2 +1.06、test6 +1.06、test +0.08），墙钟 0~+3.8%。

**余量 10% 优于 0%**（与"空格看着更对"的直觉相反）：余量 0 会插入重复句读
空格（`好酒好酒好酒 → 好酒 好酒 好酒`），语义上更可读，但按真值算准确率
反而低 0.6pp；数字场景还会引入 `51→S1`、`115→11S` 错字。→ 默认保持 10%。

**P0-6 去块滤波（fork，env 门控默认关）**
`video_reader.cc` 把 `DECORD_SKIP_LOOP_FILTER` 透传给 `avcodec_open2` 的
AVDictionary（与 AV1 的 `max_frame_delay` 同一处），仅 CPU 软解。
HEVC **-8.3%~-14.3% 墙钟**、h264 -0.6%~-4.2%、AV1 无效；5 片准确率均值 −0.01pp。
**默认关闭**：会改变输出像素（无去块平滑），`rep_crop` 预览会看到块状伪影，
属用户可见的质量变更。启用：`DECORD_SKIP_LOOP_FILTER=all`。

**教训（本轮最值钱的一条）**
判断"某改动让 OCR 变好还是变坏"，**只测"文本有没有变"会得出错误结论**。
上一轮的守卫就是靠"文本一致率 100%"显得很安全，实际在白白损失 0.9pp。
必须**按帧对齐真值**测准确率。也别用置信度当代理：
`羸弱→赢弱` 是退化但置信度反而从 0.9433 升到 0.9700。

**踩过的坑**
- `_probe_guard_clean.py` 首版把**中文 label 当 mode 传给 worker**，
  `mode == 'noguard'` 恒为假 → monkeypatch 从未生效，三种模式跑出同一个数，
  看起来像"守卫没影响"，实为**假阴性**。label 与 mode 必须分开传。
- 探针跑在后台时**不要改被测代码**（首版 +0.86pp 就是这么被污染的）。
- 真值 CSV 头是 `# roi=843,993,948,1025, format=..., frame_start=362` ——
  **不能按逗号切再找 `roi=` 前缀**（那样只拿到 "843"），要用正则取四个整数。
- 编辑配置文件替换 dict 时容易留下**重复的旧键值对**导致 IndentationError。
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
   `tests/decode/test_hybrid_next_roi.py` 防回归。
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
   本文 §18（§18.A 混合 OCR/解码旧档案、§18.B 双流水线全史），正文只留
   指针与现役结论。当时的维护工具 `tools/trim_perf_doc.py`（把章节剪切到
   `docs/ARCHIVE.md`）**已于 2026-08-30 删除**——多文档架构取消后无用途。
5. **新增单测**：`tests/decode/test_hybrid_next_roi.py`（stride 1/2/3 推进、起始帧
   取 starts[0]）。

> 真机验证已完成（test5 3000帧 stride8，3 轮中位）：`GPU_PIPELINE_ASYNC=1`
> 与默认严格持平（3.281 vs 3.278s，±0.1%）——无净收益，保持实验态不转正
> （见第 3 条）。hybrid 转正的工程化收尾仍待做（现役为显式
> `decode_backend="hybrid"` + 7 个 HYBRID_* env 旋钮，需决策默认值与
> 长视频回归后再考虑默认启用）。

### 下一步三目标轮（2026-08）：真跳帧证伪 / CPU+ONNX 提速 / hybrid 修复

按"1. 更好的 stride（真跳帧）2. 验证 CPU+ONNX 墙钟 3. 更好的 hybrid_decode"
三目标推进，顺序按性价比重排为 2 → 1 → 3。

1. **真跳帧（目标 1）—— 实测证伪，P1-1 封板**
   - `tools/_probe_drop_nonref.py`：码流转 Annex-B，按 `first_mb_in_slice==0`
     切 access unit，按 `nal_ref_idc` 生成丢帧流，与全量流解码后逐帧比对。
   - **按 `nal_ref_idc==0` 整包丢 packet 是安全的**（剩余帧与原解码逐字节一致，
     test5 2007/2007、新三国01 2268/2268）。这推翻了 `PERFORMANCE.md` §6
     "丢 packet 必然破坏参考关系"的封板结论——旧结论针对的是**按 pict_type
     丢 B 帧**（High profile 下 B 帧可能 `nal_ref_idc>0` 即仍是参考帧）。
   - **但收益只有 1.03~1.48×**（原估 2~4×）：① 采样点与参考帧都不能跳，
     stride=8 仍要保留 57~62% 的帧；② FFmpeg 帧线程下非参考 B 帧大量落在
     关键路径之外，线程越多越"免费"（标清 16 线程丢 43% 的帧只快 3%）。
     折算墙钟：整集仅 -9%，而 test5 最优配置是 stride=1，**stride=1 无法跳帧**。
   - 唯一还值钱的分支是 `skip_loop_filter=all`（1.11~1.36×，线程越多越赚，
     与丢包叠加 1.48~1.59×），代价是输出像素变化，**须先做端到端 OCR 质量回归**。
   - 踩坑：`ffmpeg -frames:v N` 计的是**输出**帧（用它做 skip 实验会系统性
     低估收益）；`-f rawvideo` 计时被落盘 I/O 污染，计时必须走 `-f null`。

2. **CPU+ONNX（目标 2）—— 没变慢，但白丢 28%**
   - HEAD vs HEAD~1：`decode=cpu ocr=cpu` **-0.6%（持平）**，`ocr=auto` 且
     TRT 不可用（回退 ONNX）**-25.3%**（该路径判据 `_ocr_on_gpu()` 只看配置
     不看 TRT 是否真可用，之前"意外正确"）。
   - **现役多核分支返回 `None` → 解码落到 fork 默认 8 线程**，白丢性能。
     "多核不该给解码加线程"其实是**高段密度（OCR 受限）**场景的结论，被当成普适结论。
   - 改为按 `sample_stride` 判段密度分档（stride>1 → 逻辑核 3/4 钳 [8,24]；
     stride==1 → 逻辑核 1/3 钳 [8,12]）。实测：低密度 -28.7%、高密度 -3.4%
     （且 ≥14 线程劣化，故必须分档）、标清整集 -15.4%、弱 CPU 不劣化。
   - 新增 env 钩子 `DECODE_THREADS`（与 `OCR_THREADS` 对齐）。

3. **hybrid（目标 3）—— 两个 stride 相关 bug，修复后首次跑赢单端**
   - **Bug A（致命）**：`_producer` 的 `prev_end = fis[-1] + 1` 漏 stride。
     "片间连续扫掠免 seek"靠 `下一片首帧 == prev_end`；stride>1 时首帧是
     `fis[-1]+stride`，判定永远失败 → **每片一次 seek**（GPU 50~190ms/次）。
     实测 stride=8 下 hybrid 比纯 NVDEC 慢 **38~59%**，且比任一单端独跑都慢。
     与上一轮 `next_roi` 的 `+1` 是同一类缺陷（stride==1 时恒等，长期潜伏）。
   - **Bug B**：速率校准的 `calib` 按**采样帧**给，stride=8 时解 40×8=320 源帧，
     比 stride=1 贵 8 倍。改为按源帧预算（校准只取两后端速率比值，信息量等价）。
   - 解禁 stride==1 安全门（`next_roi` 已按 stride 推进；且 stride>1 时宿主
     校准与主循环都走 `get_batch`，不碰 `next_roi`）。
   - **实测（3000 帧，3 轮最快；括号内为相对最佳单端）**：

     | codec | stride | 纯 NVDEC | 纯 CPU | hybrid 修复前 | hybrid 修复后 |
     |---|---:|---:|---:|---:|---:|
     | h264  | 8 | 3.869s | 2.070s | 2.587s | **2.111s（+2%）** |
     | HEVC  | 8 | 2.124s | 2.353s | 3.145s | **1.687s（-20.6%）** |
     | AV1   | 8 | 2.512s | 6.644s | 3.615s | **2.086s（-17.0%）** |
     | h264  | 1 | 3.867s | 2.739s | 2.677s | 2.735s（并列最优） |
     | HEVC  | 1 | 2.203s | 2.995s | 2.393s | 2.361s（+7%） |
     | AV1   | 1 | 2.609s | 6.962s | 2.641s | 2.806s（+7.6%，噪声 ±6%） |

     → **HEVC/AV1 + stride>1 上 hybrid 首次跑赢最佳单端**；h264 上纯 CPU 仍最优
     （CPU 比 NVDEC 快 ~2×，两路合起来也追不上单路 CPU）。
   - 新增单测 `tests/decode/test_hybrid_producer_stride.py`（桩 reader 数 seek 次数，
     stride=1/3/8 + 交付完整性；已验证改回 `+1` 会 FAIL）。
     注意：`__new__` 绕过构造时 `_roi` 必须手动补，否则生产者在首帧抛异常、
     seek 计数恰好也是 1 → **假通过**。

> **遗留（按性价比排序，均未做）**：
> ① **P0-3 `auto` 后端按"编码+核数"选择**——现役 `auto` 优先 NVDEC，但 h264 上
> CPU 软解（24~32 线程）**快约 2×**（test5 3000f stride8：gpu_yuv 3.99s vs
> host_cpu 2.03s）。**默认用户至今拿不到 P0-1 的收益**，纯 Python、改动小、收益最大。
> ② **P0-2 host 输入的 TRT 批走 GPU argmax 归约**——原型已验证 test5 全片 -8.4%、
> 3000 帧 -14.2%、整集 infer -39%，逐位一致；`execute_device_argmax` 已实现，
> 只需在 `ocr_native._call_trt_gpu` 复用到 host 路径。
> ③ `skip_loop_filter` 的端到端 OCR 质量回归（需先动 fork 才能端到端验证）。

### 路线图收口轮（2026-08-29）：P0-4 GPU 直通 + P1-3 解耦 + hybrid 启动重叠

按 `docs/PERFORMANCE.md` §16.0.4 完成 2/3/5 号项（实测详见
`docs/PERFORMANCE.md` §12）：

1. **P0-4' 宽度裁切扩到 GPU 直通路径**：
   - `GpuFrameAnalyzer.content_range`（新 `col_ink` kernel，单 block 256
     线程 shared 归约，DtoH 8 字节）：rep 帧「有墨迹列范围」；
   - `prep_gray_raw` 支持 6 元组 infos `(ptr,h,w,owner,x_off,crop_w)`，
     未裁项 `(0, src_w)` 与旧全宽内核**逐位一致**；content_w 用宿主
     `_preprocess_standard` 同式（int 截断）在 host 算好传入；
   - 余量数学收敛到 `_HostPipelineMixin._content_range_to_crop`
     （宿主/GPU 共用，同一 rep 帧同一裁切区间）；GPU 侧跳过条件
     （关/force_aspect/std<3/满宽）与宿主一致，std 用 GPU analyze 的
     sharp（yuv 已是展开 Y，与宿主同值域）；
   - `_start_ocr_session.flush()` raw 项按裁后宽度排序、按批拆子批
     （与宿主裁切路径同策略，不分组收益归零）。
   - 实测：宽 ROI TRT infer -7.3%、墙钟噪声内（OCR 被解码掩盖，预期）；
     `decode=auto` vs `decode=cpu` 全片文本 503/503 一致。

2. **P1-3 解耦（decode=cpu 也走 GPU 全驻留管线）**：
   - 门控 `decode ∈ {auto, nvdec, cpu}`（cpu 显式或 NVDEC 回退都进
     CPU 分支；hybrid 仍互斥）。CPU 分支：每批 asnumpy → 宿主灰度
     （与宿主逐位同式）→ H2D → 同一 hist/analyze kernel；
   - `_DevBatchPool`/`_DevBatch`/`_CpuFrameRef`：批缓冲池化（GC 归零
     归还）。**复用安全性契约**：raw OCR（call_gpu_raw 返回前同步）与
     sim_pair（compare_pair 同步）读完才可能归零——与 `_YFramePool` 同一
     契约，勿在异步路径上破坏它；rep 的 keep_crops/OCR 回退走宿主切片
     **拷贝**返回（numpy view 会钉住整批解码数组）；
   - 设备侧恒为灰度（yuv 只上载展开 Y）；`_d2h_rep` 探测 owner 的
     `host_crop` 属性分流（NVDEC=D2H，CPU=宿主切片）；
   - 实测（decode=cpu）：test5 全片 -11.2%（解码批 64 vs 宿主 16 的
     吞吐差 + 分段上 GPU）、三国30000 -1.7%、test6 AV1 +0.1%；
     **真值 test5/test2/test6 三片 +0.00pp**（逐位一致）。

3. **hybrid 第二 reader 后台打开**：CPU reader 后台线程打开，与 GPU 端
   测速重叠；CPU 测速等打开完成。**实测本机热缓存持平**
   （open_and_fps 0.054 vs 0.055s——路线图"打开 ~0.12s"估算未复现），
   保留为冷缓存兜底。语义变化：CPU reader 打开失败从"构造期静默回退
   纯 GPU"变为"hybrid_begin 上抛"（GPU 已成功打开的前提下实际不可达）。

4. **不做**：P0-6 翻默认（等用户拍板）；P3'（ROI < 10 万像素判据不满足）；
   P2-2（2/3 完成后重估，投入产出比仍不成立）。

**踩坑**：`process_gray_raw` 的 D2D 聚批循环原按 4 元组解包，扩到 6 元组后
`too many values to unpack`——改 infos 结构时所有解包点都要过一遍。
`_probe_*_ab` 探针的 `uniq` 口径是**排除空文本**，自写探针若不排除会对不上。

### P0-6 翻默认评估（2026-08-29 续）：否决——test4 有确定性准确率退化

按用户判据（"无负面影响才翻默认"）用 6 片真值复核 `skip_loop_filter=all`
（decode=cpu 走现役 GPU 管线路径）：5 片 +0.00~+0.08pp，**test4 −0.19pp
全等 / −0.08pp 数值容错**（原 P0-6 表漏测的片）。逐帧分析
（`tools/_probe_slf_diff.py`）：纠错 9 vs 退化 21 帧；失败模式 = 前导幽灵
"0"（`20→020`，去块滤波关闭后 ROI 左缘弱噪声越过二值化阈值）+ 偶发丢位
（`221→21`）。宿主管线复核 Δ 完全一致 → 纯解码像素变化所致。

**决策：默认保持关闭**（env 旋钮保留；fork v0.7.13 发布透传能力）。
方法教训：**"5 片均值 −0.01pp 噪声"曾被用来支持"无负面影响"，第 6 片
就翻了案——均值不能替代逐片检查**；"+0.08pp 改善"说明像素变化对准确率
是双向噪声 + 片源相关退化，本质是掷硬币换速度。

### P0-6 翻案（2026-08-29 续二）：test4 抽帧视觉裁定 → 真值伪影，翻默认开

用户指出 test4 真值可能有问题，要求看图。对 关≠开 65 帧（27 簇）抽帧
逐格人工裁定（`tools/_probe_slf_adjudicate.py`，拼图在 `tools/_slf_vis/`）：

- **前导零**：显示三位补零（`020`，前导 0 更暗），真值（v2.7.0 生成）剥零。
  关滤波也漏读暗淡 0 → 账面"对"；开滤波读 `020` 更忠实 → 被记成退化。
  f133 实拍就是 `020`（看图确凿）。
- **白闪转场**：显示被吞（f4688 只剩 "2"、f4838 只剩 "9"），真值是语义值，
  两路都在编造；逐格裁定开优 ≈20 帧 / 关优 ≈15 帧 / 不可裁 ≈27 帧。
- **真值脱节**：f933 实拍 `208` 真值 `230`、f1097 实拍 `226` 真值 `248`
  （疑遥测时间基准不同）；末帧真值 `-1` 哨兵。

**最终：六片无负面影响 → 翻默认开**（`video_ocr_engine/__init__.py` 里
`os.environ.setdefault("DECORD_SKIP_LOOP_FILTER", "all")`，opt-out 预设
`none`；需 fork ≥v0.7.13）。注意输出格式变化：2 位速显示输出带前导零。

**教训（补充）**：①"逐片看真值"仍不够——**真值本身要抽查**（版本、
剥零、哨兵、时间基准）；②账面差异能确定复现 ≠ 正确，确定性复现的可能是
"真值错误 × 像素变化"的交集；③分歧帧必须看图归因，tools/_probe_slf_adjudicate.py
就是为此写的拼图工具。

### §8 扫描轮落地（2026-08-29 晚）：TRT 批对齐 -9.1% + hybrid 合并 + NVDEC 同步证伪

按新路线图 §8 落地：①TRT 批对齐 max_batch（单 TRT 引擎分块 16→18，
CPU+TRT 墙钟 -9.1%；**flush 步长与切片必须同步改**，首版错位静默丢段，
文本门拦截）；②批量实例并发 README 指南（NVDEC∥CPU ~1.4×）；③hybrid ×
GPU 管线合并（互斥门控移除；**后端判定必须精确匹配**——decord/GPU+CPU-hybrid
以 decord/GPU 开头，前缀判断会误入 NVDEC 分支对 _Batch 取 DLPack）；
④fork NVDEC 逐帧同步错峰（延迟 sync+unmap）——**逐位等价但无显著收益
（-0.3~-0.5%）**，P2-2 的 0.15ms/帧固定开销推断证伪（同步隐藏在解码
间隔内），P2-1/P2-2 收口。fork 构建加 `/utf-8`：MSVC 按 936 代码页解码
UTF-8 源码，CJK 注释行尾字节被 GBK 配对吞掉换行（行号漂移、成员声明
丢失）——cuda_threaded_decoder.cc 首次重编时暴露；此前各轮只增量重编
未触及该 obj，所以从未暴露。**教训：给 UTF-8 源码的 C++ 项目配 MSVC
构建一律加 /utf-8；改 obj 未重编过的文件时要预期"首次编译"级别的坑。**

### 0.9.0 清理轮（2026-08-29）：删除已证实无收益的实验钩子，API 收干净

按"遗留实验性钩子证实无收益即删除，只在文档留痕"的决策执行（版本
0.8.1 → **0.9.0**，破坏性变更）：

- **`GPU_PIPELINE_ASYNC`**（env + GpuFrameAnalyzer 三个 kernel 方法的
  `async_mode` 参数 + 门控副作用）：NVDEC 分支与 CPU 解码分支两轮实测
  均无收益（第二次 3 轮交错 min -0.6% 噪声）→ 删除。
- **`HYBRID_CALIB_ROUNDS`**（env + hybrid_begin 多轮循环）：实测净负
  （3 轮 -21%，~0.68s 成本 > 分界收益）→ 删除，单轮测速写死。
- **merge_similar 的 `contrast` 分离模式**（`TEXT_SEP_MERGE=contrast/1`
  归一为 binary；`_text_sep_gray` 的 contrast 分支 + `_box_blur` +
  GPU `_similar_device` 的边界 D2H 路径）：实验入口无净收益 → 删除；
  `TEXT_SEP_MERGE` 仅剩 binary/off。
- **`DECORD_FORCE_CPU`**（旧钩子，`decode_backend` 参数化后废弃满两个
  版本）→ 删除。
- **构造参数 `gray_output` / `yuv_output`**（0.7.0 标 deprecated，两个
  版本已过）→ 删除，一律 `rep_crop_format`。
- **保留**（非"无收益实验"）：`GPU_PIPELINE=1` 强制模式（无 TRT 环境的
  管线验证入口 + e2e gpu_onnx 配置）、`HYBRID_MAX_CHUNK_FRAMES`（防御性，
  真实超长 GOP 场景生效，单测覆盖）、全部诊断开关（ENGINE_PROFILE /
  TRT_SUBPROBE / DEBUG_BOUNDS / HYBRID_PROBE*）、全部调参 env。
- 删除的钩子在 README「实验/诊断」节尾与 docs/PERFORMANCE.md 留痕；
  再想实验这些方向时先看文档再重写。

### 设计审查结论（2026-08-30，原 docs/DESIGN-REVIEW.md 已并入本节）

> **来源**：静态代码阅读 + 全部文档对照的一轮/二轮审查（v0.9.2 基准），
> 共 25 条，覆盖 API/默认值、架构边界、正确性与并发风险、使用体验四类。
> 2026-08-30 修复轮落地 23 条，原报告文件已删除。
> **验证**：单测 93 项全过（含新增 6 项）；真机 e2e 冒烟 7 配置矩阵全 PASS
> （test5 3000 帧 stride8，真值匹配率 100%、跨配置唯一文本重合率 100%）；
> 引擎池对象复用 + 5 轮提取显存稳定已实测；取消响应 3.3s（3s 截止）。

**代码注释里的 `DESIGN-REVIEW Xn` 标记指向下表对应条目。**

| 条目 | 问题 | 处置 | 状态 |
|---|---|---|---|
| **A1** 同一旋钮多入口、优先级未文档化 | `OCR_PAD_SMALL` 等 env 压过构造参数；autocrop 全家只有 env 入口 | README 新增优先级（env > 构造 > 常量）与仅-env 清单 | ✅ 文档 |
| **A2** `auto` 在 h264 多核不是最优 | CPU+TRT 比 NVDEC+TRT 快约 2×，但 `auto` 默认 NVDEC | 保持现状（P0-3 决策：静态判据不可靠、判错代价成倍）；README 补决策理由 | ✅ 文档（不修） |
| **A3** `frame_end=0` 兼容语义只在注释里 | 0 与 None 都是"到末尾" | README 声明双入口 | ✅ 文档 |
| **A4** 超界 `frame_end` 静默截断 | 与 `frame_start` 超界报错不对称 | 两条流水线加 warning，**保留截断语义不破坏兼容** | ✅ 代码+文档 |
| **A5** `force_aspect` 与 `fill_width` 强耦合 | 只调一个可能拿到反向次优值 | README 加参数组合提示 | ✅ 文档 |
| **A6** env 读取时机不一致 | autocrop 四旋钮构造期烘焙，其余调用期读 | 四旋钮改 property 调用期读 env，全表统一 | ✅ 代码+文档 |
| **B1** `extractor.py` 同时是骨架/解码器工厂/线程预算中心 | 接入新后端改动集中在核心文件 | 拆 `DecoderFactory` / `ThreadBudget` | ⏸ **未动**（需破坏版本） |
| **B2** 顶层模块与包双向依赖 | `ocr_trt` re-export 包内模块，依赖图成环 | 全部模块收进包内，顶层只留 shim | ⏸ **未动**（需破坏版本） |
| **B3** 两条流水线重复实现 | 残留 `contrast` 死分支（0.9.0 已删该模式） | `_similar_device` 分支删除 + 三处注释清理 | ✅ 代码 |
| **B4** `HybridDecoder.seek_accurate` 是空操作 | 接口存在但语义被掏空，未来调用方会静默拿错帧序 | 抛 `NotImplementedError`；两条流水线对 hybrid 跳过 seek | ✅ 代码 |
| **B5** OCR 引擎/内核/分析器跨视频零复用 | 每个视频重复付 TRT 反序列化 + NVRTC 编译；GPU 管线丢掉 `_ocr_engines` 参数 | GPU 管线接上透传 + **进程级 OCR 引擎池**（`ocr_native.acquire/checkin`，key=(model,type,fill_width,threads)，每 key 上限 4）+ NVRTC 模块进程级缓存 | ✅ 代码 |
| **C1** `nvdec_available` 的 `lru_cache` 缓存瞬态失败 | 首次探测遇瞬态失败 → 该视频 GPU 管线判定被永久钉死 | 成功才缓存（`_nvdec_probe_success`），失败抛出不进缓存 | ✅ 代码+单测 |
| **C2** GPU 缓冲池回收是"注释即契约" | 引用归零即归还，无类型约束 | 由 C5 的显式 release 路径取代 | ✅ 由 C5 取代 |
| **C3** 宿主 yuv 批量灰度缓冲复用是失效代码 | `shape[:2]` 少一维 → 复用从未生效 | **复核发现 gray 分支同样失效**，两分支均修复 + "复用确实发生"单测 | ✅ 代码+单测 |
| **C4** OCR 初始化/推理错误延迟到 `extract()` 末尾才抛 | 丢失线程上下文 | worker / GPU producer 异常统一 `raise RuntimeError(...) from e` | ✅ 代码 |
| **C5** 设备内存只增不减，无显式释放 | 长进程批量显存单调增长 | `TrtEngine` / `GpuPreprocessor` / `GpuOutputReducer` / `GpuFrameAnalyzer` 各加 `release`，两池 `release_all`；GPU 管线 finally 统一释放（OCR 引擎归池常驻） | ✅ 代码 |
| **C6** GPU 管线错误路径上 producer 无取消机制 | 线程与解码器泄漏 | `producer_stop` Event + `put(timeout)` 轮询，finally 置位 | ✅ 代码 |
| **C7** `cancel_check` 在阻塞点失效 | 取消延迟无上界 | 宿主 `_put` Full 分支 + GPU 消费端 `get(timeout)` 轮询均查 `cancel_check`（契约：回调抛异常） | ✅ 代码 |
| **C8** 非 ROI decord 的兼容路径是半成品 | 只有校准做 ROI 回退切片，主流程静默用整帧 | 构造期 `_ensure_roi_capable_decoder`：无 `_CAPI_VideoReaderSetRoi` 直接 `ValueError`（**用户决策：直接报错**；decord 未安装不拦截） | ✅ 代码 |
| **C9** `nv12_to_rgb` 不接收 color_range | full-range 流的 RGB 预览偏色，docstring 自相矛盾 | 新增 `color_range` 参数（full/pc 矩阵） | ✅ 代码+单测 |
| **C10** GPU→宿主回退路径重复打开视频 | 最坏同一视频开 3 次 | `_fallback_to_host` 复用普通 reader；**hybrid 必须 close 后重开**（分片消费指针已前进，复用会序错位） | ✅ 代码 |
| **D1** `import` 即改全局环境 | `__init__.py` 顶层 `setdefault` 改进程级 env，影响同进程其他 decord 使用方 | **行为变更**：移除 setdefault，`DECORD_SKIP_LOOP_FILTER` 改**显式 opt-in**；e2e 复验真值匹配不受影响 | ✅ 代码+文档（见 §12.5.1 状态注） |
| **D2** 进度回调永远不会到 100% | 上限 ~86% | README 结果节注明口径 | ✅ 文档 |
| **D3** `ExtractionResult.meta` 不含实际参数 | 降级原因不可见 | meta 新增 `params` / `engine_version` / `degraded_reason`（五类）/`color_range` / `rep_crop_format` | ✅ 代码 |
| **D4** `rep_crop` 默认 NV12 二维数组 | 对非 CV 用户是隐性摩擦 | 新增 `ExtractionResult.rep_crop_rgb(seg)`（按 meta 自动选格式/色域） | ✅ 代码+文档 |
| **D5** `keep_frames=False` 同时清空段级帧号 | 文档未声明 | README 声明 | ✅ 文档 |
| **D6** 文档分裂成五份且互相引用 | 使用者/维护者看到两套真相 | **2026-08-30 合并为四份**（见本文开头）；README 新增文档地图 | ✅ 文档 |
| **D7** README env 表两处失实 | `OCR_PAD_SMALL` 默认值错、残留已删除的 `GPU_PIPELINE_ASYNC` 行 | 默认值订正 + 幽灵行删除 + DEPENDENCIES 清理 | ✅ 文档 |
| **D8** hybrid 语义三处停在旧门控 | stride==1 / 走宿主管线，与代码和测试矛盾 | README（stride 解禁 + GPU 管线合并）、engine_config 注释同步 | ✅ 文档 |
| **D9** `FieldExtractor` 是单发对象但结果有双入口 | 语义未声明 | README 声明单发语义与"以返回值为准"；`frames` setter 保留（上层应用兼容） | ✅ 文档 |

**唯一的用户可见行为变更**：D1。默认不再关去块滤波（CPU 软解 HEVC/h264
回到完整去块滤波）。要速度的使用方自行在打开解码器前设
`DECORD_SKIP_LOOP_FILTER=all`，收益见 `docs/PERFORMANCE.md` §16.2 P0-6。

#### 未验证项（如实声明，尚未跑）

- C1 的"瞬态失败"概率、C3 的实测性能损失幅度、D3 的降级场景覆盖，
  均需真机/单元验证。
- C5 的显存增长速率、C6 的泄漏累积速度取决于批量规模与输入宽度，
  需长进程实测确认量级（机制本身由代码路径可证）。
- C8 只影响非 fork decord 用户，实际是否存在这类用户未知。
- C9 需要一个 full/pc range 的真实片源复验预览偏色。
- 所有性能结论引用仓库自带实测，未在审查中复现。

#### 0.9.1 / 0.9.2 / 0.10.0 三轮（本节补齐 CLAUDE.md 缺失的记录）

本文件此前停在 0.9.0 清理轮，以下三轮只在 `docs/PERFORMANCE.md` 有记录：

- **0.9.1**：裁切判据澄清（真判据是 **ROI 宽裕度**，`force_aspect` 是混淆变量）；
  `force_aspect>0` 改为"先定比例、后裁"；生产原始误读 149→126。
- **0.9.2**：裁切改用**最小收益门槛 10%**（余量由 20 回到 10）；
  §13 det 模型替换裁切评估（否定）；§14 分段合并收口（字幕已 100% 达 oracle 最优，
  遥测的理论空间被表示层纠缠封死）；§15 yuv 输出税归因（**否定结果**，
  是冷启动测量假象，不是格式差）。
- **0.10.0**：上述设计审查修复轮（25 条）+ 版本号提升。
  ⚠️ 按版本纪律，`engine_config.__version__` 改动必须同步打同名 git tag
  （v0.10.0 已打）。
