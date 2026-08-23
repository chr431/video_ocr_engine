# CLAUDE.md — 开发记录与约定

本文件用于记录 `video_ocr_engine` 的开发决策、实验结论与维护约定。
**README 只写用户向 API/使用说明**；开发过程结论（尤其“为什么这样做/为什么不做”）
写在这里和 `docs/PERFORMANCE.md`，避免污染用户文档。

## 通用约定

- 引擎是通用文本提取库，不携带速度/字幕领域后处理；领域语义由上层应用完成。
- 默认行为必须保持向后兼容：新功能默认关闭，除非明确作为新默认。
- 性能实验结论（尤其失败/无收益/最优参数）追加到 `docs/PERFORMANCE.md`，
  避免重复投入。

## 单实例双完整流水线并行（2026-08，二轮重构）

### 设计

- 一个 `FieldExtractor` 实例内，把同一视频的采样帧序列切片；两条完整
  “解码→分段→OCR”流水线作为消费者动态取片，最后按帧序合并。
- **默认互补对 = `(auto,auto) ∥ (cpu,auto)`**（二轮起）：两条流水线都用
  TensorRT 推理，仅解码侧互补。探针实测 TRT⊕ONNX 共存推理互相膨胀
  （2.9/16.5 ms/段 → 双方 ~8.4，总吞吐钉死），TRT+TRT 无膨胀。
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
  AV1 编码（CPU 软解已知净负，`RVTOL_DUAL_NO_CODEC_FALLBACK=1` 可关）。
- 默认关闭（`dual_pipeline=False`），环境变量 `RVTOL_DUAL_PIPELINE=1` 可开启。

### 探针定位的损耗来源（2026-08 二轮，勿再猜测）

1. **混配退化的真因 = 内存子系统争抢（三轮探针定位，勿再用"核饱和/
   调度饥饿/GIL/对称收敛"表述）**。判别链：CPU 占用仅 ~46%（未饱和）；
   跨进程仍退化（非 GIL）；`RVTOL_ORT_SPIN=0` 无效（非自旋）；双侧绑核
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
   补充（四轮-b）：尝试削减 ONNX 侧可裁剪的输出流量——图级追加
   ArgMax+ReduceMax（`RVTOL_ONNX_CTC=1`，需 onnx 包一次性构建缓存，
   数值与旧路径 100% 一致）。结果：共存时 TRT 退化仅 +75%→+67%、ONNX
   自身 -6%，单引擎反而 +10%（ORT 归约核不如 numpy 成块 SIMD）——证明
   混配干扰的主体是 ONNX 计算内部的激活/权重访存，而非可裁剪的 I/O
   张量；默认关闭。进一步缓解只剩模型量化路线（未立项）。
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

### 实现注意

- 每条 worker 使用“持久 OCR 会话”（`_start_ocr_session`）：一个 OCR worker +
  infer 队列跨所有切片复用，切片之间不做 join；后一片解码可与前一片 OCR 重叠。
- `RVTOL_PROFILE=1` 时各流水线 profile 按 `producer:pipeN / ocr:pipeN`
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
- decord GPU 帧直通已实现为实验路径（`RVTOL_GPU_RAW=1` 开启）：
  - 从 decord gray NDArray DLPack 解析 device ptr，代表帧 D2D 聚批后
    `prep_gray_raw` kernel 在 GPU 完成 resize+gamma+normalize+pad；
  - 默认关闭。
  - 本机 test5 1500/3000 帧 A/B：**开启反而慢 20~30%**（当前仍做每帧
    asnumpy 供分段，raw 只省代表帧，却增加 GPU kernel 与 D2D 竞争）。
  - 结论：要真正收益必须把灰度/sharp/分段也留在 GPU，当前实验路径保留参考。
- GPU 灰度/sharp/聚类分段已实现实验路径（`RVTOL_GPU_PIPELINE=1`）：
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

### 显存全驻立路径补全与争抢韧性验证（2026-08 三轮，RVTOL_GPU_CTC）

在"内存子系统争抢"结论（见双流水线小节）之后，把 GPU+TRT 路径最后两个
RAM 大触点补掉，形成显存全驻留闭环：

- **逐帧直方图校准**（`GpuFrameAnalyzer.histograms_perframe`）：每帧 256-bin
  直方图在 GPU 统计，D2H 仅 B×1KB 标量表，宿主复刻"前 50 帧 Otsu 取中位
  数"语义——校准阈值行为与单流水线逐位一致（此前全局池化直方图阈值不同，
  段数差 4×）；
- **TRT 输出 GPU argmax 归约**（`GpuOutputReducer` + `TrtEngine.
  execute_device_argmax`，env `RVTOL_GPU_CTC=1`）：(B,S,C) float32 在 GPU
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
  contrast 且未开 dual_pipeline 时自动启用；`RVTOL_GPU_PIPELINE=0` 显式
  关闭，'1' 强制尝试。yuv_output（RaceVideoToLog）暂走宿主管线。
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

### 字幕/背景分离预处理（实验，RVTOL_TEXT_SEP）

- 实验替代当前“灰度 + gamma 2.0”的 OCR 输入：
  - contrast：局部盒式背景估计 + 绝对差分，突出文字笔画/边缘；
  - binary：用分段 Otsu 阈值二值化，白字黑底。
- `RVTOL_TEXT_SEP=contrast|binary` 开启 OCR 预处理；同时会让
  `_segments_similar` 先做分离再比较，尝试让相似帧合并只关注文字变化。
- 短窗口新三国01（1500 采样帧）实测：
  - off：215 段 / 116 文本
  - contrast：216 段 / 116 文本，耗时 +9%
  - binary：214 段 / 115 文本，耗时 -6%
- 结论：当前简单版没有明显准确率/合并收益，保留为实验入口，默认关闭。
  后续需要更细的文字分割（连通域、颜色/亮度先验、形态学过滤）再评估。

### Race 跨编码实测（2026-08 一轮，1500 帧窗口，旧 CPU+ONNX 互补对）

> 以下为一轮（CPU+ONNX 互补对）的历史结论；二轮重构后短窗口由
> `DUAL_PIPELINE_MIN_FRAMES` 门控回退，全片长见上文最终实测（-27~-42%）。

- h264（test3/test5）：默认双流水线 2 片有 7~17% 收益；两条 CPU+TRT 略快。
- HEVC（test）：只有 8 片时接近持平；AV1（test6）：双流水线全部明显变慢
  （+42~87%）——CPU 软解/ONNX 路径成为瓶颈。
- 动态切片会让 GPU+TRT 路径拿到更多片（如 AV1 8 片时 GPU 5 片 / CPU 3 片），
  但“快路径做完后等待慢路径”仍存在，分片数只能缓解不能根治。
- **一轮生产结论（已被二轮取代）：保持默认关闭。**
