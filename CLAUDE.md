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

1. TRT⊕ONNX 混配单边饥饿（引擎级微基准定位，勿再用"对称收敛/共享串行门"
   表述）：干净速率 TRT 2.57 / ONNX(14T) 8.28 ms/段；混配时 ONNX 满速不变，
   **只有 TRT 被饥饿**（同进程 4.47、跨进程 4.05 → 非 GIL；nospin 无效 →
   非自旋），真实双流水线里叠加解码竞争恶化到 8.28。机制 = ONNX 真实计算
   占满物理核，TRT 宿主提交线程被调度饥饿。任何混配合计吞吐 ≤ 纯 TRT 单跑
   的 96% → 默认对不混用推理后端（见上）；显式混配时 ONNX 侧自动限流
   `DUAL_PIPELINE_ONNX_PEER_THREADS=6`（恢复 TRT 至 +32%）。
   另：早先"ONNX 单跑 16.5ms"是被其流水线自身解码/生产者污染的读数。
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
    raw OCR 转 infer 线程异步、GPU 分段批加大到 64、
    GPU 直方图 Otsu 校准、生产者线程让 analyze 与分段/OCR 重叠。
  - test5 1500 帧 A/B：开启仍比 host 慢（2.50 vs 1.74），raw 单独开启也慢
    （2.34~2.39 vs 1.71）。micro 上 raw 与 host 接近，E2E 慢主要来自
    GPU 路径与 decode/TRT 之间仍缺少真正统一的异步流水线。
  - 结论：当前实验路径已完整实现但无净收益；不建议启用。
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
