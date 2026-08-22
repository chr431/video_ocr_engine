# CLAUDE.md — 开发记录与约定

本文件用于记录 `video_ocr_engine` 的开发决策、实验结论与维护约定。
**README 只写用户向 API/使用说明**；开发过程结论（尤其“为什么这样做/为什么不做”）
写在这里和 `docs/PERFORMANCE.md`，避免污染用户文档。

## 通用约定

- 引擎是通用文本提取库，不携带速度/字幕领域后处理；领域语义由上层应用完成。
- 默认行为必须保持向后兼容：新功能默认关闭，除非明确作为新默认。
- 性能实验结论（尤其失败/无收益/最优参数）追加到 `docs/PERFORMANCE.md`，
  避免重复投入。

## 单实例双完整流水线并行（2026-08）

### 设计

- 一个 `FieldExtractor` 实例内，把同一视频的采样帧序列切成多个连续小片；
  两条完整“解码→分段→OCR”流水线作为消费者从队列动态取片，最后按帧序合并。
- 与旧 `HYBRID_DECODE` / `HYBRID_OCR` 不同：旧方案只在一个阶段内并行，
  新方案两条流水线各自拥有独立解码器和 OCR 引擎。
- 默认后端组合 = 主后端 + 互补后端（CPU ↔ GPU/TRT）；调用方可传
  `dual_backends=[(decode, ocr), (decode, ocr)]` 显式指定。
- 需要 NVDEC 和 TensorRT 均可用；否则回退单流水线（不硬报错）。
- 默认关闭（`dual_pipeline=False`），环境变量 `RVTOL_DUAL_PIPELINE=1` 可开启。

### 实测结论

- 标清 h264 字幕、stride=4、6000 采样帧（Plan B 持久 OCR 会话后）：
  - 单 auto+auto：7.32s
  - 默认双流水线（2 片）：4.97s（-32%）
  - 默认双流水线（4 片）：5.27s（-28%）
  - 默认双流水线（8 片）：5.70s（-22%）
- 分片数 2~4 最优；8 片虽已消除每片 OCR join 屏障，仍因 seek/边界处理略慢。
- 双流水线唯一非空字幕集合与单流水线一致（抽查），但噪声重复分段更少。

### 实现注意

- 每条 worker 使用“持久 OCR 会话”（`_start_ocr_session`）：一个 OCR worker +
  infer 队列跨所有切片复用，切片之间不做 join；后一片解码可与前一片 OCR 重叠。
- 解码器在 worker 内打开一次，跨片 seek，避免每片重新 open。
- 片边界会引入分段边界，但抽查未发现唯一文本缺失；字幕场景由
  `merge_similar` / 下游去重吸收。
- 当前仅支持 2 条流水线；超过 2 条的后端组合暂未开放。

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
  - E2E 仍由 NVDEC 解码瓶颈主导，test5 3000 帧约 3.46s 基本不变。
- pinned host staging 微测反而更慢（额外 CPU 拷贝），未采纳。
- 当前仍存在 1 次 HtoD（预处理后的 batch） + 1 次 DtoH（TRT 输出后处理用）。
  GPU 侧预处理 / decord GPU 帧直通显存需要更大重构，尚未落地。

### Race 跨编码实测（2026-08，1500 帧窗口）

- h264（test3/test5）：默认双流水线 2 片有 7~17% 收益；两条 CPU+TRT 略快。
- HEVC（test）：只有 8 片时接近持平；AV1（test6）：双流水线全部明显变慢
  （+42~87%）——CPU 软解/ONNX 路径成为瓶颈。
- 动态切片会让 GPU+TRT 路径拿到更多片（如 AV1 8 片时 GPU 5 片 / CPU 3 片），
  但“快路径做完后等待慢路径”仍存在，分片数只能缓解不能根治。
- **生产结论：保持默认关闭；不要作为 Race 通用默认。** 适合用户显式选择、
  h264 且 CPU 较强的场景。
