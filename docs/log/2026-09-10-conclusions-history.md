# 结论历史（dead/superseded 全文，自 CONCLUSIONS.md 降级；
# append-only，永不注入，显式检索可达——Q7 R3）

## C-06（superseded → C-05）
- 结论：项目层 hybrid 调度（v3~v7：kfe 分片/校准/折扣/在线移界/窃取）
- 前提：—
- 复评触发：—
- 证据：PERF §22 §23（历史）；DECISIONS v3/v4

## C-12（dead）
- 结论：det 裁切替换（PP-OCRv6 det 模型进热路径）：4559 段文本零变化 + 性能净负
- 前提：2026-08-30 实测
- 复评触发：PERF §13「重提条件」：自动发现 ROI / 离线 QA 工具 / 启发式无法处理的 ROI 形态
- 证据：PERF §13

## C-17（dead）
- 结论：stride>1 跳 B 帧加速（AVDISCARD_NONREF / packet 级过滤）
- 前提：两种实现均错帧；根因 = High profile 部分 B 帧是参考帧
- 复评触发：解析 slice header reference 标记的完整实现（成本 ≈ 重写跳帧解码器）
- 证据：PERF §6

## C-18（dead）
- 结论：异步批量解码（fork experiment 分支）
- 前提：测试视频均为 NVDEC 硬件解码上限
- 复评触发：硬件解码器换代后重测上限
- 证据：PERF §6

## C-19（dead）
- 结论：用码流信息（pkt_size/pict_type）判断 ROI 变化
- 前提：分离度 0.126–0.417、召回 0.269，必须熵解码 ≈ 解码本身
- 复评触发：—
- 证据：PERF §6

## C-20（dead）
- 结论：Tesseract 替代 PP-OCR
- 前提：慢约 25×，正确率 33–95% 不稳
- 复评触发：—
- 证据：PERF §6

## C-21（dead）
- 结论：FP16 / INT8 TRT 引擎
- 前提：tiny/small 非算力受限；FP16 构建反慢 2.2×
- 复评触发：更大模型 / TRT 大版本升级
- 证据：PERF §6

## C-22（dead）
- 结论：onnxruntime 线程/执行参数调优（1.29 新增参数）
- 前提：全部无收益
- 复评触发：ort 大版本升级
- 证据：PERF §6 §7

## C-23（dead）
- 结论：OCR 专职 preprocess 线程（三级流水线）
- 前提：严格 A/B 无净收益，已回滚
- 复评触发：OCR 预处理显著变重时
- 证据：PERF §6

## C-24（dead）
- 结论：冷启动 prewarm 引擎
- 前提：批量 5 集总量不变
- 复评触发：—
- 证据：PERF §17.4

## C-25（dead）
- 结论：cluster_win3 定点下沉（对默认用户无效）
- 前提：GPU 全驻留默认路径无 Python 分段
- 复评触发：宿主路径 + ROI≥4 万 px + 分段暴露为瓶颈
- 证据：PERF §17.3

## C-26（superseded → C-05）
- 结论：hybrid CPU 端线程数按核数分档（项目层实测 8→24 墙钟 −8.6%）
- 前提：旋钮保留，decord 原生下自动分档钳 [8, 16]
- 复评触发：decord 原生实现下重测线程网格
- 证据：PERF §17.2 §22

## C-27（superseded → ?）
- 结论：yuv 输出格式墙钟税
- 前提：—
- 复评触发：—
- 证据：PERF §15

## C-28（dead）
- 结论：多预处理自动选择 / 窗口重 OCR 自动化 / scipy 连通域
- 前提：均被现有方案覆盖或净负
- 复评触发：—
- 证据：PERF §6

## C-29（superseded → C-05）
- 结论：短任务 hybrid 固定开销交叉点（N*≈8182 帧前不如纯 NVDEC）
- 前提：项目层调度实测
- 复评触发：—
- 证据：PERF §22.1

## C-30（dead）
- 结论：GPU 分段异步（GPU_PIPELINE_ASYNC）
- 前提：NVDEC/CPU 分支均无收益，钩子已删
- 复评触发：—
- 证据：DECISIONS「0.9.0 清理轮」

## C-33（superseded → C-08）
- 结论：「auto 按 codec 选路（h264→CPU 软解，探测带缓存）」**已实现后回退（2026-09-10 用户拍板）**：本机实测 h264 CPU 软解快 1.7~2.8×、hevc/av1 NVDEC 快 1.5~2.2× 属实，但 auto 恒为 NVDEC 优先——弱 CPU 鲁棒性 + CPU 解码的争用/功耗代价 > 本机峰值吞吐。实现与实测保留为证据，勿再据此改 auto
- 前提：本机（7945HX+RTX 4060）强多核口径
- 复评触发：弱 CPU 数据出现且用户重新拍板
- 证据：log 2026-09-09 深度性能优化（含 2026-09-10 勘误）；tools/_ab_perf_20260909/
