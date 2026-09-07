# 现役结论索引（L1，唯一规范性结论地）

> **规则**：全仓"结论/禁令/勿再投入"只住在本文件（一行一条，预算受测试守护
> `tests/test_docs_hygiene.py`）。新结论以一行进表；实验叙事进 `docs/log/` 或
> PERF 历史，不得混写。依赖/大迁移变更时 grep 本表"前提"列并逐条翻状态。
> 参数默认值的唯一事实源是 `engine_config.py`（PERF §5 只存调参依据）。
> 证据列只写指针；指向已删除文件时不加反引号（悬空符号测试豁免见测试 docstring）。
>
> **状态**：`active`（现役有效）/ `superseded(C-xx)`（被取代，勿据此调参）/
> `dead`（已验证死路）。**"复评触发"= 前提变化时重评的条件；没有触发的 dead
> 才是永久死路。** 实测基线：7945HX + RTX 4060，decord fork 0.7.x（2026-09）。

| ID | 结论 | 状态 | 前提/边界 | 复评触发 | 证据 |
|----|------|------|-----------|----------|------|
| C-01 | 并发退化真因 = NVDEC 会话数（单硬件单元串行）；NVDEC∥CPU 互补聚合 1.83–1.87×，双 NVDEC 仅 1.01–1.20× | active | 本机单 NVDEC 单元；同视频同负载 | 多 NVDEC 单元 GPU / 驱动调度变更 | PERF §19 §21 |
| C-02 | IO 不是并发退化原因（<1% 墙钟；页缓存全命中仍退化 1.88×） | active | NVMe + 页缓存命中 | 冷盘/网络盘/超长视频使 IO 占比抬升 | PERF §19 |
| C-03 | 内存带宽不是并发变量（B_max 实测 55.8 GB/s；互补设计仅 7.8 GB/s 退化 1.02×） | active | 2×16GB DDR5-6000 独显平台 | 共享内存带宽的集成平台 / 内存减半 | PERF §20 §21 |
| C-04 | 解码后端按编码选：h264 CPU 快 ~2.9×，AV1 反转慢 ~2.6× | active | fork 0.7.x 解码路径、本机核数 | decord 侧重写解码后端 / 新增编码 | PERF §21 §22.1 |
| C-05 | hybrid = decord fork 原生（≥v0.7.15）：TRT 走 hybrid_gpu 显存直通、CPU OCR 走宿主帧；e2e hevc 1.42× / h264 1.22× vs 纯 NVDEC | active | decord fork ≥v0.7.15 | fork 版本升级；AV1 侧数据补齐 | PERF §24；DECISIONS「混合解码迁移」 |
| C-06 | 项目层 hybrid 调度（v3~v7：kfe 分片/校准/折扣/在线移界/窃取） | superseded(C-05) | — | — | PERF §22 §23（历史）；DECISIONS v3/v4 |
| C-07 | `auto` 不区分 OCR 后端、一律尝试 NVDEC：批量互补必须显式 `decode_backend="cpu"` 并核验 `_backend` | active | — | auto 实现按 ocr_backend 分叉后复核 | README 批量章；PERF §19 §21 |
| C-08 | `auto` 在 h264 多核非最优（CPU+TRT 约 2×），但静态判据不可靠、判错代价成倍 → 保持 auto | active | — | 出现可靠的运行时解码速率探测 | DECISIONS 审查 A2 |
| C-09 | GPU 分段+ONNX 无净收益，GPU 管线默认只放行 NVDEC+TRT | active | onnxruntime 1.29 CPU ep；解码为瓶颈 | ort 支持 IO binding / 新执行提供器；解码供给率大幅提升 | PERF §9 |
| C-10 | GPU_PIPELINE_STREAM 默认关：流水发射零实测收益，decord 侧 API 保留 opt-in | active | 解码仍是瓶颈（消费不反超供给） | OCR 提速使消费反超解码供给 | 提交 9b83cba / d9c96f9 |
| C-11 | 分段合并：不误合并约束下无普适空间 | active | 现有分段算法 + 5 真值视频 | 分段算法更换 / 真值扩容暴露误合并 | PERF §14 |
| C-12 | det 裁切替换（PP-OCRv6 det 模型进热路径）：4559 段文本零变化 + 性能净负 | dead | 2026-08-30 实测 | PERF §13「重提条件」：自动发现 ROI / 离线 QA 工具 / 启发式无法处理的 ROI 形态 | PERF §13 |
| C-13 | 真跳帧（丢 nal_ref_idc==0 整包）安全，但收益仅 1.03–1.48×（原估 2~4×） | active | H.264、fork 0.7.x | 新编码 / 更激进的过滤方案 | DECISIONS「下一步三目标轮」 |
| C-14 | skip_loop_filter 收益 1.11–1.36× 但改变输出像素；默认 opt-in | active | — | 下游证实对像素不敏感且需提速率 | DECISIONS「P0-6 翻案」 |
| C-15 | OCR pad 下限 224 保持（160 已回退：生产误读退化；320 已证伪） | active | — | pad / 预处理架构变更 | ARCHIVE §16.2 |
| C-16 | OCR 裁切余量 10% 优于 0%；裁切即使省不到算力也能提准确率（旧"守卫"前提是错的） | active | — | ROI 形态 / 分辨率大变 | DECISIONS「第四轮」 |
| C-17 | stride>1 跳 B 帧加速（AVDISCARD_NONREF / packet 级过滤） | dead | 两种实现均错帧；根因 = High profile 部分 B 帧是参考帧 | 解析 slice header reference 标记的完整实现（成本 ≈ 重写跳帧解码器） | PERF §6 |
| C-18 | 异步批量解码（fork experiment 分支） | dead | 测试视频均为 NVDEC 硬件解码上限 | 硬件解码器换代后重测上限 | PERF §6 |
| C-19 | 用码流信息（pkt_size/pict_type）判断 ROI 变化 | dead | 分离度 0.126–0.417、召回 0.269，必须熵解码 ≈ 解码本身 | — | PERF §6 |
| C-20 | Tesseract 替代 PP-OCR | dead | 慢约 25×，正确率 33–95% 不稳 | — | PERF §6 |
| C-21 | FP16 / INT8 TRT 引擎 | dead | tiny/small 非算力受限；FP16 构建反慢 2.2× | 更大模型 / TRT 大版本升级 | PERF §6 |
| C-22 | onnxruntime 线程/执行参数调优（1.29 新增参数） | dead | 全部无收益 | ort 大版本升级 | PERF §6 §7 |
| C-23 | OCR 专职 preprocess 线程（三级流水线） | dead | 严格 A/B 无净收益，已回滚 | OCR 预处理显著变重时 | PERF §6 |
| C-24 | 冷启动 prewarm 引擎 | dead | 批量 5 集总量不变 | — | PERF §17.4 |
| C-25 | cluster_win3 定点下沉（对默认用户无效） | dead | GPU 全驻留默认路径无 Python 分段 | 宿主路径 + ROI≥4 万 px + 分段暴露为瓶颈 | PERF §17.3 |
| C-26 | hybrid CPU 端线程数按核数分档（项目层实测 8→24 墙钟 −8.6%） | superseded(C-05) | 旋钮保留，decord 原生下自动分档钳 [8, 16] | decord 原生实现下重测线程网格 | PERF §17.2 §22 |
| C-27 | yuv 输出格式墙钟税 | superseded（税不存在 = 冷启动测量假象；yuv 输出已删除，架构改单通道灰度） | — | — | PERF §15 |
| C-28 | 多预处理自动选择 / 窗口重 OCR 自动化 / scipy 连通域 | dead | 均被现有方案覆盖或净负 | — | PERF §6 |
| C-29 | 短任务 hybrid 固定开销交叉点（N*≈8182 帧前不如纯 NVDEC） | superseded(C-05) | 项目层调度实测 | — | PERF §22.1 |
| C-30 | GPU 分段异步（GPU_PIPELINE_ASYNC） | dead | NVDEC/CPU 分支均无收益，钩子已删 | — | DECISIONS「0.9.0 清理轮」 |
