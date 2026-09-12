# 现役结论索引（L1，唯一规范性结论地）

> 本文件由 knowledge/render.py 从 knowledge/conclusions.yaml 渲染
> （人不得手写；--check 校验一致性）。状态取值：active / superseded
> （被取代，只留指针）/ dead（已降级 docs/log 历史）。规则与完整
> 说明见 yaml 头部注释。

| ID | 结论 | 前提/边界 | 复评触发 | 证据 |
|----|------|-----------|----------|------|
| C-01 | 并发退化真因 = NVDEC 会话数（单硬件单元串行）；NVDEC∥CPU 互补聚合 1.83–1.87×，双 NVDEC 仅 1.01–1.20× | 本机单 NVDEC 单元；同视频同负载 | 多 NVDEC 单元 GPU / 驱动调度变更 | PERF §19 §21 |
| C-02 | IO 不是并发退化原因（<1% 墙钟；页缓存全命中仍退化 1.88×） | NVMe + 页缓存命中 | 冷盘/网络盘/超长视频使 IO 占比抬升 | PERF §19 |
| C-03 | 内存带宽不是并发变量（B_max 实测 55.8 GB/s；互补设计仅 7.8 GB/s 退化 1.02×） | 2×16GB DDR5-6000 独显平台 | 共享内存带宽的集成平台 / 内存减半 | PERF §20 §21 |
| C-04 | 解码后端按编码选：h264 CPU 快 ~2.9×，AV1 反转慢 ~2.6× | fork 0.7.x、本机核数；0.8.1/FFmpeg9 下 av1 CPU 经济性已变（见 C-31），并行场景数字待重测 | 并行场景重测（C-31 策略修复后） | PERF §21 §22.1；log 2026-09-08 |
| C-05 | hybrid = decord fork 原生（TRT→hybrid_gpu、CPU→宿主帧）。**收益面 = TRT/设备路径**；**ONNX 宿主路径不反超** → 选 nvdec。可见性重评：h264 2.45×/hevc 17%/av1 30%、无热降；hevc 绑定侧=CPU 臂忙而慢，hevc/av1 剩余差距=GPU 臂调度闲置 | 4060/16C32T；0.8.3 已发布（b67f9eb）；忙时计数 fork ≥6da2957 | >32 核档位复测 / 份额向 GPU 再倾斜（fork 课题） | log 2026-09-11-hybrid差距分解 §11-§16；log 2026-09-13-hybrid可见性重评 |
| C-07 | `auto` 不区分 OCR 后端、一律尝试 NVDEC：批量互补必须显式 `decode_backend="cpu"` 并核验 `_backend` | — | auto 实现按 ocr_backend 分叉后复核 | README 批量章；PERF §19 §21 |
| C-08 | `auto` 在 h264 多核非最优（本机 CPU+TRT 快 1.7~2.8×），**但 auto 恒为 NVDEC 优先是刻意决策**：弱 CPU 可能反慢 + CPU 解码必带争用/功耗代价；峰值留给显式 `decode_backend="cpu"` | 稳妥性 > 本机峰值吞吐（用户拍板） | 仅当出现「NVDEC 不可用」级前提变化 | DECISIONS 审查 A2；log 2026-09-09 深度性能优化 |
| C-09 | GPU 分段+ONNX 无净收益，GPU 管线默认只放行 NVDEC+TRT | onnxruntime 1.29 CPU ep；解码为瓶颈 | ort 支持 IO binding / 新执行提供器；解码供给率大幅提升 | PERF §9 |
| C-10 | GPU_PIPELINE_STREAM 默认关：流水发射零实测收益，decord 侧 API 保留 opt-in | 解码仍是瓶颈（消费不反超供给） | OCR 提速使消费反超解码供给 | 提交 9b83cba / d9c96f9；2026-09-09 decord 0.8.2 复确认零收益（log 深度性能优化） |
| C-13 | 真跳帧（丢 nal_ref_idc==0 整包）安全，但收益仅 1.03–1.48×（原估 2~4×） | H.264、fork 0.7.x | 新编码 / 更激进的过滤方案 | DECISIONS「下一步三目标轮」 |
| C-14 | skip_loop_filter 收益 1.11–1.36× 但改变输出像素；默认 opt-in | — | 下游证实对像素不敏感且需提速率 | DECISIONS「P0-6 翻案」 |
| C-15 | **OCR 输入侧参数都不是杠杆**。pad 下限 224 保持（160/320 均证伪）。预处理跳出 gamma 的六变体：唯一非负 +7 帧，其余 −17~−1170。翻转系**实例级边缘判决**——test5/test6 裁切图强度统计近乎相同却对每个色调旋钮反向，同字形对 8→日 一侧修好、另一侧弄坏 | v6_small + racelog 内容；真实管线（宿主≡GPU） | pad/预处理架构变更；换模型或字体/ROI 形态 | log 2026-09-12-准确项 §4.1；bench/prep_ab.json |
| C-16 | OCR 裁切余量 10% 优于 0%；裁切即使省不到算力也能提准确率（旧"守卫"前提错） | — | ROI 形态 / 分辨率大变 | DECISIONS「第四轮」 |
| C-31 | decord 0.8.1 + FFmpeg9：seek 未变慢；"av1 seek 变慢"系 NT=4 口径假象 + dav1d 扩展性改善被旧线程策略（cores//2=8T）埋没。修：av1 → 逻辑核 3/4 钳 [8,24]，CPU 后端 −50%、host_cpu e2e −45% | pip wheel 0.8.2（dll md5 6597eea6，已含修复，DLL 随包自带） | fork 再升级 / 驱动或 FFmpeg 再换代 | log 2026-09-08 decord-0.8.1；tools/_ab_decord081/；DEPENDENCIES decord 节 |
| C-32 | 分段/状态机/裁切/预处理实现唯一出处 = segmentation.py：宿主直调，GPU kernel 为设备侧逐位镜像、判据引用同一文件；不做插件抽象面（GPU 侧不可插拔=降速宿主，已回退） | 0.11.0；两侧行为由真值用例逐位守护 | 出现真实的 GPU 侧算法插件需求 | log 2026-09-09 引擎四方向；tests/ 全套 |
| C-34 | S7 复评（Q4）：单视频仍受解码供给限制（h264-gpu ≈ NVDEC 1013fps），ExtractionPool 互补配对前提成立；但为新增 API，待显式立项 | 2026-09-10 S0/S6 bench（本机 4060/8GB）；触发条件①已满足 | 出现多视频批量场景 / S5 内联后解码格局变化 / 显式立项请求 | tests/golden/bench_baseline.json；log 2026-09-10-S6性能轮 §0 |
| C-36 | 冷启动 = cuda.core 0.22s + NVRTC 0.09s + TRT 反序列化 0.39–0.44s；显式 `warmup()` 把它移出首个 extract（engine_init →0.0001s，首视频 −33%），总吞吐不变；预热前移零收益已回退 | 引擎文件已缓存；首个 extract | 换后端/引擎格式 | log §3；tests/pipeline/test_warmup.py |
| C-37 | 消费端提交可批量化：归约 D2H→异步+本流同步（h264-cpu −5.19%）、keep_crops D2H 并批窗口 16（h264-gpu −2.45%），PI-14 全段逐位一致；pad 支配时按宽分组无收益 → 窗口按"ROI 宽高比 ≤ 下限比"收敛到 1 | 本机 4060；交错 A/B | 段密度翻倍 / ROI 形态使判据翻转 | log §4-§6；ocr_stage.effective_reorder_window |
| C-38 | 合并判定加「稠密簇门」（差异图 win3 ≥ SEG_C ⇒ 恒不合并，`segment.merge_dense_gate` 默认开）净 +226 帧（test5 +61 / test6 +165）；代价段数 +1.2~3.6%、墙钟不变 | 本机 4060 + 六片真值；宿主/GPU 逐位一致 | 大字号码管字体 / OCR-bound 部署 | log 2026-09-12-准确项 §3 §6 |
| C-39 | **NVML NVDEC% 是「在用」指示器非占空比**（臂仅 35% 忙时仍读 98%）——忙闲判别用 fork `[hybrid-stats] busy` 每臂忙时；NVML 价值=时钟+热降原因位。自此可分：hevc CPU 臂 91% 忙@655fps=忙而慢；GPU 臂 35% 忙+HOL=调度闲置（无热降，时钟贴顶） | full 档 + fork ≥6da2957；本机单卡 | 换卡（NVDEC% 语义随驱动/卡型变）/ fork 换代 | log 2026-09-13-hybrid可见性重评；bench/hybrid_reeval.json |

## 已取代（指针）

- C-06 → C-05
- C-11 → C-38
- C-26 → C-05
- C-27 → ?
- C-29 → C-05
- C-33 → C-08

（dead 条目已整体降级 docs/log/，含复评触发，显式检索可达。）
