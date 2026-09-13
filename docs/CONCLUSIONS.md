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
| C-05 | hybrid = decord fork 原生（TRT→hybrid_gpu、CPU→宿主帧）。**收益面 = TRT/设备路径**；**ONNX 宿主路径不反超** → 选 nvdec。可见性：h264 2.45×/hevc 17%/av1 30%、无热降；hevc 绑定侧=CPU 臂忙而慢 | 4060/16C32T；0.8.3 已发布（b67f9eb）；忙时计数 fork ≥6da2957 | >32 核档位复测 / 份额倾斜已由 C-45 落地，残余=每臂混跑干扰 | log 2026-09-11-hybrid差距分解 §11-§16；log 2026-09-13-hybrid可见性重评 |
| C-07 | `auto` 不区分 OCR 后端、一律尝试 NVDEC：批量互补必须显式 `decode_backend="cpu"` 并核验 `_backend` | — | auto 实现按 ocr_backend 分叉后复核 | README 批量章；PERF §19 §21 |
| C-08 | `auto` 在 h264 多核非最优（本机 CPU+TRT 快 1.7~2.8×），**但 auto 恒为 NVDEC 优先是刻意决策**：弱 CPU 可能反慢 + CPU 解码必带争用/功耗代价；峰值留给显式 `decode_backend="cpu"` | 稳妥性 > 本机峰值吞吐（用户拍板） | 仅当出现「NVDEC 不可用」级前提变化 | DECISIONS 审查 A2；log 2026-09-09 深度性能优化 |
| C-09 | GPU 分段+ONNX 无净收益，GPU 管线默认只放行 NVDEC+TRT | onnxruntime 1.29 CPU ep；解码为瓶颈 | ort 支持 IO binding / 新执行提供器；解码供给率大幅提升 | PERF §9 |
| C-10 | GPU_PIPELINE_STREAM 默认关：流水发射零实测收益，decord 侧 API 保留 opt-in | 解码仍是瓶颈（消费不反超供给） | OCR 提速使消费反超解码供给 | 提交 9b83cba / d9c96f9；2026-09-09 decord 0.8.2 复确认零收益（log 深度性能优化） |
| C-13 | 真跳帧（丢 nal_ref_idc==0 整包）安全，但收益仅 1.03–1.48×（原估 2~4×） | H.264、fork 0.7.x | 新编码 / 更激进的过滤方案 | DECISIONS「下一步三目标轮」 |
| C-14 | skip_loop_filter 收益 1.11–1.36× 但改变输出像素；默认 opt-in | — | 下游证实对像素不敏感且需提速率 | DECISIONS「P0-6 翻案」 |
| C-15 | **OCR 输入侧参数都不是杠杆**。pad 下限 224 保持（160/320 均证伪）；预处理六变体仅一个非负 +7 帧；翻转系实例级边缘判决（test5/6 统计近乎相同却对每个旋钮反向） | v6_small + racelog 内容；真实管线（宿主≡GPU） | pad/预处理架构变更；换模型或字体/ROI 形态 | log 2026-09-12-准确项 §4.1；bench/prep_ab.json |
| C-16 | OCR 裁切余量 10% 优于 0%；裁切即使省不到算力也能提准确率（旧"守卫"前提错） | — | ROI 形态 / 分辨率大变 | DECISIONS「第四轮」 |
| C-31 | decord 0.8.1 + FFmpeg9：seek 未变慢；"av1 seek 变慢"系 NT=4 假象 + dav1d 扩展性被旧线程策略埋没。修：av1 → 逻辑核 3/4 钳 [8,24]，CPU 后端 −50%、e2e −45% | pip wheel 0.8.2（dll md5 6597eea6，已含修复，DLL 随包自带） | fork 再升级 / 驱动或 FFmpeg 再换代 | log 2026-09-08 decord-0.8.1；tools/_ab_decord081/；DEPENDENCIES decord 节 |
| C-32 | 分段/状态机/裁切/预处理实现唯一出处 = segmentation.py：宿主直调，GPU kernel 为设备侧逐位镜像、判据引用同一文件；不做插件抽象面（GPU 侧不可插拔=降速宿主，已回退） | 0.11.0；两侧行为由真值用例逐位守护 | 出现真实的 GPU 侧算法插件需求 | log 2026-09-09 引擎四方向；tests/ 全套 |
| C-34 | S7 复评（Q4）：单视频仍受解码供给限制（h264-gpu ≈ NVDEC 1013fps），ExtractionPool 互补配对前提成立；但为新增 API，待显式立项 | 2026-09-10 S0/S6 bench（本机 4060/8GB）；触发条件①已满足 | 出现多视频批量场景 / S5 内联后解码格局变化 / 显式立项请求 | tests/golden/bench_baseline.json；log 2026-09-10-S6性能轮 §0 |
| C-36 | 冷启动 = cuda.core 0.22s + NVRTC 0.09s + TRT 反序列化 0.39–0.44s；显式 `warmup()` 移出首个 extract（首视频 −33%），总吞吐不变；预热前移零收益已回退 | 引擎文件已缓存；首个 extract | 换后端/引擎格式 | log §3；tests/pipeline/test_warmup.py |
| C-37 | 消费端提交可批量化：归约 D2H→异步+本流同步（h264-cpu −5.19%）、keep_crops D2H 并批窗 16（h264-gpu −2.45%），逐位一致；pad 支配时按宽分组无收益 | 本机 4060；交错 A/B | 段密度翻倍 / ROI 形态使判据翻转 | log §4-§6；ocr_stage.effective_reorder_window |
| C-38 | **合并判定「稠密簇门」**（win3 ≥ SEG_C 恒不合并，默认开）净 +226 帧；代价段数 +1.2~3.6%、墙钟不变 | 本机 4060 + 六片真值；宿主/GPU 逐位一致 | 大字号码管字体 / OCR-bound 部署 | log 2026-09-12-准确项 §3 §6 |
| C-39 | **NVML NVDEC% 是「在用」指示器非占空比**（臂 35% 忙仍读 98%）——忙闲判别用 fork `[hybrid-stats] busy`；NVML 价值=时钟+热降原因位（实测无热降） | full 档 + fork ≥6da2957；本机单卡 | 换卡（NVDEC% 语义随驱动/卡型变）/ fork 换代 | log 2026-09-13-hybrid可见性重评；bench/hybrid_reeval.json |
| C-40 | **hybrid 收益与视频长度相关**：h264 无交叉点（500 帧起胜）；hevc/av1 交叉点 1500~3000 帧；短窗暖态劣势仅 0.06~0.10s。对纯 CPU 臂三码全片均胜（1.10×/2.76×/1.81×） | 4060/16C32T；每臂独立子进程 + min-of-2（两臂同刻，差值消掉约 0.85s 的进程/引擎固定开销）；ocr=trt 隔离解码 | 换卡（NVDEC 与 CPU 相对速率变）/ 关键帧间隔差异大的片源 / 超短片（<500 帧） | log 2026-09-13-hybrid解码率与理论并联和 §续七 |
| C-41 | **短窗缺口量级（已更正）**：暖态 hybrid−纯GPU臂 仅 +0.059s(w=1000)/+0.096s(w=500)，w=3000 反超；首次抽取差 +0.18~0.28s 属实例化/init 非调度。播种实验证伪"盲阶段采样块"归因，已回退 | 4060/16C32T；同进程 3 次取暖态均值（首次含 engine_init 不入账）；hevc/test6_hevc；对照臂 = hybrid 内 FORCE_SIDE=gpu | 换卡 / fork 实例化优化后重测 / 需"进程首跑"口径时单独测 | log 2026-09-13-份额旋钮修复与换DLL-A-B §四（含证伪链） |
| C-42 | 段边界相似判定单次 232µs、全片≈墙钟 21%，**但不在关键路径**：短路成恒不相似（分段逐位同）wall 无改善 ⇒ 消费者空等 84~87% 完全吸收；按占比推算收益是错的，不值得做 | GPU 管线；hevc/test6_hevc 6000 帧；该窗口 merges=0（短路前后段数均 2101） | 消费者不再空等的配置（OCR 极慢 / batch 极大）/ 段边界密度大幅上升的素材 / 换卡后 GPU 争用格局变化 | log 2026-09-13-merge判定代价与短路实验；ENGINE_PROFILE 分相 |
| C-43 | **fill_width 与 force_aspect 强交互**（224 保持）：force_aspect=1.5 下 224 零误读（30664 帧），降 0 反而差；仅 force_aspect=0 时关填充更准。`fill_width=0` 作加法开关保留，默认不动 | 六片真值；真值头记生成配置（test5/6_ref force_aspect=1.5；test2 =0 且 fill_width=320） | 换片源（内容宽/字号分布变）/ OCR 模型换代 / 默认 force_aspect 变更 | log 2026-09-13-填充宽度重测；`_probe_acc_ab.py` 已自动复刻真值头配置 |
| C-45 | **hybrid 并联缺口收口（fork 4cebeef）**：可修成分=①CPU 臂银行帽 1536 系全帧字节残留且 raw/frame 合并计价（h264 对侧值日期 27% 闲置）②计划冻结用 rg 爬升值（±13% 漂）→hevc 份额偏 CPU、GPU 队尾闲置。修复=queue 按 ROI 字节重算+raw 全帧 768MB 解耦+滑动视界 4096（`DECORD_HYBRID_PLAN_HORIZON=0` 回退；REPLAN_PCT 删）。fork 全片（干净对干净）hevc +12.8%（77→87%）、h264 +14.8%（68→78%）、av1 +3.0%（88→90%，GPU 臂=物理瓶颈）；金标 28/28。残余=每臂混跑干扰（CPU busy-rate −15~17%，32T 最优），上限 ≈87/78/90% | 4060/16C32T；fork 4cebeef（已装 wheel）；hybrid_gpu 口径；引擎干净对 A/B：hevc −4.8%（23/28）、av1 −2.9%（7/8）、h264 平价（引擎税+暖启+消费端吃掉 fork 收益） | 换卡 / fork 换代 / >32 核 / 绑核立项 | log 2026-09-13-hybrid并联缺口收口（bench/gap_decomp.json、dll_ab.json） |
| C-46 | **hybrid = 包缓存 + 供料期 GOP 派工（fork fef3c4b，已随 wheel 发布）**：Push 只入压缩包缓存（512MB），泵在分配时点按当下速率贪心派工整个 GOP（前瞻+同侧保序）；**kick 必须经泵按流序注入**（错位=IDR 重置冲掉重排窗→~50 帧遮蔽→段数漂移，fork 级测不出）。引擎 A/B：hevc −6.29%（6/6）、h264 −4.78%（6/6）、av1 平价；fork 对并联和 95/78/93%；18/18 段数恒定、金标 28/28。旧机制（计划/视界/盲窗/REPLAN/债务克隆）全删 | 4060/16C32T；旋钮 PKT_CACHE_MB/KICK_OFF/KICK_BURST/AV1_CPU/DECODE_CORES；AV1 混跑保留 | 换卡 / fork 换代 / 截断流·bf16 酷刑流重跑 / >32 核 | log 2026-09-13-hybrid重设计分支 §8；bench/dll_ab.json、gap_decomp.json |

## 已取代（指针）

- C-06 → C-05
- C-11 → C-38
- C-26 → C-05
- C-27 → C-36
- C-29 → C-05
- C-33 → C-08
- C-44 → C-45
- C-45 → C-46
- C-44 → C-45

（dead 条目已整体降级 docs/log/，含复评触发，显式检索可达。）
