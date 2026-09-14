# 活动结论（唯一事实源，按行解析非 YAML；docs/CONCLUSIONS.md 为渲染产物）
# 规则（Q7）：每条以 `- id:` 起含 status/premises/revisit/evidence；superseded 只留指针；dead 降级 docs/log/。

- id: C-01
  conclusion: 并发退化真因 = NVDEC 会话数（单硬件单元串行）；NVDEC∥CPU 互补聚合 1.83–1.87×，双 NVDEC 仅 1.01–1.20×
  status: active
  premises: 本机单 NVDEC 单元；同视频同负载
  revisit: 多 NVDEC 单元 GPU / 驱动调度变更
  evidence: PERF §19 §21

- id: C-02
  conclusion: IO 不是并发退化原因（<1% 墙钟；页缓存全命中仍退化 1.88×）
  status: active
  premises: NVMe + 页缓存命中
  revisit: 冷盘/网络盘/超长视频使 IO 占比抬升
  evidence: PERF §19

- id: C-03
  conclusion: 内存带宽不是并发变量（B_max 实测 55.8 GB/s；互补设计仅 7.8 GB/s 退化 1.02×）
  status: active
  premises: 2×16GB DDR5-6000 独显平台
  revisit: 共享内存带宽的集成平台 / 内存减半
  evidence: PERF §20 §21

- id: C-04
  conclusion: 解码后端按编码选：h264 CPU 快 ~2.9×，AV1 反转慢 ~2.6×
  status: active
  premises: fork 0.7.x、本机核数；0.8.1/FFmpeg9 下 av1 CPU 经济性已变（见 C-31），并行场景数字待重测
  revisit: 并行场景重测（C-31 策略修复后）
  evidence: PERF §21 §22.1；log 2026-09-08

- id: C-05
  conclusion: hybrid = decord fork 原生（TRT→hybrid_gpu、CPU→宿主帧）。**收益面 = TRT/设备路径**；**ONNX 宿主路径不反超** → 选 nvdec。可见性：h264 2.45×/hevc 17%/av1 30%、无热降；hevc 绑定侧=CPU 臂忙而慢
  status: active
  premises: 4060/16C32T；0.8.3（b67f9eb）；忙时计数 fork ≥6da2957
  revisit: >32 核档位复测 / 份额倾斜已由 C-45 落地，残余=每臂混跑干扰
  evidence: log 2026-09-11-hybrid差距分解 §11-§16；log 2026-09-13-hybrid可见性重评

- id: C-06
  status: superseded
  replaced_by: C-05

- id: C-07
  conclusion: `auto` 不区分 OCR 后端、一律尝试 NVDEC：批量互补必须显式 `decode_backend="cpu"` 并核验 `_backend`
  status: active
  premises: —
  revisit: auto 实现按 ocr_backend 分叉后复核
  evidence: README 批量章；PERF §19 §21

- id: C-08
  conclusion: `auto` 在 h264 多核非最优（本机 CPU+TRT 快 1.7~2.8×），**但 auto 恒为 NVDEC 优先是刻意决策**：弱 CPU 可能反慢 + CPU 解码必带争用/功耗代价；峰值留给显式 `decode_backend="cpu"`
  status: active
  premises: 稳妥性 > 本机峰值吞吐（用户拍板）
  revisit: 仅当出现「NVDEC 不可用」级前提变化
  evidence: DECISIONS 审查 A2；log 2026-09-09 深度性能优化

- id: C-09
  conclusion: GPU 分段+ONNX 无净收益，GPU 管线默认只放行 NVDEC+TRT
  status: active
  premises: onnxruntime 1.29 CPU ep；解码为瓶颈
  revisit: ort 支持 IO binding / 新执行提供器；解码供给率大幅提升
  evidence: PERF §9

- id: C-10
  conclusion: GPU_PIPELINE_STREAM 默认关：流水发射零实测收益，decord 侧 API 保留 opt-in
  status: active
  premises: 解码仍是瓶颈（消费不反超供给）
  revisit: OCR 提速使消费反超解码供给
  evidence: 提交 9b83cba / d9c96f9；2026-09-09 decord 0.8.2 复确认零收益（log 深度性能优化）

- id: C-11
  status: superseded
  replaced_by: C-38

- id: C-13
  conclusion: 真跳帧（丢 nal_ref_idc==0 整包）安全，但收益仅 1.03–1.48×（原估 2~4×）
  status: active
  premises: H.264、fork 0.7.x
  revisit: 新编码 / 更激进的过滤方案
  evidence: DECISIONS「下一步三目标轮」

- id: C-14
  conclusion: skip_loop_filter 收益 1.11–1.36× 但改变输出像素；默认 opt-in
  status: active
  premises: —
  revisit: 下游证实对像素不敏感且需提速率
  evidence: DECISIONS「P0-6 翻案」

- id: C-15
  conclusion: **OCR 输入侧参数都不是杠杆**。pad 下限 224 保持（160/320 均证伪）；预处理六变体仅一个非负 +7 帧；翻转系实例级边缘判决（test5/6 统计近乎相同却对每个旋钮反向）
  status: active
  premises: v6_small + racelog 内容；真实管线（宿主≡GPU）
  revisit: pad/预处理架构变更；换模型或字体/ROI 形态
  evidence: log 2026-09-12-准确项 §4.1；bench/prep_ab.json

- id: C-16
  conclusion: OCR 裁切余量 10% 优于 0%；裁切即使省不到算力也能提准确率（旧"守卫"前提错）
  status: active
  premises: —
  revisit: ROI 形态 / 分辨率大变
  evidence: DECISIONS「第四轮」

- id: C-26
  status: superseded
  replaced_by: C-05

- id: C-27
  status: superseded
  replaced_by: C-36

- id: C-29
  status: superseded
  replaced_by: C-05

- id: C-31
  conclusion: decord 0.8.1 + FFmpeg9：seek 未变慢；"av1 seek 变慢"系 NT=4 假象 + dav1d 扩展性被旧线程策略埋没。修：av1 → 逻辑核 3/4 钳 [8,24]，CPU 后端 −50%、e2e −45%
  status: active
  premises: pip wheel 0.8.2（md5 6597eea6，DLL 随包自带）
  revisit: fork 再升级 / 驱动或 FFmpeg 再换代
  evidence: log 2026-09-08 decord-0.8.1；tools/_ab_decord081/；DEPENDENCIES decord 节

- id: C-32
  conclusion: 分段/状态机/裁切/预处理实现唯一出处 = segmentation.py：宿主直调，GPU kernel 为设备侧逐位镜像、判据引用同一文件；不做插件抽象面（GPU 侧不可插拔=降速宿主，已回退）
  status: active
  premises: 0.11.0；两侧行为由真值用例逐位守护
  revisit: 出现真实的 GPU 侧算法插件需求
  evidence: log 2026-09-09 引擎四方向；tests/ 全套

- id: C-33
  status: superseded
  replaced_by: C-08
- id: C-34
  conclusion: S7 复评：单视频受解码供给限制；ExtractionPool 已交付（pool.py），同温批量零增益（OCR-GPU-bound）不推荐批量
  status: active
  premises: 2026-09-10 S0/S6 bench（4060）；触发①已满足
  revisit: 出现多视频批量场景 / S5 内联后解码格局变化 / 显式立项请求
  evidence: tests/golden/bench_baseline.json；log 2026-09-10-S6性能轮 §0

- id: C-36
  conclusion: 冷启动 = cuda.core 0.22s + NVRTC 0.09s + TRT 反序列化 0.39–0.44s；显式 `warmup()` 移出首个 extract（首视频 −33%），总吞吐不变；预热前移零收益已回退
  status: active
  premises: 引擎文件已缓存；首个 extract
  revisit: 换后端/引擎格式
  evidence: log §3；tests/pipeline/test_warmup.py

- id: C-37
  conclusion: 消费端提交可批量化：归约 D2H→异步+本流同步（h264-cpu −5.19%）、keep_crops D2H 并批窗 16（h264-gpu −2.45%），逐位一致；按宽分组无收益
  status: active
  premises: 本机 4060；交错 A/B
  revisit: 段密度翻倍 / ROI 形态使判据翻转
  evidence: log §4-§6；ocr_stage.effective_reorder_window

- id: C-38
  conclusion: **合并判定「稠密簇门」**（win3 ≥ SEG_C 恒不合并，默认开）净 +226 帧；代价段数 +1.2~3.6%、墙钟不变
  status: active
  premises: 本机 4060 + 六片真值；宿主/GPU 逐位一致
  revisit: 大字号码管字体 / OCR-bound 部署
  evidence: log 2026-09-12-准确项 §3 §6

- id: C-39
  conclusion: **NVML NVDEC% 是「在用」指示器非占空比**（臂 35% 忙仍读 98%）——忙闲判别用 fork `[hybrid-stats] busy`；NVML 价值=时钟+热降原因位（实测无热降）
  status: active
  premises: full 档 + fork ≥6da2957；本机单卡
  revisit: 换卡（NVDEC% 语义随驱动/卡型变）/ fork 换代
  evidence: log 2026-09-13-hybrid可见性重评；bench/hybrid_reeval.json

- id: C-40
  conclusion: **hybrid 收益与视频长度相关**：h264 无交叉点（500 帧起胜）；hevc/av1 交叉点 1500~3000 帧；短窗暖态劣势仅 0.06~0.10s。对纯 CPU 臂三码全片均胜（1.10×/2.76×/1.81×）
  status: active
  premises: 4060/16C32T；每臂独立子进程 + min-of-2（差值消 0.85s 固定开销）；ocr=trt
  revisit: 换卡（NVDEC 与 CPU 相对速率变）/ 关键帧间隔差异大的片源 / 超短片（<500 帧）
  evidence: log 2026-09-13-hybrid解码率与理论并联和 §续七

- id: C-41
  conclusion: **短窗缺口量级（已更正）**：暖态 hybrid−纯GPU臂 仅 +0.059s(w=1000)/+0.096s(w=500)，w=3000 反超；首抽差 +0.18~0.28s 系实例化非调度。播种实验证伪"盲阶段采样块"归因，已回退
  status: active
  premises: 4060/16C32T；同进程 3 次取暖态均值（首次含 engine_init 不入账）；hevc/test6_hevc；对照臂 = hybrid 内 FORCE_SIDE=gpu
  revisit: 换卡 / fork 实例化优化后重测 / 需"进程首跑"口径时单独测
  evidence: log 2026-09-13-份额旋钮修复与换DLL-A-B §四（含证伪链）

- id: C-42
  conclusion: 相似判定单次 232µs、全片≈墙钟 21%，**但不在关键路径**：短路成恒不相似（分段逐位同）wall 无改善 ⇒ 消费者空等 84~87% 完全吸收；按占比推算收益是错的
  status: active
  premises: GPU 管线；hevc 6000 帧；merges=0（短路前后段数均 2101）
  revisit: 消费者不再空等的配置（OCR 极慢 / batch 极大）/ 段边界密度大幅上升的素材 / 换卡后 GPU 争用格局变化
  evidence: log 2026-09-13-merge判定代价与短路实验；ENGINE_PROFILE 分相

- id: C-43
  conclusion: **fill_width×force_aspect 强交互**（224 保持）：fa=1.5 下 224 零误读（30664 帧），降 0 反而差；仅 fa=0 时关填充更准；fill_width=0 加法开关保留
  status: active
  premises: 六片真值；真值头记配置（test5/6_ref fa=1.5；test2 =0 fw=320）
  revisit: 换片源 / OCR 模型换代 / 默认 force_aspect 变更
  evidence: log 2026-09-13-填充宽度重测；`_probe_acc_ab.py` 已自动复刻真值头配置

- id: C-44
  status: superseded
  replaced_by: C-45

- id: C-45
  status: superseded
  replaced_by: C-46

- id: C-44
  status: superseded
  replaced_by: C-45

- id: C-45
  conclusion: **hybrid 并联缺口收口（fork 4cebeef）**：可修成分=①CPU 臂银行帽 1536 系全帧字节残留且 raw/frame 合并计价②计划冻结用 rg 爬升值（±13% 漂）→hevc 份额偏 CPU、GPU 队尾闲置。修复=queue 按 ROI 字节重算+raw 768MB 解耦+滑动视界 4096（HORIZON=0 回退）。fork 全片 hevc +12.8%/h264 +14.8%/av1 +3.0%（达成率 77→87/68→78/88→90%）；金标 28/28。残余=每臂混跑干扰（CPU busy-rate −15~17%，32T 最优）
  status: active
  premises: 4060/16C32T；fork 4cebeef；引擎干净对：hevc −4.8%（23/28）、av1 −2.9%（7/8）、h264 平价（引擎税吃掉 fork 收益）
  revisit: 换卡 / fork 换代 / >32 核 / 绑核立项
  evidence: log 2026-09-13-hybrid并联缺口收口（bench/gap_decomp.json）

- id: C-46
  conclusion: **hybrid = 包缓存 + 供料期 GOP 派工（fork fef3c4b，已随 wheel 发布）**：Push 只入压缩包缓存（512MB），泵按当下速率贪心派工 GOP（前瞻+同侧保序）；**kick 必须经泵按流序注入**（错位=IDR 冲重排窗→段数漂移，fork 级测不出）。引擎 A/B：hevc −6.29%（6/6）、h264 −4.78%（6/6）、av1 平价；fork 对并联和 **96/92/96%**（hevc/h264/av1 fps_b；同会话三臂定稿，旧 78/87 系跨会话伪影）；18/18 段数恒定、金标 28/28。旧机制（计划/视界/盲窗/REPLAN/债务克隆）全删
  status: active
  premises: 4060/16C32T；达成率=同会话三臂（wheel 6ec5b1ea）；旋钮 PKT_CACHE_MB/KICK_OFF/KICK_BURST/AV1_CPU
  revisit: 换卡 / fork 换代 / 截断流·bf16 酷刑流重跑 / >32 核
  evidence: log 2026-09-13-hybrid重设计分支 §8、2026-09-14-hybrid达成率定稿测量

- id: C-47
  conclusion: **OCR 批延迟残差=GPU 链争用主导（"launch 裸奔"归因修正）**：TRT_DEFER_SYNC 深度2延迟收集机制成立、两码逐位一致但 e2e 平价（fp32 +0.26%/fp16 −0.22% 热池符号乱）——与 CUDA Graph 同判，提交重叠可回收成分≈0；再提速只能减 GPU 工作量（fp16 已做）或减争用
  status: active
  premises: 4060/16C32T；全片逐位 + bench ab 热池；trt_call 13.9ms/批占 87%、SM ~40%
  revisit: 换卡 / TRT 换代 / 解码下 GPU
  evidence: log 2026-09-14-TRT延迟收集流水；_probe_ocr_phase_split.py
