# 现役结论索引（L1，唯一规范性结论地）

> 本文件由 knowledge/render.py 从 knowledge/conclusions.md 渲染
> （人不得手写；--check 校验一致性）。状态取值：active /
> superseded（被取代）/ dead。superseded 与 dead 全文在
> docs/log/2026-09-10-conclusions-history.md（append-only，显式
> 检索可达；2026-10-09 起 superseded 不再留源内指针）。规则与
> 完整说明见 conclusions.md 头部注释。

| ID | 结论 | 前提/边界 | 复评触发 | 证据 |
|----|------|-----------|----------|------|
| C-01 | 并发退化真因=NVDEC 会话数；NVDEC∥CPU 互补聚合 1.83–1.87×，双 NVDEC 仅 1.01–1.20× | 本机单 NVDEC 单元；同视频同负载 | 多 NVDEC 单元 GPU / 驱动调度变更 | PERF §19 §21 |
| C-02 | IO 不是并发退化原因（<1% 墙钟；页缓存全命中仍退化 1.88×） | NVMe 页缓存命中 | 冷盘/网络盘使 IO 占比抬升 | PERF §19 |
| C-03 | 内存带宽不是并发变量（B_max 55.8 GB/s；互补退化 1.02×） | 2×16GB DDR5-6000 | 集成显存平台 / 内存减半 | PERF §20 §21 |
| C-04 | 解码后端按编码选：h264 CPU 快 ~2.9×，av1 慢 ~2.6× | fork 0.7.x；av1 见 C-31 | 并行场景重测（C-31 策略修复后） | PERF §21 §22.1；log 2026-09-08 |
| C-05 | hybrid=fork 原生（TRT→hybrid_gpu）；收益面=TRT；ONNX 不反超→选 nvdec；可见性 2.45×；e2e 见 C-53 | 4060/16C32T；0.8.3；fork ≥6da2957 | >32 核 / 残余=每臂混跑干扰 | log 09-11-hybrid差距分解；09-13 可见性 |
| C-07 | auto 一例试 NVDEC；批量互补须显式 `cpu` 并核验 `_backend` | — | auto 实现按 ocr_backend 分叉后复核 | README 批量章；PERF §19 |
| C-08 | h264 CPU 快 1.7~2.8× 但**恒 NVDEC 优先=刻意决策**；峰值走显式 cpu | 稳妥性>峰值吞吐（用户拍板） | 「NVDEC 不可用」级前提变化 | DECISIONS A2；09-09 深度优化 |
| C-09 | GPU 分段+ONNX 无净收益，GPU 管线默认只放行 NVDEC+TRT | onnxruntime 1.29 CPU ep；解码为瓶颈 | ort IO binding/新 EP；解码供给大升 | PERF §9 |
| C-10 | GPU_PIPELINE_STREAM 默认关（流水发射零收益；decord API 保留） | 解码仍是瓶颈（消费不反超供给） | 消费反超解码供给 | 提交 9b83cba；09-09 复确认 |
| C-13 | 真跳帧（丢 nal_ref_idc==0 整包）安全，收益仅 1.03–1.48× | h264、fork 0.7.x | 新编码/更激进过滤 | DECISIONS 三目标轮 |
| C-14 | skip_loop_filter 1.11–1.36× 但改变像素；默认 opt-in | — | 下游证实对像素不敏感且需提速率 | DECISIONS P0-6 翻案 |
| C-15 | **OCR 输入侧参数都不是杠杆**。pad 下限 224 保持（160/320 证伪）；预处理六变体仅一非负；翻转系实例级判决 | v6_small+racelog | pad/预处理架构变更；换模型或字体/ROI 形态 | log 09-12-准确项 §4.1 |
| C-16 | OCR 裁切余量 10% 优于 0%；裁切提准确率（旧"守卫"前提错） | — | ROI 形态 / 分辨率大变 | DECISIONS 第四轮 |
| C-31 | decord 0.8.1：seek 未变慢；av1 慢系 NT=4 假象；修=钳[8,24]：CPU −50%、e2e −45% | wheel 0.8.2 | fork/驱动/FFmpeg 换代 | log 2026-09-08 decord-0.8.1；DEPENDENCIES decord 节 |
| C-32 | 实现唯一出处=segmentation.py：宿主直调，GPU kernel 逐位镜像、判据同文件；不做插件抽象面 | 0.11.0；两侧行为由真值用例逐位守护 | 真实 GPU 侧插件需求出现 | log 2026-09-09 引擎四方向 |
| C-36 | 冷启动=cuda.core 0.22+NVRTC 0.09+TRT 0.4s；warmup 移出首 extract −33% | 热池；首 extract | 换后端/引擎格式 | log §3；test_warmup |
| C-37 | 消费端提交可批量化：D2H 异步+本流同步 −5.19%、keep_crops 并批窗 16 −2.45%；按宽分组无收益 | 本机 4060；交错 A/B | 段密度翻倍 / ROI 形态翻转 | log §4-§6 |
| C-38 | **合并判定稠密簇门**（win3≥SEG_C，默认开）；段数 +1.2~3.6% | 六片真值；宿主/GPU 逐位一致 | 大字号字体 / OCR-bound 部署 | log 09-12-准确项 §3 |
| C-39 | **NVML NVDEC%=在用指示器非占空比**（35% 忙仍读 98%）；忙闲用 fork busy 计数 | full 档 + fork ≥6da2957；本机单卡 | 换卡（NVDEC% 语义随卡变）/ fork 换代 | log 2026-09-13-hybrid可见性重评 |
| C-40 | **hybrid 收益与片长相关**：h264 短窗即胜 nvdec；hevc 交叉点 ~1200；av1 <3000（−16.7%）；纯 CPU 臂三码全片均胜 | 4060/16C32T；fork ≥a6cdeb7 | 换卡 / GOP 尺寸极端小 / <500 帧短片 | log 09-18-hybrid启动轮 §4 |
| C-42 | 相似判定 232µs、全片≈21% 但**不在关键路径**（短路实验无改善） | GPU 管线；hevc 6000 帧；merges=0 | 消费者不再空等 / 段边界密度大增 / 换卡 | log 2026-09-13-merge判定代价与短路实验 |
| C-43 | **fill_width×force_aspect 强交互**（224 保持）：fa=1.5 下 224 零误读；仅 fa=0 关填充更准 | 六片真值 | 换片源 / 模型换代 / 默认 force_aspect 变更 | log 2026-09-13-填充宽度重测 |
| C-46 | **hybrid=包缓存+供料期 GOP 派工（fork fef3c4b）**：Push 只入缓存，泵按速率贪心派工；kick 必须经泵按流序注入（错位=段数漂移）；机制 A/B −5~6% | 达成率=同会话三臂 | 换卡 / fork 换代 / 截断流·bf16 重跑 | log 2026-09-13-hybrid重设计分支；2026-09-14 达成率定稿 |
| C-47 | **OCR 批延迟残差=GPU 链争用主导**：DEFER_SYNC 机制成立、逐位一致但 e2e 平价；提速只能减 GPU 争用。复测：平价复确认（+0.33%±0.32%）但 e2e 周期 +4.6%（16/16）——争用缓解时可能浮出 | trt_call 13.9ms/批占 87% | 换卡 / TRT 换代 / 解码下 GPU | log 2026-09-14-TRT延迟收集流水；2026-10-09-账本产品化 |
| C-48 | **CPU OCR=OpenVINO 唯一**：模型级 2.1×；真值零差；热池 −27.96%；冻结包 73MB | 4060/Zen4；openvino 2026.3.1 | 换 CPU（非 x86）/ openvino 换代 / 内容族大变 | log 2026-09-14-OpenVINO模型级A-B |
| C-49 | **换依赖无剩余性能空间**：GPU 三码/h264-cpu 解码绑定；fork 超外部参考 989fps；PyNv 493fps+DLL 冲突；cv2 −0.26% | 4060/16C32T/Zen4；fork 0.8.3 | 换卡 / 带 ROI-first 的解码绑定 / 预处理升为关键路径 | log 2026-09-15-依赖替换 |
| C-50 | **DECODE_THREADS（CPU-OCR 口径）维持 auto=10**：复测：d32 墙钟 −1.86%±0.38%（14/16）但 e2e 周期 +13.4%（decode −16.4%）=争用税——共享桌面不值。GPU-OCR 口径 auto=32 已最优 | 16C32T；CPU OCR；h264 n=16 配对 | 独占部署（CPU 空闲换墙钟可接受）/ 核数格局变 / OCR 再提速 / decord 预取变 | log 2026-09-17-重设计 §5-§11；2026-10-09-账本产品化 |
| C-51 | **监测系统新基线（2026-09-17）**：时钟门禁+ab 轮转/判定/--aa；report（含 fork 穿透）；子相位闭合（infer_other 92%→16%）；地板 0.30%（n=50 后 0.20）；0.22.0 两档化后 telemetry-check=off vs full 单限（D30） | 4060；共享桌面 | 换卡/机器 / 协议改即 --aa 重标 / trace 转正需锁频 | log 2026-09-17-重设计 §7 |
| C-53 | **hybrid 基线=目标码最快纯臂**：h264=CPU（+6.1% 慢）；hevc/av1=NVDEC，hybrid −23.9%/−31.4%→选 hybrid。批量=逐文件串行（28.61 vs 29.3s；pool 已删） | 4060/16C32T；fork ≥a6cdeb7 | 换卡 / fork 换代 | log 09-18-hybrid启动轮 |
| C-54 | **启动+窗口尾轮（fork a6cdeb7）：熟前挂起+防饿死盲派+窗口尾 ETA**——盲派 13→2、hevc 交叉点 ~1200；H2D 聚合机制成立无净收益→默认关 | 盲承诺下界=2 GOP | 换卡 / fork 换代 / infer 暴露复现日 | log 09-18-hybrid启动轮；bench/hybrid_startup.json |
| C-55 | **池复用必须换壳**（复活对象二次死亡不触发 __del__）；旧 Y 池漏 +2.0 MiB/轮→修复 +0.000 | CPython；4 钉子 | PyPy finalizer 语义 / 池契约重构 | log 2026-09-20-审计修复轮 §2 |
| C-58 | **hybrid OCR 双车道（TRT+OV）v0.5：回归已消除（+48%→−1.8%）但本机无净收益**（OV 产能=TRT 1/3；争用 +20%）；触发条件实测见 C-60 | batch_test 字幕 stride=1；16C32T 三层占满 | NVDEC 纯解码部署 / 多 NVDEC 卡 / OCR 变重 | log 09-20 批量策略轮 §3-4 |
| C-59 | 宽 ROI 字幕：裁切文本效应=临界字形宽度彩票（C639 不翻/C781 翻），随集波动（ep01 −7/ep02 +40）；性能真收益 −6.6~−9.0%。默认维持 | 新三国01/02；dbe=cpu+TRT；视觉+抽帧复核 | 换模型 / PAD_SMALL×裁切联调 / NVDEC 纯解码 / dbe 变更 | log 2026-09-20-裁切复测轮 |
| C-60 | hybrid OCR 奖金池负结果：CPU 空闲已兑现（hevc NVDEC 纯解码 7600fps）双车道仍只兑 −4.9%±0.1（TRT 劣化 ~28%+OV 1/3 短板）；动态分配不立项——瓶颈不在可分配资源 | hevc 转码集整集 3 轮交错；阈值 5% 预注册 | OV 产能>TRT 1/2 / GPU 分离部署（解码/OCR 异卡） | log 2026-09-20-hybridOCR奖金池裁决 |
| C-62 | **硬窗架构重做（fork 0.9.0）：绝对帧区间 + marker 语义分离（0=EOF/1=WINDOW_END）+ 会话对象化（SessionState 整体重建）+ 窗模式禁替补（win_subs 结构性恒 0，缺帧响亮 FATAL）**——根除 C-57 族四根源。矩阵 12/12×5 + 哈希逐位一致 + 七套件 | fork 0.9.0 dev（52597677）；金标 D 组晚起点 5 用例 | fork 换代/窗口路径改动（matrix --repeat 5 + window 套件）/换卡/竞态再现（轮盘续钻） | log 2026-10-08 收口与窗口架构重做 |
| C-63 | **周期/指令账本可作 A/B 判据**：引擎负载下 QueryProcessCycleTime 同条件 CV≤2.3%（满载墙钟 30-36%）、nvdec 臂守恒 0.965；ncu 指令散布 2.7e-6（n=1 分辨 0.01%）。边界：SMT ±16%→交错仍需；BLAS 自旋库满载 +139% 不可用；ncu 需管理员=离线专用 | 本机 4060/16C32T/Win32；ncu 2026.2.0；引擎栈无自旋放大 | 换机/换 CPU；BLAS/OV 线程模型换代 | log 2026-10-09-周期计数测量首轮 |
| C-64 | **周期账本产品化（report v7 + bench 周期判读）**：per_phase 行含原始 cycles + cycles_e2e；`bench ab` 自动周期判读（同款纪律，不改退出码）。A/A：cyc:e2e 下限 1.58% vs 墙钟 2.85%；墙钟不可判定时的第二意见 | Windows；交错仍需（SMT 带）；C-63 边界全部继承 | 换机/换 CPU；周期 A/A 带漂移超 1×SE 重标；BLAS/OV 线程模型换代 | log 2026-10-09-账本产品化与不可判定复测 |
| C-65 | **跨线程账本（v8）+ 预处理容量**：full 档逐线程占空（宿主 ocr 0.20/infer 0.61；GPU 臂宿主≤0.11、NVDEC 99%、SM 16%）。预处理重算法预算（绑定实验）：宿主臂膝点=额外 0.5-1ms/crop（当前 0.30→2.5-3×）后 1:1 传导；GPU 臂 SM 余量 ~5×。duty 的 4× 被膝点修正（C-42）；更重≠更准（C-15） | 本机 4060/16C32T；test5 600 帧；sleep 注入；ALL_ACCESS 句柄怪癖见 resources 注 | 换机/换卡；OCR 提速（余量收窄）；真上重算法前复核膝点 | log 2026-10-09-跨线程账本与预处理容量 |
| C-66 | **hybrid 缺口 −5.6/−33/−34%（3000 帧）=CPU 臂欠配**：GPU 臂无损；assigned==delivered→非供给/计算/引擎耦合——泵 GOP 贪心近视（EWMA 冷启动锁死早期分配，av1 仅 1/10）。修法=速率比例预留/再平衡（kick 已有未触发）；短窗次要未做 | 4060/16C32T；fork 0.9.2；test6；v8/v10 账本 | 引擎耦合归因（nsys/ETW）；fork 泵改动后复跑；换卡 | log 2026-10-09/10 争用定位与三码矩阵 |

（superseded/dead 条目全文在 docs/log/2026-09-10-conclusions-history.md，含复评触发，显式检索可达。）
