# 现役结论索引（L1，唯一规范性结论地）

> 本文件由 knowledge/render.py 从 knowledge/conclusions.md 渲染
> （人不得手写；--check 校验一致性）。状态取值：active / superseded
> （被取代，只留指针）/ dead（已降级 docs/log 历史）。规则与完整
> 说明见 conclusions.md 头部注释。

| ID | 结论 | 前提/边界 | 复评触发 | 证据 |
|----|------|-----------|----------|------|
| C-01 | 并发退化真因=NVDEC 会话数；NVDEC∥CPU 互补聚合 1.83–1.87×，双 NVDEC 仅 1.01–1.20×（引擎级复证：C-52 nv1≈nv2） | 本机单 NVDEC 单元；同视频同负载 | 多 NVDEC 单元 GPU / 驱动调度变更 | PERF §19 §21 |
| C-02 | IO 不是并发退化原因（<1% 墙钟；页缓存全命中仍退化 1.88×） | NVMe 页缓存命中 | 冷盘/网络盘使 IO 占比抬升 | PERF §19 |
| C-03 | 内存带宽不是并发变量（B_max 实测 55.8 GB/s；互补设计仅 7.8 GB/s 退化 1.02×） | 2×16GB DDR5-6000 | 集成显存平台 / 内存减半 | PERF §20 §21 |
| C-04 | 解码后端按编码选：h264 CPU 快 ~2.9×，av1 慢 ~2.6× | fork 0.7.x；av1 经济性已变（C-31）；并行已重测（C-52） | 并行场景重测（C-31 策略修复后） | PERF §21 §22.1；log 2026-09-08 |
| C-05 | hybrid=fork 原生（TRT→hybrid_gpu、CPU→宿主帧）；收益面=TRT；ONNX 不反超→选 nvdec；可见性 2.45×/17%/30%；e2e 见 C-53 | 4060/16C32T；0.8.3；fork ≥6da2957 | >32 核 / 残余=每臂混跑干扰 | log 09-11-hybrid差距分解；09-13 可见性 |
| C-07 | auto 一例试 NVDEC；批量互补须显式 `cpu` 并核验 `_backend` | — | auto 实现按 ocr_backend 分叉后复核 | README 批量章；PERF §19 |
| C-08 | h264 上 CPU 快 1.7~2.8× 但**恒 NVDEC 优先是刻意决策**（稳妥>峰值）；峰值走显式 cpu | 稳妥性>峰值吞吐（用户拍板） | 「NVDEC 不可用」级前提变化 | DECISIONS A2；09-09 深度优化 |
| C-09 | GPU 分段+ONNX 无净收益，GPU 管线默认只放行 NVDEC+TRT | onnxruntime 1.29 CPU ep；解码为瓶颈 | ort IO binding/新 EP；解码供给大升 | PERF §9 |
| C-10 | GPU_PIPELINE_STREAM 默认关：流水发射零实测收益，decord 侧 API 保留 opt-in | 解码仍是瓶颈（消费不反超供给） | OCR 提速使消费反超解码供给 | 提交 9b83cba；09-09 复确认 |
| C-13 | 真跳帧（丢 nal_ref_idc==0 整包）安全，收益仅 1.03–1.48× | h264、fork 0.7.x | 新编码 / 更激进的过滤方案 | DECISIONS 三目标轮 |
| C-14 | skip_loop_filter 1.11–1.36× 但改变像素；默认 opt-in | — | 下游证实对像素不敏感且需提速率 | DECISIONS P0-6 翻案 |
| C-15 | **OCR 输入侧参数都不是杠杆**。pad 下限 224 保持（160/320 证伪）；预处理六变体仅一非负 +7 帧；翻转系实例级判决 | v6_small+racelog | pad/预处理架构变更；换模型或字体/ROI 形态 | log 09-12-准确项 §4.1 |
| C-16 | OCR 裁切余量 10% 优于 0%；裁切提准确率（旧"守卫"前提错） | — | ROI 形态 / 分辨率大变 | DECISIONS 第四轮 |
| C-31 | decord 0.8.1+FFmpeg9：seek 未变慢；av1 慢系 NT=4 假象；修=av1 钳 [8,24]，CPU −50%、e2e −45% | wheel 0.8.2 | fork/驱动/FFmpeg 换代 | log 2026-09-08 decord-0.8.1；DEPENDENCIES decord 节 |
| C-32 | 实现唯一出处=segmentation.py：宿主直调，GPU kernel 逐位镜像、判据同文件；不做插件抽象面（已回退） | 0.11.0；两侧行为由真值用例逐位守护 | 真实 GPU 侧插件需求出现 | log 2026-09-09 引擎四方向 |
| C-36 | 冷启动=cuda.core 0.22+NVRTC 0.09+TRT 0.39–0.44s；`warmup()` 移出首 extract（−33%） | 引擎已缓存；首个 extract | 换后端/引擎格式 | log §3；test_warmup |
| C-37 | 消费端提交可批量化：归约 D2H 异步+本流同步（−5.19%）、keep_crops 并批窗 16（−2.45%）；按宽分组无收益 | 本机 4060；交错 A/B | 段密度翻倍 / ROI 形态翻转 | log §4-§6 |
| C-38 | **合并判定稠密簇门**（win3≥SEG_C 恒不合并，默认开）+226 帧；段数 +1.2~3.6% | 六片真值；宿主/GPU 逐位一致 | 大字号字体 / OCR-bound 部署 | log 09-12-准确项 §3 |
| C-39 | **NVML NVDEC%=在用指示器非占空比**（35% 忙仍读 98%）；忙闲用 fork busy 计数，NVML 只管时钟/热 | full 档 + fork ≥6da2957；本机单卡 | 换卡（NVDEC% 语义随卡变）/ fork 换代 | log 2026-09-13-hybrid可见性重评 |
| C-40 | **hybrid 收益与片长相关（交叉点三改）**：h264 短窗即胜 nvdec；hevc 交叉点 ~1200（w1000 +5.4%→w3000 −14.5%）；av1 <3000（−16.7%）；纯 CPU 臂三码全片均胜 | 4060/16C32T；fork ≥a6cdeb7（熟前挂起+窗口尾 ETA 派工） | 换卡 / GOP 尺寸极端小 / <500 帧短片 | log 09-18-hybrid启动轮 §4 |
| C-42 | 相似判定 232µs、全片≈21% 但**不在关键路径**（短路实验 wall 无改善）；按占比推算收益是错的 | GPU 管线；hevc 6000 帧；merges=0 | 消费者不再空等 / 段边界密度大增 / 换卡 | log 2026-09-13-merge判定代价与短路实验 |
| C-43 | **fill_width×force_aspect 强交互**（224 保持）：fa=1.5 下 224 零误读（30664 帧）；仅 fa=0 关填充更准 | 六片真值 | 换片源 / 模型换代 / 默认 force_aspect 变更 | log 2026-09-13-填充宽度重测 |
| C-46 | **hybrid=包缓存+供料期 GOP 派工（fork fef3c4b）**：Push 只入 512MB 缓存，泵按速率贪心派工；**kick 必须经泵按流序注入**（错位=段数漂移）；机制 A/B hevc −6.29%/h264 −4.78% | 达成率=同会话三臂 | 换卡 / fork 换代 / 截断流·bf16 重跑 | log 2026-09-13-hybrid重设计分支；2026-09-14 达成率定稿 |
| C-47 | **OCR 批延迟残差=GPU 链争用主导**：DEFER_SYNC 机制成立、逐位一致但 e2e 平价；提速只能减 GPU 争用 | 逐位+ab 热池；trt_call 13.9ms/批占 87% | 换卡 / TRT 换代 / 解码下 GPU | log 2026-09-14-TRT延迟收集流水 |
| C-48 | **CPU OCR=OpenVINO 唯一**：模型级 2.1×；真值零差；热池 −27.96%；冻结包 73MB | 4060/Zen4；openvino 2026.3.1 | 换 CPU（非 x86）/ openvino 换代 / 内容族大变 | log 2026-09-14-OpenVINO模型级A-B |
| C-49 | **换依赖无剩余性能空间**：GPU 三码/h264-cpu 解码绑定；fork 超外部参考（989>901fps；ROI-first 1.75×）；PyNv 493fps+DLL 冲突；cv2 −0.26% | 4060/16C32T/Zen4；fork 0.8.3 | 换卡 / 带 ROI-first 的解码绑定 / 预处理升为关键路径 | log 2026-09-15-依赖替换 |
| C-50 | **DECODE_THREADS 10→32 无收益**（+0.97% 不可判定；冷启 +4.03%）。decode.batch −82% 被 GIL 对冲——**维持 10 档** | 16C32T；openvino CPU OCR；h264 | 核数格局变 / OCR 再提速 / decord 预取变 | log 2026-09-17-重设计 §5-§11 |
| C-51 | **监测系统新基线（2026-09-17）**：时钟门禁+ab 轮转/判定/--aa；report v6（relations/缺口派生/histograms/fork 穿透）；子相位闭合（infer_other 92%→16%）；std 地板 0.30%（n=50 后 0.20） | 4060；共享桌面 | 换卡/机器 / 协议改即 --aa 重标 / trace 转正需锁频 | log 2026-09-17-重设计 §7 |
| C-53 | **hybrid 基线=目标码最快纯臂**：h264=CPU（+6.1% 慢；h264same −22% 反例=按文件分界）；hevc/av1=NVDEC，hybrid −23.9%/−31.4%→选 hybrid。批量=逐文件串行（28.61s vs pool 29.3~29.7；pool 0.15 废弃 0.16 删） | 4060/16C32T；fork ≥a6cdeb7 | 换卡 / fork 换代 | log 09-18-hybrid启动轮 |
| C-54 | **启动+窗口尾轮（fork a6cdeb7）：熟前挂起+防饿死盲派+窗口尾 ETA**——盲派 13→2、hevc 交叉点 ~1200；h264 残余=冷税 ~95ms+尾 69ms；H2D 聚合修法机制成立（put_block −23%）无净收益→默认关 | 盲承诺下界=2 GOP | 换卡 / fork 换代 / infer 暴露复现日 | log 09-18-hybrid启动轮；bench/hybrid_startup.json |
| C-55 | **池复用必须换壳**（复活对象二次死亡不触发 __del__）；旧 Y 池两层漏 +2.0 MiB/轮→修复后 +0.000 | CPython；4 钉子 | PyPy finalizer 语义 / 池契约重构 | log 2026-09-20-审计修复轮 §2 |
| C-58 | **hybrid OCR 双车道（TRT+OV）v0.5：回归已消除（+48%→−1.8%）但本机无净收益**——容量理论 +54% 实测 +18%（OV 产能=TRT 1/3；争用 +20%；CPU 臂无余量）；触发条件实测见 C-60 | batch_test 字幕 stride=1；16C32T 三层占满 | NVDEC 纯解码部署 / 多 NVDEC 卡 / OCR 变重 | log 09-20 批量策略轮 §3-4 |
| C-59 | 宽 ROI 字幕：裁切文本效应=临界字形宽度彩票（C639≡C 不翻、C781 纯 pad 翻 14 段），随集波动（ep01 −7/ep02 +40）；性能真收益（墙钟 −6.6% 同窗口/−9.0% vs 旧默认）。默认维持；L2 同属宽度扰动 | 新三国01/02 stride=1；dbe=cpu+TRT；真值视觉+抽帧复核 | 换模型 / PAD_SMALL×裁切联调 / NVDEC 纯解码 / dbe 变更 | log 2026-09-20-裁切复测轮 |
| C-60 | hybrid OCR 奖金池负结果：CPU 空闲已兑现（hevc NVDEC 纯解码 7600fps=墙钟 15%）双车道仍只兑 −4.9%±0.1（TRT 劣化 ~28%[GPU 共享]+OV 1/3 短板），双双 hybrid >15× 病理超时；动态分配不立项——瓶颈不在可分配资源；文本 B≡A 0/48054 | hevc 转码集整集 3 轮交错；阈值 5% 预注册 | OV 产能>TRT 1/2 / GPU 分离部署（解码/OCR 异卡） | log 2026-09-20-hybridOCR奖金池裁决 |
| C-62 | **硬窗架构重做（fork 0.9.0）：绝对帧区间 + marker 语义分离（0=EOF/1=WINDOW_END）+ 会话对象化（SessionState 整体重建）+ 窗模式禁替补（win_subs 结构性恒 0，缺帧响亮 FATAL）**——结构性根除 C-57 一族四根源（eof_pushed_ 语义过载/跨类计数算术/手写重置清单/替补放大器）。矩阵 12/12 × 5 轮 + 哈希与旧架构逐位一致 + fork 七套件（新增 window 套件） | fork 0.9.0 dev（52597677）；金标 D 组晚起点 5 用例 | fork 换代/窗口路径改动（matrix --repeat 5 + window 套件）/换卡/竞态再现（轮盘续钻） | log 2026-10-08 收口与窗口架构重做 |
| C-63 | **周期/指令账本可作 A/B 判据**：引擎负载下 QueryProcessCycleTime 同条件 CV≤2.3%（满载墙钟 CV 30-36% 对照）、nvdec 臂跨负载守恒 0.965；ncu sm__inst_executed 24 次 launch 散布 2.7e-6——GPU 代码变更 n=1 采集即可分辨 0.01% 级。边界：SMT 争用均值带 ±16%→交错仍需；BLAS/自旋库满载 +139% 不可用；ncu 需管理员+重放扰动=离线专用 | 本机 4060/16C32T/Win32；ncu 2026.2.0；引擎栈 decord+OpenVINO 无自旋放大 | 换机/换 CPU；BLAS/OV 线程模型换代；bench 判据集成前先实战一轮 A/B | log 2026-10-09-周期计数测量首轮 |

## 已取代（指针）

- C-06 → C-05
- C-11 → C-38
- C-26 → C-05
- C-27 → C-36
- C-29 → C-05
- C-33 → C-08
- C-34 → C-52
- C-52 → C-53
- C-41 → C-54
- C-44 → C-45
- C-45 → C-46
- C-57 → C-61
- C-61 → C-62

（dead 条目已整体降级 docs/log/，含复评触发，显式检索可达。）
