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
| C-05 | hybrid = fork 原生（TRT→hybrid_gpu、CPU→宿主帧）。收益面=TRT 路径；ONNX 宿主不反超→选 nvdec。解码率可见性 2.45×/17%/30%；e2e 见 C-53 | 4060/16C32T；0.8.3；fork ≥6da2957 | >32 核 / 残余=每臂混跑干扰 | log 2026-09-11-hybrid差距分解 §11-16；2026-09-13-hybrid可见性重评 |
| C-07 | `auto` 不区分 OCR 后端、一律尝试 NVDEC：批量互补必须显式 `decode_backend="cpu"` 并核验 `_backend` | — | auto 实现按 ocr_backend 分叉后复核 | README 批量章；PERF §19 §21 |
| C-08 | `auto` 在 h264 非最优（CPU+TRT 快 1.7~2.8×）但**恒 NVDEC 优先是刻意决策**：弱 CPU 可能反慢+争用/功耗代价；峰值走显式 cpu | 稳妥性>峰值吞吐（用户拍板） | 「NVDEC 不可用」级前提变化 | DECISIONS A2；log 2026-09-09 深度性能优化 |
| C-09 | GPU 分段+ONNX 无净收益，GPU 管线默认只放行 NVDEC+TRT | onnxruntime 1.29 CPU ep；解码为瓶颈 | ort IO binding/新 EP；解码供给大升 | PERF §9 |
| C-10 | GPU_PIPELINE_STREAM 默认关：流水发射零实测收益，decord 侧 API 保留 opt-in | 解码仍是瓶颈（消费不反超供给） | OCR 提速使消费反超解码供给 | 提交 9b83cba；2026-09-09 复确认零收益 |
| C-13 | 真跳帧（丢 nal_ref_idc==0 整包）安全，收益仅 1.03–1.48× | h264、fork 0.7.x | 新编码 / 更激进的过滤方案 | DECISIONS「下一步三目标轮」 |
| C-14 | skip_loop_filter 1.11–1.36× 但改变像素；默认 opt-in | — | 下游证实对像素不敏感且需提速率 | DECISIONS「P0-6 翻案」 |
| C-15 | **OCR 输入侧参数都不是杠杆**。pad 下限 224 保持（160/320 均证伪）；预处理六变体仅一个非负 +7 帧；翻转系实例级边缘判决 | v6_small + racelog 内容 | pad/预处理架构变更；换模型或字体/ROI 形态 | log 2026-09-12-准确项 §4.1 |
| C-16 | OCR 裁切余量 10% 优于 0%；裁切提准确率（旧"守卫"前提错） | — | ROI 形态 / 分辨率大变 | DECISIONS「第四轮」 |
| C-31 | decord 0.8.1+FFmpeg9：seek 未变慢；av1 慢系 NT=4 假象。修：av1 线程 3/4 钳 [8,24]，CPU −50%、e2e −45% | wheel 0.8.2 | fork/驱动/FFmpeg 换代 | log 2026-09-08 decord-0.8.1；DEPENDENCIES decord 节 |
| C-32 | 分段/状态机/裁切/预处理实现唯一出处 = segmentation.py：宿主直调，GPU kernel 为设备侧逐位镜像、判据引用同一文件；不做插件抽象面（已回退） | 0.11.0；两侧行为由真值用例逐位守护 | 出现真实的 GPU 侧算法插件需求 | log 2026-09-09 引擎四方向；tests/ 全套 |
| C-36 | 冷启动 = cuda.core 0.22s + NVRTC 0.09s + TRT 反序列化 0.39–0.44s；显式 `warmup()` 移出首 extract（首视频 −33%） | 引擎已缓存；首个 extract | 换后端/引擎格式 | log §3；tests/pipeline/test_warmup.py |
| C-37 | 消费端提交可批量化：归约 D2H 异步+本流同步（−5.19%）、keep_crops 并批窗 16（−2.45%）；按宽分组无收益 | 本机 4060；交错 A/B | 段密度翻倍 / ROI 形态翻转 | log §4-§6 |
| C-38 | **合并判定「稠密簇门」**（win3≥SEG_C 恒不合并，默认开）净 +226 帧；段数 +1.2~3.6%、墙钟不变 | 六片真值；宿主/GPU 逐位一致 | 大字号字体 / OCR-bound 部署 | log 2026-09-12-准确项 §3 §6 |
| C-39 | **NVML NVDEC% 是「在用」指示器非占空比**（35% 忙仍读 98%）——忙闲用 fork busy 计数；NVML 价值=时钟+热降位 | full 档 + fork ≥6da2957；本机单卡 | 换卡（NVDEC% 语义随驱动/卡型变）/ fork 换代 | log 2026-09-13-hybrid可见性重评 |
| C-40 | **hybrid 收益与片长相关（交叉点三改）**：h264 短窗即胜 nvdec；hevc 交叉点 **~1200**（w1000 +5.4%/w1500 −6.1%/w3000 −14.5%；旧值系伪影）；av1 <3000（−16.7%）；对纯 CPU 臂三码全片均胜 | 4060/16C32T；fork ≥a6cdeb7（熟前挂起+窗口尾 ETA 派工） | 换卡 / GOP 尺寸极端小 / <500 帧短片 | log 2026-09-18-hybrid启动轮 §4 |
| C-42 | 相似判定单次 232µs、全片≈墙钟 21% 但**不在关键路径**（短路恒不相似 wall 无改善，消费者空等吸收）；按占比推算收益是错的 | GPU 管线；hevc 6000 帧；merges=0 | 消费者不再空等 / 段边界密度大增 / 换卡 | log 2026-09-13-merge判定代价与短路实验 |
| C-43 | **fill_width×force_aspect 强交互**（224 保持）：fa=1.5 下 224 零误读（30664 帧）；仅 fa=0 关填充更准；fill_width=0 开关保留 | 六片真值 | 换片源 / 模型换代 / 默认 force_aspect 变更 | log 2026-09-13-填充宽度重测 |
| C-46 | **hybrid = 包缓存+供料期 GOP 派工（fork fef3c4b）**：Push 只入 512MB 包缓存，泵按速率贪心派工 GOP；**kick 必须经泵按流序注入**（错位=段数漂移）。机制 A/B hevc −6.29%/h264 −4.78%（新旧机制对比，见 C-53）；对并联和 96/92/96% | 达成率=同会话三臂 | 换卡 / fork 换代 / 截断流·bf16 重跑 | log 2026-09-13-hybrid重设计分支 §8、2026-09-14-hybrid达成率定稿测量 |
| C-47 | **OCR 批延迟残差=GPU 链争用主导**：TRT_DEFER_SYNC 机制成立、逐位一致但 e2e 平价（与 CUDA Graph 同判）；再提速只能减 GPU 工作量/争用 | 全片逐位 + bench ab 热池；trt_call 13.9ms/批占 87%、SM ~40% | 换卡 / TRT 换代 / 解码下 GPU | log 2026-09-14-TRT延迟收集流水 |
| C-48 | **CPU OCR=OpenVINO 唯一（ORT 已移除）**：模型级 2.1×；真值零差；h264-cpu 热池 −27.96%；冻结包 73MB | 4060/Zen4；openvino 2026.3.1 | 换 CPU（非 x86）/ openvino 换代 / 内容族大变 | log 2026-09-14-OpenVINO模型级A-B/集成轮 |
| C-49 | **换依赖无剩余性能空间**：GPU 三码/h264-cpu 解码绑定；fork 超外部参考（NVDEC 989>cuvid 901fps；ROI-first 1.75×）；PyNv 493fps+DLL 冲突；cv2 −0.26% | 4060/16C32T/Zen4；fork 0.8.3 | 换卡 / 带 ROI-first 的解码绑定 / 预处理升为关键路径 | log 2026-09-15-依赖替换裁决 |
| C-50 | **DECODE_THREADS 10→32 无收益**（不可判定 +0.97%；冷启 +4.03% 8/8）。decode.batch −82% 但 GIL 争用对冲——**维持 10 档** | 16C32T；openvino CPU OCR；h264 | 核数格局变 / OCR 再提速 / decord 预取变 | log 2026-09-17-重设计 §5-§11 |
| C-51 | **监测系统新基线（2026-09-17）**：时钟门禁+ab 轮转/判定/--aa；report v6=relations+缺口派生+cycle 口径+histograms（仅 full）+fork 穿透段；子相位闭合（infer_other 92%→16%）；std 地板 0.30%（n=50 后 0.20） | 4060；共享桌面 | 换卡/机器 / 协议改即 --aa 重标 / trace 转正需锁频 | log 2026-09-17-重设计 §7 |
| C-53 | **hybrid 基线=目标码最快纯臂（口径规则）**：h264 基线=CPU（test5 +6.1% 慢→cpu；**h264same 反例 −22%：按文件 CPU 速率分界，pool 一刀切误派**）；hevc/av1 基线=NVDEC，hybrid −23.9%/−31.4%→选 hybrid（越 C-40）。**批量=逐文件 hybrid 串行**（28.61s vs pool 并发 29.3~29.7s；pool 0.15 废弃 0.16 删） | 4060/16C32T；fork ≥a6cdeb7 | 换卡 / fork 换代 | log 2026-09-18-hybrid启动轮 §4§11 |
| C-54 | **启动+窗口尾轮（fork a6cdeb7）：熟前挂起+防饿死盲派+窗口尾 ETA**——盲派 13→2、hevc 交叉点 ~1200；h264 残余=冷税 ~95ms+尾 69ms。其修法（H2D 聚合）2026-09-20 干预：机制成立（put_block −23%）无净收益（HOL 对冲、暴露未复现日）→UPLOAD_WAIT_US 默认 0 | 盲承诺下界=2 GOP | 换卡 / fork 换代 / infer 暴露复现日 | log 2026-09-18-hybrid启动轮；bench/hybrid_startup.json |
| C-55 | **池复用必须换壳**（CPython 复活对象二次死亡不触发 __del__）+放弃路径直释——旧 Y 池两层皆漏=+2.0 MiB/轮，修复后 40 轮 +0.000 | CPython；4 钉子 | PyPy 等 finalizer 语义不同 / 池契约重构 | log 2026-09-20-审计修复轮 §2 |
| C-57 | **hybrid 硬窗 × seek(start>0) fork 缺陷：尾帧 EOF 容错静默替补**（帧数守恒、尾 ~20 帧像素错；start=0 窗干净）——引擎谓词 `start<窗长` 是唯一防线 | fork 0.8.4 与 dev 均在 | fork 迟包供给专项落地后解除谓词 | log 2026-09-20-审计修复轮 §4 |
| C-58 | **hybrid OCR v0 双车道自败**：正确性成立（段数恒等/文本 3 段差），弃 raw 直通后串行宿主 resize（1.37ms/段=prep 86%）成瓶颈 → OCR-bound +51% 回归；v0.5=设备 prep 共享（留档），天花板 −15~20% | batch_test 字幕 stride=1（infer 92% 忙）；v0 opt-in 保留 | v0.5 落地后复测 | log 2026-09-20-批量策略与hybridOCR轮 §3 |

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

（dead 条目已整体降级 docs/log/，含复评触发，显式检索可达。）
