# 现役结论索引（L1，唯一规范性结论地）

> 本文件由 knowledge/render.py 从 knowledge/conclusions.md 渲染
> （人不得手写；--check 校验一致性）。状态取值：active / superseded
> （被取代，只留指针）/ dead（已降级 docs/log 历史）。规则与完整
> 说明见 conclusions.md 头部注释。

| ID | 结论 | 前提/边界 | 复评触发 | 证据 |
|----|------|-----------|----------|------|
| C-01 | 并发退化真因=NVDEC 会话数；NVDEC∥CPU 互补聚合 1.83–1.87×，双 NVDEC 仅 1.01–1.20×（引擎级复证：C-52 nv1≈nv2） | 本机单 NVDEC 单元；同视频同负载 | 多 NVDEC 单元 GPU / 驱动调度变更 | PERF §19 §21 |
| C-02 | IO 不是并发退化原因（<1% 墙钟；页缓存全命中仍退化 1.88×） | NVMe + 页缓存命中 | 冷盘/网络盘使 IO 占比抬升 | PERF §19 |
| C-03 | 内存带宽不是并发变量（B_max 实测 55.8 GB/s；互补设计仅 7.8 GB/s 退化 1.02×） | 2×16GB DDR5-6000 独显平台 | 共享内存带宽的集成平台 / 内存减半 | PERF §20 §21 |
| C-04 | 解码后端按编码选：h264 CPU 快 ~2.9×，av1 慢 ~2.6× | fork 0.7.x；av1 经济性已变（C-31）；并行已重测（C-52） | 并行场景重测（C-31 策略修复后） | PERF §21 §22.1；log 2026-09-08 |
| C-05 | hybrid = fork 原生（TRT→hybrid_gpu、CPU→宿主帧）。收益面=TRT 路径；ONNX 宿主不反超→选 nvdec。解码率可见性 h264 2.45×/hevc 17%/av1 30%；e2e 见 C-53 | 4060/16C32T；0.8.3；忙时计数 fork ≥6da2957 | >32 核复测 / C-45 已落地份额倾斜，残余=每臂混跑干扰 | log 2026-09-11-hybrid差距分解 §11-§16；2026-09-13-hybrid可见性重评 |
| C-07 | `auto` 不区分 OCR 后端、一律尝试 NVDEC：批量互补必须显式 `decode_backend="cpu"` 并核验 `_backend` | — | auto 实现按 ocr_backend 分叉后复核 | README 批量章；PERF §19 §21 |
| C-08 | `auto` 在 h264 非最优（CPU+TRT 快 1.7~2.8×）但**恒 NVDEC 优先是刻意决策**：弱 CPU 可能反慢+争用/功耗代价；峰值走显式 cpu | 稳妥性>峰值吞吐（用户拍板） | 「NVDEC 不可用」级前提变化 | DECISIONS A2；log 2026-09-09 深度性能优化 |
| C-09 | GPU 分段+ONNX 无净收益，GPU 管线默认只放行 NVDEC+TRT | onnxruntime 1.29 CPU ep；解码为瓶颈 | ort IO binding/新 EP；解码供给大升 | PERF §9 |
| C-10 | GPU_PIPELINE_STREAM 默认关：流水发射零实测收益，decord 侧 API 保留 opt-in | 解码仍是瓶颈（消费不反超供给） | OCR 提速使消费反超解码供给 | 提交 9b83cba；2026-09-09 复确认零收益 |
| C-13 | 真跳帧（丢 nal_ref_idc==0 整包）安全，收益仅 1.03–1.48× | H.264、fork 0.7.x | 新编码 / 更激进的过滤方案 | DECISIONS「下一步三目标轮」 |
| C-14 | skip_loop_filter 收益 1.11–1.36× 但改变输出像素；默认 opt-in | — | 下游证实对像素不敏感且需提速率 | DECISIONS「P0-6 翻案」 |
| C-15 | **OCR 输入侧参数都不是杠杆**。pad 下限 224 保持（160/320 均证伪）；预处理六变体仅一个非负 +7 帧；翻转系实例级边缘判决 | v6_small + racelog 内容 | pad/预处理架构变更；换模型或字体/ROI 形态 | log 2026-09-12-准确项 §4.1 |
| C-16 | OCR 裁切余量 10% 优于 0%；裁切即使省不到算力也能提准确率（旧"守卫"前提错） | — | ROI 形态 / 分辨率大变 | DECISIONS「第四轮」 |
| C-31 | decord 0.8.1+FFmpeg9：seek 未变慢；av1 慢系 NT=4 假象。修：av1 线程 3/4 钳 [8,24]，CPU −50%、e2e −45% | pip wheel 0.8.2（md5 6597eea6，DLL 随包自带） | fork 再升级 / 驱动或 FFmpeg 再换代 | log 2026-09-08 decord-0.8.1；DEPENDENCIES decord 节 |
| C-32 | 分段/状态机/裁切/预处理实现唯一出处 = segmentation.py：宿主直调，GPU kernel 为设备侧逐位镜像、判据引用同一文件；不做插件抽象面（已回退） | 0.11.0；两侧行为由真值用例逐位守护 | 出现真实的 GPU 侧算法插件需求 | log 2026-09-09 引擎四方向；tests/ 全套 |
| C-52 | **批量互补配对有真实收益（C-34 翻案）**：三码族全片 pool pair 27.0s vs 全 nvdec 34.0/34.2s=**−20.6%/−21.0%**（4/4 逐位一致）；收益主体=编码感知派工；旧判系 OCR-bound 前提已翻转 | TRT OCR + C-48 后 decode-bound；三码族（有 h264 可卸载） | 换卡 / OCR 变慢 / 素材全 hevc/av1（pair 退化为 nv2） | log 2026-09-17-重设计 §10；bench/pool_pairing.json |
| C-36 | 冷启动 = cuda.core 0.22s + NVRTC 0.09s + TRT 反序列化 0.39–0.44s；显式 `warmup()` 移出首个 extract（首视频 −33%），总吞吐不变 | 引擎已缓存；首个 extract | 换后端/引擎格式 | log §3；tests/pipeline/test_warmup.py |
| C-37 | 消费端提交可批量化：归约 D2H 异步+本流同步（h264-cpu −5.19%）、keep_crops 并批窗 16（h264-gpu −2.45%）；按宽分组无收益 | 本机 4060；交错 A/B | 段密度翻倍 / ROI 形态翻转 | log §4-§6 |
| C-38 | **合并判定「稠密簇门」**（win3≥SEG_C 恒不合并，默认开）净 +226 帧；段数 +1.2~3.6%、墙钟不变 | 六片真值；宿主/GPU 逐位一致 | 大字号字体 / OCR-bound 部署 | log 2026-09-12-准确项 §3 §6 |
| C-39 | **NVML NVDEC% 是「在用」指示器非占空比**（35% 忙仍读 98%）——忙闲用 fork busy 计数；NVML 价值=时钟+热降位 | full 档 + fork ≥6da2957；本机单卡 | 换卡（NVDEC% 语义随驱动/卡型变）/ fork 换代 | log 2026-09-13-hybrid可见性重评 |
| C-40 | **hybrid 收益与片长相关（交叉点已复测改判）**：h264 短窗即胜 nvdec；hevc 交叉点 **>3000 帧**（w3000 仍 +35% 慢，2026-09-17 硬窗界修复后复测，旧 1500~3000 系越窗税伪影）；av1 **<3000**（w3000 已 −16.3%）；对纯 CPU 臂三码全片均胜 | 4060/16C32T；硬窗界修复后口径（DecodeStats 包级直证） | 换卡 / 关键帧间隔差异大的片源 / <500 帧短片 | log 2026-09-13-hybrid解码率与理论并联和 §续七 |
| C-41 | **短窗缺口量级（已更正）**：暖态 hybrid−纯GPU 仅 +0.059s(w=1000)/+0.096s(w=500)，w=3000 反超；首抽差系实例化；"盲阶段采样块"归因已证伪回退 | 同进程 3 次热态均值（首抽不入账）；对照=FORCE_SIDE=gpu | 换卡 / fork 实例化优化后重测 | log 2026-09-13-份额旋钮修复与换DLL-A-B §四 |
| C-42 | 相似判定单次 232µs、全片≈墙钟 21% 但**不在关键路径**（短路恒不相似 wall 无改善，消费者空等吸收）；按占比推算收益是错的 | GPU 管线；hevc 6000 帧；merges=0 | 消费者不再空等的配置 / 段边界密度大增的素材 / 换卡 | log 2026-09-13-merge判定代价与短路实验 |
| C-43 | **fill_width×force_aspect 强交互**（224 保持）：fa=1.5 下 224 零误读（30664 帧）；仅 fa=0 关填充更准；fill_width=0 开关保留 | 六片真值（真值头记配置） | 换片源 / OCR 模型换代 / 默认 force_aspect 变更 | log 2026-09-13-填充宽度重测 |
| C-46 | **hybrid = 包缓存+供料期 GOP 派工（fork fef3c4b，随 wheel 发布）**：Push 只入 512MB 包缓存，泵按速率贪心派工 GOP；**kick 必须经泵按流序注入**（错位=段数漂移）。机制 A/B hevc −6.29%/h264 −4.78%（**新旧机制对比**，非 e2e——见 C-53）；对并联和 96/92/96%；18/18 段数恒定、金标 28/28 | 达成率=同会话三臂（wheel 6ec5b1ea=fef3c4b） | 换卡 / fork 换代 / 截断流·bf16 重跑 / >32 核 | log 2026-09-13-hybrid重设计分支 §8、2026-09-14-hybrid达成率定稿测量 |
| C-47 | **OCR 批延迟残差=GPU 链争用主导**：TRT_DEFER_SYNC 机制成立、逐位一致但 e2e 平价（与 CUDA Graph 同判，2026-09-17 复核维持）；再提速只能减 GPU 工作量/争用 | 全片逐位 + bench ab 热池；trt_call 13.9ms/批占 87%、SM ~40% | 换卡 / TRT 换代 / 解码下 GPU | log 2026-09-14-TRT延迟收集流水；_probe_ocr_phase_split.py |
| C-48 | **CPU OCR=OpenVINO 唯一（ORT 已移除）**：模型级 2.1×；真值零差；h264-cpu 热池 −27.96%；冻结包 73MB | 4060/Zen4；openvino 2026.3.1；仅 CPU 路径 | 换 CPU（尤其非 x86）/ openvino 换代 / 内容族大变 | log 2026-09-14-OpenVINO模型级A-B/集成轮 |
| C-49 | **换依赖无剩余性能空间**：GPU 三码解码绑定（零成本 OCR≈噪声）、h264-cpu 生产者绑定；fork 超外部参考（NVDEC 989>cuvid 901fps；ROI-first 1.75×）；PyNv 493fps+DLL 冲突；cv2 resize 墙钟 −0.26% | 4060/16C32T/Zen4；fork 0.8.3；入口 _probe_{binding,decode_ceiling,preproc_ab} | 换卡 / 带 ROI-first 的解码绑定 / 预处理升为关键路径 / 断言瓶颈前先跑绑定实验 | log 2026-09-15-依赖替换裁决 |
| C-50 | **DECODE_THREADS 10→32 无收益**（不可判定 +0.97%；冷启 +4.03% 8/8）。decode.batch −82% 但 GIL 争用对冲——**维持 10 档** | 16C32T；openvino CPU OCR；h264 | 核数格局变 / OCR 再提速 / decord 预取变 | log 2026-09-17-重设计 §5-§11 |
| C-51 | **监测系统新基线（2026-09-17）**：时钟门禁+ab 轮转/判定/--aa；report v6=relations+缺口派生+cycle 口径+TOTALS n/max+histograms（仅 full）+相位 sm_clock+fork 穿透段；子相位闭合（infer_other 92%→16%）；std 地板 0.30%（n=50 后 0.20）；trace opt-in | 4060（max3105/平台2700）；共享桌面 | 换卡/驱动 / 换机器重标 / 协议改即 --aa 重标 / trace 转正需锁频 | log 2026-09-17-重设计 §7 |
| C-53 | **hybrid 基线=目标码最快纯臂（口径规则）**：h264 基线=CPU，hybrid w3000 +39%/全片 +26% 慢（硬窗界后；暖机+慢臂结构成本）；hevc/av1 基线=NVDEC，hybrid **−23.6%/−31.2%** → **h264 选 cpu、hevc/av1 选 hybrid**；fork 遥测直通 report 'hybrid' 段。⚠️ hevc 窗口须 >3000（C-40 改判）；av1 w3000 已 −16.3% | 4060/TRT；fork 穿透版；hevc/av1 全片；h264 曾错用 nvdec 基线 | 换卡 / h264 CPU 臂速率变 / fork 换代 / OCR 变慢改瓶颈 | log 2026-09-17-重设计 §11 |

## 已取代（指针）

- C-06 → C-05
- C-11 → C-38
- C-26 → C-05
- C-27 → C-36
- C-29 → C-05
- C-33 → C-08
- C-34 → C-52
- C-44 → C-45
- C-45 → C-46

（dead 条目已整体降级 docs/log/，含复评触发，显式检索可达。）
