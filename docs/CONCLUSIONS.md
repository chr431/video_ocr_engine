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
| C-04 | 解码后端按编码选：h264 CPU 快 ~2.9×，AV1 反转慢 ~2.6× | fork 0.7.x 解码路径、本机核数；**0.8.1/FFmpeg9 下 av1 CPU 经济性已变**（顺序 393→1164fps @24T，见 C-31），并行/互补场景的选型数字待重测 | 并行场景重测（C-31 策略修复后） | PERF §21 §22.1；log 2026-09-08 |
| C-05 | hybrid = decord fork 原生（≥v0.7.15）：TRT 走 hybrid_gpu 显存直通、CPU OCR 走宿主帧。0.8.2 发布版：任一编码不优于最优单侧。**本地预路由+池深修复（未发布）后改观**：decode-only hybrid 近两侧理想和（av1 损耗 29.5%→8.2%），且随 dav1d 线程正常扩展（h264 ct32 3563fps 超纯 CPU 3504、hevc ct32 2559 超 NVDEC 2217）；引擎口径 hybrid ≈ 最优单侧 ±3-5%（消费端为共享瓶颈），**av1 上 hybrid 成为最快路径（1.67-1.75s vs NVDEC 1.80s）** | 本地 dll（fork 73e5540/d94d92e/3b96c6f，未推送/未发 wheel）；0.8.2 pip wheel 不含 | fork 发布 wheel 后重测；单侧速率格局再变 | log 2026-09-10-hybrid联调深挖 §K7/K8/P；tools/_probe_hybrid_sum_gap/_probe_hybrid_trace/_probe_hybrid_bitwise |
| C-07 | `auto` 不区分 OCR 后端、一律尝试 NVDEC：批量互补必须显式 `decode_backend="cpu"` 并核验 `_backend` | — | auto 实现按 ocr_backend 分叉后复核 | README 批量章；PERF §19 §21 |
| C-08 | `auto` 在 h264 多核非最优（本机 CPU+TRT 实测快 1.7~2.8×），**但 auto 恒为 NVDEC 优先是刻意决策（2026-09-10 重申）**：弱 CPU 上 h264 软解可能慢于 NVDEC，且 CPU 解码必然引入资源争用与整机功耗上升，NVDEC 稳妥优先；峰值吞吐留给显式 `decode_backend="cpu"` | 决策权衡稳妥性 > 本机峰值吞吐；机器/场景变化不改此方向（用户拍板） | 仅当出现「NVDEC 不可用」级别的前提变化 | DECISIONS 审查 A2；log 2026-09-09 深度性能优化（含 2026-09-10 勘误） |
| C-09 | GPU 分段+ONNX 无净收益，GPU 管线默认只放行 NVDEC+TRT | onnxruntime 1.29 CPU ep；解码为瓶颈 | ort 支持 IO binding / 新执行提供器；解码供给率大幅提升 | PERF §9 |
| C-10 | GPU_PIPELINE_STREAM 默认关：流水发射零实测收益，decord 侧 API 保留 opt-in | 解码仍是瓶颈（消费不反超供给） | OCR 提速使消费反超解码供给 | 提交 9b83cba / d9c96f9；2026-09-09 decord 0.8.2 复确认零收益（log 深度性能优化） |
| C-11 | 分段合并：不误合并约束下无普适空间 | 现有分段算法 + 5 真值视频 | 分段算法更换 / 真值扩容暴露误合并 | PERF §14 |
| C-13 | 真跳帧（丢 nal_ref_idc==0 整包）安全，但收益仅 1.03–1.48×（原估 2~4×） | H.264、fork 0.7.x | 新编码 / 更激进的过滤方案 | DECISIONS「下一步三目标轮」 |
| C-14 | skip_loop_filter 收益 1.11–1.36× 但改变输出像素；默认 opt-in | — | 下游证实对像素不敏感且需提速率 | DECISIONS「P0-6 翻案」 |
| C-15 | OCR pad 下限 224 保持（160 已回退：生产误读退化；320 已证伪） | — | pad / 预处理架构变更 | ARCHIVE §16.2 |
| C-16 | OCR 裁切余量 10% 优于 0%；裁切即使省不到算力也能提准确率（旧"守卫"前提是错的） | — | ROI 形态 / 分辨率大变 | DECISIONS「第四轮」 |
| C-31 | decord 0.8.1 + FFmpeg9 升级：seek 本身未变慢（两构建同斜率 ~2.6ms/帧×关键帧距离）；此前"av1 seek 变慢"是 NT=4 探针口径假象 + FFmpeg9 dav1d 线程扩展性改善被引擎旧 av1 线程策略（cores//2=8T，FFmpeg8 时代口径）埋没。修复：av1 → 逻辑核 3/4 钳 [8,24]（不分 OCR 位置），CPU 后端 5.47→2.72s（−50%）、hybrid 1.90→1.80s、host_cpu e2e 6.26→3.42s（−45%）；release 初版 yuv 路径必挂（shim 误用 v1 cuMemcpy2D 导出 → 201），须含 cuMemcpy2D_v2 修复（fork 7ef70f5） | pip wheel 0.8.2（dll md5 6597eea6，已含修复，DLL 随包自带） | fork 再升级 / 驱动或 FFmpeg 再换代 | log 2026-09-08 decord-0.8.1；tools/_ab_decord081/；DEPENDENCIES decord 节 |
| C-32 | 分段判定/状态机/裁切/预处理的实现唯一出处 = segmentation.py：宿主管线直接调用，GPU kernel 为其设备侧逐位镜像、判定阈值/余量/合并判据引用同一文件；不做插件抽象面（GPU 设备侧实现不可插拔，插件语义=降速到宿主管线，已回退） | 0.11.0；两侧行为由真值用例逐位守护 | 出现真实的 GPU 侧算法插件需求（需设备侧实现面，另立结论） | log 2026-09-09 引擎四方向；tests/ 全套 |
| C-34 | S7 复评（Q4）——重构后单视频仍受解码供给限制（h264-gpu 3.107s/3000 帧 ≈ NVDEC 1013fps 天花板,解码相位占主导）,ExtractionPool 互补配对的前提成立;但它是新增公共 API,实现待显式立项（触发条件②③未发生:无真实多视频批量场景诉求,S4.5/S5 未做故解码格局未变） | 2026-09-10 S0 bench（本机 4060/8GB）;触发条件①已满足 | 用户出现多视频批量场景 / S5 内联后解码格局变化 / 显式立项请求 | tests/golden/bench_baseline.json;knowledge/benchmarks.yaml |

## 已取代（指针）

- C-06 → C-05
- C-26 → C-05
- C-27 → ?
- C-29 → C-05
- C-33 → C-08

（dead 条目已整体降级 docs/log/，含复评触发，显式检索可达。）
