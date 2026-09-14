# tools/ 索引

`tools/` 现有 **100 个 `.py`**（16,870 行），其中 90 个是探针
（`_probe_*`）。本文件只做**索引**，**不移动任何文件** —— 理由见下节（有实测依据）。

> 本索引的每个数字都由 `python tools/_probe_index_audit.py` 核对（退出码非 0
> 即不一致）。改了 `tools/` 之后跑一次，别让索引漂成第二份过期文档。

## ⛔ 为什么不按子目录拆分 `tools/`

量过了，移动的成本远大于收益：

| 成本项 | 实测数 |
|---|---|
| 文档中写死 `tools/_probe_*.py` 路径的引用 | **70 处** |
| 探针之间的 `from _probe_X import ...`（库型依赖） | **6 处** |
| 探针被别的探针当子进程 worker 调用 | 3 处（`_probe_mp_scale` ← `_probe_mem_bw`；`_probe_mem_bw` ← round4 两个驱动） |
| 探针内硬编码 `sys.path.insert(0, r"D:\Repo\video_ocr_engine\tools")` | **33 / 42** |

移动一个探针 = 改文档 + 改依赖它的探针 + 改它自己硬编码的 `sys.path`。
**结论：原地不动，用本索引解决"找不到"的问题。** 真要清理，先按下面
「孤儿」一节逐个案确认无引用，再单独删。

## A. 产品 / 常备工具（随仓库发布）

| 文件 | 行 | 用途 | 引用 |
|---|---:|---|---|
| `e2e_smoke.py` | 368 | 端到端冒烟 / 真值验证（真实视频） | README「测试」节 |
| `bench_hybrid.py` | 171 | hybrid 解码基准 | PERF §4 |
| `probe_decode_rates.py` | 125 | 各后端解码速率探测 | — |
| `_probe_perf_baseline.py` | 189 | 多轮性能优化的**基线/对比驱动**：视频×后端×管线全矩阵，含预热、timing 分相、ENGINE_PROFILE、唯一文本 sha 门禁与 `--compare` | docs/log/2026-09-09-深度性能优化.md |
| `_probe_hybrid_axis.py` | 104 | hybrid 增益逐轴复现（bench口径→+ROI→+引擎线程→+批64），定位 decord 侧增益在引擎口径下的存活层；首次暴露 av1 hybrid close 崩溃 | docs/log/2026-09-10-hybrid联调深挖.md |
| `_probe_hybrid_engine_loss.py` | 73 | hybrid 引擎内流失定位（stream/批大小/轻消费对照 + ENGINE_PROFILE 分相） | docs/log/2026-09-10-hybrid联调深挖.md |
| `_probe_hybrid_sum_gap.py` | 81 | hybrid 与两侧解码器速率之和的精确差值（三速率中位 + 理想和对比） | docs/log/2026-09-10-hybrid联调深挖.md |
| `_probe_hybrid_trace.py` | 109 | hybrid 调度轨迹剖析（DECORD_HYBRID_DEBUG 分侧发射时间线/份额/慢批定位） | docs/log/2026-09-10-hybrid联调深挖.md |
| `_probe_hybrid_bitwise.py` | 72 | hybrid vs NVDEC 输出逐帧字节比对（深 prefetch 安全性 + fork 改动的正确性门禁） | docs/log/2026-09-10-hybrid联调深挖.md |
| `_probe_onnx_dcd_sweep.py` | 80 | ONNX OCR 场景解码线程数 sweep（DECODE_THREADS；h264/hevc/av1 × stride），产出 2026-09-10 新档位表 | docs/log/2026-09-10-ONNX解码线程档位.md |
| `_probe_perf_sweep.py` | 118 | 解码参数 sweep（batch/stream/threads/hybthreads），monkey-patch 模块常量；用于 C-10 复确认与 batch=32 越界 bug 的暴露 | docs/log/2026-09-09-深度性能优化.md |
| `_probe_index_audit.py` | 277 | 核对本索引的每个数字是否与磁盘一致 | 本文件（自检） |
| `_probe_discipline_audit.py` | 612 | **项目纪律审计**（12 项：硬编码路径 / 异常吞噬 / 未用 import / 未门控 print / 版本号 / 文档引用 / 注入预算…） | AGENTS.md「纪律与自动化守卫」 |
| `_doc_section.py` | 214 | **文档章节级检索**：`--toc` 看目录 / `--find` 按标题定位 / 读单章。避免整文件读，实测省 84~96% tokens | AGENTS.md「查文档前先定位」 |
| `_probe_roi_decode.py` | 105 | **否定结果**：量化「打开时 SetRoi」vs「每次 get_batch 传 roi」对 CPU 软解速率的影响。实测两者无差异（1841 vs 1849 fps），但**不传 ROI = 520 fps**（3.6× 慢）→ ROI 本身是巨大优化，两种传法等价 | PERF §22.5 |
| `_probe_nvdec_interference.py` | 141 | **推翻前一轮归因**：隔离测 NVDEC 对 CPU 软解的干扰，A 单跑 / B ∥NVDEC / C ∥忙等线程（对照）。实测 NVDEC 只造成 **−3.3%**，而等量纯抢核 **−41.9%** → 元凶是分段/OCR 流水线，不是 NVDEC | PERF §22.6 |
| `_probe_decode_contention.py` | 195 | **五组完整对照**把拖慢源分层：A 单跑 / B ∥第二路 CPU 解码 / C ∥NVDEC / D ∥忙等 / E ∥带宽 hog。B(−42%)≈D(−40%)，而 C 仅 −4.2%、E 仅 −13.8% → 第二路 CPU 解码代价远大于 NVDEC（**注**：§22.9 已用 CPU profile 推翻"host CPU 算力饱和"这一解释，本探针的 A~F 组数据仍有效）
| `_probe_hybrid_cpu_profile.py` | 130 | **直接测 CPU 占用**（不再反推）。实测 hybrid 进程只用 **2~3 核（峰值 10~12）**，纯 CPU 后端吃满 **17~21 核** → "算力争用"不成立，hybrid 是**空转**不是争用 | PERF §22.9 |
| `_run_with_switchinterval.py` | 34 | 在指定 `sys.setswitchinterval` 下跑 bench_hybrid 的 runner。**证伪 GIL 争抢**：调小间隔反而更慢（2.887→3.077s）→ 瓶颈不是"等 GIL" | PERF §22.10 |
| `_probe_roadmap_decode.py` | 101 | 路线图轮解码矩阵：三编码 × {cpu(nm sweep), nvdec, hybrid(sweep)} 顺序吞吐，双 dll 口径（pip wheel vs fork build-081fix）；ROI/批 64 口径与 `_probe_hybrid_sum_gap.py` 对齐 | 本轮路线图（2026-09-10） |
| `_probe_roadmap_ocr.py` | 191 | 路线图轮 OCR rec 微基准：ONNX/TRT × 批大小扫描 + `_resize_norm` 单帧成本 + FP32/FP16×max_b 实验引擎构建（TRT 11 无 FP16 builder flag 的实证） | 本轮路线图（2026-09-10） |
| `_probe_roadmap_profile.py` | 62 | 路线图轮 ENGINE_PROFILE 分相打印驱动（单配置一次 extract，输出 producer.*/ocr.* 全分相） | 本轮路线图（2026-09-10） |
| `_probe_r3_infer_split.py` | 58 | R3 调查：hybrid vs nvdec 的 OCR worker infer 差异分解（ENGINE_PROFILE + TRT SUBPROBE 双跑） | 路线图执行轮（2026-09-10） |
| `bench.py` | 686 | **S6 性能轮的原生度量入口**（§8.6 N-4）：`run` 跑配置矩阵并落 `bench/registry.jsonl`、`diff` 按 D10 双档逐指标对比、`ab` 交错 A/B（对抗 GPU 热降漂移——同码两次实测可差 7.7%）、`telemetry-check` = PI-15 门禁（**同进程交替 + 档位轮转 + 同轮配对差分均值 + 符号多数一致**，`--aa` 标定本机噪声带、阈值 = \|偏差\|+3×SE）、`show` 打印报告细目 | AGENTS.md 性能节 / v2 §8.6 / log 2026-09-11-资源层与门控校准 §4 |
| `_probe_golden_diff.py` | 50 | 金标分歧定位：同进程连跑两次（冷/热池）逐段对比，区分"结构漂移"与"置信度浮点敏感"（F-8 的工具） | tests/golden/FINDINGS.md F-8 |
| `_probe_trt_maxbatch.py` | 101 | **否定/定量**：同一份 ONNX 构建 batch 6 与 18 两个 profile，同批 1116 张走同一提交路径 → 0.663s vs 0.557s（**−16%**）；支撑 `TRT_PROFILE_BATCH=18`（S6） | log 2026-09-10-S6性能轮 §7；FINDINGS F-10 |
| `_probe_engine_ab.py` | 121 | 引擎产物端到端交错 A/B（worker 子进程 monkeypatch `_engine_candidates`，产品代码零开关）：h264-cpu 热轮 −5.85%、`ocr.infer` −33% | log §7；FINDINGS F-10 |
| `_probe_decode_batch_ab.py` | 127 | 解码批 sweep（交错 + 独立子进程 + 段数/文本 sha 硬门）：64/128/256 → **64 最优**（128 持平、256 墙钟 +0.96% 尽管解码相位 −4.9%） | log 2026-09-11-hybrid差距分解 §0 |
| `_probe_run_setup_cost.py` | 80 | 每 run 固定成本分解：analyzer/池/kernel 全在 0.1ms 级 → 0.083s 的 `calibrate` 主体是**首帧解码冷启**（不可省） | 同上 §0 |
| `_probe_hybrid_gap.py` | 107 | hybrid **三层差距分解**（L1 解码器能力 / L2 两路吞吐相加理想 / L3 引擎口径）：hybrid_gpu 仅达理想 67–77%；hybrid_gpu 比宿主输出慢 19–24% | log 2026-09-11-hybrid差距分解；C-05 |
| `_probe_phase_cores.py` | 91 | **每相位资源读数**（§8.6 r5 L1/L2 的消费者）：墙钟/平均并行核数/线程/RSS/磁盘速率/显存 + full 档 NVML 峰值因子。实测 h264-cpu 解码相位 **21.2/32 核** → hybrid CPU 分片必与引擎线程争核 | log 2026-09-11-资源层与门控校准 §3 |
| `_probe_upload_chain.py` | 157 | 用 fork 的 `DECORD_HYBRID_FORCE_SIDE` 拆 **mixed / 纯NVDEC / 纯CPU** 三臂（每组独立子进程 + **45–90s 硬超时** + 轮内组序轮转 + 进度落文件）。量出 mixed"低于最优单侧"（§7.1）——**该结论已被 §8 翻案**：EWMA 估计器伪影 | log 2026-09-11-hybrid差距分解 §7/§8；DECISIONS 2026-09-11 fork 修复 |
| `_probe_hol_stats.py` | 245 | 走 fork **非打印** `DECORD_HYBRID_STATS`（零打印扰动）实测队头阻塞时长/episode/搁置峰值 + BuildPlan 冻结速率/CPU 折数/时刻；decode-only 每格独立子进程+硬超时，`--set KEY=VAL` 做单变量 A/B；`--nts 16,24,32` 展开 CPU 臂线程档位（§14：hybrid_gpu 传 num_threads=0 在 decord 内隐式落 16，同口径比较必须显式给 nt） | log 2026-09-11-hybrid差距分解 §8/§14；knowledge `hybrid_cpu_rate_ratchet` |
| `_probe_stress_harness.py` | 109 | hybrid 死锁/性能**压测 harness 通用规格**：每 trial 独立子进程 + 硬超时 + 进度逐行落文件（慢消费者 = get_batch+asnumpy 保留；"无超时压测烧千秒"三踩后的纪律固化，rules.yaml 同名规则）。kick 突发修复 24/24 证据；`--nt` 覆盖 CPU 臂线程（§14 起引擎新档 24/32 同须压测） | log 2026-09-11-hybrid差距分解 §8.2/§9.1/§14.5；knowledge `hybrid_sustained_default` |
| `_probe_e2e_mode.py` | 34 | 单视频引擎 e2e + 管线模式取证（`_gpu_pipeline_mode` + `DECORD_HYBRID_STATS` 汇总）；全片配对 A/B 用它做 sustained 转默认测量 | log 2026-09-11-hybrid差距分解 §9.2；knowledge `hybrid_sustained_default` |
| `_probe_release_gate.py` | 259 | **hybrid 发布门禁**（§18 硬化）：六步 19 项一键跑——金标 28/28 / 三码×双路径 e2e（段数对表+零 stall 误报）/ 压测 / 损坏码流（现场生成 faststart 截断+坏字节，完成或干净报错）/ KICK_BURST=0 消融判别（期望挂死=机制承重）。损坏流段数不查表；⚠️ 段数期望表是内容锚点，引擎分段语义变更需同步 | log 2026-09-11-hybrid差距分解 §18；knowledge `hybrid_release_gate` |
| `_probe_hybrid_threads_e2e.py` | 137 | hybrid CPU 臂线程档位的**引擎 e2e 配对 A/B**（独立子进程槽位 + 交错 + 段数/唯一文本集 sha 门禁）：§14 定档证据链——16→24/24→32/32→48 三码矩阵，h264→32 / hevc→32 / av1→24 恰为 `_decode_num_threads` 现行策略 | log 2026-09-11-hybrid差距分解 §14；knowledge `hybrid_cpu_threads_tier` |

## B. 库型 / worker 型（**被其他探针依赖，动不得**）

| 文件 | 行 | 被谁依赖 | 形式 |
| `_probe_hooks.py` | 150 | 2026-09-13 | **探针打桩点间接层（L1）**：全部 monkeypatch 点位收敛成一张 POINTS 表——重构只改表，探针不动。重构后跑 `python tools/_probe_hooks.py` 自检，或看审计项 23 |
|---|---:|---|---|
| `_probe_mp_scale.py` | 173 | `_probe_mem_bw` | 子进程 worker（`SCALE` 常量） |
| `_probe_roi_segcost.py` | 132 | `_probe_seg_share` | 代码引用 |

## C. 证据链探针（按"它支撑的结论在哪一章"分组）

删任何一个之前，先确认对应章节的结论是否已经作废。

### §12 去块滤波 / 真值环境

| 文件 | 行 | 改于 |
|---|---:|---|
| `_probe_truth_env.py` | 260 | 2026-08-29 |
| `_probe_slf_diff.py` | 116 | 2026-08-29 |
| `_probe_slf_adjudicate.py` | 142 | 2026-08-29 |

### §13 裁切换 det 模型评估

| 文件 | 行 | 改于 |
|---|---:|---|

### §14 分段合并收口

| 文件 | 行 | 改于 |
|---|---:|---|

### §15 yuv 输出税（否定结果）

| 文件 | 行 | 改于 |
|---|---:|---|
| `_probe_yuv_tax.py` | 132 | 2026-08-30 |

### §16 路线图归档（**该章已归档，探针仅留证据**）

| 文件 | 行 | 改于 | | 文件 | 行 | 改于 |
|---|---:|---|---|---|---:|---|
| `_probe_drop_nonref.py` | 417 | 2026-08-29 | | `_probe_final.py` | 104 | 2026-08-28 |
| `_probe_ffmpeg.py` | 58 | 2026-08-28 | | `_probe_gpu_ctc.py` | 126 | 2026-08-29 |
| `_probe_crop_miscut.py` | 233 | 2026-08-30 | | `_probe_perframe.py` | 120 | 2026-08-28 |
| `_probe_autocrop_truth.py` | 152 | 2026-08-29 | | `_probe_ceiling.py` | 131 | 2026-08-28 |
| `_probe_autocrop_ab.py` | 140 | 2026-08-29 | | `_probe_e2e_ab.py` | 116 | 2026-08-28 |
| `_probe_cpu_onnx.py` | 271 | 2026-08-29 | | `_probe_threads.py` | 64 | 2026-08-28 |
| `_probe_pad_width.py` | 194 | 2026-08-29 | | `_probe_skip_frame.py` | 168 | 2026-08-29 |
| `_probe_roi_width.py` | 171 | 2026-08-29 | | `_probe_guard_clean.py` | 177 | 2026-08-29 |
| `_probe_roi_whitespace.py` | 159 | 2026-08-30 | | | | |

### §17 下一步候选

| 文件 | 行 | 改于 |
|---|---:|---|
| `_probe_python_cost.py` | 157 | 2026-08-30 |
| `_probe_seg_share.py` | 159 | 2026-08-30 |
| `_probe_hybrid_ab.py` | 119 | 2026-08-29 |
| `_probe_batch_coldstart.py` | 101 | 2026-08-31 |

### §19 / §20 / §21 IO 与内存带宽与并发争用（**现役，本轮在用**）

| 文件 | 行 | 改于 | 支撑 |
|---|---:|---|---|
| `_probe_mem_bw.py` | 647 | 2026-08-31 | §20 带宽、§21 |
| `_probe_round4_wall.py` | 119 | 2026-08-31 | §21 墙钟矩阵 |
| `_probe_round4_bw.py` | 102 | 2026-08-31 | §21 带宽矩阵 |
| `_probe_cr_roundtrip.py` | 182 | 2026-08-31 | 裸 CR 保真性（AGENTS.md 编辑护栏） |

| `tools/_audit_ext.py` | 285 | 2026-09-10 | S8 审计扩展 13..22（与纪律审计同一入口） |

### 2026-09-12 准确项（分段合并稠密簇门 · 预处理 gamma 复核）

| 文件 | 行 | 改于 | 支撑 |
|---|---:|---|---|
| `_probe_acc_baseline.py` | 214 | 2026-09-12 | 六片真值逐帧准确率基线 + 误差分类（边界类/段中类），落 `bench/acc_baseline.json` |
| `_probe_acc_ab.py` | 274 | 2026-09-12 | **准确率 A/B harness**：参数化 env 旋钮跑真实引擎全片比对（`--knob OCR_GAMMA=2.0,1.0`），落 `bench/acc_ab_<knob>.json` |
| `_probe_merge_log.py` | 103 | 2026-09-12 | merge_similar 逐次判定插桩（mean/npx/win3/merged）：证明 test5 33/33、test6 94/94 次合并全部 win3=9（稠密簇被吞） |
| `_probe_roi_dump.py` | 111 | 2026-09-12 | ROI 放大导出（6×，帧号标注）+ `--pad` 看框外内容 + `--montage` 拼竖排长图——真值可疑时目视核实原始像素的工具（2026-09-12 §2 的三处裁定都由它复现） |
| `_probe_gamma_sweep.py` | 138 | 2026-09-12 | 误读帧的**离线单帧** gamma/对比度变体扫描（落 `bench/gamma_sweep.json`）。⚠️ 其结论未在真实管线复现（见 log），仅作线索探针保留 |
| `_probe_text_ab.py` | 81 | 2026-09-12 | 字幕场景 A/B：`text_test.mp4` 时间戳式真值按 fps 映射区间取多数文本；门开关两侧字幕行命中 125/198 **完全相同**、墙钟不变（段数 403→1175）|
| `_probe_golden_drift.py` | 64 | 2026-09-12 | 金标漂移的**定性**核对：复用 `record.summarize` 口径逐用例比对唯一文本集——重锚前必须证明「段数变了但没丢文本」（实测 28 用例丢 0 / 增 3）|
| `_probe_hybrid_reeval.py` | 246 | 2026-09-13 | **hybrid 重评 harness**：引擎全片 nvdec/hybrid 双臂 × 三码，一次 run 同时拿 NVML（时钟/热降）与 hybrid-stats（每臂忙时）——忙而慢 vs 调度闲置的归因工具（C-39）|
| `_probe_busy_overhead.py` | 96 | 2026-09-13 | busy 计数器**零开销确认**：fork dll vs wheel 交错配对（轮转先后）——h264 −0.02%、hevc +3.9%（符号混合，噪声内）|
| `_probe_prep_ab.py` | 283 | 2026-09-12 | **预处理重设计 A/B**（全管线，monkeypatch `preprocess_standard`，宿主路径两侧）：6 变体全测 → 只有双三次 +7 帧，逐图拉伸 −49 / unsharp −99 / 局部平场 −1170；`base` 复现 GPU 数字做自检 |
| `_probe_crop_stats.py` | 76 | 2026-09-12 | 代表帧裁切图强度分布：test5/test6 几乎完全相同（Otsu 均 121、文字占比均 0.181）却对每个色调旋钮反向 ⇒ 逐图自适应在信息上不可行（C-15 的依据）|

### 2026-09-13 自诊断（report v3）与路径普查

| 文件 | 行 | 改于 | 支撑 |
|---|---:|---|---|
| `_probe_path_survey.py` | 141 | 2026-09-13 | **路径普查**（测量非 A/B）：五条路径单臂各跑一次、只取 std 档报告，一次给出时间去向 + 背压分诊（`get_wait`/`put_block` 的总量·次数·单次最长）。首轮结论：hybrid/TRT 比 nvdec/TRT 快 1.6~1.8×、decode 占 wall 83~87%、hybrid/cpu 是唯一生产者也被压的路径（瓶颈翻转到下游 OCR）|
| `_probe_decode_rate.py` | 173 | 2026-09-13 | **只解码速率**（工作项 0，扩展 `_probe_path_survey.py`：那条测引擎路径的时间去向，本条测 fork 侧纯解码吞吐）：与引擎 GPU 管线同参数（ctx/gray/ROI-first/线程档/DECODE_BATCH 粒度）但**不做 analyze/分段/OCR**，按 ctx 分离 fork 侧吞吐——判定 hybrid 稳态封顶（三码趋同 2214~2285 fps）落在 fork 还是引擎生产者链 |
| `_probe_dll_ab.py` | 206 | 2026-09-13 | **换 DLL 的交错 A/B**（工作项 5）：`bench ab` 是同进程内跑两臂、而换 DLL 必须每臂新进程，本探针补这个缺口——逐轮交错 + 轮转先后 + 配对差分均值/SE/符号多数，可分辨下限取 PI-15 标定值。⚠️ 实测噪声带 sd 4.6%（冷启动，比同进程 ab 大一个量级）⇒ 判 1% 级效应需 ≥20 轮 |
| `_probe_gap_decomp.py` | 213 | 2026-09-13 | **hybrid 并联缺口分解**（`_probe_decode_rate.py` 的分臂账目版）：全片 + `[hybrid-stats]` 解析，一次拿 delivered c/g、busy、plan rc/rg/份额、HOL——把 fork 级缺口拆成「CPU 臂银行帽 / 计划份额失真 / 每臂混跑干扰」三成分（C-45 的取证入口；支持 `--threads` 扫描） |
| `_probe_pool_pairing.py` | 132 | 2026-09-14 | **跨视频互补配对 A/B**（层5）：pool.run 的 pair（编码感知+LPT）vs 全 nvdec 顺序，交错 + 段/文本逐位门禁；同温零增益（OCR-GPU-bound）的取证入口 |
| `_probe_quant_static.py` | 158 | 2026-09-14 | **静态 INT8 QDQ 量化评测**（压缩轮，已判死）：校准=真帧生产预处理；全图/仅Conv 两变体——argmax 格一致 0.58、CTC 一致 0、速度 +30~81%——与动态量化（20x 劣化，ARCHIVE）同命；ORT 1.29/Win-x64 上量化方向关闭，OpenVINO EP 为 CPU 提速正路（需立项） |
| `_probe_h264_hybrid.py` | 419 | 2026-09-14 | **hybrid 达成率定稿测量**（对两解码器并联和）：同会话三臂（cpu/gpu/hybrid）逐轮交错 + min-of-N + 机器忙闲污染门（>40% 当轮重试）+ fps_b/wall 双口径 + 外部重负载进程扫描；`--landscape` 线程扫描 / `--shares` 份额扫描 / `--arm` 自定义臂。C-46 达成率数字的复测载体（达成率定稿协议见 log 2026-09-14） |
| `_probe_ocr_phase_split.py` | 137 | 2026-09-14 | **OCR 批延迟拆相**（TRT 流水轮取证）：生产 ocr.infer 按相拆解——prep 提交 / TRT 调用（enqueue+argmax+D2H+sync）或深度 2 的 submit/collect / 宿主 CTC；`--runs` 冷热两轮。定量出 trt_call 13.9ms/批占 87%、CTC 0.4ms（launch 裸奔归因的测量入口，结论见 log 2026-09-14-TRT延迟收集流水） |
| `_probe_ov_cpu_ab.py` | 97 | 2026-09-14 | **OCR CPU 后端 A/B：onnxruntime vs OpenVINO**（模型级）：同模型同线程（物理核）同形状，生产 ORT 配置 vs OV CPU 插件——本机 Zen4 实测 OV 快 **2.1×**（47.8→23.0ms@B18W224）且随机输入 argmax 100% 一致；OpenVINO 立项（ocr_backend 扩展）的裁决入口 |

### 2026-09-10 D1/D2 调查（gpu/host 段数分歧 · 线程优先级）

| 文件 | 行 | 改于 | 支撑 |
|---|---:|---|---|
| `_probe_d1_prim_diff.py` | 125 | 2026-09-10 | log《hybrid联调深挖》D1：帧级原语逐项对比（1083 vs 1042 根因定位） |
| `_probe_d1_trace.py` | 118 | 2026-09-10 | log《hybrid联调深挖》D1：分段判定逐帧 trace |
| `_probe_d1_trace2.py` | 148 | 2026-09-10 | log《hybrid联调深挖》D1：合并判定失真定位（PI-6 修复证据） |

## D. 孤儿（无任何文档引用、也无代码引用）

**共 4 个 / 550 行 / 20.7 KB**。量很小，**建议保留**——删掉的代价（断了未记录的
结论链）远大于留着的代价（一个文件名）。

> 数字由 `python tools/_probe_index_audit.py` 实测维护，别手改。

| 文件 | 行 | 改于 | 状态判断 |
|---|---:|---|---|
| `_probe_lifecycle_repeat.py` | 210 | 2026-08-31 | 生命周期修复轮的重复压测，**近期在用的可能性最高，留** |
| `_probe_cluster_dtype.py` | 105 | 2026-08-30 | `_cluster_win3` 改 uint8 的 dtype 验证，留作证据 |
| `_probe_slf_vis.py` | 110 | 2026-08-29 | 生成 `tools/_slf_vis/` 拼图（DECISIONS「P0-6 翻案」引用了该目录），留 |
| `probe_decode_rates.py` | 125 | 2026-08-28 | 探测各后端解码速率，与 §21 结论同主题，留 |

## E. 一次性迁移工具（任务已完成）

| 文件 | 行 | 说明 |
|---|---:|---|
| `_split_claude_md.py` | 263 | 2026-08-31 把 AGENTS.md 拆成注入核 + `docs/log/DECISIONS.md`。**已完成，可删** |
| `_split_perf_md.py` | 194 | 2026-08-31 按「活/归档」把 PERFORMANCE.md 切出 `docs/log/ARCHIVE.md`。**已完成，可删** |
| `_fix_probe_paths.py` | 416 | 2026-08-31 把探针里写死的路径改成 `__file__` 推导 / 环境变量。**已完成，可删** |

## 探针状态（L2）：live 14 / frozen 75（共 89）

**默认 frozen**——不在下表 `live` 名单里的探针一律视为历史证据，豁免活性检查、不做修复义务。
一个探针进 `live` 必须写清理由（被在用载体引用 / 本轮在用）；重构时的修复义务**只覆盖 live 集合**。

| live 探针 | 为何在用 |
|---|---|
| `_probe_cpu_onnx.py` | AGENTS.md |
| `_probe_discipline_audit.py` | AGENTS.md、tools/_audit_ext.py、tools/_probe_discipline_audit.py、tests/ |
| `_probe_index_audit.py` | AGENTS.md、tests/ |
| `_probe_path_survey.py` | 本轮新工具：路径普查，当前优化轮的取证入口 |
| `_probe_decode_rate.py` | 本轮新工具：工作项 0（hybrid 封顶归因）的取证入口 |
| `_probe_dll_ab.py` | 本轮新工具：工作项 5（换 DLL 交错 A/B），fork 改动的判据工具 |
| `_probe_gap_decomp.py` | 本轮新工具：并联缺口分解（C-45），fork 改动的归因入口 |
| `_probe_h264_hybrid.py` | C-46 达成率定稿口径的载体：fork 换代/换卡复评直跑（同会话三臂交错协议） |
| `_probe_ocr_phase_split.py` | 本轮新工具：OCR 批延迟拆相（TRT_DEFER_SYNC 复评时直跑） |
| `_probe_ov_cpu_ab.py` | 本轮新工具：OpenVINO vs ORT 模型级 A/B（OV 立项复评直跑） |
| `_probe_content_det.py` | 本轮新工具：hybrid 输出内容确定性对照（帧级 hash，C-46 kick 落位根因的取证入口） |
| `_probe_quant_static.py` | 本轮新工具：静态 QDQ 量化评测（已判死；OpenVINO 复评时直跑） |
| `_probe_pool_pairing.py` | 本轮新工具：层5 配对 A/B |
| `_probe_run_setup_cost.py` | tests/ |

frozen 75 个（按 §A–§D 各节原样保留）：`_probe_acc_ab.py`、`_probe_acc_baseline.py`、`_probe_autocrop_ab.py`、`_probe_autocrop_truth.py`、`_probe_batch_coldstart.py`、`_probe_busy_overhead.py`、`_probe_ceiling.py`、`_probe_cluster_dtype.py`、`_probe_cr_roundtrip.py`、`_probe_crop_miscut.py`、`_probe_crop_stats.py`、`_probe_d1_prim_diff.py`、`_probe_d1_trace.py`、`_probe_d1_trace2.py`、`_probe_decode_batch_ab.py`、`_probe_decode_contention.py`、`_probe_drop_nonref.py`、`_probe_e2e_ab.py`、`_probe_e2e_mode.py`、`_probe_engine_ab.py`、`_probe_ffmpeg.py`、`_probe_final.py`、`_probe_gamma_sweep.py`、`_probe_golden_diff.py`、`_probe_golden_drift.py`、`_probe_gpu_ctc.py`、`_probe_guard_clean.py`、`_probe_hol_stats.py`、`_probe_hybrid_ab.py`、`_probe_hybrid_axis.py`、`_probe_hybrid_bitwise.py`、`_probe_hybrid_cpu_profile.py`、`_probe_hybrid_engine_loss.py`、`_probe_hybrid_gap.py`、`_probe_hybrid_reeval.py`、`_probe_hybrid_sum_gap.py`、`_probe_hybrid_threads_e2e.py`、`_probe_hybrid_trace.py`、`_probe_lifecycle_repeat.py`、`_probe_mem_bw.py`、`_probe_merge_log.py`、`_probe_mp_scale.py`、`_probe_nvdec_interference.py`、`_probe_onnx_dcd_sweep.py`、`_probe_pad_width.py`、`_probe_perf_baseline.py`、`_probe_perf_sweep.py`、`_probe_perframe.py`、`_probe_phase_cores.py`、`_probe_prep_ab.py`、`_probe_python_cost.py`、`_probe_r3_infer_split.py`、`_probe_release_gate.py`、`_probe_roadmap_decode.py`、`_probe_roadmap_ocr.py`、`_probe_roadmap_profile.py`、`_probe_roi_decode.py`、`_probe_roi_dump.py`、`_probe_roi_segcost.py`、`_probe_roi_whitespace.py`、`_probe_roi_width.py`、`_probe_round4_bw.py`、`_probe_round4_wall.py`、`_probe_seg_share.py`、`_probe_skip_frame.py`、`_probe_slf_adjudicate.py`、`_probe_slf_diff.py`、`_probe_slf_vis.py`、`_probe_stress_harness.py`、`_probe_text_ab.py`、`_probe_threads.py`、`_probe_trt_maxbatch.py`、`_probe_truth_env.py`、`_probe_upload_chain.py`、`_probe_yuv_tax.py`
## 清理判据（想删探针时按这个顺序）

1. `grep -rn "<文件名>" README.md AGENTS.md docs/ tools/` —— 有命中就不删。
2. 命中只在 `docs/log/ARCHIVE.md` / PERF §16 归档章 → 该结论已归档，可随档一起删，
   但要确认 §16 的校正表没把它标成"仍有效"。
3. 零命中 → 归入「孤儿」，走上面 D 节的逐个案判断。
4. 删之前把结论数字抄进 `docs/log/PERFORMANCE.md` 对应章节 —— **探针可以删，
   数字不能丢**。

## 数据文件

多个探针会把结果写到 `tools/_probe_*.json` / `tools/_round4_*.json`。
⚠️ 驱动脚本会**覆写**同名 JSON，重跑前先备份（踩过：`_round4_bw.json` 被覆盖）。

| 文件 | 内容 |
|---|---|
| `tools/_round4_wall.json` | §21 墙钟矩阵 |
| `tools/_round4_bw.json` | §21 DRAM 消耗与瞬时带宽序列 |
| `tools/_round4_bw_rerun.json` | §21.8 独立重跑复核数据 |
| `tools/_index_data.json` | 生成本索引时的原始扫描数据（可删后重生成） |
