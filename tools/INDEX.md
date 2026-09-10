# tools/ 索引

`tools/` 现有 **64 个 `.py`**（10,868 行），其中 52 个是探针
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

## B. 库型 / worker 型（**被其他探针依赖，动不得**）

| 文件 | 行 | 被谁依赖 | 形式 |
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
| `_probe_python_cost.py` | 130 | 2026-08-30 |
| `_probe_seg_share.py` | 154 | 2026-08-30 |
| `_probe_hybrid_ab.py` | 119 | 2026-08-29 |
| `_probe_batch_coldstart.py` | 101 | 2026-08-31 |

### §19 / §20 / §21 IO 与内存带宽与并发争用（**现役，本轮在用**）

| 文件 | 行 | 改于 | 支撑 |
|---|---:|---|---|
| `_probe_mem_bw.py` | 647 | 2026-08-31 | §20 带宽、§21 |
| `_probe_round4_wall.py` | 119 | 2026-08-31 | §21 墙钟矩阵 |
| `_probe_round4_bw.py` | 102 | 2026-08-31 | §21 带宽矩阵 |
| `_probe_cr_roundtrip.py` | 182 | 2026-08-31 | 裸 CR 保真性（AGENTS.md 编辑护栏） |

| `tools/_audit_ext.py` | 220 | 2026-09-10 | S8 审计扩展 13..21（与纪律审计同一入口） |

### 2026-09-10 D1/D2 调查（gpu/host 段数分歧 · 线程优先级）

| 文件 | 行 | 改于 | 支撑 |
|---|---:|---|---|
| `_probe_d1_prim_diff.py` | 125 | 2026-09-10 | log《hybrid联调深挖》D1：帧级原语逐项对比（1083 vs 1042 根因定位） |
| `_probe_d1_trace.py` | 114 | 2026-09-10 | log《hybrid联调深挖》D1：分段判定逐帧 trace |
| `_probe_d1_trace2.py` | 145 | 2026-09-10 | log《hybrid联调深挖》D1：合并判定失真定位（PI-6 修复证据） |

## D. 孤儿（无任何文档引用、也无代码引用）

**共 4 个 / 550 行 / 20.6 KB**。量很小，**建议保留**——删掉的代价（断了未记录的
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
