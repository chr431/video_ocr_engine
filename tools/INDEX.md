# tools/ 索引

`tools/` 现有 **51 个 `.py`**（10,706 行 / ~423 KB）））），其中 42 个是探针
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
| `e2e_smoke.py` | 351 | 端到端冒烟 / 真值验证（真实视频） | README「测试」节 |
| `bench_hybrid.py` | 125 | hybrid 解码基准 | PERF §4 |
| `probe_decode_rates.py` | 125 | 各后端解码速率探测 | — |
| `_probe_index_audit.py` | 277 | 核对本索引的每个数字是否与磁盘一致 | 本文件（自检） |
| `_probe_discipline_audit.py` | 574 | **项目纪律审计**（12 项：硬编码路径 / 异常吞噬 / 未用 import / 未门控 print / 版本号 / 文档引用 / 注入预算…） | CLAUDE.md「纪律与自动化守卫」 |
| `_doc_section.py` | 214 | **文档章节级检索**：`--toc` 看目录 / `--find` 按标题定位 / 读单章。避免整文件读，实测省 84~96% tokens | CLAUDE.md「查文档前先定位」 |
| `_probe_roi_decode.py` | 105 | **否定结果**：量化「打开时 SetRoi」vs「每次 get_batch 传 roi」对 CPU 软解速率的影响。实测两者无差异（1841 vs 1849 fps），但**不传 ROI = 520 fps**（3.6× 慢）→ ROI 本身是巨大优化，两种传法等价 | PERF §22.1 |

## B. 库型 / worker 型（**被其他探针依赖，动不得**）

| 文件 | 行 | 被谁依赖 | 形式 |
|---|---:|---|---|
| `_probe_det_crop_eval.py` | 412 | `_probe_block_audit` / `_probe_domain_audit` / `_probe_merge_audit` | `from ... import` |
| `_probe_merge_audit.py` | 414 | `_probe_block_audit` / `_probe_domain_audit` | `from ... import` |
| `_probe_block_audit.py` | 213 | `_probe_domain_audit` | `from ... import` |
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
| `_probe_det_crop_eval.py` | 412 | 2026-08-30 |

### §14 分段合并收口

| 文件 | 行 | 改于 |
|---|---:|---|
| `_probe_merge_audit.py` | 414 | 2026-08-30 |
| `_probe_block_audit.py` | 213 | 2026-08-30 |
| `_probe_domain_audit.py` | 143 | 2026-08-30 |

### §15 yuv 输出税（否定结果）

| 文件 | 行 | 改于 |
|---|---:|---|
| `_probe_yuv_tax.py` | 132 | 2026-08-30 |

### §16 路线图归档（**该章已归档，探针仅留证据**）

| 文件 | 行 | 改于 | | 文件 | 行 | 改于 |
|---|---:|---|---|---|---:|---|
| `_probe_pad_variants.py` | 340 | 2026-08-29 | | `_probe_ffmpeg.py` | 54 | 2026-08-28 |
| `_probe_drop_nonref.py` | 417 | 2026-08-29 | | `_probe_final.py` | 104 | 2026-08-28 |
| `_probe_crop_miscut.py` | 233 | 2026-08-30 | | `_probe_perframe.py` | 120 | 2026-08-28 |
| `_probe_autocrop_truth.py` | 152 | 2026-08-29 | | `_probe_ceiling.py` | 131 | 2026-08-28 |
| `_probe_autocrop_ab.py` | 140 | 2026-08-29 | | `_probe_e2e_ab.py` | 116 | 2026-08-28 |
| `_probe_cpu_onnx.py` | 271 | 2026-08-29 | | `_probe_threads.py` | 64 | 2026-08-28 |
| `_probe_pad_width.py` | 194 | 2026-08-29 | | `_probe_skip_frame.py` | 168 | 2026-08-29 |
| `_probe_roi_crop_ocr.py` | 182 | 2026-08-29 | | `_probe_gpu_ctc.py` | 126 | 2026-08-29 |
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
| `_probe_io_budget.py` | 577 | 2026-08-31 | §19 IO、§21 |
| `_probe_mem_bw.py` | 647 | 2026-08-31 | §20 带宽、§21 |
| `_probe_round4_wall.py` | 119 | 2026-08-31 | §21 墙钟矩阵 |
| `_probe_round4_bw.py` | 102 | 2026-08-31 | §21 带宽矩阵 |
| `_probe_cr_roundtrip.py` | 182 | 2026-08-31 | 裸 CR 保真性（CLAUDE.md 编辑护栏） |

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
| `_split_claude_md.py` | 263 | 2026-08-31 把 CLAUDE.md 拆成注入核 + `docs/DECISIONS.md`。**已完成，可删** |
| `_split_perf_md.py` | 194 | 2026-08-31 按「活/归档」把 PERFORMANCE.md 切出 `docs/ARCHIVE.md`。**已完成，可删** |
| `_fix_probe_paths.py` | 416 | 2026-08-31 把探针里写死的路径改成 `__file__` 推导 / 环境变量。**已完成，可删** |

## 清理判据（想删探针时按这个顺序）

1. `grep -rn "<文件名>" README.md CLAUDE.md docs/ tools/` —— 有命中就不删。
2. 命中只在 `docs/ARCHIVE.md` / PERF §16 归档章 → 该结论已归档，可随档一起删，
   但要确认 §16 的校正表没把它标成"仍有效"。
3. 零命中 → 归入「孤儿」，走上面 D 节的逐个案判断。
4. 删之前把结论数字抄进 `docs/PERFORMANCE.md` 对应章节 —— **探针可以删，
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
