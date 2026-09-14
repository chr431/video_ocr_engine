# AGENTS.md — 开发记录与约定（注入核）

> 本文件在每个会话开头被注入，**只放"现在必须知道的"**。
> **硬上限 14 KB** — 超了就把内容迁到 `docs/log/DECISIONS.md`，这里只留指针。

## 文档地图（7 份治理文档，别再新增；另有 4 份衍生/辅助）

| 文件 | 性质 | 什么时候读 |
|---|---|---|
| `README.md` | 用户向 API / 用法 | 写调用代码时 |
| `AGENTS.md`（本文件） | 维护者向**注入核**：铁律 + 现役架构 + 结论指针 | 自动注入 |
| `docs/CONCLUSIONS.md` | **L1 结论索引**（状态/前提/复评触发），唯一允许规范性结论处 | 动手前查结论、出结论写这里 |
| `docs/log/PERFORMANCE.md` | 性能实验史（**已冻结增长**，新叙事进 `docs/log/`） | 需要实测细节/证据链时 |
| `docs/log/DECISIONS.md` | 每轮决策过程、已删除功能、设计审查结论 | 想问"为什么这么做"时 |
| `docs/log/ARCHIVE.md` | 归档（PERF §4 / §8 / §16 / §18），**编号保留勿重编** | 只看"为什么不做" |
| `docs/DEPENDENCIES.md` | 依赖版本与已知问题 | 装环境 / 报 bug 时 |
| `docs/MIGRATION.md` | API 迁移表（v0.11→v0.13；**0.14.0 删除清单在这里**） | 改导入路径 / 删 shim 前 |
| `docs/KNOBS.md` | 旋钮表（render 产物，事实源 `video_ocr_engine/config/knobs.py`） | 查旋钮默认值与依据 |
| `docs/log/README.md` | `docs/log/` 自己的写作规则（叙事不放规范性语句） | 往 log 写东西前 |
| `docs/architecture.svg` | README 配图（用户向架构图，**手绘，改架构时同步刷新版本号**） | 改架构 / 发版时 |

⚠️ **现役规则以本文件为准**；`docs/log/DECISIONS.md` 是迁出的原文存档，冲突时
以本文件为真相（避免"两套真相"，设计审查 D6）。结论的**当前状态**
（active / superseded / dead）以 `docs/CONCLUSIONS.md` 为准。

### ⛔ 查文档前先定位，不要整文件读

大文档实测（cl100k）：PERFORMANCE ≈ **30k**、ARCHIVE ≈ **28k**、
DECISIONS ≈ **17k** tokens，而本文件注入才 **~2.4k** —— **误读一次 ≈ 12 倍
注入成本**，本项目最大的 token 浪费点。

```bash
python tools/_doc_section.py --find <关键词>   # 跨文档按标题定位（≈357 tok）
python tools/_doc_section.py --toc <文件>      # 目录+token 数（≈838 tok）
python tools/_doc_section.py <文件> 21         # 只读 §21（支持 16 / 16.8 / 4.4b）
```

`--toc` + 单章 ≈ **2,800 tokens**，比整文件读**省 84~96%**；`--find` 只搜标题，
搜正文用 grep。查结论先读 `docs/CONCLUSIONS.md`（≈3k，预算受测试守护）。

## 铁律（先量后做）

1. **先量后做**：任何结论必须带实测数字，禁止凭直觉推断下结论。
2. **正确性门禁** = 段数 + 唯一文本集（或真值准确率），不是"看起来没问题"。
3. **只测"文本有没有变"会判错 OCR 的好坏** —— 必须**按帧对齐真值**测准确率；
   别用置信度当代理（`羸弱→赢弱` 是退化，置信度反而 0.9433→0.9700）。
4. **均值不能替代逐片检查**：5 片均值 ±0.01pp 曾用来支持"无负面影响"，
   第 6 片就翻了案。
5. **真值本身要抽查**：版本、剥零、哨兵、时间基准都可能错
   （见 DECISIONS「P0-6 翻案」）。
6. **别按"看起来旧"删脚本**：`tools/` 的探针是**证据链**，删前先查引用
   （绝大多数被文档、跨探针 import 或文档路径引用锚定）。
7. **探针放 `tools/_probe_*.py`**（下划线前缀 = 调查工具，不随产品发布）。
8. **结论分离**：结论一行进 `docs/CONCLUSIONS.md`（必须带状态/前提/复评
   触发），实验叙事进 `docs/log/`（PERFORMANCE.md 已冻结增长）；历史章节
   不回溯重写。版本号改动必须打同名 git tag。
9. **迁移清扫**：依赖升级 / 大迁移落地时，grep `docs/CONCLUSIONS.md` 的
   "前提/触发"列，逐条复核命中行并翻状态（active / superseded / dead）。
10. **向后兼容**：新功能默认关闭，除非明确作为新默认；新增遗留面一律先标
   deprecated、两个版本后删除。

## 现役架构

链路：**解码 → 像素分段 → 代表帧 → OCR → 相似段合并**

| 阶段 | 入口 |
|---|---|
| 解码 | `decord.VideoReader.get_batch`（**唯一入口**；decode_backend=hybrid 走 decord 原生混合解码 ctx） |
| 分段 | `video_ocr_engine/domain/segmentation.py` |
| 编排引擎 | `video_ocr_engine/pipeline/engine.py`（SegmentEngine 唯一入口） |
| 宿主后端 | `video_ocr_engine/pipeline/host_backend.py` |
| GPU 后端 | `video_ocr_engine/pipeline/gpu_backend.py`（gray+NVDEC+TRT 时默认；设备侧机制在 `video_ocr_engine/gpu/device.py`） |
| OCR 会话 | `video_ocr_engine/pipeline/ocr_stage.py`（SessionSpec 契约） |
| OCR 调度 / 引擎池 | `video_ocr_engine/ocr/native.py`（根 `ocr_native.py` 为兼容 shim） |
| TRT | `video_ocr_engine/ocr/trt.py` + `video_ocr_engine/_gpu_kernels.py` |
| 配置常量 | `video_ocr_engine/config/constants.py`（根 `engine_config.py` 为兼容 shim）；旋钮注册表 `video_ocr_engine/config/` |
| 运行报告 | `video_ocr_engine/pipeline/report.py`（RunReport schema **v2** → `meta['report']`；v2 加 `resources`/`hardware`）；指标注册表 `video_ocr_engine/domain/metrics.py`；资源层 `video_ocr_engine/domain/resources.py`（L1 边界差分 / L2 NVML） |
| 分相打桩 | `video_ocr_engine/extractor.py` 的 `_prof_end`（**单一计时脊柱**：同一 t0 喂 profile 与指标） |
| 性能 A/B | `tools/bench.py`（`run`/`diff`/`show`/`ab`/`telemetry-check`；报告落 `bench/registry.jsonl`） |

⚠️ **A/B 必须交错**（`bench ab`）：同码连跑两次实测可差 **7.7%**（GPU 热降），
顺序跑会把漂移记到 B 头上。⚠️ **门禁阈值 ≥ 本机可分辨下限**：先
`bench telemetry-check --aa` 标定（本机 A/A \|Δ\|p95 0.484%，2026-09-13
重标，n=25/inner=3 → std 限 +0.30% / full 限 +1.20%），阈值 =
\|偏差\|+3×SE，判据 = 同轮配对差分**均值** + 符号多数一致；µs 级严格性见
`tests/config/test_telemetry_cost.py`。执行体在脚本（`knowledge/rules.yaml`）。

**现役并行维度只有一个**：`decode_backend="hybrid"` 的 CPU+NVDEC 双解码，
**已由 decord fork 原生实现**（v0.7.15+ 的 `hybrid`/`hybrid_gpu` ctx，引擎只
透传解码参数；项目层 `hybrid_decode.py` 已删除，迁移记录见 DECISIONS 同名
章节与 PERF §24）。**没有 dual pipeline、没有 `DUAL_*` 环境变量、
没有 `_dual_pipeline.py`** —— 历史提及均为旧档案，勿据此调优。

其他现役事实：

- **引擎内部恒为单通道灰度**：decord 只输出 `'yuv420'`（keep_crops 且
  `rep_crop_format="yuv"`）或 `'gray'`；旧 `gray_output`/`yuv_output` 已删除。
- **相似段合并**在分离图上进行，默认 `binary`（`TEXT_SEP_MERGE=binary|off`）。
- **GPU 管线门控**：gray + `decode∈{auto,nvdec,cpu}` + `ocr≠cpu` + NVDEC/TRT
  可用时自动启用；`GPU_PIPELINE=0` 关 / `=1` 强制。GPU+ONNX 组合实测无净收益。
- **OCR 引擎池**：`_POOL_MAX_PER_KEY=4`、`_POOL_MAX_TOTAL=16`，
  key=(model, type, fill_width, threads)。

## 已封板结论 → `docs/CONCLUSIONS.md`

全部 26 条结论（20 条 active / 6 条 superseded 指针；dead 已整体降级
`docs/log/`）在 `docs/CONCLUSIONS.md`，这里只留最容易踩的六条：

- 并发退化真因 = **NVDEC 会话数**；互补配对首选 NVDEC∥CPU，聚合 1.87×（PERF §21）
- `auto` **恒为 NVDEC 优先**（刻意决策，2026-09-10 重申）：本机 h264 CPU 软解虽快
  1.7~2.8×（C-04），但弱 CPU 可能反慢且必带争用/功耗代价；批量互补仍需**显式**
  `decode_backend="cpu"`（C-07/C-08）
- GPU 分段 + ONNX OCR 无净收益，门控只放行 NVDEC+TRT（PERF §9）
- hybrid 已迁 **decord 原生**；现役架构 = **包缓存 + 供料期 GOP 派工**（C-46，fork fef3c4b 已随 wheel 发布）：Push 只入压缩包缓存（512MB），泵按当下速率贪心派工 GOP（前瞻+同侧保序）；**kick 必须经泵按流序注入**（错位=IDR 冲掉重排窗→遮蔽损坏，fork 级测不出）；引擎 A/B hevc −6.29%/h264 −4.78%/av1 平价，fork 对并联和 96/92/96%（同会话三臂交错口径）。**收益面 = TRT/设备路径；ONNX 宿主路径不反超**（选 nvdec）；h264 峰值走显式 `cpu`（C-08）。GPU 侧余量=OCR：h264 已 OCR-bound；**fp16 已产品化（TRT_FP16=1，opt-in）**——fp16 ONNX+STRONGLY_TYPED 路线，真值准确率 1.0000（30664 帧）+热池 −3%/−1%；CUDA Graph e2e 无收益且 hevc 崩，已移除（TRT 换代后重评）。旧机制（计划/视界/REPLAN/债务克隆）已删除
- **racelog_test 全部视频测量/验证一律 `sample_stride=1`**（2026-09-10 重申，
  防漏信息）；stride>1 仅用于字幕场景（字幕更新频率慢，如批量剧集字幕提取）
- **合并判定默认带「稠密簇门」**（`segment.merge_dense_gate`）：遥测内容不再合并，
  段数比此前高 1~4%（C-38）

## 编辑护栏（docs/log/PERFORMANCE.md）

✅ **纯 LF，裸 CR = 0**（2026-08-31 清除 934 处，提交 `fd2a76a`）——文本模式
（含默认 `newline=None`）、Edit 工具、二进制模式**都 100% 保真**。自检
`open(p,'rb').read().count(b'\r') == 0`；防护 `tests/test_docs_hygiene.py`；
934 处的成因与修法见 `docs/log/DECISIONS.md`「编辑护栏：934 裸 CR 清除记」。

## 纪律与自动化守卫

**改完代码跑一次**（22 项 = 12 基础 + 10 扩展，退出码非 0 即违规；关键项另有单测守护）：

```bash
python tools/_probe_discipline_audit.py          # 全量
python tools/_probe_discipline_audit.py --only 1,3,7
python tools/_probe_index_audit.py               # tools/INDEX.md 数字一致性
```

几条容易踩、且已自动化的：

- **路径不许写死**：仓库内路径一律 `__file__` 推导；外部视频走环境变量
  （`RACELOG_VIDEO_DIR` / `RACELOG_BATCH_DIR`），硬编码值只能当默认值。
  ⚠️ **`WORKER = r"""..."""` 子进程模板是特例**：它以 `python -c` 执行，
  **`-c` 下 `__file__` 未定义**，须靠父进程注入 `PROBE_ROOT`（`_probe_cpu_onnx.py`
  例外：它做新旧版本 A/B，子进程 cwd 就是旧 worktree，**必须插 cwd**）。
- **`except … pass` 不许无声**：要么 `logger.debug`，要么 `pass  # 为何可忽略`
  （存量 63 处登记在 `tools/_discipline_baseline.json`，**存量豁免、增量严格**；
  改到该文件时顺手补注释即自然脱出）。`except BaseException` 仅清理路径允许。
- **产品代码的 print 必须受 debug 开关保护**（`env_bool(DEBUG_BOUNDS_ENV)` /
  `self._probe`），否则走 `logging`。docstring 里的用法示例不算。
- **未使用的 import**：有意 re-export 加 `# noqa: F401`，否则删掉。
- **探针分层（L2）**：`tools/INDEX.md` 的「探针状态」小节是唯一权威——**只有 `live` 名单里的探针有修复义务**，其余默认 `frozen`（证据已产出、结论已封板 → 豁免活性检查，重构时不必修）。新增 live 须写理由。现状 live 5 / frozen 75 —— 这就是抑制「探针无限堆积 + 失效修复成本无限增高」的机制。
- **不得引用六个废弃根模块 shim**（`engine_config` / `gpu_setup` /
  `ocr_native` / `ocr_trt` / `segmentation` / `video_utils`）——审计项 22，
  白名单只有两个冻结 shim 可导入性的契约测试；迁移表 `docs/MIGRATION.md` §1。
- **文档裸 CR = 0、AGENTS.md ≤ 14 KB**：`tests/test_docs_hygiene.py` 守护。

（旧「只能用二进制」「Edit 工具不安全」两条规矩**双向都错**，勘误原文见
`docs/log/DECISIONS.md` 2026-09-12「编辑护栏勘误」。）

## 环境与命令

- Python：`c:/Users/eric chen/AppData/Local/Programs/Python/python313/python.exe`
  （**PATH 上的 `python` 缺 numpy，不是项目环境**）。pytest / numpy / psutil /
  cuda.bindings / decord 均可用。
- 硬件：RTX 4060（**单 NVDEC 单元**）/ 16C32T / 2×16GB DDR5-6000（实测流式
  上限 **55.8 GB/s**；WMI `Speed`=5600 是 SPD 标称，`ConfiguredClockSpeed`
  =6000 才是实际值）。
- 测试视频 `D:\Videos\racelog_test\`，真值在 `ground_truth_csv/`
  （头是 `# roi=...`，**必须用正则取四个整数**，按逗号切只能拿到第一个）。
  ⚠️ **该目录全部视频一律 `sample_stride=1`**（防漏信息）；`stride>1` 只用于
  字幕提取场景（`text_text`/`batch_test`，字幕更新频率慢）。
- 中文输出需 `sys.stdout.reconfigure(encoding="utf-8")`（GBK 控制台会崩）。
- `tools/` 子目录下 `sys.path[0]` 是 tools/，探针须 `sys.path.insert(0, 上级目录)`。
