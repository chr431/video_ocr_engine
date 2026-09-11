# AGENTS.md — 开发记录与约定（注入核）

> 本文件在每个会话开头被注入，**只放"现在必须知道的"**。
> **硬上限 12 KB** — 超了就把内容迁到 `docs/log/DECISIONS.md`，这里只留指针。

## 文档地图（7 份，别再新增）

| 文件 | 性质 | 什么时候读 |
|---|---|---|
| `README.md` | 用户向 API / 用法 | 写调用代码时 |
| `AGENTS.md`（本文件） | 维护者向**注入核**：铁律 + 现役架构 + 结论指针 | 自动注入 |
| `docs/CONCLUSIONS.md` | **L1 结论索引**（状态/前提/复评触发），唯一允许规范性结论处 | 动手前查结论、出结论写这里 |
| `docs/log/PERFORMANCE.md` | 性能实验史（**已冻结增长**，新叙事进 `docs/log/`） | 需要实测细节/证据链时 |
| `docs/log/DECISIONS.md` | 每轮决策过程、已删除功能、设计审查结论 | 想问"为什么这么做"时 |
| `docs/log/ARCHIVE.md` | 归档（PERF §4 / §8 / §16 / §18），**编号保留勿重编** | 只看"为什么不做" |
| `docs/DEPENDENCIES.md` | 依赖版本与已知问题 | 装环境 / 报 bug 时 |

⚠️ **现役规则以本文件为准**。`docs/log/DECISIONS.md` 是迁出的原文存档，
两者冲突时以本文件为真相（避免"两套真相"，见设计审查 D6）。结论的
**当前状态**（active / superseded / dead）以 `docs/CONCLUSIONS.md` 为准。

### ⛔ 查文档前先定位，不要整文件读

实测（tiktoken cl100k，2026-09-08）：PERFORMANCE.md ≈ **30k**、
ARCHIVE.md ≈ **28k**、DECISIONS.md ≈ **17k** tokens —— 而本文件每会话注入
才 **~2.4k tokens**。**误读一次大文档 ≈ 12 倍的注入成本**，这是本项目最大的
token 浪费点。查结论先读 `docs/CONCLUSIONS.md`（≈3k，预算受测试守护）。

```bash
python tools/_doc_section.py --find <关键词>        # 跨文档按标题定位，≈357 tokens
python tools/_doc_section.py --toc docs/log/PERFORMANCE.md   # 目录+token数，≈838 tokens
python tools/_doc_section.py docs/log/PERFORMANCE.md 21      # 只读 §21
python tools/_doc_section.py docs/log/ARCHIVE.md 4.4b        # 支持 16 / 16.8 / 4.4b
```

典型路径：`--toc`(838) + 读单章(≈2,000) ≈ **2,800 tokens**，比整文件读
**省 84~96%**。`--find` 只搜标题；要搜正文再用 grep。

## 铁律（先量后做）

1. **先量后做**：任何结论必须带实测数字，禁止凭直觉推断下结论。
2. **正确性门禁** = 段数 + 唯一文本集（或真值准确率），不是"看起来没问题"。
3. **判断 OCR 变好还是变坏，只测"文本有没有变"会得出错误结论** —— 必须
   **按帧对齐真值**测准确率；别用置信度当代理（`羸弱→赢弱` 是退化，
   置信度反而 0.9433→0.9700）。
4. **均值不能替代逐片检查**：5 片均值 ±0.01pp 曾用来支持"无负面影响"，
   第 6 片就翻了案。
5. **真值本身要抽查**：版本、剥零、哨兵、时间基准都可能错
   （见 DECISIONS「P0-6 翻案」）。
6. **别按"看起来旧"删脚本**：`tools/` 的探针是**证据链**，删之前先查引用
   （35/40 被文档引用，另有 5 处跨探针 import 与 69 处文档路径引用）。
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
| 分段 | `segmentation.py` |
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

⚠️ **A/B 必须交错**（`bench ab`）：同一份代码连跑两次实测可差 **7.7%**
（GPU 热降/后台占用），顺序 A 全跑再 B 全跑会把漂移记到 B 头上。
⚠️ **门禁阈值必须 ≥ 本机可分辨下限**：先 `bench telemetry-check --aa` 标定
（本机 A/A 同档两槽 \|Δ\|p95 达 1.8%），阈值 = \|偏差\|+3×SE；判据是**同轮配对
差分的均值**+符号多数一致。µs 级严格性在 `tests/config/test_telemetry_cost.py`。
`AGENTS.md` 只留指路，执行体在脚本（`knowledge/rules.yaml` 同名规则）。

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

全部 33 条结论（含状态 / 前提 / 复评触发）在 `docs/CONCLUSIONS.md`，
这里只留最容易踩的五条：

- 并发退化真因 = **NVDEC 会话数**；互补配对首选 NVDEC∥CPU，聚合 1.87×（PERF §21）
- `auto` **恒为 NVDEC 优先**（刻意决策，2026-09-10 重申）：本机 h264 CPU 软解虽快
  1.7~2.8×（C-04，显式选型依据），但弱 CPU 上可能反慢，且 CPU 解码必带争用/
  功耗代价，NVDEC 稳妥优先；批量互补仍需**显式** `decode_backend="cpu"`（C-07/C-08）
- GPU 分段 + ONNX OCR 无净收益，门控只放行 NVDEC+TRT（PERF §9）
- hybrid 已迁 **decord 原生**；本地预路由+池深修复（fork 3b96c6f，未发布）后
  decode-only 近两侧理想和，**av1 上 hybrid 是引擎内最快路径**；h264/hevc
  引擎内 ≈ 最优单侧 ±3-5%（消费端上限）；0.8.2 pip wheel 不含这些修复（C-05）
- **racelog_test 全部视频测量/验证一律 `sample_stride=1`**（2026-09-10 重申，
  防漏信息）；stride>1 仅用于字幕场景（字幕更新频率慢，如批量剧集字幕提取）

## 编辑护栏（docs/log/PERFORMANCE.md）

✅ **2026-08-31 已清除全部 934 个裸 `\r`**（提交 `fd2a76a`），现为**纯 LF 文件**，
裸 CR = **0**。**任何工具都安全** —— 文本模式（含默认 `newline=None`）、Edit 工具、
二进制模式，对纯 LF 文件都 100% 保真。

那 934 个 CR 是什么：每个 `\r` 与下一个 `\n` 之间只有空格或一个重复的 `>`，
**无一后跟正文**。CommonMark 把裸 `\r` 也当行结束符 → 154 处块引用被劈成多段、
762 处意外硬换行，**当时有 934 处渲染错误**，只是没人读渲染结果才没暴露。

- 修法（若哪天又出现）：`re.sub(r'\r(>?[ ]*)(?=\n)', '', txt)`
- 自检：`open(p,'rb').read().count(b'\r') == 0`
- 回归防护：`tests/test_docs_hygiene.py`

## 纪律与自动化守卫

**改完代码跑一次**（12 项，退出码非 0 即违规；关键项另有单测守护）：

```bash
python tools/_probe_discipline_audit.py          # 全量
python tools/_probe_discipline_audit.py --only 1,3,7
python tools/_probe_index_audit.py               # tools/INDEX.md 数字一致性
```

几条容易踩、且已自动化的：

- **路径不许写死**。仓库内路径一律 `__file__` 推导；外部测试视频走环境变量
  （`RACELOG_VIDEO_DIR` / `RACELOG_BATCH_DIR`），硬编码值只能当默认值。
  ⚠️ **`WORKER = r"""..."""` 子进程模板是特例**：它以 `python -c` 执行，
  **`-c` 下 `__file__` 未定义**，必须靠父进程注入 `PROBE_ROOT` 环境变量
  （`_probe_cpu_onnx.py` 例外：它做新旧版本 A/B，子进程 cwd 就是旧 worktree，
  **必须插 cwd** 才不会变成"新代码 vs 新代码"）。
- **`except … pass` 不许无声**。要么 `logger.debug`，要么 `pass  # 为何可忽略`。
  存量 63 处登记在 `tools/_discipline_baseline.json` —— **存量豁免、增量严格**。
  改到某个文件时顺手补注释，它自然脱出 baseline。
  `except BaseException` 只在清理路径允许（否则连 Ctrl-C 一起吞）。
- **产品代码的 print 必须受 debug 开关保护**（`env_bool(DEBUG_BOUNDS_ENV)` /
  `self._probe`），否则走 `logging`。docstring 里的用法示例不算。
- **未使用的 import**：有意 re-export 加 `# noqa: F401`，否则删掉。
- **文档裸 CR = 0、AGENTS.md ≤ 12 KB**：`tests/test_docs_hygiene.py` 守护。

⚠️ **旧规矩双向都错，别再照着做**：
- 「只能用二进制」—— 从来不是必需。文本模式显式 `newline=''` 或 `'\n'` 就
  100% 保真，唯一致命的是默认 `newline=None`（通用换行翻译，把行中 CR 全转
  成 `\n`，文件从 3450 行劈到 4366 行）。实测见 `tools/_probe_cr_roundtrip.py`。
- 「Edit 工具不安全」—— 它只做 CRLF→LF 规范化，而 `.gitattributes` 就是
  `* text=auto eol=lf`，那正是 git 期望的行为。

## 环境与命令

- Python：`c:/Users/eric chen/AppData/Local/Programs/Python/python313/python.exe`
  （**PATH 上的 `python` 缺 numpy，不是项目环境**）。pytest / numpy / psutil /
  cuda.bindings / decord 均可用。
- 硬件：RTX 4060（**单 NVDEC 单元**）/ 16 物理核 32 逻辑核 / 2×16GB DDR5-6000
  （实测流式上限 **55.8 GB/s**；WMI 的 `Speed`=5600 是 SPD 标称，
  `ConfiguredClockSpeed`=6000 才是实际值）。
- 测试视频 `D:\Videos\racelog_test\`，真值在 `ground_truth_csv/`
  （头是 `# roi=...`，**必须用正则取四个整数**，按逗号切只能拿到第一个）。
  ⚠️ **该目录全部视频一律 `sample_stride=1`**（防漏信息）；`stride>1` 只
  用于字幕提取场景（`text_text`/`batch_test`，字幕更新频率慢）。
- 中文输出需 `sys.stdout.reconfigure(encoding="utf-8")`（GBK 控制台会崩）。
- `tools/` 子目录下 `sys.path[0]` 是 tools/，探针须 `sys.path.insert(0, 上级目录)`。
