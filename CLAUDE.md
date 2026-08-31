# CLAUDE.md — 开发记录与约定（注入核）

> 本文件在每个会话开头被注入，**只放"现在必须知道的"**。
> **硬上限 12 KB** — 超了就把内容迁到 `docs/DECISIONS.md`，这里只留指针。

## 文档地图（6 份，别再新增）

| 文件 | 性质 | 什么时候读 |
|---|---|---|
| `README.md` | 用户向 API / 用法 | 写调用代码时 |
| `CLAUDE.md`（本文件） | 维护者向**注入核**：铁律 + 现役架构 + 结论指针 | 自动注入 |
| `docs/PERFORMANCE.md` | 现役性能实测（§1–§15, §17, §19–§21） | 动性能相关代码前 |
| `docs/DECISIONS.md` | 每轮决策过程、已删除功能、设计审查结论 | 想问"为什么这么做"时 |
| `docs/ARCHIVE.md` | 归档（PERF §4 / §8 / §16 / §18），**编号保留勿重编** | 只看"为什么不做" |
| `docs/DEPENDENCIES.md` | 依赖版本与已知问题 | 装环境 / 报 bug 时 |

⚠️ **现役规则以本文件为准**。`docs/DECISIONS.md` 是迁出的原文存档，
两者冲突时以本文件为真相（避免"两套真相"，见设计审查 D6）。

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
8. **新结论一律追加到现役章节，不新建文档**；版本号改动必须打同名 git tag。
9. **向后兼容**：新功能默认关闭，除非明确作为新默认；新增遗留面一律先标
   deprecated、两个版本后删除。

## 现役架构

链路：**解码 → 像素分段 → 代表帧 → OCR → 相似段合并**

| 阶段 | 入口 |
|---|---|
| 解码 | `decord.VideoReader.get_batch` / `hybrid_decode.HybridDecoder.get_batch`（**唯一入口**） |
| 分段 | `segmentation.py` |
| 宿主管线 | `video_ocr_engine/_host_pipeline.py` |
| GPU 管线 | `video_ocr_engine/_gpu_pipeline.py`（gray+NVDEC+TRT 时默认） |
| OCR 调度 / 引擎池 | `ocr_native.py`（`acquire_ocr_engine` / `checkin_ocr_engine`） |
| TRT | `ocr_trt.py` + `video_ocr_engine/_gpu_kernels.py` |
| 配置常量 | `engine_config.py`（`GPU_PIPELINE_DECODE_BATCH=64` 等） |
| 分相打桩 | `video_ocr_engine/extractor.py:317 _prof_end` |

**现役并行维度只有一个**：`decode_backend="hybrid"` 的 CPU+NVDEC 双解码生产者
竞争（`hybrid_decode.py`）。**没有 dual pipeline、没有 `DUAL_*` 环境变量、
没有 `_dual_pipeline.py`** —— 历史提及均为旧档案，勿据此调优。

其他现役事实：

- **引擎内部恒为单通道灰度**：decord 只输出 `'yuv420'`（keep_crops 且
  `rep_crop_format="yuv"`）或 `'gray'`；旧 `gray_output`/`yuv_output` 已删除。
- **相似段合并**在分离图上进行，默认 `binary`（`TEXT_SEP_MERGE=binary|off`）。
- **GPU 管线门控**：gray + `decode∈{auto,nvdec,cpu}` + `ocr≠cpu` + NVDEC/TRT
  可用时自动启用；`GPU_PIPELINE=0` 关 / `=1` 强制。GPU+ONNX 组合实测无净收益。
- **OCR 引擎池**：`_POOL_MAX_PER_KEY=4`、`_POOL_MAX_TOTAL=16`，
  key=(model, type, fill_width, threads)。

## 已封板结论（勿重复投入）

| 结论 | 详见 |
|---|---|
| **IO 不是并发退化原因**：磁盘 IO 占单次墙钟 <1%（0.03–0.05s / 5–7s），PCIe 0.01% | PERF §19 |
| **内存带宽数字被高估 1.8×**：本机 B_max 只有 **55.8 GB/s**（非 ~100） | PERF §20 |
| **并发退化真因 = NVDEC 会话数**（单硬件单元串行）。互补设计（CPU 软解+ONNX ∥ NVDEC+TRT）聚合加速 **1.87×**；两条都走 NVDEC 只有 1.01~1.20× | PERF §21 |
| **解码后端必须按编码选**：h264 CPU 快 2.88×，AV1 CPU 慢 2.56×（7.4× 反转）；解码占管线 98%+ | PERF §21 |
| **冷启动 yuv 格式税不存在**，是测量假象；冷轮多出的 ~0.55s 里 0.50–0.53s 是 TRT 引擎构造 | PERF §15 |
| **分段合并**在不误合并约束下已无普适空间 | PERF §14 |
| **GPU 分段 + ONNX OCR 无净收益**，默认门控只放行 NVDEC+TRT | PERF §9 |
| **真跳帧**（丢 `nal_ref_idc==0` 整包）安全，但收益仅 1.03~1.48×（原估 2~4×） | DECISIONS「下一步三目标轮」 |
| **`skip_loop_filter`** 1.11~1.36×，但改变输出像素；默认关闭（opt-in） | DECISIONS「P0-6 翻案」 |
| **pad 224 保持**：160 已回退（生产误读退化），320 已证伪 | ARCHIVE §16.2 |
| **裁切余量 10% 优于 0%**；裁切即使省不到算力也能提准确率（旧"守卫"前提是错的） | DECISIONS「第四轮」 |
| **`auto` 后端在 h264 多核不是最优**（CPU+TRT 快约 2×），但静态判据不可靠、判错代价成倍 → 保持现状 | DECISIONS 设计审查 A2 |

## 编辑护栏（docs/PERFORMANCE.md）

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
- **文档裸 CR = 0、CLAUDE.md ≤ 12 KB**：`tests/test_docs_hygiene.py` 守护。

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
- 中文输出需 `sys.stdout.reconfigure(encoding="utf-8")`（GBK 控制台会崩）。
- `tools/` 子目录下 `sys.path[0]` 是 tools/，探针须 `sys.path.insert(0, 上级目录)`。
