# API 迁移指南（v0.11 → v0.17）

> 面向 `RaceVideoToLog` / `video_subtitle_extractor` 及任何直接使用本引擎的
> 下游。**未列出的部分 = 语义不变**；全部变更由金标向量（28 用例逐位一致）
> 与 API 快照测试背书。

## TL;DR

绝大多数下游**零改动可跑**：`FieldExtractor` / `ExtractedSegment` /
`ExtractionResult` / `meta` 9 键 / `timing` 3 键 / env 旋钮名全部原样。
需要动作的只有四类：①导入路径迁移（旧路径 0.14.0 删除）；②若你依赖
"构造后改 env 仍生效"或"env 盖过构造参数"；③若你读取 OMP_WAIT_POLICY
被引擎隐式设置；④0.16.0 删除面（pool / v1 逃生门 / 实例兼容属性）。

**0.16.0（2026-09-27 R2 API 定型轮，破坏性收口）**：
- **删除 `ExtractionPool`（`video_ocr_engine.pipeline.pool` 整模块）**——
  0.15 公告废弃、原定 0.16 删除（0.15 未发版，废弃窗口实际在
  0.14.1→0.16.0 开发期）；批量改「逐文件 `decode_backend="hybrid"`
  顺序跑」（实测快于任何跨视频并发，C-53）。
- **删除 env `VOE_ENV_WINS`**（v1 `env>参数` 逃生门，原计划 0.14.0）：
  设了不再有任何效果（不再发 DeprecationWarning）。优先级恒为
  **显式构造参数 > env > 默认**。
- **删除实例兼容属性 `ex.frames` / `ex.crops` / `ex.timing`**（与返回值
  双真相是误用源）——分别改读 `result.frames` / `result.segments[i].rep_crop` /
  `result.timing`。**`ex.profile` 保留**：`ENGINE_PROFILE=1` 的唯一读面。
- **新增 `result.report`**（RunReport v6 的类型化只读视图：
  `result.report.spans / .gauges / .health / .pipeline`...；序列化唯一
  形态仍是 `meta['report']` 的 dict，JSON sidecar/registry 不变）。
- 引擎内部结构重组（R1：门面拆解 + 端口接线 + SegmentEngine 注入式编排）
  **不影响公共 API**——金标 28/28 逐位一致背书。

**0.15（2026-09-20 批量策略轮，未单独发版）**：`ExtractionPool.run` 标废弃
（`DeprecationWarning`，0.16.0 删除）——批量改「逐文件
`decode_backend="hybrid"` 顺序跑」（实测快于任何跨视频并发，README 批量
章）；新增 `ocr_backend="hybrid"`（双车道 TRT+OpenVINO，实验性，OCR-bound
负载 v0 实测慢于单 TRT，勿用于生产）。

**0.13.2（S6 性能轮）只做加法**：`meta` 新增 `report` 键（第 10 键）、新增
`FieldExtractor.warmup()` 与 `VOE_REPORT_FILE` 旋钮；无参数语义变更。


---

## 1. 导入路径迁移（六根模块 → 包内）——**已删除（0.14.1）**

| 旧（DeprecationWarning） | 新 |
|---|---|
| `import engine_config` / `from engine_config import X` | `from video_ocr_engine.config import constants`（或直接 `from video_ocr_engine.config.constants import X`） |
| `import segmentation` / `from segmentation import X` | `from video_ocr_engine.domain import segmentation` |
| `import video_utils` / `from video_utils import nv12_to_rgb` | `from video_ocr_engine.domain.video_utils import nv12_to_rgb` |
| `import ocr_native` / `acquire_ocr_engine` | `from video_ocr_engine.ocr.native import acquire_ocr_engine` |
| `import ocr_trt` / `TrtEngine` | `from video_ocr_engine.ocr.trt import TrtEngine` |
| `import gpu_setup` / `ensure_gpu_initialized` | `from video_ocr_engine.gpu.context import ensure_gpu_initialized` |

- 旧路径是**模块别名式 shim**：全部符号（含下划线名）可用、模块同一性
  保持（`import segmentation as s; s is video_ocr_engine.domain.segmentation`
  为 True）——monkeypatch/别名逻辑不受影响。
- 时间线（Q3 裁决）：0.12.x 起 DeprecationWarning；**0.14.0 起删除**——
  实际落地在 **0.14.1**（先完成下游迁移：RaceVideoToLog 33 文件 59 行、
  video_subtitle_extractor 无代码依赖，2026-09-19）。**上表旧路径自
  0.14.1 起不可导入**，请按右列迁移。
- `engine_config.__version__` / `OCR_THREADS_ENV` 等常量：从
  `video_ocr_engine.config.constants` 导入，名字不变。

## 2. 行为变更（§10.2 声明项）

### 2.0 CPU OCR 引擎：onnxruntime → OpenVINO（2026-09-14，0.14.0）
- **旧**：CPU 推理 = onnxruntime（`ocr_backend="cpu"`）。
- **新**：CPU 推理 = **OpenVINO 唯一**（模型级 2.1×、真值门禁三内容族
  零差、h264-cpu 热池 −27.96%；`OCR_CPU_BACKEND` 旋钮随移除删除——
  从未随版本发布，零迁移面）。**onnxruntime 依赖移除**：CPU 路径用户
  需 `pip install openvino`（缺席时构造报错并提示）。
- `engine_type="onnxruntime"` 字符串保留为 CPU 路径标识（API 不变）。
- **模型文件 = fp16 权重版**（2026-09-14）：canonical `PP-OCRv6_rec_small.onnx` 即半精度权重（21.2→10.7MB）；真值代价 ±1 帧/片（六片 47134 帧双向抖动，无系统回退）。
- 证据链：log 2026-09-14-OpenVINO模型级A-B / 集成轮 / 移除轮。

### 2.1 配置在构造期一次冻结（D6）
- **v1**：几乎全部 env 旋钮调用期读取，README 称"构造之后再改 env 同样
  生效"。
- **v0.13**：`FieldExtractor(...)` 构造时经 `config.resolve()` 一次解析并
  冻结（RunConfig）；**构造后改 env 不再生效**。需要换配置 → 新建实例。
- 影响旋钮：autocrop×3 / reorder_window / OCR_THREADS / DECODE_THREADS /
  HYBRID_CPU_THREADS / GPU_PIPELINE / GPU_PIPELINE_STREAM / TEXT_SEP_MERGE /
  OCR_GAMMA / OCR_BATCH / OCR_INSTANCES / GPU_CTC / OCR_PAD_SMALL /
  ENGINE_PROFILE / DEBUG_BOUNDS。
- **残留**：`TRT_SUBPROBE` 仍为 import 期读取（模块级开关）。

### 2.2 优先级反转：显式参数 > env > 默认（Q5）
- **v1**：`OCR_PAD_SMALL` 盖过 `fill_width`、`TEXT_SEP_MERGE` 盖过
  `merge_text_sep`（README 自称排查陷阱）。
- **v0.13**：显式传参即锁定；参数缺省时 env 照常生效。
- **逃生门**：`VOE_ENV_WINS=1` 恢复 v1 语义（DeprecationWarning），
  **0.14.0 移除**——请尽快迁移。

### 2.3 其他已生效变更（0.12.0 起）
| 变更 | v1 | v0.13 |
|---|---|---|
| 同实例二次 `extract()` | 带上一次降级原因/计时 | 每次 run 全量重置（B1） |
| 未知 `decode_backend`/`ocr_backend` | 静默走 CPU / 当 TRT | 构造期 `ValueError`（B3） |
| `import ocr_native` 副作用 | 改写 `OMP_WAIT_POLICY=PASSIVE` | 已删除（D7）——需要的使用方自行在导入前设置 |
| GPU 管线 autocropper/y_pool | 会话启动后赋值 | 构造期装配（B4/B5）；`__del__` 不再执行 CUDA 释放（B6） |
| `meta` 键 | 9 键 | 9 键不变（v2 文档原写 10 为勘误） |

## 2b. 遥测两档化（0.22.0，破坏性之一：默认行为变更）

| 变更 | 迁移 |
|---|---|
| **产品默认档 std→off** | `extract()` 默认不再产出 `meta['report']`（发布=全关）。需要报告的调用方显式 `VOE_TELEMETRY=full`（调试档：直方图+NVML+线程账本全开） |
| **std 档退役** | `VOE_TELEMETRY=std` 按受谴责别名映射为 full 并告警；**0.24.0 起拒收**。直接构造 `Metrics("std")` 已抛 ValueError（写 off/full） |
| `PI15_LIMITS.std_pct` 删除 | bench 遥测门禁只剩 full_pct=1.20；`telemetry-check` 两档互比（off vs full），`--std-limit` 参数删除 |
| bench `--telemetry` 默认翻 full | run/ab 默认带全套遥测（开销实测 +0.079%±0.064%，噪声内） |

## 3. 内部结构（仅当你的代码伸进了私有面）

| v1 私有面 | v0.13 去向 |
|---|---|
| `video_ocr_engine._host_pipeline` / `_ocr_session` | 已删除；实现迁 `pipeline/host_backend.py` / `pipeline/ocr_stage.py`（OcrSession 吃显式 SessionSpec） |
| `_GpuPipelineMixin` / `_HostPipelineMixin` | mixin 已消灭（方法并入 FieldExtractor / pipeline） |
| `FieldExtractor._run_pipelined_host/_gpu` | 仍在（门面：构建 Spec → 调 pipeline 后端）；驱动主体在 `pipeline/{host,gpu}_backend.py`（HostRunSpec/GpuRunSpec 显式契约） |
| monkeypatch 点 `_gpu_pipeline.nvdec_available` 等 | **保留**（模块级名字未动） |
| `TrtEngine.execute/execute_device`、`similar_binary`、4 个键集常量 | 死代码已删，不会复活（审计守卫） |
| `video_ocr_engine._gpu_pipeline` | 迁 `video_ocr_engine/gpu/device.py`（S9-5）；`nvdec_available`/`tensorrt_available` patch 点随之移动 |
| `video_ocr_engine._ocr_session` / `_host_pipeline` | shim/模块已删；实现见 `pipeline/ocr_stage.py` / `pipeline/host_backend.py` |
| 队列/infer 载荷（5 元组） | `SegmentTask` / `InferBatch`（`pipeline/ocr_stage.py`，S9-3）；OCR 结果 `OcrResult` (NamedTuple) |

## 4. 新增面（可选使用）

- `video_ocr_engine.config.resolve()` / `RunConfig`——唯一 env 读取点；
- `pipeline.SegmentEngine` / `RunOutcome`、`decode.port.FrameSource` /
  `ocr.port.OcrBackend` 协议（实验性，签名未冻结）；
- `pytest -m gpu`：宿主↔GPU 双后端等价门禁；
- `tests/golden/`：28 用例金标向量（`record.py --verify` 复验）。

### 4.1 S6 性能轮新增（0.13.2，只增不改）

| 面 | 说明 |
|---|---|
| `result.meta["report"]` | RunReport（schema v1）：`spans`/`counters`/`gauges`/`health`/`environment`/`pipeline`/`degradations`。**默认开（std 档）**，实测开销 −0.15%（噪声内）。`VOE_TELEMETRY=off` 则无此键 |
| `FieldExtractor.warmup()` | 显式预热 OCR 引擎池（首个 extract 的 `engine_init` 0.39s→0.0001s）。不自动触发、不改 `extract()` 行为、总吞吐不变（C-24） |
| `VOE_REPORT_FILE` | 把 RunReport 细档 JSON 写到指定路径（空=不写；写文件是副作用，须显式 opt-in） |
| `tools/bench.py` | 报告矩阵 / `diff`（D10 双档）/ `ab`（交错 A/B，对抗机器漂移）/ `telemetry-check`（PI-15） |
| `reorder_window` 生效值 | **行为变更（性能向，结果不变）**：pad 下限支配 ROI 时（ROI 上界宽高比 ≤ `fill_width/48`），按宽分组的等待窗口自动收敛到 1（实测热轮 −5.05%）。宽 ROI/`force_aspect>0` 场景保持原窗口。金标 28/28 逐位一致 |

### 4.2 S6 续轮新增（0.13.3，只增不改）

| 面 | 说明 |
|---|---|
| RunReport **schema v10** | `report_version` 9→10，只加键（full 档）：`resources.per_phase` 行新增 `thr_foreign`/`thr_foreign_n`（外来线程簇逐 TID 周期差分：decord 解码池/OMP/TBB 等命名线程之外的全体——争用定位的外来簇归因面）与 `cycles_e2e.threads_foreign` 合计。个体 TID 只用于差分，结论应下在簇级 |
| RunReport **schema v9** | `report_version` 8→9。**删键语义**：Windows 报告的 `resources.per_phase` 行不再发 `cores_avg`（tick 口径）——cycles 在场时它是与 `cores_avg_cycles` 并存的重复口径；消费者改读 `cores_avg_cycles`。非 Windows（cycles 缺席）仍发 `cores_avg` 作回退 |
| RunReport **schema v2** | `report_version` 1→2，**只加键**：`resources`（std+：每相位平均并行核数/线程数/RSS·VRAM 增量/磁盘读写速率 + 每来源真出处或 `unavailable:原因`）、`hardware`（仅 full 档采样过才出现：GPU%/NVDEC%/显存 min·p50·p99·max）。按 v1 解析的旧读者不受影响；金标只记 `meta` 键名，故无需重录（28/28 逐位一致已验） |
| `hybrid` CPU 线程档位 | 默认 12→**16**（核数//2 钳 [8,16]），并取消按 decord 版本号的门控（对 `DECORD_LIBRARY_PATH` 换 dll 的情形判错）。交错 A/B：h264-hybrid −6.5%、hevc-hybrid −12.7%；`HYBRID_CPU_THREADS` 显式覆盖仍有效 |
| PI-15 门禁校准 | 判据改"同进程交替 + 档位轮转 + 同轮配对差分**均值** + 符号多数一致"，阈值按本机 A/A 标定（**2026-09-13 重标：std +0.30% / full +1.20%**；规则 `\|偏差\|+3×SE`）。§13.2 设计目标 +0.1%/+1% 仍打印。**换机器后需 `bench.py telemetry-check --aa` 重标**。µs 级严格性移至 `tests/config/test_telemetry_cost.py`（插桩路径重放 ≤2ms/run） |
| 环境指纹 GPU 来源 | 改走 NVML（43.1ms→20.8ms、免子进程），失败回退 nvidia-smi；新增 `environment["gpu_source"]`（`nvml`/`nvidia-smi`/`unavailable`） |
| **不新增**：PCIe 速率 | 本机不可直读，v2 报告**不产该字段**（原承诺收窄为 L3 推导：counter 字节 ÷ 相位墙钟，由使用者自行换算） |

## 5. 发布节奏

| 版本 | 内容 |
|---|---|
| 0.12.0 | S0–S8：编排契约化、mixin 消灭、知识库、21 项审计 |
| 0.13.0 | S9：模块入包 + shim、D6/Q5 激活（本指南 §1/§2.1/§2.2） |
| 0.13.1 | S9 续：载荷类型化（DeviceRef/SegmentTask/InferBatch）、设备层迁 `gpu/`、打包修复（子包入 wheel） |
| 0.13.2 | S6：原生测量系统（RunReport + `meta['report']` + `bench`）、显式 `warmup()`、归约 D2H 异步化、keep_crops D2H 并批、分组窗口自适应（§4.1） |
| **0.13.3** | S6 续：**资源层落地**（`report["resources"]` L1 相位差分、full 档 `report["hardware"]` NVML）→ **RunReport schema v1→v2（只加键）**；hybrid CPU 线程档位 12→**16**（h264 −6.5% / hevc −12.7%，取消版本号门控）；PI-15 门禁按本机 A/A 校准（§4.2）；修 off 档仍查注册表的缺陷 |
| 0.14.0（计划） | 删除六根模块 shim、`VOE_ENV_WINS`；届时无新破坏 |

## 6. 0.17.0：OCR 标识与报告键更名（破坏性，无兼容层）

> 2026-09-28 审计轮：名实不符清理。下游若有命中，按本节改名即可；
> 引擎不提供任何兼容别名。

### 6.1 OCR 引擎标识 `'onnxruntime'` → `'openvino'`

C-48（2026-09-14）移除 ORT 后，`'onnxruntime'` 字符串仍作为 CPU/OpenVINO
路径的内部标识残留一轮。0.17.0 起统一为 `'openvino'`：

| 面 | 旧 | 新 |
|---|---|---|
| `FieldExtractor._ocr_engine_type()`（`ocr_backend='cpu'` 时） | `'onnxruntime'` | `'openvino'` |
| `acquire_ocr_engine(..., engine_type=)` 合法值 | `'onnxruntime'` | `'openvino'` |
| `OcrEngine(..., engine_type=)` 默认值 | `'onnxruntime'` | `'openvino'` |
| 引擎池 key 中的 type 分量 | `'onnxruntime'` | `'openvino'` |
| `ocr_backend='hybrid'` 的 backend_used 回调值 | `'tensorrt+onnxruntime'` | `'tensorrt+openvino'` |
| 常量 `OCR_ONNX_CHUNK` | — | `OCR_OV_CHUNK`（同值 16） |

`OcrEngine.backend_name` **本就返回 `'openvino'`，无变化**。单测名
`test_onnx_*` 同步更名 `test_openvino_*`。

### 6.2 报告：host 路径停发 `pipeline.decode` span

host 管线的 `pipeline.decode` 与 `pipeline.consumer` 为同区间双键
（遗留等价键）。0.17.0 起 host 只发 `pipeline.consumer`；**GPU 管线的
`pipeline.decode` 保留**（生产者线程成本的真实相位键，非等价键）。
读 host 解码段耗时的下游改读 `pipeline.consumer`（两键数值本就相同，
仅键名变化）。

### 6.3 默认值接线（行为不变）

`DEFAULT_OCR_BACKEND` / `DEFAULT_DECODE_BACKEND` / `DEFAULT_FORCE_ASPECT`
自 0.17.0 起真正作为 `FieldExtractor` 构造签名默认值（此前被同值字面量
绕过）。取值不变，仅语义收敛。

### 6.4 删除（零引用死代码）

`FieldExtractor._decord_format()` 与 `DecordFrameSource.decord_format()`
（全仓零调用的委托链）；金标与全部单测不受影响。

### 6.5 同轮 fork 侧（非引擎 API）

fork dev 构建目录 `build-081fix` → `build-dev`（`DECORD_LIBRARY_PATH`
指向需同步）；fork 根新增 `build_dev.bat`（只构建不部署）。
