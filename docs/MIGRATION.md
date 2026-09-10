# API 迁移指南（v0.11 → v0.13）

> 面向 `RaceVideoToLog` / `video_subtitle_extractor` 及任何直接使用本引擎的
> 下游。**未列出的部分 = 语义不变**；全部变更由金标向量（28 用例逐位一致）
> 与 API 快照测试背书。

## TL;DR

95% 的下游**零改动可跑**：`FieldExtractor` / `ExtractedSegment` /
`ExtractionResult` / `meta` 9 键 / `timing` 3 键 / 18 个 env 旋钮名全部原样。
需要动作的只有三类：①导入路径迁移（旧路径 0.14.0 删除）；②若你依赖
"构造后改 env 仍生效"或"env 盖过构造参数"；③若你读取 OMP_WAIT_POLICY
被引擎隐式设置。

**0.13.2（S6 性能轮）只做加法**：`meta` 新增 `report` 键（第 10 键）、新增
`FieldExtractor.warmup()` 与 `VOE_REPORT_FILE` 旋钮；无参数语义变更。


---

## 1. 导入路径迁移（六根模块 → 包内，0.14.0 删除旧路径）

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
- 时间线（Q3 裁决）：0.12.x 起 DeprecationWarning；**0.14.0 删除**。
- `engine_config.__version__` / `OCR_THREADS_ENV` 等常量：从
  `video_ocr_engine.config.constants` 导入，名字不变。

## 2. 行为变更（§10.2 声明项）

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

## 5. 发布节奏

| 版本 | 内容 |
|---|---|
| 0.12.0 | S0–S8：编排契约化、mixin 消灭、知识库、21 项审计 |
| 0.13.0 | S9：模块入包 + shim、D6/Q5 激活（本指南 §1/§2.1/§2.2） |
| 0.13.1 | S9 续：载荷类型化（DeviceRef/SegmentTask/InferBatch）、设备层迁 `gpu/`、打包修复（子包入 wheel） |
| 0.13.2 | S6：原生测量系统（RunReport + `meta['report']` + `bench`）、显式 `warmup()`、归约 D2H 异步化、keep_crops D2H 并批、分组窗口自适应（§4.1） |
| 0.14.0（计划） | 删除六根模块 shim、`VOE_ENV_WINS`；届时无新破坏 |
