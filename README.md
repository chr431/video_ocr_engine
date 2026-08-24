# video-ocr-engine

从视频**固定区域**提取文本的通用引擎：`解码 → 像素分段 → 代表帧 → OCR 文本（含置信度）`。

- **输出只有文本与置信度**——不解析领域含义。速度数字识别、字幕提取、固定字幕条
  OCR 等通用场景直接可用；上层应用自行做数值解析/纠错。
- **零领域语义**：构造参数仅引擎域（`video_path / roi / frame_start / frame_end /
  decode_backend / ocr_backend / buffer_size / fill_width ...`），不含速度/纠错参数。
- **解码器只输出 ROI**（自建 [chr431/decord](https://github.com/chr431/decord) fork 的
  ROI-first 管线），非识别区域不参与转换，显著提速。
- 自 [RaceVideoToLog](https://github.com/chr431/RaceVideoToLog) v2.15.2 拆分独立。

## 安装

### 直接使用源码（推荐，也被 RaceVideoToLog 以 submodule 方式挂载）

引擎仓库根目录即 Python 源码根目录，把本目录加入 `PYTHONPATH` / `sys.path` 即可：

```python
import sys
sys.path.insert(0, r"path\to\video_ocr_engine")   # 仓库根目录
from video_ocr_engine import FieldExtractor
```

### pip 安装

```bash
pip install -e .            # 或 pip install .（自带 OCR 模型，~21MB）
```

Python 3.11+。运行时依赖 `numpy / onnxruntime / psutil`。

### 解码后端（decord fork，必需）

解码使用自建 fork（`chr431/decord`，支持 NVDEC GPU 硬解码、ROI-only 输出、
`yuv420` 输出）。PyPI 版 decord 不支持。本地运行需安装 fork：

- 方式一：运行 `chr431/decord` 仓库的 Release workflow，把发布产物解压后将其
  Python 层 + DLL 装入环境（RaceVideoToLog 的 `setup_venv.bat` 即此做法）。
- 方式二：从源码构建 fork 并 `pip install`。

> GPU OCR（TensorRT）还需 CUDA Toolkit + TensorRT 并加入 PATH；缺失时自动回退
> CPU（onnxruntime）。

## 用法

```python
from video_ocr_engine import FieldExtractor

ex = FieldExtractor(
    video_path="subtitle_episode.mkv",
    roi=(10, 850, 1910, 940),        # 字幕条区域 (x1, y1, x2, y2)
    frame_start=0,                    # 可选
    frame_end=None,                   # 可选
    decode_backend="auto",            # auto/cpu/nvdec
    ocr_backend="cpu",                # auto/cpu/tensorrt
    yuv_output=True,                  # 代表帧保留 YUV（转 RGB 预览用）
    keep_crops=True,                  # 是否在结果中保留每段代表帧图像
    keep_frames=True,                 # 是否在结果中保留每段帧号序列
    merge_similar=True,               # 是否合并视觉相似相邻段（默认开启）
    merge_similar_threshold=3.0,      # 相似判定阈值（灰度平均绝对差）
    merge_text_sep="binary",          # 相似帧合并用的分离方案（默认 binary）
)
result = ex.extract()

print(result.fps)
for seg in result.segments:
    if seg.text:
        print(f"frames {seg.start}-{seg.end}  text={seg.text!r}  conf={seg.confidence:.4f}")
```

`decode_backend="auto"` 的默认逻辑：**优先 NVDEC，不可用时回退 CPU**。在强多核
CPU 且片源为 h264 时，可手动选 `"cpu"` 获得更高软解吞吐（NVDEC h264 解码器约
2Gp/s 上限，FFmpeg CPU 解码器最多可利用约 13 核）；弱 CPU / HEVC / AV1 场景仍
建议保持 `auto` 或 `nvdec`。

`result` 为 `ExtractionResult`：

| 字段 | 含义 |
|------|------|
| `segments` | `list[ExtractedSegment]`：`start/end/frames/rep_frame/text/confidence/rep_crop` |
| `frames` | 全部采样帧号 |
| `fps` | 自测帧率（从解码器读取，忽略外部传入） |
| `timing` | 各阶段耗时 |
| `meta` | `backend / ocr_backend / codec / n_segments` |

## 分频采样（sample_stride 参数）

`FieldExtractor(sample_stride=N)`（默认 1）：`>1` 时只解码/分段/OCR 每个第 N 帧——
字幕等 ROI 更新较慢时显著降低处理压力，时间戳仍取真实帧号（准确度基本不变）。
需 decord fork ≥v0.7.12 的 `GetBatch` 等差步长快速路径（顺序流式跳帧）；旧版退化
为逐索引 seek（仍正确但 AV1/HEVC 上更慢）。`stride=1`（默认）与 RaceVideoToLog
完全兼容（零改动）。

> 长视频/大 ROI 场景若不需要预览图，可设 `keep_crops=False`、`keep_frames=False`
> 显著降低内存占用（默认 `True` 保持兼容）。

## 单实例双完整流水线并行（默认关闭）

`FieldExtractor` 可以在一个实例内把同一视频切成多个连续小片，由两条完整
“解码 → 分段 → OCR”流水线作为消费者从队列动态取片，最后按帧序合并。与旧
“只并行解码或只并行 OCR”的方案不同，这里每条流水线都有独立的解码器与 OCR
引擎，能同时利用 CPU 与 GPU。

```python
ex = FieldExtractor(
    video_path="subtitle_episode.mkv",
    roi=(10, 850, 1910, 940),
    decode_backend="auto",       # 主流水线
    ocr_backend="auto",
    gray_output=True,
    merge_similar=True,
    dual_pipeline=True,          # 开启单实例双流水线
    # 可选：自定义两条流水线后端；默认主 + 互补（GPU/TRT ∥ CPU/ONNX）
    # dual_backends=[("auto", "auto"), ("cpu", "cpu")],
)
result = ex.extract()
```

- 默认 `dual_pipeline=False`，保持原有单流水线行为。
- 需要 **NVDEC 和 TensorRT 同时可用**；不满足时自动回退单流水线并发出警告。
- 默认互补对 = `(auto, auto) ∥ (cpu, cpu)`：一条 GPU+NVDEC+TensorRT，
  一条 CPU+ONNX，与下游 `video_subtitle_extractor --dual` 的互补策略一致，
  分别利用 GPU 与 CPU 硬件。
- 分片方法固定为 **kfe（每关键帧一片）**：头部试点×2 + 确认×2 小片预留
  （消解启动竞态并给让位判定取样），试点之外的大竞争区按剩余区域内的每个
  关键帧边界切一片交给共享队列竞争——关键帧过密时按
  `DUAL_KEYFRAME_EVERY_MIN_GAP` / `DUAL_KEYFRAME_EVERY_MAX_CHUNKS` 放大
  间距合并、片数受控，无关键帧时退化为单大竞争片。旧的“等分 N 片（dual-2）/
  按试点比例分配 / 在线优先取片”分片方法已移除。
- 混配（TRT ⊕ ONNX）默认让位阈值取 `DUAL_PIPELINE_MIXED_SLOW_RATIO=0.5`：
  两条流水线分属不同硬件，阈值过高会误让、过低/0 又无法在 AV1 极端失衡时
  止损；可用 `DUAL_SLOW_RATIO` 覆盖。
- 采样帧数 < `DUAL_PIPELINE_MIN_FRAMES`（默认 3000）时自动回退单流水线：
  双流水线的固定开销（探测/校准、第二套引擎初始化、跨片边界）在短窗口无法摊销。
- AV1 编码下 CPU 软解已知净负，默认组合自动回退单流水线
  （`DUAL_NO_CODEC_FALLBACK=1` 可关闭）。
- `dual_backends` 可显式指定两条流水线的 `(decode_backend, ocr_backend)`；
  只给一条时自动复制为两条。
- 本机实测（16 核 + RTX 4060 Laptop）：标清 h264 字幕批量 **-27%**、
  Race 速度数字全片 **-27% ~ -42%**；输出与单流水线逐段一致
  （跨片边界的相似段会被缝合合并）。

> 面向字幕提取的完整 CLI 应用已拆到独立仓库
> [chr431/video_subtitle_extractor](https://github.com/chr431/video_subtitle_extractor)：
> 提供 `--roi`/`--start-frame`/`--end-frame`/`--sample-stride` 等参数，输出
> `time_sec,text` 两列 CSV。本引擎仓库保持为通用引擎（不携带具体场景 CLI）。

## 识别链

1. 校准：前 `SEG_CALIB_FRAMES` 帧 Otsu 求二值化阈值（仅在变化显著时切段）。
2. 分段：ROI 灰度逐帧异或 + 3×3 聚类判别（`_cluster_win3`，纯 numpy）。
3. 代表帧：段内灰度 std 最大者为最清晰帧。
4. OCR：代表帧 → 48 高 resize + 灰度 gamma 2.0 → PP-OCRv6_small（ONNX/TensorRT）。

解码∥分段∥OCR 三级流水线 + 有界队列背压（`OCR_BATCH_SIZE` / `buffer_size`），
解码与 OCR 线程重叠摊薄墙钟。

### 显存全驻留管线（gray + NVDEC + TRT 时默认启用）

`gray_output=True` 且 NVDEC+TRT 可用时，识别链自动切换为**显存全驻留**
路径：NVDEC 解码、灰度、sharp/聚类分段、Otsu 校准、OCR 预处理、TensorRT
推理、CTC 预归约全部在 GPU 内闭环，过 RAM 的只有每帧两个标量、校准直方
图表与归约后的索引/概率。分段/合并/输出与宿主路径逐位一致。

- 干净环境小幅更快（窗口实测 -13%），对端大内存流量时显著更稳
  （对端 ~100GB/s 流拷贝下 -24%，退化 ×1.43 vs 宿主 ×1.64）
- 整集 stride=8 场景两路径同受 NVDEC 跳帧解码供给率限制，速度持平
- env `GPU_PIPELINE=0` 显式关闭；YUV 输出场景仍走宿主管线

## 环境变量钩子（实验）

| 变量 | 作用 |
|------|------|
| `GPU_PIPELINE` | GPU 全驻留管线：`0` 关闭回退宿主路径（gray+NVDEC+TRT 时默认启用） |
| `GPU_CTC` | `0` 关闭 TRT 输出的 GPU argmax 归约（默认开启，仅影响 GPU 管线） |
| `OCR_THREADS` | OCR 推理线程数覆盖（默认全物理核） |
| `OCR_BATCH` | OCR 批大小覆盖（默认 16） |
| `OCR_PAD_SMALL` | OCR 输入 pad 宽度下限覆盖 |
| `OCR_GAMMA` | OCR 预处理 gamma（默认 2.0） |
| `DUAL_ONNX` | `0` 关闭双 ONNX 实例（CPU 核数≥8 默认开） |
| `DUAL_PIPELINE` | `1` 开启单实例双完整流水线并行（需 NVDEC+TRT，默认关闭） |
| `DUAL_KEYFRAME_EVERY_MIN_GAP` | kfe 最小片间距（采样帧数，默认 16）：间距小于该值的关键帧不切分 |
| `DUAL_KEYFRAME_EVERY_MAX_CHUNKS` | kfe 竞争片数上限（默认 8）：关键帧过密时逐步放大间距合并，防止片数上百导致 seek 总耗时线性暴涨 |
| `DUAL_PIPELINE_INFLIGHT` | 竞争取片 in-flight 上限（片数，默认 1）：本流水线“已取但 OCR 尚未排空”的片数达到上限即暂停取片等自己 OCR 追上来——防止“解码快、OCR 慢”的路径在自由竞争中抢占过多切片却因 OCR 瓶颈拖慢整体 |
| `DUAL_PIPELINE_SEEK` | 显式 seek 控制：默认仅 CPU 软解做显式 seek_accurate（NVDEC 硬解 get_batch 内部随机定位更便宜且不衰减）；`1` 强制全部显式（旧保守）；`0` 全部跳过（实验，仅 h264 小片略快、字幕/CPU 随机访问会大幅变慢） |
| `DUAL_SLOW_RATIO` | 双流水线让位阈值覆盖（混配默认 0.5，可显式覆盖） |
| `DUAL_NO_CODEC_FALLBACK` | `1` 关闭 AV1 编码时的双流水线自动回退 |
| `TEXT_SEP_MERGE` | 相似段合并使用的分离模式（contrast/binary/off） |
| `ENGINE_PROFILE` | `1` 开启引擎细粒度性能剖面 |
| `TRT_SUBPROBE` | `1` 开启 TRT 子相位探针 |
| `DEBUG_BOUNDS` | `1` 打印分段边界调试信息 |

## 文档

- [性能调优记录](docs/PERFORMANCE.md) —— 性能基线、后端矩阵、线程预算、已锁定参数、已验证死路。
- [依赖与运行环境](docs/DEPENDENCIES.md) —— decord fork / TensorRT / onnxruntime 版本与注意事项。

## 测试

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

纯单元测试无需视频 / decord / GPU（OCR 用例用 onnxruntime CPU + 仓库内模型）。
解码集成测试在缺 decord 时显式跳过。

## 许可证

**Apache-2.0**。本引擎是独立通用库：不依赖 Qt/GUI 组件，无 copyleft 传染，
可自由用于开源与商业项目（其依赖 decord / PP-OCRv6 亦为宽松或兼容许可）。

> 来源说明：引擎由 RaceVideoToLog（GPL-3.0）拆分而来，代码为原作者原创作品，
> 拆分时已重新授权为 Apache-2.0。RaceVideoToLog 本身因依赖
> PySide6-Fluent-Widgets（GPLv3）仍保持 GPL-3.0，但可自由包含本引擎（submodule）。
