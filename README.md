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

## 单实例双完整流水线并行（实验，默认关闭）

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
    dual_pipeline_chunks=2,      # 竞争片数，默认 2
    # 可选：自定义两条流水线后端；默认主 + 互补解码（GPU/TRT ∥ CPU/TRT）
    # dual_backends=[("cpu", "auto"), ("cpu", "auto")],
)
result = ex.extract()
```

- 默认 `dual_pipeline=False`，保持原有单流水线行为。
- 需要 **NVDEC 和 TensorRT 同时可用**；不满足时自动回退单流水线并发出警告。
- 默认互补对 = `(auto, auto) ∥ (cpu, auto)`：两条流水线都用 TensorRT 推理，
  仅解码侧互补（NVDEC 有固定吞吐上限，CPU 软解是真实增量）。TRT 与 ONNX
  共存推理会互相膨胀（实测总吞吐反而钉死），因此默认不再混用两种推理后端。
- 采样帧数 < `DUAL_PIPELINE_MIN_FRAMES`（默认 3000）时自动回退单流水线：
  双流水线的固定开销（探测/校准、第二套引擎初始化、跨片边界）在短窗口无法摊销。
- AV1 编码下 CPU 软解已知净负，默认组合自动回退单流水线
  （`RVTOL_DUAL_NO_CODEC_FALLBACK=1` 可关闭）。
- 运行中按“试点片 + 确认片”实测两条流水线的处理速率，明显偏慢的一侧会让位，
  剩余片由快路径完成，避免尾部等待。
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

## 环境变量钩子（实验）

| 变量 | 作用 |
|------|------|
| `RVTOL_OCR_THREADS` | OCR 推理线程数覆盖（默认全物理核） |
| `RVTOL_OCR_BATCH` | OCR 批大小覆盖（默认 16） |
| `RVTOL_DUAL_ONNX` | `0` 关闭双 ONNX 实例（CPU 核数≥8 默认开） |
| `RVTOL_HYBRID_DECODE` | `1` 开启 CPU+NVDEC 混合解码（队列实测无提升，维持关闭） |
| `RVTOL_HYBRID_OCR` | `1` 开启 TRT+ONNX 混合 OCR（队列实测无提升，维持关闭） |
| `RVTOL_DUAL_PIPELINE` | `1` 开启单实例双完整流水线并行（需 NVDEC+TRT，默认关闭） |
| `RVTOL_SEG_GAMMA` | 分段灰度 gamma（默认 0=raw） |
| `RVTOL_OCR_GAMMA` | OCR 预处理 gamma（默认 2.0） |

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
