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
)
result = ex.extract()

print(result.fps)
for seg in result.segments:
    if seg.text:
        print(f"frames {seg.start}-{seg.end}  text={seg.text!r}  conf={seg.confidence:.4f}")
```

`result` 为 `ExtractionResult`：

| 字段 | 含义 |
|------|------|
| `segments` | `list[ExtractedSegment]`：`start/end/frames/rep_frame/text/confidence/rep_crop` |
| `frames` | 全部采样帧号 |
| `fps` | 自测帧率（从解码器读取，忽略外部传入） |
| `timing` | 各阶段耗时 |
| `meta` | `backend / ocr_backend / codec / n_segments` |

## 示例 CLI（text_extract_cli）

仓库自带一个**仅 CLI** 的文本提取测试程序（`text_extract_cli.py`；pip 安装后可
通过 `ocr-text-extract` 命令调用）。只暴露基本参数（视频 / ROI / 开始帧 / 结束帧 /
输出文件），解码与 OCR 后端用演示默认（decode=auto 自动 GPU/CPU、OCR=cpu 走 ONNX），
适合快速验证引擎在字幕等场景的识别结果。

```bash
# 源码方式（仓库根目录即源码根）
python text_extract_cli.py subtitle_ep.mkv --roi 10 850 1910 940 \
    --start-frame 0 --end-frame 3000 -o subtitles.csv

# pip 安装后
ocr-text-extract subtitle_ep.mkv --roi 10 850 1910 940 -o subtitles.csv
```

输出两列 CSV（`utf-8-sig`，对中文/含逗号文本自动加引号）：

| time_sec | text |
|---|---|
| 12 | 你好，世界 |
| 15 | 我们继续 |

- `time_sec`：段代表帧（识别帧）在视频中的实际秒数（绝对帧号 / 引擎自测 fps，
  四舍五入到秒）。
- `text`：OCR 原始文本，**原样输出**（不做速度解析 / 过滤 / 规整）。

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
| `RVTOL_HYBRID_DECODE` | `1` 开启 CPU+NVDEC 混合解码 |
| `RVTOL_HYBRID_OCR` | `1` 开启 TRT+ONNX 混合 OCR |
| `RVTOL_SEG_GAMMA` | 分段灰度 gamma（默认 0=raw） |
| `RVTOL_OCR_GAMMA` | OCR 预处理 gamma（默认 2.0） |

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
