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
    decode_backend="auto",            # auto/cpu/nvdec/hybrid
    ocr_backend="cpu",                # auto/cpu/tensorrt
    rep_crop_format="yuv",            # 代表帧格式："yuv"=packed NV12（默认；
                                      # 内部链恒为单通道灰度，外部用
                                      # video_utils.nv12_to_rgb 转 RGB）或 "gray"
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
建议保持 `auto` 或 `nvdec`。若 NVDEC 解码是瓶颈且 CPU 有空闲，可显式选
`"hybrid"`：同一实例内 NVDEC 与 CPU 软解按**实测速率比例分界**并行解码——
快端从头连续扫掠前半、慢端 seek 一次后连续扫掠后半，快端扫完自动接管慢端
剩余片（校准误差自愈）。要求 NVDEC 可用、`sample_stride==1`；不可用时自动
回退纯 NVDEC/CPU。编码不限（含 AV1）：CPU 慢于 NVDEC 的 HEVC/AV1 场景与
纯 NVDEC 持平不退化，CPU 快于 NVDEC 的 h264 场景显著更快（见
`docs/PERFORMANCE.md`）。

`result` 为 `ExtractionResult`：

| 字段 | 含义 |
|------|------|
| `segments` | `list[ExtractedSegment]`：`start/end/frames/rep_frame/text/confidence/rep_crop` |
| `frames` | 全部采样帧号 |
| `fps` | 自测帧率（从解码器读取，忽略外部传入） |
| `timing` | 各阶段耗时 |
| `meta` | `backend / ocr_backend / codec / n_segments` |

> 引擎内部链恒为单通道灰度（解码输出 `yuv420` 或 `gray`，不再输出 RGB 帧——
> RGB→灰度转换由解码侧/fork 完成）。`rep_crop` 的像素格式由
> `rep_crop_format` 决定：默认 `"yuv"` = packed NV12（`video_utils.nv12_to_rgb`
> 转 RGB 预览，BT.601 limited 矩阵，仅代表帧调用、毫秒级）；`"gray"` = 灰度。
> `keep_crops=False` 时自动退化为 `gray` 解码输出（省 UV 传输）。
> 旧参数 `gray_output` / `yuv_output` 保留为 deprecated 别名
> （`yuv_output=True` ≡ `rep_crop_format="yuv"`；`gray_output=True` ≡
> `rep_crop_format="gray"`；两者均未设置时的新默认是 `"yuv"`——旧默认 RGB
> 已移除）。

## 分频采样（sample_stride 参数）

`FieldExtractor(sample_stride=N)`（默认 1）：`>1` 时只解码/分段/OCR 每个第 N 帧——
字幕等 ROI 更新较慢时显著降低处理压力，时间戳仍取真实帧号（准确度基本不变）。
需 decord fork ≥v0.7.12 的 `GetBatch` 等差步长快速路径（顺序流式跳帧）；旧版退化
为逐索引 seek（仍正确但 AV1/HEVC 上更慢）。`stride=1`（默认）与 RaceVideoToLog
完全兼容（零改动）。

> 长视频/大 ROI 场景若不需要预览图，可设 `keep_crops=False`、`keep_frames=False`
> 显著降低内存占用（默认 `True` 保持兼容）。

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

### 显存全驻留零拷贝管线（NVDEC+TRT 时默认启用）

NVDEC+TensorRT 可用（`decode_backend∈{auto,nvdec}`、`ocr_backend≠cpu`）时，
识别链默认切换为**显存全驻留零拷贝**路径：NVDEC 解码、灰度（`yuv420` 时由
`luma_nv12` kernel 在 GPU 提取 Y 平面，与宿主逐位一致）、sharp/聚类分段、
Otsu 校准、merge_similar 判定（GPU `sim_pair`；contrast 模式在边界时 D2H 两
帧走宿主判定）、代表帧保活（gray 直通 decord 指针 / yuv 用 `_YFramePool`
池帧按需提取 Y）、TensorRT 推理（single TRT 引擎，`force_aspect` 已支持）
与 CTC 预归约全部在 GPU 内闭环——过 RAM 的只有每帧两个标量、校准直方图表、
合并判定标量（contrast 时两帧）、CTC 归约结果与 `keep_crops` 输出（每段一
张，结果给外部）。分段/合并判定/输出与宿主路径一致。

- 干净环境小幅更快（窗口实测约 -10%），对端大内存流量时显著更稳
  （对端 ~100GB/s 流拷贝下退化 ×1.36 vs 宿主 ×1.70）
- 整集 stride=8 场景两路径同受 NVDEC 跳帧解码供给率限制，速度持平
- 默认只放行"全程 raw"（NVDEC+TRT）；GPU 分段 + ONNX OCR 实测无净收益
  （见 `docs/PERFORMANCE.md` §9）→ 无 TRT / `ocr_backend="cpu"` 走宿主
- env `GPU_PIPELINE=0` 显式关闭；`=1` 强制启用（含 GPU 分段+ONNX 实验组合）；
  `decode_backend="hybrid"`/`"cpu"` 走宿主

## 环境变量钩子

> 构造参数已覆盖绝大多数用法；环境变量仅在**批量调优/诊断**时使用。
> 未设置 = 引擎默认值（已按实测调优，见 `docs/PERFORMANCE.md`"已锁定参数"）。

### 用户调参（了解影响后再动）

| 变量 | 作用 |
|------|------|
| `GPU_PIPELINE` | 零拷贝管线：未设置 = NVDEC+TRT 时默认启用；`0` 显式关闭；`1` 强制启用（含 GPU 分段+ONNX 等实验组合） |
| `OCR_THREADS` | OCR 推理线程数覆盖（默认全物理核） |
| `OCR_BATCH` | OCR 批大小覆盖（默认 16） |
| `OCR_PAD_SMALL` | OCR 输入 pad 宽度下限覆盖 |
| `OCR_GAMMA` | OCR 预处理 gamma（默认 2.0） |
| `TEXT_SEP_MERGE` | 相似段合并分离模式（contrast/binary/off） |
| `HYBRID_MAX_CHUNKS` | 混合解码分片上限（默认 16） |
| `HYBRID_CPU_THREADS` | 混合解码中 CPU 软解线程数（默认 0=核数//2） |
| `HYBRID_MAX_CHUNK_FRAMES` | 混合解码单片采样帧数上限（默认 0=不拆；>0 时超限片按关键帧/等分拆小，内存上界 = inflight × 上限） |

### 实验/诊断（排查问题时用）

| 变量 | 作用 |
|------|------|
| `GPU_CTC` | `0` 关闭 TRT 输出的 GPU argmax 归约（默认开；仅影响 GPU 管线） |
| `OCR_INSTANCES` | `0` 关闭并行双 ONNX 实例（CPU 核数≥8 默认开） |
| `ENGINE_PROFILE` | `1` 开启引擎细粒度性能剖面 |
| `TRT_SUBPROBE` | `1` 开启 TRT 子相位探针 |
| `DEBUG_BOUNDS` | `1` 打印分段边界调试信息 |
| `HYBRID_PROBE` | `1` 打印混合解码逐片时序（速率校准/分界/接管诊断） |
| `HYBRID_PROBE_CSV` | 设为 CSV 路径时，`HYBRID_PROBE` 逐片时序另落盘一份明细 |
| `HYBRID_CALIB_ROUNDS` | 混合解码速率校准轮数（默认 1；>1 取中位数更稳，但每轮约 +0.3s 成本） |

### 已废弃/兼容（勿再依赖）

| 变量 | 说明 |
|------|------|
| `DECORD_FORCE_CPU` | 旧强制 CPU 解码钩子；仅 `decode_backend="auto"` 时兼容生效，请改用 `decode_backend="cpu"` |

内部实现（`engine_config` 常量、`_gpu_pipeline` 门控等）不在本表；如需深入，
以 `engine_config.py` 为唯一事实源。

## 文档

- [性能调优记录](docs/PERFORMANCE.md) —— 性能基线、后端矩阵、线程预算、已锁定参数、已验证死路。
- [依赖与运行环境](docs/DEPENDENCIES.md) —— decord fork / TensorRT / onnxruntime 版本与注意事项。

## 测试

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

纯单元测试无需视频 / decord / GPU（OCR 用例用 onnxruntime CPU + 仓库内模型）。
解码路径的集成验证不在单元测试内——真实视频/decord 的端到端验收由下面的
`tools/e2e_smoke.py` 承担。

### 端到端冒烟 / 性能工具（真实视频）

`tools/e2e_smoke.py`：同一视频窗口跑配置矩阵（GPU 零拷贝 / GPU 灰度 raw /
keep 关闭 / 宿主+TRT / CPU+ONNX / hybrid / GPU 分段+ONNX），校验跨路径文本
一致性、代表帧格式、段序/置信度，可对照 Race ground-truth CSV 验证匹配率，
并可重复跑测墙钟与分相耗时。示例：

```bash
# 功能矩阵 + 真值验证（ROI/帧区间可自动从 truth CSV 头读）
python tools/e2e_smoke.py --video D:\Videos\racelog_test\test5.mp4 \
    --roi 843,993,948,1025 \
    --truth D:\Videos\racelog_test\ground_truth_csv\test5_ref.csv \
    --frames 5000 --stride 8 --verify

# 只探视频/后端元数据；性能测试（重复跑 + ENGINE_PROFILE）
python tools/e2e_smoke.py --video X --roi A --probe
python tools/e2e_smoke.py --video X --roi A --perf --runs 3 --frames 3000
```

运行 GPU 路径需：decord fork（ROI-first / GPU gray / yuv420）、`cuda-python`
（分段/校准/CTC kernel）、TensorRT thin binding（`tensorrt-cu13-bindings`
+ `tensorrt-cu13` 元包 shim，DLL 从系统 PATH 加载）。

## 许可证

**Apache-2.0**。本引擎是独立通用库：不依赖 Qt/GUI 组件，无 copyleft 传染，
可自由用于开源与商业项目（其依赖 decord / PP-OCRv6 亦为宽松或兼容许可）。

> 来源说明：引擎由 RaceVideoToLog（GPL-3.0）拆分而来，代码为原作者原创作品，
> 拆分时已重新授权为 Apache-2.0。RaceVideoToLog 本身因依赖
> PySide6-Fluent-Widgets（GPLv3）仍保持 GPL-3.0，但可自由包含本引擎（submodule）。
