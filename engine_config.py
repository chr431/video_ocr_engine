"""管线引擎配置（engine_config）— 解码/OCR/分段域常量。

独立引擎仓库的配置单一事实源，供 engine 内 ocr_native / ocr_trt /
segmentation / video_utils 与引擎外的上层应用直接引用。
上层应用可通过 `from engine_config import *` 聚合再导出（GUI 与
`import config` 兼容），应用侧专属常量（颜色/窗口/图表/领域后处理参数）
留在应用侧。

自拆仓（v0.1.0）起：本文件随 video_ocr_engine 独立仓库发布、独立版本线。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.8.1"

# ═══════════════════ 环境变量助手与名称常量 ═══════════════════
# 引擎全部 env 开关/覆写在此收敛（单一事实源）。布尔开关统一走 env_bool：
# 缺省取 default，值匹配真集/假集，未识别值回退 default——与各路径历史
# 语义一致；数值型 env（OCR_THREADS / HYBRID_MAX_CHUNKS 等）只在此定义名称，
# 解析在调用点。

_TRUTHY_VALUES = ("1", "true", "yes", "on")
_FALSY_VALUES = ("0", "false", "no", "off")


def env_bool(name: str, default: bool = False) -> bool:
    """读取布尔开关 env：缺省 → default；值在真/假集 → 对应布尔；其余回退 default。"""
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in _TRUTHY_VALUES:
        return True
    if v in _FALSY_VALUES:
        return False
    return default


def env_int(name: str, default: int) -> int:
    """读取整数型 env：缺省/空/非法 → default（解析收敛点，调用点不再各自 int()）。"""
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """读取浮点型 env：缺省/空/非法 → default（解析收敛点，调用点不再各自 float()）。"""
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    try:
        return float(v.strip())
    except ValueError:
        return default


# 解码 / OCR / 分段
DECORD_FORCE_CPU_ENV: str = "DECORD_FORCE_CPU"                  # 1 强制 CPU 解码（旧钩子）
OCR_THREADS_ENV: str = "OCR_THREADS"                            # OCR 推理线程数覆盖（值型）
OCR_BATCH_ENV: str = "OCR_BATCH"                                # OCR 批大小覆盖（值型）
OCR_GAMMA_ENV: str = "OCR_GAMMA"                                # OCR 预处理 gamma（值型）
OCR_PAD_SMALL_ENV: str = "OCR_PAD_SMALL"                        # OCR 输入 pad 宽下限覆盖（值型）
# ── OCR 输入宽度自适应裁切（宽 ROI 字幕省计算）──
OCR_ROI_AUTOCROP_ENV: str = "OCR_ROI_AUTOCROP"                  # 0 关闭宽度自适应裁切
OCR_ROI_AUTOCROP_MARGIN_ENV: str = "OCR_ROI_AUTOCROP_MARGIN"    # 内容两侧保留余量（占 ROI 宽 %）
OCR_REORDER_WINDOW_ENV: str = "OCR_REORDER_WINDOW"              # OCR 重排窗口（段）；0=不重排
OCR_INSTANCES_ENV: str = "OCR_INSTANCES"                        # 0 关闭并行双 ONNX 实例
TEXT_SEP_MERGE_ENV: str = "TEXT_SEP_MERGE"                      # 相似段合并分离模式
DECODE_THREADS_ENV: str = "DECODE_THREADS"                      # CPU 软解 FFmpeg 帧线程数覆盖（值型；0=自动）
# 实验/诊断开关
GPU_PIPELINE_ENV: str = "GPU_PIPELINE"                          # 0 关闭 GPU 全驻留管线
GPU_PIPELINE_ASYNC_ENV: str = "GPU_PIPELINE_ASYNC"              # 1 开启 GPU 分段 kernel 异步实验路径
GPU_CTC_ENV: str = "GPU_CTC"                                    # 0 关闭 TRT 输出 GPU 归约
ENGINE_PROFILE_ENV: str = "ENGINE_PROFILE"                      # 1 开启引擎级性能剖面
TRT_SUBPROBE_ENV: str = "TRT_SUBPROBE"                          # 1 开启 TRT 子相位探针
DEBUG_BOUNDS_ENV: str = "DEBUG_BOUNDS"                          # 1 打印分段边界调试信息
HYBRID_PROBE_ENV: str = "HYBRID_PROBE"                          # 1 打印混合解码逐片时序
HYBRID_PROBE_CSV_ENV: str = "HYBRID_PROBE_CSV"                  # 逐片时序另落盘 CSV 路径
# CPU+NVDEC 混合解码（hybrid_decode.HybridDecoder v4，decode_backend="hybrid"）：
# 仅 GPU(NVDEC) 可用、stride==1、未开 GPU 全驻留管线时生效（编码不限，
# 含 AV1）；v4 = 动态分界（慢端不拖尾约束下给慢端尽量多片）+ 稳态速率
# 折扣（短校准高估 CPU 软解稳态速率）+ 缩短校准帧数（弱 CPU 下 256 帧
# 校准 ~0.4s 会吃掉混合收益）。
# h264 CPU 软解吞吐可达 NVDEC 两倍以上，闲置 CPU 的正确用途是帮解码；
# CPU 明显慢于 NVDEC（HEVC/AV1/弱 CPU）时 decode 仍可提升（8 核亲和
# 模拟实测 h264 decode -18%、HEVC decode -3%）。
HYBRID_CPU_THREADS_ENV: str = "HYBRID_CPU_THREADS"              # CPU reader 线程数(0=核数//2)
HYBRID_MAX_CHUNKS_ENV: str = "HYBRID_MAX_CHUNKS"                # 分片上限
# 分片粒度上限：>0 时 hybrid 分片超过该采样帧数继续拆小（内存上界 =
# inflight × 该上限，防宽 ROI 字幕整集单大片 2000+ 帧一次性缓存在
# ch['data']）；0=不拆（默认，兼容 v3）。仅 decode_backend="hybrid" 生效。
HYBRID_MAX_CHUNK_FRAMES_ENV: str = "HYBRID_MAX_CHUNK_FRAMES"
# v4 新增：慢端预取上限（默认 4 片，防 decode 早结束导致 OCR 尾批堆积）；
# 慢端速率折扣（慢端=CPU 默认 0.45 修正软解缓冲衰减、=NVDEC 默认 0.85）；
# 速率校准帧数（默认 40，弱 CPU 下压缩固定开销）。
HYBRID_SLOW_INFLIGHT_ENV: str = "HYBRID_SLOW_INFLIGHT"
HYBRID_SLOW_DISCOUNT_ENV: str = "HYBRID_SLOW_DISCOUNT"
HYBRID_CALIB_FRAMES_ENV: str = "HYBRID_CALIB_FRAMES"
HYBRID_CALIB_ROUNDS_ENV: str = "HYBRID_CALIB_ROUNDS"
# v4 默认值（解析收敛：调用点统一走 env_int / env_float，勿再各自解析）
HYBRID_SLOW_INFLIGHT_DEFAULT: int = 4      # 慢端预取上限（片）
HYBRID_SLOW_DISCOUNT_DEFAULT_CPU: float = 0.45  # 慢端=CPU 软解稳态折扣
HYBRID_SLOW_DISCOUNT_DEFAULT_GPU: float = 0.85  # 慢端=NVDEC 稳态折扣
HYBRID_CALIB_FRAMES_DEFAULT: int = 40      # 速率校准帧数（弱 CPU 下压缩固定开销）
HYBRID_CALIB_ROUNDS_DEFAULT: int = 1       # 校准轮数（>1 取中位数更稳，成本 ~0.3s/轮）
# ═══════════════════ CPU 软解线程预算（按 OCR 是否在 GPU 分档）═══════════════
# 背景：decord fork 在引擎不显式传 num_threads 时，CPU 解码线程数落到
# DECORD_FFMPEG_THREAD_COUNT = clamp(hw/4, 2, 8)（fork 源码
# src/video/video_reader.cc），即把 CPU 软解钉在 8 线程。该默认值是在
# "OCR 跑 ONNX 占满物理核"的时代定的；TensorRT 成为默认后 host CPU 在解码
# 阶段基本空闲，旧上限反而成为瓶颈。
# 实测（7945HX 16C32T + RTX 4060 Laptop，TRT；段数/唯一文本逐位一致）：
#   test5 1080p h264 全片 7223 帧  ：8 线程 6.452s → 16 线程 4.875s
#   新三国01 标清整集 73430 帧 s8  ：8 线程 15.897s → 32 线程 10.812s
#   相对现役默认（NVDEC+TRT 8.112s / 21.785s）= -45% / -50%。
#   绑核 8 逻辑核（模拟弱 CPU）不劣化：1080p -6%、标清 -35%。
# 上限 32：再多只增加 FFmpeg 帧缓冲（1080p 约 3MB/帧）与调度开销，无吞吐
# 收益；下限 8：保证少核机不低于 fork 原默认值。
# 仅当 OCR 后端非 cpu（TRT）时启用；ONNX 场景按采样步长另行分档（见下）。
DECODE_THREADS_GPU_OCR_MIN: int = 8
DECODE_THREADS_GPU_OCR_MAX: int = 32
# ── OCR 在 CPU（ONNX）时的解码线程分档（按 sample_stride 判段密度）──
# 实测（7945HX 16C32T，decode=cpu ocr=cpu，test5 3000 帧；段数/唯一文本一致）：
#   低段密度 stride=8（339 段，解码受限）  dcd=8  2.841s → 24  2.026s（-27.8%）
#                                        16/20/28/32 = 2.09~2.12s（平台）
#   高段密度 stride=1（1083 段，OCR 受限） dcd=8  3.746s → 10  3.617s（-3.4%）
#                                        12 3.714 / 14 3.824 / 16 3.811（>12 劣化）
# 判据是"解码与 OCR 谁占墙钟"：stride>1 时采样帧数 ÷ stride 而解码量不变
# → 解码必然更占优；stride==1 时段数可接近采样帧数 → OCR 更占优。
# 弱 CPU（绑 8 逻辑核）复核：stride8 最优 8（4/12/16 差 ≤4.6%）、
# stride1 最优 12（8 差 2.8%）—— 取下面公式后误差 ≤5%，两端都不劣化。
DECODE_THREADS_CPU_OCR_MAX: int = 24          # stride>1：逻辑核 3/4，上限 24
DECODE_THREADS_CPU_OCR_STRIDE1_MAX: int = 12  # stride==1：逻辑核 1/3，上限 12
# 并行双 ONNX 实例的启动门限（OCR 线程数 ≥ 此值才默认拆两个实例）
OCR_INSTANCES_MIN_THREADS: int = 8

# ═══════════════════ 数据目录 ═══════════════════


def app_data_dir() -> Path:
    """程序数据目录（本文件夹内，免安装/portable 设计）。

    引擎缓存（ocr_engines/）与运行日志（logs/）都放这里 —— 不写
    %LOCALAPPDATA%，卸载/移动时删除整个程序目录即清理干净。

    frozen: exe 所在目录；源码运行: 项目根目录。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def app_logs_dir() -> Path:
    """运行日志目录（本目录/logs，与数据目录一致）。"""
    return app_data_dir() / "logs"

# ═══════════════════ 用户可配置默认值 ═══════════════════
DEFAULT_OCR_MODEL: str = "v6_small"     # 唯一 OCR 模型（v2.13 起移除 tiny / 重 OCR）
DEFAULT_REP_CROP_FORMAT: str = "yuv"    # 代表帧 keep_crops 默认格式：
                                        # "yuv"=packed NV12（内部只取 Y 平面，
                                        # 外部用 nv12_to_rgb 转 RGB——内部恒为
                                        # 单通道灰度链路，不产 RGB 帧）；"gray"=灰度
DEFAULT_BUFFER_SIZE: int = 128          # 解码∥OCR 流水线队列缓冲（段数）
                                        # 64→128：GPU 解码突发时缓冲背压，减少
                                        # 解码线程 q.put 阻塞等待（GPU+CPU wall
                                        # -0.3s；256 无进一步收益）
DEFAULT_DECODE_BACKEND: str = "auto"    # 解码后端 (auto / cpu / nvdec / hybrid)
DECODE_BACKEND_KEYS: list[str] = ["auto", "cpu", "nvdec", "hybrid"]
DECODE_BACKEND_LABELS: dict[str, str] = {"auto": "自动", "cpu": "CPU",
                                         "nvdec": "NVDEC",
                                         "hybrid": "混合(CPU+NVDEC)"}
# GPU 全驻留管线（_gpu_pipeline）解码批大小：64 为 GPU 分段实验最优（更大批
# 减少 kernel/同步次数），与宿主 DECODE_BATCH_SIZE=16 刻意不同——两条路径
# 独立调参，勿统一为一个常量。
GPU_PIPELINE_DECODE_BATCH: int = 64
DEFAULT_OCR_BACKEND: str = "auto"       # OCR 推理后端 (auto / cpu / tensorrt)
OCR_BACKEND_KEYS: list[str] = ["auto", "cpu", "tensorrt"]
OCR_BACKEND_LABELS: dict[str, str] = {"auto": "自动", "cpu": "CPU", "tensorrt": "TensorRT"}
DEFAULT_FORCE_ASPECT: float = 0.0       # 强制横向宽高比（0=不启用；>0 时宽度
                                        # 强制 = 48×此值，纠正扁宽字体）
# **2026-08-29 用真值重测后下调：224 → 160**（与 OCR_PAD_WIDTH_MIN 同步，
# 详见下方 §"OCR 输入 pad 宽度下限"的实测表）。extractor 默认把它传给
# OcrEngine(fill_width=...)，因此这**才是实际生效的默认值**。
DEFAULT_FILL_WIDTH: int = 160           # OCR 输入 pad 宽度下限（引擎 _resize_norm
                                        # pad 到该总宽）。旧值 224 的依据已作废：
                                        # 320 raw 最优 0.53% vs 224 0.67%（test5
                                        # 7→2、test6 17→5），但端到端 224 最优
                                        # （13 vs 16）—— 那是旧预处理/旧模型下
                                        # 的结论，重测后 160 在准确率与墙钟上
                                        # 双赢。GUI 可调 96-320，默认 160。
DEFAULT_SAMPLE_STRIDE: int = 1          # 分频采样步长（默认 1 = 逐帧）。>1 时只
                                        # 解码/分段/OCR 每个第 N 帧（字幕等慢更新
                                        # 内容显著降低处理压力，时间戳仍取真实帧号）。
                                        # 需 decord fork ≥0.7.12 的等差步长快速路径，
                                        # 否则退化为逐索引 seek
OCR_GAMMA: float = 2.0                  # OCR 预处理灰度 gamma 增强指数（正式预处理：
                                        # 白字黄底等背景色块场景放大高段分离；灰度
                                        # 先于 gamma——RGB 逐通道 gamma 视觉差异小、
                                        # 回归多。1.0=纯灰度不增强，0=保留 RGB）

# ═══════════════════ 段管线参数 ═══════════════════
SEG_C: float = 5.0              # 分段聚类阈值：max 3×3 窗口和 < C ⇒ 显示未变
# 相似段合并（生产默认开启）：连续两段代表帧在字幕/背景分离图上比较，
# 平均绝对差 ≤ 阈值时视为同一视觉内容（如噪声把同一条字幕切成多段），
# 合并后只 OCR 一次。
DEFAULT_MERGE_SIMILAR: bool = True
# 相似帧合并使用的分离方案：binary（黑底白字）为引擎默认。
# OCR 输入仍保持灰度+gamma，不直接使用 binary（实测会降低 OCR 准确率）。
DEFAULT_MERGE_TEXT_SEP: str = "binary"
SEG_MERGE_SIMILAR_THRESHOLD: float = 3.0
# 相似段合并的“显著变化像素”上限（ROI 面积比例）：即使平均绝对差很小，若
# 变化像素占比超过该比例，仍视为真实内容变化（防止宽 ROI 中单字短字幕被误合并）。
SEG_MERGE_MAX_CHANGED_RATIO: float = 0.01

# ═══════════════════ OCR 输入 pad 宽度下限 ═══════════════════
# **2026-08-29 用真值重测：224 是过时残留，已下调到 160。**
#
# 旧注释（已作废，保留以说明为何当初这么定）：
#   窄图（48 高后 78-160 宽）在宽 pad 下 v6_small 更准
#   （test6：224→err 0.09%，192→0.16%，48~96→0.69~1.19%；256 精度相同但更慢）
#
# 实测（现役代码 + ground_truth_csv 按帧对齐，decode=cpu ocr=auto，全等准确率）：
#
# | 视频 | 编码 | 48 | 96 | 160 | 192 | **224(旧)** | 320 |
# |---|---|---:|---:|---:|---:|---:|---:|
# | test5 | h264 | 98.962 | 98.962 | **99.031** | 98.754 | 97.951 | 98.103 |
# | test6 | av1  | **98.916** | 98.916 | 98.878 | 98.379 | 98.187 | 97.816 |
# | test  | hevc | 94.987 | 94.847 | 94.847 | 94.875 | 94.903 | 94.763 |
# | test2 | hevc | 95.265 | 95.322 | 95.550 | 95.665 | **95.722** | 95.722 |
#
# → 224 在 test5/test6 上反而最差（−0.7~−1.1pp）；四片均值最优是 160（+0.39pp）。
#   只有 test2 略偏好高位（差 ≤0.46pp，噪声量级）。
# → 墙钟方向一致：低档位全面更快（−0.5%~−4.3%），320 变慢（+2.3%~+4.1%）。
# → 宽 ROI（新三国01 30000帧）上 160 vs 224 **文本完全一致**且快 8.8%。
#
# 结论：**宽 pad 不再带来精度收益，只是白算**。取 160（均值最优且更快）。
OCR_PAD_WIDTH_MIN: int = 160
OCR_PAD_WIDTH_MIN_BY_MODEL: dict[str, int] = {
    "v6_small": 160,
}

# ═══════════════════ 运行参数（v2.15.1 起从代码中收敛） ═══════════════════
# OCR 模型固定输入高度（rapidocr resize_norm_img 语义，训练尺寸）
OCR_TARGET_H: int = 48
# ═══════════════════ OCR 输入宽度自适应裁切 ═══════════════════
# 宽 ROI 字幕（如整集 407×25）里，绝大多数字幕不占满宽度，空白列照样参与
# 卷积。用分段已算好的二值图求"有墨迹的列范围"，裁掉两侧空白再喂 OCR。
#
# 实测（新三国01 30000帧 stride8，503 段，TRT，离线对照）：
#   内容宽/ROI宽：min 0.05  p10 0.23  中位 0.69  p90 1.00
#   OCR 耗时    顺序分批 -1.7%（每批仍被满宽成员顶上去，几乎没用）
#               **按宽度排序分批 -23.9%（余量 10%）**
#   文本一致率  余量 0% → 98.0%；5% → 99.8%；**10% → 100.0%**；20% → 100%
# （余量 0 的差异几乎全是"插入多余空格"，如 好酒好酒好酒 → 好酒 好酒 好酒；
#   加 10% 余量后逐位一致。均值置信度 0.52715 → 0.52683，实质不变。）
#
# 两个前提，缺一不可：
#   1. **必须跨批按宽度分组**。OcrEngine.__call__ 的 pad 宽 = 批内最大宽，
#      它虽已在批内排序，但那只优化 host resize 顺序、不改 pad 宽。
#   2. **余量不能省**。裁太紧会改变 CTC 序列长度进而插空格。
#
# 自动失效的场景（无收益则不动）：
#   · 内容宽 ≥ ROI 宽（字幕满宽）→ 不裁
#   · 裁后宽度仍 ≤ OCR_PAD_WIDTH_MIN（224）→ pad 回去，无收益（窄 ROI 常态）
#   · force_aspect > 0 → 宽度被强制，裁切只改变缩放不省宽 → 跳过
OCR_ROI_AUTOCROP_DEFAULT: bool = True
OCR_ROI_AUTOCROP_MARGIN_PCT: int = 10        # 余量 = ROI 宽的百分比（每侧）
OCR_REORDER_WINDOW_DEFAULT: int = 64         # 重排窗口（段）；= 4 × 默认 OCR 批 16
# 高度已接近目标时跳过 resize 的相对容差（2%）
OCR_RESIZE_TOL: float = 0.02
# 灰度权重（Rec.601；分段与 OCR 预处理共用，逐位一致性依赖此权重）
GRAY_RGB_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)
# 段管线批大小：OCR 批（段数）与解码批（帧数）
OCR_BATCH_SIZE: int = 16
DECODE_BATCH_SIZE: int = 16
# 流水线队列：OCR 预处理→推理队列上限
OCR_INFER_QUEUE_SIZE: int = 4
# 分段 Otsu 阈值校准帧数（前 N 帧；seek 校准代价高，前段与全片抽样一致）
SEG_CALIB_FRAMES: int = 50
# Otsu 无法计算时的兜底阈值（0-255）
OTSU_FALLBACK_THRESH: int = 127
# 解码器无法给出 fps 时的兜底帧率
DEFAULT_FPS_FALLBACK: float = 30.0
# OCR 引擎内部：ONNX 单批上限与 CTC 归约分块（内存峰值控制）
OCR_ONNX_CHUNK: int = 16
OCR_CTC_CHUNK: int = 64
# TensorRT 引擎构建：默认 batch profile、输入宽 profile 与 workspace
TRT_PROFILE_BATCH: int = 6
TRT_PROFILE_MIN_W: int = 32
TRT_PROFILE_OPT_W: int = 320
TRT_PROFILE_MAX_W: int = 2048
TRT_WORKSPACE_BYTES: int = 1 << 30
# TRT 引擎缓存文件的 SM 后缀（引擎与 GPU 架构绑定）
TRT_ENGINE_SM: str = "sm89"
# ═══════════════════ 少核 CPU 解码线程分核预算（v2.15.2 实验） ═══════════════════
# CPU 软解 + 物理核 ≤ CPU_CORES_SPLIT_THRESHOLD 时，OCR 线程与 decord
# FFmpeg 线程各分 cores//2（显式分核），避免 FFmpeg 2 帧线程（fork 默认，
# 只用 2 核）+ OCR 全核的过订阅。实测（test5，affinity 模拟，venv）：
#   4 核 CPU+ONNX：ocrT=2/dcd=2 → 28.0s vs 现状 33.1s（-15%）
#   8 核 CPU+ONNX：ocrT=4/dcd=4 → 17.8s vs 现状 20.7s（-14%）
#   16 核：分核反而更差（12.0 vs 9.5s）→ 保持现状（OCR=全核，FFmpeg
#   默认 2 帧线程落在 SMT 份额上）；GPU(NVDEC) 解码不抢 CPU → 保持现状。
# 4 核 CPU+TRT 时解码可分更多核（dcd=4 → 13.2s），但 TRT 组合少核机器
# 罕见且收益在测量波动内，统一用 cores//2 分核。
CPU_CORES_SPLIT_THRESHOLD: int = 8