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

__version__ = "0.3.0"

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


# 解码 / OCR / 分段
DECORD_FORCE_CPU_ENV: str = "DECORD_FORCE_CPU"                  # 1 强制 CPU 解码（旧钩子）
OCR_THREADS_ENV: str = "OCR_THREADS"                            # OCR 推理线程数覆盖（值型）
OCR_BATCH_ENV: str = "OCR_BATCH"                                # OCR 批大小覆盖（值型）
OCR_GAMMA_ENV: str = "OCR_GAMMA"                                # OCR 预处理 gamma（值型）
OCR_PAD_SMALL_ENV: str = "OCR_PAD_SMALL"                        # OCR 输入 pad 宽下限覆盖（值型）
OCR_INSTANCES_ENV: str = "OCR_INSTANCES"                        # 0 关闭并行双 ONNX 实例
TEXT_SEP_MERGE_ENV: str = "TEXT_SEP_MERGE"                      # 相似段合并分离模式
# 实验/诊断开关
GPU_PIPELINE_ENV: str = "GPU_PIPELINE"                          # 0 关闭 GPU 全驻留管线
GPU_CTC_ENV: str = "GPU_CTC"                                    # 0 关闭 TRT 输出 GPU 归约
ENGINE_PROFILE_ENV: str = "ENGINE_PROFILE"                      # 1 开启引擎级性能剖面
TRT_SUBPROBE_ENV: str = "TRT_SUBPROBE"                          # 1 开启 TRT 子相位探针
DEBUG_BOUNDS_ENV: str = "DEBUG_BOUNDS"                          # 1 打印分段边界调试信息
# CPU+NVDEC 混合解码（hybrid_decode.HybridDecoder，decode_backend="hybrid"）：
# 仅 GPU(NVDEC) 可用、非 AV1、stride==1、未开 GPU 全驻留管线时生效；
# cpu_threads=0 表示 cores//2。h264 CPU 软解吞吐可达 NVDEC 两倍以上，
# 闲置 CPU 的正确用途是帮解码（全负载场景多为 NVDEC 解码受限）。
HYBRID_CPU_THREADS_ENV: str = "HYBRID_CPU_THREADS"              # CPU reader 线程数(0=核数//2)
HYBRID_MAX_CHUNKS_ENV: str = "HYBRID_MAX_CHUNKS"                # 竞争分片上限
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
DEFAULT_FILL_WIDTH: int = 224           # OCR 输入 pad 宽度下限（引擎 _resize_norm
                                        # pad 到该总宽）。扫描（test2/5/6 全量）：
                                        # 320 raw 最优 0.53% vs 224 0.67%（test5 7→2、
                                        # test6 17→5），但端到端 224 最优（13 vs 16）。
                                        # GUI 可调 160-320，默认 224
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
# 窄图（48 高后 78-160 宽）在宽 pad 下 v6_small 更准
# （test6：224→err 0.09%，192→0.16%，48~96→0.69~1.19%；256 精度相同但更慢）。
OCR_PAD_WIDTH_MIN: int = 224
OCR_PAD_WIDTH_MIN_BY_MODEL: dict[str, int] = {
    "v6_small": 224,
}

# ═══════════════════ 运行参数（v2.15.1 起从代码中收敛） ═══════════════════
# OCR 模型固定输入高度（rapidocr resize_norm_img 语义，训练尺寸）
OCR_TARGET_H: int = 48
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