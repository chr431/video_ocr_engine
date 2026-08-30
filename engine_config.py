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

# 版本单一事实源：wheel 版本号（`pyproject.toml` 用 `dynamic` + `attr` 从此处
# 读取）与运行时 `video_ocr_engine.__version__` 同源，且必须与 git tag 一致
# —— 否则"装的到底是哪个版本"无法判断。改动本值后记得同步打 tag。
__version__ = "0.10.1"

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
OCR_THREADS_ENV: str = "OCR_THREADS"                            # OCR 推理线程数覆盖（值型）
OCR_BATCH_ENV: str = "OCR_BATCH"                                # OCR 批大小覆盖（值型）
OCR_GAMMA_ENV: str = "OCR_GAMMA"                                # OCR 预处理 gamma（值型）
OCR_PAD_SMALL_ENV: str = "OCR_PAD_SMALL"                        # OCR 输入 pad 宽下限覆盖（值型）
# ── OCR 输入宽度自适应裁切（宽 ROI 字幕省计算）──
OCR_ROI_AUTOCROP_ENV: str = "OCR_ROI_AUTOCROP"                  # 0 关闭宽度自适应裁切
OCR_ROI_AUTOCROP_MARGIN_ENV: str = "OCR_ROI_AUTOCROP_MARGIN"    # 内容两侧保留余量（占 ROI 宽 %）
OCR_ROI_AUTOCROP_MIN_GAIN_ENV: str = "OCR_ROI_AUTOCROP_MIN_GAIN"  # 裁掉比例低于此值则不裁（%）
OCR_REORDER_WINDOW_ENV: str = "OCR_REORDER_WINDOW"              # OCR 重排窗口（段）；0=不重排
OCR_INSTANCES_ENV: str = "OCR_INSTANCES"                        # 0 关闭并行双 ONNX 实例
TEXT_SEP_MERGE_ENV: str = "TEXT_SEP_MERGE"                      # 相似段合并分离模式
DECODE_THREADS_ENV: str = "DECODE_THREADS"                      # CPU 软解 FFmpeg 帧线程数覆盖（值型；0=自动）
# 实验/诊断开关
GPU_PIPELINE_ENV: str = "GPU_PIPELINE"                          # 0 关闭 GPU 全驻留管线；1 强制（含 GPU 分段+ONNX 实验组合）
GPU_CTC_ENV: str = "GPU_CTC"                                    # 0 关闭 TRT 输出 GPU 归约
ENGINE_PROFILE_ENV: str = "ENGINE_PROFILE"                      # 1 开启引擎级性能剖面
TRT_SUBPROBE_ENV: str = "TRT_SUBPROBE"                          # 1 开启 TRT 子相位探针
DEBUG_BOUNDS_ENV: str = "DEBUG_BOUNDS"                          # 1 打印分段边界调试信息
HYBRID_PROBE_ENV: str = "HYBRID_PROBE"                          # 1 打印混合解码逐片时序
HYBRID_PROBE_CSV_ENV: str = "HYBRID_PROBE_CSV"                  # 逐片时序另落盘 CSV 路径
# CPU+NVDEC 混合解码（hybrid_decode.HybridDecoder v4，decode_backend="hybrid"）：
# 显式选择且 NVDEC 可用时生效（编码不限，含 AV1）；stride>1 已支持
# （分片/扫掠/校准均按采样步长推进）；GPU 全驻留管线开启时由其 CPU
# 分支消费（§8.3 合并，原互斥门控已移除）。v4 = 动态分界（慢端不拖尾
# 约束下给慢端尽量多片）+ 稳态速率折扣（短校准高估 CPU 软解稳态速率）
# + 缩短校准帧数（弱 CPU 下 256 帧校准 ~0.4s 会吃掉混合收益）。
# h264 CPU 软解吞吐可达 NVDEC 两倍以上，闲置 CPU 的正确用途是帮解码；
# CPU 明显慢于 NVDEC（HEVC/AV1/弱 CPU）时 decode 仍可提升（8 核亲和
# 模拟实测 h264 decode -18%、HEVC decode -3%）。
HYBRID_CPU_THREADS_ENV: str = "HYBRID_CPU_THREADS"              # CPU reader 线程数；
                                                                # 0 = 按核数自动分档（见下方 AUTO_MIN/MAX）
# 2026-08-30：0 的历史语义是"不传 num_threads"→ 落到 fork 的
# DECORD_FFMPEG_THREAD_COUNT=clamp(hw/4,2,8)，把 CPU 生产者钉在 8 线程。
# 实测（docs/PERFORMANCE.md §17.2，test5 3000 帧 stride1，3 轮最快）：
#   cpuT 0→16 = 2.166→2.084s，0→24 = 2.166→2.051s（decode 1.288→1.114s，
#   -13.5%），0→32 = 2.077s（略差于 24：CPU 生产者与 NVDEC/消费者抢 host CPU）。
# 故自动值取 逻辑核×3/4 钳 [8, 24]（与 _decode_num_threads 的 stride>1
# 分档同式）。段数与唯一文本在所有档位下完全一致。
HYBRID_CPU_THREADS_AUTO_MIN: int = 8
HYBRID_CPU_THREADS_AUTO_MAX: int = 24
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
# v4 默认值（解析收敛：调用点统一走 env_int / env_float，勿再各自解析）
HYBRID_SLOW_INFLIGHT_DEFAULT: int = 4      # 慢端预取上限（片）
HYBRID_SLOW_DISCOUNT_DEFAULT_CPU: float = 0.45  # 慢端=CPU 软解稳态折扣
HYBRID_SLOW_DISCOUNT_DEFAULT_GPU: float = 0.85  # 慢端=NVDEC 稳态折扣
HYBRID_CALIB_FRAMES_DEFAULT: int = 40      # 速率校准帧数（弱 CPU 下压缩固定开销）
# 注：HYBRID_CALIB_ROUNDS（多轮校准取中位）已于 0.9.0 删除——实测净负
# （3 轮 -21%：~0.68s 测速成本 > 分界精度收益），见 docs/PERFORMANCE.md §10.5。
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
# **2026-08-29 回退：160 → 224**（与 OCR_PAD_WIDTH_MIN 同步，详见下方
# §"OCR 输入 pad 宽度下限"的对照表）。extractor 默认把它传给
# OcrEngine(fill_width=...)，因此这**才是实际生效的默认值**。
# 上一轮下调到 160 是错的：只测了 force_aspect=0，而生产用 force_aspect>0，
# 那里 160 使原始误读 7→26。旧值 224 的依据（320 raw 最优 0.53% vs 224
# 0.67%，但端到端 224 最优 13 vs 16）**依然成立，不该推翻**。
DEFAULT_FILL_WIDTH: int = 224           # OCR 输入 pad 宽度下限（引擎 _resize_norm
                                        # pad 到该总宽）。GUI 可调 160-320，
                                        # 默认 224。取值依赖 force_aspect：
                                        # force_aspect>0（内容被压窄）时越大越准，
                                        # =0（内容按原宽高比）时偏小更佳。
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
# **2026-08-29 回退：160 → 224。上一轮的"下调到 160"是错的，勿再改。**
#
# 出错原因：我全程用 `force_aspect=0.0`（引擎默认）评估，而生产（RaceVideoToLog
# 2.17.x）传 `force_aspect=mw`（真值头里 test5 = 1.5）。**两者下 pad 下限的
# 作用方向完全相反**：
#
# | pad | force_aspect=0（内容 154px）| force_aspect=1.5（内容被压到 72px）|
# |---|---:|---:|
# | 160 | **6** | 26 |
# | 192 | 12 | 17 |
# | 224 | 30 | **7**  ← 基线 |
# | 256 | 31 | 6 |
# | 320 | 29 | **2** |
# （test5 代表帧 tol=1 误读数，越低越好；生产门禁 baseline = 7）
#
# 机制：pad 下限的优劣取决于**内容宽度**，不能一刀切。
#   · force_aspect=0：内容按原宽高比缩放，较宽（154px）→ **偏小 pad 更好**
#   · force_aspect>0：内容被压到固定窄宽（72px）→ **pad 越大留白越多、越准**
# 这与旧注释的经验（"窄图在宽 pad 下更准"）一致——**旧注释是对的，
# 是我的评估口径错了**，不该推翻它。
#
# 生产侧已验证（tools/accuracy_breakdown.py，5 视频全量帧，单变量
# OCR_PAD_SMALL）：160 使总原始误读 150→190（test5 7→26、test6 17→32），
# 224 恢复基线 ≈149。故保持 224。
#
# 后续若要再动这个值，**必须用生产口径**（代表帧 + tol=1 + force_aspect=真值头）
# 复测，不能只用引擎侧的 force_aspect=0 默认。
OCR_PAD_WIDTH_MIN: int = 224
OCR_PAD_WIDTH_MIN_BY_MODEL: dict[str, int] = {
    "v6_small": 224,
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
# 裁切顺序按 force_aspect 分流（两条路径**都会裁**，只是顺序不同）：
#   · fa > 0 → 先定比例、后裁（顺序 ⑦，`_crop_after_aspect`）
#   · fa = 0 → 先裁、后定比例（`_crop_to_content`）
# 顺序 ⑦ 是硬要求：fa>0 时若先裁，裁后区间会被拉伸到 force 宽度而改变
# 内容宽高比 → 畸变（实测 test5：⑦=0 vs ⑥=9 误读）。
#
# 不裁的两种情况：内容已占满 ROI（无空白可裁）、或**收益低于门槛**（见下）。

# ⚠️ force_aspect **不是**裁切收益的判据，别拿它当开关。
# fa 只是恰好与"ROI 宽裕程度"相关（本批 5 个视频里 fa=1.5 的两个恰好
# ROI 宽高比 3.3、左留白 24%，fa=0 的三个是 1.65~1.83、左留白 6~11%）。
# 真正的判据是 **ROI 相对内容的宽裕程度**：
#
# | 视频 | ROI宽高比 | 内容占比 | 左留白 | 右留白 | 裁切效果 |
# |---|---:|---:|---:|---:|---|
# | test5 | 3.28 | 0.708 | 23.6% | 5.7% | 大幅改善 7→0 |
# | test6 | 3.38 | 0.697 | 23.9% | 6.4% | 大幅改善 17→0 |
# | test  | 1.65 | 0.835 | 10.6% | 5.9% | 中性（余量足够时）|
# | test2 | 1.83 | 0.897 |  6.4% | 2.6% | 中性 |
# | test3 | 1.64 | 0.896 |  7.3% | 2.1% | 中性 |
#
# test5/test6 左留白是右留白的约 4 倍 —— 数字右对齐、ROI 按最长状态取，
# 短数字时空白全堆在左侧。留白越多，裁掉的收益越大、切到笔画的风险越小。
# 测量工具：`tools/_probe_roi_whitespace.py`。
OCR_ROI_AUTOCROP_DEFAULT: bool = True
# 余量 = ROI 宽的百分比（每侧）。**2026-08-30 由 10 改为 20**，依据见下。
#
# 余量是「裁掉空白」与「切到笔画」之间的权衡，两侧都错得起：
#   · 太小 → 切掉字符边缘笔画（紧凑 ROI 上明显）
#   · 太大 → 把留白又放回来，收益被吃掉（宽 ROI 上 30% 反而变差）
#
# 生产路径实测（tools/_probe_truth_env.py --dbe auto，段代表帧 + 数值
# tol=1 误读数，越低越好）：
#
# | 视频 | ROI宽高比 | 左留白 | 不裁 | 余量10 | 余量20 | 余量30 |
# |---|---:|---:|---:|---:|---:|---:|
# | test5 | 3.28 | 23.6% | 7  | 0  | 0  | 10 |
# | test6 | 3.38 | 23.9% | 17 | 0  | 0  | 10 |
# | test  | 1.65 | 10.6% | 78 | 80 | 78 | 78 |
# | test2 | 1.83 |  6.4% | 51 | 51 | 51 | 51 |
#
# 余量 10 时 test 退化 2（左留白 10.6% ≈ 余量，边界上切到了笔画）；
# 20 时四个视频全部不差于不裁；30% 在宽 ROI 上把留白放回来，反而更差。
#
# ⚠️ 但**靠加大余量规避误裁是错的**：余量是全局的，会把宽 ROI 的收益一起
# 削掉 —— test5 的裁掉量中位数 余量10 时 13.2% → 余量20 时只剩 3.8%
# （收益少了 71%），而 test5 在余量 10 下**零误裁**。
# 真正的解法是下面的 **最小收益门槛**（逐段自适应），余量因此**回到 10**。
OCR_ROI_AUTOCROP_MARGIN_PCT: int = 10
#
# 最小收益门槛：裁掉宽度占 ROI 宽的比例低于此值就**整段不裁**。
#
# 依据（`tools/_probe_crop_miscut.py`，前 3000 帧，误裁 = 被裁区间在宽松
# 判据「列墨迹 ≥1」下仍有内容，即引擎判据漏掉了真实笔画）：
#
# | 视频 | 余量 | 裁切率 | 裁掉量中位 | 误裁左 | 裁掉<10%的段 |
# |---|---:|---:|---:|---:|---:|
# | test  | 10% | 72% | **1.2%** | **61** | 855/1337 |
# | test  | 20% |  8% | 0.0%     | 12 | 22 |
# | test2 | 10% |  8% | 0.0%     |  3 | 13 |
# | test5 | 10% | 78% | **13.2%** | **0** | 0 |
# | test6 | 10% | 81% | **13.8%** | **0** | 0 |
#
# test 上 72% 的段被裁但**裁掉量中位数只有 1.2%**（64% 的段裁掉不到 5%）
# —— 几乎没收益却承担切笔画的风险，误裁全部集中在这些"微裁"段。
# 而 test5/test6 裁掉 13% 且零误裁。
#
# 设门槛后：紧凑 ROI（test）自动几乎不裁，宽 ROI（test5/test6）照裁，
# **收益与风险在逐段粒度上自动分开**，不需要全局折中。
# 门槛取 10% 是因为 test5/test6 的裁掉量分布里 [5%,10%) 区间为空（0 段），
# 即 10% 不会误伤宽 ROI 的任何一段。
OCR_ROI_AUTOCROP_MIN_GAIN_PCT: int = 10
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