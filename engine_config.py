"""管线引擎配置（engine_config）— 解码/OCR/分段/纠错域常量。

独立引擎仓库的配置单一事实源，供 engine 内 ocr_native / ocr_trt /
segmentation / video_utils / hybrid_decode 与引擎外的上层应用直接引用。
RaceVideoToLog 的 config.py 通过 `from engine_config import *` 聚合再导出
（GUI 与 `import config` 兼容），GUI 专属常量（颜色/窗口/图表）留在应用侧。

自拆仓（v0.1.0）起：本文件随 video_ocr_engine 独立仓库发布、独立版本线。
"""
from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

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

# ═══════════════════ 物理常量 ═══════════════════
MPS_TO_KMH: float = 3.6          # m/s → km/h 转换因子

# ═══════════════════ 用户可配置默认值 ═══════════════════
DEFAULT_OCR_MODEL: str = "v6_small"     # 唯一 OCR 模型（v2.13 起移除 tiny / 重 OCR）
DEFAULT_SPEED_FORMAT: str = "km/h"     # 速度单位 (km/h / m/s / mile/h)
DEFAULT_BUFFER_SIZE: int = 128          # 解码∥OCR 流水线队列缓冲（段数）
                                        # 64→128：GPU 解码突发时缓冲背压，减少
                                        # 解码线程 q.put 阻塞等待（GPU+CPU wall
                                        # -0.3s；256 无进一步收益）
DEFAULT_DECODE_BACKEND: str = "auto"   # 解码后端 (auto / cpu / nvdec)
DECODE_BACKEND_KEYS: list[str] = ["auto", "cpu", "nvdec"]
DECODE_BACKEND_LABELS: dict[str, str] = {"auto": "自动", "cpu": "CPU",
                                         "nvdec": "NVDEC"}
# 实验性 CPU+NVDEC 混合解码开关（不暴露给 GUI/CLI 参数）：
# 环境变量置 1/true/yes/on 后，GPU 模式（auto / nvdec）内部改走
# CPU+NVDEC 双解码器并行（CPU 前段 + GPU 后段，见 _open_hybrid_vrs）；
# 默认关闭（纯 GPU 足够好，混合收益不确定且增加复杂度）。
HYBRID_DECODE_ENV: str = "RVTOL_HYBRID_DECODE"
# 实验性 OCR 混合开关（不暴露给 GUI/CLI 参数）：置 1/true/yes/on 后
# OCR 同时用 TensorRT（GPU）+ onnxruntime（CPU）双引擎并发处理段批
# （OCR 无状态约束，结果按段索引聚合，顺序无关 → 实现简单）。
# 默认关闭（TRT 可用时 OCR 已隐藏于解码阶段之下，非瓶颈）。
HYBRID_OCR_ENV: str = "RVTOL_HYBRID_OCR"
DEFAULT_OCR_BACKEND: str = "auto"      # OCR 推理后端 (auto / cpu / tensorrt)
OCR_BACKEND_KEYS: list[str] = ["auto", "cpu", "tensorrt"]
OCR_BACKEND_LABELS: dict[str, str] = {"auto": "自动", "cpu": "CPU", "tensorrt": "TensorRT"}
DEFAULT_MAX_SPEED: float = 400.0       # 最大速度 (km/h)
DEFAULT_MAX_ACCEL: float = 50.0        # 最大加速度 (m/s²)
DEFAULT_FORCE_ASPECT: float = 0.0      # 强制横向宽高比（0=不启用；>0 时宽度
                                       # 强制 = 48×此值，纠正扁宽字体）
DEFAULT_FILL_WIDTH: int = 224          # OCR 输入 pad 宽度下限（引擎 _resize_norm
                                       # pad 到该总宽）。扫描（test2/5/6 全量）：
                                       # 320 raw 最优 0.53% vs 224 0.67%（test5 7→2、
                                       # test6 17→5），但端到端 224 最优（13 vs 16）
                                       # ——test5/6 的 raw 提升被 DP 吸收，test2 宽
                                       # pad 引入混杂邻域 DP 拉中间值（纠错 5）。
                                       # GUI 可调 160-320，默认 224
DEFAULT_SAMPLE_STRIDE: int = 1         # 分频采样步长（默认 1 = 逐帧，与
                                       # RaceVideoToLog 完全兼容）。>1 时只
                                       # 解码/分段/OCR 每个第 N 帧（字幕等慢
                                       # 更新内容显著降低处理压力，时间戳仍取
                                       # 真实帧号）。需 decord fork ≥0.7.12 的
                                       # 等差步长快速路径，否则退化为逐索引 seek
OCR_GAMMA: float = 2.0                 # OCR 预处理灰度 gamma 增强指数（正式预处理：
                                       # 白字黄底等背景色块场景放大高段分离；灰度
                                       # 先于 gamma——RGB 逐通道 gamma 视觉差异小、
                                       # 回归多。1.0=纯灰度不增强，0=保留 RGB）

# ═══════════════════ 段管线参数 ═══════════════════
HYBRID_CPU_SPLIT: float = 0.10    # 实验性混合解码（HYBRID_DECODE_ENV=1 时生效）
                                  # 的 CPU 段帧数比例（保守分法）。
                                  # 只有 CPU/GPU 吞吐相近（h264：CPU 软解
                                  # ~1260fps ≈ NVDEC 2Gp/s 上限 ~960fps）时
                                  # 对半分（55/45）才有 decode 砍半优势；
                                  # HEVC/AV1 的 CPU 软解只有 NVDEC 的 1/3~1/5，
                                  # 大份额 CPU 段反成瓶颈（test6 AV1 混合
                                  # 43.6s vs GPU 14.4s）。10% 保守分法下
                                  # wall = max(CPU 10% 耗时, GPU 90% 耗时)，
                                  # h264/HEVC ≤ 纯 GPU（实测 test HEVC 2.7 vs
                                  # 2.9s / test3 h264 3.1 vs 3.4s / test5 h264
                                  # 7.1 vs 7.6s decode）；AV1 特判：CPU 软解
                                  # AV1 极耗核且并发竞争拖慢 GPU 段（混合 19.1s
                                  # vs 纯 GPU 14.4s）→ _hybrid_split 返回 0，
                                  # 等效纯 GPU。env RVTOL_HYBRID_SPLIT 可覆盖
                                  # （实验）。
SEG_GAMMA: float = 0.0             # 分段/代表帧选择的灰度 gamma 增强指数。
                                   # 0 = raw 灰度（锁定基线，v2.14 现状：分段与
                                   # OCR 正式预处理 gray+gamma2.0 不一致但已接受）。
                                   # >0 = 255*(g/255)^g 增强后分段（与 OCR 预处理
                                   # 对齐实验，env RVTOL_SEG_GAMMA 可覆盖）。
SEG_C: float = 5.0              # 分段聚类阈值：max 3×3 窗口和 < C ⇒ 显示未变
# 相似段合并（subtitle 场景可选）：连续两段代表帧灰度平均绝对差 ≤ 该阈值时，
# 视为同一视觉内容（如噪声把同一条字幕切成多段），合并后只 OCR 一次。
# 默认关闭（速度数字场景不能合并）；video_subtitle_extractor 可显式开启。
SEG_MERGE_SIMILAR_THRESHOLD: float = 3.0
SEG_WIN: int = 30               # 段级检测带宽窗口（换算成帧：×中位段间距，上限 120 帧）
SEG_MULT: float = 2.0           # 检测门限倍率：|值-中值| > 带宽×mult ⇒ suspect
SEG_MIN_DEV: float = 6.0        # 纠正最小偏差：|插值-当前| > 此值才改
SEG_MED_K: int = 10             # 中值滤波窗口半宽（段索引）：平滑值曲线，误读=尖峰
SEG_DETECT_FLOOR: float = 3.0   # 带宽下限 (km/h)：防 ±1-2 噪声被 flag
                                # （floor4×mult2=gate8 会漏 8-off 尖峰，如
                                # test.mp4 1499 段 160 在 168 平板上）
SEG_SINGLE_FLOOR: float = 2.0   # 单帧段专用带宽下限：单帧段误读率 4.2% vs
                                # 多帧 0.3%（12.6×，80% 误读是单帧段）→ 平缓区
                                # gate 4 抓 ≥5-off 单帧误读；弯曲区按实际带宽
                                # （↓到 1.5/1.0 虽提升 ±1 召回 94.5→96.7%，但
                                #  当前纠错把正确单帧段改错 → test 19→22/23 回归，
                                #  需配合纠错保守化（下一步）才可放宽）
SEG_ANCHOR_MAX_FRAMES: float = 120.0  # 纠错锚点最大帧距离：近锚点才插值（防远锚点误插值）

# ═══════════════════ 段级置信度（中值偏差 + 急动度加权，供 DP 锚定） ═══════════════════
SEG_CONF_W_MED: float = 0.7       # 中值偏差信号权重（主导锚定：紧邻误读的
                                  # 正确段中值分高 → 被 pin，防 DP 平滑拖走）
SEG_CONF_W_JERK: float = 0.3      # 急动度信号权重：辅助区分（刹车中值低但
                                  # 急动度高 → conf 中，raw 观测保其不变）
SEG_CONF_JERK_SCALE: float = 3.0  # 急动度分指数尺度 (km/h)：100*exp(-jerk/scale)

# ═══════════════════ 段级稠密格点 DP 纠正（对齐旧 viterbi_dense） ═══════════════════
# 观测 = 纯惩罚偏离 raw（旧系统 ref 来自重 OCR，重 OCR 已删 → ref 删除）。
# 观测存在的意义：惩罚任何改动，防止把正确的改错。DP 只在转移平滑性
# （加速度约束）强烈要求时移动值。
SEG_DP_OBS_WEIGHT: float = 1.0      # 观测权重：非锚点填向局部锚点插值（曲线），
                                    # 高权重让 DP 输出精确贴合曲线（锚点插值
                                    # 本身给基线，DP 再加全局平滑处理运行）
SEG_DP_ACCEL_WEIGHT: float = 1.0    # 转移权重：超加速度约束的二次惩罚
SEG_DP_MAX_DV_CAP: float = 4.0      # 每段转移最大变化 (km/h)：max_dv = min(
                                    # max_accel×dt×3.6, cap)。长段间距时
                                    # max_accel×dt 过松（8-off 跳变免费），
                                    # cap 保证误读跳变被惩罚、DP 拉正
SEG_DP_ANCHOR_COST: float = 0.1     # 高置信段锚定代价（固定到 raw）
SEG_DP_CHANGE_THRESHOLD: float = 3.0  # |DP输出 - raw| > 此值才修正：干净视频
                                      # 1-off 拉偏不提交；放宽到 3.0 消掉 2-off
                                      # 正确段被 DP 微调改错（gamma raw 下实测
                                      # 误改 2→0，漏纠不变，最终 15→13）
SEG_DP_ANCHOR_CONF: float = 20.0   # 锚定阈值：conf ≥ 此值的段固定到 raw
                                   # （门控 conf 后正确段 p10=72 干净分离，
                                   #  T=20 pin 100% 正确、仅 9% 误读）

# ═══════════════════ 孤立尖峰豁免（A4，13→12 实测） ═══════════════════
# conf∈[20,50) 的锚定段若 jerk（二阶差分）中等 → 解除锚定交给 DP。
# 判别依据（5 视频 722 段实测）：真刹车 jerk≈0（713 段全在 [0,9] 且
# 绝大多数 [0,4]）、丢位邻居污染 jerk≥80（9 段）、孤立尖峰误读 jerk 中等
# （如 test#74 raw=107 truth=103 jerk=9 —— 锚定会保留误读）。带通 [5,40]
# 只抓尖峰：解锚 24 段中 23 误读 + 1 正确（正确段也未被改坏），13→12
# 零误改。参数敏感性：下界 0 灾难（刹车全解锚，78 误改）、下界 3-8 ×
# 上界 20-60 全部稳定 12。0=禁用豁免。
SEG_DP_DEANCHOR_JERK_MIN: float = 5.0
SEG_DP_DEANCHOR_JERK_MAX: float = 40.0

# ═══════════════════ OCR 输入 pad 宽度下限 ═══════════════════
# 速度数字是窄图（48 高后 78-160 宽）。v6_small 在宽 pad 更准
# （test6：224→err 0.09%，192→0.16%，48~96→0.69~1.19%；256 精度相同但更慢）。
OCR_PAD_WIDTH_MIN: int = 224
OCR_PAD_WIDTH_MIN_BY_MODEL: dict[str, int] = {
    "v6_small": 224,
}

# ═══════════════════ 速度单位转换 ═══════════════════
SOURCE_TO_KMH: dict[str, float] = {
    "m/s": MPS_TO_KMH,
    "km/h": 1.0,
    "mile/h": 1.609344,
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
# 流水线队列：混合解码各后端队列上限；OCR 预处理→推理队列上限
HYBRID_QUEUE_SIZE: int = 8
OCR_INFER_QUEUE_SIZE: int = 4
# 分段 Otsu 阈值校准帧数（前 N 帧；seek 校准代价高，前段与全片抽样一致）
SEG_CALIB_FRAMES: int = 50
# Otsu 无法计算时的兜底阈值（0-255）
OTSU_FALLBACK_THRESH: int = 127
# 解码器无法给出 fps 时的兜底帧率
DEFAULT_FPS_FALLBACK: float = 30.0
# _local_bandwidth 的帧窗口上限（config.SEG_WIN 注释中 "上限 120 帧" 的实体）
SEG_WIN_MAX_FRAMES: float = 120.0
# 段级置信度的结构性门槛（历史调参结论，勿单独改动）
SEG_CONF_MIN_NEIGHBORS: int = 3
SEG_CONF_SHORT_NEIGHBOR: float = 30.0
SEG_CONF_EDGE: float = 100.0
SEG_CONF_MED_GATE: float = 50.0
# 一致性孤岛下限（近似/带波动 run）：累计帧数少于该值，即使局部中值贴合，
# conf 也封顶到 SHORT_RUN_CAP（不能成为 HIGH_TRUST/DP 锚点）。
# 127,128 这类 2 帧近似孤岛 → 不信任；带小波动的 3 帧以上才允许高置信。
SEG_CONF_MIN_CONSISTENT_FRAMES: int = 3
# 完全相同（无内部波动）的 run 需要更多帧才允许高置信：4 帧连续 127 这种
# 短促平坦孤岛仍不锚定；带小波动的近似 run 反而 3 帧即可信（更像真实斜坡）。
SEG_CONF_MIN_CONSISTENT_FRAMES_EXACT: int = 5
SEG_CONF_SHORT_RUN_CAP: float = 15.0
# 一致性孤岛的“近似相同”容差 (km/h)：孤岛内允许的小波动范围。
# 只靠完全相同会把 127,128 这种两个误读互相撑腰的短孤岛漏掉。
SEG_CONF_ISLAND_TOL: float = 2.0
# 一致性孤岛还需“脱离曲线”：短近似相同值相对 run 外邻居中值的偏差
# > 局部带宽×该倍率 才封顶（防止误伤坡道上的正常短段）
SEG_CONF_ISLAND_DEV_MULT: float = 3.0
# A4 孤立尖峰豁免的 conf 上界（下界=SEG_DP_ANCHOR_CONF）
SEG_DP_DEANCHOR_CONF_MAX: float = 50.0

# ═══════════════════ 第二遍尖峰检测（孤立 2-off 单帧误读，v2.16 实验） ═══════════════════
# 生产第一遍（detect+conf+DP）去污染后，对未改动的 len=1 段做"孤立尖峰"
# 判别：±k 段窗口两侧中值一致偏离 ≥ thresh 且 raw 值在邻域内不重复。
# 修正目标 = 离 raw 更远的一侧中值；|raw-target| ≥ min_fix 才提交。
# 实测（5 视频夹具全量）：final 11→5（test 3→0 / test2 8→5），harm=0，
# 零误改；剩余 5 个为真信息论极限（truth 瞬时跳变/1-off 锚点误差传播）。
SEG_SPIKE_K: int = 2            # 侧窗口中值半宽（段索引）
SEG_SPIKE_THRESH: float = 2.0   # 至少一侧偏离阈值 (km/h)
SEG_SPIKE_MIN_FIX: float = 2.0  # 提交改动最小偏差 (km/h)
SEG_SPIKE_MIN_NBR: int = 2      # 每侧最少邻居数
# 第二遍尖峰检测的最低帧率：低于该帧率跳过（30fps 模拟实测：低帧率下
# 相邻段真实速度变化 1-2 km/h（赛车急加速 30-60 km/h/s），正确段的孤立
# 凸起与 2-off 误读不可区分 → 误改 9/修对 2 净负；57fps 下修 6 零误改）。
SEG_SPIKE_MIN_FPS: float = 40.0
# OCR 引擎内部：ONNX 单批上限与 CTC 归约分块（内存峰值控制）
OCR_ONNX_CHUNK: int = 16
OCR_CTC_CHUNK: int = 64
# TensorRT 引擎构建：默认 batch profile、输入宽 profile 与 workspace
TRT_PROFILE_BATCH: int = 6
TRT_PROFILE_MIN_W: int = 32
TRT_PROFILE_OPT_W: int = 320
TRT_PROFILE_MAX_W: int = 2048
TRT_WORKSPACE_BYTES: int = 1 << 30
# TRT 引擎缓存文件名的 SM 后缀（引擎与 GPU 架构绑定）
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
