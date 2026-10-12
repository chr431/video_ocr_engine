"""测量原语复用层 —— 常见实验的原语库（2026-10-11 可拓展性轮）。

动机
----
视频注册表（VIDS/ROI）此前在 bench.py 与 record.py 各有一份完全相同的
拷贝，test_gpu_equivalence / test_leak_regression 再硬编码同一个 ROI
元组，window_matrix / h264_hybrid 又各有变体——共 5 处。新建一个探针
平均要重抄一遍"视频路径 + ROI + 预热计时 + 结果落盘"。本模块把这些
原语收敛为单一事实源，新实验直接 import，不再复制粘贴。

内容（只收已有 ≥2 个真实调用方的原语，不做前瞻抽象）：
  · VIDS / ROI          —— 视频注册表（bench.py / record.py 已接入；
                           经 env_probe.video_dir() 解析，RACELOG_VIDEO_DIR
                           生效，符合"外部视频走环境变量"纪律）
  · capture_shas        —— 提取结果的指纹摘要（record.py 已接入）
  · paired_ab           —— 配对交错 A/B 计时（臂序轮转 + 同轮配对；
                           带宽/判定复用 bench._band，单一实现）
  · quick_guard         —— 真值视频环境守卫（缺视频 → pytest.skip）

纪律：新增第 3 个重复实现前必须下沉到这里（rules.yaml「测量原语复用」）。
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from typing import Callable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from env_probe import video_dir  # noqa: E402  环境事实源（RACELOG_VIDEO_DIR）

# ── 视频注册表（单一事实源；bench.py / record.py 均 import 自此）────────
VIDS: dict[str, str] = {
    "test5": os.path.join(video_dir(), "test5.mp4"),  # h264, 7761 帧
    "test6_av1": os.path.join(video_dir(), "test6.mp4"),  # av1, 23970 帧
    "test6_h264": os.path.join(video_dir(), "test6_h264.mp4"),
    "test6_hevc": os.path.join(video_dir(), "test6_hevc.mp4"),
}
ROI: dict[str, tuple] = {"test5": (843, 993, 948, 1025), "test6": (841, 994, 949, 1026)}

#: 真值视频齐套判定（pytest fixture / 探针守卫共用）
has_real_videos = all(os.path.isfile(p) for p in VIDS.values())


def quick_guard(reason: str = "需要真值视频（RACELOG_VIDEO_DIR）") -> None:
    """缺真值视频时的统一 skip 出口（pytest 环境外为 no-op）。"""
    if not has_real_videos:
        try:
            import pytest

            pytest.skip(reason)
        except ImportError:  # noqa: S110 探针上下文：调用方自行判 has_real_videos
            pass  # 探针上下文：调用方自行判 has_real_videos


# ── 提取结果指纹（record.py 的 sha 摘要逻辑收敛于此）───────────────────
def capture_shas(result, keep_crops: bool = False) -> dict:
    """ExtractionResult → 指纹摘要（段结构 / 文本 / 裁切字节）。"""
    segs = [[s.start, s.end, s.rep_frame, s.text, s.confidence] for s in result.segments]
    out = {
        "n_segments": len(segs),
        "seg_struct_sha": hashlib.sha256(
            json.dumps([[s[0], s[1], s[2]] for s in segs]).encode()
        ).hexdigest(),
        "text_sha": hashlib.sha256("".join(s[3] for s in segs).encode()).hexdigest(),
    }
    if keep_crops:
        out["crop_sha"] = [
            hashlib.sha256((s.rep_crop.tobytes() if s.rep_crop is not None else b"")).hexdigest()[
                :16
            ]
            for s in result.segments
        ]
    return out


# ── 配对交错 A/B 计时（臂序轮转 + 同轮配对差分）────────────────────────
def paired_ab(
    run_a: Callable[[], float], run_b: Callable[[], float], pairs: int = 6, warmup: int = 1
) -> dict:
    """交错执行两臂（AB/BA 逐对轮转），返回配对差分与带宽摘要。

    先各跑 warmup 次预热（引擎池/时钟），再 pairs 对 AB/BA 轮转。
    返回 {"a": [...], "b": [...], "diffs_pct": [...], "band": {...}}；
    band 复用 bench._band（均值 + 3×SE 判据的单一实现）。
    """
    from bench import _band  # 惰性 import 防环

    for _ in range(warmup):
        run_a()
        run_b()
    a: list[float] = []
    b: list[float] = []
    for i in range(pairs):
        if i % 2 == 0:  # AB/BA 臂序逐对轮转
            a.append(run_a())
            b.append(run_b())  # noqa: E702 — 交错对紧凑写法
        else:
            b.append(run_b())
            a.append(run_a())  # noqa: E702
    diffs = [(y - x) / x * 100.0 for x, y in zip(a, b)]
    return {
        "a": a,
        "b": b,
        "diffs_pct": diffs,
        "band": _band(diffs),
        "median": (statistics.median(a), statistics.median(b)),
    }


# ── 后台负载门禁（R-perf-quiet，S3 Phase E）────────────────────────────
# 旧制度"标定/实验期间机器独占，后台任务禁止并行"是 prose（从未执行过）；
# 本节把它变成硬门禁：性能测量启动前采样外部进程负载，超阈立即报错退出，
# 不浪费整轮测量（同码两跑差 7.7% 的教训之一即后台噪声）。

LOAD_GATE_ERROR_HINT = "关掉后台任务后重测（R-perf-quiet）"


class BackgroundLoadError(RuntimeError):
    """后台负载超阈——测量前中止（R-perf-quiet）。"""


def _judge_quiet(readings, gpu_pct, cpu_pct, gpu_pct_limit=50.0):
    """判据核心（可注入假数据单测）：任一轮外部 CPU 超阈或 GPU 超阈即违规。

    GPU 阈值校准（2026-10-12 实测）：桌面合成与 agent 界面常态 20-40%
    （ZCode.exe≈31% + dwm≈28%），不属"大负载"；外来计算负载（编码/训练/
    渲染作业）才会持续 >50%。CPU 按整机折算，15% ≈ 5 个满核（32LP 机）。
    """
    for i, (sys_pct, _top) in enumerate(readings):
        if sys_pct > cpu_pct:
            return f"第{i + 1}轮外部 CPU {sys_pct:.0f}% > {cpu_pct:.0f}%"
    if gpu_pct > gpu_pct_limit:
        return f"GPU 利用率 {gpu_pct:.0f}% > {gpu_pct_limit:.0f}%"
    return None


def _gpu_util():
    """GPU 利用率（%）；无 nvidia-smi/超时=环境事实，返回 0 跳过该判据。"""
    import subprocess as _sp

    try:
        r = _sp.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001,S110 — 无卡/无工具/超时=环境事实
        return 0.0


def _prime_foreign_processes():
    """外部进程列表（排除自身进程树与 Idle）并 prime cpu_percent 计数器。"""
    import os

    import psutil

    own = {os.getpid()}
    try:
        own |= {q.pid for q in psutil.Process(os.getpid()).children(recursive=True)}
    except psutil.Error:  # noqa: BLE001 — 子进程枚举失败按无子进程处理
        pass
    procs = []
    for q in psutil.process_iter(["pid", "name"]):
        try:
            if q.pid in own or q.pid == 0:
                continue
            if "idle" in (q.info.get("name") or "").lower():
                continue
            q.cpu_percent(interval=None)  # 首调恒 0，仅 prime
            procs.append(q)
        except psutil.Error:  # noqa: BLE001 — 进程消失=正常竞态
            continue
    return procs


def assert_machine_quiet(
    cpu_pct: float = 15.0,
    samples: int = 3,
    interval: float = 1.0,
    gpu: bool = True,
    gpu_pct_limit: float = 50.0,
) -> None:
    """性能测量前的机器独占门禁：超阈抛 BackgroundLoadError（含元凶清单）。

    外部 CPU 按进程求和折算整机百分比（psutil 的 cpu_percent 是单核归一）；
    本进程与子进程不计入（测量自身的负载不是"后台"）。
    """
    import time as _t

    import psutil

    procs = _prime_foreign_processes()
    ncpu = psutil.cpu_count() or 1
    readings = []
    for _ in range(samples):
        _t.sleep(interval)
        total, top = 0.0, []
        for q in procs:
            try:
                c = q.cpu_percent(interval=None)
                total += c
                if c > 3.0:
                    top.append((c, q.info.get("name") or "?"))
            except psutil.Error:  # noqa: BLE001 — 进程消失=正常竞态
                continue
        readings.append((total / ncpu, sorted(top, reverse=True)))
    gpu_pct = _gpu_util() if gpu else 0.0
    bad = _judge_quiet(readings, gpu_pct, cpu_pct, gpu_pct_limit)
    if bad:
        detail = "；".join("%s=%.0f%%" % (n, c) for c, n in readings[-1][1][:5])
        raise BackgroundLoadError(
            "后台负载超阈（%s）——%s。元凶：%s" % (bad, LOAD_GATE_ERROR_HINT, detail or "GPU")
        )
