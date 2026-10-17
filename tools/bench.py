"""bench —— 性能报告矩阵、run registry 与 `bench diff`。

把"每次 A/B 重写一段对照脚本"变成一条命令：
  python tools/bench.py run --label before --telemetry full
  python tools/bench.py run --label after  --telemetry full
  python tools/bench.py diff before after      # 双档判定

报告落地 `bench/registry.jsonl`（gitignored、append-only）；A/B 的数字一律
来自报告对比，不来自"探针对探针"的手工对照。
`telemetry-check` 是遥测成本互比门禁：**同轮配对差分** + 符号一致性判
失败，阈值按本机 A/A 噪声带标定（`--aa` 现测；阈值须高于同码可分辨力）。
"""

from __future__ import annotations

import argparse
import json
import os
import re as _re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# 遥测成本门禁（**校准版**）：阈值由本机 A/A 标定得出（同档两槽、同进程
# 交替、每槽 3 次均值、25 对：均值偏差 −0.179%、sd 1.053%、SE 0.211%
# → 0.179+3×0.211 ≈ 0.81（现役限见 PI15_LIMITS）。
#   - 只有一条 full 可分辨下限；µs 级严格性由
#     `tests/config/test_telemetry_cost.py` 的确定性成本守卫承担
#     ——那里有 10⁴ 倍于墙钟的分辨力。
#   - 判失败还需**符号多数一致**（70%），落在噪声带里的差值判"不可判定"。
#   - 换机器/换窗口/改配对数 n 后必须 `telemetry-check --aa` 重标（SE∝1/√n）。
#   - 实配对 full−off ≈ +0.1%（限 PI15_LIMITS["full_pct"]）；成本门禁配方 = --inner 1
#     （A/A floor ≪ 限，3× 便宜，bench/aa_floor_curve.json）。
PI15_LIMITS = {"full_pct": 1.20, "sign_majority": 0.70}
#: 阈值下限（抽成常量）：取 0.20%——标定证据 |均值|+3SE =
#: 0.291/0.209/0.274/0.198% 中，n=50 已触 0.20，地板再高会 Masking
#: 真实分辨率。建议值随 n 增大收紧（SE∝1/√n），默认 n=25 多在
#: 0.25~0.30；PI15_LIMITS 的限与 n=25 匹配。
PI15_FLOOR_PCT = 0.20

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
REGISTRY = ROOT / "bench" / "registry.jsonl"
#: A/A 标定账本（P2-2）：最近一次 `ab --aa` 的噪声带 + 机器指纹，append 落盘。
#: 存在理由：噪声带只在本机当时有效，换机/换窗口必须重标——此前这条
#: 纪律只写在文档里（R-threshold-resolvable 的 --aa 提示），没有执行体，
#: 跑 ab 的人无从知道自己手里的阈值是否已经漂了。现在 ab 启动时自动
#: 比对：指纹不符或带显著变宽即告警（不判失败——只提示重标）。
CALIBRATION = ROOT / "bench" / "aa_calibration.jsonl"

# 视频注册表单一事实源 = tools/_measure_lib.py；经 RACELOG_VIDEO_DIR 解析。
from _measure_lib import ROI, VIDS, capture_shas  # noqa: E402

#: 矩阵视图可扫的旋钮（名 → 值类型）；值经 _round 的同名 kwarg 透传。
MATRIX_KNOBS = {
    "fill_width": int,
    "force_aspect": float,
    "buffer_size": int,
}

#: 配置集（与 tests/golden/bench_baseline.py 同口径，可比）
CONFIGS = {
    "h264-gpu": dict(video="test5", decode_backend="nvdec"),
    "h264-cpu": dict(video="test5", decode_backend="cpu"),
    "hevc-nvdec": dict(video="test6_hevc", decode_backend="nvdec"),
    "av1-nvdec": dict(video="test6_av1", decode_backend="nvdec"),
    "h264-hybrid": dict(video="test5", decode_backend="hybrid"),
    "av1-hybrid": dict(video="test6_av1", decode_backend="hybrid"),
    # hybrid 诊断：解码器层实测 hybrid_gpu(2311) > nvdec(2032)
    # —— 引擎口径必须自己量。
    "hevc-hybrid": dict(video="test6_hevc", decode_backend="hybrid"),
    "hevc-cpu": dict(video="test6_hevc", decode_backend="cpu"),
    "av1-cpu": dict(video="test6_av1", decode_backend="cpu"),
    # 同内容族 h264（跨编码对照的第三支柱）；解码后端由 --decode-backend
    # 覆盖，故此处默认值只作占位。
    "h264same-nvdec": dict(video="test6_h264", decode_backend="nvdec"),
    "h264same-cpu": dict(video="test6_h264", decode_backend="cpu"),
    "h264same-hybrid": dict(video="test6_h264", decode_backend="hybrid"),
}


def _round(
    cfg_name: str,
    window: int,
    telemetry: str,
    keep_crops: bool,
    ocr_backend: str,
    rep_format: str = "",
    buffer_size: int | None = None,
    decode_override: str = "",
    fill_width: int | None = None,
    video: str = "",
    force_aspect: float | None = None,
) -> dict:
    """一轮 extract（含指纹：段结构 sha + 文本 sha + 段数）。

    `video`（P2-1）：覆盖 config 的视频键（空=用 config 自带的），供
    批量视图在同一配置下逐文件跑——C-53 口径的"逐文件顺序"批量形态。
    指纹（capture_shas）随之入账：跨文件/跨轮可做段落结构一致性回归
    （文本 sha 跨轮不一致 = 非确定性缺陷的直接读数）。
    `force_aspect`：生产调用透传（真值头 test5=1.5）；None=引擎默认。
    """
    from video_ocr_engine import FieldExtractor

    cfg = CONFIGS[cfg_name]
    vid = video or cfg["video"]
    kw = {}
    if rep_format:
        kw["rep_crop_format"] = rep_format
    if buffer_size:
        kw["buffer_size"] = buffer_size
    if fill_width is not None:
        kw["fill_width"] = fill_width  # 0 = 关闭填充（批内自适应）
    if force_aspect is not None:
        kw["force_aspect"] = float(force_aspect)
    # 遥测档位经 env 进引擎（resolve 的 env 优先级链）；进程内改
    # os.environ 与 telemetry-check 的 in-process 口径一致。
    _prev_tier = os.environ.get("VOE_TELEMETRY")
    os.environ["VOE_TELEMETRY"] = telemetry
    try:
        ex = FieldExtractor(
            VIDS[vid],
            ROI["test5" if vid == "test5" else "test6"],
            frame_start=0,
            frame_end=window,
            decode_backend=(decode_override or cfg["decode_backend"]),
            ocr_backend=ocr_backend,
            keep_crops=keep_crops,
            sample_stride=cfg.get("sample_stride", 1),
            **kw,
        )
        t0 = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t0
    finally:
        if _prev_tier is None:
            os.environ.pop("VOE_TELEMETRY", None)
        else:
            os.environ["VOE_TELEMETRY"] = _prev_tier
    return {
        "wall": round(wall, 4),
        "n_segments": len(r.segments),
        "timing": {k: round(v, 4) for k, v in r.timing.items()},
        "report": r.meta.get("report"),
        "shas": capture_shas(r),
        "video": vid,
    }


def _gpu_state() -> str:
    """GPU 状态串（A/B 记录用）：掉频/热降是本地最大噪声源。

    NVML 直读（nvidia-smi 子进程实测 43.1ms/次；NVML 会话进程级复用后
    微秒级，与 report 环境指纹共用 `resources.nvml_handle`）。
    """
    try:
        import ctypes
        from video_ocr_engine.domain.resources import nvml_handle

        nvml, h = nvml_handle()
        temp, sm, pw = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        parts = []
        if int(nvml.nvmlDeviceGetTemperature(h, 0, ctypes.byref(temp))) == 0:
            parts.append("temp=%dC" % temp.value)
        if int(nvml.nvmlDeviceGetClockInfo(h, 1, ctypes.byref(sm))) == 0:
            parts.append("sm=%dMHz" % sm.value)
        if int(nvml.nvmlDeviceGetPowerUsage(h, ctypes.byref(pw))) == 0:
            parts.append("power=%dmW" % (pw.value // 1000))
        return " ".join(parts) or "n/a"
    except Exception:
        return "n/a"


# ── 时钟门禁：协议降噪 ─────────────────────────────────────────
# 判据层：把 NVML sm_clock min 与主动压频位登记进每一轮读数，"时钟未起"
# 的轮次不进聚合——不可比读数应当剔除，而不是被平均掉（冷轮
# sm_min 345~450MHz，弃冷后噪声带收窄 54×）。
# k 校准（bench/clock_gate.json）：冷轮 sm_min/最大SM ≈ 0.145、热轮平台
# ≈ 0.870（boost 上限 3105、负载平台 2700），k=0.75 居分离带中段
# （两侧 >10pp 余量；k=0.85 仅剩 2.3% 余量，k=0.9 会把热轮全拒）。
CLOCK_GATE_K = 0.75
#: 主动压频位（resources.py THROTTLE_BITS 的"非常态"集合：gpu_idle/
#: apps_clocks_setting/sync_boost 为常态）。任一出现 = 该 tick 主动压时钟。
_THROTTLE_BAD = 0x4 | 0x8 | 0x20 | 0x40 | 0x80 | 0x100


def _sm_max_clock() -> int | None:
    """nvmlDeviceGetMaxClockInfo(SM)；不可用 → None（门禁降级为"无信号"）。"""
    try:
        import ctypes
        from video_ocr_engine.domain.resources import nvml_handle

        nvml, h = nvml_handle()
        c = ctypes.c_uint()
        if int(nvml.nvmlDeviceGetMaxClockInfo(h, 1, ctypes.byref(c))) != 0:
            return None
        return int(c.value)
    except Exception:  # 无 NVML = 本机不可判，不是错误
        return None


def _gpu_valid(gpu: dict | None) -> bool:
    """记录级门禁判定：无 gpu 字段 / gate=off/unavailable → True。

    门禁只在有明确低时钟/压频证据时剔除（fail-open），且**不追溯改判**
    缺 gpu 字段的记录（缺失视为 valid）。
    """
    if not gpu:
        return True
    return bool(gpu.get("valid", True))


class _RoundClockWatch:
    """cmd_run 每轮的 GPU 时钟观察：NvmlSampler 包住一轮 extract。

    `with` 退出后 `block()` 给出 registry 的 gpu 字段：
    {sm_min, sm_p50, sm_max, throttle_bad, valid, gate}。
    valid = 无主动压频 tick 且 sm_min ≥ CLOCK_GATE_K×最大SM 时钟；
    采样点不足（run 短于首个 0.2s tick）判 valid=True 并注明——门禁
    不能替无数据，也不能无证据剔除。

    **边界（P2-3 实测）**：门禁假设"两臂负载相当，低时钟=环境干扰"。
    当被测旋钮本身改变 GPU 负载（如 fill_width 关填充 → OCR 更省 →
    GPU 更闲 → SM 时钟自然落到 2070~2250 < 0.75×最大）时，它把**处理
    效应**误判成干扰并整臂剔除（A/B 退化为"无有效配对轮"）。此类对比
    须 `--no-clock-gate`，改由同轮交错（`matrix`）或臂序轮转承受漂移。
    """

    def __init__(self, gate: bool = True, label: str = "") -> None:
        self._gate = gate
        self._label = label
        self._sm_max = None
        self._sampler = None
        self._pts: list = []
        self.unavailable = ""
        if gate:
            self._sm_max = _sm_max_clock()  # NVML 会话进程级一次初始化
            if self._sm_max is None:
                self.unavailable = "unavailable:NVML 或最大SM时钟不可读"

    def __enter__(self):
        if self._gate and not self.unavailable:
            from video_ocr_engine.domain.resources import NvmlSampler

            self._sampler = NvmlSampler(interval_s=0.2)
            self._sampler.start()
        return self

    def __exit__(self, *exc):
        if self._sampler is not None:
            self._sampler.stop()
            self._pts = list(self._sampler._pts)
            self._sampler = None
        return False

    def block(self) -> dict:
        if not self._gate:
            return {"valid": True, "gate": self._label or "off"}
        if self.unavailable:
            return {"valid": True, "gate": self.unavailable}
        pts = [(p[4], p[7]) for p in self._pts if p[4] is not None]
        if not pts:
            return {
                "valid": True,
                "sm_min": None,
                "sm_p50": None,
                "throttle_bad": 0,
                "gate": "unavailable:采样点不足",
            }
        sm = sorted(p[0] for p in pts)
        bad = sum(1 for _, m in pts if (m or 0) & _THROTTLE_BAD)
        sm_min, sm_p50 = sm[0], sm[len(sm) // 2]
        valid = bad == 0 and sm_min >= CLOCK_GATE_K * (self._sm_max or 0)
        return {
            "sm_min": sm_min,
            "sm_p50": sm_p50,
            "sm_max": self._sm_max,
            "throttle_bad": bad,
            "valid": valid,
            "gate": "nvml",
        }


def _set_priority_above_normal() -> bool:
    """SetPriorityClass(ABOVE_NORMAL)（实测可用、无需管理员）。

    伪句柄 -1 直传，绕开 GetCurrentProcess 的 restype 截断陷阱（64 位
    伪句柄若按默认 c_int 返回会被截成 32 位、调用静默失败）。
    """
    try:
        import ctypes

        k = ctypes.windll.kernel32  # type: ignore[attr-defined]
        return bool(
            k.SetPriorityClass(ctypes.c_void_p(-1), 0x00008000)
        )  # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:
        return False


def cmd_run(args) -> int:
    names = (
        args.config.split(",")
        if args.config
        else ["h264-gpu", "h264-cpu", "hevc-nvdec", "av1-nvdec"]
    )
    REGISTRY.parent.mkdir(exist_ok=True)
    rows = []
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    gate = not args.no_clock_gate
    if gate:
        _sm_max_clock()  # NVML 会话一次性初始化（~20ms 不进首轮）
    if args.priority and not _set_priority_above_normal():
        print("⚠ 优先级提升失败（忽略，按普通优先级继续）")
    for name in names:
        # 门禁只对真用 GPU 的配置生效（nvdec/hybrid 解码或 OCR≠cpu）：
        # 纯 CPU 配置下 GPU 空闲时钟恒低，判了只会全 invalid 再回退。
        uses_gpu = (
            CONFIGS[name]["decode_backend"] in ("nvdec", "hybrid") or args.ocr_backend != "cpu"
        )
        rounds = []
        for i in range(args.rounds):
            with _RoundClockWatch(
                gate=gate and uses_gpu, label="off:纯CPU配置不判GPU时钟"
            ) as watch:
                rec = _round(
                    name,
                    args.window,
                    args.telemetry,
                    args.keep_crops,
                    args.ocr_backend,
                    args.rep_format,
                    args.buffer_size,
                    getattr(args, "decode_backend", ""),
                    getattr(args, "fill_width", None),
                    force_aspect=getattr(args, "force_aspect", None),
                )
            rec["gpu"] = watch.block()
            rec["round"] = i + 1
            rounds.append(rec)
            gpu = rec["gpu"]
            note = "" if gpu.get("valid") else "  ⚠低时钟/压频（剔除出热轮）"
            print(
                "  %-12s round %d  %.4fs  %d 段%s"
                % (name, i + 1, rec["wall"], rec["n_segments"], note)
            )
        # 热轮 = 第 2 轮起（引擎热）∧ 时钟门禁 valid；全剔则回退全收——
        # 门禁不能替无数据。dropped 只数"本可入热轮但被门禁剔"的轮，
        # 引擎冷启动的第 1 轮是既有排除口径、不算门禁剔除。
        hot_pool = [r for r in rounds if r["round"] > 1 or args.rounds == 1]
        hot = [r["wall"] for r in hot_pool if r["gpu"].get("valid")]
        dropped = len(hot_pool) - len(hot)
        if not hot:
            hot = [r["wall"] for r in rounds]
            dropped = 0
            print("  ⚠ 时钟门禁后无热轮，回退全收")
        med = statistics.median(hot)
        spread = ((max(hot) - min(hot)) / med * 100) if len(hot) > 1 else 0.0
        print(
            "%-12s 热轮中位 %.4fs  散布 %.2f%%  段数 %d%s"
            % (
                name,
                med,
                spread,
                rounds[-1]["n_segments"],
                ("  剔除 %d 轮低时钟" % dropped) if dropped else "",
            )
        )
        for rec in rounds:
            rec.update(
                {
                    "label": args.label,
                    "config": name,
                    "commit": commit,
                    "telemetry": args.telemetry,
                    "window": args.window,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            rows.append(rec)
    with open(REGISTRY, "a", encoding="utf-8", newline="\n") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("已入库 %d 条 → %s" % (len(rows), REGISTRY))
    return 0


def _load(label: str) -> dict:
    """按 label 聚合：config → {wall 中位, 指标快照, 有效热轮}。

    `hot` 只收「第 2 轮起 ∧ 时钟门禁 valid」的 wall（无 gpu 字段
    视为 valid，不追溯改判）；`n_excluded` 是被门禁剔除的热轮数。
    `round_reports`：[(round, valid, report)]——逐指标配对用（P0-4）。
    """
    out: dict = {}
    if not REGISTRY.exists():
        raise SystemExit("registry 不存在：%s" % REGISTRY)
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("label") != label:
            continue
        c = out.setdefault(
            rec["config"],
            {
                "walls": [],
                "reports": [],
                "n_segments": [],
                "hot": [],
                "n_excluded": 0,
                "round_reports": [],
            },
        )
        c["walls"].append(rec["wall"])
        c["n_segments"].append(rec["n_segments"])
        if rec.get("round", 1) > 1:
            if _gpu_valid(rec.get("gpu")):
                c["hot"].append(rec["wall"])
            else:
                c["n_excluded"] += 1
        if rec.get("report"):
            c["reports"].append(rec["report"])
            c["round_reports"].append(
                (rec.get("round", 1), _gpu_valid(rec.get("gpu")), rec["report"])
            )
    if not out:
        raise SystemExit("label 无记录：%s" % label)
    return out


def _flatten(report: dict) -> dict:
    """报告 → 扁平指标表（wall 之外的逐项对比口径）。"""
    flat = {}
    for k, v in (report.get("spans") or {}).items():
        flat["span.sum:" + k] = v["sum"]
        flat["span.n:" + k] = v["n"]
    for k, v in (report.get("counters") or {}).items():
        flat["counter:" + k] = v
    for k, v in (report.get("gauges") or {}).items():
        flat["gauge:" + k] = v
    # 周期账本（C-63）：resources.per_phase 的原始 cycles 差分 → 配对
    # 差分口径（与 span/gauge 同款进归因表；非 Windows/来源缺席时无键）。
    per_phase = (report.get("resources") or {}).get("per_phase") or {}
    for ph, row in per_phase.items():
        if isinstance(row, dict) and row.get("cycles") is not None:
            flat["cyc:" + ph] = row["cycles"]
    e2e = per_phase.get("cycles_e2e")
    if isinstance(e2e, dict) and e2e.get("total") is not None:
        flat["cyc:e2e"] = e2e["total"]
        # 线程级全程账本（full 档；bench ab 需 --telemetry full）
        for name, c in (e2e.get("threads") or {}).items():
            flat["cyct:" + name] = c
    return flat


def _sig_stats(vals: list) -> dict:
    """配对差分的统计摘要（与 `ab` 的 `_paired_metric_diffs` 同款口径）。

    返回 {n, mean, se, p95abs, pos}；n<2 返 {}（无 SE 可算，调用方据此
    只报幅值不标显著性——不伪造统计量）。
    """
    import math

    n = len(vals)
    if n < 2:
        return {}
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals)
    se = sd / math.sqrt(n)
    av = sorted(abs(x) for x in vals)
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "p95abs": av[min(n - 1, int(0.95 * (n - 1)))],
        "pos": sum(1 for x in vals if x > 0),
    }


def _health_gate(reports: list, label: str) -> list:
    """health 红旗判定（P0-1）：任一报告的任一 PI 判据失败即红。

    返回 [(label, 键, 值, 限, 次数)] 事实清单（无红旗时空表）。判据本体在
    产品 `report.health`（PI_LIMITS 单源）——bench 只做"红旗 → 退出码"的
    接线，不在此复制阈值（复制即双事实源，阈值改一处忘一处必假绿）。
    逐轮重复命中去重（保留次数：N 轮全红 vs 偶发一轮性质不同）。
    """
    from video_ocr_engine.pipeline.report import PI_LIMITS, red_flags

    seen: dict = {}
    order: list = []
    for rep in reports:
        if not rep:
            continue
        for k in red_flags(rep):
            v = (rep.get("health") or {}).get(k) or {}
            lim = PI_LIMITS.get(k)
            key = (label, k, v.get("value"), lim[1] if lim else None)
            if key not in seen:
                seen[key] = 0
                order.append(key)
            seen[key] += 1
    return [(lbl, k, v, lim, seen[(lbl, k, v, lim)]) for lbl, k, v, lim in order]


def cmd_diff(args) -> int:
    a, b = _load(args.a), _load(args.b)
    hard, warn = args.hard, args.warn
    print("双档：硬失败 >%.1f%% / 告警 >%.1f%%（噪声底 %.1f%%）" % (hard, warn, args.noise))
    print("%-12s %10s %10s %9s  %s" % ("config", args.a, args.b, "Δ%", "判定"))
    worst = 0.0
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            print("%-12s  仅一侧有记录，跳过" % name)
            continue
        # 优先用门禁后的有效热轮；空则回退（walls[1:] / 全量）
        ha = a[name]["hot"] or a[name]["walls"][1:] or a[name]["walls"]
        hb = b[name]["hot"] or b[name]["walls"][1:] or b[name]["walls"]
        ma = statistics.median(ha)
        mb = statistics.median(hb)
        d = (mb - ma) / ma * 100
        worst = max(worst, abs(d))
        verdict = "硬失败" if abs(d) > hard else "告警" if abs(d) > warn else "通过"
        cut = ""
        if a[name]["n_excluded"] or b[name]["n_excluded"]:
            cut = "  [门禁剔除热轮 A:%d B:%d]" % (a[name]["n_excluded"], b[name]["n_excluded"])
        print("%-12s %10.4f %10.4f %+8.2f%%  %s%s" % (name, ma, mb, d, verdict, cut))
    # health 红旗（P0-1）：墙钟之外的失败判据，任一臂任一 PI 失败即退出 1
    red: list = []
    for lbl, data in ((args.a, a), (args.b, b)):
        for name in sorted(data):
            red += _health_gate(data[name]["reports"], "%s/%s" % (lbl, name))
    if red:
        print("\n⚠ health 红旗（PI 判据失败，与墙钟无关也判失败）：")
        for lbl, k, v, lim, n in red:
            print("  %-24s %-6s 值=%s（限 %s）×%d" % (lbl, k, v, lim, n))
    print("\n逐指标（配对差分口径，仅列变化 >%.1f%% 的项）：" % warn)
    # P0-4：与 `ab` 的 _attr_report 同款配对判据。diff 的两侧 label 是各自
    # 独立记录的（没有同轮交错关系），但**同轮号配对**仍成立：每侧第 k 轮
    # 报告互配，差分即"同一顺序位置上两 label 的差"。位置效应不能完全对消
    # （无交错），故判据比 ab 严一档且只作归因不改退出码的墙钟判据。
    # |均值|>3×SE ∧ 符号多数 ≥70% → `**`（强）；仅幅值超 warn → `~`。
    shown = 0
    for name in sorted(set(a) & set(b)):
        ra = {rn: rep for rn, valid, rep in a[name]["round_reports"] if rn > 1 and valid}
        rb = {rn: rep for rn, valid, rep in b[name]["round_reports"] if rn > 1 and valid}
        ks = sorted(set(ra) & set(rb))
        if not ks:
            continue
        fa = {rn: _flatten(ra[rn]) for rn in ks}
        fb = {rn: _flatten(rb[rn]) for rn in ks}
        keys = set().union(*[set(fa[k]) for k in ks]) if ks else set()
        for k in sorted(keys):
            diffs, bases = [], []
            for rn in ks:
                x, y = fa[rn].get(k), fb[rn].get(k)
                if x is None or y is None or x in (0, 0.0):
                    continue  # 缺键/零基值跳过（与旧口径一致）
                diffs.append((y - x) / x * 100.0)
                bases.append(x)
            if not diffs or abs(statistics.fmean(diffs)) <= warn:
                continue
            n = len(diffs)
            st = _sig_stats(diffs)
            sig = st and abs(st["mean"]) > max(3 * st["se"], 0.05)
            need = max(2, int(round(n * PI15_LIMITS["sign_majority"] + 0.5)))
            majority = sig and ((n - st["pos"]) >= need or st["pos"] >= need)
            mark = "**" if (sig and majority) else ("~" if sig else "  ")
            print(
                "  %-12s %-28s %10.4f → %10.4f  %+7.2f%%  SE %5.2f%% %2d/%-2d %s"
                % (
                    name,
                    k,
                    statistics.median(bases),
                    statistics.median(
                        [
                            fb[rn][k]
                            for rn in ks
                            if k in fb[rn] and fa[rn].get(k) not in (None, 0, 0.0)
                        ]
                    ),
                    statistics.fmean(diffs),
                    st["se"] if st else 0.0,
                    st["pos"] if st else 0,
                    n,
                    mark,
                )
            )
            shown += 1
    if not shown:
        print("  （无）")
    print(
        "  标记：** = |均值|>3×SE ∧ 符号多数≥70%（强）；~ = 仅幅值显著；"
        "判读同 `ab` 口径（C-42：归因≠可回收量）"
    )
    return 1 if (worst > hard or red) else 0


def _inproc_rounds(
    cfg: str,
    window: int,
    tiers: list,
    rounds: int,
    ocr_backend: str,
    warmup: int = 2,
    inner: int = 1,
) -> dict:
    """**同进程内交替**跑各档：返回 {(round, tier): wall}。

    为什么用同进程：子进程模式下每个测量点都要重付"解释器 + CUDA 上下文 +
    TRT 反序列化"，这些与插桩无关的方差远大于被测量本身（A/A 实测同档两槽
    |Δ|p95 2.33%）。同进程 + 预热轮让各档共享同一份热池与 OS 缓存，唯一差别
    就是插桩本身 → 配对差分才真正"配对"。

    `inner>1`：每槽连跑 inner 次取均值（槽内复制）。同档两槽实测 sd=1.89%，
    复制 √inner 倍压噪，比无脑加轮数便宜。
    """
    from video_ocr_engine import FieldExtractor

    c = CONFIGS[cfg]
    vid = c["video"]
    per: dict = {}
    for i in range(-warmup, rounds):
        k = i + warmup
        order = list(tiers)
        order = order[k % len(order) :] + order[: k % len(order)]  # 位置轮转
        for tier in order:
            os.environ["VOE_TELEMETRY"] = "full" if tier == "full2" else tier
            walls = []
            for _ in range(max(1, inner)):
                ex = FieldExtractor(
                    VIDS[vid],
                    ROI["test5" if vid == "test5" else "test6"],
                    frame_start=0,
                    frame_end=window,
                    decode_backend=c["decode_backend"],
                    ocr_backend=ocr_backend,
                    sample_stride=c.get("sample_stride", 1),
                )
                t = time.perf_counter()
                ex.extract()
                walls.append(time.perf_counter() - t)
            w = statistics.fmean(walls)
            if i >= 0:
                per[(i, tier)] = w
            print(
                "    [%s%d] %-5s %.4fs%s"
                % (
                    "warm" if i < 0 else "run",
                    i,
                    tier,
                    w,
                    "" if inner == 1 else "  (%d 次均值)" % inner,
                ),
                flush=True,
            )
    return per


def _subproc_rounds(args, tiers: list) -> dict:
    """冷启真口径：每档独立子进程（含解释器/CUDA/TRT 初始化）。

    保留作 `--subproc` 用：当怀疑"分档开销只在进程生命周期上体现"（例如
    细档采样器分配的大块内存在 teardown 时的成本）时用这条路径。默认判
    跨档插桩开销用同进程交替——子进程模式的噪声远大于被测量（A/A 实测）。
    """
    run_id = time.strftime("%m%d-%H%M%S")
    n = args.inner_rounds or 2
    for i in range(args.rounds):
        order = list(tiers)
        order = order[i % len(order) :] + order[: i % len(order)]
        for tier in order:
            real = "full" if tier == "full2" else tier
            env = dict(os.environ)
            env["VOE_TELEMETRY"] = real
            r = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "run",
                    "--label",
                    "pi15-%s#%d@%s" % (tier, i, run_id),
                    "--config",
                    args.config,
                    "--rounds",
                    str(n),
                    "--window",
                    str(args.window),
                    "--telemetry",
                    real,
                    "--ocr-backend",
                    args.ocr_backend,
                ],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if r.returncode != 0:
                print(r.stdout[-2000:] + r.stderr[-2000:])
                raise SystemExit(1)
            print(
                "  [%d] %-5s %s"
                % (i, tier, [ln for ln in r.stdout.splitlines() if "热轮中位" in ln])
            )
            time.sleep(args.gap)  # 连续起进程会让 TRT CUDA 初始化瞬态失败
    per: dict = {}
    rows = [
        json.loads(ln) for ln in REGISTRY.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    for rec in rows:
        lab = rec.get("label", "")
        if not lab.endswith("@" + run_id) or rec.get("round", 1) <= 1:
            continue  # 只统计本次 invocation 的热轮
        m = _re.match(r"pi15-(\w+)#(\d+)@", lab)
        if m and m.group(1) in tiers:
            per[(int(m.group(2)), m.group(1))] = rec["wall"]
    return per


def cmd_telemetry_check(args) -> int:
    """遥测成本两档互比（**同进程交替 + 同轮配对差分** + 符号一致性判失败）。

    方法学（口径设计）：
    1. 逐档交错，而非"两档各自连跑 N 轮"——后者会把机器漂移记进档位差
       （同码两次可差 7.7%，而阈值只有 0.1%）。
    2. 每轮轮转档位顺序（拉丁方），而非固定 off→std→full——固定顺序让轮内
       热漂移系统性落在最后一档（full 稳定 +3.3%，而 NVML/差分本身只值 µs 级）。
    3. **同轮内配对差分**，而非两档各自中位数相比——后者公共漂移没被减掉
       （同码两次判读可给出 std −0.50% 与 std +0.93%）。
    4. **同进程交替**，而非子进程模式——后者 A/A（同档两槽）噪声带实测
       |Δ|p50 0.24% / p95 2.33%，远大于被测量；同进程共享热池与 OS 缓存，
       唯一差别是插桩本身。
    5. 阈值按 A/A 现测校准（`--aa`），并要求**符号多数一致**才判失败——
       落在噪声带里的差值判"不可判定"，不误报。
    6. 两档互比：std 开销与 full 同在噪声内（0.115% vs 0.038%），互比收敛
       为 off vs full 单一成本限（full_pct）；A/A 槽位用 full/full2。
    """
    tiers = ["off", "full"]
    if args.aa:
        tiers = ["full", "full2"]  # A/A：两槽同档 → 差分即纯噪声
    if args.subproc:
        per = _subproc_rounds(args, tiers)
    else:
        per = _inproc_rounds(
            args.config, args.window, tiers, args.rounds, args.ocr_backend, inner=args.inner
        )
    med = {
        t: statistics.median([v for (i, tt), v in per.items() if tt == t and v is not None])
        for t in tiers
    }
    if any(v is None or v != v for v in med.values()):
        print("样本不足：%s" % med)
        return 1
    if args.aa:
        return _report_aa(per, med, args)
    return _report_pi15(per, med, args)


def _paired(per: dict, rounds: int, a: str, b: str) -> list:
    """同轮配对差分 %（b 相对 a）；缺任一读数的轮次跳过。"""
    out = []
    for i in range(rounds):
        va, vb = per.get((i, a)), per.get((i, b))
        if va and vb:
            out.append((vb - va) / va * 100.0)
    return out


def _band(vals: list) -> dict:
    """配对差分分布摘要（**统计量取均值**，见下方位置效应说明）。

    为什么是均值不是中位数：A/A 实测（同档两槽、25 对）给出**中位 +1.267%**
    ——同一段代码两槽不可能真有 1.3% 差别，真因是"**轮内位置效应**"：每轮里
    先跑的那一档系统性慢 ~1.3%（前一档的收尾/GC 还没落定）。档位顺序按轮
    轮转后，差值分布变成 ±X 双峰——**中位数落在其中一峰上（所以 A/A 中位
    非零），而均值把两峰对消**（13 正 12 负 → 残差仅 ~0.05%）。故判失败用
    均值 + 3×SE，并同时要求**符号多数一致**。
    """
    import math

    v = sorted(vals)
    av = sorted(abs(x) for x in vals)
    n = len(v)

    def q(arr, f):
        return arr[min(len(arr) - 1, int(f * (len(arr) - 1)))]

    med = statistics.median(v)
    mean = statistics.fmean(v) if n else 0.0
    sd = statistics.stdev(v) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    mad = statistics.median([abs(x - med) for x in v]) * 1.4826
    return {
        "n": n,
        "mean": round(mean, 3),
        "median": round(med, 3),
        "p50abs": round(q(av, 0.5), 3),
        "p95abs": round(q(av, 0.95), 3),
        "maxabs": round(av[-1], 3) if av else 0.0,
        "sd": round(sd, 3),
        "se": round(se, 3),
        "sigma_mad": round(mad, 3),
        "pos": sum(1 for x in v if x > 0),
    }


def _limit_from_band(b: dict) -> float:
    """阈值规则（写死，免得每次拍脑袋）：
    |A/A 均值偏差| + 3×SE（单侧 ~99.7%），向上取整到 0.05%，下限 0.30%。
    含义：**只有超过"同档两槽都能测出来的差别"的档位差才允许判失败**。
    标定用对数 n 必须与门禁一致（SE∝1/√n）。
    """
    cand = abs(b["mean"]) + 3 * b["se"]
    return max(PI15_FLOOR_PCT, -(-cand // 0.05) * 0.05)


def _report_aa(per: dict, med: dict, args) -> int:
    """A/A 标定：两槽同档 → 差分分布即纯机器噪声带。"""
    d = _paired(per, args.rounds, "full", "full2")
    if not d:
        print("A/A 无有效配对轮次")
        return 1
    b = _band(d)
    print(
        "\nA/A 噪声标定（同档两槽，%d 对，%s，config=%s window=%d）："
        % (b["n"], "子进程" if args.subproc else "同进程交替", args.config, args.window)
    )
    print(
        "  均值 %+.3f%%  中位 %+.3f%%  |Δ|p50 %.3f%%  |Δ|p95 %.3f%%  "
        "max|Δ| %.3f%%  sd=%.3f%% SE=%.3f%%"
        % (b["mean"], b["median"], b["p50abs"], b["p95abs"], b["maxabs"], b["sd"], b["se"])
    )
    rec = _limit_from_band(b)
    print(
        "  → 本机在 %d 对下可分辨 full 阈值 = +%.2f%%"
        "（规则 |均值偏差|+3×SE 上取整 0.05%%，下限 0.20%%；"
        "判失败另需符号多数一致）" % (b["n"], rec)
    )
    print(
        "  提示：阈值随配对数收紧（SE∝1/√n），标定 n 必须与门禁 n 一致；换机器/换窗口后必须重标。"
    )
    return 0


def _report_pi15(per: dict, med: dict, args) -> int:
    base = med["off"]
    d_full = _paired(per, args.rounds, "off", "full")
    lim_full = args.full_limit or PI15_LIMITS["full_pct"]
    need = max(2, int(round(len(d_full) * PI15_LIMITS["sign_majority"] + 0.5))) if d_full else 2

    def judge(d, lim):
        if not d:
            return False, "无配对样本"
        b = _band(d)
        m = b["mean"]  # 统计量=均值（位置效应对消，见 _band）
        signs = b["pos"] >= need or (len(d) - b["pos"]) >= need
        ok = (m <= lim) or not signs
        note = "" if ok or m <= lim else "（超阈且符号一致）"
        return ok, (
            "均值 %+.3f%%（SE %.3f%%，限 +%.2f%%）符号 %d/%d%s%s → %s"
            % (
                m,
                b["se"],
                lim,
                b["pos"],
                len(d),
                "一致" if signs else "不一致=噪声内",
                note,
                "通过" if ok else "失败",
            )
        )

    ok_f, txt_f = judge(d_full, lim_full)
    print(
        "\n遥测成本（%s，同轮**配对差分** %d 对；off 中位 %.4fs /"
        " full %.4fs）"
        % ("子进程" if args.subproc else "同进程交替", len(d_full), base, med["full"])
    )
    print("  full vs off ：%s" % txt_f)
    print(
        "  两档化后唯一成本限 full ≤ +%.2f%%；本机可分辨下限见 "
        "`bench.py telemetry-check --aa`（阈值=校准值 %s，用户裁决"
        "放松过严阈值以消除误报）" % (lim_full, lim_full)
    )
    ok = bool(ok_f)
    print("判定：%s" % ("通过" if ok else "失败（插桩密度或 off 档实现退化）"))
    return 0 if ok else 1


def cmd_show(args) -> int:
    """打印某 label 的报告细目（相位/计数/量值）——不看探针输出，看报告。"""
    data = _load(args.label)
    for name in sorted(data):
        if args.config and args.config != name:
            continue
        reps = [r for r in data[name]["reports"]]
        walls = data[name]["walls"]
        print(
            "\n=== %s / %s  wall=%s（中位 %.4fs，%d 段）"
            % (
                args.label,
                name,
                [round(w, 3) for w in walls],
                statistics.median(walls),
                data[name]["n_segments"][-1],
            )
        )
        if not reps:
            print("  （无报告：telemetry=off 档）")
            continue
        rep = reps[-1]
        if args.tier and rep.get("tier") != args.tier:
            print("  （报告档位 %s）" % rep.get("tier"))
        tot = 0.0
        for k in sorted(rep.get("spans", {})):
            v = rep["spans"][k]
            tot += v["sum"]
            print(
                "  span   %-24s n=%-5d sum=%8.4f p50=%8.5f max=%8.5f"
                % (k, v["n"], v["sum"], v["p50"], v["max"])
            )
        print(
            "  ---- span 合计 %.4f s（wall %.4f s，占比 %.1f%%）"
            % (tot, rep["wall_s"], tot / max(rep["wall_s"], 1e-9) * 100)
        )
        for k in sorted(rep.get("gauges", {})):
            print("  gauge  %-24s %12.5f" % (k, rep["gauges"][k]))
        for k in sorted(rep.get("counters", {})):
            print("  cnt    %-24s %12d" % (k, rep["counters"][k]))
        if rep.get("health"):
            print("  health %s" % json.dumps(rep["health"], ensure_ascii=False))
    return 0


def _pair_idx(label: str) -> int:
    """从 ab 的 `label#i@runid` 取配对轮号 i（归因配对键的组成部分）。"""
    m = _re.search(r"#(\d+)@", label)
    return int(m.group(1)) if m else -1


def _paired_metric_diffs(reports_a: list, reports_b: list, mode: str = "hot") -> dict:
    """归因：逐对配对差分的**指标级**口径（与 wall 判据同款统计纪律）。

    `reports_*` = [(label, round, valid, report)]。配对键 = **(对序号, 轮号)**
    ——对序号从 label 的 `#i@` 解析。⚠️ 按裸 round 号配对是错的：6 对
    的 12 份报告会被塌缩成 2 个键，表里的 "n=2" 实为两个塌缩伪样本（同码
    都能"显著"出 −44%±52% 的伪强信号）。

    mode="hot"：round>1 ∧ 时钟门禁 valid（与 wall 判定同口径）；
    mode="cold"：round==1（冷轮不过门禁——时钟爬坡正是冷启动的一部分）。

    为什么不用"两臂跨轮中位数相减"：那正是被否决的口径——
    公共漂移没减掉，而配对口径把同轮的机器状态差掉了。
    """
    import math

    slots = []
    for reports, is_hot in ((reports_a, True), (reports_b, True)):
        d: dict = {}
        for lab, rn, valid, rep in reports:
            if is_hot:
                if rn < 2 or not valid:
                    continue
            elif rn != 1:
                continue
            d[(_pair_idx(lab), rn)] = rep
        slots.append(d)
    ka, kb = slots
    out: dict = {}
    for key in sorted(set(ka) & set(kb)):
        fa = _flatten(ka[key])
        fb = _flatten(kb[key])
        for k in set(fa) & set(fb):
            base = fa[k]
            if base in (0, 0.0):
                continue
            rec = out.get(k)
            if rec is None:
                rec = out[k] = {"vals": [], "base": []}
            rec["vals"].append((fb[k] - base) / base * 100.0)
            rec["base"].append(base)
    res: dict = {}
    for k, rec in out.items():
        vals = rec["vals"]
        if len(vals) < 2:
            continue
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals)
        se = sd / math.sqrt(len(vals))
        av = sorted(abs(x) for x in vals)
        res[k] = {
            "n": len(vals),
            "mean": mean,
            "sd": sd,
            "se": se,
            "pos": sum(1 for x in vals if x > 0),
            "p95abs": av[min(len(av) - 1, int(0.95 * len(av)))],
            "base": statistics.median(rec["base"]),
        }
    return res


def _attr_report(name: str, ra: dict, rb: dict, pairs: list, limit: float) -> None:
    """逐指标归因表：逐对配对差分 + 显著性标记（**只归因，不判失败**）。

    判据（与 wall 判据同款，但只用于"这个指标动了没有"的陈述）：
      - `CI 排零`：|mean| > 3×SE（单侧 ~99.7%，与 `_limit_from_band` 同源）
      - `符号多数`：正号或负号 ≥ 70% 配对（对消位置效应后的稳定性）
    两者同时成立标 `**`（强归因）；仅幅值超限标 `~`（方向不稳）。
    ⚠️ C-42：归因≠可回收量——占比/差分只说明"谁动了"，可回收量必须靠
    绑定实验（零成本替换口径）判定。

    样本量门：**有效配对数 < max(3, ⌈pairs/3⌉) 的键不出表**——2 个样本的
    SE 没有意义（实测出现过 n=2 却算出 "−44%±52%" 的伪强信号）。样本不足
    会在表尾如实统计（不静默丢弃）。符号多数的分母用**该键的有效 n**，
    不是请求的对数（个别对缺键/被门禁剔除时诚实缩表）。
    """
    ra_list = ra.get(name, [])
    rb_list = rb.get(name, [])
    diffs = _paired_metric_diffs(ra_list, rb_list, mode="hot")
    if not diffs:
        return
    n_min = max(3, -(-len(pairs) // 3))  # ⌈pairs/3⌉ 上取整
    rows, skipped = [], []
    for k, v in diffs.items():
        if v["n"] < n_min:
            skipped.append((k, v["n"]))
            continue
        need = max(2, int(round(v["n"] * PI15_LIMITS["sign_majority"] + 0.5)))
        sig_ci = abs(v["mean"]) > max(3 * v["se"], 0.05)
        pos, n = v["pos"], v["n"]
        sig_sign = pos >= need or (n - pos) >= need
        mark = "**" if (sig_ci and sig_sign) else ("~" if sig_ci else "  ")
        rows.append((mark, k, v))
    if not rows:
        # 全被样本门挡下：如实说明（不静默）——4 对以下几乎必然发生，
        # 归因表要 ≥6 对才有统计意义（判据线本身也按对数缩放）。
        print(
            "逐指标归因：跳过（有效配对不足，%d 键全部 n<%d——归因请用 "
            "--repeat ≥6）" % (len(skipped), n_min)
        )
        return
    rows.sort(key=lambda t: -abs(t[2]["mean"]))
    strong = [r for r in rows if r[0] == "**"]
    weak = [r for r in rows if r[0] != "**"]
    show = (strong + weak)[:18]
    nz = len(strong)
    print(
        "逐指标归因（逐对配对差分，热轮口径；** = |均值|>3×SE ∧ 符号多数"
        "≥70%（按该键有效 n）；~ = 幅值够但方向不稳；符号列=正号数/n）："
    )
    print(
        "  %-30s %11s %11s %8s %7s %5s  %s"
        % ("指标", "A(中位)", "Δ均值%", "SE%", "p95|Δ|%", "符号", "")
    )
    for mark, k, v in show:
        print(
            "  %-30s %11.4f %+10.2f%% %7.2f%% %6.2f%% %2d/%-2d %s"
            % (k, v["base"], v["mean"], v["se"], v["p95abs"], v["pos"], v["n"], mark)
        )
    if skipped:
        print(
            "  （样本不足未列 %d 键：%s）"
            % (len(skipped), ", ".join("%s n=%d" % (k, n) for k, n in skipped[:6]))
        )
    print("  强归因 %d 项（**）；⚠ C-42：归因≠可回收量，可回收量须跑绑定实验" % nz)


def _cycle_verdict(name: str, ra: dict, rb: dict) -> None:
    """周期账本判据（C-63）：cycles 配对差分的判读输出。

    与 wall 判定同款统计纪律（CI 排零 ∧ 符号多数 ≥70%），但**只判读不改
    退出码**——wall 仍是唯一失败判据；cycles 的角色是墙钟分辨率不够时的
    第二意见（配对差分自带配对消噪，SMT 争用均值带见 C-63 前提）。
    主判据 = cyc:e2e / cyc:decode 两键；cyc:ocr / cyc:calibrate 与
    cyct:*（线程级，full 档才有）为信息性展示——小相位样本少，A/A 带宽
    达 27%/8.6%（C-64），不作结论。
    """
    diffs = _paired_metric_diffs(ra.get(name, []), rb.get(name, []), mode="hot")
    keys = [k for k in ("cyc:e2e", "cyc:decode", "cyc:ocr", "cyc:calibrate") if k in diffs]
    keys += sorted(k for k in diffs if k.startswith("cyct:"))
    if not keys:
        return
    print(
        "周期账本判据（C-63；同款纪律：|均值|>3×SE ∧ 符号多数≥70%；"
        "只判读不改退出码；正号=B 周期多=B 更费 CPU）："
    )
    for k in keys:
        v = diffs[k]
        need = max(2, int(round(v["n"] * PI15_LIMITS["sign_majority"] + 0.5)))
        sig = abs(v["mean"]) > max(3 * v["se"], 0.05)
        pos, n = v["pos"], v["n"]
        if sig and pos >= need:
            verdict = "B 周期显著多"
        elif sig and (n - pos) >= need:
            verdict = "B 周期显著少"
        else:
            verdict = "不可判定"
        print(
            "  %-14s A(中位)%8.3fG  Δ均值 %+7.2f%%  SE %5.2f%%  "
            "符号 %d/%d  → %s"
            % (k.split(":", 1)[1], v["base"] / 1e9, v["mean"], v["se"], pos, n, verdict)
        )


def _machine_fingerprint() -> dict:
    """本机指纹（P2-2）：噪声带只在"同机器同驱动同核数"下可比。

    GPU 名与驱动经 NVML（与 report.environment 同源）；不可用时记 None
    ——指纹缺失**不冒充一致**（比较端见 None 即降级为"无法核对"）。
    """
    import os

    fp: dict = {"cpu_count": os.cpu_count() or 0}
    try:
        from video_ocr_engine.domain.resources import gpu_identity

        ident = gpu_identity()
        fp["gpu"] = ident[0] if ident else None
        fp["driver"] = ident[1] if ident else None
    except Exception:  # NVML 缺席=环境事实
        fp["gpu"] = fp["driver"] = None
    return fp


def _fp_key(fp: dict) -> str:
    return "%s|%s|cpu%s" % (fp.get("gpu"), fp.get("driver"), fp.get("cpu_count"))


def _load_latest_calibration() -> dict | None:
    """最近一条 A/A 标定记录（append-only 账本的最后一行）。"""
    if not CALIBRATION.exists():
        return None
    last = None
    for line in CALIBRATION.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return last


def _save_calibration(rec: dict) -> None:
    CALIBRATION.parent.mkdir(exist_ok=True)
    with open(CALIBRATION, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _calibration_context(config: str, window: int) -> str:
    """ab 启动时的标定上下文（P2-2；只读提示，不改退出码）。

    三种状态如实区分：无标定 / 指纹不符（换机器——带不可比）/ 指纹相符
    （报年龄与建议阈值，供读者核对手里的 --hard 是否还是本机标定的）。
    """
    rec = _load_latest_calibration()
    if not rec:
        return "标定基线：无（建议先跑一次 ab --aa 固化本机噪声带）"
    fp_now = _machine_fingerprint()
    same = rec.get("fingerprint") == _fp_key(fp_now)
    age = ""
    try:
        import datetime as _dt

        t0 = _dt.datetime.strptime(rec["ts"], "%Y-%m-%dT%H:%M:%S")
        days = (_dt.datetime.now() - t0).days
        age = "，%d 天前" % days
    except Exception:  # noqa: S110 — 时间戳缺失/格式变=记录本身旧：只影响年龄列
        pass  # 时间戳缺失/格式变=记录本身旧：只影响年龄列，不阻断比对
    cfg = rec.get("config", "")
    win = rec.get("window", "")
    base = "标定基线：%s%s（config=%s window=%s，hard=%s%%，hard_cold=%s%%，n=%s）" % (
        same and "同机" or "**指纹不符**",
        age,
        cfg,
        win,
        rec.get("suggested_hard"),
        rec.get("suggested_hard_cold"),
        rec.get("n"),
    )
    if not same:
        base += " → 标定来自不同机器/驱动：噪声带不可比，必须重跑 ab --aa"
    return base


def cmd_ab(args) -> int:
    """交错 A/B（臂序轮转 + 时钟门禁 + 自动判定）。

    背景：**同一份代码**连跑两次 `bench run`，h264-cpu
    1.2567 → 1.1599s（−7.7%）、hevc −7.7%——功耗/热状态与后台进程造成的
    漂移**大于**被测量本身，顺序跑会把漂移记到 B 头上 → 必须交错。每个
    variant 起独立子进程（状态干净、各含一轮预热）。三层降噪：
      1. **臂序逐对轮转**（AB/BA 交替）——对消轮内位置效应（先跑的臂
         系统性慢 ~1.3%，telemetry-check 同款结论）；
      2. **每轮 GPU 时钟入账**（子进程 `bench run` 内做，registry `gpu`
         字段），低时钟/压频热轮不进聚合（冷轮 CV 6.6% → 热轮 0.12%）；
      3. **自动判定**：逐对热轮中位配对差分 → 均值 + SE + 符号多数（≥70%）
         vs `--hard` 阈值；B 显著慢 → 退出码 1（`--no-verdict` 回纯展示，
         只关墙钟判定）；
      4. **health 红旗**（P0-1）：报告的 PI 判据（engine_init/syncs）失败
         与墙钟无关也判失败——`--no-verdict` 不豁免它（正确性判据不能因
         "只展示"而静默）。判据本体在产品 `report.health`，此处只接线。
    `--aa`：两臂同配置（忽略 env/args 差异）→ 差分即子进程协议噪声带，
    打印建议阈值（规则 |均值偏差|+3×SE，下限 PI15_FLOOR_PCT）。

    用法：
      python tools/bench.py ab --config h264-cpu --repeat 3 \
          --a base --b opt --env-b VOE_XXX=1
    """
    import os

    run_id = time.strftime("%m%d-%H%M%S")
    aa = bool(args.aa)
    gate = not args.no_clock_gate
    pairs: list = []
    for i in range(args.repeat):
        row = {}
        arm_specs = (("A", args.a, args.env_a, args.args_a), ("B", args.b, args.env_b, args.args_b))
        if i % 2 == 1:
            arm_specs = arm_specs[::-1]  # 臂序轮转：对消位置效应
        for tag, label, envspec, extra in arm_specs:
            if aa:
                envspec, extra = "", ""  # A/A：两臂完全同配置
            lab = "%s#%d@%s" % (label, i, run_id)
            env = dict(os.environ)
            for kv in filter(None, envspec.split(",")):
                k, _, v = kv.partition("=")
                env[k.strip()] = v.strip()
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run",
                "--label",
                lab,
                "--config",
                args.config,
                "--rounds",
                str(args.rounds),
                "--window",
                str(args.window),
                "--telemetry",
                args.telemetry,
                "--ocr-backend",
                args.ocr_backend,
            ]
            if args.keep_crops:
                cmd.append("--keep-crops")
            if not gate:
                cmd.append("--no-clock-gate")
            if args.priority:
                cmd.append("--priority")
            cmd += [a for a in extra.split() if a]
            r = subprocess.run(
                cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if r.returncode != 0:
                print(r.stdout[-2000:] + r.stderr[-2000:])
                return 1
            row[tag] = lab
            print(
                "  [%d] %s %s  gpu=%s"
                % (i, tag, [ln for ln in r.stdout.splitlines() if "热轮中位" in ln], _gpu_state())
            )
        if args.cooldown:
            time.sleep(args.cooldown)  # 热降是本地最大噪声源（实测 2× 偏差）
        pairs.append(row)

    # 聚合：直接按 registry 精确 label 集合（防历史同名污染）
    def _arm_rounds(labels: set) -> dict:
        out: dict = {}
        for line in REGISTRY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("label") not in labels:
                continue
            out.setdefault(rec["config"], []).append(
                (rec.get("label"), rec.get("round", 1), rec["wall"], _gpu_valid(rec.get("gpu")))
            )
        return out

    def _arm_reports(labels: set) -> dict:
        """config → [(label, round, valid, report)]（归因逐对配对用）。"""
        out: dict = {}
        for line in REGISTRY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("label") in labels and rec.get("report"):
                out.setdefault(rec["config"], []).append(
                    (
                        rec.get("label"),
                        rec.get("round", 1),
                        _gpu_valid(rec.get("gpu")),
                        rec["report"],
                    )
                )
        return out

    wa = _arm_rounds({row["A"] for row in pairs})
    wb = _arm_rounds({row["B"] for row in pairs})
    ra_all = _arm_reports({row["A"] for row in pairs})
    rb_all = _arm_reports({row["B"] for row in pairs})
    print("本次运行 label 后缀 @%s（聚合只认这些；臂序逐对轮转 AB/BA）" % run_id)
    print(_calibration_context(args.config, args.window))
    rc = 0
    for name in sorted(set(wa) | set(wb)):
        if name not in wa or name not in wb:
            continue
        print(
            "\n=== %s（每 variant %d 次独立进程；冷=第 1 轮，热=第 2 轮起）==="
            % (name, args.repeat)
        )
        hot_diffs: list = []
        cold_diffs: list = []
        for mode in ("cold", "hot"):
            print(
                "\n-- %s --"
                % (
                    "冷启动（进程内首个 extract，未过门禁）"
                    if mode == "cold"
                    else "热池（引擎复用 ∧ 时钟门禁）"
                )
            )
            print("%-12s %10s %10s %9s  %s" % ("config", args.a, args.b, "Δ%", "逐对符号"))
            per_pair = []
            n_excluded = 0
            for row in pairs:
                la, lb = row["A"], row["B"]
                va = [
                    w
                    for l_, rn, w, v in wa.get(name, [])
                    if l_ == la and (rn == 1 if mode == "cold" else rn > 1 and v)
                ]
                vb = [
                    w
                    for l_, rn, w, v in wb.get(name, [])
                    if l_ == lb and (rn == 1 if mode == "cold" else rn > 1 and v)
                ]
                if mode == "hot":
                    n_excluded += sum(
                        1 for l_, rn, w, v in wa.get(name, []) if l_ == la and rn > 1 and not v
                    ) + sum(1 for l_, rn, w, v in wb.get(name, []) if l_ == lb and rn > 1 and not v)
                if va and vb:
                    per_pair.append((statistics.median(va), statistics.median(vb)))
            if not per_pair:
                print("%-12s  无有效配对轮" % name)
                continue
            ma = statistics.median([p[0] for p in per_pair])
            mb = statistics.median([p[1] for p in per_pair])
            d = (mb - ma) / ma * 100
            signs = ["+" if (b_ - a_) > 0 else "-" for a_, b_ in per_pair]
            anomaly = any(abs(b_ - a_) / a_ > 0.20 for a_, b_ in per_pair)
            cut = ("  [门禁剔除 %d 热轮轮次]" % n_excluded) if n_excluded else ""
            print(
                "%-12s %10.4f %10.4f %+8.2f%%  %s%s%s"
                % (
                    name,
                    ma,
                    mb,
                    d,
                    "".join(signs),
                    cut,
                    "  ⚠环境异常对（|Δ|>20%：GPU 掉频/后台占用）" if anomaly else "",
                )
            )
            if mode == "hot":
                hot_diffs = [(b_ - a_) / a_ * 100.0 for a_, b_ in per_pair]
            else:
                cold_diffs = [(b_ - a_) / a_ * 100.0 for a_, b_ in per_pair]
        # health 红旗（P0-1）：PI 判据失败与墙钟判定正交——任一臂任一
        # PI 失败即退出 1。**放在墙钟判定的早退之前**：热轮配对可能因
        # 时钟门禁全被剔除（"无有效配对轮"），但引擎真跑了、报告真产了，
        # 正确性判据没有理由跟着缺席。`--no-verdict`/`--aa` 同样不豁免
        # （红旗是正确性判据不是性能判据；标定轮也真跑了引擎）。
        rc = _ab_health_gate(args, name, ra_all, rb_all, rc)
        if not hot_diffs:
            continue
        # 热轮自动判定 / A/A 标定
        band = _band(hot_diffs)
        need = max(2, int(round(len(hot_diffs) * PI15_LIMITS["sign_majority"] + 0.5)))
        pos = band["pos"]
        neg = len(hot_diffs) - pos
        if aa:
            rec_ = _limit_from_band(band)
            print(
                "\nA/A 标定（子进程协议，%d 对）：均值 %+.3f%%  |Δ|p95 %.3f%%  "
                "sd=%.3f%%  SE=%.3f%%  符号 %d/%d"
                % (band["n"], band["mean"], band["p95abs"], band["sd"], band["se"], pos, neg)
            )
            print(
                "  → 建议 --hard 阈值 = +%.2f%%（|均值偏差|+3×SE 上取整 "
                "0.05%%，下限 %.2f%%）" % (rec_, PI15_FLOOR_PCT)
            )
            cold_rec = None
            # 冷启动带同表标定（引擎构建期 + 时钟爬坡，噪声面远大于热轮）
            if cold_diffs:
                cb = _band(cold_diffs)
                crec = _limit_from_band(cb)
                cneed = max(2, int(round(len(cold_diffs) * PI15_LIMITS["sign_majority"] + 0.5)))
                cold_rec = {"mean": cb["mean"], "se": cb["se"], "hard": crec, "n": cb["n"]}
                print(
                    "  冷启动带（%d 对）：均值 %+.3f%%  |Δ|p95 %.3f%%  "
                    "sd=%.3f%%  SE=%.3f%%  符号 %d/%d"
                    % (
                        cb["n"],
                        cb["mean"],
                        cb["p95abs"],
                        cb["sd"],
                        cb["se"],
                        cb["pos"],
                        cb["n"] - cb["pos"],
                    )
                )
                print(
                    "  → 建议 --hard-cold 阈值 = +%.2f%%（同规则；%d/%d 符号多数）"
                    % (crec, cneed, cb["n"])
                )
            # P2-2：标定持久化 + 与上一条比对（漂移 >1×SE 即提示）。
            # "上一条"只取**同 config 同窗口**（不同配置的噪声带不可比）；
            # 漂移判据 = |Δmean| > 上一条 se（本机可分辨下限口径）。
            prev = _load_latest_calibration()
            drift_note = ""
            if prev and prev.get("config") == name and prev.get("window") == args.window:
                dmean = abs(band["mean"] - prev.get("mean", 0.0))
                # 漂移阈 = max(上次 SE, 本机地板)：上次 n 小时 SE 可能为 0，
                # 用 0 当阈会把任何差异都当漂移（噪声内抖动也报警）。
                thr = max(prev.get("se", 0.0), PI15_FLOOR_PCT)
                if dmean > thr:
                    drift_note = (
                        "  ⚠ 噪声带漂移：本次均值 %+.3f%% vs 上次 %+.3f%%"
                        "（差 %.3f%% > 阈 %.3f%%）——阈值以本次为准，"
                        "建议复核在用 --hard" % (band["mean"], prev["mean"], dmean, thr)
                    )
            _save_calibration(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "fingerprint": _fp_key(_machine_fingerprint()),
                    "config": name,
                    "window": args.window,
                    "repeat": args.repeat,
                    "n": band["n"],
                    "mean": band["mean"],
                    "sd": band["sd"],
                    "se": band["se"],
                    "p95abs": band["p95abs"],
                    "suggested_hard": rec_,
                    "suggested_hard_cold": (cold_rec or {}).get("hard"),
                }
            )
            print("  标定已入账 bench/aa_calibration.jsonl（ab 启动时自动比对）")
            if drift_note:
                print(drift_note)
            # C-64：A/A 周期带直接打印——aa
            # 分支在周期判读前 continue，周期带要事后从 registry 手工算。
            # 只列主判据两键（cyc:ocr/cyc:calibrate 的 A/A 带 27%/8.6%，
            # 小相位样本少，见 _cycle_verdict 注）。A/A=两臂同码，故这里
            # 的"可分辨下限"即噪声带估计（只标定不判定）。
            if args.telemetry != "off":
                cyd = _paired_metric_diffs(ra_all.get(name, []), rb_all.get(name, []), mode="hot")
                for key in ("cyc:e2e", "cyc:decode"):
                    v = cyd.get(key)
                    if v:
                        print(
                            "  周期 A/A（%s，%d 对）：Δ均值 %+.3f%%  "
                            "SE %.3f%%  → 可分辨下限 +%.2f%%（|μ|+3SE "
                            "上取整 0.05%%；只标定不判定）"
                            % (key, v["n"], v["mean"], v["se"], _limit_from_band(v))
                        )
            continue
        if args.no_verdict:
            print(
                "\n判定关闭（--no-verdict）：均值 %+.3f%%（SE %.3f%%）符号 %d/%d"
                % (band["mean"], band["se"], pos, neg)
            )
        else:
            lim = args.hard
            if band["mean"] > lim and pos >= need:
                print(
                    "\n判定：**B 显著慢 %+.2f%%**（均值 %+.3f%%，SE %.3f%%，"
                    "限 +%.2f%%，符号 %d/%d）"
                    % (band["mean"], band["mean"], band["se"], lim, pos, neg)
                )
                rc = 1
            elif band["mean"] < -lim and neg >= need:
                print(
                    "\n判定：B 显著快 %+.2f%%（均值 %+.3f%%，SE %.3f%%，符号 "
                    "%d/%d）" % (band["mean"], band["mean"], band["se"], neg, len(hot_diffs))
                )
            else:
                print(
                    "\n判定：不可判定（落在噪声带内；均值 %+.3f%%，SE %.3f%%，"
                    "限 ±%.2f%%，符号 %d/%d）" % (band["mean"], band["se"], lim, pos, neg)
                )
            # P2-2：与标定账本比对（只提示不改判）。
            note = _sd_drift_note(name, args.window, band)
            if note:
                print(note)
            # 冷启动判定：单独阈值（冷带含引擎构建 + 时钟爬坡，幅值大得多）
            if cold_diffs and args.hard_cold > 0:
                cb = _band(cold_diffs)
                cneed = max(2, int(round(len(cold_diffs) * PI15_LIMITS["sign_majority"] + 0.5)))
                cpos = cb["pos"]
                cneg = cb["n"] - cpos
                clim = args.hard_cold
                if cb["mean"] > clim and cpos >= cneed:
                    print(
                        "冷启动判定：**B 显著慢 %+.2f%%**（均值 %+.3f%%，SE "
                        "%.3f%%，限 +%.2f%%，符号 %d/%d）"
                        % (cb["mean"], cb["mean"], cb["se"], clim, cpos, cneg)
                    )
                    rc = max(rc, 1)
                elif cb["mean"] < -clim and cneg >= cneed:
                    print(
                        "冷启动判定：B 显著快 %+.2f%%（均值 %+.3f%%，符号 %d/%d）"
                        % (cb["mean"], cb["mean"], cneg, cb["n"])
                    )
                else:
                    print(
                        "冷启动判定：不可判定（均值 %+.3f%%，SE %.3f%%，限 "
                        "±%.2f%%，符号 %d/%d）" % (cb["mean"], cb["se"], clim, cpos, cneg)
                    )
        # 逐指标归因（只归因不判失败；C-42：归因≠可回收量）
        if args.telemetry != "off":
            _attr_report(name, ra_all, rb_all, pairs, args.hard)
            # 周期账本判据（C-63）：cycles 第二意见（不改退出码）
            _cycle_verdict(name, ra_all, rb_all)
        # health 红旗已在本 config 入口处判过（早退之前，见上）
    if not aa:
        print("\n判读补充：符号一致=可信；符号混乱=落在漂移内。A=%s B=%s" % (args.a, args.b))
    return rc


def _sd_drift_note(config: str, window: int, band: dict) -> str:
    """本轮配对散布 vs 标定带（P2-2；返回告警串，无则空串）。

    正常 A/B 轮的配对差分含真实效应，不能直接当噪声带用；能比的是
    **离散度**（sd）——效应为常数时不改变 sd，故 sd 显著宽于标定带
    意味着环境噪声变了（或效应异质），两种情况下标定地板都不再适用。
    标定 sd 为 0（样本过少）时不比（避免用 0 当阈把噪声内抖动报警）。
    """
    prev = _load_latest_calibration()
    if not prev or prev.get("config") != config or prev.get("window") != window:
        return ""
    cal_sd = prev.get("sd", 0.0)
    if cal_sd <= 0:
        return ""
    thr = cal_sd + max(3 * prev.get("se", 0.0), PI15_FLOOR_PCT)
    if band["sd"] <= thr:
        return ""
    return (
        "  ⚠ 本轮配对散布 %.3f%% 显著宽于标定带 %.3f%%（阈 %.3f%%）"
        "——环境噪声或效应异质，标定地板可能不再适用（建议重跑 ab --aa 复核）"
        % (band["sd"], cal_sd, thr)
    )


def _ab_health_gate(args, name: str, ra_all: dict, rb_all: dict, rc: int) -> int:
    """ab 的 health 红旗收口（P0-1）：红则退出码 1；无报告（off 档）跳过。

    独立成函数的原因：`aa`（A/A 标定）分支会提前 continue，若红旗检查
    写在标定分支之后，`ab --aa` 整轮都不判红旗——标定轮也真跑了引擎，
    正确性判据没有豁免的理由。故本调用点放在 `aa` 分支**之前**。
    """
    if args.telemetry == "off":
        return rc
    red: list = []
    for lbl, data in ((args.a, ra_all), (args.b, rb_all)):
        red += _health_gate([rep for _l, _r, _v, rep in data.get(name, [])], lbl)
    if red:
        print("\n⚠ health 红旗（PI 判据失败，与墙钟无关也判失败）：")
        for lbl, k, v, lim, n in red:
            print("  %-24s %-6s 值=%s（限 %s）×%d" % (lbl, k, v, lim, n))
        rc = max(rc, 1)
    return rc


def _matrix_cells(knobs: dict) -> list:
    """旋钮 → 全因子格列表（[{name: level, ...}, ...]，按字典序）。

    纯函数（单测入口）：值域两水平时产 2^k 格；k 上限 3（2³=8 格；再往上
    单轮成本压不住，且"交互项"的可解释性急剧下降——三阶交互已无工程意义）。
    """
    import itertools

    names = sorted(knobs)
    grids = [knobs[n] for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*grids)]


def _matrix_effects(cells: list, walls: dict) -> dict:
    """格读数 → 主效应 + 交互项（差中差）。纯函数（单测入口）。

    `walls` = {cell_key: [每轮墙钟]}；`cell_key` 由 `_cell_key` 生成。
    主效应 = 该旋钮在**其余旋钮各水平上**的效应（逐轮配对差分，对消他因子
    水平）；交互 = 某因子效应在两个水平上的差。
    双因子 2×2 时给出唯一的交互项；三因子给出两两交互（固定第三因子取
    首水平——完整三阶交互不报，见 `_matrix_cells` 注）。
    """

    def eff(ca: dict, cb: dict) -> list:
        """逐轮配对差分 %（cb 相对 ca）。"""
        wa, wb = walls[_cell_key(ca)], walls[_cell_key(cb)]
        return [(y - x) / x * 100.0 for x, y in zip(wa, wb)]

    names = sorted(cells[0]) if cells else []
    out: dict = {"main": {}, "interaction": {}}
    import itertools

    for n in names:
        base_lvl = min(c[n] for c in cells)
        others = [m for m in names if m != n]
        other_grids = [sorted({c[m] for c in cells}) for m in others]
        per_level: dict = {}
        for combo in itertools.product(*other_grids):
            fixed = dict(zip(others, combo))
            ca = dict(fixed, **{n: base_lvl})
            for lvl in sorted({c[n] for c in cells}):
                if lvl == base_lvl:
                    continue
                cb = dict(fixed, **{n: lvl})
                per_level.setdefault(lvl, []).extend(eff(ca, cb))
        for lvl, diffs in per_level.items():
            out["main"]["%s=%s" % (n, lvl)] = _band(diffs)
    # 两两交互（2×2 子面）：固定第三因子的首水平
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            third = [m for m in names if m not in (a, b)]
            fixed3 = {m: min(c[m] for c in cells) for m in third}
            alv = sorted({c[a] for c in cells})
            blv = sorted({c[b] for c in cells})
            if len(alv) != 2 or len(blv) != 2:
                continue
            a0, a1 = alv
            b0, b1 = blv

            def cell(aa, bb):
                return dict(fixed3, **{a: aa, b: bb})

            e_a_at_b0 = eff(cell(a0, b0), cell(a1, b0))
            e_a_at_b1 = eff(cell(a0, b1), cell(a1, b1))
            inter = [y - x for x, y in zip(e_a_at_b0, e_a_at_b1)]
            out["interaction"]["%s×%s" % (a, b)] = {
                "effect_a_at_%s=%s" % (b, b0): _band(e_a_at_b0),
                "effect_a_at_%s=%s" % (b, b1): _band(e_a_at_b1),
                "diff_of_diff": _band(inter),
            }
    return out


def _cell_key(cell: dict) -> str:
    return ",".join("%s=%s" % (k, cell[k]) for k in sorted(cell))


def cmd_matrix(args) -> int:
    """多旋钮矩阵屏（P2-3）：已知交互对的全因子 + 差中差判读。

    为什么建（取证，本轮 `_scratch/p2_3_matrix.py` → bench/p2_3_matrix.json）：
    fill_width × force_aspect 2×2 实测**真交互**——fw224→0 的效应在
    fa=1.5 时 −0.32%（不可判定）、在 fa=0 时 +2.70%（5/5 显著），
    差中差 +3.02%（SE 0.96%）且反向复核 +2.97%（SE 1.30%）一致。
    一阶（逐项单独扫）在两个方向都会给错答案：结论依赖被固定的另一旋钮
    水平。故需要**同轮交错的全因子 + 交互项读数**（照 _band 判据：
    |均值|>3×SE ∧ 符号多数），而不是逐个旋钮串行扫。

    实现口径与 `ab` 一致但**同进程 + 不过时钟门禁**：
      · 同进程：多格共享热池，唯一差别是旋钮本身（子进程模式每格重付
        引擎初始化，方差远大于被测量）；格序按轮轮转对消位置效应。
      · 不过 GPU 时钟门禁：门禁判据是"sm_min ≥ 0.75×最大 SM 时钟"，
        而**改变 GPU 负载的旋钮本身就会压低对方的时钟**——实测
        fill_width=0 臂的 OCR 更省，GPU 更闲，sm_min 掉到 2070~2250
        （被门禁判无效），于是整个 A/B 变成"无有效配对轮"。门禁在此
        把**处理效应**误判为**环境干扰**（它是为"两臂码相同"的 A/B
        设计的）。矩阵屏改以同轮交错对消机器漂移，不用绝对时钟门。
    """
    knobs: dict = {}
    for spec in args.knob:
        name, _, vals = spec.partition("=")
        name = name.strip()
        if name not in MATRIX_KNOBS:
            raise SystemExit("未知旋钮 %s（可扫：%s）" % (name, ",".join(MATRIX_KNOBS)))
        typ = MATRIX_KNOBS[name]
        levels = []
        for v in vals.split(","):
            v = v.strip()
            if not v:
                continue
            levels.append(typ(v))
        if len(levels) < 2:
            raise SystemExit("旋钮 %s 需 ≥2 个水平（收到 %r）" % (name, vals))
        knobs[name] = sorted(set(levels))
    if not knobs or len(knobs) > 3:
        raise SystemExit("需 1..3 个旋钮（每个 2+ 水平）：--knob name=v1,v2 [--knob ...]")
    cells = _matrix_cells(knobs)
    cfg = args.config
    if cfg not in CONFIGS:
        raise SystemExit("未知配置 %s（可用：%s）" % (cfg, ",".join(CONFIGS)))
    print(
        "矩阵屏：config=%s  旋钮=%s  %d 格 × %d 对（同进程交错，窗口 %d）"
        % (
            cfg,
            {k: v for k, v in knobs.items()},
            len(cells),
            args.pairs,
            args.window,
        )
    )
    runners = {}
    for c in cells:
        kw = {k: c[k] for k in c}
        runners[_cell_key(c)] = _matrix_runner(cfg, args, kw)
    # 预热：每格一次（共享热池，之后每格唯一差别是旋钮）
    walls: dict = {k: [] for k in runners}
    for k, run in runners.items():
        run()
    # 交错轮转：每轮格序按轮号轮转（对消"先跑的格系统性慢"的位置效应，
    # 与 ab 的 AB/BA 及 telemetry-check 的拉丁方同源）
    keys = list(runners)
    for i in range(args.pairs):
        order = keys[i % len(keys) :] + keys[: i % len(keys)]
        for k in order:
            walls[k].append(runners[k]())
        print("  第 %d 轮：%s" % (i + 1, {k: round(walls[k][-1], 4) for k in keys}), flush=True)
    for k in keys:
        print(
            "  %-28s 中位 %.4fs  散布 %5.2f%%"
            % (
                k,
                statistics.median(walls[k]),
                (max(walls[k]) - min(walls[k])) / statistics.median(walls[k]) * 100,
            )
        )
    eff = _matrix_effects(cells, walls)
    print("\n主效应（对其余旋钮的首水平；|均值|>3×SE ∧ 符号多数 = 显著）：")
    for k in sorted(eff["main"]):
        b = eff["main"][k]
        ok = abs(b["mean"]) > max(3 * b["se"], 0.05)
        need = max(2, int(round(b["n"] * PI15_LIMITS["sign_majority"] + 0.5)))
        sig = b["pos"] >= need or (b["n"] - b["pos"]) >= need
        mark = "**" if (ok and sig) else ("~" if ok else "")
        print(
            "  %-28s Δ均值 %+7.3f%%  SE %5.3f%%  符号 %d/%-2d %s"
            % (k, b["mean"], b["se"], b["pos"], b["n"], mark)
        )
    if eff["interaction"]:
        print("\n交互项（差中差；|均值|>3×SE ∧ 符号多数 = 真交互，两旋钮不可单独判读）：")

        def _sig(b: dict) -> bool:
            need = max(2, int(round(b["n"] * PI15_LIMITS["sign_majority"] + 0.5)))
            sign_ok = b["pos"] >= need or (b["n"] - b["pos"]) >= need
            return abs(b["mean"]) > max(3 * b["se"], 0.05) and sign_ok

        for pair, d in sorted(eff["interaction"].items()):
            print("  —— %s ——" % pair)
            for k, b in sorted(d.items()):
                print(
                    "     %-28s Δ均值 %+7.3f%%  SE %5.3f%%  符号 %d/%-2d %s"
                    % (k, b["mean"], b["se"], b["pos"], b["n"], "**" if _sig(b) else "")
                )
        any_sig = any(_sig(v["diff_of_diff"]) for v in eff["interaction"].values())
        print(
            "  → 存在显著交互：这两个旋钮的最优值互相依赖，"
            "单独扫描（一阶方法）会把另一旋钮的水平当成常量，结论会错。"
            if any_sig
            else "  → 交互不显著：一阶方法在此对上够用。"
        )
    # health 红旗（与 ab 同纪律：矩阵屏也真跑了引擎）
    if args.telemetry != "off":
        red = []
        for k, run in runners.items():
            rep = getattr(run, "last_report", None)
            if rep:
                red += _health_gate([rep], k)
        if red:
            print("\n⚠ health 红旗：")
            for lbl, k, v, lim, n in red:
                print("  %-28s %-6s 值=%s（限 %s）×%d" % (lbl, k, v, lim, n))
            return 1
    return 0


def _matrix_runner(cfg_name: str, args, knob_kw: dict):
    """矩阵格 runner：返回墙钟，并把最近一次报告挂到 `run.last_report`。

    `last_report` 供 health 红旗判读（与 ab 同纪律：矩阵屏也真跑了引擎）。
    """

    def run() -> float:
        rec = _round(
            cfg_name,
            args.window,
            args.telemetry,
            args.keep_crops,
            args.ocr_backend,
            getattr(args, "rep_format", ""),
            getattr(args, "buffer_size", None),
            getattr(args, "decode_backend", ""),
            knob_kw.get("fill_width"),
            video=args.video,
            force_aspect=knob_kw.get("force_aspect"),
        )
        run.last_report = rec.get("report")  # type: ignore[attr-defined]
        return rec["wall"]

    run.last_report = None  # type: ignore[attr-defined]
    return run


def _file_summary(rounds: list) -> dict:
    """单文件多轮 → 摘要（纯函数，单测入口）。

    热轮 = round>1 ∧ 时钟门禁 valid；全剔时回退全收并置 `fell_back`
    （冷轮混入会把散布吹大，报告给读者看，不藏）。指纹一致性按**全部轮**
    判（含冷轮——确定性缺陷与冷热无关，任何两轮不同都是缺陷）。
    """
    hot = [r["wall"] for r in rounds if r["round"] > 1 and r["gpu"].get("valid")]
    fell_back = False
    if not hot:  # 门禁全剔：回退全收，但如实标记
        hot = [r["wall"] for r in rounds]
        fell_back = True
    med = statistics.median(hot)
    texts = {(r.get("shas") or {}).get("text_sha") for r in rounds}
    structs = {(r.get("shas") or {}).get("seg_struct_sha") for r in rounds}
    segs = {r["n_segments"] for r in rounds}
    return {
        "hot": hot,
        "median": med,
        "spread": ((max(hot) - min(hot)) / med * 100) if len(hot) > 1 else 0.0,
        "fell_back": fell_back,
        "text_stable": len(texts) == 1,
        "struct_stable": len(structs) == 1,
        "segs_stable": len(segs) == 1,
        "n_segments": rounds[-1]["n_segments"],
    }


def cmd_bulk(args) -> int:
    """多文件批量视图（P2-1）：**同一配置逐文件顺序跑**，输出 per-file 读数。

    为什么必须有这个视图（取证，v0.26.2 快照回放，窗口 3000 帧、cpu 后端
    三码）：跨文件墙钟散布中位 **106.6%**（h264 1.04 / av1 2.51 / hevc
    3.57s），而同文件重复轮散布中位仅 **1.02%**——差 ~100×。文件身份是
    方差主因子：把多文件聚合成一个数字，等于把"文件差异"混进"版本差异"，
    单文件回归会被文件间散布整个吃掉。故批量形态（C-53 逐文件顺序跑）
    的监测最小单位 = 文件。

    本轮读数：per-file 墙钟（热轮中位/散布）+ 段数 + 指纹一致性
    （同文件跨轮 text_sha 相同 = 确定性；不同 = 直接暴露非确定性缺陷）
    + 跨文件散布（文件身份贡献的直接读数）。入账 config 沿用原配置名，
    `video` 独立成键——`diff` 按 (label, config) 聚合，故批量与单文件的
    记录天然可混比。
    """
    names = [n.strip() for n in args.videos.split(",") if n.strip()]
    bad = [n for n in names if n not in VIDS]
    if bad:
        raise SystemExit("未知视频键 %s（可用：%s）" % (bad, ",".join(VIDS)))
    cfg_name = args.config
    if cfg_name not in CONFIGS:
        raise SystemExit("未知配置 %s（可用：%s）" % (cfg_name, ",".join(CONFIGS)))
    REGISTRY.parent.mkdir(exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    gate = not args.no_clock_gate
    if gate:
        _sm_max_clock()
    uses_gpu = (
        CONFIGS[cfg_name]["decode_backend"] in ("nvdec", "hybrid") or args.ocr_backend != "cpu"
    )
    rows: list = []
    print(
        "批量视图：config=%s  文件=%s  轮数=%d  窗=%d" % (cfg_name, names, args.rounds, args.window)
    )
    per_file: dict = {}
    for vid in names:
        rounds = []
        for i in range(args.rounds):
            with _RoundClockWatch(
                gate=gate and uses_gpu, label="off:纯CPU配置不判GPU时钟"
            ) as watch:
                rec = _round(
                    cfg_name,
                    args.window,
                    args.telemetry,
                    args.keep_crops,
                    args.ocr_backend,
                    video=vid,
                )
            rec["gpu"] = watch.block()
            rec["round"] = i + 1
            rounds.append(rec)
        summary = _file_summary(rounds)
        per_file[vid] = summary
        med = summary["median"]
        for rec in rounds:
            rec.update(
                {
                    "label": args.label,
                    "config": cfg_name,
                    "video": vid,
                    "commit": commit,
                    "telemetry": args.telemetry,
                    "window": args.window,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            rows.append(rec)
        flag = (
            "" if (per_file[vid]["text_stable"] and per_file[vid]["segs_stable"]) else "  ⚠指纹不稳"
        )
        if summary["fell_back"]:
            flag += "  ⚠门禁全剔（冷轮混入，散布不可比）"
        print(
            "  %-13s 中位 %.4fs  散布 %5.2f%%  %5d 段%s"
            % (vid, med, summary["spread"], rounds[-1]["n_segments"], flag)
        )
    with open(REGISTRY, "a", encoding="utf-8", newline="\n") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("\n已入库 %d 条 → %s" % (len(rows), REGISTRY))
    # 跨文件散布：文件身份贡献的方差（对照：同文件轮内散布）。
    # 对照只取**未回退**文件（冷轮混入的散布不可比，不参与中位）。
    meds = [v["median"] for v in per_file.values()]
    clean_spreads = [v["spread"] for v in per_file.values() if not v["fell_back"]]
    if len(meds) > 1:
        mx, mn = max(meds), min(meds)
        ref = (
            "同文件轮内中位 %.2f%%" % statistics.median(clean_spreads)
            if clean_spreads
            else "同文件轮内样本不足（门禁全剔）"
        )
        print(
            "跨文件散布 %.1f%%（%.4fs..%.4fs）——对照 %s；"
            "批量口径的方差主因子是文件身份，per-file 分辨率是回归定位的最小单位"
            % ((mx - mn) / statistics.fmean(meds) * 100, mn, mx, ref)
        )
    unstable = [v for v, d in per_file.items() if not (d["text_stable"] and d["segs_stable"])]
    if unstable:
        print("⚠ 确定性缺陷：%s 的跨轮指纹不一致（同文件同参数应逐位相同）" % ",".join(unstable))
        return 1
    return 0


def cmd_list(args) -> int:
    if not REGISTRY.exists():
        print("（registry 为空）")
        return 0
    seen: dict = {}
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            seen.setdefault(rec.get("label"), set()).add(rec["config"])
    for label, cfgs in seen.items():
        print("%-20s %s" % (label, ",".join(sorted(cfgs))))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="性能矩阵 / registry / diff")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="跑配置矩阵并入库")
    r.add_argument("--config", default="", help="逗号分隔；默认四配置")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--window", type=int, default=3000)
    r.add_argument("--telemetry", default="full", choices=("off", "full"))
    r.add_argument("--ocr-backend", default="tensorrt")
    r.add_argument("--keep-crops", action="store_true")
    r.add_argument("--rep-format", default="", help="yuv|gray（默认按引擎规则）")
    r.add_argument("--buffer-size", type=int, default=0, help="生产者队列深度")
    r.add_argument("--label", required=True)
    r.add_argument(
        "--fill-width",
        type=int,
        default=None,
        help="OCR 输入 pad 下限（px）；0 = 关闭填充（批内自适应）",
    )
    r.add_argument(
        "--force-aspect",
        type=float,
        default=None,
        help="强制宽高比（生产调用透传；真值头 test5=1.5。默认=None 用引擎默认）",
    )
    r.add_argument(
        "--decode-backend",
        default="",
        choices=("", "auto", "cpu", "nvdec", "hybrid"),
        help="覆盖 config 的解码后端（空=用 config 值）。"
        "配合 ab 的 --args-a/--args-b 即可做跨解码器交错 A/B",
    )
    r.add_argument(
        "--no-clock-gate",
        action="store_true",
        help="关闭每轮 GPU 时钟门禁（默认开：NVML 包轮，低时钟/压频轮不进热轮聚合；校准 k=0.75）",
    )
    r.add_argument(
        "--priority",
        action="store_true",
        help="本进程提到 ABOVE_NORMAL 优先级（降抢占，opt-in：会改变被测调度 regime）",
    )
    r.set_defaults(func=cmd_run)
    b = sub.add_parser(
        "bulk",
        help="多文件批量视图（同一配置逐文件顺序跑，C-53 口径；per-file 墙钟/段数/指纹）",
    )
    b.add_argument("--config", default="h264-cpu", help="单个配置名（批量视图一次一配置）")
    b.add_argument(
        "--videos",
        default=",".join(VIDS),
        help="逗号分隔视频键（默认全部：%s）" % ",".join(VIDS),
    )
    b.add_argument("--rounds", type=int, default=3, help="每文件轮数（≥3 才有 ≥2 热轮，散布可比）")
    b.add_argument("--window", type=int, default=3000)
    b.add_argument("--telemetry", default="full", choices=("off", "full"))
    b.add_argument("--ocr-backend", default="tensorrt")
    b.add_argument("--keep-crops", action="store_true")
    b.add_argument("--label", required=True)
    b.add_argument("--no-clock-gate", action="store_true")
    b.set_defaults(func=cmd_bulk)
    mx = sub.add_parser(
        "matrix",
        help="多旋钮矩阵屏（全因子 + 差中差交互判读；P2-3）",
    )
    mx.add_argument("--config", default="h264-cpu")
    mx.add_argument(
        "--knob",
        action="append",
        required=True,
        help="name=v1,v2[,...]（可重复；可扫：%s）" % ",".join(MATRIX_KNOBS),
    )
    mx.add_argument("--pairs", type=int, default=5, help="交错轮数（≥5 才有符号判据）")
    mx.add_argument("--window", type=int, default=3000)
    mx.add_argument("--telemetry", default="full", choices=("off", "full"))
    mx.add_argument("--ocr-backend", default="tensorrt")
    mx.add_argument("--keep-crops", action="store_true")
    mx.add_argument("--video", default="", help="覆盖 config 的视频键（空=config 自带）")
    mx.set_defaults(func=cmd_matrix)
    d = sub.add_parser("diff", help="两个 label 的逐指标对比（双档判定）")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--hard", type=float, default=5.0)
    d.add_argument("--warn", type=float, default=1.0)
    d.add_argument("--noise", type=float, default=1.3)
    d.set_defaults(func=cmd_diff)
    t = sub.add_parser(
        "telemetry-check", help="遥测成本互比门禁（交错+轮转+同轮配对差分；--aa 做噪声标定）"
    )
    t.add_argument("--config", default="h264-gpu")
    t.add_argument(
        "--rounds",
        type=int,
        default=25,
        help="交错轮数（默认 25 对——PI15_LIMITS 即按 n=25/inner=3 标定；改小需用 --aa 重标）",
    )
    t.add_argument(
        "--inner-rounds", type=int, default=2, help="每个子进程内轮数（>1 时取热轮，避开冷启动）"
    )
    t.add_argument("--window", type=int, default=3000)
    t.add_argument("--ocr-backend", default="tensorrt")
    t.add_argument("--cooldown", type=float, default=6.0)
    t.add_argument(
        "--gap",
        type=float,
        default=2.5,
        help="子进程之间的间隔（连续起进程会触发瞬时 CUDA 初始化失败）",
    )
    t.add_argument(
        "--aa",
        action="store_true",
        help="A/A 标定模式：同档两槽 → 差分分布即本机噪声带，打印建议阈值",
    )
    t.add_argument(
        "--subproc",
        action="store_true",
        help="用独立子进程测（含冷启成本，噪声大；默认同进程交替）",
    )
    t.add_argument(
        "--inner",
        type=int,
        default=3,
        help="同进程模式下每槽连跑次数（取均值压噪，默认 3）。"
        "成本门禁配方（实测，aa_floor_curve.json）："
        "判 full 成本 ≤1.2%% 用 --inner 1（floor ≪ 限，"
        "3× 便宜 178s vs ~500s）；µs 级 micro 对比（--full-limit "
        "0.2~0.3）仍用默认 inner=3 + n=50",
    )
    t.add_argument(
        "--full-limit",
        type=float,
        default=0.0,
        help="覆盖 full 阈值（%%），0=用 PI15_LIMITS 校准值",
    )
    t.set_defaults(func=cmd_telemetry_check)
    li = sub.add_parser("list", help="列已入库的 label")
    li.set_defaults(func=cmd_list)
    sh = sub.add_parser("show", help="打印某 label 的报告细目")
    sh.add_argument("label")
    sh.add_argument("--config", default="")
    sh.add_argument("--tier", default="")
    sh.set_defaults(func=cmd_show)
    ab = sub.add_parser("ab", help="交错 A/B（臂序轮转+时钟门禁+自动判定+health 红旗）")
    ab.add_argument("--a", required=True)
    ab.add_argument("--b", required=True)
    ab.add_argument("--env-a", default="")
    ab.add_argument("--env-b", default="")
    ab.add_argument("--config", default="h264-cpu")
    ab.add_argument("--repeat", type=int, default=3)
    ab.add_argument("--rounds", type=int, default=2)
    ab.add_argument("--window", type=int, default=3000)
    ab.add_argument("--telemetry", default="full")
    ab.add_argument("--ocr-backend", default="tensorrt")
    ab.add_argument("--keep-crops", action="store_true")
    ab.add_argument("--args-a", default="", help="A 变体附加 CLI 参数（空格分隔）")
    ab.add_argument("--args-b", default="", help="B 变体附加 CLI 参数")
    ab.add_argument(
        "--hard",
        type=float,
        default=1.0,
        help="自动判定阈值 %%（B 慢过此值且符号多数 → 退出码 1）；"
        "建议先跑 ab --aa 标定再回填（默认 1.0）",
    )
    ab.add_argument(
        "--hard-cold",
        type=float,
        default=0.0,
        help="冷启动判定阈值 %%；0=不判冷（默认）。冷带含"
        "引擎构建/时钟爬坡，幅值远大于热轮——用 `ab --aa` "
        "输出的「建议 --hard-cold」回填",
    )
    ab.add_argument(
        "--cooldown", type=float, default=8.0, help="每组之间的冷却秒数（对抗 GPU 热降）"
    )
    ab.add_argument(
        "--no-verdict",
        action="store_true",
        help="关闭墙钟自动判定（回到纯展示，恒退出码 0——但 health 红旗仍判失败）",
    )
    ab.add_argument(
        "--no-clock-gate", action="store_true", help="子进程关闭时钟门禁（透传 bench run）"
    )
    ab.add_argument(
        "--priority", action="store_true", help="子进程提到 ABOVE_NORMAL 优先级（透传 bench run）"
    )
    ab.add_argument(
        "--aa",
        action="store_true",
        help="A/A 标定：两臂同配置（忽略 env/args 差异），差分即"
        "子进程协议噪声带，打印建议 --hard 阈值",
    )
    ab.set_defaults(func=cmd_ab)
    args = ap.parse_args()
    if args.cmd in ("run", "diff", "ab", "telemetry-check", "bulk", "matrix"):
        _load_gate("bench " + args.cmd)
    return args.func(args)


def _load_gate(label: str) -> None:
    """性能测量前的后台负载门禁（R-perf-quiet）——超阈立即退出不浪费整轮。"""
    from _measure_lib import BackgroundLoadError, assert_machine_quiet

    try:
        assert_machine_quiet()
    except BackgroundLoadError as e:
        print(f"[{label}] {e}")
        raise SystemExit(2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
