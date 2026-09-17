"""bench —— 性能报告矩阵、run registry 与 `bench diff`（v2 §8.6 N-4）。

把"每次 A/B 重写一段对照脚本"变成一条命令：
  python tools/bench.py run --label s6a-before --telemetry std
  python tools/bench.py run --label s6a-after  --telemetry std
  python tools/bench.py diff s6a-before s6a-after      # D10 双档判定

报告落地 `bench/registry.jsonl`（gitignored、append-only）；A/B 的数字一律
来自报告对比，不来自"探针对探针"的手工对照（S6 门禁口径）。
`telemetry-check` 是 PI-15 的三档互比门禁：**同轮配对差分** + 符号一致性判
失败，阈值按本机 A/A 噪声带标定（`--aa` 现测；r11 用户裁决放松过严的
0.1%/1%——同码两判曾给出 −0.50% 与 +0.93%，说明旧阈值低于可分辨力）。
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

# PI-15 门禁（r11 用户裁决"门控过严导致误报，适当放松"后的**校准版**）：
# 阈值由本机 A/A 标定得出（同档两槽、同进程交替、每槽 3 次均值、25 对：
# 均值偏差 −0.179%、sd 1.053%、SE 0.211% → 0.179+3×0.211 ≈ 0.81 → 0.85）。
# 出处：knowledge/benchmarks.yaml:pi15_gate_calibration。
#   - std 与 full 用**同一条**可分辨下限（测量地板对两档一样；设计目标
#     +0.1%/+1% 仍然打印，µs 级严格性由 `tests/config/test_telemetry_cost.py`
#     的确定性成本守卫承担——那里有 10⁴ 倍于墙钟的分辨力）。
#   - 判失败还需**符号多数一致**（70%），落在噪声带里的差值不再误报。
#   - 换机器/换窗口/改配对数 n 后必须 `telemetry-check --aa` 重标（SE∝1/√n）。
PI15_LIMITS = {"std_pct": 0.30, "full_pct": 1.20, "sign_majority": 0.70}
#: 阈值下限（P1 抽成常量）：P3 重标后按新 A/A 证据更新（只收紧不放松）。
PI15_FLOOR_PCT = 0.30

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REGISTRY = ROOT / "bench" / "registry.jsonl"

VIDS = {"test5": r"D:\Videos\racelog_test\test5.mp4",
        "test6_av1": r"D:\Videos\racelog_test\test6.mp4",
        "test6_hevc": r"D:\Videos\racelog_test\test6_hevc.mp4",
        # 与 test6_av1/test6_hevc **同内容**重编码 → 编码对照必须用它
        "test6_h264": r"D:\Videos\racelog_test\test6_h264.mp4"}
ROI = {"test5": (843, 993, 948, 1025), "test6": (841, 994, 949, 1026)}
#: §12 三配置展开为 4（与 tests/golden/bench_baseline.py 同口径，可比）
CONFIGS = {
    "h264-gpu": dict(video="test5", decode_backend="nvdec"),
    "h264-cpu": dict(video="test5", decode_backend="cpu"),
    "hevc-nvdec": dict(video="test6_hevc", decode_backend="nvdec"),
    "av1-nvdec": dict(video="test6_av1", decode_backend="nvdec"),
    "h264-hybrid": dict(video="test5", decode_backend="hybrid"),
    "av1-hybrid": dict(video="test6_av1", decode_backend="hybrid"),
    # S6 续（hybrid 诊断）：hevc 的 hybrid/CPU 侧此前没进矩阵，而解码器
    # 层实测 hybrid_gpu(2311) > nvdec(2032) —— 引擎口径必须自己量。
    "hevc-hybrid": dict(video="test6_hevc", decode_backend="hybrid"),
    "hevc-cpu": dict(video="test6_hevc", decode_backend="cpu"),
    "av1-cpu": dict(video="test6_av1", decode_backend="cpu"),
    # 同内容族 h264（跨编码对照的第三支柱）；解码后端由 --decode-backend
    # 覆盖，故此处默认值只作占位。
    "h264same-nvdec": dict(video="test6_h264", decode_backend="nvdec"),
    "h264same-cpu": dict(video="test6_h264", decode_backend="cpu"),
    "h264same-hybrid": dict(video="test6_h264", decode_backend="hybrid"),
}


def _round(cfg_name: str, window: int, telemetry: str, keep_crops: bool,
           ocr_backend: str, rep_format: str = "",
           buffer_size: int | None = None,
           decode_override: str = "",
           fill_width: int | None = None) -> dict:
    from video_ocr_engine import FieldExtractor
    cfg = CONFIGS[cfg_name]
    vid = cfg["video"]
    kw = {}
    if rep_format:
        kw["rep_crop_format"] = rep_format
    if buffer_size:
        kw["buffer_size"] = buffer_size
    if fill_width is not None:
        kw["fill_width"] = fill_width     # 0 = 关闭填充（批内自适应）
    # 遥测档位经 env 进引擎（resolve 的 env 优先级链）——此前 `telemetry`
    # 参数被收下却从未使用：`bench run --telemetry full` 一直是静默无效
    # （2026-09-17 续轮 W1 排查直方图缺失时暴露）。进程内改 os.environ
    # 与 telemetry-check 的 in-process 口径一致。
    _prev_tier = os.environ.get("VOE_TELEMETRY")
    os.environ["VOE_TELEMETRY"] = telemetry
    try:
        ex = FieldExtractor(
            VIDS[vid], ROI["test5" if vid == "test5" else "test6"],
            frame_start=0, frame_end=window,
            decode_backend=(decode_override or cfg["decode_backend"]),
            ocr_backend=ocr_backend, keep_crops=keep_crops,
            sample_stride=cfg.get("sample_stride", 1), **kw)
        t0 = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t0
    finally:
        if _prev_tier is None:
            os.environ.pop("VOE_TELEMETRY", None)
        else:
            os.environ["VOE_TELEMETRY"] = _prev_tier
    return {"wall": round(wall, 4), "n_segments": len(r.segments),
            "timing": {k: round(v, 4) for k, v in r.timing.items()},
            "report": r.meta.get("report")}


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
    except Exception:  # noqa: BLE001
        return "n/a"


# ── 时钟门禁（2026-09-17 §8.6 落地，P1 协议降噪）──────────────────────
# 判据层：把 NVML sm_clock min 与主动压频位登记进每一轮读数，"时钟未起"
# 的轮次不进聚合——不可比读数应当剔除，而不是被平均掉（§8.2 实测：冷轮
# sm_min 345~450MHz，弃冷后噪声带收窄 54×）。
# k 校准（tools/_probe_clock_gate.py → bench/clock_gate.json）：冷轮
# sm_min/最大SM ≈ 0.145、热轮平台 ≈ 0.870（boost 上限 3105、负载平台
# 2700），k=0.75 居分离带中段（两侧 >10pp 余量；k=0.85 仅剩 2.3% 余量，
# k=0.9 会把热轮全拒——见探针 k 扫描）。
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
    except Exception:  # noqa: BLE001 无 NVML = 本机不可判，不是错误
        return None


def _gpu_valid(gpu: dict | None) -> bool:
    """记录级门禁判定：无 gpu 字段（历史）/ gate=off/unavailable → True。

    门禁只在有明确低时钟/压频证据时剔除（fail-open），且**不追溯改判**
    历史记录——它们写库时还没有这个字段。
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
    """

    def __init__(self, gate: bool = True, label: str = "") -> None:
        self._gate = gate
        self._label = label
        self._sm_max = None
        self._sampler = None
        self._pts: list = []
        self.unavailable = ""
        if gate:
            self._sm_max = _sm_max_clock()   # NVML 会话进程级一次初始化
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
            return {"valid": True, "sm_min": None, "sm_p50": None,
                    "throttle_bad": 0, "gate": "unavailable:采样点不足"}
        sm = sorted(p[0] for p in pts)
        bad = sum(1 for _, m in pts if (m or 0) & _THROTTLE_BAD)
        sm_min, sm_p50 = sm[0], sm[len(sm) // 2]
        valid = bad == 0 and sm_min >= CLOCK_GATE_K * (self._sm_max or 0)
        return {"sm_min": sm_min, "sm_p50": sm_p50, "sm_max": self._sm_max,
                "throttle_bad": bad, "valid": valid, "gate": "nvml"}


def _set_priority_above_normal() -> bool:
    """SetPriorityClass(ABOVE_NORMAL)（§8.5：实测可用、无需管理员）。

    伪句柄 -1 直传，绕开 GetCurrentProcess 的 restype 截断陷阱（64 位
    伪句柄若按默认 c_int 返回会被截成 32 位、调用静默失败）。
    """
    try:
        import ctypes
        k = ctypes.windll.kernel32  # type: ignore[attr-defined]
        return bool(k.SetPriorityClass(
            ctypes.c_void_p(-1), 0x00008000))  # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:  # noqa: BLE001
        return False


def cmd_run(args) -> int:
    names = (args.config.split(",") if args.config
             else ["h264-gpu", "h264-cpu", "hevc-nvdec", "av1-nvdec"])
    REGISTRY.parent.mkdir(exist_ok=True)
    rows = []
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=str(ROOT), capture_output=True, text=True,
                            encoding="utf-8").stdout.strip()
    gate = not args.no_clock_gate
    if gate:
        _sm_max_clock()            # NVML 会话一次性初始化（~20ms 不进首轮）
    if args.priority and not _set_priority_above_normal():
        print("⚠ 优先级提升失败（忽略，按普通优先级继续）")
    for name in names:
        # 门禁只对真用 GPU 的配置生效（nvdec/hybrid 解码或 OCR≠cpu）：
        # 纯 CPU 配置下 GPU 空闲时钟恒低，判了只会全 invalid 再回退。
        uses_gpu = (CONFIGS[name]["decode_backend"] in ("nvdec", "hybrid")
                    or args.ocr_backend != "cpu")
        rounds = []
        for i in range(args.rounds):
            with _RoundClockWatch(
                    gate=gate and uses_gpu,
                    label="off:纯CPU配置不判GPU时钟") as watch:
                rec = _round(name, args.window, args.telemetry, args.keep_crops,
                             args.ocr_backend, args.rep_format, args.buffer_size,
                             getattr(args, "decode_backend", ""),
                             getattr(args, "fill_width", None))
            rec["gpu"] = watch.block()
            rec["round"] = i + 1
            rounds.append(rec)
            gpu = rec["gpu"]
            note = "" if gpu.get("valid") else "  ⚠低时钟/压频（剔除出热轮）"
            print("  %-12s round %d  %.4fs  %d 段%s" % (
                name, i + 1, rec["wall"], rec["n_segments"], note))
        # 热轮 = 第 2 轮起（引擎热）∧ 时钟门禁 valid；全剔则回退全收——
        # 门禁不能替无数据。dropped 只数"本可入热轮但被门禁剔"的轮，
        # 引擎冷启动的第 1 轮是既有排除口径、不算门禁剔除。
        hot_pool = [r for r in rounds
                    if r["round"] > 1 or args.rounds == 1]
        hot = [r["wall"] for r in hot_pool if r["gpu"].get("valid")]
        dropped = len(hot_pool) - len(hot)
        if not hot:
            hot = [r["wall"] for r in rounds]
            dropped = 0
            print("  ⚠ 时钟门禁后无热轮，回退全收")
        med = statistics.median(hot)
        spread = ((max(hot) - min(hot)) / med * 100) if len(hot) > 1 else 0.0
        print("%-12s 热轮中位 %.4fs  散布 %.2f%%  段数 %d%s" % (
            name, med, spread, rounds[-1]["n_segments"],
            ("  剔除 %d 轮低时钟" % dropped) if dropped else ""))
        for rec in rounds:
            rec.update({"label": args.label, "config": name, "commit": commit,
                        "telemetry": args.telemetry, "window": args.window,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
            rows.append(rec)
    with open(REGISTRY, "a", encoding="utf-8", newline="\n") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("已入库 %d 条 → %s" % (len(rows), REGISTRY))
    return 0


def _load(label: str) -> dict:
    """按 label 聚合：config → {wall 中位, 指标快照, 有效热轮}。

    `hot` 只收「第 2 轮起 ∧ 时钟门禁 valid」的 wall（历史无 gpu 字段
    视为 valid，不追溯改判）；`n_excluded` 是被门禁剔除的热轮数。
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
        c = out.setdefault(rec["config"], {"walls": [], "reports": [],
                                           "n_segments": [], "hot": [],
                                           "n_excluded": 0})
        c["walls"].append(rec["wall"])
        c["n_segments"].append(rec["n_segments"])
        if rec.get("round", 1) > 1:
            if _gpu_valid(rec.get("gpu")):
                c["hot"].append(rec["wall"])
            else:
                c["n_excluded"] += 1
        if rec.get("report"):
            c["reports"].append(rec["report"])
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
    return flat


def _median_flat(reports: list) -> dict:
    flats = [_flatten(r) for r in reports]
    keys = set().union(*[set(f) for f in flats]) if flats else set()
    return {k: statistics.median([f[k] for f in flats if k in f])
            for k in keys}


def cmd_diff(args) -> int:
    a, b = _load(args.a), _load(args.b)
    hard, warn = args.hard, args.warn
    print("D10 双档：硬失败 >%.1f%% / 告警 >%.1f%%（噪声底 %.1f%%，§12）"
          % (hard, warn, args.noise))
    print("%-12s %10s %10s %9s  %s" % ("config", args.a, args.b, "Δ%", "判定"))
    worst = 0.0
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            print("%-12s  仅一侧有记录，跳过" % name)
            continue
        # 优先用门禁后的有效热轮；空则回退旧口径（walls[1:] / 全量）
        ha = a[name]["hot"] or a[name]["walls"][1:] or a[name]["walls"]
        hb = b[name]["hot"] or b[name]["walls"][1:] or b[name]["walls"]
        ma = statistics.median(ha)
        mb = statistics.median(hb)
        d = (mb - ma) / ma * 100
        worst = max(worst, abs(d))
        verdict = ("硬失败" if abs(d) > hard
                   else "告警" if abs(d) > warn else "通过")
        cut = ""
        if a[name]["n_excluded"] or b[name]["n_excluded"]:
            cut = "  [门禁剔除热轮 A:%d B:%d]" % (a[name]["n_excluded"],
                                                  b[name]["n_excluded"])
        print("%-12s %10.4f %10.4f %+8.2f%%  %s%s"
              % (name, ma, mb, d, verdict, cut))
    print("\n逐指标（报告口径，中位；仅列变化 >%.1f%% 的项）：" % warn)
    shown = 0
    for name in sorted(set(a) & set(b)):
        fa = _median_flat(a[name]["reports"])
        fb = _median_flat(b[name]["reports"])
        for k in sorted(set(fa) & set(fb)):
            base = fa[k]
            if base in (0, 0.0):
                continue
            d = (fb[k] - base) / base * 100
            if abs(d) > warn:
                print("  %-12s %-28s %10.4f → %10.4f  %+7.2f%%"
                      % (name, k, base, fb[k], d))
                shown += 1
    if not shown:
        print("  （无）")
    return 1 if worst > hard else 0


def _inproc_rounds(cfg: str, window: int, tiers: list, rounds: int,
                   ocr_backend: str, warmup: int = 2,
                   inner: int = 1) -> dict:
    """**同进程内交替**跑各档：返回 {(round, tier): wall}。

    为什么改成同进程：子进程模式下每个测量点都要重付"解释器 + CUDA 上下文 +
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
        order = order[k % len(order):] + order[:k % len(order)]   # 位置轮转
        for tier in order:
            os.environ["VOE_TELEMETRY"] = "std" if tier == "std2" else tier
            walls = []
            for _ in range(max(1, inner)):
                ex = FieldExtractor(
                    VIDS[vid], ROI["test5" if vid == "test5" else "test6"],
                    frame_start=0, frame_end=window,
                    decode_backend=c["decode_backend"],
                    ocr_backend=ocr_backend,
                    sample_stride=c.get("sample_stride", 1))
                t = time.perf_counter()
                ex.extract()
                walls.append(time.perf_counter() - t)
            w = statistics.fmean(walls)
            if i >= 0:
                per[(i, tier)] = w
            print("    [%s%d] %-5s %.4fs%s" % (
                "warm" if i < 0 else "run", i, tier, w,
                "" if inner == 1 else "  (%d 次均值)" % inner), flush=True)
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
        order = order[i % len(order):] + order[:i % len(order)]
        for tier in order:
            real = "std" if tier == "std2" else tier
            env = dict(os.environ)
            env["VOE_TELEMETRY"] = real
            r = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "run",
                 "--label", "pi15-%s#%d@%s" % (tier, i, run_id),
                 "--config", args.config,
                 "--rounds", str(n), "--window", str(args.window),
                 "--telemetry", real, "--ocr-backend", args.ocr_backend],
                env=env, capture_output=True, text=True, encoding="utf-8",
                errors="replace")
            if r.returncode != 0:
                print(r.stdout[-2000:] + r.stderr[-2000:])
                raise SystemExit(1)
            print("  [%d] %-5s %s" % (i, tier,
                  [ln for ln in r.stdout.splitlines() if "热轮中位" in ln]))
            time.sleep(args.gap)      # 连续起进程会让 TRT CUDA 初始化瞬态失败
    per: dict = {}
    rows = [json.loads(ln) for ln in
            REGISTRY.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for rec in rows:
        lab = rec.get("label", "")
        if not lab.endswith("@" + run_id) or rec.get("round", 1) <= 1:
            continue                  # 只统计本次 invocation 的热轮
        m = _re.match(r"pi15-(\w+)#(\d+)@", lab)
        if m and m.group(1) in tiers:
            per[(int(m.group(2)), m.group(1))] = rec["wall"]
    return per


def cmd_telemetry_check(args) -> int:
    """PI-15：三档互比（**同进程交替 + 同轮配对差分** + 符号一致性判失败）。

    方法学演进（每一步都是被数据逼出来的，全部留档）：
    1. 原实现"三档各自连跑 N 轮"把机器漂移记进档位差——同码两次可差 7.7%，
       而阈值只有 0.1% → 改为每档独立子进程 + 逐轮交错（S6）。
    2. 固定 off→std→full 顺序让轮内热漂移系统性落在最后一档：full 稳定
       +3.3%，而 NVML/差分本身只值 µs 级 → 每轮轮转档位顺序（拉丁方）。
    3. 交错后仍不稳：同一份代码两次判读给出 std −0.50% 与 std +0.93%。根因是
       **用三档各自中位数相比**，公共漂移没被减掉 → 改为**同轮内配对差分**。
    4. 子进程模式的 A/A（同档两槽）噪声带实测 |Δ|p50 0.24% / p95 2.33%，远大于
       被测量 → 改为**同进程交替**（共享热池与 OS 缓存，唯一差别是插桩本身）。
    5. 阈值按 A/A 现测校准（`--aa`），并要求**符号多数一致**才判失败——
       落在噪声带里的差值判"不可判定"，不再误报（r11 用户裁决）。
    """
    tiers = ["off", "std", "full"]
    if args.aa:
        tiers = ["std", "std2"]        # A/A：两槽同档 → 差分即纯噪声
    if args.subproc:
        per = _subproc_rounds(args, tiers)
    else:
        per = _inproc_rounds(args.config, args.window, tiers, args.rounds,
                             args.ocr_backend, inner=args.inner)
    med = {t: statistics.median([v for (i, tt), v in per.items()
                                 if tt == t and v is not None])
           for t in tiers}
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
    return {"n": n, "mean": round(mean, 3), "median": round(med, 3),
            "p50abs": round(q(av, 0.5), 3), "p95abs": round(q(av, 0.95), 3),
            "maxabs": round(av[-1], 3) if av else 0.0,
            "sd": round(sd, 3), "se": round(se, 3), "sigma_mad": round(mad, 3),
            "pos": sum(1 for x in v if x > 0)}


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
    d = _paired(per, args.rounds, "std", "std2")
    if not d:
        print("A/A 无有效配对轮次")
        return 1
    b = _band(d)
    print("\nA/A 噪声标定（同档两槽，%d 对，%s，config=%s window=%d）："
          % (b["n"], "子进程" if args.subproc else "同进程交替",
             args.config, args.window))
    print("  均值 %+.3f%%  中位 %+.3f%%  |Δ|p50 %.3f%%  |Δ|p95 %.3f%%  "
          "max|Δ| %.3f%%  sd=%.3f%% SE=%.3f%%"
          % (b["mean"], b["median"], b["p50abs"], b["p95abs"], b["maxabs"],
             b["sd"], b["se"]))
    rec = _limit_from_band(b)
    print("  → 本机在 %d 对下可分辨 std 阈值 = +%.2f%%、full = +%.2f%%"
          "（规则 |均值偏差|+3×SE 上取整 0.05%%，下限 0.30%%；"
          "判失败另需符号多数一致）" % (b["n"], rec, 4 * rec))
    print("  提示：阈值随配对数收紧（SE∝1/√n），标定 n 必须与门禁 n 一致；"
          "换机器/换窗口后必须重标。")
    return 0


def _report_pi15(per: dict, med: dict, args) -> int:
    base = med["off"]
    d_std = _paired(per, args.rounds, "off", "std")
    d_full = _paired(per, args.rounds, "off", "full")
    lim_std = args.std_limit or PI15_LIMITS["std_pct"]
    lim_full = args.full_limit or PI15_LIMITS["full_pct"]
    need = max(2, int(round(len(d_std) * PI15_LIMITS["sign_majority"] + 0.5))) \
        if d_std else 2

    def judge(d, lim):
        if not d:
            return False, "无配对样本"
        b = _band(d)
        m = b["mean"]                      # 统计量=均值（位置效应对消，见 _band）
        signs = b["pos"] >= need or (len(d) - b["pos"]) >= need
        ok = (m <= lim) or not signs
        note = "" if ok or m <= lim else "（超阈且符号一致）"
        return ok, ("均值 %+.3f%%（SE %.3f%%，限 +%.2f%%）符号 %d/%d%s%s → %s"
                    % (m, b["se"], lim, b["pos"], len(d),
                       "一致" if signs else "不一致=噪声内", note,
                       "通过" if ok else "失败"))
    ok_s, txt_s = judge(d_std, lim_std)
    ok_f, txt_f = judge(d_full, lim_full)
    print("\nPI-15（%s，同轮**配对差分** %d 对；off 中位 %.4fs / std %.4fs /"
          " full %.4fs）" % ("子进程" if args.subproc else "同进程交替",
                             len(d_std), base, med["std"], med["full"]))
    print("  std  vs off ：%s" % txt_s)
    print("  full vs off ：%s" % txt_f)
    print("  设计目标（§13.2）std ≤ +0.1%%、full ≤ +1%%；本机可分辨下限见 "
          "`bench.py telemetry-check --aa`（阈值=校准值 %s/%s，r11 用户裁决"
          "放松过严阈值以消除误报）" % (lim_std, lim_full))
    ok = bool(ok_s) and bool(ok_f)
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
        print("\n=== %s / %s  wall=%s（中位 %.4fs，%d 段）" % (
            args.label, name, [round(w, 3) for w in walls],
            statistics.median(walls), data[name]["n_segments"][-1]))
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
            print("  span   %-24s n=%-5d sum=%8.4f p50=%8.5f max=%8.5f"
                  % (k, v["n"], v["sum"], v["p50"], v["max"]))
        print("  ---- span 合计 %.4f s（wall %.4f s，占比 %.1f%%）"
              % (tot, rep["wall_s"], tot / max(rep["wall_s"], 1e-9) * 100))
        for k in sorted(rep.get("gauges", {})):
            print("  gauge  %-24s %12.5f" % (k, rep["gauges"][k]))
        for k in sorted(rep.get("counters", {})):
            print("  cnt    %-24s %12d" % (k, rep["counters"][k]))
        if rep.get("health"):
            print("  health %s" % json.dumps(rep["health"], ensure_ascii=False))
    return 0


def cmd_ab(args) -> int:
    """交错 A/B（S6 口径 → P1 加固：臂序轮转 + 时钟门禁 + 自动判定）。

    背景（实测 2026-09-10）：**同一份代码**连跑两次 `bench run`，h264-cpu
    1.2567 → 1.1599s（−7.7%）、hevc −7.7%——功耗/热状态与后台进程造成的
    漂移**大于**被测量本身，顺序跑会把漂移记到 B 头上 → 必须交错。每个
    variant 起独立子进程（状态干净、各含一轮预热）。P1 再加三层降噪：
      1. **臂序逐对轮转**（AB/BA 交替）——对消轮内位置效应（先跑的臂
         系统性慢 ~1.3%，telemetry-check 同款结论）；
      2. **每轮 GPU 时钟入账**（子进程 `bench run` 内做，registry `gpu`
         字段），低时钟/压频热轮不进聚合（P0b：冷轮 CV 6.6% → 热轮 0.12%）；
      3. **自动判定**：逐对热轮中位配对差分 → 均值 + SE + 符号多数（≥70%）
         vs `--hard` 阈值；B 显著慢 → 退出码 1（`--no-verdict` 回纯展示）。
    `--aa`：两臂同配置（忽略 env/args 差异）→ 差分即子进程协议噪声带，
    打印建议阈值（规则 |均值偏差|+3×SE，下限 PI15_FLOOR_PCT）。

    用法：
      python tools/bench.py ab --config h264-cpu --repeat 3 \
          --a base --b s6c --env-b VOE_S6C_ASYNC=1
    """
    import os
    run_id = time.strftime("%m%d-%H%M%S")
    aa = bool(args.aa)
    gate = not args.no_clock_gate
    pairs: list = []
    for i in range(args.repeat):
        row = {}
        arm_specs = (("A", args.a, args.env_a, args.args_a),
                     ("B", args.b, args.env_b, args.args_b))
        if i % 2 == 1:
            arm_specs = arm_specs[::-1]     # 臂序轮转：对消位置效应（P1）
        for tag, label, envspec, extra in arm_specs:
            if aa:
                envspec, extra = "", ""     # A/A：两臂完全同配置
            lab = "%s#%d@%s" % (label, i, run_id)
            env = dict(os.environ)
            for kv in filter(None, envspec.split(",")):
                k, _, v = kv.partition("=")
                env[k.strip()] = v.strip()
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "run", "--label", lab, "--config", args.config,
                   "--rounds", str(args.rounds), "--window", str(args.window),
                   "--telemetry", args.telemetry,
                   "--ocr-backend", args.ocr_backend]
            if args.keep_crops:
                cmd.append("--keep-crops")
            if not gate:
                cmd.append("--no-clock-gate")
            if args.priority:
                cmd.append("--priority")
            cmd += [a for a in extra.split() if a]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print(r.stdout[-2000:] + r.stderr[-2000:])
                return 1
            row[tag] = lab
            print("  [%d] %s %s  gpu=%s" % (
                i, tag, [ln for ln in r.stdout.splitlines()
                         if "热轮中位" in ln], _gpu_state()))
        if args.cooldown:
            time.sleep(args.cooldown)     # 热降是本地最大噪声源（实测 2× 偏差）
        pairs.append(row)

    # 聚合：直接按 registry 精确 label 集合（防历史同名污染，2026-09-13 实测）
    def _arm_rounds(labels: set) -> dict:
        out: dict = {}
        for line in REGISTRY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("label") not in labels:
                continue
            out.setdefault(rec["config"], []).append(
                (rec.get("label"), rec.get("round", 1), rec["wall"],
                 _gpu_valid(rec.get("gpu"))))
        return out

    def _arm_reports(labels: set) -> dict:
        out: dict = {}
        for line in REGISTRY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("label") in labels and rec.get("report"):
                out.setdefault(rec["config"], []).append(rec["report"])
        return out

    wa = _arm_rounds({row["A"] for row in pairs})
    wb = _arm_rounds({row["B"] for row in pairs})
    ra_all = _arm_reports({row["A"] for row in pairs})
    rb_all = _arm_reports({row["B"] for row in pairs})
    print("本次运行 label 后缀 @%s（聚合只认这些；臂序逐对轮转 AB/BA）" % run_id)
    rc = 0
    for name in sorted(set(wa) | set(wb)):
        if name not in wa or name not in wb:
            continue
        print("\n=== %s（每 variant %d 次独立进程；冷=第 1 轮，热=第 2 轮起）==="
              % (name, args.repeat))
        hot_diffs: list = []
        for mode in ("cold", "hot"):
            print("\n-- %s --" % ("冷启动（进程内首个 extract，未过门禁）"
                                  if mode == "cold"
                                  else "热池（引擎复用 ∧ 时钟门禁）"))
            print("%-12s %10s %10s %9s  %s" % ("config", args.a, args.b, "Δ%",
                                               "逐对符号"))
            per_pair = []
            n_excluded = 0
            for row in pairs:
                la, lb = row["A"], row["B"]
                va = [w for l_, rn, w, v in wa.get(name, [])
                      if l_ == la and (rn == 1 if mode == "cold" else
                                       rn > 1 and v)]
                vb = [w for l_, rn, w, v in wb.get(name, [])
                      if l_ == lb and (rn == 1 if mode == "cold" else
                                       rn > 1 and v)]
                if mode == "hot":
                    n_excluded += (sum(1 for l_, rn, w, v in wa.get(name, [])
                                       if l_ == la and rn > 1 and not v)
                                   + sum(1 for l_, rn, w, v in wb.get(name, [])
                                         if l_ == lb and rn > 1 and not v))
                if va and vb:
                    per_pair.append((statistics.median(va),
                                     statistics.median(vb)))
            if not per_pair:
                print("%-12s  无有效配对轮" % name)
                continue
            ma = statistics.median([p[0] for p in per_pair])
            mb = statistics.median([p[1] for p in per_pair])
            d = (mb - ma) / ma * 100
            signs = ["+" if (b_ - a_) > 0 else "-" for a_, b_ in per_pair]
            anomaly = any(abs(b_ - a_) / a_ > 0.20 for a_, b_ in per_pair)
            cut = ("  [门禁剔除 %d 热轮轮次]" % n_excluded) if n_excluded else ""
            print("%-12s %10.4f %10.4f %+8.2f%%  %s%s%s"
                  % (name, ma, mb, d, "".join(signs), cut,
                     "  ⚠环境异常对（|Δ|>20%：GPU 掉频/后台占用）"
                     if anomaly else ""))
            if mode == "hot":
                hot_diffs = [(b_ - a_) / a_ * 100.0 for a_, b_ in per_pair]
        if not hot_diffs:
            continue
        # 热轮自动判定 / A/A 标定
        band = _band(hot_diffs)
        need = max(2, int(round(len(hot_diffs) * PI15_LIMITS["sign_majority"]
                               + 0.5)))
        pos = band["pos"]
        neg = len(hot_diffs) - pos
        if aa:
            rec_ = _limit_from_band(band)
            print("\nA/A 标定（子进程协议，%d 对）：均值 %+.3f%%  |Δ|p95 %.3f%%  "
                  "sd=%.3f%%  SE=%.3f%%  符号 %d/%d"
                  % (band["n"], band["mean"], band["p95abs"], band["sd"],
                     band["se"], pos, neg))
            print("  → 建议 --hard 阈值 = +%.2f%%（|均值偏差|+3×SE 上取整 "
                  "0.05%%，下限 %.2f%%）" % (rec_, PI15_FLOOR_PCT))
            continue
        if args.no_verdict:
            print("\n判定关闭（--no-verdict）：均值 %+.3f%%（SE %.3f%%）符号 %d/%d"
                  % (band["mean"], band["se"], pos, neg))
        else:
            lim = args.hard
            if band["mean"] > lim and pos >= need:
                print("\n判定：**B 显著慢 %+.2f%%**（均值 %+.3f%%，SE %.3f%%，"
                      "限 +%.2f%%，符号 %d/%d）"
                      % (band["mean"], band["mean"], band["se"], lim, pos, neg))
                rc = 1
            elif band["mean"] < -lim and neg >= need:
                print("\n判定：B 显著快 %+.2f%%（均值 %+.3f%%，SE %.3f%%，符号 "
                      "%d/%d）" % (band["mean"], band["mean"], band["se"], neg))
            else:
                print("\n判定：不可判定（落在噪声带内；均值 %+.3f%%，SE %.3f%%，"
                      "限 ±%.2f%%，符号 %d/%d）"
                      % (band["mean"], band["se"], lim, pos, neg))
        # 逐 span 归因（只展示不判；C-42：占比/差分≠可回收量，绑定实验才是口径）
        if args.telemetry != "off":
            fa = _median_flat(ra_all.get(name, []))
            fb = _median_flat(rb_all.get(name, []))
            moved = [(k, fa[k], fb[k]) for k in sorted(set(fa) & set(fb))
                     if fa[k] not in (0, 0.0)
                     and abs((fb[k] - fa[k]) / fa[k] * 100) > 1.0]
            if moved:
                print("逐 span 归因（中位，|Δ|>1%；⚠ C-42 归因≠可回收量）：")
                for k, va_, vb_ in moved[:20]:
                    print("  %-40s %12.4f → %12.4f  %+7.2f%%"
                          % (k, va_, vb_, (vb_ - va_) / va_ * 100))
    if not aa:
        print("\n判读补充：符号一致=可信；符号混乱=落在漂移内。A=%s B=%s"
              % (args.a, args.b))
    return rc


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
    r.add_argument("--telemetry", default="std", choices=("off", "std", "full"))
    r.add_argument("--ocr-backend", default="tensorrt")
    r.add_argument("--keep-crops", action="store_true")
    r.add_argument("--rep-format", default="", help="yuv|gray（默认按引擎规则）")
    r.add_argument("--buffer-size", type=int, default=0, help="生产者队列深度")
    r.add_argument("--label", required=True)
    r.add_argument("--fill-width", type=int, default=None,
                   help="OCR 输入 pad 下限（px）；0 = 关闭填充（批内自适应）")
    r.add_argument("--decode-backend", default="",
                   choices=("", "auto", "cpu", "nvdec", "hybrid"),
                   help="覆盖 config 的解码后端（空=用 config 值）。"
                        "配合 ab 的 --args-a/--args-b 即可做跨解码器交错 A/B")
    r.add_argument("--no-clock-gate", action="store_true",
                   help="关闭每轮 GPU 时钟门禁（默认开：NVML 包轮，低时钟/"
                        "压频轮不进热轮聚合；P0b 校准 k=0.75）")
    r.add_argument("--priority", action="store_true",
                   help="本进程提到 ABOVE_NORMAL 优先级（§8.5 降抢占，"
                        "opt-in：会改变被测调度 regime）")
    r.set_defaults(func=cmd_run)
    d = sub.add_parser("diff", help="两个 label 的逐指标对比（D10 双档）")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--hard", type=float, default=5.0)
    d.add_argument("--warn", type=float, default=1.0)
    d.add_argument("--noise", type=float, default=1.3)
    d.set_defaults(func=cmd_diff)
    t = sub.add_parser("telemetry-check",
                       help="PI-15 三档互比门禁（交错+轮转+同轮配对差分；--aa 做噪声标定）")
    t.add_argument("--config", default="h264-gpu")
    t.add_argument("--rounds", type=int, default=25,
                   help="交错轮数（默认 25 对——PI15_LIMITS 即按 n=25/inner=3 "
                        "标定；改小需用 --aa 重标）")
    t.add_argument("--inner-rounds", type=int, default=2,
                   help="每个子进程内轮数（>1 时取热轮，避开冷启动）")
    t.add_argument("--window", type=int, default=3000)
    t.add_argument("--ocr-backend", default="tensorrt")
    t.add_argument("--cooldown", type=float, default=6.0)
    t.add_argument("--gap", type=float, default=2.5,
                   help="子进程之间的间隔（连续起进程会触发瞬时 CUDA 初始化失败）")
    t.add_argument("--aa", action="store_true",
                   help="A/A 标定模式：同档两槽 → 差分分布即本机噪声带，打印建议阈值")
    t.add_argument("--subproc", action="store_true",
                   help="用独立子进程测（含冷启成本，噪声大；默认同进程交替）")
    t.add_argument("--inner", type=int, default=3,
                   help="同进程模式下每槽连跑次数（取均值压噪，默认 3）")
    t.add_argument("--std-limit", type=float, default=0.0,
                   help="覆盖 std 阈值（%%），0=用 PI15_LIMITS 校准值")
    t.add_argument("--full-limit", type=float, default=0.0,
                   help="覆盖 full 阈值（%%），0=用 PI15_LIMITS 校准值")
    t.set_defaults(func=cmd_telemetry_check)
    li = sub.add_parser("list", help="列已入库的 label")
    li.set_defaults(func=cmd_list)
    sh = sub.add_parser("show", help="打印某 label 的报告细目")
    sh.add_argument("label")
    sh.add_argument("--config", default="")
    sh.add_argument("--tier", default="")
    sh.set_defaults(func=cmd_show)
    ab = sub.add_parser("ab", help="交错 A/B（臂序轮转+时钟门禁+自动判定）")
    ab.add_argument("--a", required=True)
    ab.add_argument("--b", required=True)
    ab.add_argument("--env-a", default="")
    ab.add_argument("--env-b", default="")
    ab.add_argument("--config", default="h264-cpu")
    ab.add_argument("--repeat", type=int, default=3)
    ab.add_argument("--rounds", type=int, default=2)
    ab.add_argument("--window", type=int, default=3000)
    ab.add_argument("--telemetry", default="std")
    ab.add_argument("--ocr-backend", default="tensorrt")
    ab.add_argument("--keep-crops", action="store_true")
    ab.add_argument("--args-a", default="", help="A 变体附加 CLI 参数（空格分隔）")
    ab.add_argument("--args-b", default="", help="B 变体附加 CLI 参数")
    ab.add_argument("--hard", type=float, default=1.0,
                    help="自动判定阈值 %%（B 慢过此值且符号多数 → 退出码 1）；"
                         "建议先跑 ab --aa 标定再回填（默认 1.0）")
    ab.add_argument("--cooldown", type=float, default=8.0,
                    help="每组之间的冷却秒数（对抗 GPU 热降）")
    ab.add_argument("--no-verdict", action="store_true",
                    help="关闭自动判定（回到纯展示，恒退出码 0）")
    ab.add_argument("--no-clock-gate", action="store_true",
                    help="子进程关闭时钟门禁（透传 bench run）")
    ab.add_argument("--priority", action="store_true",
                    help="子进程提到 ABOVE_NORMAL 优先级（透传 bench run）")
    ab.add_argument("--aa", action="store_true",
                    help="A/A 标定：两臂同配置（忽略 env/args 差异），差分即"
                         "子进程协议噪声带，打印建议 --hard 阈值")
    ab.set_defaults(func=cmd_ab)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
