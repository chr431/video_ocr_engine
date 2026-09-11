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
PI15_LIMITS = {"std_pct": 0.85, "full_pct": 0.85, "sign_majority": 0.70}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REGISTRY = ROOT / "bench" / "registry.jsonl"

VIDS = {"test5": r"D:\Videos\racelog_test\test5.mp4",
        "test6_av1": r"D:\Videos\racelog_test\test6.mp4",
        "test6_hevc": r"D:\Videos\racelog_test\test6_hevc.mp4"}
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
}


def _round(cfg_name: str, window: int, telemetry: str, keep_crops: bool,
           ocr_backend: str, rep_format: str = "",
           buffer_size: int | None = None) -> dict:
    from video_ocr_engine import FieldExtractor
    cfg = CONFIGS[cfg_name]
    vid = cfg["video"]
    kw = {}
    if rep_format:
        kw["rep_crop_format"] = rep_format
    if buffer_size:
        kw["buffer_size"] = buffer_size
    ex = FieldExtractor(VIDS[vid], ROI["test5" if vid == "test5" else "test6"],
                        frame_start=0, frame_end=window,
                        decode_backend=cfg["decode_backend"],
                        ocr_backend=ocr_backend, keep_crops=keep_crops,
                        sample_stride=cfg.get("sample_stride", 1), **kw)
    t0 = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t0
    return {"wall": round(wall, 4), "n_segments": len(r.segments),
            "timing": {k: round(v, 4) for k, v in r.timing.items()},
            "report": r.meta.get("report")}


def _gpu_state() -> str:
    """GPU 状态串（A/B 记录用）：掉频/热降是本地最大噪声源。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10).stdout.strip()
        return out
    except Exception:  # noqa: BLE001
        return "n/a"


def cmd_run(args) -> int:
    names = (args.config.split(",") if args.config
             else ["h264-gpu", "h264-cpu", "hevc-nvdec", "av1-nvdec"])
    REGISTRY.parent.mkdir(exist_ok=True)
    rows = []
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=str(ROOT), capture_output=True, text=True,
                            encoding="utf-8").stdout.strip()
    for name in names:
        rounds = []
        for i in range(args.rounds):
            rec = _round(name, args.window, args.telemetry, args.keep_crops,
                         args.ocr_backend, args.rep_format, args.buffer_size)
            rec["round"] = i + 1
            rounds.append(rec)
            print("  %-12s round %d  %.4fs  %d 段" % (
                name, i + 1, rec["wall"], rec["n_segments"]))
        walls = [r["wall"] for r in rounds]
        hot = walls[1:] if len(walls) > 1 else walls
        med = statistics.median(hot)
        spread = ((max(hot) - min(hot)) / med * 100) if len(hot) > 1 else 0.0
        print("%-12s 热轮中位 %.4fs  散布 %.2f%%  段数 %d" % (
            name, med, spread, rounds[-1]["n_segments"]))
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
    """按 label 聚合：config → {wall 中位, 指标快照}。"""
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
                                           "n_segments": []})
        c["walls"].append(rec["wall"])
        c["n_segments"].append(rec["n_segments"])
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
        ma = statistics.median(a[name]["walls"][1:] or a[name]["walls"])
        mb = statistics.median(b[name]["walls"][1:] or b[name]["walls"])
        d = (mb - ma) / ma * 100
        worst = max(worst, abs(d))
        verdict = ("硬失败" if abs(d) > hard
                   else "告警" if abs(d) > warn else "通过")
        print("%-12s %10.4f %10.4f %+8.2f%%  %s" % (name, ma, mb, d, verdict))
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
    return max(0.30, -(-cand // 0.05) * 0.05)


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
    """交错 A/B（S6 口径修正）：同一窗口内 A/B/A/B 交替测量。

    背景（实测 2026-09-10）：**同一份代码**连跑两次 `bench run`，h264-cpu
    1.2567 → 1.1599s（−7.7%）、hevc −7.7%——笔记本功耗/热状态与后台进程
    造成的漂移**大于**被测量本身。顺序 A 全跑再 B 全跑会把漂移记到 B 头上，
    因此 S6 各项一律用本命令：每个 variant 起独立子进程（状态干净、各自
    含一轮预热），交错重复后在窗口内取中位。

    用法：
      python tools/bench.py ab --config h264-cpu --repeat 3 \
          --a base --b s6c --env-b VOE_S6C_ASYNC=1
    """
    import os
    pairs: list = []
    for i in range(args.repeat):
        row = {}
        for tag, label, envspec, extra in (("A", args.a, args.env_a, args.args_a),
                                           ("B", args.b, args.env_b, args.args_b)):
            lab = "%s#%d" % (label, i)
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
    data = None  # 直接按 registry 聚合，不做二次缓存
    # 聚合：每个 variant 在窗口内的全部轮次（冷=第 1 轮 / 热=其余）
    def _walls(label):
        out: dict = {}
        for line in REGISTRY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            lab = rec.get("label", "")
            if "".join(lab.split("#")[:-1]) != label:
                continue
            out.setdefault(rec["config"], []).append(
                (lab, rec.get("round", 1), rec["wall"]))
        return out

    wa, wb = _walls(args.a), _walls(args.b)
    print("\n交错 A/B（每 variant %d 次独立进程；冷=第 1 轮，热=第 2 轮起）"
          % args.repeat)
    for mode in ("cold", "hot"):
        print("\n-- %s --" % ("冷启动（进程内首个 extract）" if mode == "cold"
                              else "热池（引擎复用）"))
        print("%-12s %10s %10s %9s  %s" % ("config", args.a, args.b, "Δ%",
                                           "逐对符号"))
        for name in sorted(set(wa) | set(wb)):
            per_pair = []
            for row in pairs:
                la, lb = row["A"], row["B"]
                sel = (lambda r: r[1] == 1) if mode == "cold" else (
                    lambda r: r[1] > 1)
                va = [w for lab, rn, w in wa.get(name, [])
                      if lab == la and sel((lab, rn, w))]
                vb = [w for lab, rn, w in wb.get(name, [])
                      if lab == lb and sel((lab, rn, w))]
                if va and vb:
                    per_pair.append((statistics.median(va),
                                     statistics.median(vb)))
            if not per_pair:
                continue
            ma = statistics.median([p[0] for p in per_pair])
            mb = statistics.median([p[1] for p in per_pair])
            d = (mb - ma) / ma * 100
            signs = ["+" if (b_ - a_) > 0 else "-" for a_, b_ in per_pair]
            anomaly = any(abs(b_ - a_) / a_ > 0.20 for a_, b_ in per_pair)
            print("%-12s %10.4f %10.4f %+8.2f%%  %s%s"
                  % (name, ma, mb, d, "".join(signs),
                     "  ⚠环境异常对（|Δ|>20%：GPU 掉频/后台占用）"
                     if anomaly else ""))
    print("\n判读：符号一致=可信；符号混乱=落在漂移内。A=%s B=%s"
          % (args.a, args.b))
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
    r.add_argument("--telemetry", default="std", choices=("off", "std", "full"))
    r.add_argument("--ocr-backend", default="tensorrt")
    r.add_argument("--keep-crops", action="store_true")
    r.add_argument("--rep-format", default="", help="yuv|gray（默认按引擎规则）")
    r.add_argument("--buffer-size", type=int, default=0, help="生产者队列深度")
    r.add_argument("--label", required=True)
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
    ab = sub.add_parser("ab", help="交错 A/B（抵消机器漂移）")
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
    ab.add_argument("--hard", type=float, default=5.0)
    ab.add_argument("--cooldown", type=float, default=8.0,
                    help="每组之间的冷却秒数（对抗 GPU 热降）")
    ab.set_defaults(func=cmd_ab)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
