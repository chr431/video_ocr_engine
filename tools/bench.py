"""bench —— 性能报告矩阵、run registry 与 `bench diff`（v2 §8.6 N-4）。

把"每次 A/B 重写一段对照脚本"变成一条命令：
  python tools/bench.py run --label s6a-before --telemetry std
  python tools/bench.py run --label s6a-after  --telemetry std
  python tools/bench.py diff s6a-before s6a-after      # D10 双档判定

报告落地 `bench/registry.jsonl`（gitignored、append-only）；A/B 的数字一律
来自报告对比，不来自"探针对探针"的手工对照（S6 门禁口径）。
`telemetry-check` 是 PI-15 的三档互比门禁（std vs off ≤ +0.1%、full vs off ≤ +1%）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

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


def cmd_telemetry_check(args) -> int:
    """PI-15：三档互比（std vs off ≤ +0.1%、full vs off ≤ +1%）。

    **必须交错**（S6 修正）：原实现"三档各自连跑 N 轮"把机器漂移记进了档位
    差——同码两次实测可差 7.7%，而阈值只有 0.1%。现为每档独立子进程 +
    off/std/full 逐轮交错 + 取各自热轮中位（与 `bench ab` 同一方法学）。
    """
    import os
    run_id = time.strftime("%m%d-%H%M%S")
    per_tier: dict = {t: [] for t in ("off", "std", "full")}
    for i in range(args.rounds):
        for tier in ("off", "std", "full"):
            env = dict(os.environ)
            env["VOE_TELEMETRY"] = tier
            r = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "run",
                 "--label", "pi15-%s#%d@%s" % (tier, i, run_id),
                 "--config", args.config,
                 "--rounds", str(args.inner_rounds), "--window", str(args.window),
                 "--telemetry", tier, "--ocr-backend", args.ocr_backend],
                env=env, capture_output=True, text=True, encoding="utf-8",
                errors="replace")
            if r.returncode != 0:
                print(r.stdout[-2000:] + r.stderr[-2000:])
                return 1
            print("  [%d] %-5s %s" % (i, tier, [ln for ln in r.stdout.splitlines()
                                                if "热轮中位" in ln]))
            time.sleep(args.gap)      # 进程间留缝：连续起进程会让 TRT 的 CUDA
            #                           初始化瞬态失败（cudaError 35，实测）
    # 取各自的热轮（子进程内第 2 轮起）
    rows = [json.loads(ln) for ln in
            REGISTRY.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for rec in rows:
        lab = rec.get("label", "")
        if not lab.endswith("@" + run_id):
            continue                 # 只统计本次 invocation（防混入历史窗口）
        for tier in per_tier:
            if lab.startswith("pi15-%s#" % tier) and rec.get("round", 1) > 1:
                per_tier[tier].append(rec["wall"])
    med = {t: statistics.median(v) for t, v in per_tier.items() if v}
    if len(med) < 3:
        print("样本不足：%s" % med)
        return 1
    base = med["off"]
    d_std = (med["std"] - base) / base * 100
    d_full = (med["full"] - base) / base * 100
    print("\nPI-15（交错 %d 轮 × 每档热轮 %d 次）：off %.4fs / std %.4fs "
          "(%+.3f%%，限 +0.1%%) / full %.4fs (%+.3f%%，限 +1%%)"
          % (args.rounds, args.inner_rounds, base, med["std"], d_std,
             med["full"], d_full))
    ok = d_std <= 0.1 and d_full <= 1.0
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
    t = sub.add_parser("telemetry-check", help="PI-15 三档互比门禁（交错）")
    t.add_argument("--config", default="h264-gpu")
    t.add_argument("--rounds", type=int, default=3, help="交错轮数（每轮三档各一次）")
    t.add_argument("--inner-rounds", type=int, default=2,
                   help="每个子进程内轮数（>1 时取热轮，避开冷启动）")
    t.add_argument("--window", type=int, default=3000)
    t.add_argument("--ocr-backend", default="tensorrt")
    t.add_argument("--cooldown", type=float, default=6.0)
    t.add_argument("--gap", type=float, default=2.5,
                   help="子进程之间的间隔（连续起进程会触发瞬时 CUDA 初始化失败）")
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
