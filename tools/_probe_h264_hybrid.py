"""_probe_h264_hybrid.py —— h264 hybrid 达成率（对两解码器并联和）归因载体。

命题
----
hybrid 的**理论并联和** = CPU 单臂速率 + NVDEC 单臂速率（两侧都不闲着时的
上界）。C-46 记录三码达成率 hevc 95% / **h264 78%** / av1 93% —— h264 明显
偏低。本探针是这条命题的取证工具：同一次会话内先量两臂单臂，再量多个
hybrid 变体（线程档 / 份额 / DLL），**全部独立子进程 + 逐轮交错 + min-of-N**，
避免跨会话漂移把"单臂和"与"hybrid"比出假缺口。

口径（与 `_probe_gap_decomp.py` 同源，便于与 C-45/C-46 数字互认）
--------------------------------------------------------------
- ROI-first + `output_format='gray'` + `DECODE_BATCH_SIZE` 的 get_batch 粒度、
  全片、首批（冷）不入账；调用方只取批，不做任何 analyze/分段/OCR。
- 分臂账目/忙时/HOL 一律解 `DECORD_HYBRID_STATS=1` 的 `[hybrid-stats]` 行；
  `DECORD_HYBRID_DEBUG=1` 时另收 `[emit-tl]`/`[hybrid-budget]` 时间线。
- 每臂独立子进程；跨臂比较用同会话读数（`--reps` 轮内交错）。

用法
----
    python tools/_probe_h264_hybrid.py --landscape          # 单臂 + 线程扫描
    python tools/_probe_h264_hybrid.py --shares 0.5,0.7,0.9 # 份额扫描
    python tools/_probe_h264_hybrid.py --arm hybrid_gpu:t24:DECORD_HYBRID_GPU_GAMMA=0.5
    python tools/_probe_h264_hybrid.py --landscape --dll-dir <含 FFmpeg 运行库的目录>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

_VDIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))
OUT = ROOT / "bench" / "h264_hybrid.json"

VIDS = {"h264": ("test5.mp4", (843, 993, 948, 1025), 7761),
        "hevc": ("test6_hevc.mp4", (841, 994, 949, 1026), 23970),
        "av1": ("test6.mp4", (841, 994, 949, 1026), 23970)}
THREADS = {"h264": 32, "hevc": 32, "av1": 24}
# min-of-N 轮数按片长缩放（省时）：短片的单轮成本低
REPS_DEFAULT = {"h264": 3, "hevc": 2, "av1": 2}

WORKER = r'''
import json, os, sys, time
ROOT = os.environ["PROBE_ROOT"]
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import decord
from decord import VideoReader

ctx_name, path, roi_s, n, threads, batch = sys.argv[1:7]
roi = tuple(int(v) for v in roi_s.split(","))
n, threads, batch = int(n), int(threads), int(batch)
if ctx_name == "hybrid_gpu":
    ctx = decord.hybrid_gpu(0)
elif ctx_name == "hybrid":
    ctx = decord.hybrid(0)
elif ctx_name == "gpu":
    ctx = decord.gpu(0)
else:
    ctx = decord.cpu(0)
kw = {"num_threads": threads} if ctx_name != "gpu" else {}
vr = VideoReader(path, ctx=ctx, output_format="gray", roi=roi, **kw)
frames = list(range(n))
first = None
lst = []
t_all0 = time.perf_counter()
for bstart in range(0, len(frames), batch):
    bend = min(bstart + batch, len(frames))
    t0 = time.perf_counter()
    nds = vr.get_batch(frames[bstart:bend])
    dt = time.perf_counter() - t0
    nfr = int(nds.shape[0]) if getattr(nds, "shape", None) else (bend - bstart)
    if first is None:
        first = dt
    else:
        lst.append((nfr, dt))
t_all = time.perf_counter() - t_all0
try:
    vr.close()
except Exception:
    pass  # 关闭失败不影响已测速率；hybrid 生产者线程由析构兜底
tot_f = sum(f for f, _ in lst)
tot_t = sum(t for _, t in lst)
# ⚠️ 键名必须是 decoded_n（不能叫 frames）：[hybrid-stats] 解析也写 "frames"
# 键，撞名会让"单臂无 stats 行"的情况把 int 当四元组解包（2026-09-14 踩过）。
print(json.dumps({"ctx": ctx_name, "decoded_n": tot_f, "secs": tot_t,
                  "wall_all": t_all, "first": first,
                  "fps": (tot_f / tot_t) if tot_t else 0.0}))
'''

_P = {
    "frames": re.compile(r"frames c=(\d+) g=(\d+) chunks c=(\d+) g=(\d+)"),
    "plan": re.compile(r"plan rc=(\d+) rg=(\d+) frames c=(\d+) g=(\d+)"),
    "hol": re.compile(r"hol cpu-head us=(\d+) ev=(\d+).*?gpu-head us=(\d+) ev=(\d+)"),
    "busy": re.compile(r"busy cpu_us=(\d+) pkts=\d+ \| gpu_us=(\d+) pics=(\d+)"),
    "disp": re.compile(r"disp gops c=(\d+) g=(\d+) cache_peak=(\d+)MB "
                       r"late=(\d+) strag=(\d+) rc_now=(\d+) rg_now=(\d+) "
                       r"assigned c=(\d+) g=(\d+)"),
    "budget": re.compile(r"\[hybrid-budget\][^\n]*"),
    "emit": re.compile(r"\[emit-tl\] t=([\d.]+) total=(\d+) c=(\d+)\(\+(\d+)\) "
                       r"g=(\d+)\(\+(\d+)\) rate=(\d+)"),
}


def foreign_jobs() -> list[str]:
    """列出**别人**的重负载进程（bench/probe/decord 子进程）。

    ⚠️ 2026-09-14 两次踩到：上一会话遗留的实验进程树在本会话里继续跑
    （一次是 `_probe_dll_ab.py --rounds 16`，一次是 `bench.py ab … GAMMA`，
    两者都吃掉 12~14 核）⇒ 同期一切读数作废。本函数只报告、不杀：
    杀进程是破坏性动作，留给人或上层决定。
    """
    out = []
    try:
        import psutil
        me = os.getpid()
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            if p.info["pid"] == me:
                continue
            name = (p.info["name"] or "").lower()
            if "python" not in name and "ninja" not in name and "cl" != name:
                continue
            cl = " ".join(p.info["cmdline"] or [])
            if "multisim_mcp" in cl:
                continue          # 常驻 MCP 服务，实测 <1% 占用
            if any(k in cl for k in ("bench.py", "_probe_", "decord")):
                out.append("pid=%d %s" % (p.info["pid"], cl[:150]))
    except Exception:  # noqa: BLE001
        pass  # 外部进程探查失败不应阻断测量
    return out


def load_pct(sample_s: float = 0.6) -> float:
    """机器忙闲采样（psutil，采样窗内的总占用）。

    ️ 2026-09-14 教训：上一会话遗留的 `_probe_dll_ab.py`（16 轮）在后台吃掉
    12.8 核，而同刻本探针报出 cpu-single 同配置 3146/1887 fps（±40%）——
    一切并发实验的读数都是废的。故本探针每臂前后各采一次忙闲并落盘，
    忙闲 > `--max-load` 时**当轮作废并告警**（不静默）。
    """
    try:
        import psutil
        return float(psutil.cpu_percent(interval=sample_s))
    except Exception:  # noqa: BLE001 — 无 psutil 时不阻断，退化为"未知"
        return -1.0


def env_note() -> dict:
    """测量的环境旁证（绝对速率随电源方案/频率漂移，必须随读数留档）。"""
    note = {}
    try:
        import psutil
        note["cpu_freq_mhz"] = int(psutil.cpu_freq().current)
        note["logical"] = psutil.cpu_count()
    except Exception:  # noqa: BLE001
        pass  # 环境旁证缺失不阻断测量
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=10).stdout
        note["power_scheme"] = out.strip()[-40:]
    except Exception:  # noqa: BLE001
        pass  # 同上：powercfg 不在/超时都不阻断测量
    return note


def run_clean(ctx_name: str, codec: str, th, frames, env, dll_dir, want_emit,
              settle: float, max_load: float, retries: int) -> dict:
    """跑一臂并在**机器污染**时重试（最多 retries 次），返回首个干净读数。

    ⚠️ 本机外部负载是块状出现的（实测某轮 pre-load 85~90% 持续两三分钟）：
    最常见触发源是**刚拷过大文件**（Defender 实时扫描 ~150MB FFmpeg DLL 运行库）
    或构建。旧版只在污染时标一个 flag、不做重试 ⇒ 整轮 6 臂读数全废、
    白跑十分钟。故污染一律重跑；全部尝试都污染时返回最好一次并留标记。
    """
    best = None
    for attempt in range(retries + 1):
        if settle:
            time.sleep(settle)
        l0 = load_pct()
        r = run_arm(ctx_name, codec, th, frames, env, dll_dir, want_emit)
        r["load_pre"] = l0
        time.sleep(1.5)   # 等子进程 CUDA/线程收尾，否则误判污染（实测 65%+）
        r["load_post"] = min(load_pct(0.3), load_pct(0.3))
        bad = max(l0, r["load_post"]) > max_load
        r["contaminated"] = bad
        r["attempt"] = attempt
        if "error" not in r:
            if best is None or (best.get("contaminated") and not bad) or (
                    not bad and not best.get("contaminated")
                    and r["fps"] > best["fps"]):
                best = r
        if not bad and "error" not in r:
            return best
        if bad:
            print("     ↻ 机器忙闲 %.0f%%/%.0f%% > %.0f%%：重试 %d/%d"
                  % (l0, r["load_post"], max_load, attempt + 1, retries))
    return best if best is not None else r


def run_arm(ctx_name: str, codec: str, threads: int | None, frames: int | None,
            env_extra: dict, dll_dir: str | None, want_emit: bool,
            timeout: int = 1800) -> dict:
    vid, roi, full_n = VIDS[codec]
    path = str(_VDIR / vid)
    threads = THREADS[codec] if threads is None else threads
    frames = full_n if frames is None else frames
    try:
        from video_ocr_engine.config import constants as cfg
        batch = int(cfg.DECODE_BATCH_SIZE)
    except Exception:  # noqa: BLE001 — 探针可独立运行，缺配置时按历史默认
        batch = 16
    env = dict(os.environ)
    env["PROBE_ROOT"] = str(ROOT)
    env["DECORD_HYBRID_STATS"] = "1"
    env.pop("DECORD_HYBRID_DEBUG", None)
    if want_emit:
        env["DECORD_HYBRID_DEBUG"] = "1"
    if dll_dir:
        env["DECORD_LIBRARY_PATH"] = dll_dir
    env.update(env_extra)
    p = subprocess.run([sys.executable, "-c", WORKER, ctx_name, path,
                        ",".join(map(str, roi)), str(frames), str(threads),
                        str(batch)],
                       capture_output=True, text=True, encoding="utf-8",
                       env=env, timeout=timeout)
    if p.returncode != 0:
        return {"ctx": ctx_name, "error": (p.stderr or "")[-600:]}
    row = json.loads(p.stdout.strip().splitlines()[-1])
    # ⚠️ 两套口径必须分开记账：`fps` 是**批计时和**（首批冷启动已剔除，与
    # `_probe_decode_rate.py` 同源）；`wall_fps` 是**端到端墙钟**（含首批）。
    # 达成率一律用 wall_fps 对 wall_fps（混用会把 hybrid 的冷启动白算成收益：
    # 实测 h264 同轮 3840(batch) vs 3678(wall)，差 4.4%）。
    row["wall_fps"] = (row["decoded_n"] + 0) / row["wall_all"] if row["wall_all"] else 0.0
    err = p.stderr or ""
    for name, pat in _P.items():
        if name in ("budget", "emit"):
            continue
        m = pat.search(err)
        if m:
            row[name] = [int(g) for g in m.groups()]
    mb = _P["budget"].search(err)
    row["budget"] = mb.group(0) if mb else ""
    row["emit_tl"] = [[float(g[0])] + [int(v) for v in g[1:]]
                      for g in _P["emit"].findall(err)]
    row["stderr_tail"] = err[-400:]
    return row


def fmt(label: str, r: dict, sum_fps: float | None) -> str:
    if "error" in r:
        return "%-24s 失败: %s" % (label, r["error"][:200].replace("\n", " "))
    bits = ["wall=%.2fs fps_b=%.0f fps_w=%.0f"
            % (r["wall_all"], r["fps"], r["wall_fps"])]
    if r.get("contaminated"):
        bits.append("⚠️污染读数")
    if sum_fps:
        bits.append("**达成率=%.0f%%**(和=%.0f)" % (100.0 * r["wall_fps"] / sum_fps,
                                                   sum_fps))
    if "frames" in r and isinstance(r["frames"], list):
        fc, fg, cc, cg = r["frames"]
        w = r["wall_all"]
        bits.append("deliv c=%d(%.0f) g=%d(%.0f) | gops c=%d g=%d"
                    % (fc, fc / w, fg, fg / w, cc, cg))
    if "busy" in r:
        bits.append("busy c=%.0f%% g=%.0f%%"
                    % (100.0 * r["busy"][0] / 1e6 / r["wall_all"],
                       100.0 * r["busy"][1] / 1e6 / r["wall_all"]))
    if "hol" in r:
        bits.append("hol c=%.2fs g=%.2fs" % (r["hol"][0] / 1e6,
                                             r["hol"][2] / 1e6))
    if r.get("plan"):
        rc, rg, pc, pg = r["plan"]
        bits.append("plan rc=%d rg=%d share_c=%.3f" % (rc, rg, pc / max(pc + pg, 1)))
    return "%-24s %s" % (label, "  ".join(bits))


def parse_arm(spec: str) -> tuple[str, int | None, dict, str]:
    """`<ctx>[:t<threads>][:K=V,K=V][@label]` → (ctx, threads, env, label)。"""
    label = ""
    if "@" in spec:
        spec, _, label = spec.partition("@")
    parts = spec.split(":")
    ctx = parts[0]
    threads = None
    env: dict = {}
    for p in parts[1:]:
        if not p:
            continue
        if p.startswith("t") and p[1:].isdigit():
            threads = int(p[1:])
        elif "=" in p:
            for kv in p.split(","):
                k, _, v = kv.partition("=")
                env[k] = v
    return ctx, threads, env, (label or spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="h264", choices=sorted(VIDS))
    ap.add_argument("--landscape", action="store_true",
                    help="单臂 cpu/gpu + hybrid 线程扫描")
    ap.add_argument("--threads", default="16,20,24,28,32")
    ap.add_argument("--shares", default="",
                    help="份额扫描（DECORD_HYBRID_FORCE_SHARE），逗号分隔")
    ap.add_argument("--arm", action="append", default=[],
                    help="自定义臂，可重复：<ctx>[:t<threads>][:K=V,K=V][@label]")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--reps", type=int, default=0, help="0=按片长取默认")
    ap.add_argument("--dll-dir", default="")
    ap.add_argument("--emit-tl", action="store_true")
    ap.add_argument("--settle", type=float, default=4.0,
                    help="臂间静默秒数（等上一进程的 CUDA/线程收尾；"
                         "实测 1.5s 不够——自身收尾尾巴会被误判为机器污染）")
    ap.add_argument("--retries", type=int, default=3,
                    help="机器被污染时该臂的重试次数（默认 3）")
    ap.add_argument("--max-load", type=float, default=40.0,
                    help="机器忙闲上限（百分数）；超过则该轮读数标 contaminated。"
                         "本机基线（msedge/svchost/Legion 工具）约 10~20%%")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    arms: list[tuple[str, int | None, dict, str]] = []
    if args.landscape:
        arms.append(("cpu", None, {}, "cpu-single"))
        arms.append(("gpu", None, {}, "gpu-single"))
        for t in (int(x) for x in args.threads.split(",") if x.strip()):
            arms.append(("hybrid_gpu", t, {}, "hybrid-t%d" % t))
    for s in (x for x in args.shares.split(",") if x.strip()):
        arms.append(("hybrid_gpu", None,
                     {"DECORD_HYBRID_FORCE_SHARE": s}, "hybrid-share%s" % s))
    for a in args.arm:
        arms.append(parse_arm(a))
    if not arms:
        ap.error("至少要给 --landscape / --shares / --arm 之一")

    reps = args.reps or REPS_DEFAULT[args.codec]
    fj = foreign_jobs()
    if fj:
        print("⚠️⚠️ 检测到**外部重负载进程**（读数会作废，建议先停）：")
        for x in fj:
            print("      " + x)
    print("codec=%s  arms=%d  reps=%d  dll=%s"
          % (args.codec, len(arms), reps, args.dll_dir or "(安装包默认)"))
    best: dict[str, dict] = {}
    for i in range(reps):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for ctx, th, env, label in order:
            r = run_clean(ctx, args.codec, th, args.frames, env, args.dll_dir,
                          args.emit_tl, args.settle, args.max_load, args.retries)
            prev = best.get(label)
            if "error" not in r and not r.get("contaminated") and (
                    prev is None or prev.get("contaminated")
                    or r["fps"] > prev["fps"]):
                best[label] = r
            print("  [r%d] %-24s %s%s%s" % (
                i, label,
                "fps=%.0f" % r["fps"] if "error" not in r
                else "失败: " + r["error"][:120].replace("\n", " "),
                "  (污染)" if r.get("contaminated") else "",
                "  %d" % r["attempt"] if r.get("attempt") else ""))

    sum_fps = None
    if "cpu-single" in best and "gpu-single" in best:
        n = args.frames or VIDS[args.codec][2]
        sum_fps = n / best["cpu-single"]["wall_all"] + n / best["gpu-single"]["wall_all"]
    print("\n== 结果（min-of-%d） ==" % reps)
    out = {}
    for ctx, th, env, label in arms:
        r = best.get(label)
        if not r:
            continue
        print(fmt(label, r, sum_fps if ctx in ("hybrid_gpu", "hybrid") else None))
        out[label] = {k: v for k, v in r.items() if k != "stderr_tail"}
        out[label]["threads"] = th
        out[label]["env"] = env
    if "hybrid-default" in best and sum_fps:
        a = best["hybrid-default"]["wall_fps"]
        out["_verdict"] = {
            "hybrid_wall_fps": a, "sum_fps": sum_fps,
            "attainment_pct": 100.0 * a / sum_fps,
            "cpu_wall_fps": VIDS[args.codec][2] / best["cpu-single"]["wall_all"],
            "gpu_wall_fps": VIDS[args.codec][2] / best["gpu-single"]["wall_all"],
        }
    if sum_fps:
        print("\n并联和 = cpu %.0f + gpu %.0f = %.0f fps（本会话同刻读数，wall 口径）"
              % (VIDS[args.codec][2] / best["cpu-single"]["wall_all"],
                 VIDS[args.codec][2] / best["gpu-single"]["wall_all"], sum_fps))
    payload = {"codec": args.codec, "tag": args.tag, "reps": reps,
               "dll_dir": args.dll_dir, "sum_fps": sum_fps, "arms": out,
               "env_note": env_note()}
    old = {}
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — 旧文件损坏不应阻断本轮取证
            old = {}
    old.setdefault(args.codec, {})
    old[args.codec].update(out)
    old["%s_meta" % args.codec] = {"tag": args.tag, "reps": reps,
                                   "dll_dir": args.dll_dir, "sum_fps": sum_fps}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
    print("落盘", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())