"""hybrid 发射队头阻塞（HOL）与规划速率的**非打印**实测（§7.2-1/2 重采）。

背景：DECORD_HYBRID_DEBUG 一次运行 13 万行 stderr，墙钟 2940 vs 3100fps
——打印本身扰动测量，此前"764 次 flush"一类归因数字全部带着这个污染。
fork 现在（dll 96C0293C029001B2 起）提供 `DECORD_HYBRID_STATS`：热路径只做
relaxed 整数累加，**析构时一次性**打印 [hybrid-stats] 汇总。本探针按解码
only 口径（同 _probe_hybrid_gap：gray/批64/num_threads=32）跑各 case×ctx，
从 stderr 解析 stats，回答三个问题：

1. HOL 占墙钟百分之几？卡在 CPU-head 还是 GPU-head？被搁置的对侧存货多深？
2. BuildPlan 冻结的速率 rc/rg 与单臂 decode-only 实测（knowledge
   `decode_rates_fork_dll_batch64`）相符吗？规划份额 vs 实际交付份额。
3. （hybrid_gpu）上载批次的真实 avg_batch / nobuf / cpu-empty——不带打印扰动。

方法学：每格**独立子进程 + 硬超时**（一次 del vr 出一份 stats）；轮内
顺序轮转；`DECORD_HYBRID_DEBUG` 一律清除；墙钟只含解码循环。

用法（先设 DECORD_FORK_BUILD 指向 fork 构建树，否则多半用 0.8.2 wheel，
不含 stats——WORKER_FAIL/无 stats 行即为该信号）：
  python tools/_probe_hol_stats.py --cases h264,hevc,av1 --ctxs hybrid
  python tools/_probe_hol_stats.py --cases hevc --ctxs hybrid_gpu --reps 2
  python tools/_probe_hol_stats.py --set DECORD_HYBRID_FORCE_SHARE=0.3   # 份额实验
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKER = r'''
import os, sys, time, json
sys.path.insert(0, os.environ["PROBE_ROOT"])
import decord
from decord import cpu, gpu, hybrid, hybrid_gpu
name = os.environ["PROBE_CTX"]
CTX = {"hybrid": hybrid, "hybrid_gpu": hybrid_gpu,
       "cpu": cpu, "nvdec": gpu}[name]
vr = decord.VideoReader(os.environ["PROBE_VID"], ctx=CTX(0),
                        output_format="gray",
                        num_threads=32 if name in ("cpu", "hybrid") else 0,
                        roi=tuple(int(x) for x in
                                  os.environ["PROBE_ROI"].split(",")))
n = min(int(os.environ["PROBE_FRAMES"]), len(vr))
idx = list(range(n))
t = time.perf_counter()
got = 0
for s in range(0, n, 64):
    got += vr.get_batch(idx[s:s + 64]).shape[0]
wall = time.perf_counter() - t
del vr   # 析构 → Stop() → 一次性 [hybrid-stats]
print("HOLJSON " + json.dumps({"wall": wall, "got": got}))
'''

# 视频目录约定同 bench.py / _probe_upload_chain.py（换机器不坏）
_VDIR = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
CASES = {"h264": (os.path.join(_VDIR, "test5.mp4"), "843,993,949,1026"),
         "hevc": (os.path.join(_VDIR, "test6_hevc.mp4"), "841,994,950,1027"),
         "av1": (os.path.join(_VDIR, "test6.mp4"), "841,994,950,1027"),
         # test6(av1) 的 h264 转码（同内容同长度 23970 帧、同 GOP 300、
         # High profile、libx264 crf19 medium）——长片单臂 vs 混合的
         # 控制变量臂，编码/长度/GOP 与 av1 源全同（§11 后续实验）
         "h264lg": (os.path.join(_VDIR, "test6_h264.mp4"), "841,994,950,1027")}

_RE = {
    "mode": re.compile(r"mode=(\S+) frames c=(\d+) g=(\d+) chunks c=(\d+) g=(\d+)"),
    "plan": re.compile(r"plan rc=(\d+) rg=(\d+) frames c=(\d+) g=(\d+)"
                       r"(?: folds=(\d+) age=(\d+)ms)?"),
    "hol": re.compile(r"hol cpu-head us=(\d+) ev=(\d+) strandmax=(\d+)"
                      r" \| gpu-head us=(\d+) ev=(\d+) strandmax=(\d+)"),
    "up": re.compile(r"upload flushes=(\d+) frames=(\d+) avg_batch=([\d.]+)"
                     r" nobuf=(\d+) cpuempty=(\d+)"),
}


def parse_stats(err: str) -> dict | None:
    if "[hybrid-stats]" not in err:
        return None
    out: dict = {}
    for ln in err.splitlines():
        if "[hybrid-stats] mode=" in ln:
            m = _RE["mode"].search(ln)
            out.update(mode=m.group(1), fc=int(m.group(2)), fg=int(m.group(3)))
        elif "[hybrid-stats] plan " in ln:
            m = _RE["plan"].search(ln)
            out.update(rc=int(m.group(1)), rg=int(m.group(2)),
                       plan_c=int(m.group(3)), plan_g=int(m.group(4)))
            if m.group(5):
                out.update(plan_folds=int(m.group(5)), plan_age_ms=int(m.group(6)))
        elif "[hybrid-stats] hol " in ln:
            m = _RE["hol"].search(ln)
            out.update(hol_us_c=int(m.group(1)), hol_ev_c=int(m.group(2)),
                       strand_c=int(m.group(3)), hol_us_g=int(m.group(4)),
                       hol_ev_g=int(m.group(5)), strand_g=int(m.group(6)))
        elif "[hybrid-stats] upload " in ln:
            m = _RE["up"].search(ln)
            out.update(flushes=int(m.group(1)), up_frames=int(m.group(2)),
                       avg_batch=float(m.group(3)), nobuf=int(m.group(4)),
                       cpuempty=int(m.group(5)))
    return out


def run_cell(vid: str, roi: str, ctx: str, frames: int, timeout: float,
             extra: dict) -> tuple[dict | None, str | None]:
    env = dict(os.environ)
    env.update({"PROBE_ROOT": str(ROOT), "PROBE_VID": vid, "PROBE_ROI": roi,
                "PROBE_CTX": ctx, "PROBE_FRAMES": str(frames),
                "DECORD_HYBRID_STATS": "1"})
    env.pop("DECORD_HYBRID_DEBUG", None)
    env.pop("DECORD_HYBRID_FORCE_SIDE", None)
    env.update(extra)
    try:
        p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    for ln in p.stdout.splitlines():
        if ln.startswith("HOLJSON "):
            r = json.loads(ln[len("HOLJSON "):])
            st = parse_stats(p.stderr) or {}
            r["stats"] = st
            return r, None
    return None, "WORKER_FAIL:" + (p.stderr.strip().splitlines()
                                   or ["?"])[-1][:80]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="h264,hevc,av1")
    ap.add_argument("--ctxs", default="hybrid,hybrid_gpu")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--fork", default=os.environ.get("DECORD_FORK_BUILD", ""))
    ap.add_argument("--set", action="append", default=[],
                    help="附加 env，KEY=VAL，可重复（份额/预算实验用）")
    args = ap.parse_args()
    extra = dict(kv.split("=", 1) for kv in args.set)
    if args.fork:
        os.environ["DECORD_LIBRARY_PATH"] = args.fork
    else:
        print("⚠️ 未给 DECORD_FORK_BUILD → 用已安装 decord（0.8.2 wheel 无"
              " stats，会报无 [hybrid-stats]）")
    cells = [(c, x) for c in args.cases.split(",") for x in args.ctxs.split(",")]
    res: dict = {}
    fails = 0
    for i in range(args.reps):
        rot = cells[i % len(cells):] + cells[:i % len(cells)]
        for case, ctx in rot:
            vid, roi = CASES[case]
            t0 = time.perf_counter()
            r, err = run_cell(vid, roi, ctx, args.frames, args.timeout, extra)
            res.setdefault((case, ctx), []).append(r)
            if r is None:
                fails += 1
                print("  [%d] %-5s %-10s %s" % (i, case, ctx, err), flush=True)
                continue
            st = r["stats"]
            hol_ms = (st.get("hol_us_c", 0) + st.get("hol_us_g", 0)) / 1000
            print("  [%d] %-5s %-10s %6.0f fps  hol=%.0fms/%.0fms (%.1f%%)"
                  "  plan rc/rg=%d/%d share_g=%.0f%%→real %.0f%%"
                  " folds=%d age=%dms%s" % (
                      i, case, ctx, r["got"] / r["wall"], hol_ms,
                      r["wall"] * 1000, hol_ms / (r["wall"] * 10) if r["wall"] else 0,
                      st.get("rc", 0), st.get("rg", 0),
                      100 * st.get("plan_g", 0) / max(st.get("plan_g", 0) + st.get("plan_c", 0), 1),
                      100 * st.get("fg", 0) / max(st.get("fc", 0) + st.get("fg", 0), 1),
                      st.get("plan_folds", -1), st.get("plan_age_ms", -1),
                      ("  batch=%.1f nobuf=%d cempty=%d" % (
                          st.get("avg_batch", 0), st.get("nobuf", 0),
                          st.get("cpuempty", 0))) if "avg_batch" in st else ""),
                  flush=True)
    print("\n%-5s %-10s %7s %7s %9s %9s %13s %9s" % (
        "编码", "ctx", "fps", "HOL%", "c-head秒", "g-head秒",
        "strandmax c/g", "批均"))
    for case, ctx in cells:
        v = [x for x in res.get((case, ctx), []) if x]
        if not v:
            print("%-5s %-10s 无成功槽位" % (case, ctx))
            continue
        w = statistics.median(x["wall"] for x in v)
        med = min(v, key=lambda x: abs(x["wall"] - w))
        st = med["stats"]
        hol_c = st.get("hol_us_c", 0) / 1e6
        hol_g = st.get("hol_us_g", 0) / 1e6
        print("%-5s %-10s %7.0f %6.1f%% %9.2f %9.2f %7d/%-5d %9s" % (
            case, ctx, med["got"] / med["wall"],
            100 * (hol_c + hol_g) / med["wall"], hol_c, hol_g,
            st.get("strand_c", 0), st.get("strand_g", 0),
            "%.1f" % st["avg_batch"] if "avg_batch" in st else "—"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
