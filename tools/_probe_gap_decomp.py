"""_probe_gap_decomp.py —— hybrid vs 两解码器之和的缺口分解（fork 级，带分臂账目）。

背景（工作项 6：并联效率 72~89% 的构成）
------------------------------------------
工作项 0 已定"封顶在 fork"。本探针进一步把 fork 级缺口拆成三块：

  · 每臂混跑内速率退化：delivered_c/wall、delivered_g/wall 对比
    同刻单臂（ctx=cpu / ctx=gpu）速率；
  · 计划份额失配：[hybrid-stats] plan frames c/g 的规划份额 vs
    "混跑内速率"意义下的最优份额（rc'/(rc'+rg')），失配部分表现为
    队尾闲置（一侧先做完）；
  · 队头阻塞（HOL）：stats 的 hol 行直接读。

口径
----
- 与 `_probe_decode_rate.py` 同源（ROI-first + gray + DECODE_BATCH 粒度 +
  首批丢弃），差异：**全片**、解析 `[hybrid-stats]`（DECORD_HYBRID_STATS=1）、
  支持 `--threads` 扫描 CPU 臂线程档。
- GPU 臂 busy 口径是 cuvidDecodePicture 提交时长（≠ 解码管线利用率），
  只作参考；分臂速率一律用 delivered/wall。
- 每臂独立子进程；跨臂比较用同刻差值。

用法
----
    python tools/_probe_gap_decomp.py --codec hevc                 # 单码分解
    python tools/_probe_gap_decomp.py --matrix                     # 三码 × 三 ctx
    python tools/_probe_gap_decomp.py --codec hevc --threads 24    # 线程扫描臂
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

_VDIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))
OUT = ROOT / "bench" / "gap_decomp.json"

VIDS = {"h264": ("test5.mp4", (843, 993, 948, 1025), 7761),
        "hevc": ("test6_hevc.mp4", (841, 994, 949, 1026), 23970),
        "av1": ("test6.mp4", (841, 994, 949, 1026), 23970)}
# 引擎 §14 档位（与 _probe_decode_rate.py 同源）
THREADS = {"h264": 32, "hevc": 32, "av1": 24}

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
    # ⚠️ len(nds) 是元素总数不是批帧数（见 _probe_decode_rate.py 留档）
    if first is None:
        first = dt
    else:
        lst.append((nfr, dt))
t_all = time.perf_counter() - t_all0
try:
    vr.close()
except Exception:
    pass
tot_f = sum(f for f, _ in lst)
tot_t = sum(t for _, t in lst)
print(json.dumps({"ctx": ctx_name, "frames": tot_f, "secs": tot_t,
                  "wall_all": t_all, "first": first,
                  "fps": (tot_f / tot_t) if tot_t else 0.0}))
'''

_STATS_PATTERNS = {
    "frames": re.compile(r"\[hybrid-stats\] mode=\S+ frames c=(\d+) g=(\d+) chunks c=(\d+) g=(\d+)"),
    "plan": re.compile(r"\[hybrid-stats\] plan rc=(\d+) rg=(\d+) frames c=(\d+) g=(\d+)"),
    "hol": re.compile(r"\[hybrid-stats\] hol cpu-head us=(\d+) ev=(\d+).*gpu-head us=(\d+) ev=(\d+)"),
    "busy": re.compile(r"\[hybrid-stats\] busy cpu_us=(\d+) pkts=\d+ \| gpu_us=(\d+) pics=(\d+)"),
}


def run_arm(ctx_name: str, codec: str, threads: int | None, reps: int,
            frames: int | None) -> dict:
    vid, roi, full_n = VIDS[codec]
    path = str(_VDIR / vid)
    if threads is None:
        threads = THREADS[codec]
    if frames is None:
        frames = full_n
    try:
        from video_ocr_engine.config import constants as cfg
        batch = int(cfg.DECODE_BATCH_SIZE)
    except Exception:
        batch = 16
    env = dict(os.environ)
    env["PROBE_ROOT"] = str(ROOT)
    env["DECORD_HYBRID_STATS"] = "1"
    fork_build = os.environ.get("DECORD_FORK_BUILD")
    if fork_build:
        env["DECORD_LIBRARY_PATH"] = fork_build
    rows = []
    for _ in range(reps):
        p = subprocess.run([sys.executable, "-c", WORKER, ctx_name, path,
                            ",".join(map(str, roi)), str(frames), str(threads),
                            str(batch)],
                           capture_output=True, text=True, encoding="utf-8",
                           env=env, timeout=1200)
        if p.returncode != 0:
            return {"ctx": ctx_name, "codec": codec,
                    "error": (p.stderr or "")[-500:]}
        row = json.loads(p.stdout.strip().splitlines()[-1])
        for name, pat in _STATS_PATTERNS.items():
            m = pat.search(p.stderr or "")
            if m:
                row[name] = [int(g) for g in m.groups()]
        rows.append(row)
    best = max(rows, key=lambda r: r["fps"])
    best.update(codec=codec, threads=threads, frames_req=frames, reps=reps,
                fps_all=[round(r["fps"], 1) for r in rows])
    return best


def decompile(r: dict) -> str:
    """人读分解行：in-hybrid 分臂速率 / 退化 / 计划份额失配 / HOL。"""
    if "error" in r:
        return "失败: %s" % r["error"][:120]
    wall = r["wall_all"]
    fps = r["fps"]
    bits = ["wall=%.2fs fps=%.0f" % (wall, fps)]
    if "frames" in r:
        fc, fg, cc, cg = r["frames"]
        rc_eff = fc / wall
        rg_eff = fg / wall
        bits.append("deliv c=%d(%.0ffps) g=%d(%.0ffps)" % (fc, rc_eff, fg, rg_eff))
        if "plan" in r:
            prc, prg, pfc, pfg = r["plan"]
            tot = pfc + pfg
            plan_share = pfc / tot if tot else 0.0
            opt = rc_eff / (rc_eff + rg_eff) if (rc_eff + rg_eff) else 0.0
            bits.append("plan rc=%d rg=%d share_c=%.3f(opt≈%.3f)" %
                        (prc, prg, plan_share, opt))
        if "busy" in r:
            bc, bg = r["busy"][0], r["busy"][1]
            bits.append("busy c=%.0f%% g=%.0f%%" % (100.0 * bc / 1e6 / wall,
                                                    100.0 * bg / 1e6 / wall))
        if "hol" in r:
            hc, _, hg, _ = r["hol"]
            bits.append("hol c=%.2fs g=%.2fs" % (hc / 1e6, hg / 1e6))
    return "  ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="hevc", choices=sorted(VIDS))
    ap.add_argument("--ctx", default="hybrid_gpu",
                    choices=["hybrid_gpu", "gpu", "cpu"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--matrix", action="store_true")
    args = ap.parse_args()

    combos = ([(c, ctx) for c in VIDS for ctx in ("hybrid_gpu", "gpu", "cpu")]
              if args.matrix else [(args.codec, args.ctx)])
    out = {}
    for codec, ctx in combos:
        r = run_arm(ctx, codec, args.threads, args.reps, args.frames)
        key = "%s/%s" % (codec, ctx)
        if args.threads:
            key += "/t%d" % args.threads
        out[key] = r
        print("%-22s %s" % (key, decompile(r)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    old.update(out)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
    print("落盘", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
