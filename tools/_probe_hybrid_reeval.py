"""hybrid 重评（2026-09-12 可见性落地后）：忙而慢 vs 调度闲置的逐码归因。

每码两臂（nvdec / hybrid，引擎全片、GPU 管线 + TRT、VOE_TELEMETRY=full），
子进程隔离：stdout=引擎摘要，stderr=[hybrid-stats] 由父进程解析。

判读（本次新增的三件套）：
  - report.hardware: NVDEC 占用率 + SM/VIDEO 时钟 + 热降原因位
  - [hybrid-stats] busy: 每臂实际解码忙时（busy fps 与忙碌占比）
  - [hybrid-stats] hol/plan: 队头阻塞与规划份额兑现

用法：python tools/_probe_hybrid_reeval.py [--reps 2]
落盘 bench/hybrid_reeval.json
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
# fork 构建树（含 busy 计数器，2026-09-13 起）：与 _probe_hol_stats 同一 env 约定
FORK = os.environ.get("DECORD_FORK_BUILD", "")
sys.stdout.reconfigure(encoding="utf-8")
if not FORK:
    print("需 DECORD_FORK_BUILD 指向含 busy 计数器的 fork 构建树"
          "（≥6da2957）")
    raise SystemExit(2)

_VDIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))
OUT = ROOT / "bench" / "hybrid_reeval.json"

CASES = {"h264": ("test5.mp4", (843, 993, 948, 1025)),
         "hevc": ("test6_hevc.mp4", (841, 994, 949, 1026)),
         "av1": ("test6.mp4", (841, 994, 949, 1026))}

RE_BUSY = re.compile(r"busy cpu_us=(\d+) pkts=(\d+) \| gpu_us=(\d+) pics=(\d+)")
RE_MODE = re.compile(r"mode=(\S+) frames c=(\d+) g=(\d+)")
RE_HOL = re.compile(r"hol cpu-head us=(\d+) ev=(\d+) strandmax=(\d+)"
                    r" \| gpu-head us=(\d+) ev=(\d+) strandmax=(\d+)")
RE_UP = re.compile(r"upload flushes=(\d+) frames=(\d+) avg_batch=([\d.]+)"
                   r" nobuf=(\d+) cpuempty=(\d+)")

WORKER = r"""
import os, sys, time, json
sys.path.insert(0, os.environ["PROBE_ROOT"])
sys.stdout.reconfigure(encoding="utf-8")
path, roi_s, backend = sys.argv[1], sys.argv[2], sys.argv[3]
_fe = int(sys.argv[4]) if len(sys.argv) > 4 else 0
roi = tuple(int(x) for x in roi_s.split(","))
from video_ocr_engine import FieldExtractor
_kw = {"frame_end": _fe} if _fe > 0 else {}
ex = FieldExtractor(path, roi, decode_backend=backend,
                    ocr_backend="tensorrt", keep_crops=False, **_kw)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
rep = r.meta.get("report") or {}
hw = rep.get("hardware") or {}
out = {"wall": round(wall, 3), "segs": len(r.segments),
       "decode_s": round((rep.get("spans", {}) or {}).get(
           "pipeline.decode", {}).get("sum", 0.0), 3),
       "backend": r.meta.get("backend"),
       "nvdec_util": hw.get("nvdec_util_pct"),
       "gpu_util": hw.get("gpu_util_pct"),
       "sm_clock": hw.get("sm_clock_mhz"),
       "video_clock": hw.get("video_clock_mhz"),
       "throttle": hw.get("throttle"),
       "hw_sources": hw.get("sources"), "hw_n": hw.get("n")}
print("E2EJSON " + json.dumps(out))
"""


def parse_stats(err: str) -> dict:
    st = {}
    for ln in err.splitlines():
        m = RE_MODE.search(ln)
        if m:
            st.update(mode=m.group(1), fc=int(m.group(2)), fg=int(m.group(3)))
        m = RE_BUSY.search(ln)
        if m:
            st.update(cpu_us=int(m.group(1)), pkts=int(m.group(2)),
                      gpu_us=int(m.group(3)), pics=int(m.group(4)))
        m = RE_HOL.search(ln)
        if m:
            st.update(hol_us_c=int(m.group(1)), hol_us_g=int(m.group(4)))
        m = RE_UP.search(ln)
        if m:
            st.update(up_avg=float(m.group(3)), up_nobuf=int(m.group(4)),
                      up_cempty=int(m.group(5)))
    return st


def run(case, vid, roi, backend, extra_env=None, window=0):
    env = dict(os.environ)
    env["PROBE_ROOT"] = str(ROOT)
    env["VOE_TELEMETRY"] = "full"
    env["DECORD_HYBRID_STATS"] = "1"
    env["DECORD_LIBRARY_PATH"] = FORK
    if extra_env:
        env.update(extra_env)
    _argv = [str(_VDIR / vid), ",".join(map(str, roi)), backend]
    if window:
        _argv.append(str(window))
    p = subprocess.run([sys.executable, "-c", WORKER] + _argv,
                       env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    out = None
    for ln in p.stdout.splitlines():
        if ln.startswith("E2EJSON "):
            out = json.loads(ln[8:])
    if out is None:
        return None, (p.stderr or "")[-300:]
    out["stats"] = parse_stats(p.stderr or "")
    return out, ""


def share_sweep(args) -> int:
    """扫 `DECORD_HYBRID_FORCE_SHARE`（CPU 侧份额），找 wall 最低点。

    "auto" = 不设 env（用 fork 按实测速率算的默认份额）。
    """
    shares = [s.strip() for s in args.share.split(",") if s.strip()]
    out: dict = {}
    for case in (args.cases.split(",") if args.cases else list(CASES)):
        vid, roi = CASES[case]
        print("\n== %s（%s）==" % (case, vid))
        print("%-8s %8s %8s %9s %9s %10s %7s" % (
            "share", "wall", "decode", "busy_g%", "busy_c%", "hol_c_ms", "segs"))
        rows = []
        for sh in shares:
            if sh == "auto":
                envx = {}
            elif sh.startswith("side:"):
                envx = {"DECORD_HYBRID_FORCE_SIDE": sh.split(":", 1)[1]}
            elif sh.startswith("env:"):
                envx = {}
                for kv in sh[4:].split("+"):
                    k, _, v = kv.partition("=")
                    envx[k.strip()] = v.strip()
            else:
                envx = {"DECORD_HYBRID_FORCE_SHARE": sh}
            best = None
            for _ in range(args.reps):
                r, err = run(case, vid, roi, "hybrid", envx, args.window)
                if r is None:
                    print("  %-8s FAIL %s" % (sh, err[-120:]))
                    continue
                if best is None or r["wall"] < best["wall"]:
                    best = r
            if best is None:
                continue
            st, wall = best["stats"], best["wall"]
            wg = st.get("gpu_us", 0) / 1e6
            wc = st.get("cpu_us", 0) / 1e6
            print("%-8s %8.3f %8.3f %9.1f %9.1f %10.1f %7d" % (
                sh, wall, best.get("decode_s") or 0.0,
                100 * wg / wall if wall else 0, 100 * wc / wall if wall else 0,
                st.get("hol_us_c", 0) / 1000.0, best["segs"]))
            rows.append({"share": sh, "wall": wall,
                         "decode_s": best.get("decode_s"),
                         "busy_g_pct": 100 * wg / wall if wall else 0,
                         "busy_c_pct": 100 * wc / wall if wall else 0,
                         "busy_g_fps": st.get("pics", 0) / wg if wg else 0,
                         "busy_c_fps": st.get("pkts", 0) / wc if wc else 0,
                         "hol_us_c": st.get("hol_us_c"),
                         "hol_us_g": st.get("hol_us_g"),
                         "stats": {k: v for k, v in st.items()
                                   if k not in ("stats",)}})
        out[case] = rows
        if rows:
            b = min(rows, key=lambda x: x["wall"])
            print("  → 最优 share=%s（wall %.3fs）" % (b["share"], b["wall"]))
    dst = ROOT / "bench" / "hybrid_share_sweep.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                   encoding="utf-8", newline="\n")
    print("\n落盘 %s" % dst)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--backends", default="nvdec,hybrid")
    ap.add_argument("--share", default="",
                    help="份额/路由扫描：逗号列表。auto=默认；数字=FORCE_SHARE"
                         "（⚠️ fork 里该形参未参与分配，实测无效）；"
                         "side:gpu / side:cpu = FORCE_SIDE（有效）")
    ap.add_argument("--cases", default="",
                    help="只跑指定编码，逗号分隔（h264,hevc,av1）")
    ap.add_argument("--window", type=int, default=0,
                    help="帧窗上限（0=全片）；用于量化启动期摊销")
    args = ap.parse_args()
    if args.share:
        return share_sweep(args)
    backends = args.backends.split(",")
    rep = {}
    for case, (vid, roi) in CASES.items():
        rep[case] = {}
        for be in backends:
            runs = []
            for i in range(args.reps):
                r, err = run(case, vid, roi, be, window=args.window)
                if r is None:
                    print("  %s/%s FAIL: %s" % (case, be, err[-160:]))
                    continue
                runs.append(r)
                st = r["stats"]
                wall = r["wall"]
                nu = (r.get("nvdec_util") or {})
                busy = ""
                if st.get("gpu_us") or st.get("cpu_us"):
                    wg = st.get("gpu_us", 0) / 1e6
                    wc = st.get("cpu_us", 0) / 1e6
                    busy = (" busy_g=%.0ffps/%.0f%% busy_c=%.0ffps/%.0f%%"
                            % (st.get("pics", 0) / wg if wg else 0,
                               100 * wg / wall if wall else 0,
                               st.get("pkts", 0) / wc if wc else 0,
                               100 * wc / wall if wall else 0))
                print("  [%d] %-5s %-7s wall=%.2fs segs=%d nvdec%%=%s%s"
                      % (i, case, be, wall, r["segs"],
                         nu.get("p50"), busy), flush=True)
            rep[case][be] = runs
        # 汇总：热轮中位
        row = []
        for be in backends:
            walls = sorted(x["wall"] for x in rep[case][be])
            if walls:
                row.append((be, walls[len(walls) // 2]))
        if len(row) == 2 and row[0][1]:
            print("== %s: %s %.2fs vs %s %.2fs → %+.1f%%" % (
                case, row[0][0], row[0][1], row[1][0], row[1][1],
                (row[1][1] - row[0][1]) / row[0][1] * 100))
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("落盘 %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
