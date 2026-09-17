"""P4 trace/timeline 实验：采集 + 视图（文本甘特/重叠矩阵/空洞清单）。

两种用法：
  python tools/_probe_trace_view.py --video test5 --decode cpu --ocr cpu
      跑一次 extract（VOE_TRACE_FILE=bench/trace_<tag>.json）并就地分析
  python tools/_probe_trace_view.py --trace bench/trace_xxx.json
      只分析已有时间线

输出五块（全部来自逐事件时间线，聚合报告给不出的口径）：
  1. 线程摘要（事件数/忙时/按 sum 的 top 事件）
  2. 文本甘特图（线程 × 时间桶，密度字符；NVML sm 时钟并行一排）
  3. 线程重叠矩阵（两两同时忙的时长占比——并发结构的直接读数）
  4. 空洞清单（每线程最大空闲段——"谁在等谁"的起点）
  5. 成本：trace on/off 两次墙钟对照 + 事件量 + 文件体积
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VID = {"test5": ("test5.mp4", (843, 993, 948, 1025)),
       "test6_hevc": ("test6_hevc.mp4", (841, 994, 949, 1026))}
BENCH = Path(__file__).resolve().parents[1] / "bench"


def _merge(intervals: list) -> list:
    """区间并集（忙时不受嵌套/重叠双计影响）。"""
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def analyze(d: dict) -> None:
    evs = d["events"]
    wall = d.get("wall_s") or (evs[-1][2] if evs else 0.0)
    names = [e[0] for e in evs]
    print("wall %.4fs  事件 %d  键 %d  meta.backend=%s" % (
        wall, len(evs), len(set(names)),
        (d.get("meta") or {}).get("backend")))
    by_thread: dict = {}
    for n, t0, t1, tid in evs:
        by_thread.setdefault(tid, []).append((n, t0, t1))
    # 1) 线程摘要
    print("\n── 线程摘要 ──")
    thread_busy = {}
    for tid, rows in by_thread.items():
        busy = sum(b - a for a, b in _merge([(t0, t1) for _, t0, t1 in rows]))
        thread_busy[tid] = busy
        byname: dict = {}
        for n, t0, t1 in rows:
            s, c = byname.get(n, (0.0, 0))
            byname[n] = (s + (t1 - t0), c + 1)
        top = sorted(byname.items(), key=lambda kv: -kv[1][0])[:3]
        print("  tid %-10s 事件 %5d  忙时 %7.3fs（wall 的 %5.1f%%）  %s"
              % (tid, len(rows), busy, busy / wall * 100,
                 "; ".join("%s×%d=%.3fs" % (n, c, s) for n, (s, c) in top)))
    # 2) 文本甘特（60 桶）
    COLS = 60
    print("\n── 甘特图（%d 桶，█>75%% ▓>50%% ▒>25%% ░>0 ·空闲）──" % COLS)
    ramp = "█▓▒░"
    tids = sorted(by_thread, key=lambda t: -thread_busy[t])
    for tid in tids[:8]:
        rows = by_thread[tid]
        line = []
        for c in range(COLS):
            a, b = wall * c / COLS, wall * (c + 1) / COLS
            ov = sum(min(e, b) - max(s, a) for s, e in
                     _merge([(t0, t1) for _, t0, t1 in rows])
                     if min(e, b) > max(s, a))
            f = ov / (wall / COLS)
            line.append("·" if f <= 0 else
                        ramp[3] if f <= 0.25 else
                        ramp[2] if f <= 0.5 else
                        ramp[1] if f <= 0.75 else ramp[0])
        print("  tid %-10s %s  忙%.1f%%" % (tid, "".join(line),
                                            thread_busy[tid] / wall * 100))
    nvml = d.get("nvml_points") or []
    if nvml:
        line = []
        for c in range(COLS):
            a, b = wall * c / COLS, wall * (c + 1) / COLS
            pts = [p for p in nvml if a <= p[0] < b and p[4] is not None]
            sm = (min(p[4] for p in pts) if pts else None)
            line.append("·" if sm is None else
                        "%X" % min(15, int(sm / 3200 * 16)))
        sms = [p[4] for p in nvml if p[4] is not None]
        bad = sum(1 for p in nvml if (p[7] or 0) & 0x1EC)
        print("  sm_clock(MHz)   %s  min=%d max=%d 压频tick=%d"
              % ("".join(line), min(sms), max(sms), bad))
    # 3) 重叠矩阵
    print("\n── 线程重叠矩阵（同时忙时长 / wall）──")
    tlist = tids[:8]
    print("        " + "".join("%-9s" % str(t)[-9:] for t in tlist))
    for ta in tlist:
        ua = _merge([(t0, t1) for _, t0, t1 in by_thread[ta]])
        row = []
        for tb in tlist:
            if ta == tb:
                row.append("   —    ")
                continue
            ub = _merge([(t0, t1) for _, t0, t1 in by_thread[tb]])
            ov = 0.0
            i = j = 0
            while i < len(ua) and j < len(ub):
                lo = max(ua[i][0], ub[j][0])
                hi = min(ua[i][1], ub[j][1])
                if hi > lo:
                    ov += hi - lo
                if ua[i][1] < ub[j][1]:
                    i += 1
                else:
                    j += 1
            row.append("%7.1f%%" % (ov / wall * 100))
        print("  %-5s %s" % (str(ta)[-5:], "".join(r.ljust(9) for r in row)))
    # 4) 空洞清单（每线程最大 3 段空闲；只看事件覆盖 ≥ wall 90% 的线程）
    print("\n── 空洞清单（空闲 ≥ 50ms 才列）──")
    for tid in tids[:8]:
        cov = _merge([(t0, t1) for _, t0, t1 in by_thread[tid]])
        busy = thread_busy[tid]
        gaps = []
        prev = 0.0
        for a, b in cov:
            if a - prev >= 0.05:
                gaps.append((a - prev, prev))
            prev = b
        if wall - prev >= 0.05:
            gaps.append((wall - prev, prev))
        if busy / wall < 0.10 or not gaps:
            continue
        gaps.sort(reverse=True)
        print("  tid %-10s 最大空洞: %s" % (
            tid, "; ".join("%.0fms@%.2fs" % (g * 1e3, p)
                           for g, p in gaps[:3])))


def run_and_trace(args) -> Path:
    from video_ocr_engine import FieldExtractor
    name, roi = VID[args.video]
    path = str(Path(os.environ.get("RACELOG_VIDEO_DIR",
                                   r"D:\Videos\racelog_test")) / name)
    tag = "%s_%s%s" % (args.video, args.decode,
                       "_gpu" if args.decode == "nvdec" else "")
    out = BENCH / ("trace_%s.json" % tag)
    # 冷轮（建引擎）+ 热轮（带 trace）+ 无 trace 对照轮
    ex = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                        decode_backend=args.decode, ocr_backend=args.ocr,
                        keep_crops=False)
    ex.extract()
    os.environ["VOE_TRACE_FILE"] = str(out)
    ex2 = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                         decode_backend=args.decode, ocr_backend=args.ocr,
                         keep_crops=False)
    t0 = time.perf_counter()
    r2 = ex2.extract()
    wall_on = time.perf_counter() - t0
    n2 = len(r2.segments)
    os.environ.pop("VOE_TRACE_FILE")
    ex3 = FieldExtractor(path, roi, frame_start=0, frame_end=args.frames,
                         decode_backend=args.decode, ocr_backend=args.ocr,
                         keep_crops=False)
    t0 = time.perf_counter()
    r3 = ex3.extract()
    wall_off = time.perf_counter() - t0
    n3 = len(r3.segments)
    d = json.loads(out.read_text(encoding="utf-8"))
    print("成本对照（热轮）：trace on %.4fs / off %.4fs（Δ%+.2f%%）  "
          "段数 %d/%d %s  事件 %d  文件 %.0f KB"
          % (wall_on, wall_off, (wall_on - wall_off) / wall_off * 100,
             n2, n3, "一致" if n2 == n3 else "不一致!", d["n_events"],
             out.stat().st_size / 1024))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="", help="分析已有时间线 JSON")
    ap.add_argument("--video", default="test5")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--decode", default="cpu")
    ap.add_argument("--ocr", default="cpu")
    args = ap.parse_args()
    if args.trace:
        src = Path(args.trace)
    else:
        src = run_and_trace(args)
    analyze(json.loads(src.read_text(encoding="utf-8")))
    print("\n→ %s" % src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
