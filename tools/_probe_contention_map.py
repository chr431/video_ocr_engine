"""争用定位账本（L1 探针，2026-10-09 争用定位轮）。

方法（三层里的第一层，纯消费 full 档报告 v8/v10，零产品侵入）：
  差分归因——同一逻辑工作在孤立臂 vs 并发臂下的**单位成本**对比，
  膨胀者=受害者，与其重叠的相位/外来簇=加害方；duty×等待判别
  等待性质（duty 高+等待长=供给侧忙等；duty 低+等待长=干净阻塞）。

`--hybrid-gap`（本轮问题）：hybrid 解码产出速率 < 双臂理论和的缺口分解。
  三臂（cpu / nvdec / hybrid）× reps，hevc（hybrid 优选码，C-53）：
    理论和 = rate(cpu 臂) + rate(nvdec 臂)
    实际 = rate(hybrid 臂)，fork hybrid_stats 给出臂内实际分率
            （frames_c/frames_g ÷ decode 相位墙钟）
    缺口分解 = CPU 臂损耗 + NVDEC 臂损耗 + 合并侧天花板（consumer
            duty 饱和=结构性上限）+ 争用税（各执行器 cycles/单位膨胀）
  外来簇（thr_foreign，v10）：按"解码线程数随 DECODE_THREADS 旋钮变"
  的差分可辨识性归簇（本探针只报簇总量与每帧成本，跨臂对比定损耗）。

用法：
  python tools/_probe_contention_map.py --hybrid-gap [--reps 3 --window 3000]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _extract(vid: str, roi, backend: str, frames: int):
    from video_ocr_engine import FieldExtractor
    import os as _os
    ex = FieldExtractor(vid, roi, frame_start=0, frame_end=frames,
                        decode_backend=backend,
                        ocr_backend=_os.environ.get("PROBE_OCR", "tensorrt"))
    t0 = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t0
    return wall, r


def _one_arm(vid: str, roi, backend: str, frames: int, reps: int) -> dict:
    """跑一个臂（预热 + reps），聚合 full 档报告的定位原料。"""
    walls, rows = [], []
    for i in range(reps + 1):
        wall, r = _extract(vid, roi, backend, frames)
        if i == 0:
            continue   # 预热（引擎池/上下文冷启动不进测量）
        walls.append(wall)
        rep = r.meta.get("report") or {}
        rows.append(rep)
    segs = statistics.median([rep["pipeline"]["n_segments"] for rep in rows])
    agg = dict(
        backend=backend,
        e2e_ms=statistics.median(walls) * 1000,
        segs=segs,
        reps=rows,
    )
    return agg


def _dec_rate(arm: dict, frames: int) -> tuple[float, dict]:
    """decode 相位产出速率（fps）与该相位行（duty/周期/外来簇）。

    逐 rep 墙钟取中位（h264 hybrid 方差大时，单 rep 的 fork 计数与
    中位墙混算会自相矛盾——逐 rep 全打印，聚合只对同源量取中位）。
    """
    walls, rows = [], []
    for rep in arm["reps"]:
        row = (rep.get("resources") or {}).get("per_phase", {}).get("decode")
        if row:
            walls.append(row["wall"])
            rows.append(row)
    wall = statistics.median(walls)
    rate = frames / wall
    return rate, rows[len(rows) // 2]


def _hybrid_reps(arm: dict) -> list:
    """hybrid 臂逐 rep 的 fork 供给计数（帧分率/速率 EWMA/饥饿计数）。"""
    out = []
    for rep in arm["reps"]:
        h = rep.get("hybrid") or {}
        if h:
            out.append(h)
    return out


def _fmt_cyc(v: float) -> str:
    return "%.2fG" % (v / 1e9) if v >= 1e8 else "%.1fM" % (v / 1e6)


def hybrid_gap(args) -> int:
    base = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
    vids = {   # test6 同内容三码版（编码对照必须用这套，bench CONFIGS 同源）
        "hevc": ("test6_hevc.mp4", (841, 994, 949, 1026)),
        "h264": ("test6_h264.mp4", (841, 994, 949, 1026)),
        "av1": ("test6.mp4", (841, 994, 949, 1026)),
    }
    fname, roi = vids[args.video]
    vid = os.path.join(base, fname)
    frames = args.window

    print("== 三臂采样（full 档；预热后 %d reps 中位；%s %d 帧 =="
          % (args.reps, vid, frames))
    arms = {}
    for backend in ("nvdec", "cpu", "hybrid"):
        arms[backend] = _one_arm(vid, roi, backend, frames, args.reps)
        print("  %-6s e2e %.0fms 段数 %.0f"
              % (backend, arms[backend]["e2e_ms"], arms[backend]["segs"]))

    r_nv, row_nv = _dec_rate(arms["nvdec"], frames)
    r_cp, row_cp = _dec_rate(arms["cpu"], frames)
    r_hy, row_hy = _dec_rate(arms["hybrid"], frames)
    theory = r_nv + r_cp
    gap = (1 - r_hy / theory) * 100
    hy_wall = frames / r_hy

    # fork hybrid_stats：逐 rep 打印（h264 方差大，单 rep 会误导）
    hreps = _hybrid_reps(arms["hybrid"])

    print("\n== 缺口分解 ==")
    print("纯臂速率：nvdec %.0ffps / cpu %.0ffps → 理论和 %.0ffps"
          % (r_nv, r_cp, theory))
    print("hybrid 实际 %.0ffps（decode 相位 %.2fs）→ 缺口 −%.1f%%"
          % (r_hy, hy_wall, gap))
    cpu_share = []
    for i, h in enumerate(hreps):
        fc, fg = h.get("frames_c", 0), h.get("frames_g", 0)
        wall = frames / r_hy   # 帧分率按中位墙折算仅作展示，逐 rep 见下
        print("  rep%d frames_c=%d frames_g=%d assigned_c=%s assigned_g=%s "
              "rc_now=%.0f rg_now=%.0f up_cempty=%d"
              % (i, fc, fg, h.get("assigned_c"), h.get("assigned_g"),
                 h.get("rc_now", 0), h.get("rg_now", 0),
                 h.get("up_cempty", 0)))
        if fc + fg:
            cpu_share.append(fc / (fc + fg))
    if cpu_share:
        share = statistics.median(cpu_share)
        fc_med = share * frames
        fg_med = frames - fc_med
        print("  臂内分率（中位）：CPU 臂 %.0ffps（%+.0f%% vs 纯臂 %.0f）/"
              " GPU 臂 %.0ffps（%+.0f%% vs 纯臂 %.0f）"
              % (fc_med / hy_wall, (fc_med / hy_wall / r_cp - 1) * 100, r_cp,
                 fg_med / hy_wall, (fg_med / hy_wall / r_nv - 1) * 100, r_nv))
    hs = hreps[len(hreps) // 2] if hreps else {}
    for k in ("up_cempty", "up_nobuf", "hol_ev_c", "hol_us_c", "kicks_c",
              "kicks_g", "clones", "strag", "late", "cache_peak_mb",
              "gpu_arm_stall"):
        if k in hs:
            print("    %-14s %s" % (k, hs[k]))

    print("\n== 逐执行器定位表（decode 相位；单位成本=cycles/帧或/crop）==")
    hdr = "%-14s %10s %10s %10s" % ("执行器", "nvdec", "cpu", "hybrid")
    print(hdr)

    def cell(row: dict, key: str, denom_frames: int) -> str:
        thr = row.get("thr") or {}
        e2e_thr = row.get("thr") or {}
        v = thr.get(key) or e2e_thr.get(key)
        return _fmt_cyc(v / denom_frames) if v else "—"

    segs = arms["hybrid"]["segs"]
    for key, label in (("consumer", "consumer/帧"),
                       ("ocr", "ocr/crop"), ("infer0", "infer0/批")):
        print("%-14s %10s %10s %10s"
              % (label,
                 cell(row_nv, key, frames), cell(row_cp, key, frames),
                 cell(row_hy, key, frames)))
    # 外来簇（decord 池 + OMP/TBB）：总量与每帧
    for name, row in (("nvdec", row_nv), ("cpu", row_cp), ("hybrid", row_hy)):
        pass   # 下方统一打印

    def foreign(row: dict) -> tuple[int, float]:
        d = row.get("thr_foreign") or {}
        return len(d), sum(d.values())

    print("%-14s %10s %10s %10s  (簇数)" % (
        "外来簇/帧",
        "%s(%d)" % (_fmt_cyc(foreign(row_nv)[1] / frames), foreign(row_nv)[0]),
        "%s(%d)" % (_fmt_cyc(foreign(row_cp)[1] / frames), foreign(row_cp)[0]),
        "%s(%d)" % (_fmt_cyc(foreign(row_hy)[1] / frames), foreign(row_hy)[0])))

    print("\n== duty×等待判别（decode 相位）==")
    for name, row in (("nvdec", row_nv), ("cpu", row_cp), ("hybrid", row_hy)):
        duty = row.get("thr_duty") or {}
        print("  %-6s duty=%s cores=%.1f 墙 %.2fs"
              % (name,
                 {k: round(v, 2) for k, v in sorted(duty.items())},
                 row.get("cores_avg_cycles", -1), row["wall"]))
    for name, arm in (("nvdec", arms["nvdec"]), ("cpu", arms["cpu"]),
                      ("hybrid", arms["hybrid"])):
        waits = []
        for rep in arm["reps"]:
            g = rep.get("gauges", {}).get("pipeline.q_get_wait")
            if g:
                waits.append(g)
        if waits:
            print("  %-6s consumer 等帧中位 %.3fs" % (name,
                                                     statistics.median(waits)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-gap", action="store_true", default=True)
    ap.add_argument("--video", default="hevc", choices=("hevc", "h264", "av1"),
                    help="test6 同内容三码版")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--window", type=int, default=3000)
    args = ap.parse_args()
    os.environ.setdefault("VOE_TELEMETRY", "full")
    return hybrid_gap(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
