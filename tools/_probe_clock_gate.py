"""测量学证据（2026-09-17 §8.2/§8.3 固化）：GPU 时钟爬坡与时钟门禁校准。

协议：GPU 路径（nvdec+tensorrt）同进程连跑 N 轮，每轮独立包一个
`NvmlSampler`（0.2s tick），逐轮记 wall / sm_clock min·p50 / 主动压频 tick 数。

产出三个数字，供 bench.py 时钟门禁落地引用：
1. 冷轮效应复测：含冷轮 CV vs 弃前 2 轮 CV（§8.2 实测 54× 收窄）。
2. 门禁比率 k 的分离度：valid = sm_min ≥ k×最大 SM 时钟（nvmlDeviceGetMaxClockInfo）
   且无主动压频位（sw_power_cap/hw_slowdown/热/功率刹车/display，即 ~0x1EC）。
   对 k∈[0.5,1.0] 扫描，找"冷轮全拒、热轮全留"的 k 区间。
3. 本机最大 SM 时钟与热态 sm_clock 水位（供日志留档）。

产物：bench/clock_gate.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parents[1] / "bench" / "clock_gate.json"

#: 主动压频位（resources.py THROTTLE_BITS 的"非常态"集合）：
#: gpu_idle(0x1)/apps_clocks_setting(0x2)/sync_boost(0x10) 为常态。
BAD_THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80 | 0x100


def _max_sm_clock() -> int | None:
    """nvmlDeviceGetMaxClockInfo(SM)；失败 None（门禁退化为不可用）。"""
    import ctypes
    from video_ocr_engine.domain.resources import nvml_handle
    try:
        nvml, h = nvml_handle()
        c = ctypes.c_uint()
        if int(nvml.nvmlDeviceGetMaxClockInfo(h, 1, ctypes.byref(c))) != 0:
            return None
        return int(c.value)
    except Exception:  # noqa: BLE001 无 NVML → 探针只报不可用
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--frames", type=int, default=3000)
    args = ap.parse_args()

    from video_ocr_engine import FieldExtractor
    from video_ocr_engine.domain.resources import NvmlSampler

    vid = Path(__import__("os").environ.get(
        "RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")) / "test5.mp4"
    roi = (843, 993, 948, 1025)

    sm_max = _max_sm_clock()
    print("最大 SM 时钟 = %s MHz" % sm_max)

    rows = []
    for i in range(args.rounds):
        ex = FieldExtractor(str(vid), roi, frame_start=0, frame_end=args.frames,
                            decode_backend="nvdec", ocr_backend="tensorrt",
                            keep_crops=False)
        s = NvmlSampler(interval_s=0.2)
        s.start()
        t0 = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t0
        s.stop()
        pts = [(p[4], p[7]) for p in s._pts if p[4] is not None]
        sm = sorted(p[0] for p in pts)
        bad = sum(1 for _, m in pts if (m or 0) & BAD_THROTTLE_MASK)
        row = {"round": i, "wall": round(wall, 4), "n_pts": len(pts),
               "sm_min": sm[0] if sm else None,
               "sm_p50": sm[len(sm) // 2] if sm else None,
               "throttle_bad": bad, "segments": len(r.segments)}
        rows.append(row)
        print("  round %d  wall %.4fs  sm min/p50 = %s/%s MHz  压频tick %d"
              % (i, wall, row["sm_min"], row["sm_p50"], bad))

    walls = [r["wall"] for r in rows]
    hot = walls[2:] if len(walls) > 2 else walls

    def cv(xs):
        return statistics.pstdev(xs) / statistics.fmean(xs) * 100 if xs else 0.0

    # k 扫描：valid 集合随 k 变化；目标 = 拒掉冷轮、保留热轮。
    sweep = []
    if sm_max:
        for k10 in range(5, 11):
            k = k10 / 10.0
            valid = [r["round"] for r in rows if r["sm_min"] is not None
                     and r["sm_min"] >= k * sm_max and r["throttle_bad"] == 0]
            hot_walls = [r["wall"] for r in rows if r["round"] in valid]
            sweep.append({"k": k, "valid_rounds": valid,
                          "n_valid": len(valid),
                          "cv_valid_pct": round(cv(hot_walls), 3) if hot_walls else None})
            print("  k=%.1f → valid=%s  CV(全程)=%.3f%%"
                  % (k, valid, sweep[-1]["cv_valid_pct"] or float("nan")))

    result = {"sm_max_mhz": sm_max, "rounds": rows,
              "cv_all_pct": round(cv(walls), 3),
              "cv_drop2_pct": round(cv(hot), 3),
              "narrow_factor": round(cv(walls) / cv(hot), 1) if cv(hot) else None,
              "k_sweep": sweep,
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("CV 含全部 %.3f%% → 弃前2轮 %.3f%%（×%.1f）"
          % (result["cv_all_pct"], result["cv_drop2_pct"],
             result["narrow_factor"] or 0))
    print("→ %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
