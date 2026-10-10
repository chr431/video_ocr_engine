"""telemetry-check A/A 噪声带 vs (rounds, inner) 的地板曲线实验采样。

动机（2026-10-10 测试减负轮）
------------------------------
telemetry-check 一次 = rounds × 2 档 × inner 次引擎 run（默认 25×2×3 =
150 次）。判失败用均值 + 3×SE，SE∝1/√n——重复次数是判定可靠性的数学
约束，但 **inner（槽内复制压噪）与 rounds（配对轮数）对 SE 的贡献不同**，
谁更便宜要用数据说。本探针在同一机器状态顺序采两组 A/A 原始配对样本：

  A. inner=3, rounds=25（现行默认口径，150 次 run）
  B. inner=1, rounds=25（50 次 run，3× 便宜）

离线 bootstrap 给出 floor(n) 曲线（阈值 = |均值|+3×SE 上取整 0.05），
回答：inner 3→1 换来 3× 时长，地板涨多少？是否仍 ≪ full 档成本门限
（PI15_LIMITS.full_pct=1.20%）？

产出：bench/aa_floor_curve.json（原始 per-round 墙钟）+ 摘要。
标定期间机器独占（rules.yaml：标定/实验禁止并行负载）。
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from bench import CONFIGS, _band, _inproc_rounds, _limit_from_band, _paired  # noqa: E402

OUT = os.path.join(ROOT, "bench", "aa_floor_curve.json")
CFG, WINDOW, ROUNDS, OCR = "h264-gpu", 3000, 25, "tensorrt"


def collect(inner: int) -> dict:
    print("── inner=%d rounds=%d（%d 次引擎 run）──" % (inner, ROUNDS, ROUNDS * 2 * inner),
          flush=True)
    t0 = time.perf_counter()
    per = _inproc_rounds(CFG, WINDOW, ["full", "full2"], ROUNDS, OCR, inner=inner)
    per = {k: v for k, v in per.items() if k[0] >= 0}
    d = _paired(per, ROUNDS, "full", "full2")
    b = _band(d)
    print("   %.0fs | 均值 %+.3f%% sd %.3f%% SE %.3f%% → floor %.2f%%"
          % (time.perf_counter() - t0, b["mean"], b["sd"], b["se"], _limit_from_band(b)))
    return {"inner": inner, "rounds": ROUNDS, "wall_by_round": {
        "%d|%s" % k: v for k, v in sorted(per.items())}}


def bootstrap_floor(d: list[float], ns=(10, 15, 20, 25), k=20000, seed=7) -> dict:
    rng = random.Random(seed)
    out = {}
    for n in ns:
        floors = []
        for _ in range(k):
            s = rng.choices(d, k=n)
            m = statistics.fmean(s)
            se = statistics.stdev(s) / (n ** 0.5)
            cand = abs(m) + 3 * se
            floors.append(max(0.20, -(-cand // 0.05) * 0.05))
        floors.sort()
        out[n] = {"p05": floors[int(0.05 * k)], "p50": floors[k // 2],
                  "p95": floors[int(0.95 * k)]}
    return out


def main() -> int:
    results = []
    for inner in (3, 1):
        r = collect(inner)
        # 从 wall_by_round 重建 (round, tier) 键 → 配对差分
        per = {}
        for k, v in r["wall_by_round"].items():
            rnd, tier = k.split("|")
            per[(int(rnd), tier)] = v
        d = _paired(per, r["rounds"], "full", "full2")
        r["paired_diff_pct"] = d
        r["floor_p50"] = _limit_from_band(_band(d))
        r["bootstrap"] = bootstrap_floor(d)
        results.append(r)

    print("\nfloor(n) 曲线（|均值|+3×SE，上取整 0.05%，下限 0.20%）：")
    for r in results:
        print("  inner=%d：" % r["inner"])
        for n, q in r["bootstrap"].items():
            print("    n=%-3d p05 %.2f%%  p50 %.2f%%  p95 %.2f%%" % (n, q["p05"], q["p50"], q["p95"]))
    print("\n门限参照：PI15 full_pct = 1.20%%（遥测成本门禁）；floor ≪ 1.20 即不损害该门禁。")

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"cfg": CFG, "window": WINDOW, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "results": results}, f, indent=1)
    print("→ %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
