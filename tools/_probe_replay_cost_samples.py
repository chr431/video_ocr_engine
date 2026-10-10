"""REPLAYS 重复次数 vs 中位数稳定性的实验采样（2026-10-10 测试减负轮）。

背景
----
`tests/config/test_telemetry_cost.py` 曾占 pytest 全套 94.6s/126.2s（75%），
根因不是重复次数太多，而是两个叠加 bug：`_median_cost` 循环硬编码 full
档（增量口径形同虚设）+ v10 把 39ms/次的全系统线程快照挂进每个
checkpoint（154ms/run，被前面那条失效守卫漏过）。双修复后单文件
1.18s，REPLAYS=300 白拿稳定性。

本探针在新口径下采集**逐次**重放样本，离线 bootstrap 回答：
REPLAYS 若将来要降，降到多少中位数增量的抖动仍远小于预算余量——
"减少重复次数不损害判定可靠性"的量化依据。

产出：bench/replay_cost_samples.json + 控制台 bootstrap 摘要。
口径与被测测试一致（fresh 实例、记录面=full−off 增量、采样面=绝对值）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

REPLAYS = 300
OUT = os.path.join(ROOT, "bench", "replay_cost_samples.json")


def _load_test_module():
    path = os.path.join(ROOT, "tests", "config", "test_telemetry_cost.py")
    spec = importlib.util.spec_from_file_location("ttelemetry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fresh_replays(mod, fn, tier, n) -> list[float]:
    mod._warm_env_fingerprint()
    fn(mod.Metrics(tier))                    # 预热（NVML 单例/字典建立）
    out = []
    for _ in range(n):
        m = mod.Metrics(tier)
        t = time.perf_counter()
        fn(m)
        out.append((time.perf_counter() - t) * 1000.0)
    return out


def _ckpts(m) -> None:
    for ph in ("open", "calibrate", "decode", "ocr"):
        m.checkpoint(ph)


def main() -> int:
    mod = _load_test_module()
    print("采集逐次重放样本（REPLAYS=%d；记录面 off/full + 采样面 full）..." % REPLAYS)
    t0 = time.perf_counter()
    samples = {
        "base_off": _fresh_replays(mod, mod._one_run_base, "off", REPLAYS),
        "base_full": _fresh_replays(mod, mod._one_run_base, "full", REPLAYS),
        "ckpt_full": _fresh_replays(mod, _ckpts, "full", REPLAYS),
    }
    dt = time.perf_counter() - t0
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"replays": REPLAYS, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                   **samples}, f, indent=1)
    print("采集 %.1fs → %s" % (dt, OUT))

    rng = random.Random(42)
    off, full, ck = (samples[k] for k in ("base_off", "base_full", "ckpt_full"))
    point = statistics.median(full) - statistics.median(off)
    print("\n记录面点估计增量 = %.3fms（full %.3f − off %.3f），预算余量 %.3fms"
          % (point, statistics.median(full), statistics.median(off), 2.0 - point))
    print("采样面点估计 = %.3fms，预算余量 %.3fms" % (statistics.median(ck), 2.0 - statistics.median(ck)))
    print("\nbootstrap（各档独立重采样 %d 次）：n → 增量 p05/p50/p95、P(误报)" % 20000)
    for n in (30, 50, 100, 200, 300):
        deltas, ckpt = [], []
        for _ in range(20000):
            o = statistics.median(rng.choices(off, k=n))
            f = statistics.median(rng.choices(full, k=n))
            deltas.append(f - o)
            ckpt.append(statistics.median(rng.choices(ck, k=n)))
        deltas.sort(); ckpt.sort()
        flap = sum(1 for d in deltas if d > 2.0) / 20000
        flap_ck = sum(1 for d in ckpt if d > 2.0) / 20000
        print("  n=%-4d %.3f / %.3f / %.3f ms  P(记录面误报)=%.5f  P(采样面误报)=%.5f"
              % (n, deltas[int(0.05 * 20000)], deltas[10000], deltas[int(0.95 * 20000)],
                 flap, flap_ck))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
