"""每相位资源读数（§8.6 r5 L1/L2 的消费者，诊断用）。

回答过去只能靠猜的问题："这个相位到底吃了多少核 / 多少磁盘 / 多少显存？"
典型用途：**hybrid 的引擎侧损耗到底是 CPU 争用还是解码器调度**——看 hybrid
与 cpu/nvdec 配置在 decode 相位的 `cores_avg` 对比即可判定。

每配置跑 2 轮（冷/热），报**热轮**（冷轮含 TRT 反序列化，会污染差分）。

用法：
  python tools/_probe_phase_cores.py --tier std --configs h264-cpu,h264-hybrid
  python tools/_probe_phase_cores.py --tier full --configs h264-gpu   # 带 NVML
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
sys.path.insert(0, str(ROOT / "tools"))

from bench import CONFIGS, ROI, VIDS  # noqa: E402  复用同一配置矩阵

_PHASES = ("open", "calibrate", "decode", "ocr")


def one(cfg: str, tier: str, frames: int) -> dict:
    from video_ocr_engine import FieldExtractor
    c = CONFIGS[cfg]
    vid = c["video"]
    walls, runs = [], []
    for _ in range(2):
        ex = FieldExtractor(VIDS[vid], ROI["test5" if vid == "test5" else "test6"],
                            frame_start=0, frame_end=frames,
                            decode_backend=c["decode_backend"],
                            ocr_backend="tensorrt", keep_crops=False)
        # 分档在 RunConfig 解析时读取 → 必须每次 run 前设好
        os.environ["VOE_TELEMETRY"] = tier
        t = time.perf_counter()
        r = ex.extract()
        walls.append(time.perf_counter() - t)
        runs.append(r.meta["report"])
    hot = runs[1]
    out = {"cfg": cfg, "wall_hot": statistics.median(walls[1:]),
           "wall_cold": walls[0], "phases": hot["resources"]["per_phase"],
           "sources": hot["resources"]["sources"]}
    if "hardware" in hot:
        out["hardware"] = hot["hardware"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="std", choices=["std", "full"])
    ap.add_argument("--configs", default="h264-cpu,h264-hybrid,h264-gpu")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--cooldown", type=float, default=2.0)
    args = ap.parse_args()
    for cfg in args.configs.split(","):
        d = one(cfg.strip(), args.tier, args.frames)
        print("\n== %s (%s 档)  冷 %.3fs / 热 %.3fs" % (
            cfg, args.tier, d["wall_cold"], d["wall_hot"]))
        print("   %-10s %8s %8s %7s %9s %10s %8s" % (
            "相位", "墙钟s", "平均核数", "线程", "RSSΔMiB", "读MB/s", "显存MiB"))
        for p in _PHASES:
            v = d["phases"].get(p)
            if not v:
                continue
            print("   %-10s %8.4f %8.2f %7d %9.1f %10.2f %8s" % (
                p, v.get("wall", 0), v.get("cores_avg", 0),
                v.get("threads", 0), v.get("rss_delta_mib", 0),
                v.get("disk_read_mbps", 0),
                v.get("vram_used_mib", "n/a")))
        h = d.get("hardware")
        if h:
            g, nv = h.get("gpu_util_pct") or {}, h.get("nvdec_util_pct") or {}
            print("   L2[%s] n=%d 失败=%d GPU p50/p99=%s/%s  NVDEC p50/p99=%s/%s"
                  % (h["sources"], h.get("n", 0), h.get("sample_failures", 0),
                     g.get("p50"), g.get("p99"), nv.get("p50"), nv.get("p99")))
        time.sleep(args.cooldown)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
