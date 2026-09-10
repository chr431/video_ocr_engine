"""S0 性能基线录制（v2 ARCHITECTURE.md §11 S0 / D10）。

四配置 × 3 轮（同进程顺序跑，第 1 轮含冷启动 init、2–3 轮池热），
记录墙钟 + timing 分解 + 轮间散布。散布即 D10 双档门禁的**噪声底**依据
（PR 硬失败阈值 5% 的初值由此校准）。

用法：python tests/golden/bench_baseline.py   # 须在 GPU 空闲时跑
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VIDS = {"test5": r"D:\Videos\racelog_test\test5.mp4",
        "test6_av1": r"D:\Videos\racelog_test\test6.mp4",
        "test6_hevc": r"D:\Videos\racelog_test\test6_hevc.mp4"}
ROI = {"test5": (843, 993, 948, 1025), "test6": (841, 994, 949, 1026)}
WIN = (0, 3000)

CONFIGS = {  # §12 三配置展开为 4：h264 gpu/cpu + hevc nvdec + av1 nvdec
    "h264-gpu":   dict(video="test5", decode_backend="nvdec"),
    "h264-cpu":   dict(video="test5", decode_backend="cpu"),
    "hevc-nvdec": dict(video="test6_hevc", decode_backend="nvdec"),
    "av1-nvdec":  dict(video="test6_av1", decode_backend="nvdec"),
}
ROUNDS = 3


def one_round(cfg: dict) -> dict:
    from video_ocr_engine import FieldExtractor
    ex = FieldExtractor(VIDS[cfg["video"]], ROI["test5" if cfg["video"] == "test5"
                                                  else "test6"],
                        frame_start=WIN[0], frame_end=WIN[1],
                        decode_backend=cfg["decode_backend"],
                        ocr_backend="tensorrt", keep_crops=False)
    t0 = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t0
    return {"wall": round(wall, 3), "n_segments": len(r.segments),
            "timing": {k: round(v, 3) for k, v in r.timing.items()}}


def main() -> int:
    out = {"environment": {
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                 capture_output=True, text=True).stdout.strip(),
        "note": "GPU 须空闲；3000 帧窗口；TRT + GPU_PIPELINE=1"},
           "window": list(WIN), "rounds": ROUNDS, "configs": {}}
    noise = []
    for name, cfg in CONFIGS.items():
        rounds = [one_round(cfg) for _ in range(ROUNDS)]
        walls = [r["wall"] for r in rounds]
        med = statistics.median(walls[1:])          # 热轮中位（第 1 轮含冷启动 init）
        spread = (max(walls[1:]) - min(walls[1:])) / med * 100  # 噪声底=热轮散布
        noise.append(spread)
        out["configs"][name] = {"rounds": rounds, "median_wall_warm": round(med, 3),
                                "cold_wall": walls[0],
                                "spread_warm_pct": round(spread, 2)}
        print("%-11s 热轮中位 %.3fs  冷轮 %.3fs  热轮散布 %.1f%%  段数 %d" % (
            name, med, walls[0], spread, rounds[-1]["n_segments"]))
    out["noise_floor_pct"] = round(max(noise), 2)
    print("噪声底（各配置热轮散布最大值）: %.1f%%  → D10 PR 硬失败阈值 5%% 的校准依据"
          % out["noise_floor_pct"])
    Path(__file__).parent.joinpath("bench_baseline.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
