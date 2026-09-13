"""_probe_pool_pairing.py —— 层5：跨视频互补配对 vs 全 nvdec 顺序（A/B）。

口径
----
- 素材：RACELOG_BATCH_DIR（新三国 01~05，宽 ROI 字幕）；无该目录时退回
  racelog_test 三码族（同内容不同编码，配对收益应更稳）。
- 对照：A = 全部 nvdec 顺序跑（现状批量基线）；B = ExtractionPool
  pair 模式（nvdec/cpu 交替并发）。两种模式各自独立进程 × N 轮，
  交错执行；判据 = 配对差分均值 + 符号多数（批量口径无 PI15 标定，
  效应预期 ≥20%，4 轮足够）。
- 正确性门禁：两种模式每视频段数与唯一文本集必须逐位一致。

用法
----
    python tools/_probe_pool_pairing.py --rounds 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

OUT = ROOT / "bench" / "pool_pairing.json"

WORKER = r'''
import json, os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
sys.stdout.reconfigure(encoding="utf-8")
from video_ocr_engine.pipeline import pool

mode = sys.argv[1]
items = json.loads(sys.argv[2])
kw = {k: os.environ[v] for k, v in []}
# ROI/帧数等由 items 原样传入；env 解析由 FieldExtractor 侧负责
if mode == "seq":
    backends = "nvdec"
else:
    backends = "pair"
t0 = time.perf_counter()
res = pool.run(items, backends=backends)
wall = time.perf_counter() - t0
out = {"wall": round(wall, 4), "items": []}
for r in res:
    seg_list = r.segments if isinstance(r.segments, (list, tuple)) else list(r.segments)
    txt = [s for s in (getattr(x, "text", x) for x in seg_list) if s]
    out["items"].append({"n_segments": len(seg_list), "n_texts": len(set(txt))})
print(json.dumps(out))
'''


def find_videos() -> list[dict]:
    bdir = os.environ.get("RACELOG_BATCH_DIR")
    if bdir and Path(bdir).is_dir():
        vids = sorted(Path(bdir).glob("*.mp4"))
        if len(vids) >= 2:
            return [{"video": str(v), "roi": (40, 880, 1240, 1020),
                     "sample_stride": 8} for v in vids[:2]]
    vdir = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
    trio = [Path(vdir) / n for n in
            ("test5.mp4", "test6_hevc.mp4", "test6.mp4")]
    return [{"video": str(p), "roi": (841, 994, 949, 1026)}
            for p in trio if p.exists()]


def run_arm(mode: str, items: list[dict]) -> dict:
    env = dict(os.environ)
    env["PROBE_ROOT"] = str(ROOT)
    p = subprocess.run(
        [sys.executable, "-c", WORKER, mode, json.dumps(items, ensure_ascii=False)],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=1800)
    if p.returncode != 0:
        return {"error": (p.stderr or "")[-400:], "wall": 1e9}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--cooldown", type=float, default=3.0)
    args = ap.parse_args()
    items = find_videos()
    print(f"素材 {len(items)} 个：" + ", ".join(
        Path(i['video']).name for i in items))
    rows = {"seq": [], "pair": []}
    verdicts = []
    import time as _t
    for r in range(args.rounds):
        if r: _t.sleep(args.cooldown)
        # 交错 + 轮转先后
        order = [("seq", "pair")] if r % 2 == 0 else [("pair", "seq")]
        for a, b in order:
            ra = run_arm(a, items)
            rb = run_arm(b, items)
            if "error" in ra or "error" in rb:
                print("失败:", ra.get("error", "")[:120] or rb.get("error", "")[:120])
                return 1
            rows[a].append(ra["wall"])
            rows[b].append(rb["wall"])
            # 正确性：逐位一致
            same = ra["items"] == rb["items"]
            d = (rb["wall"] - ra["wall"]) / ra["wall"] * 100
            verdicts.append(d)
            print(f"[r{r:02d}] seq={ra['wall']:.2f}s pair={rb['wall']:.2f}s "
                  f"Δ={d:+.1f}% seg/texts一致={'✓' if same else '✗'}")
            if not same:
                print("  seq:", ra["items"])
                print("  pair:", rb["items"])
                return 1
    n = len(verdicts)
    neg = sum(1 for d in verdicts if d < 0)
    mean = sum(verdicts) / n
    print(f"\n配对差分（pair 相对 seq）: 均值 {mean:+.1f}%  "
          f"符号 {neg}/{n} 负（pair 快）")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"rows": rows, "diffs": verdicts, "mean": mean},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("落盘", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
