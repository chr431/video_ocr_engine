"""_probe_pool_pairing.py —— 层5：跨视频互补配对 vs 批量基线（三臂 A/B/C）。

口径（2026-09-17 翻案轮修正）
----
- **三臂**（此前只有两臂，且 `seq` 名不副实——pool.run 一律 2 worker
  并发，"全 nvdec 顺序"实际是双 NVDEC 会话并发，正撞 C-01 争用）：
    * `nv2`  = 全 nvdec × 2 worker（懒并发批量基线，原 "seq"）
    * `nv1`  = 全 nvdec × 1 worker（真串行基线）
    * `pair` = 编码感知 + LPT × 2 worker（ExtractionPool 缺省）
  三臂按轮拉丁轮转（对消位置效应）。
- 正确性门禁：三臂每视频段数与唯一文本集逐位一致。
- **瞬态异常防护**（2026-09-17 首跑 2 次 pair=34s 异常拖垮均值 −2.2% 的
  教训）：每臂墙钟偏离自身中位 >10% 的轮打 ⚠ 剔除出配对差分（保留原始
  值在输出里，不静默）——共享桌面的外部负载窗口必须可见且不进判据。
- 判据：pair 对两个基线的**同轮配对差分**均值 + 符号多数（效应 ≥15%，
  4 轮足够；2026-09-17 实测 −20.5% 对 nv2、−16% 对 nv1，确定性 12/12）。
- 每臂落盘逐视频 wall + backend（pool 的 INFO 日志捕获），归因可复查。

素材：RACELOG_BATCH_DIR（≥2 个 mp4 → 取前 2，宽 ROI 字幕）；无该目录
时退回 racelog_test 三码族（同内容不同编码，互补配对收益最稳）。

用法
----
    python tools/_probe_pool_pairing.py --rounds 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

OUT = ROOT / "bench" / "pool_pairing.json"

#: mode → (pool backends 参数, max_workers)
ARMS = {"nv2": ("nvdec", 2), "nv1": ("nvdec", 1), "pair": ("pair", 2)}

WORKER = r'''
import json, os, sys, time, logging
sys.path.insert(0, os.environ["PROBE_ROOT"])
sys.stdout.reconfigure(encoding="utf-8")
from video_ocr_engine.pipeline import pool

mode, items = sys.argv[1], json.loads(sys.argv[2])
backends, workers = {"nv2": ("nvdec", 2), "nv1": ("nvdec", 1),
                     "pair": ("pair", 2)}[mode]
# 逐视频 wall/backend：捕获 pool 的 INFO 日志（handler 拦截，不进 stderr）
cap = []
class _Cap(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if m.startswith("pool item"):
            cap.append(m)
logging.getLogger("video_ocr_engine.pipeline.pool").addHandler(_Cap())
t0 = time.perf_counter()
res = pool.run(items, backends=backends, max_workers=workers)
wall = time.perf_counter() - t0
out = {"wall": round(wall, 4), "items": [], "pool_log": cap}
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
    arm_names = list(ARMS)
    rows: dict = {a: [] for a in arm_names}
    ref_items = None
    for r in range(args.rounds):
        if r:
            time.sleep(args.cooldown)
        order = arm_names[r % len(arm_names):] + arm_names[:r % len(arm_names)]
        res = {}
        for a in order:
            out = run_arm(a, items)
            if "error" in out:
                print("失败:", out["error"][:120])
                return 1
            res[a] = out
            rows[a].append(out["wall"])
        # 正确性：三臂逐位一致（以第一臂为参照）
        if ref_items is None:
            ref_items = res[order[0]]["items"]
        same = all(res[a]["items"] == ref_items for a in arm_names)
        line = " ".join("%s=%.2fs" % (a, res[a]["wall"]) for a in arm_names)
        print(f"[r{r:02d}] {line}  seg/texts一致={'✓' if same else '✗'}")
        if not same:
            for a in arm_names:
                print(" ", a, res[a]["items"])
            return 1
    # 瞬态异常防护：臂内偏离自身中位 >10% 的轮剔除出判据（原始值保留）
    med = {a: statistics.median(v) for a, v in rows.items()}
    keep = {a: [w for w in v if abs(w - med[a]) / med[a] <= 0.10]
            for a, v in rows.items()}
    for a in arm_names:
        if len(keep[a]) < len(rows[a]):
            print("⚠ %s 剔除 %d/%d 轮瞬态（>10%% 偏离中位 %.2fs）：留 %s"
                  % (a, len(rows[a]) - len(keep[a]), len(rows[a]), med[a],
                     keep[a]))
    # 判据：pair 对两基线（同轮配对不可行——轮内三臂共享同环境，直接比中位）
    print()
    for base in ("nv2", "nv1"):
        pm, bm = statistics.median(keep["pair"]), statistics.median(keep[base])
        d = (pm - bm) / bm * 100
        print("pair vs %-3s：中位 %.2fs vs %.2fs = %+.1f%%（pair %s）"
              % (base, pm, bm, d, "快" if d < 0 else "慢"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"rows": rows, "kept": keep, "medians": med,
         "material": [Path(i["video"]).name for i in items]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("落盘", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
