"""TRT profile batch 的端到端交错 A/B（S6 轮）。

`tools/_probe_trt_maxbatch.py` 量的是同一条 TRT 提交路径上的微基准（batch
6→18 快 16%）。本探针把它落到**真实管线**：同一个视频窗口、同一份代码，
只换引擎产物（`TrtEngine._engine_candidates` 指向指定 .engine），
A/B/A/B 交错跑，报告墙钟 + `ocr.infer` + 子批/同步计数。

**不往产品代码里加开关**：引擎选择在 worker 子进程里 monkeypatch
（`python -c` + PROBE_ENGINE 环境变量），跑完即消失。

用法：
  python tools/_probe_engine_ab.py --a <engineA> --b <engineB> \
      [--config h264-cpu] [--frames 3000] [--repeat 3]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDS = {"h264-cpu": ("test5.mp4", (843, 993, 948, 1025), "cpu"),
        "h264-gpu": ("test5.mp4", (843, 993, 948, 1025), "nvdec"),
        "hevc-nvdec": ("test6_hevc.mp4", (841, 994, 949, 1026), "nvdec")}

WORKER = r'''
import json, os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
from pathlib import Path
from video_ocr_engine.ocr.trt import TrtEngine
_eng = Path(os.environ["PROBE_ENGINE"])
TrtEngine._engine_candidates = staticmethod(lambda size: [_eng])
from video_ocr_engine import FieldExtractor
vid = os.path.join(os.environ["RACELOG_VIDEO_DIR"],
                   os.environ["PROBE_VIDEO"])
roi = tuple(int(x) for x in os.environ["PROBE_ROI"].split(","))
outs = []
for i in range(2):                      # 第 1 轮冷、第 2 轮热
    ex = FieldExtractor(vid, roi, frame_start=0,
                        frame_end=int(os.environ["PROBE_FRAMES"]),
                        decode_backend=os.environ["PROBE_DEC"],
                        ocr_backend="tensorrt", keep_crops=False)
    t = time.perf_counter()
    r = ex.extract()
    rep = r.meta.get("report") or {}
    outs.append({"wall": time.perf_counter() - t,
                 "segs": len(r.segments),
                 "infer": (rep.get("spans", {}).get("ocr.infer") or {}).get("sum"),
                 "init": (rep.get("gauges") or {}).get("ocr.engine_init"),
                 "chunks": (rep.get("counters") or {}).get("ocr.chunks"),
                 "sub": (rep.get("counters") or {}).get("ocr.sub_chunks")})
print("PROBE_JSON " + json.dumps(outs))
'''


def one(engine: str, cfg: str, frames: int) -> dict:
    vid, roi, dec = VIDS[cfg]
    env = dict(os.environ)
    env.update({"PROBE_ROOT": str(ROOT), "PROBE_ENGINE": engine,
                "PROBE_VIDEO": vid, "PROBE_ROI": ",".join(map(str, roi)),
                "PROBE_DEC": dec, "PROBE_FRAMES": str(frames),
                "RACELOG_VIDEO_DIR": env.get("RACELOG_VIDEO_DIR",
                                             r"D:\Videos\racelog_test")})
    p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    raise SystemExit("worker 失败：\n%s\n%s" % (p.stdout[-1500:], p.stderr[-1500:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--config", default="h264-cpu")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=8.0)
    args = ap.parse_args()
    cfgs = args.config.split(",")
    rows: dict = {c: {"A": [], "B": []} for c in cfgs}
    for i in range(args.repeat):
        for cfg in cfgs:
            for tag, eng in (("A", args.a), ("B", args.b)):
                out = one(eng, cfg, args.frames)
                rows[cfg][tag].append(out)
                print("  [%d] %-11s %s cold=%.3f hot=%.3f infer=%.3f sub=%s"
                      % (i, cfg, tag, out[0]["wall"], out[1]["wall"],
                         out[1]["infer"] or -1, out[1]["sub"]))
            time.sleep(args.cooldown)
    print("\n引擎 A=%s\n     B=%s" % (Path(args.a).name, Path(args.b).name))
    print("%-11s %10s %10s %9s  %s" % ("config", "A(hot)", "B(hot)", "Δ%",
                                       "逐对符号"))
    for cfg in cfgs:
        pairs = [(r["A"][1]["wall"], r["B"][1]["wall"]) for r in (
            [{k: rows[cfg][k][i] for k in ("A", "B")} for i in range(args.repeat)])]
        ma = statistics.median([p[0] for p in pairs])
        mb = statistics.median([p[1] for p in pairs])
        signs = "".join("+" if b > a else "-" for a, b in pairs)
        print("%-11s %10.4f %10.4f %+8.2f%%  %s"
              % (cfg, ma, mb, (mb - ma) / ma * 100, signs))
        ia = statistics.median([r["A"][1]["infer"] for r in (
            [{k: rows[cfg][k][i] for k in ("A", "B")} for i in range(args.repeat)])])
        ib = statistics.median([r["B"][1]["infer"] for r in (
            [{k: rows[cfg][k][i] for k in ("A", "B")} for i in range(args.repeat)])])
        print("%-11s %10.4f %10.4f %+8.2f%%   (ocr.infer)"
              % ("", ia, ib, (ib - ia) / ia * 100))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
