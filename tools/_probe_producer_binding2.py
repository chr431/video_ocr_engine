"""生产者侧 numpy 是否真在关键路径：**双侧**判定（加速 + 减速同表）。

背景（本轮实测）：
  · `_probe_patch_verify.py`：生产者侧 numpy 实际占墙钟 **24.7%**
    （`_segments_similar` 0.255s + `_cluster_win3` 0.182s + `_text_sep_binary` 0.088s）；
  · `_probe_numpy_replace_ab.py`：三者换成 cv2（逐位一致、内核快 29~57%）
    → **墙钟 +0.37%（符号 -++，噪声内）**，producer 也没降。

矛盾。C-42 的教训是"按占比推算收益是错的"，但方向反了也同样是陷阱：
**"加速了没收益"还不能断定不在关键路径**——可能是替换臂没生效（已用
`_probe_patch_verify.py` 排除）、也可能是被别的阻塞吸收。

**双侧判据**：把生产者 numpy 人为做 **N 倍冗余计算**（不改任何输出，
段数/指纹逐位不变），看墙钟是否随之上升。
  · 若 N 倍后墙钟**明显上升** → 该计算在关键路径，加速应当兑现；
  · 若 N 倍后墙钟**几乎不动** → 该计算被别的等待吸收，换库不可能提速
    （双侧都钝 = 铁证，比单侧"加速无收益"强）。

同时测 **decord get_batch 是否释放 GIL**（生产者 1.35s 都在它里面，
若它不释放 GIL，则同进程的 numpy/cv2 改动都会被 GIL 串行化）。

用法：
  python tools/_probe_producer_binding2.py [--config h264-cpu] \
      [--frames 3000] [--repeat 3] [--mult 5]
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

VIDS = {"h264-cpu": ("test5.mp4", (843, 993, 948, 1025), "cpu", "cpu"),
        "hevc-cpu": ("test6_hevc.mp4", (841, 994, 949, 1026), "cpu", "cpu"),
        "h264-gpu": ("test5.mp4", (843, 993, 948, 1025), "nvdec", "tensorrt")}

#: 冗余臂（生产者侧）：把生产者三个 numpy 内核各跑 N 次（结果用第 1 次，
#: 其余丢弃）——**输出逐位不变**，只是把该计算的成本放大 N 倍。
#: 这正是"零成本臂会改判据"问题的干净替代（零成本会让段数从 1090 变 3000，
#: 见 `_probe_producer_gap.py` 的 nocluster 教训）。
REDUNDANT = r'''
def _install_redundant(mult):
    import video_ocr_engine.domain.segmentation as seg
    import video_ocr_engine.extractor as ex_mod

    _orig_cluster = seg._cluster_win3
    _orig_sep = ex_mod._text_sep_binary

    def slow_cluster(diff):
        r = _orig_cluster(diff)
        for _ in range(mult - 1):
            _orig_cluster(diff)          # 冗余：结果丢弃
        return r

    def slow_sep(gray, th):
        r = _orig_sep(gray, th)
        for _ in range(mult - 1):
            _orig_sep(gray, th)
        return r

    seg._cluster_win3 = slow_cluster
    ex_mod._text_sep_binary = slow_sep

    # _segments_similar：整法包裹，额外重复 diff 数学（不改返回值）
    import numpy as np
    _orig_sim = ex_mod.FieldExtractor._segments_similar

    def slow_sim(self, a, b):
        r = _orig_sim(self, a, b)
        aa, bb = a, b
        for _ in range(mult - 1):
            try:
                d = np.abs(aa.astype(np.int16) - bb.astype(np.int16))
                _ = float(d.mean()), int(np.sum(d > 10))
            except Exception:
                break
        return r

    ex_mod.FieldExtractor._segments_similar = slow_sim
'''

#: 冗余臂（消费者侧）：重复 OCR 预处理（resize+gamma 归一化），
#: 结果用第 1 次 → 输出逐位不变。用于验证"消费者侧是否同样被吸收"
#: （与生产者侧对称，双侧都钝才是铁证）。
REDUNDANT_CONSUMER = r'''
def _install_redundant_consumer(mult):
    import video_ocr_engine.ocr.native as nat

    _orig_rn = nat.OcrEngine._resize_norm

    def slow_resize_norm(img, max_wh_ratio, height=48):
        r = _orig_rn(img, max_wh_ratio, height)
        for _ in range(mult - 1):
            _orig_rn(img, max_wh_ratio, height)
        return r

    nat.OcrEngine._resize_norm = staticmethod(slow_resize_norm)
'''

WORKER = r'''
import hashlib, json, os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
import numpy as np

''' + REDUNDANT + "\n" + REDUNDANT_CONSUMER + r'''

MULT = int(os.environ["PROBE_MULT"])
SIDE = os.environ.get("PROBE_SIDE", "producer")
if MULT > 1:
    if SIDE in ("producer", "both"):
        _install_redundant(MULT)
    if SIDE in ("consumer", "both"):
        _install_redundant_consumer(MULT)

from video_ocr_engine import FieldExtractor

vid = os.path.join(os.environ["RACELOG_VIDEO_DIR"], os.environ["PROBE_VIDEO"])
roi = tuple(int(x) for x in os.environ["PROBE_ROI"].split(","))
outs = []
for i in range(2):
    ex = FieldExtractor(vid, roi, frame_start=0,
                        frame_end=int(os.environ["PROBE_FRAMES"]),
                        decode_backend=os.environ["PROBE_DEC"],
                        ocr_backend=os.environ["PROBE_OCR"], keep_crops=False)
    t = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t
    rep = r.meta.get("report") or {}
    sp, ga = rep.get("spans", {}), rep.get("gauges", {})
    fp = hashlib.sha256(
        "\n".join("%s|%.5f" % (s.text, s.confidence) for s in r.segments
                  ).encode("utf-8")).hexdigest()[:16]
    outs.append({"wall": wall, "segs": len(r.segments), "fp": fp,
                 "producer": (sp.get("pipeline.consumer") or {}).get("sum"),
                 "decode_batch": (sp.get("decode.batch") or {}).get("sum"),
                 "infer": (sp.get("ocr.infer") or {}).get("sum"),
                 "q_get_wait": ga.get("pipeline.q_get_wait"),
                 "q_put_block": ga.get("pipeline.q_put_block")})
print("PROBE_JSON " + json.dumps(outs))
'''


def one(mult: int, cfg: str, frames: int, side: str = "producer") -> list:
    vid, roi, dec, ocr = VIDS[cfg]
    env = dict(os.environ)
    env.update({"PROBE_ROOT": str(ROOT), "PROBE_MULT": str(mult),
                "PROBE_SIDE": side,
                "PROBE_VIDEO": vid, "PROBE_ROI": ",".join(map(str, roi)),
                "PROBE_DEC": dec, "PROBE_OCR": ocr,
                "PROBE_FRAMES": str(frames),
                "RACELOG_VIDEO_DIR": env.get("RACELOG_VIDEO_DIR",
                                             r"D:\Videos\racelog_test")})
    p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    raise SystemExit("worker(mult=%d/%s/%s) 失败：\n%s\n%s"
                     % (mult, cfg, side, p.stdout[-1500:], p.stderr[-2000:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="h264-cpu")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=4.0)
    ap.add_argument("--mult", type=int, default=5)
    ap.add_argument("--side", default="producer",
                    choices=["producer", "consumer", "both"],
                    help="冗余哪一侧（consumer=重复 OCR 预处理）")
    args = ap.parse_args()

    mults = [1, args.mult]
    cfgs = args.config.split(",")
    rows: dict = {c: {m: [] for m in mults} for c in cfgs}
    for i in range(args.repeat):
        for cfg in cfgs:
            for m in mults:
                out = one(m, cfg, args.frames, args.side)
                rows[cfg][m].append(out)
                h = out[-1]
                print("  [%d] %-10s mult=%d hot=%.4f segs=%d fp=%s "
                      "producer=%.3f decode=%.3f wait=%.3f"
                      % (i, cfg, m, h["wall"], h["segs"], h["fp"],
                         h["producer"] or -1, h["decode_batch"] or -1,
                         h["q_get_wait"] or -1))
            time.sleep(args.cooldown)

    summary = {"side": args.side, "cfgs": {}}
    print("\n%-10s %6s %10s %10s %10s %9s %s"
          % ("config", "mult", "hot wall", "producer", "decode.bat", "Δwall", "门禁"))
    for cfg in cfgs:
        base = statistics.median([rows[cfg][1][i][1]["wall"]
                                  for i in range(args.repeat)])
        segs0 = rows[cfg][1][-1][1]["segs"]
        fp0 = rows[cfg][1][-1][1]["fp"]
        summary["cfgs"][cfg] = {"mults": {}}
        for m in mults:
            ws = [rows[cfg][m][i][1]["wall"] for i in range(args.repeat)]
            med = statistics.median(ws)
            prod = statistics.median([rows[cfg][m][i][1]["producer"] or 0
                                      for i in range(args.repeat)])
            decb = statistics.median([rows[cfg][m][i][1]["decode_batch"] or 0
                                      for i in range(args.repeat)])
            signs = "".join("+" if rows[cfg][m][i][1]["wall"]
                            > rows[cfg][1][i][1]["wall"] else "-"
                            for i in range(args.repeat)) if m != 1 else "-"
            segs_ok = all(rows[cfg][m][i][1]["segs"] == segs0
                          for i in range(args.repeat))
            fp_ok = all(rows[cfg][m][i][1]["fp"] == fp0
                        for i in range(args.repeat))
            print("%-10s %6d %10.4f %10.4f %10.4f %+8.2f%% %s/%s %s"
                  % (cfg if m == mults[0] else "", m, med, prod, decb,
                     (med - base) / base * 100 if m != 1 else 0.0, signs,
                     "段数OK" if segs_ok else "段数FAIL",
                     "指纹OK" if fp_ok else "指纹FAIL"))
            summary["cfgs"][cfg]["mults"][m] = {
                "wall": med, "producer": prod, "decode_batch": decb,
                "delta_pct": (med - base) / base * 100 if m != 1 else 0.0,
                "signs": signs, "segs_ok": segs_ok, "fp_ok": fp_ok}

    out = ROOT / "bench" / ("producer_binding2_%s.json" % args.side)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "raw": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n→ %s" % out)
    print("判读（side=%s）：mult=%d 后墙钟若明显上升 → 该侧在关键路径；"
          "若几乎不动 → 被等待吸收，换库无法提速" % (args.side, args.mult))
    print("本机可分辨下限 |Δ|p95 = 0.484%")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
