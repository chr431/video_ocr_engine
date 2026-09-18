"""_probe_content_det.py —— hybrid 输出内容确定性对照（帧级 hash）。

口径：fork 级全片解码，每帧 xxhash64（无依赖用 md5 前 8 字节）。
跑 N 轮 hybrid（redesign）+ 1 轮纯 GPU 参照，逐帧比对：
  - hybrid 各轮间差异帧号集合（内容不定 = concealment/乱序证据）
  - hybrid vs 纯 GPU 差异帧号集合（与换侧点的相关性）
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

WORKER = r'''
import hashlib, os, sys
sys.path.insert(0, os.environ["PROBE_ROOT"])
sys.stdout.reconfigure(encoding="utf-8")
import decord
from decord import VideoReader

ctx_name, path, roi_s, nt = sys.argv[1:5]
roi = tuple(int(v) for v in roi_s.split(","))
ctx = (decord.hybrid_gpu(0) if ctx_name == "hybrid"
       else decord.gpu(0))
vr = VideoReader(path, ctx=ctx, output_format="gray", roi=roi,
                 num_threads=int(nt) if ctx_name != "gpu" else 0)
n = len(vr)
hs = []
B = 64
for s in range(0, n, B):
    nds = vr.get_batch(list(range(s, min(s + B, n)))).asnumpy()
    for k in range(nds.shape[0]):
        hs.append(hashlib.md5(nds[k].tobytes()).hexdigest()[:10])
vr.close()
print("HASHES " + " ".join(hs))
'''


def run(ctx_name: str, video: str, roi, nt: str) -> list:
    env = dict(os.environ)
    env["PROBE_ROOT"] = str(ROOT)
    p = subprocess.run(
        [sys.executable, "-c", WORKER, ctx_name, video,
         ",".join(map(str, roi)), nt],
        capture_output=True, text=True, encoding="utf-8", env=env,
        timeout=90)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "")[-300:])
    line = [l for l in p.stdout.splitlines() if l.startswith("HASHES ")][0]
    return line.split()[1:]


def main() -> int:
    vid = os.path.join(os.environ.get("RACELOG_VIDEO_DIR",
                                      r"D:\Videos\racelog_test"),
                       "test6_hevc.mp4")
    from video_ocr_engine.config.decode_caliber import (
        decode_num_threads, roi_for_decord)
    roi = roi_for_decord((841, 994, 949, 1026))  # 口径轮：+1 换算
    nt = str(decode_num_threads("hevc"))
    ref = run("gpu", vid, roi, nt)
    print("gpu 参照帧数", len(ref))
    runs = []
    for i in range(3):
        h = run("hybrid", vid, roi, nt)
        runs.append(h)
        d_ref = [j for j in range(min(len(h), len(ref))) if h[j] != ref[j]]
        print(f"hybrid#{i}: n={len(h)} vs gpu 差异帧 {len(d_ref)} 个，"
              f"首 12 个: {d_ref[:12]}")
        if i > 0:
            d_prev = [j for j in range(len(h))
                      if h[j] != runs[0][j]]
            print(f"   vs hybrid#0 差异帧 {len(d_prev)} 个，首 12: {d_prev[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
