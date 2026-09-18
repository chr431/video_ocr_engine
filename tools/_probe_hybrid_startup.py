"""hybrid 启动期派工诊断：盲阶段（速率未熟）到底分掉了多少 GOP。

假说（2026-09-18 待证）：demux 远快于解码，rc 出版（CPU 臂 4 折×16=64
帧，hevc ~85ms+）之前已关闭的 GOP 全部走 50/50 交替保守派工 → CPU 臂
超分 → 慢臂尾部 = hevc w3000 hybrid +35% 慢的大头。

口径：fork 级 decode-only（hybrid_gpu ctx + ROI + 引擎同款线程档），
冷进程，硬窗界 set_decode_window(n)，DECORD_HYBRID_TRACE=1。
判据：assigned_c vs 稳态份额 f×n（trace 的 share_bp 稳态值）；
盲派工数 = chunks_total − trace 点数（trace 只记速率就绪后的派工）。

用法：python tools/_probe_hybrid_startup.py [--video test6_hevc] [--n 3000]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VID = {  # name → (file, roi（真值口径 x1,y1,x2,y2 闭）)
    "test5": ("test5.mp4", (843, 993, 948, 1025)),
    "test6_hevc": ("test6_hevc.mp4", (841, 994, 949, 1026)),
    "test6_av1": ("test6.mp4", (841, 994, 949, 1026)),
}
# 口径轮 2026-09-18：线程档/format 改引擎同源派生——
# decode_num_threads(codec)（DECODE_THREADS env 生效）+ 缺省 gray
# （引擎 GPU 管线生产口径；--format yuv420 对齐 keep_crops+yuv）。

WORKER = r'''
import sys, time, json
sys.stdout.reconfigure(encoding="utf-8")
from decord import VideoReader, gpu, hybrid, hybrid_gpu
path = sys.argv[1]
x1, y1, x2, y2 = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
nt, n, mode, fmt = int(sys.argv[6]), int(sys.argv[7]), sys.argv[8], sys.argv[9]
ctx = {"gpu": gpu(0), "hybrid": hybrid(0), "hybrid_gpu": hybrid_gpu(0)}[mode]
t_ctor0 = time.perf_counter()
vr = VideoReader(path, ctx=ctx, output_format=fmt,
                 roi=(x1, y1, x2, y2), num_threads=nt)
t_ctor = time.perf_counter() - t_ctor0
sdw = getattr(vr, 'set_decode_window', None)
if sdw is not None and n < len(vr):
    sdw(n)
vr.seek(0)
got, t0, batches = 0, time.perf_counter(), []
while got < n:
    e = min(got + 64, n)
    tb = time.perf_counter()
    b = vr.get_batch(list(range(got, e)))
    batches.append([got, round(time.perf_counter() - tb, 5)])
    got += b.shape[0]
    del b
wall = time.perf_counter() - t0
stats = vr.hybrid_stats() if mode.startswith("hybrid") else {}
vr.close()
print("RESULT " + json.dumps({
    "mode": mode, "n": got, "wall": round(wall, 4),
    "fps": round(got / wall, 1), "ctor_s": round(t_ctor, 4),
    "batches": batches, "stats": stats}))
'''


def run_one(python: str, path: str, roi, nt: int, n: int, mode: str,
            fmt: str = 'gray'):
    env = dict(os.environ)
    env["DECORD_HYBRID_TRACE"] = "1"
    proc = subprocess.run(
        [python, "-c", WORKER, path, str(roi[0]), str(roi[1]),
         str(roi[2]), str(roi[3]), str(nt), str(n), mode, fmt],
        capture_output=True, text=True, timeout=600, env=env,
        cwd=str(Path(__file__).resolve().parent))
    out = ""
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            out = line[7:]
    if not out:
        print(proc.stdout[-2000:], proc.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"{mode} run failed: {proc.returncode}")
    return json.loads(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test6_hevc")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--modes", default="hybrid_gpu,gpu")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--format", default="gray", choices=("gray", "yuv420"),
                    help="decord 输出格式（缺省=引擎 GPU 管线生产口径）")
    args = ap.parse_args()

    from video_ocr_engine.config.decode_caliber import (
        decode_num_threads, roi_for_decord)
    name, roi = VID[args.video]
    codec = {'test5': 'h264', 'test6_hevc': 'hevc',
             'test6_av1': 'av1'}[args.video]
    roi = roi_for_decord(roi)
    nt = decode_num_threads(codec)
    vdir = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
    path = str(Path(vdir) / name)

    rows = []
    for mode in args.modes.split(","):
        for r in range(args.runs):
            res = run_one(args.python, path, roi, nt, args.n, mode,
                          args.format)
            res["run"] = r
            rows.append(res)
            st = res.get("stats") or {}
            tr = st.get("trace") or []
            steady = [q[1] for q in tr[len(tr) // 2:]] if tr else []
            print(f"[{mode} #{r}] wall={res['wall']}s fps={res['fps']} "
                  f"ctor={res['ctor_s']}s")
            if st:
                print(f"  assigned c/g={st['assigned_c']}/{st['assigned_g']} "
                      f"chunks c/g={st['chunks_c']}/{st['chunks_g']} "
                      f"rc/rg={st['rc_now']:.0f}/{st['rg_now']:.0f} "
                      f"blind_chunks={(st['chunks_c']+st['chunks_g'])-len(tr)} "
                      f"trace_pts={len(tr)}")
                if steady:
                    print(f"  steady share_bp: p50={statistics.median(steady):.0f} "
                          f"first={steady[0]:.0f} last={steady[-1]:.0f}")
                print(f"  hol c/g={st['hol_us_c']/1e6:.3f}/{st['hol_us_g']/1e6:.3f}s "
                      f"busy c/g={st['busy_cpu_us']/1e6:.3f}/{st['busy_gpu_us']/1e6:.3f}s "
                      f"kicks c/g={st['kicks_c']}/{st['kicks_g']} clones={st['clones']}")

    out = Path(__file__).resolve().parents[1] / "bench" / "hybrid_startup.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"video": name, "n": args.n, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
