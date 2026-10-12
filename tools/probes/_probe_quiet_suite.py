"""静默机器全量补测（2026-10-11 夜）：一次跑完被会话噪声淹没的测量。

A. 直读供给 B=64 / B=300（批尺寸效应，同会话同口径）
B. FORCE cpu / gpu（独立子进程——ForcedSide static 缓存勘误）+ 均衡
C. 引擎 decode 相位（span 口径）vs A —— Tier-2 残差判定
D. 三码缺口（C-68 口径复测，0.9.3 队列修复后）
"""
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, "tools")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

os.environ["GPU_PIPELINE"] = "1"

from _measure_lib import ROI, VIDS  # noqa: E402
from video_ocr_engine import FieldExtractor  # noqa: E402

H264 = VIDS["test6_h264"]
ROI6 = ROI["test6"]
OUT = {"t0": time.strftime("%Y-%m-%d %H:%M:%S")}


def supply_direct(vid, cpu_t, B, reps=3):
    import decord
    ctx = decord.hybrid(0)
    vr = decord.VideoReader(VIDS[vid], ctx=ctx, output_format="gray",
                            roi=ROI[vid == "test5" and "test5" or "test6"],
                            num_threads=cpu_t)
    n, xs = 3000, []
    for rep in range(reps):
        t0 = time.perf_counter()
        for i in range(0, n, B):
            vr.get_batch(list(range(i, min(i + B, n))))
        xs.append(time.perf_counter() - t0)
        if rep < reps - 1:
            vr.seek(0)
    return n / statistics.median(xs)


def engine_rate(vid, backend, ocr="tensorrt", reps=3):
    xs, spans = [], None
    ex = FieldExtractor(VIDS[vid], ROI6 if vid != "test5" else ROI["test5"],
                        frame_start=0, frame_end=3000, decode_backend=backend,
                        ocr_backend=ocr, sample_stride=1)
    ex.extract()
    for _ in range(reps):
        ex2 = FieldExtractor(VIDS[vid], ROI6 if vid != "test5" else ROI["test5"],
                             frame_start=0, frame_end=3000, decode_backend=backend,
                             ocr_backend=ocr, sample_stride=1)
        t0 = time.perf_counter()
        r = ex2.extract()
        xs.append(time.perf_counter() - t0)
        spans = (r.meta.get("report") or {}).get("spans") or {}
    return 3000 / statistics.median(xs), spans


# ── A. 直读供给 B=64 / B=300 ──
OUT["supply_b64"] = supply_direct("test6_h264", 32, 64)
OUT["supply_b300"] = supply_direct("test6_h264", 32, 300)

# ── B. FORCE 分臂（独立子进程）+ 均衡 ──
FORCE_SNIP = """
import sys, time, statistics
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import decord
from _measure_lib import ROI, VIDS
# R-perf-quiet：后台负载门禁——超阈立即退出，不浪费整轮测量
from _measure_lib import BackgroundLoadError, assert_machine_quiet

try:
    assert_machine_quiet()
except BackgroundLoadError as _e:  # noqa: BLE001 — 门禁语义：超阈退出
    print(_e)
    raise SystemExit(2)

ctx = decord.hybrid(0)
vr = decord.VideoReader(VIDS["test6_h264"], ctx=ctx, output_format="gray",
                        roi=ROI["test6"], num_threads=32)
n, B, xs = 3000, 64, []
for rep in range(3):
    t0 = time.perf_counter()
    for i in range(0, n, B):
        vr.get_batch(list(range(i, min(i + B, n))))
    xs.append(time.perf_counter() - t0)
    if rep < 2:
        vr.seek(0)
print("%.1f" % (n / statistics.median(xs)))
"""
for side in ("cpu", "gpu"):
    env = dict(os.environ, DECORD_HYBRID_FORCE_SIDE=side)
    r = subprocess.run([sys.executable, "-c", FORCE_SNIP], env=env,
                       capture_output=True, text=True, timeout=300)
    OUT["force_" + side] = float(r.stdout.strip().splitlines()[-1])

# ── C. 引擎 h264 hybrid（decode span 口径）──
OUT["engine_h264"], spans = engine_rate("test6_h264", "hybrid")
OUT["engine_decode_batch_sum"] = (spans.get("decode.batch") or {}).get("sum")

# ── D. 三码缺口 ──
for name, vid in (("hevc", "test6_hevc"), ("h264", "test6_h264"), ("av1", "test6_av1")):
    rg, _ = engine_rate(vid, "nvdec")
    rc, _ = engine_rate(vid, "cpu")
    rh, _ = engine_rate(vid, "hybrid")
    OUT["gap_" + name] = {"nvdec": round(rg), "cpu": round(rc),
                          "theo": round(rg + rc), "hybrid": round(rh),
                          "gap_pct": round((rh - rg - rc) / (rg + rc) * 100, 1)}

with open("bench/quiet_machine_suite.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
