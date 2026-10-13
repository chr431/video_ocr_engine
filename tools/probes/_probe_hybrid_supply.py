"""hybrid 纯解码供给上限测量（C-69：绕过引擎，直接 decord.hybrid 双臂）。

口径：ctx=decord.hybrid(0) + VideoReader(gray, roi, num_threads=按码档位)，
get_batch 300/次拉满 3000 帧，热池 3 reps 中位。无 OCR 消费——测的是
泵供给线（demux→parse→push→分侧队列）+ 双臂解码的合计上限。

用途（C-69 判据）：供给上限 vs 理论和（双臂纯速率相加）的差 = 供给约束；
引擎内 hybrid 实测 vs 供给上限的差 = 引擎消费耦合损失。
"""
import statistics
import sys
import time

sys.path.insert(0, "tools")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import decord  # noqa: E402

from _measure_lib import ROI, VIDS  # noqa: E402

# R-perf-quiet：后台负载门禁——超阈立即退出，不浪费整轮测量
from _measure_lib import BackgroundLoadError, assert_machine_quiet

try:
    assert_machine_quiet()
except BackgroundLoadError as _e:  # noqa: BLE001 — 门禁语义：超阈退出
    print(_e)
    raise SystemExit(2)

# ⚠️ 若要加 FORCE_SIDE 诊断：ForcedSide() 的 env 是进程级 static 缓存，
# 必须每臂独立子进程（同进程连测三值全为首个缓存语义）。
CASES = [("h264", "test6_h264", 32), ("av1", "test6_av1", 24), ("hevc", "test6_hevc", 32)]
WIN, BATCH = 3000, 300


def supply(vid: str, cpu_t: int) -> float:
    ctx = decord.hybrid(0)
    vr = decord.VideoReader(VIDS[vid], ctx=ctx, output_format="gray",
                            roi=ROI["test6"], num_threads=cpu_t)
    xs = []
    for rep in range(3):
        t0 = time.perf_counter()
        for i in range(0, WIN, BATCH):
            vr.get_batch(list(range(i, min(i + BATCH, WIN))))
        xs.append(time.perf_counter() - t0)
        if rep < 2:
            vr.seek(0)
    return WIN / statistics.median(xs)


if __name__ == "__main__":
    for name, vid, ct in CASES:
        print("%-5s cpuT=%d  hybrid 纯解码供给: %.0f fps" % (name, ct, supply(vid, ct)))
