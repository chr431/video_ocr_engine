"""混合解码器**上载链**的成本归属（fork 侧优化的前置测量，无需重编译）。

假设（来自 fork 内计数器）：`hybrid_gpu` 比宿主输出 `hybrid` 慢 19–24% 的
主因是 CPU 分片帧的 H2D 上载链——`UploadStep()` 里批次数组 `bufs[]` 是
**栈上局部量**，所以 `cpu_.Pop()` 一空就必须 `flush()`（整条流 sync）：
实测 3000 帧触发 764 次 flush / ~2340 个 CPU 帧 = 每批仅 3.1 帧（设计批 8），
批次化名存实亡。按代码注释自陈的单次 sync 0.4–0.7ms，764×0.4ms ≈ **0.31s**，
与 h264 的实测缺口（理想 0.66s vs 实际 0.98s ≈ 0.33s）同量级。

用 fork 自带的 `DECORD_HYBRID_FORCE_SIDE` 拆路径：

| 组 | 含义 | 用途 |
|---|---|---|
| mixed | 真实混合路由 | 现值 |
| gpu   | 全部 NVDEC（**零上载**） | mixed − gpu = 含 CPU 分片那一侧的全部代价 |
| cpu   | 全部 CPU（**每帧上载**） | ⚠️ **实测会挂**（worker 忙等，见下）—— 默认不跑 |

**方法学**：
- 每组每轮**独立子进程**且带 `--timeout` 硬超时：挂起会被记成 TIMEOUT 继续，
  而不是把整张表卡死（第一版没有超时，一个 `cpu` 槽位烧了 735s CPU 仍未退出）。
- 轮内组顺序按轮轮转（消位置效应），取中位。
- `DECORD_HYBRID_DEBUG` 一律清除（开一次 stderr 13 万行，墙钟完全失真）。
- 进度**逐行落文件**（`--log`），因为管道缓冲会让"正在跑"看起来像"挂起"。

用法：
  python tools/_probe_upload_chain.py --cases h264,av1 --groups mixed,gpu --reps 3
  python tools/_probe_upload_chain.py --groups cpu --frames 300 --timeout 60   # 复现挂起
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKER = r'''
import os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
from decord import VideoReader, hybrid_gpu
vr = VideoReader(os.environ["PROBE_VID"], ctx=hybrid_gpu(0),
                 output_format="gray", num_threads=16,
                 roi=tuple(int(x) for x in os.environ["PROBE_ROI"].split(",")))
n = min(int(os.environ["PROBE_FRAMES"]), len(vr))
idx = list(range(n))
best = 0.0
for _ in range(int(os.environ.get("PROBE_SWEEPS", "2"))):
    t = time.perf_counter()
    got = 0
    for s in range(0, n, 64):
        got += vr.get_batch(idx[s:s + 64]).shape[0]
    best = max(best, got / (time.perf_counter() - t))
print("UPJSON " + str(round(best, 1)))
'''

# 视频目录可用 RACELOG_VIDEO_DIR 覆盖（与 bench.py 同一约定，换机器不坏）
_VDIR = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
CASES = {"h264": (os.path.join(_VDIR, "test5.mp4"), "843,993,949,1026"),
         "hevc": (os.path.join(_VDIR, "test6_hevc.mp4"), "841,994,950,1027"),
         "av1": (os.path.join(_VDIR, "test6.mp4"), "841,994,950,1027")}


def run(vid: str, roi: str, force: str, frames: int, timeout: float):
    env = dict(os.environ)
    env.update({"PROBE_ROOT": str(ROOT), "PROBE_VID": vid, "PROBE_ROI": roi,
                "PROBE_FRAMES": str(frames)})
    env.pop("DECORD_HYBRID_DEBUG", None)
    env.pop("DECORD_HYBRID_FORCE_SIDE", None)
    if force:
        env["DECORD_HYBRID_FORCE_SIDE"] = force
    try:
        p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    for ln in p.stdout.splitlines():
        if ln.startswith("UPJSON "):
            return float(ln[len("UPJSON "):]), None
    return None, "WORKER_FAIL:" + (p.stderr.strip().splitlines()
                                   or ["?"])[-1][:70]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cases", default="h264,av1")
    ap.add_argument("--groups", default="mixed,gpu",
                    help="mixed/gpu/cpu 子集（cpu 实测挂起，需显式点）")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--fork", default=os.environ.get("DECORD_FORK_BUILD", ""),
                    help="fork 构建树目录（建议用 DECORD_FORK_BUILD 环境变量传，"
                         "不写死路径——审计会把 tools/ 里的绝对路径记为警告）")
    ap.add_argument("--log", default=str(ROOT / "bench" / "upload_chain.log"))
    args = ap.parse_args()
    if args.fork:
        os.environ["DECORD_LIBRARY_PATH"] = args.fork
    else:
        print("⚠️ 未给 --fork/DECORD_FORK_BUILD → 用**已安装**的 decord"
              "（多半是 0.8.2 wheel，不含 fork 修复）")
    Path(args.log).parent.mkdir(exist_ok=True)
    log = open(args.log, "w", encoding="utf-8", buffering=1)   # 行缓冲：实时可tail
    groups = args.groups.split(",")
    res: dict = {}
    hangs = 0
    for i in range(args.reps):
        order = groups[i % len(groups):] + groups[:i % len(groups)]
        for case in args.cases.split(","):
            vid, roi = CASES[case]
            for g in order:
                t0 = time.perf_counter()
                fps, err = run(vid, roi, "" if g == "mixed" else g,
                               args.frames, args.timeout)
                res.setdefault((case, g), []).append(fps)
                line = "  [%d] %-5s %-6s %s  (壁钟 %.1fs)" % (
                    i, case, g,
                    ("TIMEOUT/挂起" if fps is None else "%7.0f fps" % fps),
                    time.perf_counter() - t0)
                if err:
                    line += "  " + err
                    hangs += 1
                print(line, flush=True)
                log.write(line + "\n")
    log.close()
    print("\n%-6s %10s %10s %10s   %s" % ("编码", "mixed", "纯NVDEC", "纯CPU",
                                          "mixed 相对纯 NVDEC 臂的差（**不是** CPU 侧代价的归因）"))
    for case in args.cases.split(","):
        def med(g):
            v = [x for x in res.get((case, g), []) if x]
            return statistics.median(v) if v else None
        m, gp, cp = med("mixed"), med("gpu"), med("cpu")
        n = args.frames
        extra = ""
        if m and gp and gp > m:
            # mixed 比"全 NVDEC"还慢 → 差值即 CPU 侧（解码+上载+调度）的净代价
            d_ms = (n / m - n / gp) * 1000
            extra = ("mixed 比纯 NVDEC 慢 %.0f ms/%d 帧 = %.1f µs/帧"
                     % (d_ms, n, d_ms * 1000 / n))
        print("%-6s %10s %10s %10s   %s" % (
            case,
            "%.0f" % m if m else "—", "%.0f" % gp if gp else "—",
            "%.0f" % cp if cp else ("挂起" if "cpu" in groups else "未测"),
            extra))
    if hangs:
        print("\n⚠️ 有 %d 个槽位超时/失败——见 %s" % (hangs, args.log))
    return 1 if hangs else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
