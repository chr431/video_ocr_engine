"""第四轮配套：互补设计的 DRAM 消耗测量 + 瞬时带宽序列。

墙钟只能告诉我们「退化了多少」，要回答「是不是内存带宽争用」，还得量它到底
吃多少带宽 —— 这正是第三轮推翻历史结论的那把尺子（B_max = 55.8 GB/s）。

本轮的新问题：**CPU 解码是访存大户**。第三轮量的 ONNX 单跑 11.7 GB/s 是
ONNX 走 **NVDEC** 时的值；本轮对端改成 **CPU 解码** 后，它的 DRAM 消耗预期
显著更高。若互补设计在「高 DRAM 消耗」下退化反而**更小**，就再次证明带宽
不是支配变量。

组合：
  bw  --work onnx_cpu   T6   : CPU 解码 + ONNX 单跑（访存大户本体）
  bw  --work mixed_cpu  T6+T6: 互补设计，两条都跑 AV1（重载）
  bw  --work mixed_cpu  T6+T5: 互补设计，默认档位
  bw  --work trt2_cpu   T6+T6: 对照
  ts  --work mixed_cpu  T6+T6: 瞬时带宽，查突发尖峰

探针口径固定 2 进程 × 4 线程（与第三轮一致，保证数字可比），窗口 6s。
"""
import json
import subprocess
import sys
import os
from pathlib import Path
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


PY = sys.executable
HERE = __file__.rsplit("\\", 1)[0] + "\\"
SCRIPT = HERE + "_probe_mem_bw.py"

T6 = str(_VIDEO_DIR / "test6.mp4")
T5 = str(_VIDEO_DIR / "test5.mp4")

BMAX = 55.8          # 第三轮标定：本机 copy 口径可达带宽上限

# (标签, mode, work, video, video2, warmup, secs, 额外参数)
# 探针口径固定 2 进程 × 4 线程、窗口 6s —— 与第三轮标定 B_max=55.8 完全同口径，
# 否则 `B_max − B_with` 的差值不可比。
RUNS = [
    ("bw/onnx_cpu_T6",  "bw", "onnx_cpu",  T6, T6, 25, 6, []),
    ("bw/mixed_cpu_T6", "bw", "mixed_cpu", T6, T6, 25, 6, []),
    ("bw/mixed_cpu_T5", "bw", "mixed_cpu", T6, T5, 25, 6, []),
    ("bw/trt2_cpu_T6",  "bw", "trt2_cpu",  T6, T6, 25, 6, []),
    ("ts/mixed_cpu_T6", "ts", "mixed_cpu", T6, T6, 18, 0,
     ["--slice", "0.15", "--window", "25"]),
]


def main():
    want = sys.argv[1:]
    out = {}
    for label, mode, kind, v, v2, warmup, secs, extra in RUNS:
        if want and not any(w in label for w in want):
            continue
        print("=" * 72, flush=True)
        print(f"### {label}   mode={mode} work={kind}", flush=True)
        print("=" * 72, flush=True)
        cmd = [PY, SCRIPT, "--mode", mode, "--work", kind,
               "--video", v, "--video2", v2,
               "--procs", "2", "--threads", "4",
               "--warmup", str(warmup), "--kernels", "copy"]
        if mode == "bw":
            cmd += ["--window", str(secs)]
        cmd += extra
        if mode == "ts":
            cmd += ["--warmup", str(warmup)]
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        txt = (p.stdout or "") + (p.stderr or "")
        print(txt, flush=True)
        recs = []
        for line in txt.splitlines():
            if line.startswith("@@JSON@@"):
                recs.append(json.loads(line[len("@@JSON@@"):]))
        out[label] = recs

    print("\n" + "=" * 72)
    print("### 汇总（B_max = %.1f GB/s）" % BMAX)
    print("=" * 72)
    for label, recs in out.items():
        for r in recs:
            if r.get("mode") == "bw":
                b = r["bw"].get("copy", 0.0)
                print(f"{label:18s} 剩余 {b:6.1f} GB/s  "
                      f"→ 负载消耗 {BMAX - b:6.1f} GB/s")
            elif r.get("mode") == "ts":
                s = sorted(r["series"])
                n = len(s)
                if n:
                    q = lambda p: s[min(n - 1, int(p * n))]        # noqa: E731
                    print(f"{label:18s} 瞬时 中位 {q(.5):5.1f} | "
                          f"p05 {q(.05):5.1f} | min {s[0]:5.1f} GB/s "
                          f"（n={n}）→ 中位消耗 {BMAX - q(.5):5.1f}，"
                          f"峰值消耗 {BMAX - s[0]:5.1f}")
    with open(HERE + "_round4_bw.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
