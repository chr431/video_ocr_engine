"""第四轮：早期「完全互补双流水线」设计的争用测量（串行执行，避免互相污染）。

## 要回答的问题

用户：最早期的双流水线是**完全互补**的（一条 CPU 解码 + ONNX，一条 NVDEC + TRT），
但争用依然严重。测一下这种情况。

## 配置矩阵：把 NVDEC 会话数 与 GPU 上下文数 分离

记 P1 = 跑 test6 的主流水线（TRT/NVDEC，恒定不变，作参照），P2 = 对端。

| work       | P1 (test6)   | P2         | NVDEC 会话 | GPU 上下文 |
|------------|--------------|------------|-----------|-----------|
| trt        | TRT/NVDEC    | —          | 1         | 1         |
| mixed      | TRT/NVDEC    | ONNX/NVDEC | 2         | 2         |
| trt2       | TRT/NVDEC    | TRT/NVDEC  | 2         | 2         |
| mixed_cpu  | TRT/NVDEC    | ONNX/CPU解 | **1**     | 2         |
| trt2_cpu   | TRT/NVDEC    | TRT/CPU解  | **1**     | 2         |

## ⚠️ 必须控住的两个混淆变量

1. **视频不同**：默认 P1 跑 test6（**AV1** 318.7MB），P2 跑 test5（**h264** 166.5MB）。
   而 h264 的 CPU 解码**快于** NVDEC，AV1 的 CPU 解码**慢于** NVDEC —— 所以
   `mixed_cpu` 默认把对端放在 CPU 解码最有利的档位上，会**低估**互补设计的争用。
   对策：补测 `--video2 test6`（两条都跑 AV1）的重载版本。
2. **对端没有单跑基线**：不知道对端自己多贵，就无法判断它给 P1 施加了多少压力。
   对策：把每个对端配置在**它自己那条视频上**单独跑一遍。

## 聚合吞吐指标

两条流水线**持续并发跑同一段时间 D**，各自完成 D/t_conc 次迭代。串行做完同样
工作量需要 Σ (D/t_conc,i × t_solo,i)。故

    Speedup = Σ_i ( t_solo,i / t_conc,i )

完美并行 + 无干扰 + 两条一样快 = 2.00。用它回答「该不该开两条」，
用单侧退化倍数回答「互相伤害多少」，两者不是一回事。
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

# (标签, work, video, video2, secs)
RUNS = [
    # ── test5 单独基线（对端的成本，必须在它自己那条视频上量）──
    ("solo/T5/trt-nvdec",      "trt",      T5, T5, 60),
    ("solo/T5/trt-cpudec",     "trt_cpu",  T5, T5, 60),
    ("solo/T5/onnx-nvdec",     "onnx",     T5, T5, 60),
    ("solo/T5/onnx-cpudec",    "onnx_cpu", T5, T5, 70),
    # ── test6 缺的那条 ──
    ("solo/T6/trt-cpudec",     "trt_cpu",  T6, T6, 70),
    # ── 并发，P2 在 test5（默认档位）──
    ("conc/mixed",             "mixed",     T6, T5, 70),
    ("conc/trt2",              "trt2",      T6, T5, 70),
    ("conc/mixed_cpu",         "mixed_cpu", T6, T5, 70),
    ("conc/trt2_cpu",          "trt2_cpu",  T6, T5, 70),
    # ── 并发，两条都跑 AV1（重载档位，CPU 解码最不利）──
    ("conc6/mixed_cpu",        "mixed_cpu", T6, T6, 80),
    ("conc6/trt2_cpu",         "trt2_cpu",  T6, T6, 70),
]


def run_one(label, kind, video, video2, secs):
    print("=" * 72, flush=True)
    print(f"### {label}   work={kind}  video={video[-9:]}  video2={video2[-9:]}",
          flush=True)
    print("=" * 72, flush=True)
    p = subprocess.run(
        [PY, SCRIPT, "--mode", "wall", "--work", kind,
         "--video", video, "--video2", video2, "--secs", str(secs)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = (p.stdout or "") + (p.stderr or "")
    print(txt, flush=True)
    rows = []
    for line in txt.splitlines():
        if line.startswith("@@JSON@@"):
            rows.append(json.loads(line[len("@@JSON@@"):]))
    for w in rows:                       # 标注各自跑的是哪条视频
        w["_label"] = label
    return rows


def main():
    want = sys.argv[1:]
    out = {}
    for label, kind, v, v2, secs in RUNS:
        if want and not any(w in label for w in want):
            continue
        out[label] = run_one(label, kind, v, v2, secs)

    print("\n" + "=" * 72)
    print("### 汇总")
    print("=" * 72)
    print(f"{'标签':22s} {'OCR':9s} {'解码':12s} {'中位墙钟':>9s} "
          f"{'最快':>8s} {'次数':>4s} {'段数':>6s} {'唯一':>5s}")
    for label, rows in out.items():
        for w in rows:
            print(f"{label:22s} {w['ocr_backend']:9s} "
                  f"{w.get('used_decode', '?'):12s} "
                  f"{w['wall_median']:8.3f}s {w['wall_min']:7.3f}s "
                  f"{w['iters']:4d} {w['segs']:6d} {w.get('uniq', '?'):5}")
    with open(HERE + "_round4_wall.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
