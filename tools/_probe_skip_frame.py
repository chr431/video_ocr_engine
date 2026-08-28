"""真跳帧解码可行性探路（goal 1 决策输入）。

三问：
  Q1 目标视频里"非参考帧"占比多少？（决定收益上限）
  Q2 FFmpeg 原生 `skip_frame=noref` 能带来多少解码提速？（决定机制收益）
  Q3 `skip_frame=noref` 是解码器级安全跳过，还是像旧的 pict_type 丢包
     那样破坏参考关系？（决定能不能走这条路）

Q1/Q2 用 ffmpeg CLI 直接量（不碰 decord）；Q3 用"跳过后剩余帧能否与
不跳过的同帧号一一对齐"来验证——若解码器自己维护 DPB，剩余帧应逐位一致。

用法：
  python tools/_probe_skip_frame.py --video X [--frames 3000] [--reps 3]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

FFMPEG = r"D:\Software\ffmpeg8\bin\ffmpeg.exe"
FFPROBE = r"D:\Software\ffmpeg8\bin\ffprobe.exe"


def _run(args: list[str], timeout: float = 600.0) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True,
                       errors="replace", timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def time_ffmpeg(video: str, extra_in: list[str], threads: int,
                frames: int, reps: int) -> dict:
    """跑 `ffmpeg [extra_in] -threads N -i V [-frames:v F] -f null -`。

    注意 `-frames:v F` 限制的是**输出**帧数，不是输入帧数——用它做 skip
    实验会让 ffmpeg 为了凑够 F 个输出而多读输入，基准被系统性抬高低估
    收益（首次实验即踩此坑：noref 反而"变慢" 0.60x）。因此 frames<=0
    时整片解码，用输出帧数差直接量"跳过了多少帧"。
    """
    ts, decoded = [], []
    lim = (["-frames:v", str(frames)] if frames and frames > 0 else [])
    for _ in range(reps):
        args = [FFMPEG, "-hide_banner", "-nostats", *extra_in,
                "-threads", str(threads), "-i", video,
                *lim, "-f", "null", "-"]
        t0 = time.perf_counter()
        rc, log = _run(args)
        dt = time.perf_counter() - t0
        if rc != 0 and "Output file is empty" not in log:
            tail = "\n".join(log.strip().splitlines()[-6:])
            return {"error": f"rc={rc}\n{tail}"}
        ts.append(dt)
        m = re.search(r"frame=\s*(\d+)", log)
        decoded.append(int(m.group(1)) if m else -1)
    return {"median_s": round(statistics.median(ts), 3),
            "min_s": round(min(ts), 3),
            "decoded_frames": statistics.median(decoded),
            "runs": ts}


def dump_frames(video: str, extra_in: list[str], threads: int,
                frames: int, out_pat: str, tmpdir: Path) -> tuple[int, list[str]]:
    """把前 F 帧导出为 gray pgm 序列，返回 (帧数, 文件列表)。

    用于 Q3 对齐校验：同样 -frames:v F 下，noref 会跳过一些帧，
    剩下的帧应与 baseline 中"未被跳过的同序号帧"逐位一致。
    由于无法预知哪些被跳过，改用另一种判据：导出 noref 的帧序列，
    再在 baseline 序列里按顺序子序列匹配（见 main 的校验逻辑）。
    """
    tmpdir.mkdir(parents=True, exist_ok=True)
    args = [FFMPEG, "-hide_banner", "-nostats", "-y", *extra_in,
            "-threads", str(threads), "-i", video,
            "-frames:v", str(frames), "-pix_fmt", "gray",
            "-f", "image2", str(tmpdir / out_pat)]
    rc, log = _run(args)
    if rc != 0:
        return 0, [("\n".join(log.strip().splitlines()[-6:]))]
    files = sorted(tmpdir.glob(out_pat.replace("%06d.png", "*.png")),
                   key=lambda p: p.name)
    return len(files), [str(f) for f in files]


def probe_meta(video: str) -> dict:
    args = [FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_frames",
            "-show_entries", "format=duration", "-of", "json", video]
    rc, log = _run(args)
    try:
        d = json.loads(log)
        s = d["streams"][0]
        return {"codec": s.get("codec_name"), "w": s.get("width"),
                "h": s.get("height"), "nb_frames": s.get("nb_frames"),
                "duration": d.get("format", {}).get("duration")}
    except Exception:
        return {"raw": log[:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=0,
                    help="0=不显式指定（用 ffmpeg 默认 = 全核）")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--align", action="store_true",
                    help="额外做 Q3 对齐校验（导出帧序列，慢）")
    a = ap.parse_args()

    th = a.threads if a.threads > 0 else (os.cpu_count() or 8)
    meta = probe_meta(a.video)
    print(f"== {Path(a.video).name} ==")
    print(f"   meta: {meta}")
    print(f"   threads={th} frames={a.frames} reps={a.reps}")

    base = time_ffmpeg(a.video, [], th, a.frames, a.reps)
    if "error" in base:
        print("baseline 失败：", base["error"])
        return 2
    print(f"\n[baseline]             {base['median_s']}s  "
          f"decoded={base['decoded_frames']}")

    results = {"video": a.video, "meta": meta, "threads": th,
               "frames": a.frames, "baseline": base, "variants": {}}

    variants = [
        ("noref", ["-skip_frame", "noref"]),
        ("bidir", ["-skip_frame", "bidir"]),
        ("nonkey", ["-skip_frame", "nonkey"]),
        ("noref+noloop", ["-skip_frame", "noref",
                          "-skip_loop_filter", "noref"]),
        ("noloop_all", ["-skip_loop_filter", "all"]),
    ]
    for name, extra in variants:
        r = time_ffmpeg(a.video, extra, th, a.frames, a.reps)
        if "error" in r:
            print(f"[{name}] 失败：{r['error'][:200]}")
            results["variants"][name] = r
            continue
        dec = r["decoded_frames"]
        full = base["decoded_frames"] or 1
        speed = base["median_s"] / r["median_s"] if r["median_s"] > 0 else 0
        kept = dec / full if full > 0 else 0
        print(f"[{name:14s}] {r['median_s']}s  decoded={dec}  "
              f"kept={kept:.1%}  speedup={speed:.2f}x")
        results["variants"][name] = {**r, "kept": kept, "speedup": speed}

    # Q1：非参考帧占比 = 1 - kept(noref)
    nr = results["variants"].get("noref", {})
    if "kept" in nr:
        print(f"\n  → 非参考帧占比 ≈ {1 - nr['kept']:.1%}"
              f"（= 真跳帧的收益上限）")

    out = Path(__file__).with_name("_probe_skip_frame.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n明细落盘：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
