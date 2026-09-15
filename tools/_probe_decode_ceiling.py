"""解码侧依赖替换评估：现役 decord fork 是否已达硬件/实现天花板。

背景：本轮 `_probe_binding.py` 实测三条 GPU 路径**全部解码绑定**
（把 OCR 推理换成零成本，墙钟 +0.1~0.2% = 噪声）——
所以"换依赖能否再快"这个问题**只可能在解码侧**成立。

本探针不猜，用三路独立证据判定 decord 是否还有余量：

  ① **外部参考实现天花板**：用 ffmpeg 自带 `h264_cuvid`（NVDEC）与软解
     跑同一窗口 → 与引擎墙钟折算的 fps 对比。若 decord ≥ ffmpeg 参考，
     说明换一个 NVDEC 封装（PyNvVideoCodec / OpenCV cudacodec / PyAV）
     的**上界**也就是参考实现水平，不可能带来数量级收益。
  ② **ROI-first 价值量化**：同窗口测「全帧解码」vs「ROI 解码」吞吐。
     第三方面向通用场景的库（PyNvVideoCodec/PyAV/OpenCV）都缺 ROI-first，
     换过去必须自己补偿这块（ARCHIVE §4 P2-3 记 PyAV -46%）。
  ③ **固定开销占比**：批大小扫描，看每帧固定成本是否仍是瓶颈。

用法：
  python tools/_probe_decode_ceiling.py [--video test5.mp4] [--frames 3000]
      [--roi 843,993,948,1025]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
FFMPEG_DIR = Path(os.environ.get(
    "FFMPEG_DIR", r"D:\Software\ffmpeg-n9.0-latest-win64-gpl-shared-9.0\bin"))


def ffmpeg_window(video: str, frames: int, hwaccel: str, codec: str | None) -> dict:
    """用 ffmpeg 自身解同一窗口（外部参考实现天花板），返回秒/帧率。"""
    ff = FFMPEG_DIR / "ffmpeg.exe"
    if not ff.exists():
        return {"error": "ffmpeg 不在 %s" % FFMPEG_DIR}
    cmd = [str(ff), "-hide_banner", "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    if codec:
        cmd += ["-c:v", codec]
    cmd += ["-i", video, "-frames:v", str(frames), "-f", "null", "-"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    el = time.perf_counter() - t0
    return {"wall_s": round(el, 4), "fps": round(frames / el, 1),
            "rc": p.returncode}


def decord_roi(video: str, roi, frames: int, decode: str,
               batches=(64,), reps: int = 2) -> dict:
    """decord 生产口径（ROI + gray）批大小扫描。"""
    from decord import VideoReader, cpu
    out = {}
    for bs in batches:
        ts = []
        for _ in range(reps):
            kw = {}
            if decode == "nvdec":
                try:
                    from decord import gpu
                    ctx = gpu(0)
                except Exception:  # noqa: BLE001  # GPU 不可用则退回 CPU 对照
                    ctx = cpu(0)
            else:
                ctx = cpu(0)
            vr = VideoReader(video, ctx=ctx, output_format="gray",
                             width=roi[2] - roi[0], height=roi[3] - roi[1],
                             roi=roi, **kw)
            n = min(frames, len(vr))
            t0 = time.perf_counter()
            for s in range(0, n, bs):
                vr.get_batch(list(range(s, min(s + bs, n)))).asnumpy()
            ts.append(time.perf_counter() - t0)
            del vr
        best = min(ts)
        out["b%d" % bs] = {"wall_s": round(best, 4),
                           "fps": round(min(frames, n) / best, 1)}
    return out


def fullframe_roi_tax(video: str, roi, frames: int, decode: str) -> dict:
    """ROI-first 的价值：同窗口「ROI 解码」vs「全帧解码」。"""
    from decord import VideoReader, cpu
    res = {}
    for tag, use_roi in (("roi", True), ("full", False)):
        ctx = cpu(0)
        if decode == "nvdec":
            try:
                from decord import gpu
                ctx = gpu(0)
            except Exception:  # noqa: BLE001  # GPU 不可用则用 CPU 对照
                ctx = cpu(0)
        kw = dict(output_format="gray")
        if use_roi:
            kw.update(width=roi[2] - roi[0], height=roi[3] - roi[1], roi=roi)
        vr = VideoReader(video, ctx=ctx, **kw)
        n = min(frames, len(vr))
        t0 = time.perf_counter()
        for s in range(0, n, 64):
            vr.get_batch(list(range(s, min(s + 64, n)))).asnumpy()
        el = time.perf_counter() - t0
        res[tag] = {"wall_s": round(el, 4), "fps": round(n / el, 1)}
        del vr
    if res.get("full", {}).get("fps") and res.get("roi", {}).get("fps"):
        res["roi_speedup"] = round(
            res["roi"]["fps"] / res["full"]["fps"], 2)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5.mp4")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--decode", default="cpu", choices=["cpu", "nvdec"])
    ap.add_argument("--batches", default="64,128")
    args = ap.parse_args()

    roi = tuple(int(x) for x in args.roi.split(","))
    video = str(Path(os.environ.get("RACELOG_VIDEO_DIR",
                                    r"D:\Videos\racelog_test")) / args.video)
    is_h264 = "h264" in args.video or args.video in ("test5.mp4", "test3.mp4")

    res: dict = {"video": args.video, "frames": args.frames, "roi": list(roi)}
    print("视频 %s  窗口 %d 帧  ROI %s" % (args.video, args.frames, roi))

    print("\n① 外部参考实现（ffmpeg 自带解码器）天花板：")
    codec_name = "h264_cuvid" if is_h264 else "hevc_cuvid"
    ref = {}
    ref["ffmpeg_nvdec"] = ffmpeg_window(video, args.frames, "cuda", codec_name)
    ref["ffmpeg_sw"] = ffmpeg_window(video, args.frames, None, None)
    for k, v in ref.items():
        if "error" in v:
            print("   %-16s %s" % (k, v["error"]))
        else:
            print("   %-16s %8.4fs  %8.1f fps" % (k, v["wall_s"], v["fps"]))
    # ffmpeg 计时含进程启动（~0.15s），注明为**保守下限**
    res["ffmpeg_ref"] = ref

    print("\n② ROI-first 的价值（同窗口 ROI vs 全帧）：")
    tax = fullframe_roi_tax(video, roi, args.frames, args.decode)
    for k in ("roi", "full"):
        if k in tax:
            print("   %-16s %8.4fs  %8.1f fps" % (k, tax[k]["wall_s"], tax[k]["fps"]))
    if "roi_speedup" in tax:
        print("   ROI-first 加速 = %.2f×（通用库无此能力）" % tax["roi_speedup"])
    res["roi_tax"] = tax

    print("\n③ decord 批大小扫描（每帧固定开销占比）：")
    batches = [int(b) for b in args.batches.split(",")]
    scan = decord_roi(video, roi, args.frames, args.decode, batches)
    for k, v in scan.items():
        print("   %-16s %8.4fs  %8.1f fps" % (k, v["wall_s"], v["fps"]))
    res["decord_scan"] = scan

    out = ROOT / "bench" / "decode_ceiling.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
