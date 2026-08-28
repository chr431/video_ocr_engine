"""临时探针：量化解码天花板与各路径的相位占比（分析用，非生产工具）。

目的：验证"decode 是否真的是墙钟主项"，以及"NVDEC 是否已到硬件像素率上限"。
测量：
  1. 视频元数据（分辨率/编码/fps/帧数）
  2. 纯解码吞吐：ROI-only vs 全帧；NVDEC vs CPU；stride 1 / 8
  3. 端到端 extract 的相位剖面
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def meta(path, gpu=True):
    from decord import VideoReader, cpu, gpu as _g
    import decord.video_reader as vrm
    roi_api = hasattr(vrm, '_CAPI_VideoReaderSetRoi')
    print(f"  ROI-first API: {roi_api}")
    for tag, ctx in (("NVDEC", _g(0)), ("CPU", cpu(0))):
        try:
            vr = VideoReader(path, ctx=ctx)
        except Exception as e:
            print(f"  {tag}: open fail {e}")
            continue
        n = len(vr)
        try:
            fps = vr.get_avg_fps()
        except Exception:
            fps = 0
        try:
            codec = vr.get_codec()
        except Exception:
            codec = "?"
        # 全帧尺寸：读一帧
        try:
            f = vr[0].asnumpy()
            shape = f.shape
        except Exception:
            shape = "?"
        print(f"  {tag}: n={n} fps={fps:.2f} codec={codec} frame={shape}")
        del vr
    return n


def bench_decode(path, roi, nframes, stride, ctx_tag, with_roi=True):
    """纯解码吞吐（不分段不 OCR）：只做 get_batch + asnumpy。"""
    from decord import VideoReader, cpu, gpu as _g
    import decord.video_reader as vrm
    roi_api = hasattr(vrm, '_CAPI_VideoReaderSetRoi')
    ctx = _g(0) if ctx_tag == "NVDEC" else cpu(0)
    kw = {}
    if with_roi and roi_api:
        kw["roi"] = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)
    vr = VideoReader(path, ctx=ctx, **kw)
    total = len(vr)
    n = min(nframes, total)
    frames = list(range(0, n, stride))
    # warmup
    vr.get_batch(frames[:16], **({"roi": kw["roi"]} if "roi" in kw else {})).asnumpy()
    t0 = time.perf_counter()
    B = 64
    for i in range(0, len(frames), B):
        vr.get_batch(frames[i:i + B], **({"roi": kw["roi"]} if "roi" in kw else {})).asnumpy()
    dt = time.perf_counter() - t0
    del vr
    rate = len(frames) / dt
    print(f"    {ctx_tag:6s} roi={'Y' if with_roi else 'N'} stride={stride}: "
          f"{len(frames)} 帧 / {dt:.3f}s = {rate:.0f} fps")
    return rate


def bench_full(path, roi, nframes, stride, backend, ocr):
    """端到端 extract（含分段+OCR）。"""
    os.environ.setdefault("ENGINE_PROFILE", "1")
    import engine_config  # noqa
    if os.environ.get("ENGINE_PROFILE") != "1":
        os.environ["ENGINE_PROFILE"] = "1"
    from video_ocr_engine import FieldExtractor
    ex = FieldExtractor(path, roi, frame_end=nframes,
                        decode_backend=backend, ocr_backend=ocr,
                        sample_stride=stride)
    t0 = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t0
    print(f"    [{backend}/{ocr}] wall={wall:.3f}s segs={len(r.segments)}")
    print(f"      timing: { {k: round(v,3) for k,v in ex.timing.items()} }")
    if ex.profile:
        for g, d in ex.profile.items():
            print(f"      {g}: { {k: round(v,3) for k,v in sorted(d.items(), key=lambda kv:-kv[1])} }")
    return wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test5.mp4")
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--full", action="store_true", help="跑端到端 + 剖面")
    args = ap.parse_args()
    roi = tuple(int(x) for x in args.roi.split(","))

    print(f"=== 元数据 {os.path.basename(args.video)} ===")
    meta(args.video)
    print("=== 纯解码吞吐（无分段/OCR）===")
    for tag in ("NVDEC", "CPU"):
        for wr in (True, False):
            for st in (1, 8):
                try:
                    bench_decode(args.video, roi, args.frames, st, tag, wr)
                except Exception as e:
                    print(f"    {tag} roi={wr} stride={st}: FAIL {type(e).__name__} {e}")
    if args.full:
        print("=== 端到端 ===")
        for be, oc in (("auto", "auto"), ("cpu", "cpu")):
            try:
                bench_full(args.video, roi, args.frames, args.stride, be, oc)
            except Exception as e:
                print(f"    {be}/{oc}: FAIL {type(e).__name__} {e}")


if __name__ == "__main__":
    main()
