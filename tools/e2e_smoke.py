"""引擎端到端冒烟验收 + 性能工具（真实视频，可作 CI 入口）。

功能矩阵（同一视频窗口 × 配置组合）：
  1. gpu_yuv   decode=auto ocr=auto rep=yuv  —— GPU 零拷贝管线（默认路线）
  2. gpu_gray  decode=auto ocr=auto rep=gray —— GPU 管线 + raw 直通（灰度帧）
  3. gpu_keepoff decode=auto ocr=auto rep=gray keep_crops=False（零 D2H 输出）
  4. host_trt  decode=auto ocr=auto rep=yuv force_aspect=1.5 —— 宿主+TRT
     （truth 对齐口径；force_aspect 使 GPU 管线门控回退宿主）
  5. host_cpu  decode=cpu  ocr=cpu  rep=yuv —— 全宿主 CPU 软解 + ONNX
  6. hybrid    decode=hybrid ocr=auto rep=yuv —— CPU+NVDEC 双解码生产者
  7. gpu_onnx  decode=auto ocr=cpu  rep=yuv —— GPU 分段 + ONNX 宿主 OCR

用法：
  python tools/e2e_smoke.py --video D:\\Videos\\racelog_test\\test5.mp4 \\
      --roi 843,993,948,1025 [--truth ...csv] [--frames 5000] [--stride 8]
  python tools/e2e_smoke.py --video X --roi A --probe        # 只探元数据
  python tools/e2e_smoke.py --video X --roi A --configs gpu_yuv,host_cpu
  python tools/e2e_smoke.py --video X --roi A --perf --runs 3 --frames 3000

--truth CSV：Race 输出格式（'#' 头可含 roi/frame_start/frame_end/force_aspect/
fps/codec），数据行 frame,text,...。--roi-from-truth 自动读头；--verify 按
帧对齐比较文本（归一化后全等或数值近似），--min-match 为最低匹配率（0-1）。

退出码：0=通过；1=功能/性能校验失败（墙钟异常）；2=参数错误。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from video_ocr_engine import FieldExtractor  # noqa: E402
from video_utils import nvdec_available, tensorrt_available  # noqa: E402

CONFIGS = {
    "gpu_yuv": dict(ocr_backend="auto", rep_crop_format="yuv"),
    "gpu_gray": dict(ocr_backend="auto", rep_crop_format="gray"),
    "gpu_keepoff": dict(ocr_backend="auto", rep_crop_format="gray",
                        keep_crops=False),
    "host_trt": dict(ocr_backend="auto", rep_crop_format="yuv",
                     force_aspect=1.5),
    "host_cpu": dict(decode_backend="cpu", ocr_backend="cpu",
                     rep_crop_format="yuv"),
    "hybrid": dict(decode_backend="hybrid", rep_crop_format="yuv"),
    "gpu_onnx": dict(ocr_backend="cpu", rep_crop_format="yuv"),
}


def _norm_text(t) -> str:
    if not t:
        return ""
    return "".join(str(t).split())


def _text_match(a: str, b: str) -> bool:
    a, b = _norm_text(a), _norm_text(b)
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        # 数值近似（速度读数常见 OCR 尾位误差：0.00 vs 0）
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= max(1e-3, 0.02 * max(abs(fa), abs(fb), 1.0))
    except ValueError:
        return a[:1] == b[:1] and len(a) <= len(b) + 2


def parse_truth(path: Path):
    """Race CSV：解析 '#' 头元数据 + (frame, text) 行。"""
    meta, rows = {}, []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith("#"):
                m = line[1:].strip().split("=")
                for kv in line.strip().lstrip("#").split(","):
                    kv = kv.strip()
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        try:
                            meta[k.strip()] = int(v)
                        except ValueError:
                            try:
                                meta[k.strip()] = float(v)
                            except ValueError:
                                meta[k.strip()] = v
                    else:
                        pass
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0].lstrip("-").isdigit():
                # Race CSV 数据行：frame,time_sec,text,conf
                rows.append((int(parts[0]), parts[2]))
            elif len(parts) == 2 and parts[0].lstrip("-").isdigit():
                rows.append((int(parts[0]), parts[1]))
    return meta, rows


def run_once(video, roi, frames, stride, cfg, truth_meta):
    kwargs = dict(CONFIGS[cfg])
    if truth_meta and cfg.startswith("host_trt") and "force_aspect" not in kwargs:
        pass  # host_trt 固定 force_aspect=1.5
    ex = FieldExtractor(
        video, roi,
        frame_start=truth_meta.get("frame_start", 0),
        frame_end=(truth_meta.get("frame_start", 0) + frames
                   if truth_meta.get("frame_start") is not None else frames),
        sample_stride=stride, keep_frames=True, **kwargs)
    t0 = time.perf_counter()
    result = ex.extract()
    wall = time.perf_counter() - t0
    return ex, result, wall


def check_result(ex, res, cfg, roi, stride):
    errs = []
    segs = res.segments
    if not segs:
        errs.append(f"{cfg}: 无分段（视频窗口/ROI 可能无文本变化）")
    ys = [s.start for s in segs]
    if any(b < a for a, b in zip(ys[:-1], ys[1:])):
        errs.append(f"{cfg}: 段序非单调")
    if segs and (segs[0].start < (0 if not ex._frame_start else ex._frame_start)):
        errs.append(f"{cfg}: 段首帧越界 {segs[0].start}")
    empty = sum(1 for s in segs if not s.text)
    if segs and empty / len(segs) > 0.5:
        errs.append(f"{cfg}: OCR 空文本占比过高 {empty}/{len(segs)}")
    for s in segs:
        if not (0.0 <= s.confidence <= 1.0):
            errs.append(f"{cfg}: confidence 越界 {s.confidence}")
            break
    # 代表帧格式
    for s in segs[:5]:
        c = s.rep_crop
        if c is not None:
            if ex._yuv_output:
                h, w = roi[3] - roi[1] + 1, roi[2] - roi[0] + 1
                if c.ndim != 2 or c.shape[0] != h + (h + 1) // 2 or c.shape[1] != w:
                    errs.append(f"{cfg}: yuv rep_crop 形状异常 {c.shape} 期望 "
                                f"({h + (h + 1) // 2},{w})")
                    break
            elif c.ndim not in (2, 3) or c.shape[-3:-1] != (
                    roi[3] - roi[1] + 1, roi[2] - roi[0] + 1) and c.shape[:2] != (
                    roi[3] - roi[1] + 1, roi[2] - roi[0] + 1):
                errs.append(f"{cfg}: gray rep_crop 形状异常 {c.shape}")
                break
    return errs


def verify_truth(res, meta, rows, min_match):
    segs = [s for s in res.segments if s.text]
    truth_by_frame = {f: t for f, t in rows}
    matched = 0
    examples = []
    for s in segs:
        t = truth_by_frame.get(s.start)
        if t is None:
            t = min(truth_by_frame.items(),
                    key=lambda kv: abs(kv[0] - s.start))[1]
        if _text_match(s.text, t):
            matched += 1
        elif len(examples) < 6:
            examples.append((s.start, s.text, t))
    ratio = matched / max(1, len(segs))
    ok = ratio >= min_match
    return ratio, ok, examples


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", help="x1,y1,x2,y2（缺省从 --truth 头读）")
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--roi-from-truth", action="store_true")
    ap.add_argument("--frames", type=int, default=3000, help="采样源帧窗口长")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="功能矩阵配置名，逗号分隔")
    ap.add_argument("--probe", action="store_true", help="只打印视频元数据")
    ap.add_argument("--verify", action="store_true", help="与 --truth 对比匹配率")
    ap.add_argument("--perf", action="store_true", help="性能测试")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--min-match", type=float, default=0.9)
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"视频不存在: {video}")
        return 2

    truth_meta, truth_rows = ({}, [])
    if args.truth:
        truth_meta, truth_rows = parse_truth(args.truth)
        print(f"truth 元数据: roi={truth_meta.get('roi')} "
              f"frames={truth_meta.get('frame_start')}.."
              f"{truth_meta.get('frame_end')} codec={truth_meta.get('codec')} "
              f"fps={truth_meta.get('fps')} 行数={len(truth_rows)}")

    roi = None
    if args.roi:
        roi = tuple(int(v) for v in args.roi.split(","))
    elif args.roi_from_truth and truth_meta.get("roi"):
        roi = tuple(int(v) for v in str(truth_meta["roi"]).split(","))
    if len(roi or ()) != 4:
        print("需要 --roi 'x1,y1,x2,y2' 或 --roi-from-truth")
        return 2

    # ── probe：解码器元数据 + 后端可用性 ──
    if args.probe:
        from decord import VideoReader, cpu
        vr = VideoReader(str(video), ctx=cpu(0), num_threads=4)
        print(f"视频: {video.name}")
        print(f"  frames={len(vr)} codec={vr.get_codec()} "
              f"avg_fps={vr.get_avg_fps()} ")
        try:
            import decord.video_reader as _vrm
            print("  roi_api=", hasattr(_vrm, "_CAPI_VideoReaderSetRoi"))
        except Exception:
            pass
        print(f"  nvdec_available={nvdec_available(str(video))} "
              f"tensorrt_available={tensorrt_available()}")
        return 0

    cfgs = [c.strip() for c in args.configs.split(",") if c.strip() in CONFIGS]
    unknown = [c.strip() for c in args.configs.split(",")
               if c.strip() and c.strip() not in CONFIGS]
    if unknown:
        print(f"未知配置: {unknown}（可选 {list(CONFIGS)}）")
        return 2

    frame_start = int(truth_meta.get("frame_start", 0))
    frame_end = min(frame_start + args.frames,
                    int(truth_meta.get("frame_end", 1 << 60)))
    print(f"\n== 功能矩阵 ==  roi={roi} frames=[{frame_start},{frame_end}) "
          f"stride={args.stride}")

    all_ok = True
    base_texts = None
    results = {}
    for cfg in cfgs:
        if cfg == "hybrid" and args.stride != 1:
            print(f"  [hybrid] 注：混合解码仅 stride==1 激活（安全门），"
                  f"当前 stride={args.stride} → 按纯 GPU 跑")
        try:
            ex, res, wall = run_once(str(video), roi, frame_end - frame_start,
                                     args.stride, cfg, truth_meta)
        except Exception as e:
            all_ok = False
            import traceback
            traceback.print_exc()
            print(f"  [{cfg}] 失败: {type(e).__name__}: {e}")
            if not args.keep_going:
                break
            continue
        errs = check_result(ex, res, cfg, roi, args.stride)
        texts = sorted({s.text for s in res.segments if s.text})
        print(f"  [{cfg}] 段={len(res.segments)} 文本={len(texts)} "
              f"墙钟={wall:.2f}s 后端={res.meta.get('backend')} "
              f"OCR={res.meta.get('ocr_backend')} "
              f"空文本={sum(1 for s in res.segments if not s.text)}")
        for e in errs:
            all_ok = False
            print(f"      ✗ {e}")
        if base_texts is None and cfgs.index(cfg) == 0:
            base_texts = set(texts)
        elif base_texts is not None:
            inter = len(base_texts & set(texts))
            cov = inter / max(1, len(base_texts))
            print(f"      与首配置唯一文本重合率={cov:.1%}")
            if cov < 0.9:
                all_ok = False
                print("      ✗ 跨路径文本一致性过低")
        results[cfg] = res

    # ── verify：与 ground truth 逐段对比 ──
    if args.verify and truth_rows:
        print("\n== 文本验证 vs ground truth ==")
        for cfg in (cfgs or [list(results)[0]]):
            if cfg not in results:
                continue
            ratio, ok, examples = verify_truth(results[cfg], truth_meta,
                                               truth_rows, args.min_match)
            all_ok &= ok
            print(f"  [{cfg}] 匹配率={ratio:.1%}"
                  f"{' ✓' if ok else ' ✗ (<' + str(args.min_match) + ')'}")
            for fr, got, want in examples:
                print(f"      帧{fr}: 提取={got!r} 真值={want!r}")

    # ── perf：重复跑 + 分相剖面 ──
    if args.perf:
        perf_cfg = args.configs.split(",")[0] if not cfgs else cfgs[0]
        print(f"\n== 性能测试 ==  配置={perf_cfg} runs={args.runs} "
              f"frames={frame_end - frame_start} stride={args.stride}")
        walls, timings = [], []
        for i in range(args.runs):
            enve = os.environ.get("ENGINE_PROFILE")
            if i == args.runs - 1:
                os.environ["ENGINE_PROFILE"] = "1"
            try:
                ex, res, wall = run_once(str(video), roi,
                                         frame_end - frame_start, args.stride,
                                         perf_cfg, truth_meta)
            finally:
                if enve is None:
                    os.environ.pop("ENGINE_PROFILE", None)
                else:
                    os.environ["ENGINE_PROFILE"] = enve
            walls.append(wall)
            timings.append(dict(res.timing))
            print(f"  run{i + 1}: 墙钟={wall:.3f}s "
                  f"分段={len(res.segments)} 时序={ {k: round(v, 3) for k, v in res.timing.items()} }")
        print(f"  中位数墙钟={statistics.median(walls):.3f}s "
              f"min={min(walls):.3f}s max={max(walls):.3f}s")
        base = float(truth_meta.get("fps") or 0) or max(1.0, res.fps)
        n_src = frame_end - frame_start
        print(f"  采样帧={n_src // args.stride} 等效源帧吞吐≈"
              f"{n_src / statistics.median(walls):.0f} fps(源帧) "
              f"@视频 {base:.1f} fps")
    print("\n== 结论 ==")
    print("  PASS" if all_ok else "  FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
