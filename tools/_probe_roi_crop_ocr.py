"""ROI 宽度自适应裁切：OCR 侧收益与**文本一致性**的离线对照实验。

## 问题
字幕 ROI 很宽（整集 407×25 → OCR 输入 781px），但字幕常常不占满宽度。
空白列照样参与卷积。裁掉空白再喂 OCR 能省多少？会不会切坏字符？

## 关键约束
`OcrEngine.__call__` 的 pad 宽度 = `max(224/48, 批内最大宽高比)`，
且它**已经在批内按宽度排序**——排序只优化 host resize 顺序，**不改变 pad 宽度**。
所以只有**跨批**把宽度相近的段分到同一批，pad 宽度才会真的降下来。

## 三种批处理
  A 全宽 + 顺序分批（现役）
  B 内容裁切 + 顺序分批（裁了，但每批仍被满宽成员顶上去）
  C 内容裁切 + 按宽度排序分批（跨批分组）

## 正确性门
三种模式的**每段文本 + 置信度**必须与 A 一致。
裁切会改变 resize 后的宽度与 pad，因此结果**可能**变化——变了就说明这优化有损。

用法：
  python tools/_probe_roi_crop_ocr.py --video X --roi a,b,c,d [--frames N]
      [--engine tensorrt|onnxruntime] [--batch 16] [--margin 0] [--reps 3]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine_config as config  # noqa: E402
from ocr_native import OcrEngine  # noqa: E402
from segmentation import _otsu  # noqa: E402
from video_ocr_engine import FieldExtractor  # noqa: E402


def ink_span(g: np.ndarray) -> tuple[int, int]:
    """内容列范围 [lo, hi)：Otsu 二值化 + 列投影（每列墨迹 ≥2 才算）。

    rep_crop 形状 (H, W) 或 (H, W, 1)——通道在最后。
    动态范围过小（std<3）时 Otsu 无意义 → 保守返回全宽。
    """
    if g.ndim == 3:
        g = g[..., 0] if g.shape[-1] == 1 else g.mean(axis=2)
    if float(g.std()) < 3.0:
        return 0, int(g.shape[1])
    th = _otsu(g)
    bright = float((g > th).sum())
    mask = (g > th) if bright <= float((g <= th).sum()) else (g <= th)
    cols = np.nonzero(mask.sum(axis=0) >= 2)[0]
    if len(cols) == 0:
        return 0, int(g.shape[1])
    return int(cols[0]), int(cols[-1]) + 1


def run_batches(eng, imgs, batch: int, order: list[int] | None = None):
    """按给定（或自然）顺序分批推理，返回 (text, conf) 列表 + 耗时。"""
    idxs = order if order is not None else list(range(len(imgs)))
    out: list = [None] * len(imgs)
    t0 = time.perf_counter()
    for i in range(0, len(idxs), batch):
        chunk = idxs[i:i + batch]
        res = eng([imgs[j] for j in chunk])
        for k, j in enumerate(chunk):
            out[j] = (res[k].txts[0], round(float(res[k].scores[0]), 5))
    return out, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--engine", default="tensorrt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--margin", type=int, default=0,
                    help="内容两侧额外保留的列数（防切边）")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    roi = tuple(int(x) for x in a.roi.split(","))
    print(f"== {Path(a.video).name} ROI={roi} engine={a.engine} ==")

    ex = FieldExtractor(a.video, roi, frame_end=(a.frames or None),
                        sample_stride=a.stride, decode_backend="cpu",
                        ocr_backend="auto", rep_crop_format="gray",
                        keep_crops=True)
    r = ex.extract()
    crops = [s.rep_crop for s in r.segments if s.rep_crop is not None]
    n = len(crops)
    roi_h, roi_w = crops[0].shape[0], crops[0].shape[1]
    tgt = config.OCR_TARGET_H
    floor = config.OCR_PAD_WIDTH_MIN_BY_MODEL.get("v6_small",
                                                 config.OCR_PAD_WIDTH_MIN)
    print(f"   段数 {n}  crop {roi_h}×{roi_w}  全宽 OCR 输入 "
          f"{int(round(tgt * roi_w / roi_h))}px  pad 下限 {floor}px")

    spans = [ink_span(c) for c in crops]
    m = a.margin
    spans = [(max(0, lo - m), min(roi_w, hi + m)) for lo, hi in spans]
    cw = [hi - lo for lo, hi in spans]
    fr = sorted(x / roi_w for x in cw)
    print(f"   内容宽/ROI宽：min {fr[0]:.2f}  p10 "
          f"{fr[int(.1 * n)]:.2f}  中位 {statistics.median(fr):.2f}  "
          f"p90 {fr[int(.9 * n)]:.2f}  max {fr[-1]:.2f}")

    # 必须复现生产链路：先过 _preprocess_standard（缩放到 OCR_TARGET_H 高），
    # 再喂 OcrEngine。直接把原始 crop 喂进去会让 _call_trt_gpu 用 crop 高度
    # 当 h0 → 模型收到 (3,26,408) → TRT 输出 seq=0 直接崩。
    from video_utils import _preprocess_standard
    procs_full = [_preprocess_standard(c) for c in crops]
    procs_crop = [_preprocess_standard(crops[i][:, spans[i][0]:spans[i][1], :])
                  for i in range(n)]

    eng = OcrEngine(config.DEFAULT_OCR_MODEL, a.engine)

    # 预热（首次推理含 CUDA/引擎初始化）
    eng([procs_full[0]])

    B = a.batch
    res: dict[str, tuple[list, float]] = {}
    # A：全宽 + 顺序
    res["A 全宽+顺序"] = min((run_batches(eng, procs_full, B)
                            for _ in range(a.reps)), key=lambda x: x[1])
    # B：裁切 + 顺序
    res["B 裁切+顺序"] = min((run_batches(eng, procs_crop, B)
                            for _ in range(a.reps)), key=lambda x: x[1])
    # C：裁切 + 按宽排序分批
    order = sorted(range(n), key=lambda i: cw[i])
    res["C 裁切+按宽排序"] = min(
        (run_batches(eng, procs_crop, B, order)
         for _ in range(a.reps)), key=lambda x: x[1])

    base_t = res["A 全宽+顺序"][1]
    print(f"\n   {'模式':<18s} {'耗时':>8s}  {'相对A':>8s}  {'与A文本一致':>12s}")
    ref = res["A 全宽+顺序"][0]
    for name, (out, t) in res.items():
        same = sum(1 for x, y in zip(ref, out) if x == y)
        print(f"   {name:<18s} {t:7.3f}s  {(t / base_t - 1) * 100:+7.1f}%  "
              f"{same}/{n} ({same / n:.1%})")

    # 逐段统计 pad 宽度（解释收益来源）
    def pad_sum(widths, order=None):
        idxs = order if order is not None else list(range(n))
        s = 0
        for i in range(0, len(idxs), B):
            mx = max(widths[j] for j in idxs[i:i + B])
            s += max(floor, int(round(tgt * mx / roi_h))) * len(idxs[i:i + B])
        return s

    w_full = pad_sum([roi_w] * n)
    w_crop = pad_sum(cw)
    w_sort = pad_sum(cw, order)
    print(f"\n   pad 像素总量（宽×批内条数）：全宽 {w_full/1e6:.2f}M → "
          f"裁切 {w_crop/1e6:.2f}M ({(w_crop/w_full-1)*100:+.1f}%) → "
          f"裁切+排序 {w_sort/1e6:.2f}M ({(w_sort/w_full-1)*100:+.1f}%)")

    out_p = Path(__file__).with_name("_probe_roi_crop_ocr.json")
    out_p.write_text(json.dumps({
        "video": a.video, "roi": roi, "engine": a.engine, "batch": B,
        "n": n, "roi_w": int(roi_w), "roi_h": int(roi_h),
        "content_frac_median": statistics.median(fr),
        "times": {k: round(v[1], 3) for k, v in res.items()},
        "agree_with_A": {k: sum(1 for x, y in zip(ref, v[0]) if x == y)
                         for k, v in res.items()},
        "pad_px": {"full": w_full, "crop": w_crop, "crop_sorted": w_sort},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细落盘：{out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
