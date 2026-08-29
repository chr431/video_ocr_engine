"""ROI 宽度自适应裁切的可行性量化（OCR 省计算）。

## 动机
字幕场景的 ROI 很宽（如整集 407×25），但绝大多数字幕不会占满宽度。
OCR 输入是 (3, 48, W) 且**整批 pad 到批内最大宽**——空白列照样参与
卷积计算。若能用分段时已有的 Otsu 二值图求出"有墨迹的列范围"，
裁掉两侧空白再喂 OCR，宽度越窄算力越省。

## 关键约束（决定这主意能不能成立）
1. `OCR_PAD_WIDTH_MIN = 224`（v6_small）：**窄图会被 pad 回 224**。
   且 engine_config 的注释明确说宽 pad 下更准（224→err 0.09%，
   48~96→0.69~1.19%）。所以只有"内容宽 > 224 且 < ROI 全宽"才有得赚，
   窄 ROI 应当自动失效。
2. `force_aspect > 0` 时宽度被强制，裁切不省宽（只改变缩放）→ 不适用。
3. 整批 pad 到最大宽：单条裁窄没用，要看**批内最大宽**下降多少。
   （必要时可按宽度排序分批，让窄的和窄的一起走。）

## 本探针回答
  Q1 代表帧的内容宽 / ROI 宽 分布如何？（有没有得裁）
  Q2 按批（B=16）聚合后，批内最大宽下降多少？（真实收益）
  Q3 OCR 输入宽（含 224 下限）实际下降多少？
  Q4 裁切会不会切掉字符？（左/右边缘留多少余量安全）

用法：
  python tools/_probe_roi_width.py --video X --roi a,b,c,d [--frames N] [--stride S]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine_config as config  # noqa: E402
from segmentation import _otsu  # noqa: E402
from video_ocr_engine import FieldExtractor  # noqa: E402


def ink_span(g: np.ndarray) -> tuple[int, int]:
    """Otsu 二值化后求"有墨迹"的列范围 [lo, hi)（闭开区间）。

    注意 rep_crop 的形状是 (H, W) 或 (H, W, 1)——**通道在最后**
    （首版误按 (C,H,W) 取 c[0]，得到 (W,1)，列数恒为 1 → 占比全 1.00）。
    动态范围过小（std < 3，纯黑/纯白帧）时 Otsu 阈值无意义 → 保守返回全宽。
    """
    if g.ndim == 3:
        g = g[..., 0] if g.shape[-1] == 1 else g.mean(axis=2)
    if float(g.std()) < 3.0:
        return 0, int(g.shape[1])
    th = _otsu(g)
    mask = (g > th) if _ink_is_bright(g, th) else (g <= th)
    colsum = mask.sum(axis=0).astype(np.int32)
    cols = np.nonzero(colsum >= 2)[0]
    if len(cols) == 0:
        return 0, int(g.shape[1])
    return int(cols[0]), int(cols[-1]) + 1


def _ink_is_bright(g: np.ndarray, th: float) -> bool:
    """判定前景是亮还是暗（字幕多为亮字暗底；也有反色）。"""
    return float((g > th).sum()) <= float((g <= th).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--dcd", type=int, default=0)
    ap.add_argument("--ocr-backend", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pad", type=int, default=0,
                    help="内容两侧保留的余量（像素），默认 0（容差另测）")
    a = ap.parse_args()

    roi = tuple(int(x) for x in a.roi.split(","))
    roi_w, roi_h = roi[2] - roi[0], roi[3] - roi[1]
    tgt_h = config.OCR_TARGET_H
    floor = config.OCR_PAD_WIDTH_MIN_BY_MODEL.get(
        "v6_small", config.OCR_PAD_WIDTH_MIN)
    w_full = int(round(tgt_h * roi_w / roi_h))

    print(f"== {Path(a.video).name} ROI={roi} ({roi_w}×{roi_h}) ==")
    print(f"   OCR 目标高 {tgt_h} → 全宽输入 w_full = {w_full}px"
          f"；pad 下限 {floor}px")
    print(f"   有效全宽 = max({floor}, {w_full}) = {max(floor, w_full)}px")

    ex = FieldExtractor(a.video, roi, frame_end=(a.frames or None),
                        sample_stride=a.stride,
                        decode_backend="cpu", ocr_backend=a.ocr_backend,
                        rep_crop_format="gray", keep_crops=True)
    r = ex.extract()
    crops = [s.rep_crop for s in r.segments if s.rep_crop is not None]
    print(f"   段数 {len(r.segments)}，取到代表帧 {len(crops)}")

    rows = []
    for c in crops:
        g = np.asarray(c)
        lo, hi = ink_span(g)
        pad = a.pad
        lo2, hi2 = max(0, lo - pad), min(g.shape[1], hi + pad)
        cw = hi2 - lo2
        rows.append({
            "roi_w": int(g.shape[1]),
            "content_w": int(cw),
            "frac": float(cw) / max(1, int(g.shape[1])),
            "w_ocr": max(floor, int(round(tgt_h * cw / roi_h))),
            "left": int(lo), "right": int(g.shape[1] - hi),
            "text": None,
        })

    fr = sorted(x["frac"] for x in rows)
    n = len(fr)
    q = lambda p: fr[min(n - 1, int(p * n))]  # noqa: E731
    print(f"\n[Q1] 内容宽 / ROI 宽 分布（n={n}）")
    print(f"   min={fr[0]:.2f}  p10={q(.10):.2f}  中位={statistics.median(fr):.2f}"
          f"  p90={q(.90):.2f}  max={fr[-1]:.2f}  均值={statistics.mean(fr):.2f}")

    B = a.batch
    w_eff_full = max(floor, w_full)
    batch_full, batch_crop = [], []
    for i in range(0, n, B):
        b = rows[i:i + B]
        batch_full.append(w_eff_full)
        batch_crop.append(max(x["w_ocr"] for x in b))
    sav = 1 - (sum(batch_crop) / sum(batch_full))
    print(f"\n[Q2/Q3] 按批 B={B} 聚合（共 {len(batch_full)} 批）")
    print(f"   批内最大宽：全宽恒 {w_eff_full}px → 裁切后"
          f" 中位 {int(statistics.median(batch_crop))}px、"
          f"均值 {statistics.mean(batch_crop):.0f}px")
    print(f"   → OCR 输入宽度总量下降 {sav:.1%}")

    # 若按宽度排序分批（窄的和窄的一起走）
    srt = sorted(rows, key=lambda x: x["w_ocr"])
    bc2 = [max(x["w_ocr"] for x in srt[i:i + B])
           for i in range(0, n, B)]
    sav2 = 1 - (sum(bc2) / sum(batch_full))
    print(f"   若按宽度排序分批：下降 {sav2:.1%}"
          f"（相对不排序 {(sav2 - sav) * 100:+.1f} 个百分点）")

    # Q4：边缘余量安全性——分布左/右空白
    lf = sorted(x["left"] for x in rows)
    rt = sorted(x["right"] for x in rows)
    print(f"\n[Q4] 墨迹距 ROI 边缘的空白（像素，原分辨率）")
    print(f"   左空白 中位 {int(statistics.median(lf))}  p90 "
          f"{lf[min(n-1,int(.9*n))]}  max {lf[-1]}")
    print(f"   右空白 中位 {int(statistics.median(rt))}  p90 "
          f"{rt[min(n-1,int(.9*n))]}  max {rt[-1]}")

    out = Path(__file__).with_name("_probe_roi_width.json")
    out.write_text(json.dumps({
        "video": a.video, "roi": roi, "roi_w": roi_w, "roi_h": roi_h,
        "w_full": w_full, "pad_floor": floor,
        "n_segments": n,
        "frac_median": statistics.median(fr),
        "frac_p10": q(.10), "frac_p90": q(.90),
        "batch_saving": sav, "batch_saving_sorted": sav2,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细落盘：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
