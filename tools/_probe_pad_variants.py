r"""留白 / 缩放 / 裁切顺序 变体对照（离线，一次抽取 + 多次 OCR）。

## 背景
生产（fa>0）下 pad 越大越准（160→26 / 224→7 / 320→2），但 fa=0 下相反。
说明**留白本身在起作用**，于是要弄清三件事：

1. **留白放哪边**？现役固定放右侧（`_resize_norm`: pad[:, :, :resized_w] = resized）。
   左侧 / 均匀（居中）会不会更好？
2. **标清 ROI 高 < 48 时，不缩放只加留白**是否有用？现役把 33px 高的 crop
   **放大**到 48（插值模糊）；改为原像素放进 48 高画布 + 纵向留白会怎样？
   （test5 h=33、test6 h=32、test2 h=42 都 < 48，都有真值，可直接判。）
3. **裁切能否在 fa>0 下也生效**：顺序是「先裁再定比例」还是「先定比例再裁」。

## 方法
一次抽取拿到段代表帧 crop，之后对每个变体单独做预处理 + OCR，避免
重复解码。判定用生产口径：**段代表帧 + 数值 tol=1 误读数**。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

GT = Path(r"D:\Videos\racelog_test\ground_truth_csv")
VID = Path(r"D:\Videos\racelog_test")

# ── 抽取：一次拿到 (rep_frame, crop_path) ──
EXTRACT = r"""
import sys, json, os
sys.path.insert(0, r"D:\Repo\video_ocr_engine")
import numpy as np
from video_ocr_engine import FieldExtractor
vp, roi_s, s, e, stride, fa, out = sys.argv[1:8]
roi = tuple(int(x) for x in roi_s.split(','))
ex = FieldExtractor(vp, roi, frame_start=int(s), frame_end=int(e),
                    sample_stride=int(stride), decode_backend='auto',
                    ocr_backend='auto', rep_crop_format='gray', keep_crops=True,
                    force_aspect=float(fa))
r = ex.extract()
rows = []
crops = []
for seg in r.segments:
    c = seg.rep_crop
    if c is None:
        continue
    g = c[..., 0] if c.ndim == 3 else c
    crops.append(np.ascontiguousarray(g))
    rows.append(int(seg.rep_frame if seg.rep_frame is not None else seg.start))
np.savez_compressed(out, frames=np.array(rows, dtype=np.int64),
                    **{f"c{i}": a for i, a in enumerate(crops)})
print(json.dumps({"n": len(rows)}))
"""

# ── 变体评估：读 npz，按变体预处理 + OCR，与真值比对 ──
EVAL = r"""
import sys, json, os, math
sys.path.insert(0, r"D:\Repo\video_ocr_engine")
import numpy as np
import engine_config as config
from video_utils import _preprocess_standard, _np_resize
from ocr_native import OcrEngine

npz, truth_json, tol, fa, pad_width, spec = sys.argv[1:7]
tol = float(tol); fa = float(fa); pad_width = int(pad_width)
spec = json.loads(spec)
truth = {int(k): v for k, v in json.load(open(truth_json, encoding='utf-8')).items()}

z = np.load(npz)
frames = z["frames"].tolist()
crops = [z[f"c{i}"] for i in range(len(frames))]

# ── 变体参数 ──
pad_side = spec.get("pad", "right")        # right | left | center
vresize = spec.get("vresize", True)        # True=缩放到48高；False=原像素不缩放
vpad = spec.get("vpad", "bottom")          # bottom | center（vresize=False 时纵向放哪）
crop_content = spec.get("crop", False)     # 是否按内容列裁切
crop_order = spec.get("order", "pre")      # pre=先裁再定比例；post=先定比例再裁
margin_pct = spec.get("margin", 10)

def content_span(g, margin_pct):
    # 内容列范围：Otsu 二值化 + 列投影（与 _crop_to_content 同判据）
    if float(g.std()) < 3.0:
        return 0, int(g.shape[1])
    # Otsu
    hist, _ = np.histogram(g.ravel(), bins=256, range=(0, 256))
    tot = hist.sum()
    w_ = np.arange(256)
    sum_b = 0.0; w_b = 0; max_v = -1.0; th = 0
    sum_all = float((hist * w_).sum())
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = tot - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        v = w_b * w_f * (m_b - m_f) ** 2
        if v > max_v:
            max_v = v; th = t
    bright = float((g > th).sum())
    mask = (g > th) if bright <= float((g <= th).sum()) else (g <= th)
    cols = np.nonzero(mask.sum(axis=0) >= 2)[0]
    if len(cols) == 0:
        return 0, int(g.shape[1])
    m = max(1, int(round(g.shape[1] * margin_pct / 100.0)))
    return max(0, int(cols[0]) - m), min(int(g.shape[1]), int(cols[-1]) + 1 + m)

def prep(g):
    # 返回 (3, 48, W) float32 —— 已按变体规则放置到 48 高画布
    if crop_content and crop_order == "pre":
        lo, hi = content_span(g, margin_pct)
        g = g[:, lo:hi]
    if vresize:
        img = _preprocess_standard(g, force_aspect=fa)      # 缩放到 48 高
    else:
        new_h = min(int(g.shape[0]), config.OCR_TARGET_H)
        if fa > 0:
            new_w = max(1, int(round(config.OCR_TARGET_H * fa)))
        else:
            new_w = max(1, int(round(g.shape[1] * new_h / g.shape[0])))
        img = _np_resize(g, new_w, new_h) if (new_h != g.shape[0]
                                              or new_w != g.shape[1]) \
            else g.astype(np.float32)
        if crop_content and crop_order == "post":
            lo, hi = content_span(img, margin_pct)
            img = img[:, lo:hi]
        # 纵向放到 48 高画布
        h = img.shape[0]
        canvas = np.zeros((config.OCR_TARGET_H, img.shape[1], 3), dtype=np.float32)
        if vpad == "center":
            top = (config.OCR_TARGET_H - h) // 2
        else:
            top = 0
        if img.ndim == 2:
            for c in range(3):
                canvas[top:top + h, :, c] = img
        else:
            canvas[top:top + h, :, :img.shape[2]] = img
        out = canvas.transpose((2, 0, 1)).astype(np.float32)
        return (out / 255.0 - 0.5) / 0.5
    if crop_content and crop_order == "post":
        # 已在 48 高：按通道维度裁（img 是 H,W,3）
        lo, hi = content_span(img[..., 0], margin_pct)
        img = img[:, lo:hi]
    # ⚠️ _preprocess_standard / _np_resize 输出仍是 0..255，
    # 必须补上与 _resize_norm 相同的归一化，否则全批识别成垃圾。
    out = img.transpose((2, 0, 1)).astype(np.float32)
    out = (out / 255.0 - 0.5) / 0.5
    return out

imgs = [prep(g) for g in crops]
max_w = max(im.shape[2] for im in imgs)
if pad_width > 0:
    max_w = max(max_w, pad_width)

def place(im, max_w):
    # 按 pad_side 把 (3,48,w) 放到 (3,48,max_w)
    out = np.zeros((3, im.shape[1], max_w), dtype=np.float32)
    w = min(im.shape[2], max_w)
    if pad_side == "right":
        off = 0
    elif pad_side == "left":
        off = max_w - w
    else:                       # center
        off = (max_w - w) // 2
    out[:, :, off:off + w] = im[:, :, :w]
    return out

eng = OcrEngine(config.DEFAULT_OCR_MODEL, 'tensorrt')
B = 16
res = {}
order = list(range(len(imgs)))
for i in range(0, len(order), B):
    ch = order[i:i + B]
    batch = np.stack([place(imgs[j], max_w) for j in ch])
    r = eng.call_model(batch) if hasattr(eng, 'call_model') else None
    if r is None:
        # 走公开路径：__call__ 内部会再 pad，这里已 pad 好，直接推理
        r = eng._infer(batch)
        from ocr_native import RecOut
        if r.ndim == 3:
            outs = eng._ctc_decode_batch(r)
        else:
            outs = [eng._ctc_decode(x) for x in r]
    else:
        outs = r
    for k, j in enumerate(ch):
        o = outs[k]
        res[frames[j]] = (o.txts[0] if o.txts else "",
                          float(o.scores[0]) if o.scores else 0.0)

def num(x):
    try:
        return float(x)
    except Exception:
        return None

bad = 0; n = 0; conf = []
for f, (txt, sc) in res.items():
    if f not in truth:
        continue
    n += 1
    conf.append(sc)
    fx, fy = num(txt), num(truth[f])
    ok = (abs(fx - fy) <= tol) if (fx is not None and fy is not None) \
        else (txt == truth[f])
    if not ok:
        bad += 1
print(json.dumps({"n": n, "misread": bad,
                  "mean_conf": round(sum(conf) / len(conf), 5) if conf else 0.0,
                  "max_w": int(max_w)}))
"""


def truth_meta(p: Path) -> dict:
    out: dict = {}
    for line in open(p, encoding="utf-8-sig"):
        if not line.startswith("#"):
            break
        m = re.search(r"roi\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", line)
        if m and "roi" not in out:
            out["roi"] = ",".join(m.groups())
        for key in ("frame_start", "frame_end", "force_aspect", "fill_width"):
            m = re.search(rf"{key}\s*=\s*(-?[\d.]+)", line)
            if m:
                v = float(m.group(1))
                if key in ("frame_start", "frame_end", "fill_width"):
                    out[key] = int(v)
                else:
                    out[key] = v
    return out


def load_truth(p: Path) -> dict:
    o = {}
    for line in open(p, encoding="utf-8-sig"):
        line = line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) >= 3 and parts[0].lstrip("-").isdigit():
            o[int(parts[0])] = parts[2].strip()
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5")
    ap.add_argument("--force-aspect", type=float, default=None,
                    help="默认跟随真值头；0 表示强制关掉")
    ap.add_argument("--pad-width", type=int, default=None,
                    help="默认跟随真值头 fill_width（生产口径），未指定则 224")
    ap.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args()

    tp = GT / f"{a.video}_ref.csv"
    if not tp.exists():
        tp = GT / f"{a.video}_truth.csv"
    meta = truth_meta(tp)
    truth = load_truth(tp)
    fa = a.force_aspect if a.force_aspect is not None else \
        float(meta.get("force_aspect", 0.0))
    # 生产口径：pad 宽跟随真值头的 fill_width（test2 是 320，其余多为 224）
    pad_w = a.pad_width if a.pad_width is not None else \
        int(meta.get("fill_width") or 224)
    roi = meta["roi"]
    s, e = meta.get("frame_start", 0), meta.get("frame_end", 0)
    print(f"=== {a.video}  roi={roi} (h={int(roi.split(',')[3]) - int(roi.split(',')[1])})"
          f"  force_aspect={fa}  pad_width={pad_w}"
          f"{'(真值头)' if a.pad_width is None else '(命令行)'}  tol={a.tol} ===")

    npz = Path(__file__).with_name(f"_crops_{a.video}.npz")
    if not npz.exists():
        print("  抽取代表帧 crop ...")
        p = subprocess.run(
            [sys.executable, "-c", EXTRACT, str(VID / f"{a.video}.mp4"), roi,
             str(s), str(e), "1", str(fa), str(npz)],
            capture_output=True, text=True)
        if p.returncode != 0:
            print((p.stderr or "").strip()[-400:])
            return 2
    tj = Path(__file__).with_name(f"_truth_{a.video}.json")
    tj.write_text(json.dumps(truth, ensure_ascii=False), encoding="utf-8")

    variants = [
        ("① 右pad(现役)", {"pad": "right", "vresize": True}),
        ("② 左pad", {"pad": "left", "vresize": True}),
        ("③ 均匀pad(居中)", {"pad": "center", "vresize": True}),
        ("④ 不缩放+右下", {"pad": "right", "vresize": False, "vpad": "bottom"}),
        ("⑤ 不缩放+居中", {"pad": "center", "vresize": False, "vpad": "center"}),
        ("⑥ 裁切+右pad", {"pad": "right", "vresize": True, "crop": True,
                          "order": "pre"}),
        ("⑦ 先定比例再裁", {"pad": "right", "vresize": True, "crop": True,
                            "order": "post"}),
        ("⑧ 裁切+不缩放", {"pad": "center", "vresize": False, "vpad": "center",
                           "crop": True, "order": "pre"}),
    ]
    print(f"  {'变体':<20s} {'比对':>6s} {'误读':>6s} {'均置信':>9s} {'最大宽':>7s}")
    rows = []
    for name, spec in variants:
        p = subprocess.run(
            [sys.executable, "-c", EVAL, str(npz), str(tj), str(a.tol),
             str(fa), str(pad_w), json.dumps(spec)],
            capture_output=True, text=True)
        out = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not out:
            print(f"  {name:<20s} FAIL {(p.stderr or '').strip()[-200:]}")
            continue
        d = json.loads(out[-1])
        rows.append((name, d))
        print(f"  {name:<20s} {d['n']:6d} {d['misread']:6d} "
              f"{d['mean_conf']:9.5f} {d['max_w']:7d}")

    if rows:
        base = rows[0][1]["misread"]
        print()
        for name, d in rows[1:]:
            print(f"    {name} vs ①: 误读 {base}→{d['misread']} "
                  f"({d['misread'] - base:+d})")
    outp = Path(__file__).with_name("_probe_pad_variants.json")
    outp.write_text(json.dumps(
        {n: d for n, d in rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细落盘：{outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
