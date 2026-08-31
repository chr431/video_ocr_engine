r"""实证探针：OCR 输入宽度自适应裁切 —— 启发式（墨迹列范围）vs PP-OCRv6 det。

当前引擎裁切 = 分段二值图「有墨迹列范围」+ 余量 10% + 最小收益门槛 10%
（`_crop_to_content` / `_crop_after_aspect` / GPU col_ink 三路同判据）。
本探针评估「换成 PP-OCRv6 det 系列检测模型来定裁切区间」的影响：

Stage A（--stage a）间隔对照 + 耗时：
  1. 抽各视频段代表帧，构造引擎真实 OCR 输入形态（_preprocess_standard：
     48 高 + gamma 2.0 + force_aspect，与生产口径一致）
  2. 逐位复刻引擎启发式 → 裁切区间；PP-OCRv6_tiny_det 概率图列范围 →
     套用同一套余量/门槛数学（_content_range_to_crop）→ 裁切区间
  3. 对照：区间一致性、det 更紧时是否切到墨迹（宽松判据 ≥1 行）、
     det 失效模式（全黑/近满宽）
  4. 耗时：det batch=1 延迟 / batch=16 吞吐，对照同机 ONNX rec batch=16

Stage B（--stage b）真值评分：
  monkeypatch 引擎裁切为 det 区间 → 重跑 extract → 段代表帧 vs 真值
  （生产口径：数值 tol=1），与基线（启发式）对比墙钟/误读/置信度。

det 模型：PP-OCRv6_tiny_det ONNX（1.8MB，ModelScope RapidAI/RapidOCR），
与本探针同目录放置 `tiny_det.onnx`。

用法：
  python tools/_probe_det_crop_eval.py --stage a --videos test5,test,test2,test6
  python tools/_probe_det_crop_eval.py --stage b --videos test5,test [--frames 3000]
"""
from __future__ import annotations

import argparse
import importlib
import re
import statistics
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import onnxruntime as ort
import os
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


GT = _VIDEO_DIR / "ground_truth_csv"
VID = _VIDEO_DIR
DET = Path(__file__).resolve().parent / "tiny_det.onnx"
DET_THRESH = 0.3


def truth_meta(p: Path) -> dict:
    out: dict = {}
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.search(r"roi\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", line)
            if m and "roi" not in out:
                out["roi"] = tuple(int(x) for x in m.groups())
            for key in ("frame_start", "frame_end", "force_aspect"):
                m = re.search(rf"{key}\s*=\s*(-?[\d.]+)", line)
                if m:
                    out[key] = float(m.group(1))
    return out


def load_truth(p: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
                continue
            out[int(parts[0])] = parts[2].strip()
    return out


def num_eq_tol(x: str, y: str, tol: float) -> bool:
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return x == y
    return abs(fx - fy) <= tol


_det_sess = None


def det_session() -> ort.InferenceSession:
    global _det_sess
    if _det_sess is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8   # 线程过多反而放大单图小输入的延迟抖动
        _det_sess = ort.InferenceSession(str(DET), so,
                                         providers=["CPUExecutionProvider"])
    return _det_sess


def det_cols(g: np.ndarray, thresh: float = DET_THRESH):
    """g: 2D float (h,w) 0-255 → ((first,last), prob) 或 (None, prob)。

    预处理与 RapidOCR det 对齐：3ch、(x/255-0.5)/0.5、pad 到 32 倍数
    （黑底 -1）、prob 图取同分辨率「有文字概率」列范围。
    """
    sess = det_session()
    h, w = g.shape
    x = np.stack([g] * 3, axis=-1).astype(np.float32)
    x = (x / 255.0 - 0.5) / 0.5
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)), constant_values=-1.0)
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    prob = out[0, 0][:h, :w]
    cols = np.nonzero((prob > thresh).any(axis=0))[0]
    if len(cols) == 0:
        return None, prob
    return (int(cols[0]), int(cols[-1])), prob


def heuristic_cols(ex, g: np.ndarray, fa: float):
    """逐位复刻引擎判据（_crop_to_content fa=0 / _crop_after_aspect fa>0）。"""
    w = int(g.shape[1])
    if w <= 8 or float(g.std()) < 3.0:
        return None
    if fa and fa > 0:
        from segmentation import _otsu
        th = _otsu(np.clip(g, 0, 255).astype(np.uint8))
    else:
        th = ex._bin_thresh
    cols = np.nonzero((g > th).sum(axis=0) >= 2)[0]
    if len(cols) == 0:
        return None
    return int(cols[0]), int(cols[-1])


def to_interval(ex, rng, w):
    if rng is None:
        return None
    return ex._content_range_to_crop(rng[0], rng[1], w)


def loose_ink_mask(g: np.ndarray) -> np.ndarray:
    """miscut 探针口径：每图 Otsu，墨迹取少侧，列宽松判据 ≥1 行。"""
    from segmentation import _otsu
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    th = _otsu(g8)
    br = float((g > th).sum())
    mask = (g > th) if br <= float((g <= th).sum()) else (g <= th)
    return mask.sum(axis=0) >= 1


def build_extractor(video: str, meta: dict, frames: int, keep_crops: bool):
    from video_ocr_engine import FieldExtractor
    fa = float(meta.get("force_aspect", 0.0))
    s = int(meta.get("frame_start", 0))
    e = int(meta.get("frame_end", 0))
    if frames > 0:
        e = s + frames
    kw = dict(frame_start=s, frame_end=e, sample_stride=1,
              decode_backend="cpu", ocr_backend="cpu",
              keep_crops=keep_crops, force_aspect=fa)
    if keep_crops:
        kw["rep_crop_format"] = "gray"
    return FieldExtractor(str(VID / f"{video}.mp4"), meta["roi"], **kw), fa


def prep_ocr_input(crop, fa: float) -> np.ndarray:
    """引擎真实 OCR 输入形态（48 高 + gamma + force_aspect），3ch float。"""
    from video_utils import _preprocess_standard
    p = _preprocess_standard(crop, force_aspect=fa)
    if p.ndim == 2:
        p = np.stack([p] * 3, axis=-1)
    return p


def det_input(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    x = np.stack([g] * 3, axis=-1).astype(np.float32)
    x = (x / 255.0 - 0.5) / 0.5
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)), constant_values=-1.0)
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


# ───────────────────────────── Stage A ─────────────────────────────

def stage_a(videos: list[str], frames: int) -> int:
    for video in videos:
        tp = GT / f"{video}_ref.csv"
        if not tp.exists():
            tp = GT / f"{video}_truth.csv"
        meta = truth_meta(tp)
        ex, fa = build_extractor(video, meta, frames, keep_crops=True)
        t0 = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t0

        rows = []
        for seg in r.segments:
            c = seg.rep_crop
            if c is None:
                continue
            g = prep_ocr_input(c, fa)[..., 0]
            w = int(g.shape[1])
            if w <= 8 or float(g.std()) < 3.0:
                continue
            hrng = heuristic_cols(ex, g, fa)
            drng, prob = det_cols(g)
            ink = loose_ink_mask(g)
            rows.append(dict(w=w, hrng=hrng, drng=drng,
                             hint=to_interval(ex, hrng, w),
                             dint=to_interval(ex, drng, w),
                             ink=ink, pmax=float(prob.max())))

        n = len(rows)
        same = det_narrow_cut_ink = heur_narrow_cut_ink = 0
        det_blind = det_full = both_crop = h_only = d_only = 0
        cut_det, cut_heur = [], []
        for row in rows:
            h_, d_ = row["hrng"], row["drng"]
            if row["hint"] is not None:
                cut_heur.append((row["w"] - (row["hint"][1] - row["hint"][0])) / row["w"])
            if row["dint"] is not None:
                cut_det.append((row["w"] - (row["dint"][1] - row["dint"][0])) / row["w"])
            if d_ is None:
                det_blind += 1
            elif (d_[1] - d_[0]) / row["w"] > 0.95:
                det_full += 1
            if h_ is not None and d_ is not None:
                if abs(h_[0] - d_[0]) <= 1 and abs(h_[1] - d_[1]) <= 1:
                    same += 1
                if row["hint"] is not None and row["dint"] is not None:
                    both_crop += 1
                    hl, hw_ = row["hint"]
                    dl, dw_ = row["dint"]
                    if dl > hl and row["ink"][hl:dl].any():
                        det_narrow_cut_ink += 1
                    if hl > dl and row["ink"][dl:hl].any():
                        heur_narrow_cut_ink += 1
                    dh = dl + dw_
                    hh = hl + hw_
                    if dh < hh and row["ink"][dh:hh].any():
                        det_narrow_cut_ink += 1
                    if hh < dh and row["ink"][hh:dh].any():
                        heur_narrow_cut_ink += 1
            elif h_ is not None and d_ is None:
                h_only += 1
            elif d_ is not None and h_ is None:
                d_only += 1

        # 耗时：det batch=1 延迟 + batch=16，对照 ONNX rec batch=16（同机同输入）
        samples3 = [prep_ocr_input(s.rep_crop, fa)
                    for s in r.segments[:64] if s.rep_crop is not None]
        samples = [x[..., 0] for x in samples3 if x.shape[1] > 8]
        sess = det_session()
        xs = [det_input(g) for g in samples]
        for x in xs[:10]:
            sess.run(None, {sess.get_inputs()[0].name: x})
        t1 = time.perf_counter()
        for x in xs[:50]:
            sess.run(None, {sess.get_inputs()[0].name: x})
        det1_ms = (time.perf_counter() - t1) / min(50, len(xs)) * 1000
        xb = np.concatenate(xs[:16], axis=0)
        for _ in range(5):
            sess.run(None, {sess.get_inputs()[0].name: xb})
        t2 = time.perf_counter()
        for _ in range(20):
            sess.run(None, {sess.get_inputs()[0].name: xb})
        detb_ms = (time.perf_counter() - t2) / 20 * 1000

        from ocr_native import OcrEngine
        oe = OcrEngine("v6_small", "onnxruntime")
        crops48 = samples3[:16]
        oe(crops48)
        t3 = time.perf_counter()
        for _ in range(5):
            oe(crops48)
        rec16_ms = (time.perf_counter() - t3) / 5 * 1000

        print(f"\n=== {video}  fa={fa}  段 {len(r.segments)}  比对 {n}  "
              f"extract {wall:.2f}s ===")
        if cut_heur:
            cut_heur.sort(); cut_det.sort()
            print(f"  裁掉量中位（裁切段）：启发式 {cut_heur[len(cut_heur)//2]:.1%}  "
                  f"det {cut_det[len(cut_det)//2]:.1%}")
        print(f"  内容区间(±1px)一致：{same}/{n} ({same/n:.1%})   "
              f"最终都裁 {both_crop}  仅启发式裁 {h_only}  仅 det 裁 {d_only}")
        print(f"  det 失效：全黑 {det_blind}  近满宽 {det_full}  "
              f"prob.max 中位 {statistics.median(x['pmax'] for x in rows):.3f}")
        print(f"  det 更紧且切到墨迹：{det_narrow_cut_ink} 段  |  "
              f"启发式更紧且切到墨迹：{heur_narrow_cut_ink} 段")
        print(f"  耗时/段：det b1 {det1_ms:.2f}ms  det b16 {detb_ms/16:.2f}ms  "
              f"rec b16 {rec16_ms/16:.2f}ms  "
              f"→ det = rec 的 {det1_ms/(rec16_ms/16):.1f}×（b1 口径）／"
              f"{detb_ms/rec16_ms:.1f}×（b16 口径）")
    return 0


# ───────────────────────────── Stage B ─────────────────────────────

def _patch_det_crop():
    """把引擎裁切换成 det 区间（同一 _content_range_to_crop 余量/门槛数学）。"""
    import video_ocr_engine._host_pipeline as hp

    def crop_to_content_det(self, crop):
        if not self._ocr_autocrop or getattr(self, "_force_aspect", 0):
            return crop
        g = crop[..., 0] if crop.ndim == 3 else crop
        w = int(g.shape[1])
        if w <= 8 or float(g.std()) < 3.0:
            return crop
        rng, _ = det_cols(g)
        iv = to_interval(self, rng, w)
        if iv is None:
            return crop
        lo, cw = iv
        return crop[:, lo:lo + cw]

    def crop_after_aspect_det(self, img):
        if not self._ocr_autocrop:
            return img
        g = img[..., 0] if img.ndim == 3 else img
        w = int(g.shape[1])
        if w <= 8 or float(g.std()) < 3.0:
            return img
        rng, _ = det_cols(g)
        iv = to_interval(self, rng, w)
        if iv is None:
            return img
        lo, cw = iv
        return img[:, lo:lo + cw]

    hp._HostPipelineMixin._crop_to_content = crop_to_content_det
    hp._HostPipelineMixin._crop_after_aspect = crop_after_aspect_det


def _unpatch():
    import video_ocr_engine._host_pipeline as hp
    importlib.reload(hp)


def run_scored(video: str, meta: dict, frames: int, use_det: bool) -> dict:
    if use_det:
        _patch_det_crop()
    try:
        ex, fa = build_extractor(video, meta, frames, keep_crops=False)
        t0 = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t0
    finally:
        if use_det:
            _unpatch()
    return {"wall": wall, "segs": len(r.segments),
            "reps": {int(s.rep_frame if s.rep_frame is not None else s.start):
                     (s.text or "", round(float(s.confidence), 5))
                     for s in r.segments}}


def stage_b(videos: list[str], frames: int) -> int:
    for video in videos:
        tp = GT / f"{video}_ref.csv"
        if not tp.exists():
            tp = GT / f"{video}_truth.csv"
        meta = truth_meta(tp)
        truth = load_truth(tp)
        base = run_scored(video, meta, frames, use_det=False)
        det = run_scored(video, meta, frames, use_det=True)
        print(f"\n=== {video}  fa={float(meta.get('force_aspect', 0.0))}  "
              f"段 {base['segs']}→{det['segs']}  "
              f"墙钟 {base['wall']:.2f}s→{det['wall']:.2f}s "
              f"({(det['wall']/base['wall']-1)*100:+.1f}%) ===")
        for name, res in (("基线(启发式)", base), ("det 裁切", det)):
            got = {k: v[0] for k, v in res["reps"].items()}
            both = [f for f in got if f in truth]
            ok = sum(1 for f in both if got[f] == truth[f])
            bad = sum(1 for f in both if not num_eq_tol(got[f], truth[f], 1.0))
            conf = (statistics.mean(v[1] for v in res["reps"].values())
                    if res["reps"] else 0.0)
            print(f"  {name:<10s} 比对 {len(both)}  误读(tol1) {bad}  "
                  f"全等 {ok} ({ok/max(1,len(both)):.2%})  数值容错 "
                  f"{(len(both)-bad)/max(1,len(both)):.2%}  均置信 {conf:.4f}")
        bt = {k: v[0] for k, v in base["reps"].items()}
        dt = {k: v[0] for k, v in det["reps"].items()}
        diff = [f for f in bt if f in dt and bt[f] != dt[f]]
        line = f"  文本变化帧：{len(diff)}/{len(bt)}"
        if diff:
            line += "  例：" + "; ".join(f"{f}: {bt[f]!r}→{dt[f]!r}" for f in diff[:6])
        print(line)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("a", "b"), required=True)
    ap.add_argument("--videos", default="test5,test,test2,test6")
    ap.add_argument("--frames", type=int, default=3000)
    a = ap.parse_args()
    vids = [v for v in a.videos.split(",") if v]
    return stage_a(vids, a.frames) if a.stage == "a" else stage_b(vids, a.frames)


if __name__ == "__main__":
    sys.exit(main())
