r"""离线审计 v3：merge_similar 的合并空间、误合并 oracle 与候选判据。

标签口径（v1/v2 教训）：
- 真值 tol=1 在连续遥测上把 257→258 标「相同」→ 严格字符串相等。
- **oracle**：OCR(rep_i) == OCR(rep_i+1) ⇔ 合并后输出逐位不变（安全性
  判据）。oracle 最优分组（同文本相邻 run）= OCR 调用下限。
- 已证伪的判据（v2，勿重试）：时间平均二值图的全图均值差 / 平均图逐像素
  变化计数 / 稳定像素翻转 / 差异腐蚀 —— 失效根源：ROI 内滚动背景 + 过曝
  瞬变帧（rep 选择偏爱 std 最大的过曝帧，二值图被亮背景淹没），同文本
  对的像素差异可与异文本对同量级。
- 本版候选：逐 rep Otsu 二值化（_crop_after_aspect 同款思路，抗全局
  亮度瞬变）对比 rep 对。

数据源：
- racelog（默认）：test/test2/test5/test6，真值 CSV 帧级标签。
- --xsg：新三国批量测试集（字幕 CSV 时间轴 → 帧标签）。

用法：
  python tools/_probe_merge_audit.py --videos test,test2 [--frames 3000] [--dump 6]
  python tools/_probe_merge_audit.py --xsg --videos 新三国01 --stride 8 --frames 30000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Repo\video_ocr_engine")
sys.path.insert(0, r"D:\Repo\video_ocr_engine\tools")

import numpy as np

from _probe_det_crop_eval import GT, VID, load_truth, truth_meta

BATCH_DIR = Path(r"D:\Videos\batch_test")


def load_xsg(video: str, frames_end: int):
    """新三国：batch_params.txt 取 ROI；合并字幕.csv 时间轴 → 帧标签。"""
    import decord
    txt = (BATCH_DIR / "batch_params.txt").read_text(encoding="utf-8")
    m = re.search(r"ROI=\[(\d+),(\d+),(\d+),(\d+)\]", txt)
    roi = tuple(int(x) for x in m.groups())
    vr = decord.VideoReader(str(BATCH_DIR / f"{video}.mkv"), num_threads=8)
    fps = float(vr.get_avg_fps()) or 25.0
    del vr

    def hms2f(t: str) -> int:
        hh, mm, ss = t.split(":")
        return int(round((int(hh) * 3600 + int(mm) * 60 + float(ss)) * fps))

    rows = []
    with open(BATCH_DIR / "合并字幕.csv", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\r\n").split(",")
            if len(parts) >= 3 and parts[0] == f"{video}.mkv":
                rows.append((hms2f(parts[1]), parts[2].strip()))
    truth = {}
    fe_all = frames_end if frames_end > 0 else 10 ** 9
    for k, (fs, tx) in enumerate(rows):
        fe = rows[k + 1][0] - 1 if k + 1 < len(rows) else fe_all
        for fr in range(max(0, fs), min(fe, fe_all) + 1):
            truth[fr] = tx
    meta = {"roi": roi, "frame_start": 0, "frame_end": fe_all,
            "force_aspect": 0.0}
    return meta, truth


def build(video: str, meta: dict, frames: int, merge: bool, stride: int = 1,
          xsg: bool = False):
    from video_ocr_engine import FieldExtractor
    fa = float(meta.get("force_aspect", 0.0))
    s = int(meta.get("frame_start", 0))
    e = int(meta.get("frame_end", 0))
    if frames > 0:
        e = s + frames
    vp = ((BATCH_DIR / f"{video}.mkv") if xsg
          else (VID / f"{video}.mp4"))
    return FieldExtractor(str(vp), meta["roi"],
                          frame_start=s, frame_end=e, sample_stride=stride,
                          decode_backend="cpu", ocr_backend="cpu",
                          keep_crops=True, rep_crop_format="gray",
                          force_aspect=fa, merge_similar=merge), fa


def actual_merges(raw_bounds, mrg_bounds):
    """从边界对齐推导在线实际合并（合并段覆盖的连续原始段组）。"""
    pairs = []
    i = j = 0
    cur = []
    while i < len(raw_bounds) and j < len(mrg_bounds):
        rs, re_ = raw_bounds[i]
        ms, me = mrg_bounds[j]
        if rs >= ms and re_ <= me:
            cur.append(i)
            i += 1
        else:
            if cur:
                for a, b in zip(cur, cur[1:]):
                    pairs.append((a, b))
                cur = []
            j += 1
    for a, b in zip(cur, cur[1:]):
        pairs.append((a, b))
    return pairs


def dump_pair(path, g1, g2):
    from PIL import Image
    a = np.asarray(g1, dtype=np.uint8)
    b = np.asarray(g2, dtype=np.uint8)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    h, w = a.shape
    pad = 2
    canvas = np.zeros((h, w * 3 + pad * 2), dtype=np.uint8)
    for k, img in enumerate((a, b, d)):
        canvas[:, k * (w + pad):k * (w + pad) + w] = img
    Image.fromarray(canvas).resize((canvas.shape[1] * 4, canvas.shape[0] * 4),
                                   Image.NEAREST).save(path)


def vis_mismerge(a) -> int:
    """导出实际合并对的边界帧拼图：目视裁定误合并是否存在。"""
    from PIL import Image, ImageDraw
    from ocr_native import OcrEngine
    from video_utils import _preprocess_standard, _text_sep_gray

    for video in [v for v in a.videos.split(",") if v]:
        meta, truth = ((load_xsg(video, a.frames) if a.xsg else None)
                       if a.xsg else
                       (truth_meta(GT / f"{video}_ref.csv")
                        if (GT / f"{video}_ref.csv").exists()
                        else truth_meta(GT / f"{video}_truth.csv"), None))
        if not a.xsg:
            tp = (GT / f"{video}_ref.csv")
            if not tp.exists():
                tp = GT / f"{video}_truth.csv"
            truth = load_truth(tp)
        else:
            meta, truth = load_xsg(video, a.frames)

        exm, fa = build(video, meta, 0 if a.xsg else a.frames,
                        merge=True, stride=a.stride, xsg=a.xsg)
        r_mrg = exm.extract()
        bin_th = exm._bin_thresh
        crops = exm.crops
        exr, _ = build(video, meta, 0 if a.xsg else a.frames,
                       merge=False, stride=a.stride, xsg=a.xsg)
        r_raw = exr.extract()
        raw_bounds = [(s.start, s.end) for s in r_raw.segments]
        mrg_bounds = [(s.start, s.end) for s in r_mrg.segments]
        merged_pairs = actual_merges(raw_bounds, mrg_bounds)
        raw_reps = [s.rep_frame for s in r_raw.segments]

        # 每个原始段的 OCR 文本（rep 裁切取自原始分段跑的 crops）
        oe = OcrEngine("v6_small", "onnxruntime")
        rep_crops = [exr.crops.get(f) for f in raw_reps]
        ocr_texts = [""] * len(raw_reps)
        B = 64
        for i in range(0, len(raw_reps), B):
            idx = [k for k in range(i, min(i + B, len(raw_reps)))
                   if rep_crops[k] is not None]
            if not idx:
                continue
            batch = [_preprocess_standard(rep_crops[k], force_aspect=fa)
                     for k in idx]
            for k, r in zip(idx, oe(batch)):
                ocr_texts[k] = r.txts[0] or ""

        # 现行判据值（全局阈值 rep-vs-rep；crops 用原始分段跑的——
        # 被合并段的 rep 只在 merge=False 时才会存进 crops）
        def pair_metric(i):
            c1, c2 = exr.crops.get(raw_reps[i]), exr.crops.get(raw_reps[i + 1])
            g1 = c1[..., 0] if (c1 is not None and c1.ndim == 3) else c1
            g2 = c2[..., 0] if (c2 is not None and c2.ndim == 3) else c2
            if g1 is None or g2 is None:
                return None, None
            ab = _text_sep_gray(np.asarray(g1, dtype=np.float32),
                                "binary", th=bin_th)
            bb = _text_sep_gray(np.asarray(g2, dtype=np.float32),
                                "binary", th=bin_th)
            d = np.abs(ab.astype(np.int16) - bb.astype(np.int16))
            return float(d.mean()), int((d > 10).sum())

        # 解码器（拼图用原始 ROI 帧）
        import decord
        from segmentation import _gray_seg_batch
        vp = ((BATCH_DIR / f"{video}.mkv") if a.xsg
              else (VID / f"{video}.mp4"))
        x1, y1, x2, y2 = meta["roi"]
        vr = decord.VideoReader(str(vp), num_threads=8)

        def grab(fs):
            need = [f for f in fs if f not in grab.cache]
            for b in range(0, len(need), 64):
                chunk = need[b:b + 64]
                cs = vr.get_batch(chunk).asnumpy()[:, y1:y2 + 1,
                                                   x1:x2 + 1]
                g = _gray_seg_batch(cs)
                for k, f in enumerate(chunk):
                    grab.cache[f] = g[k]
            return [grab.cache[f] for f in fs]
        grab.cache = {}

        out = Path(a.dump_dir)
        out.mkdir(exist_ok=True)
        print(f"\n=== {video}  实际合并 {len(merged_pairs)} 对 ===")
        print(f"  {'对':>8s} {'bin_mean':>8s} {'bin_chg':>7s} "
              f"{'OCR A→B':>16s} {'真值A→B':>16s}")
        for (i, j) in merged_pairs:
            bm, bc = pair_metric(i)
            sa, sb = raw_bounds[i], raw_bounds[j]
            # 边界帧窗口：A 段尾 3 帧 + B 段头 3 帧
            fs = (list(range(max(sa[0], sa[1] - 2), sa[1] + 1))
                  + list(range(sb[0], min(sb[1], sb[0] + 2) + 1)))
            fs = sorted(set(fs))
            imgs = grab(fs)
            tA = [truth.get(f, "?") for f in fs]
            scale = 5
            h, w = imgs[0].shape
            cols = 4
            rows = (len(imgs) + cols - 1) // cols
            pad = 14
            canvas = Image.new("L", (cols * (w * scale + 4) + 4,
                                     rows * (h * scale + pad + 4) + 4), 0)
            dr = ImageDraw.Draw(canvas)
            for k, (f, img) in enumerate(zip(fs, imgs)):
                r_, c_ = divmod(k, cols)
                x0 = 4 + c_ * (w * scale + 4)
                y0 = 4 + r_ * (h * scale + pad + 4)
                canvas.paste(Image.fromarray(img).resize(
                    (w * scale, h * scale), Image.NEAREST), (x0, y0 + pad))
                dr.text((x0, y0), f"f{f} truth={truth.get(f, '?')}",
                        fill=255)
            fn = out / f"{video}_m{i}_f{fs[0]}.png"
            canvas.save(fn)
            print(f"  {i:4d}->{j:<4d} {bm:8.2f} {bc:7d} "
                  f"{ocr_texts[i]!r:>6s}→{ocr_texts[j]!r:<6s} "
                  f"{str(truth.get(raw_reps[i]))!s:>6s}→"
                  f"{str(truth.get(raw_reps[j]))!s:<6s} {fn.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="test,test2,test5,test6")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--dump", type=int, default=0)
    ap.add_argument("--dump-dir",
                    default=str(Path(__file__).parent / "_merge_vis"))
    ap.add_argument("--xsg", action="store_true",
                    help="新三国批量测试集（字幕 CSV 时间轴标签）")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--vis-mismerge", action="store_true",
                    help="导出实际合并对的边界帧拼图（目视裁定误合并）")
    a = ap.parse_args()
    if a.vis_mismerge:
        return vis_mismerge(a)

    for video in [v for v in a.videos.split(",") if v]:
        if a.xsg:
            meta, truth = load_xsg(video, a.frames)
        else:
            tp = GT / f"{video}_ref.csv"
            if not tp.exists():
                tp = GT / f"{video}_truth.csv"
            meta = truth_meta(tp)
            truth = load_truth(tp)

        exr, fa = build(video, meta, 0 if a.xsg else a.frames,
                        merge=False, stride=a.stride, xsg=a.xsg)
        r_raw = exr.extract()
        bin_th = exr._bin_thresh
        crops = exr.crops
        raw_reps = [s.rep_frame for s in r_raw.segments]
        raw_bounds = [(s.start, s.end) for s in r_raw.segments]
        n = len(raw_reps)

        exm, _ = build(video, meta, 0 if a.xsg else a.frames,
                       merge=True, stride=a.stride, xsg=a.xsg)
        r_mrg = exm.extract()
        mrg_bounds = [(s.start, s.end) for s in r_mrg.segments]
        merged_pairs = actual_merges(raw_bounds, mrg_bounds)

        # OCR 文本（oracle）
        from ocr_native import OcrEngine
        from video_utils import _preprocess_standard
        oe = OcrEngine("v6_small", "onnxruntime")
        rep_crops = [crops.get(f) for f in raw_reps]
        ocr_texts = [""] * n
        B = 64
        for i in range(0, n, B):
            idx = [k for k in range(i, min(i + B, n))
                   if rep_crops[k] is not None]
            if not idx:
                continue
            batch = [_preprocess_standard(rep_crops[k], force_aspect=fa)
                     for k in idx]
            for k, r in zip(idx, oe(batch)):
                ocr_texts[k] = r.txts[0] or ""

        runs = 1 + sum(1 for i in range(n - 1)
                       if ocr_texts[i] != ocr_texts[i + 1])
        print(f"\n=== {video}  fa={fa}  stride={a.stride}  原始段 {n} → "
              f"现役合并后 {len(mrg_bounds)}（实际合并 {len(merged_pairs)} 对）"
              f" ===")
        print(f"  oracle OCR 调用下限 {runs}（当前 {len(mrg_bounds)}，"
              f"理论再省 {len(mrg_bounds) - runs}）")

        from segmentation import _otsu
        from video_utils import _text_sep_gray

        def rep_pair(i):
            c1, c2 = crops.get(raw_reps[i]), crops.get(raw_reps[i + 1])
            if c1 is None or c2 is None:
                return None, None
            g1 = c1[..., 0] if c1.ndim == 3 else c1
            g2 = c2[..., 0] if c2.ndim == 3 else c2
            return (np.asarray(g1, dtype=np.float32),
                    np.asarray(g2, dtype=np.float32))

        rows = []
        for i in range(n - 1):
            g1, g2 = rep_pair(i)
            if g1 is None:
                continue
            # 现行：全局校准阈值 rep-vs-rep
            ab = _text_sep_gray(g1, "binary", th=bin_th)
            bb = _text_sep_gray(g2, "binary", th=bin_th)
            d = np.abs(ab.astype(np.int16) - bb.astype(np.int16))
            # 候选：逐 rep Otsu
            o1 = _otsu(np.clip(g1, 0, 255).astype(np.uint8))
            o2_ = _otsu(np.clip(g2, 0, 255).astype(np.uint8))
            ab2 = (g1 > o1).astype(np.float32) * 255.0
            bb2 = (g2 > o2_).astype(np.float32) * 255.0
            d2 = np.abs(ab2.astype(np.int16) - bb2.astype(np.int16))
            t1, t2 = truth.get(raw_reps[i]), truth.get(raw_reps[i + 1])
            rows.append(dict(
                i=i, area=int(g1.size),
                bin_mean=float(d.mean()),
                bin_changed=int((d > 10).sum()),
                obin_mean=float(d2.mean()),
                obin_changed=int((d2 > 10).sum()),
                ocr_same=ocr_texts[i] == ocr_texts[i + 1],
                truth_same=(t1 is not None and t1 == t2),
                actually_merged=(i, i + 1) in set(merged_pairs)))

        merged_set = set(merged_pairs)
        th_m, th_c = 3.0, exr._merge_max_changed_pixels
        proxy = [r for r in rows
                 if r["bin_mean"] <= th_m and r["bin_changed"] <= th_c]
        pm = [r for r in proxy if not r["ocr_same"]]
        miss = [r for r in rows if r["ocr_same"]
                and not (r["bin_mean"] <= th_m and r["bin_changed"] <= th_c)]
        dm_act = [r for r in rows if r["actually_merged"]
                  and not r["ocr_same"]]
        print(f"  [现行] 重放命中 {len(proxy)}（有损 {len(pm)}）"
              f"  漏合并 {len(miss)}  实际合并 {len(merged_pairs)}"
              f"（oracle 有损 {len(dm_act)}）")
        # 逐 rep Otsu 工作点扫描（含现行双条件变体）
        for tm in (3.0, 5.0, 6.0, 8.0):
            m = [r for r in rows if r["obin_mean"] <= tm]
            dmg = [r for r in m if not r["ocr_same"]]
            dt = [r for r in dmg if r["truth_same"] is False]
            print(f"  [逐repOtsu mean≤{tm}] 合并 {len(m)}  有损 "
                  f"{len(dmg)}（其中真值也异 {len(dt)}）  无损 {len(m) - len(dmg)}")
        for tm in (3.0, 5.0, 6.0):
            m = [r for r in rows if r["obin_mean"] <= tm
                 and r["obin_changed"] <= th_c]
            dmg = [r for r in m if not r["ocr_same"]]
            dt = [r for r in dmg if r["truth_same"] is False]
            print(f"  [逐repOtsu mean≤{tm}&chg≤1%] 合并 {len(m)}  有损 "
                  f"{len(dmg)}（真值也异 {len(dt)}）  无损 {len(m) - len(dmg)}")
        # 列出损伤对明细（真值裁定用）
        for tm in (6.0,):
            m = [r for r in rows if r["obin_mean"] <= tm
                 and not r["ocr_same"]]
            for r in m[:8]:
                i = r["i"]
                t1, t2 = truth.get(raw_reps[i]), truth.get(raw_reps[i + 1])
                print(f"    损伤对 seg{i}(f{raw_reps[i]})→seg{i+1}(f{raw_reps[i+1]}): "
                      f"文本 {ocr_texts[i]!r}→{ocr_texts[i+1]!r}  真值 "
                      f"{t1!r}→{t2!r}  obin_mean {r['obin_mean']:.2f}")
        ob = sorted(r["obin_mean"] for r in rows if r["ocr_same"])
        ox = sorted(r["obin_mean"] for r in rows if not r["ocr_same"])
        if ob and ox:
            print(f"  逐repOtsu mean：oracle同 p50 {ob[len(ob)//2]:.2f} "
                  f"p90 {ob[len(ob)*9//10]:.2f} | oracle异 p10 "
                  f"{ox[len(ox)//10]:.2f} p50 {ox[len(ox)//2]:.2f}")

        if a.dump:
            vis = Path(a.dump_dir)
            vis.mkdir(exist_ok=True)
            cands = miss[:a.dump]
            for r in cands:
                i = r["i"]
                g1, g2 = rep_pair(i)
                if g1 is None:
                    continue
                dump_pair(str(vis / f"{video}_miss_p{i}_{raw_reps[i]}.png"),
                          g1, g2)
            print(f"  目视（漏合并对）{len(cands)} → {vis}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
