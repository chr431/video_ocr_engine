r"""专项审计：「最大连通变化块」判据能否区分误合并与安全合并。

用户假设：真实内容变化 → 连通像素块（字体段/整字）；噪声/重渲染 →
分散小差块。用「最大连通块占比」替代/追加「总变化像素占比」。

人群（全部为已通过现行判据 mean≤3 & chg≤1% 的实际合并对）：
- 误合并类：test5/test6 的 19 对（目视已证实为真值跳值吸收）
- 安全类：test 48 对 + test2 16 对 + 新三国01 9 对（oracle 已证无损）

另测两组边缘人群（判据改动会影响的对）：
- B1：mean≤3 但 chg>1%（现被拒）——若「替换」总占比条件，这批会被
  重新考虑：其 max_block 与 oracle 标签决定替换是否引入新损伤
- B2：漏合并对（oracle 同文本但被拒，test/test2）——若 max_block 比
  总占比松，这批能多合并多少（收益侧）

连通块：8 连通，BFS（掩码 ≤1% 面积，纯 python 足够快）。

用法：
  python tools/_probe_block_audit.py [--frames 3000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Repo\video_ocr_engine")
sys.path.insert(0, r"D:\Repo\video_ocr_engine\tools")

import numpy as np

from _probe_det_crop_eval import GT, VID, load_truth, truth_meta
from _probe_merge_audit import (BATCH_DIR, actual_merges, build, load_xsg)


def max_block(mask: np.ndarray) -> tuple[int, list[int]]:
    """8 连通最大块（像素数）与全部块大小降序（掩码为 bool (H,W)）。"""
    seen = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    pts = set(zip(ys.tolist(), xs.tolist()))
    sizes = []
    for start in list(pts):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in pts and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        sizes.append(size)
    sizes.sort(reverse=True)
    return (sizes[0] if sizes else 0), sizes


def pair_diff(exr, crops, reps, i, bin_th):
    from video_utils import _text_sep_gray
    c1, c2 = crops.get(reps[i]), crops.get(reps[i + 1])
    if c1 is None or c2 is None:
        return None
    g1 = c1[..., 0] if c1.ndim == 3 else c1
    g2 = c2[..., 0] if c2.ndim == 3 else c2
    ab = _text_sep_gray(np.asarray(g1, np.float32), "binary", th=bin_th)
    bb = _text_sep_gray(np.asarray(g2, np.float32), "binary", th=bin_th)
    d = np.abs(ab.astype(np.int16) - bb.astype(np.int16))
    mean = float(d.mean())
    mask = d > 10
    chg = int(mask.sum())
    mb, sizes = max_block(mask)
    return dict(mean=mean, chg=chg, mb=mb, mb2=sizes[1] if len(sizes) > 1 else 0,
                nblocks=len(sizes), area=int(g1.size))


def audit_video(video, frames_n, xsg=False, stride=1):
    if xsg:
        meta, truth = load_xsg(video, frames_n)
    else:
        tp = GT / f"{video}_ref.csv"
        if not tp.exists():
            tp = GT / f"{video}_truth.csv"
        meta = truth_meta(tp)
        truth = load_truth(tp)

    exr, fa = build(video, meta, 0 if xsg else frames_n, merge=False,
                    stride=stride, xsg=xsg)
    r_raw = exr.extract()
    bin_th = exr._bin_thresh
    crops = exr.crops
    reps = [s.rep_frame for s in r_raw.segments]
    raw_bounds = [(s.start, s.end) for s in r_raw.segments]
    exm, _ = build(video, meta, 0 if xsg else frames_n, merge=True,
                   stride=stride, xsg=xsg)
    merged_pairs = set(actual_merges(
        raw_bounds, [(s.start, s.end) for s in exm.extract().segments]))

    # oracle 文本
    from ocr_native import OcrEngine
    from video_utils import _preprocess_standard
    oe = OcrEngine("v6_small", "onnxruntime")
    ocr_texts = [""] * len(reps)
    B = 64
    for k0 in range(0, len(reps), B):
        idx = [k for k in range(k0, min(k0 + B, len(reps)))
               if crops.get(reps[k]) is not None]
        if not idx:
            continue
        batch = [_preprocess_standard(crops[reps[k]], force_aspect=fa)
                 for k in idx]
        for k, r in zip(idx, oe(batch)):
            ocr_texts[k] = r.txts[0] or ""

    th_m = 3.0
    out = {"merged": [], "b1": [], "b2": []}
    for i in range(len(reps) - 1):
        m = pair_diff(exr, crops, reps, i, bin_th)
        if m is None:
            continue
        m["i"] = i
        m["ocr_same"] = ocr_texts[i] == ocr_texts[i + 1]
        m["merged"] = (i, i + 1) in merged_pairs
        cap = 0.01 * m["area"]
        if m["merged"]:
            out["merged"].append(m)
        elif m["mean"] <= th_m and m["chg"] > cap:
            out["b1"].append(m)     # 替换条件会重新考虑的边缘人群
        elif (not m["ocr_same"]) and False:
            pass
        if m["ocr_same"] and not (m["mean"] <= th_m and m["chg"] <= cap):
            out["b2"].append(m)     # 漏合并（收益侧）
    return out


def summarize(name, rows):
    if not rows:
        print(f"  {name}: 无")
        return
    mb = sorted(r["mb"] for r in rows)
    chg = sorted(r["chg"] for r in rows)
    print(f"  {name}: n={len(rows)}  max_block px: min {mb[0]} "
          f"p50 {mb[len(mb)//2]} max {mb[-1]} | 总chg: min {chg[0]} "
          f"p50 {chg[len(chg)//2]} max {chg[-1]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3000)
    a = ap.parse_args()

    def merge_dicts(ds):
        out = {"merged": [], "b1": [], "b2": []}
        for d in ds:
            for k in out:
                out[k].extend(d[k])
        return out

    mis = merge_dicts([audit_video("test5", a.frames),
                       audit_video("test6", a.frames)])
    safe = merge_dicts([
        audit_video("test", a.frames),
        audit_video("test2", a.frames),
        audit_video("新三国01", 30000, xsg=True, stride=8)])

    print("=== A 类：实际合并对（已过现行判据）===")
    summarize("误合并(test5/6)", mis["merged"])
    m_mis = [r for r in mis["merged"] if not r["ocr_same"]]
    m_safe = [r for r in safe["merged"] if r["ocr_same"]]
    summarize("  其中 oracle 有损", m_mis)
    summarize("安全合并(test/test2/xsg)", safe["merged"])
    summarize("  其中 oracle 无损", m_safe)
    if m_mis and m_safe:
        lo = max(r["mb"] for r in m_safe)
        hi = min(r["mb"] for r in m_mis)
        print(f"  → 可分性：安全类 max_block ≤ {lo}px，误合并类 ≥ {hi}px"
              + ("  **完全可分**" if lo < hi else "  **重叠，不可分**"))
        # 阈值下的混淆
        for cap in (6, 8, 10, 12, 15, 20):
            tp_ = sum(1 for r in m_mis if r["mb"] > cap)   # 拦截误合并
            fp_ = sum(1 for r in m_safe if r["mb"] > cap)  # 误杀安全合并
            print(f"    cap={cap:>2d}px：拦截误合并 {tp_}/{len(m_mis)}"
                  f"  误杀安全 {fp_}/{len(m_safe)}")

    print("=== B1 类：mean≤3 但 chg>1%（现被拒；『替换』会重新考虑）===")
    b1 = mis["b1"] + safe["b1"]
    dmg = [r for r in b1 if not r["ocr_same"]]
    summarize("全部", b1)
    summarize("  其中 oracle 有损", dmg)
    if b1:
        for cap in (6, 8, 10, 12, 15, 20):
            newm = [r for r in b1 if r["mb"] <= cap]
            nd = sum(1 for r in newm if not r["ocr_same"])
            print(f"    cap={cap:>2d}px：新合并 {len(newm)}（有损 {nd}）")

    print("=== B2 类：漏合并（oracle 同文本；收益侧）===")
    b2 = mis["b2"] + safe["b2"]
    summarize("全部漏合并", b2)
    for cap in (6, 8, 10, 12, 15, 20):
        newm = [r for r in b2 if r["mb"] <= cap]
        print(f"    cap={cap:>2d}px：可解锁 {len(newm)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
