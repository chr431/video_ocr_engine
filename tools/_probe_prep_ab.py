"""预处理变体 A/B（**全管线**，非离线）——回答「能否跳出现有 gamma 框架」。

背景与纪律：`_probe_gamma_sweep.py` 的**离线单帧**扫描曾给出与真实管线
相反的结论（C-39），所以本探针一律走**真实引擎全片**：worker 子进程里
monkeypatch `ocr_stage.preprocess_standard`，两侧都强制 `GPU_PIPELINE=0`
（宿主路径共用同一 `preprocess_standard` 调用点，GPU 侧是它的逐位镜像，
故变体实验用宿主路径等价且无需先写 kernel）。

变体（`--variants` 选）：
  base        现役：resize(bilinear)→gray→gamma 2.0
  stretch     逐图百分位拉伸到全量程（**改用自适应曲线、不用 gamma**）
  stretch_g2  逐图拉伸后再 gamma 2.0（自适应 + 现役曲线）
  unsharp     gamma 2.0 + 3×3 unsharp（对抗运动模糊）
  flatten     局部平场（大半径盒均值背景消除）后 gamma 2.0——针对
              已目视证实的机制：test4 有移动过曝白带扫过（§2.2）
  bicubic     只换重采样核（双三次，+0.5 锐度）—— 与色调无关的几何杠杆
  nearest     最近邻（对照：确认重采样核是否敏感）

用法：python tools/_probe_prep_ab.py --variants base,stretch
      python tools/_probe_prep_ab.py --videos test5,test6
落盘 bench/prep_ab.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

_VDIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))
_TRUTH = _VDIR / "ground_truth_csv"
OUT = ROOT / "bench" / "prep_ab.json"

WORKER = r"""
import os, sys, json, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
sys.stdout.reconfigure(encoding="utf-8")
os.environ["GPU_PIPELINE"] = "0"          # 两侧同走宿主路径
import numpy as np
import video_ocr_engine.pipeline.ocr_stage as _os_mod
import video_ocr_engine.domain.segmentation as _seg
from video_ocr_engine.domain.video_utils import _np_resize

path, roi_s, fs, fe, variant = sys.argv[1:6]
roi = tuple(int(x) for x in roi_s.split(','))

_orig = _seg.preprocess_standard


def _blur3(g):
    from video_ocr_engine.domain.segmentation import _cluster_win3   # noqa
    p = np.pad(g, 1, mode='edge')
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def _to_gray(c):
    c = c.astype(np.float32)
    if c.ndim == 2:
        return c
    if c.shape[-1] == 1:
        return c[..., 0]
    return c @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _pack(g):
    return np.stack([g] * 3, axis=-1).astype(np.float32)


def _stretch(g, lo_p=2.0, hi_p=98.0):
    lo, hi = np.percentile(g, (lo_p, hi_p))
    if hi - lo < 20:
        return g
    return np.clip((g - lo) / (hi - lo) * 255.0, 0, 255)


def _box_blur_r(g, r):
    # (2r+1)^2 盒均值，积分图实现（与半径无关的 O(N)）
    p = np.pad(g, r + 1, mode='edge')
    c = p.cumsum(0).cumsum(1)
    h, w = g.shape
    a = c[r:r + h, r:r + w]
    b = c[:h, r:r + w]
    cc = c[r:r + h, :w]
    d = c[:h, :w]
    k = float((2 * r + 1) ** 2)
    return (a - b - cc + d) / k


def preprocess_variant(crop, force_aspect=0.0, gamma=None):
    if variant == 'base':
        return _orig(crop, force_aspect=force_aspect, gamma=2.0)
    target_h = _seg.config.OCR_TARGET_H
    h, w = crop.shape[:2]
    new_w = max(1, int(round(target_h * force_aspect))) if force_aspect > 0 \
        else (max(1, int(w * target_h / h)) if h > 0 else w)
    if variant == 'nearest':
        yi = np.clip((np.arange(target_h) + 0.5) * h / target_h - 0.5,
                     0, h - 1).astype(np.int32)
        xi = np.clip((np.arange(new_w) + 0.5) * w / new_w - 0.5,
                     0, w - 1).astype(np.int32)
        rs = crop.astype(np.float32)[yi[:, None], xi[None, :]]
    else:
        rs = _np_resize(crop, new_w, target_h)
    g = _to_gray(rs)
    if variant == 'stretch':
        return _pack(_stretch(g))
    if variant == 'stretch_g2':
        return _pack(255.0 * np.power(_stretch(g) / 255.0, 2.0))
    g2 = 255.0 * np.power(np.clip(g, 0, 255) / 255.0, 2.0)
    if variant == 'flatten':
        r = max(4, g.shape[0] // 2)          # 半径 ≫ 笔画宽(≈3px)、≈字高
        flat = np.clip(g - _box_blur_r(g, r) + np.percentile(g, 50), 0, 255)
        return _pack(255.0 * np.power(flat / 255.0, 2.0))
    if variant == 'unsharp':
        return _pack(np.clip(g2 + 0.8 * (g2 - _blur3(g2)), 0, 255))
    if variant == 'bicubic':
        # 双三次（Catmull-Rom）核，替代 _np_resize 的双线性
        def _cubic(t):
            t = np.abs(t)
            return np.where(t <= 1, 1.5 * t**3 - 2.5 * t**2 + 1,
                            np.where(t < 2, -0.5 * t**3 + 2.5 * t**2 - 4 * t + 2, 0))
        src = crop.astype(np.float32)
        sh, sw = src.shape[:2]
        yy = (np.arange(target_h) + 0.5) * sh / target_h - 0.5
        xx = (np.arange(new_w) + 0.5) * sw / new_w - 0.5
        y0 = np.floor(yy).astype(np.int32); x0 = np.floor(xx).astype(np.int32)
        out = np.zeros((target_h, new_w) + src.shape[2:], dtype=np.float32)
        for dy in range(-1, 3):
            wy = _cubic(yy - (y0 + dy))[:, None]
            iy = np.clip(y0 + dy, 0, sh - 1)
            for dx in range(-1, 3):
                wx = _cubic(xx - (x0 + dx))[None, :]
                ix = np.clip(x0 + dx, 0, sw - 1)
                out += (wy * wx)[..., None] * src[iy[:, None], ix[None, :]]
        out = np.clip(out, 0, 255)
        return _pack(255.0 * np.power(_to_gray(out) / 255.0, 2.0))
    return _orig(crop, force_aspect=force_aspect, gamma=2.0)


_os_mod.preprocess_standard = preprocess_variant

from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_start=int(fs), frame_end=int(fe),
                    sample_stride=1, decode_backend="auto",
                    ocr_backend="auto", keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
got = {}
for s in r.segments:
    for f in (s.frames or (s.start,)):
        got[int(f)] = (s.text or "")
print("ACCJSON " + json.dumps({"wall": round(wall, 2),
      "segs": len(r.segments), "got": got}, ensure_ascii=False))
"""

PAIRS = [("test.mp4", "test_truth.csv", "truth"),
         ("test2.mp4", "test2_truth.csv", "truth"),
         ("test3.mp4", "test3_truth.csv", "truth"),
         ("test4.mp4", "test4_truth.csv", "truth"),
         ("test5.mp4", "test5_ref.csv", "ref"),
         ("test6.mp4", "test6_ref.csv", "ref"),
         # 同内容异编码对照（test6 的内容，h264 重编码）：用来判定
         # test5/test6 在色调旋钮上的反向到底是 **编码** 还是 **内容**
         ("test6_h264.mp4", "test6_ref.csv", "ref")]

VARIANTS = ["base", "stretch", "stretch_g2", "unsharp", "bicubic", "nearest"]


def load_truth(p: Path):
    roi, fs, fe = None, 0, None
    rows: dict[int, str] = {}
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("#"):
            m = re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", line)
            if m:
                roi = tuple(int(x) for x in m.groups())
            m = re.search(r"frame_start=(\d+)", line)
            if m:
                fs = int(m.group(1))
            m = re.search(r"frame_end=(\d+)", line)
            if m:
                fe = int(m.group(1))
            continue
        pp = line.split(",")
        if len(pp) >= 3 and pp[0].lstrip("-").isdigit():
            rows[int(pp[0])] = pp[2].strip()
    return roi, fs, fe, rows


def strip0(t: str) -> str:
    return t.lstrip("0") or "0"


def run_case(video, roi, fs, fe, variant):
    env = dict(os.environ)
    env["PROBE_ROOT"] = str(ROOT)
    try:
        p = subprocess.run(
            [sys.executable, "-c", WORKER, str(_VDIR / video),
             ",".join(map(str, roi)), str(fs), str(fe), variant],
            env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=900)
    except subprocess.TimeoutExpired:
        return None
    for ln in p.stdout.splitlines():
        if ln.startswith("ACCJSON "):
            return json.loads(ln[8:])
    print("    WORKER_FAIL[%s]: %s" % (
        variant, ((p.stderr or "").strip().splitlines() or ["?"])[-1][:150]))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,stretch,stretch_g2,unsharp")
    ap.add_argument("--videos", default="")
    ap.add_argument("--dump-got", action="store_true",
                    help="落盘逐帧读数（供逐帧 diff 定性：哪些字形对被改动）")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    want = ({t.strip() if t.strip().endswith(".mp4") else t.strip() + ".mp4"
             for t in args.videos.split(",")} if args.videos else None)

    report = {"variants": variants, "cases": {}}
    for video, truthf, kind in PAIRS:
        if want and video not in want:
            continue
        roi, fs, fe, truth = load_truth(_TRUTH / truthf)
        if roi is None or fe is None:
            continue
        per = {}
        for v in variants:
            r = run_case(video, roi, fs, fe, v)
            if r is None:
                per[v] = None
                continue
            got = {int(k): x for k, x in r["got"].items()}
            ex_n = s0_n = 0
            for f, t in truth.items():
                g = got.get(f)
                if g is None:
                    continue
                if g == t:
                    ex_n += 1
                    s0_n += 1
                elif strip0(g) == strip0(t):
                    s0_n += 1
            per[v] = {"exact_n": ex_n, "strip0_n": s0_n,
                      "frames": len(truth), "segs": r["segs"],
                      "wall": r["wall"]}
            if args.dump_got:
                per[v]["got"] = {str(k): x for k, x in got.items()}
        report["cases"][video] = {"kind": kind, "arms": per}
        b = per.get("base")
        print("== %s (%s) n=%d" % (video, kind, b["frames"] if b else -1))
        for v in variants:
            c = per[v]
            if c is None:
                print("   %-11s FAIL" % v)
                continue
            d = "" if (b is None or v == "base") else \
                "  Δexact=%+d  Δ剥零=%+d" % (c["exact_n"] - b["exact_n"],
                                            c["strip0_n"] - b["strip0_n"])
            print("   %-11s exact_n=%5d 剥零_n=%5d segs=%4d %.1fs%s" % (
                v, c["exact_n"], c["strip0_n"], c["segs"], c["wall"], d))
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("落盘 %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
