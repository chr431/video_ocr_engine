r"""干净重测：pad 下限守卫对窄 ROI 准确率的影响。

## 为什么必须重测
上一版 `_probe_autocrop_truth.py` 得出"窄 ROI 裁切 +0.86pp 准确率"，
但那个探针是**在后台跑的同时我改了代码**（加了 pad 下限守卫）→
结论被污染。按守卫逻辑，窄 ROI（test5 106×33，_min_cw=155）应当
100% 被跳过、根本不该有差异 —— 两者矛盾，必须查清。

已用插桩确认：test5 上 679/679 全被守卫跳过、实际裁切 0 次；
新三国01（408×26，_min_cw=122）77 次调用中实际裁 41 次。

## 本探针
对窄 ROI（有真值的 test5/test6）跑三种模式：
  off     不裁（基线）
  guard   现役默认（裁切 + pad 下限守卫）
  noguard 绕过守卫（monkeypatch，只留"内容满宽/动态范围"的保守判断）

用法：python tools/_probe_guard_clean.py [--videos test5,test6] [--reps 2]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# WORKER 以 `python -c` 执行，-c 下 __file__ 未定义，
# 故本进程算好根目录后经 PROBE_ROOT 传给子进程。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROBE_ROOT"] = ROOT  # 供 `python -c` 的 WORKER 子进程使用
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


GT = _VIDEO_DIR / "ground_truth_csv"
VID = _VIDEO_DIR

WORKER = r"""
import os, sys, time, json
sys.path.insert(0, os.environ["PROBE_ROOT"])
mode = sys.argv[1]                      # off | guard | noguard
if mode == 'noguard':
    # 绕过 pad 下限守卫：只保留"动态范围过小 / 内容满宽"的保守跳过
    import numpy as np
    from video_ocr_engine._host_pipeline import _HostPipelineMixin

    def _crop(self, crop):
        g = crop[..., 0] if crop.ndim == 3 else crop
        w = int(g.shape[1])
        if w <= 8 or float(g.std()) < 3.0:
            return crop
        cols = np.nonzero((g > self._bin_thresh).sum(axis=0) >= 2)[0]
        if len(cols) == 0:
            return crop
        m = max(1, int(round(w * self._ocr_autocrop_margin_pct / 100.0)))
        lo = max(0, int(cols[0]) - m)
        hi = min(w, int(cols[-1]) + 1 + m)
        return crop[:, lo:hi] if not (lo == 0 and hi == w) else crop
    _HostPipelineMixin._crop_to_content = _crop

path, roi_s, start, end, stride = sys.argv[2:7]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_start=int(start), frame_end=int(end),
                    sample_stride=int(stride), decode_backend='cpu',
                    ocr_backend='auto', keep_crops=False)
t = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t
got = {}
for s in r.segments:
    for f in (s.frames or (s.start,)):
        got[int(f)] = (s.text or "", round(float(s.confidence), 5))
print(json.dumps({'wall': round(wall, 3), 'segs': len(r.segments),
                  'got': got}))
"""


def meta(p: Path) -> dict:
    o: dict = {}
    for line in open(p, encoding="utf-8-sig"):
        if not line.startswith("#"):
            break
        m = re.search(r"roi\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", line)
        if m and "roi" not in o:
            o["roi"] = ",".join(m.groups())
        for k in ("frame_start", "frame_end"):
            m = re.search(rf"{k}\s*=\s*(-?\d+)", line)
            if m:
                o[k] = int(m.group(1))
    return o


def truth(p: Path) -> dict[int, str]:
    o: dict[int, str] = {}
    for line in open(p, encoding="utf-8-sig"):
        line = line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        ps = line.split(",")
        if len(ps) >= 3 and ps[0].lstrip("-").isdigit():
            o[int(ps[0])] = ps[2].strip()
    return o


def neq(x: str, y: str) -> bool:
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return False
    return abs(fx - fy) <= max(1e-3, 0.02 * max(abs(fx), abs(fy), 1.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="test5,test6")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    print("=== pad 下限守卫 x 窄 ROI 准确率（干净重测）===")
    for vid in (x.strip() for x in a.videos.split(",") if x.strip()):
        tp = GT / f"{vid}_ref.csv"
        if not tp.exists():
            tp = GT / f"{vid}_truth.csv"
        M = meta(tp)
        T = truth(tp)
        print(f"\n--- {vid}  roi={M['roi']}  真值 {len(T)} 帧 ---")
        rows = {}
        # ⚠️ 注意 label 与 mode 必须分开传：worker 里判的是 mode
        # （'off'/'guard'/'noguard'）。首版把中文 label 当 mode 传进去，
        # 结果 `mode == 'noguard'` 恒为假 → monkeypatch 从未生效，
        # 三种模式跑出同一个数（看起来像"守卫没影响"，实为假阴性）。
        for label, mode, env in (
                ("off(不裁)", "off", {"OCR_ROI_AUTOCROP": "0"}),
                ("guard(现役默认)", "guard", {"OCR_ROI_AUTOCROP": "1"}),
                ("noguard(绕过守卫)", "noguard", {"OCR_ROI_AUTOCROP": "1"})):
            e = dict(os.environ)
            e.pop("OCR_ROI_AUTOCROP", None)
            e.update(env)
            best = None
            for _ in range(a.reps):
                p = subprocess.run(
                    [sys.executable, "-c", WORKER, mode,
                     str(VID / f"{vid}.mp4"), M["roi"],
                     str(M.get("frame_start", 0)),
                     str(M.get("frame_end", 0)), str(a.stride)],
                    capture_output=True, text=True, env=e)
                out = (p.stdout or "").strip().splitlines()
                if p.returncode != 0 or not out:
                    print(f"  {label}: FAIL {(p.stderr or '').strip()[-300:]}")
                    break
                d = json.loads(out[-1])
                if best is None or d["wall"] < best["wall"]:
                    best = d
            if best is None:
                continue
            got = {int(k): v[0] for k, v in best["got"].items()}
            both = [f for f in got if f in T]
            ok = sum(1 for f in both if got[f] == T[f])
            okn = sum(1 for f in both
                      if got[f] == T[f] or neq(got[f], T[f]))
            rows[label] = {"wall": best["wall"], "segs": best["segs"],
                          "acc": ok / len(both), "accn": okn / len(both)}
            print(f"  {label:18s} {best['wall']:7.3f}s 段={best['segs']:5d} "
                  f"全等 {ok:6d}/{len(both)} ({ok/len(both):7.3%})  "
                  f"数值容错 ({okn/len(both):7.3%})")
        if "off(不裁)" in rows and "noguard(绕过守卫)" in rows:
            b, g = rows["off(不裁)"], rows["noguard(绕过守卫)"]
            print(f"  -> 不裁 -> 绕过守卫: 墙钟 {(g['wall']/b['wall']-1)*100:+.1f}%  "
                  f"全等 {(g['acc']-b['acc'])*100:+.2f}pp  "
                  f"数值容错 {(g['accn']-b['accn'])*100:+.2f}pp")
        if "guard(现役默认)" in rows and "off(不裁)" in rows:
            same = abs(rows["guard(现役默认)"]["acc"]
                       - rows["off(不裁)"]["acc"]) < 1e-9
            print("  -> 现役守卫下与基线同结果: %s"
                  "（守卫把窄 ROI 全跳过了，符合预期）" % same)
    return 0


if __name__ == "__main__":
    sys.exit(main())
