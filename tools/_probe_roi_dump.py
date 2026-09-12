"""ROI 截图导出（准确率复核用：把指定帧的原始 ROI 放大落盘 PNG）。

用途：真值可疑时直接看原始 ROI 复核（image-analyst 子智能体消费）。
放大 6×（双线性），底部标注帧号——标注条单独占 24px，不污染 ROI 像素。

用法：
  python tools/_probe_roi_dump.py test5 687,688,689,695
  python tools/_probe_roi_dump.py test6 285,286,287 --out d:/tmp/roi
帧号 = 视频绝对帧号（与真值 CSV 同基准）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import decord
from PIL import Image, ImageDraw

_VDIR = Path(__file__).resolve().parents[2] / ".." / "Videos" / "racelog_test"
_VDIR = (Path(r"D:\Videos\racelog_test") if _VDIR.exists() else _VDIR.resolve())

TRUTH = {
    "test": ("test.mp4", "test_truth.csv"),
    "test2": ("test2.mp4", "test2_truth.csv"),
    "test3": ("test3.mp4", "test3_truth.csv"),
    "test4": ("test4.mp4", "test4_truth.csv"),
    "test5": ("test5.mp4", "test5_ref.csv"),
    "test6": ("test6.mp4", "test6_ref.csv"),
}


def load_header(p: Path):
    txt = p.read_text(encoding="utf-8-sig")
    roi = tuple(int(x) for x in re.search(r"roi=(\d+),(\d+),(\d+),(\d+)", txt).groups())
    return roi


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("frames")          # 逗号分隔，支持 a..b 步进 1
    ap.add_argument("--out", default=str(ROOT / "bench" / "roi_dump"))
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--pad", type=int, default=0,
                    help="左右各外扩 N 像素（看声明 ROI 之外的内容——"
                         "判「真值严于可见像素」时用，见 2026-09-12 准确项 §2.2）")
    ap.add_argument("--pad-y", type=int, default=None,
                    help="上下各外扩（默认同 --pad）")
    ap.add_argument("--montage", action="store_true",
                    help="拼成单张竖排长图（一次 Read 看完整个过渡过程，"
                         "比逐帧读省得多；输出 <video>_montage_<首>_<末>.png）")
    args = ap.parse_args()

    mp4, csv = TRUTH[args.video]
    roi = load_header(_VDIR / "ground_truth_csv" / csv)
    x0, y0, x1, y1 = roi
    py = args.pad if args.pad_y is None else args.pad_y
    if args.pad or py:
        x0, x1 = max(0, x0 - args.pad), x1 + args.pad
        y0, y1 = max(0, y0 - py), y1 + py
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)

    vr = decord.VideoReader(str(_VDIR / mp4), num_threads=2)
    fids = []
    for tok in args.frames.split(","):
        if ".." in tok:
            a, b = tok.split(".."); fids.extend(range(int(a), int(b) + 1))
        else:
            fids.append(int(tok))

    imgs, tiles = [], []
    for f in fids:
        arr = vr[f].asnumpy()[y0:y1, x0:x1]        # (h,w,3) RGB
        im = Image.fromarray(arr).resize((arr.shape[1] * args.scale,
                                          arr.shape[0] * args.scale),
                                         Image.BILINEAR)
        canvas = Image.new("RGB", (im.width, im.height + 24), (0, 0, 0))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((4, im.height + 4),
               f"{args.video} frame {f}  ROI {roi}"
               + (" +pad%d" % args.pad if args.pad else ""),
               fill=(255, 255, 0))
        p = outd / f"{args.video}_f{f}.png"
        canvas.save(p)
        imgs.append(str(p))
        tiles.append(canvas)
    if args.montage and tiles:
        M = Image.new("RGB", (max(t.width for t in tiles),
                              sum(t.height for t in tiles)), (0, 0, 0))
        y = 0
        for t in tiles:
            M.paste(t, (0, y)); y += t.height
        mp = outd / ("%s_montage_%d_%d.png" % (args.video, fids[0], fids[-1]))
        M.save(mp)
        imgs.append(str(mp))
    print("\n".join(imgs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
