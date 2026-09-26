"""hybrid OCR 奖金池裁决探针（R0''，2026-09-20 动态分配立项前置）。

C-58 触发条件（「NVDEC 纯解码（CPU 空闲）+ OCR 需求高」）从未实测过
——原轮用的 h264 恰是死角（CPU 软解本就是快臂，OV 车道无容量可分）。
本探针在 **hevc 转码版**上三臂交错整集跑，量化"理想门控"的奖金池：

  A  decode=nvdec  ocr=tensorrt   ← 基线（现默认语义）
  B  decode=nvdec  ocr=hybrid     ← 双车道 + CPU 空闲（理想门控的静态近似）
  C  decode=hybrid ocr=hybrid     ← 双双 hybrid（C-58 已知被挤压的现状）

裁决口径：**B−A 墙钟差 = 闭环准入控制器的奖金池上限**。
≥10% → 立项闭环门控（不立项学习器）；<5% → 容量边界负结果封板。

用法：
  python tools/_probe_hybridocr_prize.py --rounds 3
  python tools/_probe_hybridocr_prize.py --rounds 3 --video <hevc 路径>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from _worker_lib import (acquire_probe_lock, run_worker, watchdog_prelude,
                         worker_prelude)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_BATCH_DIR = Path(os.environ.get("RACELOG_BATCH_DIR", r"D:\Videos\batch_test"))

#: 臂 → (decode_backend, ocr_backend, 收集文本)；D = 纯解码上限参考
ARMS: dict[str, tuple[str, str, bool]] = {
    "A": ("nvdec", "tensorrt", True),
    "B": ("nvdec", "hybrid", True),
    "C": ("hybrid", "hybrid", False),
    "D": ("decode-only:nvdec", "", False),
}

WORKER = worker_prelude + watchdog_prelude + r"""
import time, json
os.environ['ENGINE_PROFILE'] = '1'
path, roi_s, n, stride, dbe, obe, want_texts = sys.argv[1:8]
roi = tuple(int(x) for x in roi_s.split(','))

if dbe.startswith('decode-only:'):
    # 纯解码上限：同 ROI/帧清单批量拉取（GPU 管线消费口径减去分段/OCR）
    import decord
    ctx = (decord.gpu(0) if 'nvdec' in dbe else decord.cpu(0))
    from decord import VideoReader
    x1, y1, x2, y2 = roi
    vr = VideoReader(path, ctx=ctx, output_format='gray',
                     roi=(x1, y1, x2 + 1, y2 + 1))
    total = min(int(n) or len(vr), len(vr))
    frames = list(range(0, total, int(stride)))
    t0 = time.perf_counter()
    got = 0
    for b in range(0, len(frames), 64):
        nds = vr.get_batch(frames[b:b + 64])
        got += int(nds.shape[0])
        del nds
    wall = time.perf_counter() - t0
    vr.close()
    print(json.dumps({'wall': round(wall, 3), 'segs': got, 'timing': {},
                      'ocr': {}, 'backend': dbe, 'ocr_backend': '-'}))
    raise SystemExit

from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_end=int(n) or None,
                    sample_stride=int(stride),
                    decode_backend=dbe, ocr_backend=obe, keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
out = {
    'wall': round(wall, 3), 'segs': len(r.segments),
    'timing': {k: round(v, 3) for k, v in ex.timing.items()},
    'ocr': {k: round(v, 3) for k, v in ex.profile.get('ocr', {}).items()},
    'backend': r.meta['backend'], 'ocr_backend': r.meta.get('ocr_backend'),
}
if want_texts == '1':
    out['texts'] = [s.text for s in r.segments]
print(json.dumps(out))
"""


def run_arm(video, roi, n, stride, arm):
    dbe, obe, want_texts = ARMS[arm]
    d = run_worker(WORKER, [video, roi, n, stride, dbe, obe,
                            "1" if want_texts else "0"])
    if "err" in d:
        raise RuntimeError("%s FAIL: %s" % (arm, d["err"]))
    return d


def _fmt(d):
    return (f"wall={d['wall']:7.2f}  segs={d['segs']}  "
            f"decode={d['timing'].get('decode', 0):6.2f}  "
            f"infer={d['ocr'].get('infer', 0):6.2f}  "
            f"preproc={d['ocr'].get('preprocess', 0):5.2f}  "
            f"q_wait={d['ocr'].get('q_get_wait', 0):6.2f}  "
            f"[{d['ocr_backend']}/{d['backend']}]")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arms", default="A,B,C",
                    help="逗号分隔臂集（如 A,B,D）")
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--video",
                    default=str(_BATCH_DIR / "新三国01_hevc.mkv"))
    ap.add_argument("--roi", default="144,398,551,423")
    ap.add_argument("--frames", type=int, default=73500)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=r"D:\Repo\_scratch\hybridocr_prize")
    a = ap.parse_args()
    arms = [t.strip() for t in a.arms.split(",") if t.strip()]
    print(f"=== hybrid OCR 奖金池裁决：{Path(a.video).name} "
          f"{a.frames}帧 stride={a.stride} × {a.rounds} 轮交错 "
          f"臂={','.join(arms)} ===")
    data = {arm: [] for arm in arms}
    texts = {}
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with acquire_probe_lock("hybridocr_prize"):
        for r in range(a.rounds):
            order = arms[r % len(arms):] + arms[:r % len(arms)]   # 轮转臂序
            for arm in order:
                d = run_arm(a.video, a.roi, a.frames, a.stride, arm)
                data[arm].append(d)
                if "texts" in d and arm not in texts:
                    texts[arm] = d.pop("texts")
                print(f"  r{r} {arm} {_fmt(d)}")
                time.sleep(a.cooldown)
            if texts:
                (out_dir / f"{Path(a.video).stem}_texts_r{r}.json").write_text(
                    json.dumps({k: v for k, v in texts.items()},
                               ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{Path(a.video).stem}_runs.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n  逐臂均值（wall / decode / infer）：")
    for arm in arms:
        ds = data[arm]
        if not ds:
            continue
        mw = statistics.mean(d["wall"] for d in ds)
        md = statistics.mean(d["timing"].get("decode", 0) for d in ds)
        mi = statistics.mean(d["ocr"].get("infer", 0) for d in ds)
        print(f"    {arm}: wall={mw:6.2f}  decode={md:6.2f}  infer={mi:6.2f}"
              f"  (n={len(ds)})")

    print("\n  配对差分（同轮配对，% vs A）：")
    for arm in arms[1:]:
        pairs = [(b["wall"], x["wall"]) for b, x in zip(data["A"], data[arm])
                 if b and x]
        if len(pairs) < 2:
            print(f"    {arm}: 有效配对不足")
            continue
        # pairs = [(A_wall, arm_wall)] → 差分 = arm/A − 1
        ds_ = [(aw / a_w - 1.0) * 100.0 for a_w, aw in pairs]
        mean = statistics.mean(ds_)
        se = (statistics.stdev(ds_) / len(ds_) ** 0.5
              if len(ds_) > 1 else 0.0)
        worse = sum(1 for d in ds_ if d > 0)
        print(f"    {arm} vs A：均值 {mean:+7.2f}%  SE {se:5.2f}  "
              f"慢 {worse}/{len(ds_)}  逐轮 {['%+.2f' % d for d in ds_]}")
    if "A" in texts and "B" in texts:
        diffs = [(i, x, y) for i, (x, y) in enumerate(
            zip(texts["A"], texts["B"])) if x != y]
        print(f"\n  正确性旁证：B vs A 文本差异 {len(diffs)} / "
              f"{len(texts['A'])} 段")
        for i, x, y in diffs[:8]:
            print(f"      [{i}] {x!r} → {y!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
