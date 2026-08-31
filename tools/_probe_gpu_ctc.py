"""P0-2 验收探针：host 输入的 TRT 批走 GPU argmax 归约（GPU_CTC）。

P0-2 已落地进 `ocr_native._call_trt_gpu`（不再是 monkeypatch 原型），
因此本探针只切 `GPU_CTC` env 做 A/B：
  GPU_CTC=0 → `_infer_trt_device`：DtoH 整批 (B,S,18710) float32
              （B=16 / S≈80 时 ≈95MB/批）
  默认(=1)  → `execute_device_argmax`：DtoH 仅 (B,S) idx+prob（≈12KB）

一致性门：**全段**文本 + 置信度（round 5）逐位一致，不是抽前 40 段。

注意：只有 host 输入路径（decode=cpu + TRT）才走 `_call_trt_gpu`；
NVDEC 直通的 `call_gpu_raw` 本来就走 GPU 归约，不受 GPU_CTC 影响
（用 decode=auto 测会看不到差异）。

用法：
  python tools/_probe_gpu_ctc.py --frames 3000 --stride 1 --reps 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# WORKER 以 `python -c` 执行，-c 下 __file__ 未定义，
# 故本进程算好根目录后经 PROBE_ROOT 传给子进程。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROBE_ROOT"] = ROOT  # 供 `python -c` 的 WORKER 子进程使用
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


PY = sys.executable

WORKER = r"""
import os, sys, time, json, hashlib
sys.path.insert(0, os.environ["PROBE_ROOT"])
os.environ['ENGINE_PROFILE'] = '1'
path, roi_s, n, dbe, obe, stride = sys.argv[1:7]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_end=int(n), sample_stride=int(stride),
                    decode_backend=dbe, ocr_backend=obe, keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
texts = [(s.rep_frame, s.text, round(s.confidence, 5)) for s in r.segments]
sig = hashlib.blake2b(
    json.dumps(texts, ensure_ascii=False).encode(), digest_size=16).hexdigest()
print(json.dumps({
    'wall': round(wall, 3), 'segs': len(r.segments),
    'uniq': len({t for _, t, _ in texts}),
    'sig': sig,
    'timing': {k: round(v, 3) for k, v in ex.timing.items()},
    'ocr': {k: round(v, 3) for k, v in
            sorted(ex.profile.get('ocr', {}).items(),
                   key=lambda kv: -kv[1])},
    'producer': {k: round(v, 3) for k, v in
                 sorted(ex.profile.get('producer', {}).items(),
                        key=lambda kv: -kv[1])[:5]},
    'backend': r.meta['backend'],
    'ocr_backend': r.meta.get('ocr_backend'),
}))
"""


def run(video, roi, n, dbe, obe, env=None, reps=2, stride=1):
    e = dict(os.environ)
    e.pop("GPU_CTC", None)
    if env:
        e.update(env)
    best = None
    for _ in range(reps):
        p = subprocess.run(
            [PY, "-c", WORKER, video, roi, str(n), dbe, obe, str(stride)],
            capture_output=True, text=True, env=e)
        out = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not out:
            return {"err": (p.stderr or "").strip()[-400:]}
        d = json.loads(out[-1])
        if best is None or d["wall"] < best["wall"]:
            best = d
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(_VIDEO_DIR / "test5.mp4"))
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    roi = ",".join(str(x) for x in (int(x) for x in a.roi.split(",")))
    print(f"=== P0-2 验收：{os.path.basename(a.video)} {a.frames}帧 "
          f"stride={a.stride} reps={a.reps} ===")
    print("    配置 decode=cpu ocr=auto（host 输入 + TRT，才走 _call_trt_gpu）")

    res = {}
    for mode, env in (("off", {"GPU_CTC": "0"}), ("on", {})):
        d = run(a.video, roi, a.frames, "cpu", "auto", env, a.reps, a.stride)
        if "err" in d:
            print(f"  GPU_CTC={mode}: FAIL {d['err']}")
            continue
        res[mode] = d
        print(f"  GPU_CTC={mode:3s}: wall={d['wall']:6.3f}s segs={d['segs']} "
              f"uniq={d['uniq']} decode={d['timing'].get('decode', 0):.3f} "
              f"ocr_tail={d['timing'].get('ocr_tail', 0):.3f} "
              f"[{d['backend']}/{d.get('ocr_backend')}]")
        print(f"      ocr:      {d['ocr']}")

    if "off" in res and "on" in res:
        b, g = res["off"], res["on"]
        print(f"\n  → 墙钟 {b['wall']:.3f}s → {g['wall']:.3f}s "
              f"({(g['wall']/b['wall']-1)*100:+.1f}%)")
        for k in ("infer", "ctc", "pre", "batch_wait"):
            if k in b["ocr"] and k in g["ocr"]:
                print(f"  → ocr.{k:10s} {b['ocr'][k]:.3f}s → "
                      f"{g['ocr'][k]:.3f}s "
                      f"({(g['ocr'][k]/max(b['ocr'][k], 1e-6)-1)*100:+.1f}%)")
        same = (b["sig"] == g["sig"] and b["segs"] == g["segs"])
        print(f"  → 全段(帧号,文本,置信度)逐位一致: {'OK' if same else 'FAIL'}")
        if not same:
            print(f"      off sig={b['sig']} segs={b['segs']}")
            print(f"      on  sig={g['sig']} segs={g['segs']}")
        return 0 if same else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
