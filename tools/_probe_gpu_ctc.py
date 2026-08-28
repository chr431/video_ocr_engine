"""临时探针 G：host 输入的 TRT 批也走 GPU argmax 归约（原型验证）。

现役：_call_trt_gpu → _infer_trt_device → DtoH 整批 (B,S,18710) float32
      （B=16 / S≈80 时 ≈95MB/批 → 1083 段 ≈ 2.0s）
原型：_call_trt_gpu → execute_device_argmax → DtoH 仅 (B,S) idx+prob ≈ 12KB
输入已在显存（GpuPreprocessor.process 返回 dev_ptr），切换是一行。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PY = sys.executable

WORKER = r"""
import os, sys, time, json
sys.path.insert(0, r"D:\Repo\video_ocr_engine")
os.environ['ENGINE_PROFILE'] = '1'
mode = sys.argv[1]          # 'base' | 'gpu_ctc'
path, roi_s, n, dbe, obe = sys.argv[2:7]
stride = int(sys.argv[7]) if len(sys.argv) > 7 else 1
roi = tuple(int(x) for x in roi_s.split(','))

if mode == 'gpu_ctc':
    from ocr_native import OcrEngine
    def _call_trt_gpu(self, img_list, max_wh, h0):
        self._get_gpu_pre()
        out_width = int(h0 * max_wh)
        dev_ptr, shape = self._gpu_pre.process(img_list, out_width)
        if getattr(self, '_gpu_ctc_mode', False):
            idx2d, prob2d = self._trt.execute_device_argmax(dev_ptr, shape)
            return self._ctc_from_idxprob(idx2d, prob2d)
        preds = self._infer_trt_device(dev_ptr, shape)
        results = []
        if preds.ndim == 3:
            results = self._ctc_decode_batch(preds)
        else:
            for k in range(len(preds)):
                results.append(self._ctc_decode(preds[k]))
        return results
    OcrEngine._call_trt_gpu = _call_trt_gpu

from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_end=int(n), sample_stride=stride,
                    decode_backend=dbe, ocr_backend=obe, keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
texts = [(s.text, round(s.confidence, 4)) for s in r.segments]
print(json.dumps({
    'wall': round(wall, 3), 'segs': len(r.segments),
    'uniq': len({t for t, _ in texts}),
    'timing': {k: round(v, 3) for k, v in ex.timing.items()},
    'ocr': {k: round(v, 3) for k, v in
            sorted(ex.profile.get('ocr', {}).items(),
                   key=lambda kv: -kv[1])},
    'producer': {k: round(v, 3) for k, v in
                 sorted(ex.profile.get('producer', {}).items(),
                        key=lambda kv: -kv[1])[:5]},
    'backend': r.meta['backend'],
    'sig': texts[:40],
}))
"""


def run(mode, video, roi, n, dbe, obe, env=None, reps=2, stride=1):
    e = dict(os.environ)
    if env:
        e.update(env)
    best = None
    for _ in range(reps):
        p = subprocess.run(
            [PY, "-c", WORKER, mode, video, roi, str(n), dbe, obe, str(stride)],
            capture_output=True, text=True, env=e)
        out = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not out:
            return {"err": (p.stderr or "").strip()[-400:]}
        d = json.loads(out[-1])
        if best is None or d["wall"] < best["wall"]:
            best = d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test5.mp4")
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    roi = ",".join(str(x) for x in (int(x) for x in a.roi.split(",")))
    print(f"=== host 输入 TRT 批 GPU argmax 原型: "
          f"{os.path.basename(a.video)} {a.frames}帧 ===")
    env = {"DECORD_FFMPEG_THREAD_COUNT": os.environ.get("DCDT", "24")}
    res = {}
    for mode, name in (("base", "现役 (DtoH 整批 logits)"),
                       ("gpu_ctc", "原型 (GPU argmax)")):
        d = run(mode, a.video, roi, a.frames, "cpu", "auto", env, a.reps,
                a.stride)
        if "err" in d:
            print(f"  {name}: FAIL {d['err']}")
            continue
        res[mode] = d
        print(f"  {name:24s}: wall={d['wall']:6.3f}s  segs={d['segs']} "
              f"uniq={d['uniq']}  decode={d['timing'].get('decode', 0):.3f} "
              f"ocr_tail={d['timing'].get('ocr_tail', 0):.3f}")
        print(f"      ocr:      {d['ocr']}")
        print(f"      producer: {d['producer']}")
    if "base" in res and "gpu_ctc" in res:
        b, g = res["base"], res["gpu_ctc"]
        print(f"  → 墙钟 {b['wall']:.3f}s → {g['wall']:.3f}s "
              f"({(g['wall']/b['wall']-1)*100:+.1f}%)")
        same = b["sig"] == g["sig"]
        print(f"  → 前 40 段文本+置信度逐位一致: {same}")
        if not same:
            for i, (x, y) in enumerate(zip(b["sig"], g["sig"])):
                if x != y:
                    print(f"      差异[{i}]: {x} != {y}")
                    break


if __name__ == "__main__":
    main()
