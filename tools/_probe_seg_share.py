"""临时探针：分段层在端到端墙钟中的真实占比（打桩计时，非推断）。

§17.3 的问题：cluster_win3 下沉的「ROI ≥10 万像素」判据从未被生产场景验证。
ROI 面积 A/B（_probe_roi_segcost.py）有混淆——ROI 变大同时改变 OCR 输入宽度
（infer +134%）与解码侧转换量，无法分离出分段成本。

本探针直接给两个函数打计时桩（包装原函数，行为不变）：
  · video_ocr_engine._host_pipeline._cluster_win3 —— 每帧一次，热点候选
  · video_ocr_engine.extractor._host_segment_frames —— 宿主分段全过程
累计其绝对耗时，与墙钟相比得到真实占比。段数/唯一文本作为正确性校验
（包装不应改变任何行为）。

A/B 交错 + 取 min（§15 方法学，避免冷启动被归因到首个配置）。
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
path, roi_s, n, dbe, obe, st = sys.argv[1:7]
roi = tuple(int(x) for x in roi_s.split(','))

import video_ocr_engine._host_pipeline as hp
import video_ocr_engine.extractor as ex_mod

acc = {'cluster': 0.0, 'cluster_n': 0, 'seg': 0.0, 'seg_n': 0}

_orig_cluster = hp._cluster_win3
def _w_cluster(d):
    t0 = time.perf_counter()
    r = _orig_cluster(d)
    acc['cluster'] += time.perf_counter() - t0
    acc['cluster_n'] += 1
    return r
hp._cluster_win3 = _w_cluster

_orig_seg = ex_mod._host_segment_frames
def _w_seg(*a, **kw):
    t0 = time.perf_counter()
    r = _orig_seg(*a, **kw)
    acc['seg'] += time.perf_counter() - t0
    acc['seg_n'] += 1
    return r
ex_mod._host_segment_frames = _w_seg

from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_end=int(n), sample_stride=int(st),
                    decode_backend=dbe, ocr_backend=obe, keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
texts = sorted({s.text for s in r.segments if s.text})
print(json.dumps({
    'wall': round(wall, 3),
    'seg_s': round(acc['seg'], 3), 'seg_n': acc['seg_n'],
    'cluster_s': round(acc['cluster'], 3), 'cluster_n': acc['cluster_n'],
    'segs': len(r.segments), 'uniq': len(texts),
    'decode': round(ex.timing.get('decode', 0), 3),
    'backend': r.meta.get('backend'), 'ocr_backend': r.meta.get('ocr_backend'),
    'ocr': {k: round(v, 3) for k, v in
            sorted((ex.profile or {}).get('ocr', {}).items(),
                   key=lambda kv: -kv[1])[:4]},
}))
"""


def run(video, roi, n, dbe, obe, st, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run([PY, "-c", WORKER, video, roi, str(n), dbe, obe, str(st)],
                       capture_output=True, text=True, env=e)
    out = (p.stdout or "").strip().splitlines()
    if p.returncode != 0 or not out:
        return {"err": (p.stderr or "").strip()[-400:]}
    return json.loads(out[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test5.mp4")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dbe", default="auto")
    ap.add_argument("--obe", default="auto")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--host-only", action="store_true",
                    help="只测宿主路径（GPU_PIPELINE=0）")
    a = ap.parse_args()

    narrow, wide = "843,993,948,1025", "843,993,1643,1045"
    cases = []
    if not a.host_only:
        cases += [
            ("GPU管线 窄 3.5k px", narrow, {}),
            ("GPU管线 宽 41.6k px", wide, {}),
        ]
    cases += [
        ("宿主路径 窄 3.5k px", narrow, {"GPU_PIPELINE": "0"}),
        ("宿主路径 宽 41.6k px", wide, {"GPU_PIPELINE": "0"}),
    ]
    best = {name: None for name, _, _ in cases}
    for rnd in range(a.rounds + 1):
        for name, roi, env in cases:
            d = run(a.video, roi, a.frames, a.dbe, a.obe, a.stride, env)
            if "err" in d:
                print(f"[轮{rnd}] {name}: 失败 {d['err']}")
                return
            tag = "" if rnd else "  (首轮丢弃)"
            print(f"[轮{rnd}] {name}: wall={d['wall']} 段数={d['segs']} "
                  f"uniq={d['uniq']} 分段={d['seg_s']}s "
                  f"cluster={d['cluster_s']}s decode={d['decode']}{tag}")
            if rnd == 0:
                continue
            if best[name] is None or d["wall"] < best[name]["wall"]:
                best[name] = d

    print(f"\n=== 分段层占墙钟的真实比例"
          f"（{a.rounds} 轮取 min，{a.frames} 帧 stride={a.stride} "
          f"dbe={a.dbe} obe={a.obe}）===")
    print(f"{'':22s} {'墙钟':>7s} {'decode':>7s} {'分段总':>8s} {'占比':>6s} "
          f"{'cluster':>8s} {'占比':>6s} {'调用':>7s}")
    for name, _, _ in cases:
        d = best[name]
        print(f"{name:22s} {d['wall']:7.3f}s {d['decode']:7.3f}s "
              f"{d['seg_s']:8.3f}s {d['seg_s']/d['wall']*100:5.1f}% "
              f"{d['cluster_s']:8.3f}s "
              f"{d['cluster_s']/d['wall']*100:5.1f}% {d['cluster_n']:7d}")
    print("\n注1：GPU 管线下分段在 GPU 上做，_host_segment_frames 不被调用"
          "（调用次数 0 = 该次跑的是 GPU 管线）。")
    print("注2：占比 = 分段绝对耗时 / 墙钟。若分段跑在消费者线程且被解码"
          "掩盖，实际可优化空间小于该值。")


if __name__ == "__main__":
    main()
