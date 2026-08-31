"""临时探针：Python 分段成本是否真正转化为墙钟（ROI 面积 A/B）。

§17.3 的判据问题：cluster_win3 下沉的「ROI ≥10 万像素」门槛从未被生产场景
验证过。合成帧能测出「Python 逐帧成本」，但测不出这个成本**是否转化为墙钟**
——宿主路径里分段在消费者线程，与解码生产者并行，可能被完全掩盖。

方法：同一视频、同一 stride，**只改 ROI 面积**（106×33 = 3.5k px vs
800×52 = 41.6k px）。§1.1 已证 ROI 面积对解码吞吐无影响（965 fps 不变），
所以墙钟差 ≈ 分段成本差 × 转化系数。

    A/B 交错 + 取 min（§15 方法学：同进程多配置必须丢首轮/交错取 min，
    避免新配置替进程付 CUDA 上下文 / TRT 反序列化 / NVRTC 编译的 ~1.8s）。
"""
from __future__ import annotations

import argparse
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
import os, sys, time, json
sys.path.insert(0, os.environ["PROBE_ROOT"])
os.environ['ENGINE_PROFILE'] = '1'
path, roi_s, n, dbe, obe, st = sys.argv[1:7]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
ex = FieldExtractor(path, roi, frame_end=int(n), sample_stride=int(st),
                    decode_backend=dbe, ocr_backend=obe, keep_crops=False)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
texts = sorted({s.text for s in r.segments if s.text})
prof = ex.profile or {}
print(json.dumps({
    'wall': round(wall, 3), 'segs': len(r.segments), 'uniq': len(texts),
    'decode': round(ex.timing.get('decode', 0), 3),
    'backend': r.meta.get('backend'), 'ocr_backend': r.meta.get('ocr_backend'),
    'producer': {k: round(v, 3) for k, v in
                 sorted(prof.get('producer', {}).items(),
                        key=lambda kv: -kv[1])[:6]},
    'ocr': {k: round(v, 3) for k, v in
            sorted(prof.get('ocr', {}).items(), key=lambda kv: -kv[1])[:4]},
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
    ap.add_argument("--video", default=str(_VIDEO_DIR / "test5.mp4"))
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dbe", default="auto")
    ap.add_argument("--obe", default="auto")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    # 窄 ROI（生产速度数字）与宽 ROI（text_test 同尺寸），左边界对齐
    cases = [
        ("窄 ROI 106x33 (3.5k px)", "843,993,948,1025"),
        ("宽 ROI 800x52 (41.6k px)", "843,993,1643,1045"),
    ]

    best = {name: None for name, _ in cases}
    for rnd in range(a.rounds + 1):          # +1 = 首轮丢弃
        for name, roi in cases:
            d = run(a.video, roi, a.frames, a.dbe, a.obe, a.stride)
            if "err" in d:
                print(f"[轮{rnd}] {name}: 失败 {d['err']}")
                return
            warm = "" if rnd else " (首轮丢弃)"
            print(f"[轮{rnd}] {name}: wall={d['wall']}s segs={d['segs']} "
                  f"uniq={d['uniq']} decode={d['decode']}s{warm}")
            if rnd == 0:
                continue
            if best[name] is None or d["wall"] < best[name]["wall"]:
                best[name] = d

    print(f"\n=== 结果（{a.rounds} 轮取 min，帧={a.frames} stride={a.stride} "
          f"dbe={a.dbe} obe={a.obe}）===")
    print(f"后端: {best[cases[0][0]]['backend']} / "
          f"{best[cases[0][0]]['ocr_backend']}")
    for name, _ in cases:
        d = best[name]
        print(f"  {name}: wall={d['wall']}s  decode={d['decode']}s  "
              f"段数={d['segs']}  唯一文本={d['uniq']}")
        print(f"      producer 前几项: {d['producer']}")
        print(f"      ocr 前几项: {d['ocr']}")

    wn = best[cases[0][0]]["wall"]
    ww = best[cases[1][0]]["wall"]
    dn = best[cases[0][0]]["decode"]
    dw = best[cases[1][0]]["decode"]
    n = a.frames // a.stride
    print(f"\n采样帧数 = {n}（{a.frames} 源帧 / stride {a.stride}）")
    print(f"墙钟差 = {ww - wn:+.3f}s    解码差 = {dw - dn:+.3f}s")
    print("合成帧测得的逐帧成本差 = 55.32 µs/帧 "
          "(84.92 @41.6k px − 29.60 @3.5k px)")
    print(f"若 100% 转化为墙钟，预期差 = {55.32 * n / 1e6:+.3f}s")
    if ww - wn > 0.02:
        ratio = (ww - wn) / (55.32 * n / 1e6)
        print(f"**转化系数 = {ratio:.2f}**（1.0 = 完全未被解码掩盖）")
    else:
        print("**转化系数 ≈ 0** —— 分段成本被解码完全掩盖")


if __name__ == "__main__":
    main()
