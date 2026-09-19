"""TRT 残余泄漏定位探针（2026-09-20 泄漏专项）。

背景：资源长跑轮（2026-09-19）修掉池 GC 回归后仍剩 +0.9~2.0 MiB/轮，
排除链收敛到「TRT 特有、只在完整 extract 组合下出现」。本探针不改产品
代码，双层打点：

  A 层（释放边界 Δ）：monkeypatch 候选释放点（checkin_ocr_engine /
     _gpu_release_partial / Analyzer.release / 池 release_all），前后各
     采样 cudaMemGetInfo（进程口径）——若某边界处显存下降不足，即定位。
  B 层（Python 侧分配追踪）：全局包一层 cuda.bindings.runtime 的
     cudaMalloc/cudaFree/cudaMallocHost/cudaFreeHost，按调用点聚合
     「分配未释放」字节数——Python 可见的泄漏直接给出肇事调用点。
     （TRT 原生内部分配不经过这里，B 层干净 = 泄漏在原生侧。）

用法：
  python tools/_probe_leak_pinpoint.py --rounds 8 [--video test5]
  python tools/_probe_leak_pinpoint.py --rounds 8 --ocr cpu   # OV 对照（应零）
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VIDS = {
    "test5": ("test5.mp4", (843, 993, 948, 1025), 7761),
    "test6_h264": ("test6_h264.mp4", (841, 994, 949, 1026), 23970),
}


def _ctx_free_mib() -> float:
    """本进程 CUDA 上下文口径的空闲显存（cudaMemGetInfo；NVML 会把桌面
    合成算进去，资源长跑轮已证必须用进程口径）。"""
    from cuda.bindings import runtime as cudart
    free = cudart.cudaMemGetInfo()[1]
    return free / 1048576.0


_allocs = {}          # ptr -> (size, site)
_site_stats = defaultdict(lambda: [0, 0])   # site -> [live_bytes, n_live]


def _install_alloc_tracking():
    """B 层：包 cuda.bindings.runtime 的分配/释放，按调用点聚合存活量。"""
    from cuda.bindings import runtime as cudart

    def site_of() -> str:
        # depth 2 = 直接调用行（真正的分配点）；再向上取一个引擎帧作
        # 上下文。此前从 depth 3 起找会跳过分配行、误报到上层调用者
        # （2026-09-20 实测把 trt 侧分配误记到 device.py）。
        def _fmt(fr) -> str:
            return "%s:%d" % (Path(fr.f_code.co_filename).name,
                              fr.f_lineno)
        direct = sys._getframe(2)
        up = ""
        for depth in range(3, 9):
            try:
                fr = sys._getframe(depth)
            except ValueError:
                break
            fn = fr.f_code.co_filename
            if "video_ocr_engine" in fn or fn.startswith("D:\\Repo"):
                up = " <- " + _fmt(fr)
                break
        return _fmt(direct) + up

    orig_malloc = cudart.cudaMalloc
    orig_free = cudart.cudaFree
    orig_mh = cudart.cudaMallocHost
    orig_fh = cudart.cudaFreeHost

    def w_malloc(size):
        r, p = orig_malloc(size)
        if r == 0 and p:
            s = site_of()
            _allocs[p] = (size, s)
            st = _site_stats[s]
            st[0] += size
            st[1] += 1
        return r, p

    def w_free(p):
        if p in _allocs:
            size, s = _allocs.pop(p)
            st = _site_stats[s]
            st[0] -= size
            st[1] -= 1
        return orig_free(p)

    def w_mh(size):
        r, p = orig_mh(size)
        if r == 0 and p:
            s = site_of()
            _allocs[("host", p)] = (size, s)
            st = _site_stats["host:" + s]
            st[0] += size
            st[1] += 1
        return r, p

    def w_fh(p):
        key = ("host", p)
        if key in _allocs:
            size, s = _allocs.pop(key)
            st = _site_stats["host:" + s]
            st[0] -= size
            st[1] -= 1
        return orig_fh(p)

    cudart.cudaMalloc = w_malloc
    cudart.cudaFree = w_free
    cudart.cudaMallocHost = w_mh
    cudart.cudaFreeHost = w_fh


_BOUNDARY_DELTAS = []


def _install_boundary_tracking():
    """A 层：候选释放点前后采样空闲显存（Δ<0 = 该边界释放了；漏判=泄漏
    不在任何已知边界）。"""
    from video_ocr_engine.ocr import native
    from video_ocr_engine.gpu import device as dev

    orig_checkin = native.checkin_ocr_engine
    orig_release_partial = dev._gpu_release_partial

    def probe(name, fn):
        def wrapped(*a, **kw):
            before = _ctx_free_mib()
            r = fn(*a, **kw)
            after = _ctx_free_mib()
            _BOUNDARY_DELTAS.append((name, before - after))
            return r
        return wrapped

    native.checkin_ocr_engine = probe("checkin_ocr_engine", orig_checkin)
    if hasattr(dev, "_gpu_release_partial"):
        dev._gpu_release_partial = probe("_gpu_release_partial",
                                         orig_release_partial)
    # 消费方（gpu_backend:138 / ocr_stage:157,297）全是函数级 from-import，
    # 每次调用时取当前属性 → patch 源模块即生效。


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5", choices=sorted(VIDS))
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--decode", default="hybrid")
    ap.add_argument("--ocr", default="tensorrt")
    args = ap.parse_args()

    name, roi, full = VIDS[args.video]
    vdir = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
    path = str(Path(vdir) / name)
    n = min(args.frames, full)

    _install_alloc_tracking()
    _install_boundary_tracking()

    from video_ocr_engine import FieldExtractor

    print("config: %s w%d decode=%s ocr=%s rounds=%d"
          % (name, n, args.decode, args.ocr, args.rounds))
    free_prev = None
    for i in range(args.rounds):
        ex = FieldExtractor(path, roi, frame_start=0, frame_end=n,
                            decode_backend=args.decode, ocr_backend=args.ocr)
        r = ex.extract()
        nseg = len(r.segments)
        del ex, r
        gc.collect()
        free = _ctx_free_mib()
        delta = free_prev - free if free_prev is not None else float("nan")
        print("轮 %2d segs=%d free=%.0fMiB Δ=%+.2f"
              % (i + 1, nseg, free, delta), flush=True)
        free_prev = free

    print("\n── A 层：释放边界 Δ（正=释放量 MiB；最后一轮）──")
    for name_, d in _BOUNDARY_DELTAS[-6:]:
        print("  %-24s Δ=%+.3f" % (name_, d))

    print("\n── B 层：Python 侧 cudaMalloc/MallocHost 存活（按调用点）──")
    rows = sorted(_site_stats.items(), key=lambda kv: -kv[1][0])[:12]
    for site, (live, cnt) in rows:
        if live > 0 or cnt > 0:
            print("  %-40s live=%8.2f MiB n=%d" % (site, live / 1048576.0, cnt))
    if not rows:
        print("  （无存活项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
