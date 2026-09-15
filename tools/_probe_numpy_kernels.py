"""numpy 版本对热内核的影响（跨解释器口径：同一份纯 numpy 代码跑两版 numpy）。

动机：`_probe_producer_profile.py` 定位宿主生产者缺口的构成 =
`_segments_similar` 0.329s + `_cluster_win3` 0.211s + `_np_resize` 0.179s
（全是 numpy 热内核，合计 ≈0.72s cumtime）。
numpy 2.5.3 已发布（本机装 2.4.6）→ "换依赖"最直接的候选就是升级 numpy。

本探针把热内核**原样复制**进来（不 import 产品模块，避免解释器间依赖差异），
在指定解释器下计时并打印 JSON；由驱动脚本用两个解释器各跑一次对比。
逐位一致性：两版 numpy 的同一组输入输出应逐位相同（同机同 CPU）。

用法（由 _numpy_ab 驱动，也可单独跑）：
  <python> tools/_probe_numpy_kernels.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]


def _cluster_win3(diff):
    """复制自 segmentation._cluster_win3（uint8 零拷贝版）。"""
    import numpy as np
    if not diff.any():
        return 0.0
    s = diff.view(np.uint8) if diff.flags.c_contiguous else diff.astype(np.uint8)
    c3 = s.copy()
    c3[:, 1:] += s[:, :-1]
    c3[:, :-1] += s[:, 1:]
    w3 = c3.copy()
    w3[1:, :] += c3[:-1, :]
    w3[:-1, :] += c3[1:, :]
    return float(w3.max())


def _resize_map(src_w, src_h, new_w, new_h):
    import numpy as np
    scale_x = src_w / new_w
    scale_y = src_h / new_h
    src_x = np.clip((np.arange(new_w) + 0.5) * scale_x - 0.5, 0, src_w - 1)
    src_y = np.clip((np.arange(new_h) + 0.5) * scale_y - 0.5, 0, src_h - 1)
    x0 = src_x.astype(np.int32)
    y0 = src_y.astype(np.int32)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    wx = (src_x - x0).astype(np.float32)
    wy = (src_y - y0).astype(np.float32)
    return x0, x1, y0, y1, wx, wy


def _np_resize(img, new_w, new_h):
    """复制自 video_utils._np_resize（逐轴 take 版）。"""
    import numpy as np
    src_h, src_w = img.shape[:2]
    if new_w == src_w and new_h == src_h:
        return img.astype(np.float32)
    x0, x1, y0, y1, wx, wy = _resize_map(src_w, src_h, new_w, new_h)
    f = img.astype(np.float32)
    one_ch = f.ndim == 2
    if one_ch:
        f = f[..., None]
    wx3 = wx[None, :, None]
    wy3 = wy[:, None, None]
    g0 = np.take(f, y0, axis=0)
    g1 = np.take(f, y1, axis=0)
    a = np.take(g0, x0, axis=1)
    b = np.take(g0, x1, axis=1)
    c = np.take(g1, x0, axis=1)
    d = np.take(g1, x1, axis=1)
    out = ((1 - wx3) * (1 - wy3) * a +
           wx3 * (1 - wy3) * b +
           (1 - wx3) * wy3 * c +
           wx3 * wy3 * d)
    return out[..., 0] if one_ch else out


def _similar_math(g0, g1, thresh):
    """sim_pair 宿主侧的统计部分：mean abs diff + 显著变化像素数。"""
    import numpy as np
    d = np.abs(g0.astype(np.float32) - g1.astype(np.float32))
    changed = int((d > thresh).sum())
    return float(d.mean()), changed


def bench(fn, reps, warm=5):
    for _ in range(warm):
        fn()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--frames", type=int, default=2950)
    args = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(7)
    h, w = 33, 106
    prev = (rng.random((h, w)) > 0.5)
    cur = prev.copy()
    cur[10:14, 40:50] = ~cur[10:14, 40:50]
    diff_c = prev != cur
    diff_s = prev != prev
    img = (rng.random((h, w)) * 255).astype(np.uint8)
    img3 = (rng.random((h, w, 1)) * 255).astype(np.uint8)
    g0 = rng.random((h, w)).astype(np.float32)
    g1 = g0.copy()
    g1[5:9, 20:30] += 0.4

    res = {
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "us": {},
    }
    px = h * w
    res["us"]["cluster_win3_same"] = bench(lambda: _cluster_win3(diff_s), args.reps)
    res["us"]["cluster_win3_changed"] = bench(lambda: _cluster_win3(diff_c), args.reps)
    res["us"]["resize_106x33_to_154x48"] = bench(lambda: _np_resize(img3, 154, 48), args.reps)
    res["us"]["similar_math"] = bench(lambda: _similar_math(g0, g1, 0.02), args.reps)
    res["us"]["binarize"] = bench(lambda: (img > 128), args.reps)
    res["us"]["astype_f32"] = bench(lambda: img.astype(np.float32), args.reps)
    res["us"]["std"] = bench(lambda: float(img.std()), args.reps)

    # 逐位指纹（跨版本等价性）：把几个内核的输出摘要出来
    res["fingerprint"] = {
        "cluster_same": _cluster_win3(diff_s),
        "cluster_changed": _cluster_win3(diff_c),
        "resize_sum": float(_np_resize(img3, 154, 48).sum()),
        "similar": _similar_math(g0, g1, 0.02),
    }

    if args.json:
        print("PROBE_JSON " + json.dumps(res))
    else:
        print("numpy %s / python %s" % (res["numpy"], res["python"]))
        for k, v in res["us"].items():
            print("  %-28s %9.2f µs" % (k, v))
        print("  fingerprint: %s" % res["fingerprint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
