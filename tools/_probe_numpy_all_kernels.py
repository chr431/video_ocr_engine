"""全量 numpy 热点的替换候选 µbench（真实形状 + 逐位一致性同表）。

命题（用户）：把所有用 numpy 的地方都换成更高效的库，看墙钟是否缩减。

上轮只测了 resize 一处（cv2 快 19× 但墙钟 −0.26%）。本探针把**生产链路上
所有 numpy 热点**逐项列出并给出候选实现，用于决定"全换"这一臂该换哪些：

| 热点 | 出处 | profile tottime |
|---|---|---|
| `_np_resize` | `video_utils.py` | 0.064s（cum 0.179s） |
| `_cluster_win3` | `segmentation.py` | 0.121s（cum 0.211s） |
| `_text_sep_binary` | `segmentation.py` | 0.061s |
| `_segments_similar` diff 数学 | `extractor.py` | 0.043s（cum 0.329s） |
| `crop @ _GRAY_W`（灰度） | `segmentation.py` | （in feed/resize） |
| `std`（sharp 选帧） | `host_backend.py` | numpy `_var` 0.062s |
| `binarize` `g > th` | `host_backend.py` | — |
| gamma `255*(g/255)**γ` | `segmentation.py` | numpy `power` |
| `_nv12_luma_full` | `video_utils.py` | — |
| CTC `argmax`/`max` | `native.py` | — |

**判定口径**（项目铁律）：
  · 速度用 min-of-N（µs 级严格性）；
  · **逐位一致性必须同表**——`_np_resize` 的反例留档证明分组顺序变了会漂
    金标；不能为了速度引入未验证的数值差；
  · 形状用**生产真实形状**（`--shapes` 采集：ROI 33×106 一类小图），
    不用教科书 1080p。

用法：
  python tools/_probe_numpy_all_kernels.py [--reps 300] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
GRAY_RGB_WEIGHTS = (0.299, 0.587, 0.114)


def bench(fn, reps, warm=5):
    for _ in range(warm):
        fn()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e6


# ══════════════ 基线实现（逐字复制产品代码语义） ══════════════

def _mk_map(src_w, src_h, new_w, new_h):
    import numpy as np
    scale_x = src_w / new_w
    scale_y = src_h / new_h
    src_x = np.clip((np.arange(new_w) + 0.5) * scale_x - 0.5, 0, src_w - 1)
    src_y = np.clip((np.arange(new_h) + 0.5) * scale_y - 0.5, 0, src_h - 1)
    x0 = src_x.astype(np.int32)
    y0 = src_y.astype(np.int32)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    return x0, x1, y0, y1, (src_x - x0).astype(np.float32), (src_y - y0).astype(np.float32)


def np_resize(img, new_w, new_h):
    import numpy as np
    src_h, src_w = img.shape[:2]
    if new_w == src_w and new_h == src_h:
        return img.astype(np.float32)
    x0, x1, y0, y1, wx, wy = _mk_map(src_w, src_h, new_w, new_h)
    f = img.astype(np.float32)
    one_ch = f.ndim == 2
    if one_ch:
        f = f[..., None]
    wx3, wy3 = wx[None, :, None], wy[:, None, None]
    g0, g1 = np.take(f, y0, axis=0), np.take(f, y1, axis=0)
    a, b = np.take(g0, x0, axis=1), np.take(g0, x1, axis=1)
    c, d = np.take(g1, x0, axis=1), np.take(g1, x1, axis=1)
    out = ((1 - wx3) * (1 - wy3) * a + wx3 * (1 - wy3) * b +
           (1 - wx3) * wy3 * c + wx3 * wy3 * d)
    return out[..., 0] if one_ch else out


def cluster_win3(diff):
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


def text_sep_binary(gray, th):
    import numpy as np
    g = gray.astype(np.float32)
    return np.where(g > th, 255.0, 0.0).astype(np.float32)


def sim_diff_math(a, b):
    """`_segments_similar` 的 diff 数学部分（int16 精确差 + mean + count）。"""
    import numpy as np
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff.mean()), int(np.sum(diff > 10))


def gray_from_rgb(crop):
    """产品语义：`(crop.astype(float32) @ _GRAY_W)`。`_GRAY_W` 是 **float32**
    权重（uint8 转换会把 0.299 截断成 0 → 全黑，别踩）。"""
    import numpy as np
    w = np.asarray(GRAY_RGB_WEIGHTS, dtype=np.float32)
    return crop.astype(np.float32) @ w


def gamma_apply(g, gamma=2.0):
    import numpy as np
    return 255.0 * np.power(g / 255.0, gamma)


def sharp_std(g):
    return g.std(axis=(1, 2))


def binarize(g, th):
    return g > th


# ══════════════ 候选实现 ══════════════

def cv_resize(img, new_w, new_h):
    import cv2
    import numpy as np
    src_h, src_w = img.shape[:2]
    if new_w == src_w and new_h == src_h:
        return img.astype(np.float32)
    f = np.ascontiguousarray(img.astype(np.float32))
    out = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 3 and out.ndim == 2:
        out = out[..., None]
    return out


def _box3_zero_pad(s, ddepth):
    """padding=1 零边界 + boxes 3×3 → 裁回原尺寸（anchor 居中，故 [1:-1,1:-1]
    才是原像素的 3×3 邻域和）。"""
    import cv2
    padded = cv2.copyMakeBorder(s, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    w3 = cv2.boxFilter(padded, ddepth=ddepth, ksize=(3, 3),
                       normalize=False, borderType=cv2.BORDER_ISOLATED)
    return w3[1:-1, 1:-1]


def cv_cluster_win3_boxf(diff):
    """cv2.boxFilter（非归一化，零边界）→ 与逐轴切片加法同值（整数和）。

    ⚠️ cv2 5.0 的 `boxFilter` **不接受 borderValue**（只有 borderType），
    零边界要靠 `copyMakeBorder` 自补，否则默认 BORDER_REFLECT_101 会改边界值。
    """
    import cv2
    import numpy as np
    if not diff.any():
        return 0.0
    s = diff.view(np.uint8) if diff.flags.c_contiguous else diff.astype(np.uint8)
    return float(_box3_zero_pad(s, cv2.CV_16U).max())


def cv_cluster_win3_32s(diff):
    """同上，去掉 cv2 的 16U 饱和风险（CV_32S）。"""
    import cv2
    import numpy as np
    if not diff.any():
        return 0.0
    s = diff.view(np.uint8) if diff.flags.c_contiguous else diff.astype(np.uint8)
    return float(_box3_zero_pad(s, cv2.CV_32S).max())


def cv_text_sep_binary(gray, th):
    """cv2.compare + convertScaleAbs：等价于 where(g>th,255,0)。"""
    import cv2
    import numpy as np
    g8 = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
    mask = cv2.compare(g8, int(th), cv2.CMP_GT)          # 0/255 uint8
    return mask.astype(np.float32)


def np_text_sep_lut(gray, th):
    """numpy LUT 版：查表替代 where（uint8 域内精确）。"""
    import numpy as np
    if not hasattr(np_text_sep_lut, "_lut"):
        lut = np.zeros(256, dtype=np.float32)
        lut[min(max(int(th) + 1, 0), 255):] = 255.0
        np_text_sep_lut._lut = lut
    g8 = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
    return np_text_sep_lut._lut[g8]


def cv_sim_diff_math(a, b):
    """cv2.absdiff(int16) + cv2.mean + countNonZero。"""
    import cv2
    import numpy as np
    d = cv2.absdiff(a.astype(np.int16), b.astype(np.int16))
    mean = float(cv2.mean(d)[0])
    changed = int(cv2.countNonZero(cv2.compare(d, 10, cv2.CMP_GT)))
    return mean, changed


def cv_gray_from_rgb(crop):
    """cv2.cvtColor BGR2GRAY（注意通道序与权重顺序需一致才等价）。"""
    import cv2
    return cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop


def cv_gamma_apply(g, gamma=2.0):
    """cv2.pow 图内 gamma（float32）。cv2.multiply 的 src2 **必须也是
    numpy 数组**（标量会走 Overload resolution failed）。"""
    import cv2
    import numpy as np
    norm = cv2.multiply(g, np.full((), 1.0 / 255.0, dtype=np.float32))
    return cv2.multiply(cv2.pow(norm, float(gamma)),
                        np.full((), 255.0, dtype=np.float32))


def cv_sharp_std(g):
    """cv2.meanStdDev 逐图 std（等价 g.std(axis=(1,2)) 仅当图像为 2D 时）。"""
    import cv2
    import numpy as np
    if g.ndim == 3:
        # 逐图展开：meanStdDev 只吃单图/多通道，不吃 batch
        return np.array([cv2.meanStdDev(f)[1][0, 0] for f in g],
                        dtype=np.float64)
    return float(cv2.meanStdDev(g)[1][0, 0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--roi-w", type=int, default=106)
    ap.add_argument("--roi-h", type=int, default=33)
    args = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(11)
    H, W = args.roi_h, args.roi_w
    res: dict = {"reps": args.reps, "roi": [H, W], "kernels": {}}

    # 生产输入：ROI 灰度 uint8（gray 输出），少量帧有真实变化
    g0 = (rng.random((H, W)) * 255).astype(np.uint8)
    g1 = g0.copy()
    g1[12:18, 45:60] = (rng.random((6, 15)) * 255).astype(np.uint8)
    g3 = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    rgb3 = g3  # 命名清晰：3 通道 ROI 裁切
    grayf = g0.astype(np.float32)
    diff_same = np.zeros((H, W), dtype=bool)
    diff_chg = (g0 != g1)

    def record(name, base_fn, base_args, cands, baseline_label="numpy"):
        """跑一组：基线 + 候选，速度与逐位一致性同表。"""
        entry = {"baseline": baseline_label, "baseline_us": None, "cands": {}}
        try:
            ref = base_fn(*base_args)
        except Exception as e:  # noqa: BLE001
            res["kernels"][name] = {"error": "baseline: %s" % e}
            return
        entry["baseline_us"] = round(bench(lambda: base_fn(*base_args), args.reps), 2)
        for label, fn in cands.items():
            try:
                got = fn(*base_args)
                if isinstance(ref, tuple):
                    same = all(np.array_equal(np.asarray(x), np.asarray(y))
                               for x, y in zip(ref, got))
                    maxd = max((float(np.max(np.abs(np.asarray(x, dtype=np.float64)
                                                    - np.asarray(y, dtype=np.float64))))
                                if np.asarray(x).shape == np.asarray(y).shape
                                else float("nan"))
                               for x, y in zip(ref, got))
                else:
                    ra, ga = np.asarray(ref), np.asarray(got)
                    same = (ra.shape == ga.shape) and bool(np.array_equal(ra, ga))
                    maxd = (float(np.max(np.abs(ra.astype(np.float64)
                                               - ga.astype(np.float64))))
                            if ra.shape == ga.shape else float("nan"))
                us = bench(lambda: fn(*base_args), args.reps)
                entry["cands"][label] = {
                    "us": round(us, 2),
                    "vs_base_pct": round((us - entry["baseline_us"])
                                         / entry["baseline_us"] * 100, 1)
                    if entry["baseline_us"] else None,
                    "bitwise": same, "max_abs_delta": maxd}
            except Exception as e:  # noqa: BLE001
                entry["cands"][label] = {"error": "%s: %s" % (type(e).__name__, e)}
        res["kernels"][name] = entry

    record("resize_33x106->48x154", np_resize, (g0[..., None], 154, 48),
           {"cv2.resize": cv_resize})
    record("resize_33x106->48x72", np_resize, (g0[..., None], 72, 48),
           {"cv2.resize": cv_resize})
    record("cluster_win3__same_frames", cluster_win3, (diff_same,),
           {"cv2.boxFilter16U": cv_cluster_win3_boxf,
            "cv2.boxFilter32S": cv_cluster_win3_32s})
    record("cluster_win3__changed", cluster_win3, (diff_chg,),
           {"cv2.boxFilter16U": cv_cluster_win3_boxf,
            "cv2.boxFilter32S": cv_cluster_win3_32s})
    record("text_sep_binary", text_sep_binary, (g0, 128),
           {"cv2.compare": cv_text_sep_binary, "numpy.LUT": np_text_sep_lut})
    record("sim_diff_math", sim_diff_math, (g0, g1),
           {"cv2.absdiff+mean+count": cv_sim_diff_math})
    record("gray_from_rgb3", gray_from_rgb, (rgb3,),
           {"cv2.cvtColor": cv_gray_from_rgb})
    record("gamma_pow", gamma_apply, (grayf, 2.0),
           {"cv2.pow": cv_gamma_apply})
    record("sharp_std_batch8", sharp_std,
           (np.repeat(g0[None, ...], 8, axis=0),),
           {"cv2.meanStdDev": cv_sharp_std})
    record("binarize", binarize, (g0, 128), {})

    if args.json:
        print("PROBE_JSON " + json.dumps(res))
        return 0

    print("ROI %dx%d  reps=%d  （µs，越小越好；bitwise = 与 numpy 基线逐位一致）\n"
          % (H, W, args.reps))
    print("%-26s %-22s %10s %9s %9s %11s" % (
        "kernel", "candidate", "base µs", "cand µs", "vs base", "bitwise"))
    for name, e in res["kernels"].items():
        if "error" in e:
            print("%-26s %-22s %s" % (name, "-", e["error"]))
            continue
        if not e["cands"]:
            print("%-26s %-22s %10.2f %9s %9s %11s"
                  % (name, "(无候选)", e["baseline_us"], "-", "-", "-"))
            continue
        first = True
        for label, c in e["cands"].items():
            if "error" in c:
                print("%-26s %-22s %s" % (name if first else "", label,
                                          c["error"]))
            else:
                print("%-26s %-22s %10.2f %9.2f %8.1f%% %11s"
                      % (name if first else "", label, e["baseline_us"],
                         c["us"], c["vs_base_pct"],
                         "yes" if c["bitwise"] else "NO(%.2g)" % c["max_abs_delta"]))
            first = False

    out = ROOT / "bench" / "numpy_all_kernels.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
