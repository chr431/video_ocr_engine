"""预处理 resize 的依赖替换评估：真实形状采集 + 候选实现 µbench。

命题（用户）：能否通过更换依赖进一步优化性能。

本探针只做两件事，都是"先量后做"的第一层：

1. **真实形状采集**：在生产管线（h264-cpu / host）里插桩 `_np_resize`，
   记录每次调用的 `(src_h, src_w, new_w, new_h)` 与累计调用量——候选实现
   必须按**生产的形状分布**加权评价，不能用教科书 1080p 形状。
2. **候选 µbench**：对采集到的形状跑基线（逐轴 take）与候选择优，
   **同时报告逐位一致性与最大偏差**——浮点分组顺序变了会漂金标
   （见 `_np_resize` docstring 的反例留档），所以速度必须与等价性同表。

候选：
  - `numpy-take`（基线，现役）
  - `cv2.resize`（INTER_LINEAR，OpenCV 像素对齐原生实现）
  - `numpy-separable`（先竖后横两遍；**已知会改分组顺序**，作为反例入表）
  - `scipy.ndimage.zoom`（order=1，坐标语义与 OpenCV 不同，作对照）

用法：
  python tools/_probe_preproc_dep.py --shapes        # 采集形状
  python tools/_probe_preproc_dep.py --bench         # µbench（用采集结果或内置分布）
  python tools/_probe_preproc_dep.py --bench --video <path> --roi a,b,c,d
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SHAPES_CACHE = ROOT / "bench" / "preproc_shapes.json"


# ══════════════════ 1. 形状采集 ══════════════════

def collect_shapes(video: str, roi, frames: int, decode: str,
                   ocr_backend: str, window: int) -> dict:
    """插桩 `_np_resize` 采集真实调用形状（不改产品代码，进程内 monkeypatch）。"""
    from video_ocr_engine.domain import video_utils

    counter: Counter = Counter()
    orig = video_utils._np_resize
    # 产品内部是 `from ... import _np_resize` 的**函数级**引用（见
    # native.py `_resize_norm` 与 segmentation.py），逐模块打补丁才拦得住。
    targets = []
    import video_ocr_engine.domain.segmentation as seg
    import video_ocr_engine.ocr.native as nat
    for mod in (video_utils, seg, nat):
        if hasattr(mod, "_np_resize"):
            targets.append(mod)

    def spy(img, new_w, new_h):
        src = getattr(img, "shape", None)
        if src is not None and len(src) >= 2:
            counter[(int(src[-3]) if img.ndim == 3 else int(src[0]),
                     int(src[-2]) if img.ndim == 3 else int(src[1]),
                     int(new_w), int(new_h), str(img.dtype))] += 1
        return orig(img, new_w, new_h)

    for mod in targets:
        mod._np_resize = spy
    try:
        from video_ocr_engine import FieldExtractor
        ex = FieldExtractor(video, roi, frame_start=0, frame_end=frames,
                            decode_backend=decode, ocr_backend=ocr_backend,
                            keep_crops=False)
        t = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t
    finally:
        for mod in targets:
            mod._np_resize = orig

    rows = [{"shape": list(k[:4]), "dtype": k[4], "n": v}
            for k, v in counter.most_common()]
    return {"video": Path(video).name, "frames": frames, "decode": decode,
            "ocr_backend": ocr_backend, "wall": round(wall, 4),
            "n_segments": len(r.segments), "n_calls": sum(counter.values()),
            "rows": rows}


# ══════════════════ 2. 候选实现 ══════════════════

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
    wx = (src_x - x0).astype(np.float32)
    wy = (src_y - y0).astype(np.float32)
    return x0, x1, y0, y1, wx, wy


def impl_baseline(img, new_w, new_h):
    """现役：逐轴 np.take + 四项加权和（分组顺序 = 产品语义）。"""
    import numpy as np
    src_h, src_w = img.shape[:2]
    if new_w == src_w and new_h == src_h:
        return img.astype(np.float32)
    x0, x1, y0, y1, wx, wy = _mk_map(src_w, src_h, new_w, new_h)
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


def impl_cv2(img, new_w, new_h):
    """cv2.resize INTER_LINEAR（像素中心对齐语义与现役一致）。

    先 astype(float32) 再 resize：cv2 的 **uint8 路径是定点实现**，数值语义
    与现役 float32 插值不同 → 会改金标，故不在候选内（见 use_u8 分支）。
    cv2 会压掉单通道维 (h,w,1)→(h,w)，须显式还原，否则调用方 `[..., 0]`
    与归一化形状全错。
    """
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


def impl_separable(img, new_w, new_h):
    """反例：先竖后横两遍（改分组顺序 → 浮点舍入不同）。"""
    import numpy as np
    src_h, src_w = img.shape[:2]
    if new_w == src_w and new_h == src_h:
        return img.astype(np.float32)
    x0, x1, y0, y1, wx, wy = _mk_map(src_w, src_h, new_w, new_h)
    f = img.astype(np.float32)
    wy3 = wy[:, None] if f.ndim == 2 else wy[:, None, None]
    wx3 = wx[None, :] if f.ndim == 2 else wx[None, :, None]
    g0 = np.take(f, y0, axis=0)
    g1 = np.take(f, y1, axis=0)
    vert = g0 * (1 - wy3) + g1 * wy3
    h0 = np.take(vert, x0, axis=1)
    h1 = np.take(vert, x1, axis=1)
    return h0 * (1 - wx3) + h1 * wx3


IMPLS = {"numpy-take": impl_baseline, "cv2": impl_cv2,
         "numpy-separable": impl_separable}


def bench(shapes, reps: int, out_json: Path | None, use_u8: bool = True) -> dict:
    """µbench。`use_u8` = 用生产实际 dtype（uint8 裁切）作为输入。

    生产链路：decord 裁切输出 uint8 → `preprocess_standard` → `_np_resize`
    （内部 astype(float32)）。**必须按 uint8 入口计时**，否则量的是
    float32 入口（少一次 astype）。
    """
    import numpy as np
    res: dict = {"reps": reps, "impls": {}, "shapes": [], "use_u8": use_u8}
    # 逐位一致性只对**同一份输入**做一次
    for sh in shapes:
        src_h, src_w, new_w, new_h = sh["shape"]
        rng = np.random.default_rng(12345)
        img = (rng.random((src_h, src_w, 1)) * 255).astype(
            np.uint8 if use_u8 else np.float32)
        ref = impl_baseline(img, new_w, new_h)
        rec = {"shape": [src_h, src_w, new_w, new_h], "n": sh["n"]}
        for name, fn in IMPLS.items():
            try:
                got = fn(img, new_w, new_h)
            except Exception as e:  # noqa: BLE001
                rec[name] = {"error": "%s: %s" % (type(e).__name__, e)}
                continue
            same = (ref.shape == got.shape) and bool(np.array_equal(ref, got))
            maxd = (float(np.max(np.abs(ref.astype(np.float64)
                                      - got.astype(np.float64))))
                    if ref.shape == got.shape else float("nan"))
            # 计时：预热后取 min of reps（µs 级严格性）
            for _ in range(3):
                fn(img, new_w, new_h)
            best = 1e9
            for _ in range(reps):
                t = time.perf_counter()
                fn(img, new_w, new_h)
                best = min(best, time.perf_counter() - t)
            rec[name] = {"us": round(best * 1e6, 2), "bitwise": same,
                         "max_abs_delta": maxd}
        res["shapes"].append(rec)

    # 按采集权重加权（生产口径）
    total = sum(s["n"] for s in shapes) or 1
    for name in IMPLS:
        base_us = sum(s["numpy-take"]["us"] * s["n"] for s in res["shapes"]
                      if "us" in s["numpy-take"]) / total
        try:
            us = sum(s[name]["us"] * s["n"] for s in res["shapes"]
                     if "us" in s.get(name, {})) / total
        except Exception:  # noqa: BLE001  # 缺项即不可加权，跳过该候选
            continue
        res["impls"][name] = {
            "weighted_us": round(us, 2),
            "vs_baseline_pct": round((us - base_us) / base_us * 100, 2) if base_us else None,
            "all_bitwise": all(s.get(name, {}).get("bitwise")
                               for s in res["shapes"]),
            "max_abs_delta": max((s.get(name, {}).get("max_abs_delta") or 0.0)
                                 for s in res["shapes"]),
        }
    if out_json:
        out_json.parent.mkdir(exist_ok=True)
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--video", default="")
    ap.add_argument("--roi", default="")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--decode", default="cpu")
    ap.add_argument("--ocr-backend", default="cpu")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--f32", action="store_true",
                    help="用 float32 入口计时（默认按生产的 uint8 入口）")
    args = ap.parse_args()

    if args.shapes:
        vid = args.video or str(Path(
            __import__("os").environ.get("RACELOG_VIDEO_DIR",
                                         r"D:\Videos\racelog_test")) / "test5.mp4")
        roi = tuple(int(x) for x in (args.roi or "843,993,948,1025").split(","))
        out = collect_shapes(vid, roi, args.frames, args.decode,
                             args.ocr_backend, args.window)
        SHAPES_CACHE.parent.mkdir(exist_ok=True)
        SHAPES_CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print("采集：%d 次调用 / %d 段 / wall %.3fs" % (
            out["n_calls"], out["n_segments"], out["wall"]))
        print("%-24s %8s %10s" % ("(h, w, new_w, new_h)", "dtype", "次数"))
        for r in out["rows"][:20]:
            print("%-24s %8s %10d" % (tuple(r["shape"]), r["dtype"], r["n"]))
        print("→ %s" % SHAPES_CACHE)
        return 0

    if args.bench:
        if SHAPES_CACHE.exists():
            shapes = json.loads(SHAPES_CACHE.read_text(encoding="utf-8"))["rows"]
            print("形状来源：%s（生产采集）" % SHAPES_CACHE.name)
        else:
            shapes = [{"shape": [48, 72, 48, 72], "n": 100},
                      {"shape": [32, 105, 48, 224], "n": 100}]
            print("形状来源：内置回退分布")
        res = bench(shapes, args.reps, ROOT / "bench" / "preproc_dep.json",
                    use_u8=not args.f32)
        print("%-18s %12s %10s %9s %12s" % ("impl", "weighted_us", "vs base",
                                            "bitwise", "max|Δ|"))
        for name, v in res["impls"].items():
            print("%-18s %12.2f %9s%% %9s %12.3g" % (
                name, v["weighted_us"], v["vs_baseline_pct"],
                "yes" if v["all_bitwise"] else "NO", v["max_abs_delta"]))
        print("→ bench/preproc_dep.json")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
