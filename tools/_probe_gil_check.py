"""GIL 判定：生产者侧 numpy 替换为何不兑现墙钟（决定性实验）。

现象（`_probe_patch_verify.py` + `_probe_numpy_replace_ab.py`）：
  · 生产者侧 numpy 实际耗时 = `_segments_similar` 0.255s +
    `_cluster_win3` 0.182s + `_text_sep_binary` 0.088s = **0.53s（占墙钟 24.7%）**；
  · 但把三者全换成 cv2（逐位一致）后**墙钟不动**（+0.37%，符号 -++）。

两个可能：
  ① **cv2 也在 GIL 下**（若 cv2 不释放 GIL，换库只改"谁持有 GIL"，
     不减少生产者线程的**墙钟占用**）；
  ② 生产者侧省下的时间被别处吃掉（如生产者本就在等 GIL/被抢占）。

判定方法：**并发对照**——在后台线程持续做纯 Python 计数（GIL 占用者），
主线程分别跑 numpy 版与 cv2 版热点，看：
  · 单跑耗时（无争用）与并发耗时（有争用）之比 = 该实现的"抗 GIL 争用能力"；
  · 若 cv2 与 numpy 的并发/单跑比相同 → 二者都被 GIL 串行化（可能①）。

同时直接测 `cv2.setNumThreads` 的影响与 cv2 是否释放 GIL
（用 threading 在另一线程计时，看能否并行推进）。

用法：
  python tools/_probe_gil_check.py [--reps 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]


def _mk_map(src_w, src_h, new_w, new_h):
    import numpy as np
    scale_x, scale_y = src_w / new_w, src_h / new_h
    src_x = np.clip((np.arange(new_w) + 0.5) * scale_x - 0.5, 0, src_w - 1)
    src_y = np.clip((np.arange(new_h) + 0.5) * scale_y - 0.5, 0, src_h - 1)
    x0, y0 = src_x.astype(np.int32), src_y.astype(np.int32)
    return (x0, np.minimum(x0 + 1, src_w - 1), y0, np.minimum(y0 + 1, src_h - 1),
            (src_x - x0).astype(np.float32), (src_y - y0).astype(np.float32))


def np_resize(img, new_w, new_h):
    import numpy as np
    src_h, src_w = img.shape[:2]
    x0, x1, y0, y1, wx, wy = _mk_map(src_w, src_h, new_w, new_h)
    f = img.astype(np.float32)
    if f.ndim == 2:
        f = f[..., None]
    wx3, wy3 = wx[None, :, None], wy[:, None, None]
    g0, g1 = np.take(f, y0, axis=0), np.take(f, y1, axis=0)
    a, b = np.take(g0, x0, axis=1), np.take(g0, x1, axis=1)
    c, d = np.take(g1, x0, axis=1), np.take(g1, x1, axis=1)
    return ((1 - wx3) * (1 - wy3) * a + wx3 * (1 - wy3) * b +
            (1 - wx3) * wy3 * c + wx3 * wy3 * d)


def cv_resize(img, new_w, new_h):
    import cv2
    import numpy as np
    f = np.ascontiguousarray(img.astype(np.float32))
    out = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return out[..., None] if img.ndim == 3 and out.ndim == 2 else out


def np_cluster(diff):
    import numpy as np
    s = diff.view(np.uint8) if diff.flags.c_contiguous else diff.astype(np.uint8)
    c3 = s.copy()
    c3[:, 1:] += s[:, :-1]
    c3[:, :-1] += s[:, 1:]
    w3 = c3.copy()
    w3[1:, :] += c3[:-1, :]
    w3[:-1, :] += c3[1:, :]
    return float(w3.max())


def cv_cluster(diff):
    import cv2
    import numpy as np
    s = diff.view(np.uint8) if diff.flags.c_contiguous else diff.astype(np.uint8)
    padded = cv2.copyMakeBorder(s, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    w3 = cv2.boxFilter(padded, ddepth=cv2.CV_16U, ksize=(3, 3),
                       normalize=False, borderType=cv2.BORDER_ISOLATED)
    return float(w3[1:-1, 1:-1].max())


class GILHog(threading.Thread):
    """纯 Python 忙循环：持续持有 GIL（模拟引擎里 OCR/QC 的 Python 工作）。"""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.count = 0

    def run(self):
        while not self.stop:
            for _ in range(10000):
                self.count += 1


def timed(fn, reps, hog: GILHog | None):
    for _ in range(5):
        fn()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    import numpy as np
    import cv2
    rng = np.random.default_rng(3)
    H, W = 33, 106
    g = (rng.random((H, W)) * 255).astype(np.uint8)
    g3 = g[..., None]
    g2 = g.copy()
    g2[10:16, 40:55] = 200
    diff = (g != g2)

    cv2.setNumThreads(1)   # 与生产一致（引擎不设 → 默认可能多线程）
    cases = {
        "np_resize": lambda: np_resize(g3, 154, 48),
        "cv_resize": lambda: cv_resize(g3, 154, 48),
        "np_cluster": lambda: np_cluster(diff),
        "cv_cluster": lambda: cv_cluster(diff),
    }

    res: dict = {"reps": args.reps, "cv2_threads": cv2.getNumThreads(),
                 "cases": {}}
    print("cv2.getNumThreads() = %d\n" % cv2.getNumThreads())
    print("%-12s %11s %11s %9s %s" % ("case", "单跑µs", "并发µs", "放大", "判定"))
    for name, fn in cases.items():
        solo = timed(fn, args.reps, None)
        hog = GILHog()
        hog.start()
        busy = timed(fn, args.reps, hog)
        hog.stop = True
        time.sleep(0.05)
        ratio = busy / solo
        verdict = ("释放 GIL（可并行）" if ratio < 1.6
                   else "被 GIL 串行化")
        print("%-12s %11.2f %11.2f %8.2fx %s"
              % (name, solo, busy, ratio, verdict))
        res["cases"][name] = {"solo_us": round(solo, 2),
                              "contended_us": round(busy, 2),
                              "ratio": round(ratio, 2),
                              "verdict": verdict}

    # 直接测：两个线程分别跑同一实现，总吞吐是否翻倍（真并行的判据）
    print("\n双线程并行吞吐（同实现同时跑 2 份，理想=2.0x）:")
    for name, fn in (("np_resize", cases["np_resize"]),
                     ("cv_resize", cases["cv_resize"])):
        def worker(n, fn=fn):
            for _ in range(n):
                fn()
        n = 400
        t = time.perf_counter()
        worker(n)
        solo = time.perf_counter() - t
        th = [threading.Thread(target=worker, args=(n,)) for _ in range(2)]
        t = time.perf_counter()
        for x in th:
            x.start()
        for x in th:
            x.join()
        both = time.perf_counter() - t
        speedup = (2 * n) / (both / (n / solo))
        print("  %-12s 单线程 %.3fs  双线程 %.3fs  并行度 %.2fx"
              % (name, solo, both, (2 * solo) / both))
        res["cases"][name]["parallel_speedup"] = round((2 * solo) / both, 2)

    out = ROOT / "bench" / "gil_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
