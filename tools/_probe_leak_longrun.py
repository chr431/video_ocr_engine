"""显存/内存泄漏长跑检测（2026-09-19 审查轮验证工具）。

背景：审查轮修掉两个资源泄漏缺陷——
  · `_YFramePool`/`_DevBatchPool.recycle()` 顺序反了（置空 pool 后入列 →
    复用时 __del__ 撞 None 被吞 = 每轮泄漏一块显存）
  · `OcrSession` 注入路径引擎既不使用也不归还
本探针用「同进程多轮 extract + 每轮采样 VRAM/RSS」验证修复：健康曲线
应为**首轮爬升后平台化**（池按需增长到上限即复用），而非单调增长。

判据（泄漏 vs 平台）：
  · VRAM：后 1/3 轮的斜率（MiB/轮）——健康 < 0.5，泄漏 = 每轮 +池块大小
  · RSS：同口径；Python 侧有分配器缓存，允许更大噪声，看趋势不看绝对值
  · 反向对照：`--expect-leak` 用未修复语义复现（monkeypatch recycle 为旧
    顺序）证明探针能抓到泄漏（防「探针测不出问题」的假绿）

用法：
  python tools/_probe_leak_longrun.py --video test5 --rounds 20
  python tools/_probe_leak_longrun.py --video test5 --rounds 8 --expect-leak
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VIDS = {
    "test5": ("test5.mp4", (843, 993, 948, 1025), 7761),
    "test6_h264": ("test6_h264.mp4", (841, 994, 949, 1026), 23970),
    "test6_hevc": ("test6_hevc.mp4", (841, 994, 949, 1026), 23970),
    "test6_av1": ("test6.mp4", (841, 994, 949, 1026), 23970),
}


def _nvml_used_mib():
    """显存已用（MiB）；NVML 不可用返回 None。

    ctypes 调用口径与 resources.NvmlSampler 一致（返回码 + byref 结构）。"""
    try:
        import ctypes as _ct
        from video_ocr_engine.domain.resources import nvml_handle
        nvml, h = nvml_handle()
        mem = _ct.c_ulonglong(0)   # nvmlMemory_t 首字段 total，次字段 free，
                                   # 第三字段 used；按结构体布局取 used
        class _Mem(_ct.Structure):
            _fields_ = [("total", _ct.c_ulonglong),
                        ("free", _ct.c_ulonglong),
                        ("used", _ct.c_ulonglong)]
        m = _Mem()
        if int(nvml.nvmlDeviceGetMemoryInfo(h, _ct.byref(m))) == 0:
            return m.used / 1048576.0
        return None
    except Exception:
        return None


def _rss_mib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _patch_old_recycle_order():
    """反向对照：把池的 recycle 恢复成旧顺序（置空 pool 后入列）。"""
    from video_ocr_engine.gpu import device as dev

    def _old_yframe_recycle(self, frame):
        frame.pool = None
        self._recycle(frame)

    def _old_devbatch_recycle(self, b):
        b.pool = None
        self._recycle(b)

    dev._YFramePool.recycle = _old_yframe_recycle
    dev._DevBatchPool.recycle = _old_devbatch_recycle
    print("⚠️ 已注入旧 recycle 顺序（反向对照：应看到显存单调增长）")


def _slope_per_round(xs, ys):
    """后 1/3 轮的最小二乘斜率（单位/轮）。"""
    if len(xs) < 4:
        return float("nan")
    k = max(2, len(xs) // 3)
    xs2, ys2 = xs[-k:], ys[-k:]
    mx = sum(xs2) / len(xs2)
    my = sum(ys2) / len(ys2)
    den = sum((x - mx) ** 2 for x in xs2)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs2, ys2)) / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5", choices=sorted(VIDS))
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--decode", default="hybrid",
                    help="解码后端（hybrid 走 GPU 管线 CPU 分支）")
    ap.add_argument("--ocr", default="tensorrt")
    ap.add_argument("--expect-leak", action="store_true",
                    help="注入旧 recycle 顺序做反向对照")
    ap.add_argument("--keep-crops", action="store_true",
                    help="开启 keep_crops（yuv420 代表帧路径）")
    ap.add_argument("--rep-format", default="",
                    help="yuv|gray：yuv 走 _YFramePool（池泄漏的实际触发面）")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    name, roi, full = VIDS[args.video]
    vdir = os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test")
    path = str(Path(vdir) / name)
    n = min(args.frames, full)

    if args.expect_leak:
        _patch_old_recycle_order()

    from video_ocr_engine import FieldExtractor

    gc.collect()
    v0, r0 = _nvml_used_mib(), _rss_mib()
    print("基线: VRAM=%s MiB  RSS=%s MiB  (%s w%d decode=%s ocr=%s)"
          % (None if v0 is None else "%.0f" % v0,
             None if r0 is None else "%.0f" % r0,
             name, n, args.decode, args.ocr))

    vram, rss, walls, segs = [], [], [], []
    for i in range(args.rounds):
        _kw = {}
        if args.rep_format:
            _kw["rep_crop_format"] = args.rep_format
        ex = FieldExtractor(path, roi, frame_start=0, frame_end=n,
                            decode_backend=args.decode, ocr_backend=args.ocr,
                            keep_crops=args.keep_crops, **_kw)
        t0 = time.perf_counter()
        r = ex.extract()
        walls.append(time.perf_counter() - t0)
        segs.append(len(r.segments))
        del ex, r
        gc.collect()
        v, m = _nvml_used_mib(), _rss_mib()
        vram.append(v)
        rss.append(m)
        print("  轮 %2d  wall=%.3fs  VRAM=%s  RSS=%s"
              % (i + 1, walls[-1],
                 "%.0f" % v if v is not None else "-",
                 "%.0f" % m if m is not None else "-"), flush=True)

    xs = list(range(1, args.rounds + 1))
    print("\n== 汇总（%s）==" % (args.label or args.video))
    if vram[0] is not None and all(v is not None for v in vram):
        slope_v = _slope_per_round(xs, vram)
        print("VRAM: 首轮 %.0f → 末轮 %.0f MiB；后段斜率 %+.3f MiB/轮"
              % (vram[0], vram[-1], slope_v))
        verdict_v = "平台（无泄漏）" if slope_v < 0.5 else "**持续增长（疑似泄漏）**"
        print("      判定: %s" % verdict_v)
    if rss[0] is not None and all(m is not None for m in rss):
        slope_r = _slope_per_round(xs, rss)
        print("RSS : 首轮 %.0f → 末轮 %.0f MiB；后段斜率 %+.3f MiB/轮"
              % (rss[0], rss[-1], slope_r))
        verdict_r = ("平台（无泄漏）" if slope_r < 2.0
                     else "**持续增长（疑似泄漏）**")
        print("      判定: %s" % verdict_r)
    print("墙钟: 中位 %.3fs  散布 %.1f%%  段数: %s"
          % (statistics.median(walls),
             (max(walls) - min(walls)) / statistics.median(walls) * 100,
             "恒定 %d" % segs[0] if len(set(segs)) == 1 else "**漂移** %s" % segs))

    out = Path(__file__).resolve().parents[1] / "bench" / "leak_longrun.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "video": name, "frames": n, "rounds": args.rounds,
        "decode": args.decode, "ocr": args.ocr,
        "expect_leak": args.expect_leak,
        "vram_mib": vram, "rss_mib": rss, "walls": walls, "segments": segs,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
