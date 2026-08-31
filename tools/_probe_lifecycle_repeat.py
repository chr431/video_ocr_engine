"""生命周期稳定性探针：同进程连续重复提取，检测退化 / 线程残留 / 结果漂移。

针对 PERFORMACE_REPORT.txt「优先验证的生命周期问题」中可直接观测的部分：
  1. 长进程连续提取是否越跑越慢（墙钟漂移：max/median、末轮/首轮）；
  2. extract() 返回后是否残留生产者 / OCR worker 线程（下一任务被上一任务
     干扰的典型来源）；
  3. 各轮段数 / 唯一文本集是否一致（资源复用污染的行为信号）；
  4. 显存是否随轮次单调增长（cudaMemGetInfo 观测，非精确归因）。

每轮新建 FieldExtractor（模拟批量任务逐视频调用），keep_crops=False 以
减少代表帧缓存带来的内存噪音。

用法：
  python tools/_probe_lifecycle_repeat.py --video X --roi a,b,c,d \\
      [--rounds 5] [--backend auto] [--frame-start 362] [--frames 3000]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time

# tools/ 下直接运行时 sys.path[0] 是 tools/，仓库根不在路径上 → 插到最前，
# 保证 import 到本仓源码（而非其它环境里可能存在的同名已安装包）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:                                  # 中文输出在 GBK 控制台会 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _vram_mb() -> float | None:
    """当前进程可见的已用显存（MB）；无 CUDA 时返回 None。"""
    try:
        from cuda.bindings import runtime as cudart
        err, free, total = cudart.cudaMemGetInfo()
        if err != 0:
            return None
        return (total - free) / (1024.0 * 1024.0)
    except Exception:
        return None


class _Cancelled(Exception):
    """探针内部取消信号（模拟用户中途取消提取）。"""


def _run_once(video: str, roi: tuple, backend: str, ocr_backend: str,
              frame_start: int, frames: int, cancel_after: int = 0) -> dict:
    from video_ocr_engine import FieldExtractor
    kw = {}
    if cancel_after:
        state = {"n": 0}

        def _cancel() -> None:
            state["n"] += 1
            if state["n"] >= cancel_after:
                raise _Cancelled("probe cancel")
        kw["cancel_check"] = _cancel
    ex = FieldExtractor(video, roi,
                        frame_start=frame_start,
                        frame_end=frame_start + frames,
                        decode_backend=backend,
                        ocr_backend=ocr_backend,
                        keep_crops=False, **kw)
    t0 = time.perf_counter()
    try:
        res = ex.extract()
    except _Cancelled:
        return {"cancelled": True, "segs": -1, "texts": 0,
                "uset": frozenset(), "wall": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    texts = [s.text for s in res.segments if s.text]
    return {"segs": len(res.segments), "texts": len(texts),
            "uset": frozenset(texts), "wall": wall}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True, help="x1,y1,x2,y2")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--backend", default="auto",
                    help="auto/cpu/nvdec/hybrid")
    ap.add_argument("--ocr-backend", default="auto", help="auto/cpu/tensorrt")
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--cancel-after", type=int, default=0,
                    help="第 N 次 cancel_check 后抛异常，走取消路径 A/B"
                         "（0=不取消）；用于验证取消后 producer/worker 是否"
                         "残留、下一任务是否被干扰")
    args = ap.parse_args()
    roi = tuple(int(x) for x in args.roi.split(","))

    if args.cancel_after:
        return _cancel_probe(args, roi)
    return _repeat_probe(args, roi)


def _leftover(base_ids: set) -> list[str]:
    return [t.name for t in threading.enumerate()
            if t.ident not in base_ids and t is not threading.main_thread()]


def _cancel_probe(args, roi: tuple) -> int:
    """取消路径：正常轮 → 取消轮 → 正常轮，检查残留线程与可恢复性。"""
    print(f"== 取消路径 == backend={args.backend} ocr={args.ocr_backend} "
          f"cancel-after={args.cancel_after} "
          f"frames=[{args.frame_start},{args.frame_start + args.frames})")
    base_ids = {t.ident for t in threading.enumerate()}
    ok = True

    r1 = _run_once(args.video, roi, args.backend, args.ocr_backend,
                   args.frame_start, args.frames)
    print(f"  [正常 1] 段={r1['segs']} 唯一={len(r1['uset'])} "
          f"墙钟={r1['wall']:.3f}s")

    rc = _run_once(args.video, roi, args.backend, args.ocr_backend,
                   args.frame_start, args.frames,
                   cancel_after=args.cancel_after)
    imm = _leftover(base_ids)
    time.sleep(1.0)                    # 给后台线程退出时间（join 应已保证）
    delayed = _leftover(base_ids)
    print(f"  [取消]   cancelled={rc.get('cancelled', False)} "
          f"墙钟={rc['wall']:.3f}s 残留线程 立即={len(imm)}{imm or ''} "
          f"1s后={len(delayed)}{delayed or ''}")
    if not rc.get("cancelled"):
        print("  [FAIL] cancel_check 未被触发，取消路径未真正走到")
        ok = False
    if imm:
        print(f"  [FAIL] 取消返回瞬间仍有残留线程: {imm}")
        ok = False
    if delayed:
        print(f"  [FAIL] 取消 1s 后仍有残留线程: {delayed}")
        ok = False

    r2 = _run_once(args.video, roi, args.backend, args.ocr_backend,
                   args.frame_start, args.frames)
    same = (r2["segs"] == r1["segs"] and r2["uset"] == r1["uset"])
    print(f"  [正常 2] 段={r2['segs']} 唯一={len(r2['uset'])} "
          f"墙钟={r2['wall']:.3f}s 与正常1一致={same}")
    if not same:
        print("  [FAIL] 取消后的下一任务结果与取消前不一致")
        ok = False
    print(f"== 结论 ==  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _repeat_probe(args, roi: tuple) -> int:
    base_ids = {t.ident for t in threading.enumerate()}
    rows: list[dict] = []
    print(f"== 连续提取稳定性 == backend={args.backend} "
          f"ocr={args.ocr_backend} rounds={args.rounds} "
          f"frames=[{args.frame_start},{args.frame_start + args.frames})")
    for i in range(args.rounds):
        r = _run_once(args.video, roi, args.backend, args.ocr_backend,
                      args.frame_start, args.frames)
        leftover = [t.name for t in threading.enumerate()
                    if t.ident not in base_ids
                    and t is not threading.main_thread()]
        vram = _vram_mb()
        r.update(leftover=leftover, vram=vram)
        rows.append(r)
        vs = "n/a" if vram is None else f"{vram:.0f}MB"
        print(f"  round {i + 1}: 段={r['segs']} 文本={r['texts']} "
              f"唯一={len(r['uset'])} 墙钟={r['wall']:.3f}s "
              f"残留线程={len(leftover)}{leftover if leftover else ''} "
              f"显存={vs}")

    walls = [r["wall"] for r in rows]
    med = statistics.median(walls)
    print("== 汇总 ==")
    print(f"  段数集合      : {sorted({r['segs'] for r in rows})}")
    print(f"  唯一文本数集合: {sorted({len(r['uset']) for r in rows})}")
    print(f"  墙钟 median   : {med:.3f}s  max={max(walls):.3f}s  "
          f"min={min(walls):.3f}s")
    print(f"  漂移          : max/median={max(walls) / med:.3f}  "
          f"末轮/首轮={walls[-1] / walls[0]:.3f}")
    vr = [r["vram"] for r in rows if r["vram"] is not None]
    if len(vr) >= 2:
        print(f"  显存 首={vr[0]:.0f}MB 末={vr[-1]:.0f}MB "
              f"增长={vr[-1] - vr[0]:+.0f}MB")

    ok = True
    if len({r["segs"] for r in rows}) != 1:
        print("  [FAIL] 各轮段数不一致 —— 存在跨轮状态污染")
        ok = False
    if any(r["uset"] != rows[0]["uset"] for r in rows):
        print("  [FAIL] 各轮唯一文本集不一致")
        ok = False
    if any(r["leftover"] for r in rows):
        names = sorted({n for r in rows for n in r["leftover"]})
        print(f"  [FAIL] extract 返回后仍有残留线程: {names}")
        ok = False
    if len(vr) >= 2 and vr[-1] - vr[0] > 200:
        print(f"  [WARN] 显存随轮次增长 {vr[-1] - vr[0]:+.0f}MB（>200MB）")
        ok = False
    print(f"== 结论 ==  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
