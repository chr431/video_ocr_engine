"""预处理容量膝点探针（跨线程账本轮，2026-10-09）。

问题：「预处理换成更重的图像增强算法」流水线吃得住吗？绑定实验回答
（C-42 纪律：占空/归因≠可回收量，容量必须实测）：给宿主预处理
`preprocess_standard` 注入每 crop N ms 的额外耗时（sleep=纯时间线占用，
模型化"算法变重变慢"），扫 N 找 e2e 从平价转入 1:1 上涨的膝点。
膝点前的注入被流水线空转吸收（免费），膝点后直接加墙钟——膝点即
"更重算法的时间预算"。

口径：test5.mp4 600 帧 stride=1，cpu 解码 + OpenVINO OCR（宿主全链路，
预处理住 OCR worker 线程；GPU 臂预处理在设备侧 kernel，本探针不覆盖，
其容量由 NVML/占空数据推导，见叙事）。

用法：python tools/_probe_prep_headroom.py [--steps 0,0.5,1,2,4,8 --reps 3]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="0,0.5,1,2,4,8",
                    help="每 crop 注入毫秒序列（逗号分隔）")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()
    steps = [float(x) for x in args.steps.split(",") if x.strip()]

    import os
    vid = os.path.join(
        os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"),
        "test5.mp4")
    roi = (843, 993, 948, 1025)

    from video_ocr_engine import FieldExtractor
    import video_ocr_engine.pipeline.ocr_stage as _stage

    _orig = _stage.preprocess_standard

    def run_once(ms: float) -> tuple[float, float, int]:
        """返回 (e2e 墙钟, preprocess span 和, 段数)。"""
        if ms <= 0:
            _stage.preprocess_standard = _orig
        else:
            def wrapped(crop, force_aspect=0.0, gamma=0.0):
                time.sleep(ms / 1000.0)
                return _orig(crop, force_aspect=force_aspect, gamma=gamma)
            _stage.preprocess_standard = wrapped
        try:
            ex = FieldExtractor(vid, roi, frame_start=0, frame_end=args.frames,
                                decode_backend="cpu", ocr_backend="cpu")
            t0 = time.perf_counter()
            r = ex.extract()
            wall = time.perf_counter() - t0
            rep = r.meta.get("report") or {}
            sp = (rep.get("spans") or {}).get("ocr.preprocess", {})
            return wall, float(sp.get("sum", 0.0)), len(r.segments)
        finally:
            _stage.preprocess_standard = _orig

    # 预热（引擎池/NVRTC 冷启动不进测量）
    run_once(0)

    print("注入模型：sleep/crop（纯时间线占用；每步 %d reps 取中位）"
          % args.reps)
    print("%-8s %10s %10s %8s  %s"
          % ("ms/crop", "e2e(s)", "prep(s)", "Δe2e%", "段数"))
    base = None
    for ms in steps:
        walls = []
        prep = None
        segs = 0
        for _ in range(args.reps):
            w, p, s = run_once(ms)
            walls.append(w)
            prep, segs = p, s
        med = statistics.median(walls)
        if base is None:
            base = med
        print("%-8.1f %10.4f %10.4f %+7.1f%%  %d"
              % (ms, med, prep, (med - base) / base * 100, segs))
    print("\n判读：Δe2e%% ≈0 的最右注入 = 免费预算；此后每 +1ms/crop ≈ "
          "e2e +（crop 数 × 1ms）——膝点即重算法的时间预算上限")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
