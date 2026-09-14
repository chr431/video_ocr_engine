"""_probe_ocr_phase_split.py —— h264 生产路径 OCR 批延迟拆相（launch 裸奔定量）。

命题
----
GPU 侧立项轮（log 2026-09-14-GPU侧立项轮 §0）：h264 生产 `ocr.infer`
p50 12.2ms/批 vs 裸 TRT 紧循环 5.6ms（同形 B=18/W=224），差额归因
"launch 裸奔"。本探针把生产批延迟拆成四相，为深度 2 延迟收集流水
（TRT_DEFER_SYNC）提供前后对照：

  prep_submit  GPU 预处理提交（GpuPreprocessor.process_gray_raw 的
               host 侧耗时：launch 提交，不含 GPU 执行）
  trt_call     execute_device_argmax 整段（TRT enqueue 提交 + argmax
               kernel 提交 + D2H 提交 + 流同步等待——同步在 reducer.reduce 内）
  ctc_host     _ctc_from_idxprob（宿主 CTC 组装）
  其余          ocr.infer 总和 − 三相之和（infer_worker 调度、metrics 等）

口径：test5 全片（7761 帧）、hybrid 解码 + TRT + GPU_CTC=1（生产默认）、
ENGINE_PROFILE=1。跑两轮取第二轮（热池=生产口径；首轮含 TRT 反序列化）。
逐批 p50 由探针自记（infer_worker 的 prof span 是累计和，p50 需自采）。

用法
----
    python tools/_probe_ocr_phase_split.py [--frames N] [--runs 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

VID = os.path.join(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"),
                   "test5.mp4")
ROI = (843, 993, 948, 1025)


def run_once(frames: int, tag: str) -> None:
    os.environ["ENGINE_PROFILE"] = "1"
    import video_ocr_engine.ocr.trt as trt_mod
    from video_ocr_engine import FieldExtractor
    from video_ocr_engine._gpu_kernels import GpuPreprocessor

    acc = {"prep_submit": [0.0, 0], "trt_call": [0.0, 0],
           "submit": [0.0, 0], "collect": [0.0, 0], "ctc_host": [0.0, 0]}
    per_batch: list[float] = []

    p_gs = GpuPreprocessor.process_gray_raw
    def t_gs(self, *a, **kw):
        t0 = time.perf_counter()
        r = p_gs(self, *a, **kw)
        acc["prep_submit"][0] += time.perf_counter() - t0
        acc["prep_submit"][1] += 1
        return r
    GpuPreprocessor.process_gray_raw = t_gs

    p_ea = trt_mod.TrtEngine.execute_device_argmax
    def t_ea(self, *a, **kw):
        t0 = time.perf_counter()
        r = p_ea(self, *a, **kw)
        dt = time.perf_counter() - t0
        acc["trt_call"][0] += dt
        acc["trt_call"][1] += 1
        per_batch.append(dt)
        return r
    trt_mod.TrtEngine.execute_device_argmax = t_ea

    from video_ocr_engine.ocr.native import OcrEngine
    p_ctc = OcrEngine._ctc_from_idxprob
    def t_ctc(self, *a, **kw):
        t0 = time.perf_counter()
        r = p_ctc(self, *a, **kw)
        acc["ctc_host"][0] += time.perf_counter() - t0
        acc["ctc_host"][1] += 1
        return r
    OcrEngine._ctc_from_idxprob = t_ctc

    # 深度 2 路径（TRT_DEFER_SYNC=1）：submit（提交整批）/ collect
    # （事件同步+读环+CTC 前的一切）。trt_call 仅同步回退批计入。
    p_sub = OcrEngine._gpu_raw_submit
    def t_sub(self, *a, **kw):
        t0 = time.perf_counter()
        r = p_sub(self, *a, **kw)
        acc["submit"][0] += time.perf_counter() - t0
        acc["submit"][1] += 1
        return r
    OcrEngine._gpu_raw_submit = t_sub
    p_col = OcrEngine._gpu_raw_collect
    def t_col(self, *a, **kw):
        t0 = time.perf_counter()
        r = p_col(self, *a, **kw)
        acc["collect"][0] += time.perf_counter() - t0
        acc["collect"][1] += 1
        return r
    OcrEngine._gpu_raw_collect = t_col

    ex = FieldExtractor(VID, ROI, frame_end=frames, sample_stride=1,
                        decode_backend="hybrid", ocr_backend="tensorrt")
    t0 = time.perf_counter()
    res = ex.extract()
    wall = time.perf_counter() - t0
    n = acc["trt_call"][1]
    inf = ex.profile.get("ocr", {}).get("infer", 0.0)
    ctc_prof = ex.profile.get("ocr", {}).get("ctc_decode", 0.0)
    pb = sorted(per_batch)
    p50 = pb[len(pb) // 2] if pb else 0.0
    p90 = pb[int(len(pb) * 0.9)] if pb else 0.0
    print("\n===== %s wall=%.3fs segs=%d batches=%d" % (tag, wall,
          len(res.segments), n), flush=True)
    print("  ocr.infer(sum)=%.3fs  批 p50=%.1fms p90=%.1fms  ctc_decode=%.3fs"
          % (inf, p50 * 1e3, p90 * 1e3, ctc_prof), flush=True)
    for k in ("prep_submit", "trt_call", "submit", "collect", "ctc_host"):
        s, c = acc[k]
        print("  %-12s sum=%.3fs  均值=%.2fms/批(n=%d)"
              % (k, s, (s / c * 1e3) if c else 0, c), flush=True)
    rest = inf - sum(acc[k][0] for k in acc)
    print("  其余(infer−三相)=%.3fs（均值 %.2fms/批）"
          % (rest, (rest / n * 1e3) if n else 0), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=7761)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    for i in range(args.runs):
        run_once(args.frames, "run%d%s" % (i, "/热" if i else "/冷"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
