"""TRT profile max_batch 的收益上界（S6 轮；消费端提交开销的最后一块）。

背景：`ocr.infer` 实测 1.006s / 1083 张 = 0.93 ms/张，而 rec 齐批上限
1323 fps = 0.756 ms/张 → 差 ~0.17 ms/张落在**每子批一次的 launch+归约+D2H+
形状 sync**（max_batch=6 → 186 子批 / 3000 帧）。本探针用同一份 ONNX 构建
两个 profile（batch = 6 与 18），在**同一批输入**上走同一条 TRT 提交路径，
把"改 profile 能拿回多少"量成数字——不先建引擎就谈这条收益是猜。

用法：python tools/_probe_trt_maxbatch.py [--images 1116] [--maxb 6,18]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def models_dir() -> Path:
    from video_ocr_engine.config import constants as config
    for cand in (config.models_dir(),
                 Path(config.app_data_dir()) / "ocr_engines" / "models"):
        if (cand / "PP-OCRv6_rec_small.onnx").exists():
            return cand
    raise SystemExit("找不到 PP-OCRv6_rec_small.onnx（assets/ocr_models）")


def make_engine(opt_b: int, path: Path, size: str = "small"):
    """构建（若缺）+ 加载指定 profile batch 的引擎。"""
    from video_ocr_engine.config import constants as config
    from video_ocr_engine.ocr.trt import TrtEngine
    models = models_dir()
    if not path.exists():
        saved = config.TRT_PROFILE_BATCH
        config.TRT_PROFILE_BATCH = opt_b
        try:
            inst = TrtEngine.__new__(TrtEngine)
            inst._progress_cb = None
            inst.engine_path = None
            t = time.perf_counter()
            inst._build(models, size, path)
            print("构建 max_batch=%-3d 引擎 %.1fs → %s"
                  % (opt_b, time.perf_counter() - t, path.name))
        finally:
            config.TRT_PROFILE_BATCH = saved
    orig = TrtEngine._engine_candidates
    TrtEngine._engine_candidates = staticmethod(lambda s: [path])
    try:
        eng = TrtEngine(models, size)
    finally:
        TrtEngine._engine_candidates = orig
    return eng


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=1116)
    ap.add_argument("--maxb", default="6,18")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    import numpy as np
    from video_ocr_engine.config import constants as config

    rng = np.random.default_rng(7)
    H, W = config.OCR_TARGET_H, 224
    imgs = [rng.integers(0, 255, (H, W), dtype=np.uint8)
            for _ in range(args.images)]
    x_all = np.stack(imgs).astype(np.float32)
    x_all = x_all.reshape(len(imgs), 1, H, W).repeat(3, axis=1)

    tmp = Path(config.app_data_dir()) / "ocr_engines" / "_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    for mb in (int(v) for v in args.maxb.split(",")):
        eng = make_engine(mb, tmp / ("probe_maxb%d.engine" % mb))
        walls = []
        for _ in range(args.reps):
            t = time.perf_counter()
            for s in range(0, len(imgs), mb):
                eng.execute_async(x_all[s:s + mb])
                eng.synchronize()
            walls.append(time.perf_counter() - t)
        best = min(walls)
        out[mb] = best
        print("max_batch=%-3d %4d 张  %.3fs → %.1f 张/s（子批 %d，每次同步）"
              % (mb, len(imgs), best, len(imgs) / best, -(-len(imgs) // mb)))
    ref = out[min(out)]
    for mb, v in sorted(out.items()):
        print("  max_batch=%-3d vs %d：%+.1f%%" % (mb, min(out), (v - ref) / ref * 100))
    print("RESULT " + json.dumps({str(k): round(v, 4) for k, v in out.items()}))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
