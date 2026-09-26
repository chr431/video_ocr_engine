"""轻量泄漏回归（P8，2026-09-20 稳健性轮）：多轮 extract 的 RSS/VRAM 斜率。

`tools/_probe_leak_longrun.py` 的 pytest 化核心——C-55 池复活泄漏
（旧 Y 池 +2.0 MiB/轮，修复后 +0.000）的防复发锚。

**子进程隔离**：首版在测试进程内直接量 RSS，单跑通过、全套运行翻车
（前序 GPU 测试的分配器保留态污染基线）——泄漏测量必须在干净进程里
做，子进程跑 13 轮（warmup 3）回传逐轮 RSS/VRAM 序列，父进程断言。
无视频/无 decord 环境自动 skip（CI 上不跑）。

口径：decode=cpu + ocr=auto（GPU 管线 CPU 分支）——设备侧池
（_DevBatchPool/Y pool）与宿主路径都被覆盖。斜率用中位数（首跑实测
单轮 +130MB 瞬态保留会打穿均值口径；单调泄漏仍敏）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from _paths import ROOT

VID = os.path.join(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"),
                   "test5.mp4")
ROI = (843, 993, 948, 1025)
WARMUP = 3
MEASURE = 10
SLOPE_CAP_MB = 1.5      # C-55 修复前是 +2.0 MiB/轮；留噪声余量

_WORKER = r"""
import gc, json, sys
sys.path.insert(0, sys.argv[1])
vid, roi_s, n_iters = sys.argv[2], sys.argv[3], int(sys.argv[4])
roi = tuple(int(x) for x in roi_s.split(','))
import psutil
proc = psutil.Process()
rss, vram_free = [], []
try:
    from cuda.bindings import runtime as cudart
    def _vf():
        err, free, _t = cudart.cudaMemGetInfo()
        return None if err != 0 else free / (1048576.0)
except Exception:
    _vf = lambda: None
from video_ocr_engine import FieldExtractor
for i in range(n_iters):
    ex = FieldExtractor(vid, roi, frame_end=400, decode_backend="cpu",
                        ocr_backend="auto", keep_crops=False)
    ex.extract()
    del ex
    gc.collect()
    rss.append(proc.memory_info().rss / 1048576.0)
    vf = _vf()
    vram_free.append(vf)
print(json.dumps({"rss": rss, "vram_free": vram_free}))
"""


def _available() -> bool:
    if not os.path.exists(VID):
        return False
    try:
        import decord  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _available(),
                                reason="需要真值视频 + decord（本地回归口径）")


def _slope(seq: list[float]) -> float:
    # 中位数而非均值：单轮瞬态保留（Windows 分配器，实测 +130MB 单尖峰
    # 后回落）不打穿；C-55 型单调泄漏（+2.0 MiB/轮）仍被首尾中位数差捕获
    import statistics
    n = max(3, len(seq) // 3)
    head = statistics.median(seq[:n])
    tail = statistics.median(seq[-n:])
    return (tail - head) / max(1, len(seq) - n)


def test_repeat_extract_no_rss_or_vram_drift():
    p = subprocess.run(
        [sys.executable, "-c", _WORKER, str(ROOT), VID,
         ",".join(str(v) for v in ROI), str(WARMUP + MEASURE)],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert p.returncode == 0, (p.stderr or "")[-400:]
    d = json.loads(p.stdout.strip().splitlines()[-1])
    rss = d["rss"][WARMUP:]
    rss_slope = _slope(rss)
    assert rss_slope < SLOPE_CAP_MB, (
        f"RSS 漂移 {rss_slope:+.2f} MiB/轮 超 {SLOPE_CAP_MB}（C-55 类池泄漏"
        f"回归？逐轮：{[round(x, 1) for x in d['rss']]}）")
    vf = [v for v in d["vram_free"][WARMUP:] if v is not None]
    if len(vf) == MEASURE:
        # free 递减 = 显存累积；取负号统一为「每轮增量」
        v_slope = -_slope(vf)
        assert v_slope < SLOPE_CAP_MB, (
            f"VRAM 漂移 {v_slope:+.2f} MiB/轮 超 {SLOPE_CAP_MB}"
            f"（逐轮 free：{[round(x, 1) for x in vf]}）")
