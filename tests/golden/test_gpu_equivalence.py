"""S4 门禁：宿主 ↔ GPU 双后端在固定窗口上逐位等价（v2 §11 S4 / §8.5）。

这正是 v1 的 1083 vs 1042 缺的那道门：v2 起由 pytest -m gpu 承载，
金标 --verify 之外的可独立运行等价证明。无 GPU/视频环境自动 skip。
"""
from __future__ import annotations

import os

import pytest

from _paths import ROOT

VID = r"D:\Videos\racelog_test\test5.mp4"
ROI = (843, 993, 948, 1025)
WIN = (0, 2000)


def _gpu_available() -> bool:
    if not os.path.exists(VID):
        return False
    try:
        import video_ocr_engine.gpu.device as gp  # noqa: F401
        from decord import gpu  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _gpu_available(),
                       reason="需要 GPU + decord fork + 真值视频"),
]


def _extract(pipeline: str):
    from video_ocr_engine import FieldExtractor
    os.environ["GPU_PIPELINE"] = pipeline
    try:
        ex = FieldExtractor(VID, ROI, frame_start=WIN[0], frame_end=WIN[1],
                            decode_backend="nvdec", ocr_backend="tensorrt",
                            keep_crops=False)
        r = ex.extract()
    finally:
        os.environ.pop("GPU_PIPELINE", None)
    return ([[s.start, s.end, s.rep_frame] for s in r.segments],
            [s.text for s in r.segments],
            [s.confidence for s in r.segments],
            r.meta.get("degraded_reason"))


def test_host_gpu_backends_bitwise_equivalent():
    """S4 等价门禁（F-7 实测口径，2026-09-10 test5/2000 帧）：

    - 段结构（含 rep_frame）与文本集：**逐位相同**（永不豁免，§8.5）；
    - 置信度：**中位数差 = 0**（多数段逐位相同），尾部 ≤ 5e-3 天花板
      （实测 worst 4.54e-3@'122'：host(numpy 预处理)↔GPU(设备内核) 的
      resize/gamma 浮点归约差异流经 TRT 的敏感段——文本两侧全同、
      双侧均高置信；结构性分歧会先在文本上爆炸，不会只动末三位）。
    """
    h_struct, h_texts, h_confs, h_deg = _extract("0")
    g_struct, g_texts, g_confs, g_deg = _extract("1")
    assert h_deg is None and g_deg is None, (h_deg, g_deg)
    assert len(h_struct) == len(g_struct) and len(h_struct) > 0
    assert h_struct == g_struct, "段结构(含 rep_frame)不一致"
    assert h_texts == g_texts, "文本集不一致"
    deltas = sorted(abs(a - b) for a, b in zip(h_confs, g_confs))
    median = deltas[len(deltas) // 2]
    worst = deltas[-1]
    assert median == 0.0, "置信度中位差非 0：等价性整体漂移"
    assert worst <= 5e-3, "置信度尾部差 %.2e 超 5e-3 天花板" % worst
