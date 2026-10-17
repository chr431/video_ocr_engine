"""OCR 推理冒烟（CPU/OpenVINO 后端；需仓库内模型资产，无需 decord/GPU）。"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from video_ocr_engine.ocr.native import OcrEngine


def _probe_openvino() -> tuple[bool, str]:
    """子进程探测本机能否安全跑 OpenVINO 推理。

    CI 的 windows runner 存在 VM 异构：同一 openvino 版本在部分 runner 上以
    0xC000001D（非法指令）硬崩，进程直接退出，pytest 连 FAILED 都写不出来
    （pyproject 依赖注释已记录 2026.4.0 的同类现象）。这类崩溃无法在进程内
    catch，只能先放子进程里试：**只有崩溃类返回码**才判"环境不满足前提"并
    SKIP；导入失败、缺模型资产等真实问题照常让测试失败，不冒充通过。
    """
    code = (
        "import numpy as np\n"
        "from video_ocr_engine.ocr.native import OcrEngine\n"
        "e = OcrEngine('v6_small', 'openvino', fill_width=224, num_threads=1)\n"
        "r = e([np.zeros((48, 96, 3), dtype=np.float32)])\n"
        "assert len(r) == 1\n"
    )
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "OpenVINO 探测超时"
    err = (p.stderr or b"").decode("utf-8", "replace")
    crash = (
        p.returncode < 0  # POSIX 信号（SIGILL=-4 等）
        or p.returncode >= 0xC0000000  # Windows NTSTATUS 崩溃码
        or "Windows fatal exception" in err
    )
    if not crash:
        return True, ""
    return False, "本机 OpenVINO 推理原生崩溃（returncode=%s）——环境不满足前提" % p.returncode


_OV_OK, _OV_REASON = _probe_openvino()
pytestmark = pytest.mark.skipif(not _OV_OK, reason=_OV_REASON)


@pytest.fixture(scope="module")
def engine():
    return OcrEngine("v6_small", "openvino", fill_width=224, num_threads=2)


def test_backend_name_is_openvino(engine):
    # ORT 后端已移除（历史见 git 档案）；CPU OCR 引擎必须为 openvino
    # 引擎标识为 'openvino'（历史迁移表见 git 档案 archive/pre-rework:docs/MIGRATION.md）：
    # CPU 路径唯一引擎 = OpenVINO。
    assert engine.backend_name == "openvino"


def test_openvino_backend_returns_one_result_per_input(engine):
    # 输入为 _preprocess_standard 输出风格 (48, w, 3) float32
    img = np.zeros((48, 96, 3), dtype=np.float32)
    res = engine([img, img.copy()])
    assert len(res) == 2
    for r in res:
        assert isinstance(r.txts, tuple)
        assert len(r.txts) == 1
        assert len(r.scores) == 1
        assert 0.0 <= r.scores[0] <= 1.0


def test_openvino_empty_input(engine):
    assert engine([]) == []


def test_ctc_decode_blank_only(engine):
    # 全 blank(logits 0 = blank) → 空文本、置信度 0
    pred = np.zeros((1, 8, 6906), dtype=np.float32)
    pred[:, 1:, 0] = 1.0  # 除首帧外选 blank
    out = engine._ctc_decode_batch(pred)
    assert len(out) == 1
    assert out[0].txts == ("",)
    assert out[0].scores[0] == 0.0
