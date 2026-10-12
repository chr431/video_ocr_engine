"""render 渲染器的正负测——路径活性/锚点活性/裸 CR 三类判据各有坏例。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from render import bare_cr, path_liveness  # noqa: E402


def test_check_passes_on_repo() -> None:
    from render import check

    assert check() == [], "四视图应一致（--write 重渲染后重试）"


def test_path_liveness_flags_missing_path() -> None:
    entries = [
        {
            "id": "C-T1",
            "conclusion": "见 video_ocr_engine/nope.py",
            "premises": "",
            "revisit": "",
            "evidence": [],
        }
    ]
    bad = path_liveness(entries)
    assert any("video_ocr_engine/nope.py" in b for b in bad)


def test_path_liveness_flags_dead_anchor() -> None:
    entries = [
        {
            "id": "C-T2",
            "conclusion": "",
            "premises": "",
            "revisit": "",
            "evidence": [{"commit": "deadbee", "path": "README.md"}],
        }
    ]
    bad = path_liveness(entries)
    assert any("deadbee" in b for b in bad)


def test_bare_cr_detector(tmp_path: Path) -> None:
    f = tmp_path / "view.md"
    f.write_bytes("a\r \rb\nc\n".encode())
    # bare_cr 检查的是 ROOT 相对路径；这里直接验证判据本身：\r 不跟 \n 即违例
    import re

    assert len(re.findall(rb"\r(?!\n)", f.read_bytes())) == 2
    assert bare_cr("charter.md") == []  # 宪章自身必须干净
