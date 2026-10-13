"""pytest 配置。

三个作用：

1. 把**仓库根**加入 `sys.path` —— 否则从非仓库根目录跑 pytest 时，
   `import engine_config` / `import video_ocr_engine` 会
   `ModuleNotFoundError`（`python -m pytest` 只把 CWD 放进 sys.path）。
2. 把 `tests/` 加入 `sys.path`，使各子目录的测试都能
   `from _paths import ROOT`（见 `_paths.py`）——不要按目录深度猜仓库根。
3. 作为 rootdir 锚点：`pytest tests/ -v` 与各子目录单独跑行为一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent

for _p in (str(ROOT), str(TESTS), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── 资源守卫 fixtures ──────────────────────────────────────────────────
# 新测试不再各自实现"视频在不在/GPU 可不可用"的检查——用这两个 fixture。
# 视频注册表单一事实源 = tools/_measure_lib.py（VIDS/ROI）。
import pytest  # noqa: E402


def _videos_ready() -> bool:
    try:
        from _measure_lib import has_real_videos

        return has_real_videos
    except Exception:
        return False


def _gpu_ready() -> bool:
    if not _videos_ready():
        return False
    try:
        import decord  # noqa: F401
        import video_ocr_engine.gpu.device  # noqa: F401

        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def real_video():
    """真值视频环境（缺视频 → skip）。yield (视频路径, ROI)。"""
    if not _videos_ready():
        pytest.skip("需要真值视频（RACELOG_VIDEO_DIR）")
    from _measure_lib import ROI, VIDS

    return VIDS["test5"], ROI["test5"]


@pytest.fixture(scope="session")
def gpu_env():
    """GPU + decord fork + 真值视频（缺任一 → skip）。"""
    if not _gpu_ready():
        pytest.skip("需要 GPU + decord fork + 真值视频")
    from _measure_lib import ROI, VIDS

    return VIDS["test5"], ROI["test5"]
