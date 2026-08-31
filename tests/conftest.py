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

for _p in (str(ROOT), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
