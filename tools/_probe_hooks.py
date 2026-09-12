"""探针打桩点间接层（L1）：把「探针要碰的内部点」收敛成**一张表**。

## 要解决的问题
探针靠 monkeypatch 私有符号取数（`_cluster_win3` / `_segment_frames` /
`SegmentStateMachine`…），这些符号无任何稳定性承诺 → 每次重构批量破。
2026-09-13 实测：S9 模块化一次打破 4 个探针，且**每个都要重新读代码找
新调用方**（旧点位 `_host_pipeline.py` 已删，新点位分散在两个后端模块）。

本层把该成本从 O(探针数 × 重构次数) 降到 **O(重构次数)**：重构后只改本表
的 `POINTS`，全部探针不动。

## 用法
    from _probe_hooks import hooks
    with hooks.using("STATE_MACHINE", RecMachine):
        ex.extract()                       # 自动装桩 / 卸桩
    # 或者手工：
    hooks.patch("CLUSTER_WIN3", _w)
    hooks.restore("CLUSTER_WIN3")

## 规则（tools/INDEX.md 同名条目）
- 新增探针**一律经本表取点**，不得直接 `import` 产品私有符号。
- 重构改了内部布局 → 只改 `POINTS`，然后跑 `python tools/_probe_hooks.py`
  自检（它逐个解析表项，任一项失效即非 0 退出）。
"""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

# 项目约定（AGENTS.md）：探针及其依赖库须自己把仓库根放进 sys.path——
# tools/ 下 sys.path[0] 是 tools/，直接 import 产品包会 ModuleNotFoundError。
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: 逻辑名 → (模块路径, 符号名)。**重构后只改这里。**
#:
#: 选点原则：取「调用点所在模块的模块属性」——monkeypatch 模块属性才能拦住
#: 内部调用。注意 `extractor` 走函数内 `from ... import`（每次调用重解析），
#: 所以打 `domain.segmentation` 的模块属性即可拦住它，无需另打一处。
POINTS: dict[str, tuple[str, str]] = {
    # 分段状态机：两个执行后端各自 by-value 导入，故必须分别打桩
    "STATE_MACHINE": ("video_ocr_engine.pipeline.host_backend",
                      "SegmentStateMachine"),
    "STATE_MACHINE_GPU": ("video_ocr_engine.pipeline.gpu_backend",
                          "SegmentStateMachine"),
    # 宿主分段接线（原 _host_pipeline._host_segment_frames，S9 迁入并改名）
    "SEGMENT_FRAMES": ("video_ocr_engine.pipeline.host_backend",
                       "_segment_frames"),
    # 分段原语（调用点即 segmentation 内的模块全局）
    "CLUSTER_WIN3": ("video_ocr_engine.domain.segmentation", "_cluster_win3"),
    "OTSU": ("video_ocr_engine.domain.segmentation", "_otsu"),
    "TEXT_SEP_BINARY": ("video_ocr_engine.domain.segmentation",
                        "_text_sep_binary"),
    # 相似段判定（FieldExtractor 方法，按实例打桩）
    "SEGMENTS_SIMILAR_METHOD": ("video_ocr_engine.extractor",
                                "FieldExtractor"),
}


class _Hooks:
    def __init__(self) -> None:
        self._saved: dict[str, object] = {}

    # ── 解析 ────────────────────────────────────────────────────
    @staticmethod
    def point(name: str) -> tuple[str, str]:
        try:
            return POINTS[name]
        except KeyError:
            raise KeyError(
                "未知打桩点 %r；已登记：%s（如需新点位请改 "
                "tools/_probe_hooks.py 的 POINTS，不要在探针里直接 import "
                "私有符号）" % (name, ", ".join(sorted(POINTS)))) from None

    def module(self, name: str):
        mod, _sym = self.point(name)
        return importlib.import_module(mod)

    def get(self, name: str):
        _mod, sym = self.point(name)
        return getattr(self.module(name), sym)

    # ── 装桩 / 卸桩 ────────────────────────────────────────────
    def patch(self, name: str, obj) -> object:
        """把打桩点替换为 obj，返回原对象（便于调用方自行恢复）。"""
        mod = self.module(name)
        _m, sym = self.point(name)
        orig = getattr(mod, sym)
        if name not in self._saved:
            self._saved[name] = orig
        setattr(mod, sym, obj)
        return orig

    def restore(self, name: str) -> None:
        if name in self._saved:
            mod = self.module(name)
            _m, sym = self.point(name)
            setattr(mod, sym, self._saved.pop(name))

    def restore_all(self) -> None:
        for name in list(self._saved):
            self.restore(name)

    @contextmanager
    def using(self, name: str, obj):
        self.patch(name, obj)
        try:
            yield obj
        finally:
            self.restore(name)

    # ── 自检 ────────────────────────────────────────────────────
    def audit(self) -> list[str]:
        """逐项解析 POINTS；返回失效列表（空 = 全绿）。"""
        bad = []
        for name, (mod, sym) in sorted(POINTS.items()):
            try:
                m = importlib.import_module(mod)
            except Exception as e:              # noqa: BLE001
                bad.append("%s: 模块 %s 不可导入（%s）" % (name, mod, e))
                continue
            if not hasattr(m, sym):
                bad.append("%s: %s 里已无符号 %s —— 请更新 POINTS 表"
                           % (name, mod, sym))
        return bad


hooks = _Hooks()


def main() -> int:
    bad = hooks.audit()
    print("打桩点自检：%d 项" % len(POINTS))
    for name, (mod, sym) in sorted(POINTS.items()):
        print("  %-24s -> %s.%s" % (name, mod, sym))
    if bad:
        print("\n✗ 失效 %d 项：" % len(bad))
        for b in bad:
            print("   ", b)
        return 1
    print("\n✓ 全部可解析")
    return 0


if __name__ == "__main__":
    sys.exit(main())
