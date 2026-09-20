"""P4 护栏：相位/遥测键注册表一致性（静态 AST 扫描）。

防两类历史漂移（证据：「P2c 对齐」系列——宿主缺 'open' 边界、CPU 解码
分支零打桩、merge_pair 只在 GPU 路径产出）：

1. 新增 ``prof_end`` 键未进 ``domain/metrics.py`` 映射表 → 指标静默缺席
   （profile 字典有、telemetry 没有，diff 读数对不上）；
2. 某条路径（宿主 / GPU / CPU 解码分支）的键集悄悄变化 → L1 per_phase
   两侧读数不可比。

规则：**改键 = 必须同步改本文件的注册表**（diff 可见，评审时刻即
"另一条路径要不要同键"的检查时刻）。扫描对象 = pipeline/ + gpu/ +
extractor.py 的全部 ``_prof``/``prof_end``/``.checkpoint`` 调用点
（字面量参数；转发壳天然跳过）。
"""
from __future__ import annotations

import ast
from pathlib import Path

from _paths import ROOT

# ── 注册表（唯一允许的键集声明处；改键先改这里）──────────────────────
# 路径相对仓库根；只列「有键」的模块——新模块带键而未声明 = 失败。
PHASE_KEYS_BY_MODULE: dict[str, set[tuple[str, str]]] = {
    "video_ocr_engine/pipeline/_driver.py": {
        ("producer", "calib_total"),
        ("producer", "consume_feed"),
        ("producer", "consumer_total"),
        ("producer", "open_and_fps"),
    },
    "video_ocr_engine/pipeline/host_backend.py": {
        ("producer", "bin_batch"),
        ("producer", "decode_batch"),
        ("producer", "gray_batch"),
        ("producer", "merge_pair"),
        ("producer", "q_put_block"),
        ("producer", "sharp_batch"),
    },
    "video_ocr_engine/pipeline/gpu_backend.py": {
        ("producer", "emit_autocrop"),
        ("producer", "emit_d2h"),
        ("producer", "emit_put"),
        ("producer", "merge_pair"),
        ("producer", "q_put_block"),
    },
    "video_ocr_engine/gpu/device.py": {
        ("producer", "decode_batch"),
        ("producer", "gray_batch"),
        ("producer", "stream_analyze"),
    },
    "video_ocr_engine/pipeline/ocr_stage.py": {
        ("ocr", "ctc_decode"),
        ("ocr", "engine_init"),
        ("ocr", "infer"),
        ("ocr", "preprocess"),
        ("ocr", "preproc_autocrop"),
        ("ocr", "preproc_luma"),
        ("ocr", "preproc_resize"),
        ("ocr", "q_get_wait"),
    },
}

# L1 资源边界 checkpoint（§8.6 r5）：P1 驱动合一后由 _driver.py 单点发出
# （旧两后端各发一份时的「宿主缺 open」漂移即 P2c 所修）。
CHECKPOINTS_BY_MODULE: dict[str, set[str]] = {
    "video_ocr_engine/pipeline/_driver.py":
        {"open", "calibrate", "decode", "ocr"},
}

_SCAN_ROOTS = ("video_ocr_engine/pipeline", "video_ocr_engine/gpu")
_SCAN_FILES = ("video_ocr_engine/extractor.py",)


def _iter_scan_paths() -> list[Path]:
    out: list[Path] = []
    for rel in _SCAN_ROOTS:
        out.extend(sorted((ROOT / rel).glob("*.py")))
    for rel in _SCAN_FILES:
        out.append(ROOT / rel)
    return [p for p in out if p.exists()]


def _collect(path: Path) -> tuple[set[tuple[str, str]], set[str]]:
    """返回 (prof 键集, checkpoint 集)，只认字面量参数。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keys: set[tuple[str, str]] = set()
    checkpoints: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.attr if isinstance(node.func, ast.Attribute)
                else node.func.id if isinstance(node.func, ast.Name) else "")
        args = node.args
        if name == "checkpoint":
            if args and isinstance(args[0], ast.Constant):
                checkpoints.add(args[0].value)
            continue
        if name in ("_prof", "prof_end", "_prof_end"):
            # _prof(spec, group, key, t0) / prof_end(group, key, t0)
            gi, ki = (1, 2) if name == "_prof" else (0, 1)
            if (len(args) > ki
                    and isinstance(args[gi], ast.Constant)
                    and isinstance(args[ki], ast.Constant)):
                keys.add((args[gi].value, args[ki].value))
    return keys, checkpoints


def test_all_prof_keys_are_mapped_in_metrics_registry():
    """漂移 1：任何调用点键都必须进 metrics 映射表（不进 = 指标静默缺席）。"""
    from video_ocr_engine.domain.metrics import (
        PROFILE_GAUGES, PROFILE_SPANS, PROFILE_TOTALS)
    mapped = set(PROFILE_SPANS) | set(PROFILE_TOTALS) | set(PROFILE_GAUGES)
    unmapped: list[str] = []
    for path in _iter_scan_paths():
        keys, _ = _collect(path)
        bad = keys - mapped
        unmapped += [f"{path.relative_to(ROOT)}: {g}.{k}" for g, k in
                     sorted(bad)]
    assert not unmapped, (
        f"存在未映射的 prof 键（domain/metrics.py 缺席，telemetry 将静默"
        f"无此指标）：{unmapped}")


def test_phase_key_sets_match_registry():
    """漂移 2：各模块键集必须与注册表逐一致（改键 = 同步改注册表）。"""
    problems: list[str] = []
    for path in _iter_scan_paths():
        rel = path.relative_to(ROOT).as_posix()
        keys, _ = _collect(path)
        declared = PHASE_KEYS_BY_MODULE.get(rel, set())
        if keys != declared:
            if keys - declared:
                problems.append(
                    f"{rel}: 出现未注册键 {sorted(keys - declared)}"
                    f"（先更新 PHASE_KEYS_BY_MODULE，并检查另一条路径"
                    f"是否应同键——P2c 类漂移的检查时刻就是现在）")
            if declared - keys:
                problems.append(
                    f"{rel}: 注册表有而代码无 {sorted(declared - keys)}"
                    f"（键被删/改名，注册表同步清理）")
    assert not problems, "\n".join(problems)


def test_backend_checkpoint_sets_identical():
    """两后端 L1 checkpoint 必须同集（per_phase 差分可比性的前提）。"""
    per_mod: dict[str, set[str]] = {}
    for path in _iter_scan_paths():
        rel = path.relative_to(ROOT).as_posix()
        _, cps = _collect(path)
        if cps:
            per_mod[rel] = cps
    assert per_mod == CHECKPOINTS_BY_MODULE, (
        f"checkpoint 集与注册表不符：{per_mod} != {CHECKPOINTS_BY_MODULE}；"
        f"宿主/GPU 两后端的 L1 相位边界集合不同，diff 读数不可比")
