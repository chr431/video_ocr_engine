"""S8 审计扩展（v2 §8.2 目标 21 项中的新增 9 项，13..21）。

由 _probe_discipline_audit.py 经 CHECKS_EXT 并入同一入口（钩子/CI 共用）。
每项失败即违规；历史豁免沿用既有机制（--since 快检兼容）。
"""
from __future__ import annotations

import ast
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def _product_files():
    files = [ROOT / f for f in (
        "engine_config.py", "gpu_setup.py", "ocr_native.py", "ocr_trt.py",
        "segmentation.py", "video_utils.py")]
    pkg = ROOT / "video_ocr_engine"
    files += [p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts]
    return [p for p in files if p.exists()]


def check_layering() -> str | None:
    """[13] 模块级导入图无环（P0-1b：v1 本就是 DAG，测试守住）。"""
    import importlib.util
    names = {p.stem: p for p in _product_files()}
    graph = {}
    for name, p in names.items():
        try:
            tree = ast.parse(io.open(p, encoding="utf-8").read())
        except SyntaxError:
            continue
        deps = set()
        for n in ast.walk(tree):
            if (isinstance(n, ast.ImportFrom) and n.module
                    and n.level == 0 and n.module in names):
                deps.add(n.module)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name in names:
                        deps.add(a.name)
        graph[name] = deps
    seen, stack = set(), []
    def dfs(n):
        if n in stack:
            cycle = " → ".join(stack[stack.index(n):] + [n])
            return cycle
        if n in seen:
            return None
        seen.add(n); stack.append(n)
        for d in graph.get(n, ()):
            r = dfs(d)
            if r:
                return r
        stack.pop(); return None
    for n in graph:
        r = dfs(n)
        if r:
            return "模块级导入环：%s" % r
    return None


def check_no_shim_duplication() -> str | None:
    """[14] 转发壳/重复符号回归守卫（S2 已清除的名字不得复活）。"""
    banned = ("_otsu_from_hist", "_otsu_median_threshold",
              "_preprocess_standard", "_text_sep_gray", "_gray_mean_abs_diff",
              "similar_binary")
    for p in (ROOT / "video_utils.py", ROOT / "video_ocr_engine" / "_helpers.py"):
        if not p.exists():
            continue
        t = io.open(p, encoding="utf-8").read()
        for b in banned:
            if re.search(r"def %s\b" % b, t):
                return "%s 重新定义了已清除的转发壳 %s" % (p.name, b)
    return None


def check_dead_code_regression() -> str | None:
    """[15] 已删死代码不得复活（TrtEngine.execute*/四个键集常量）。"""
    trt = ROOT / "ocr_trt.py"
    if trt.exists():
        t = io.open(trt, encoding="utf-8").read()
        for name in ("def execute(", "def execute_device("):
            if name in t:
                return "ocr_trt 复活了零调用方法 %s" % name[4:-1]
    ec = ROOT / "engine_config.py"
    if ec.exists():
        t = io.open(ec, encoding="utf-8").read()
        for name in ("DECODE_BACKEND_KEYS", "DECODE_BACKEND_LABELS",
                     "OCR_BACKEND_KEYS", "OCR_BACKEND_LABELS"):
            if re.search(r"^%s\s*[:=]" % name, t, re.M):
                return "engine_config 复活了零引用常量 %s" % name
    return None


def check_knob_coverage() -> str | None:
    """[16] 每个旋钮有依据锚点；注册表不超上限。"""
    from video_ocr_engine.config import KNOBS
    from video_ocr_engine.config.registry import _REGISTRY_CAP
    if len(KNOBS.knobs) > _REGISTRY_CAP:
        return "旋钮注册表 %d 超上限 %d" % (len(KNOBS.knobs), _REGISTRY_CAP)
    for k in KNOBS.knobs:
        if not k.rationale_id:
            return "旋钮 %s 缺依据锚点（rationale_id）" % k.name
    return None


def check_no_import_side_effects() -> str | None:
    """[17] 产品模块 import 期不得改写 os.environ（D7）。"""
    for p in _product_files():
        try:
            tree = ast.parse(io.open(p, encoding="utf-8").read())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            # 只看模块顶层（col_offset==0）的赋值/调用
            if getattr(n, "col_offset", 1) != 0:
                continue
            src = ast.unparse(n)
            if "environ[" in src or "environ.setdefault" in src or \
                    "putenv" in src:
                if "add_dll_directory" in src or "getenv" in src:
                    continue  # 读取与 DLL 注册（gpu_setup 有 _gpu_initialized 守卫的调用期路径）
                return "%s import 期改写进程环境：%s" % (
                    p.name, src[:60])
    return None


def check_docs_budget() -> str | None:
    """[18] 知识库与活文档预算（§14.2）。"""
    limits = (("knowledge/conclusions.yaml", 8 * 1024),
              ("knowledge/knobs.yaml", 24 * 1024),
              ("AGENTS.md", 12 * 1024))
    for rel, cap in limits:
        p = ROOT / rel
        if p.exists() and p.stat().st_size > cap:
            return "%s %dB 超 %dB 预算" % (rel, p.stat().st_size, cap)
    total = sum(p.stat().st_size for p in (ROOT / "docs").glob("*.md"))
    if total > 60 * 1024:
        return "docs/*.md（历史除外）合计 %dB 超 60KB" % total
    return None


def check_metrics_coverage() -> str | None:
    """[19] 指标注册表：上限 + v1 现役键不丢。"""
    from video_ocr_engine.domain.metrics import METRICS, METRIC_CAP
    if len(METRICS.names()) > METRIC_CAP:
        return "指标注册表超 %d 上限" % METRIC_CAP
    for name in ("pipeline.decode", "pipeline.ocr", "pipeline.ocr_tail",
                 "pipeline.producer", "pipeline.q_get_wait"):
        if name not in METRICS:
            return "v1 现役指标 %s 从注册表丢失" % name
    return None


def check_rules_ladder() -> str | None:
    """[20] 注入核规则：每条有 mechanism；prose 须附 why。"""
    p = ROOT / "knowledge" / "rules.yaml"
    if not p.exists():
        return "knowledge/rules.yaml 缺失"
    t = io.open(p, encoding="utf-8").read()
    n_rule = len(re.findall(r"^- rule:", t, re.M))
    n_mech = len(re.findall(r"^  mechanism:", t, re.M))
    n_why = len(re.findall(r"^  why:", t, re.M))
    n_prose = len(re.findall(r"^  mechanism: prose", t, re.M))
    if n_rule == 0:
        return "rules.yaml 无规则"
    if n_mech != n_rule:
        return "rules.yaml 有 %d 条规则缺 mechanism（%d/%d）" % (n_rule - n_mech, n_mech, n_rule)
    if n_why != n_prose:
        return "prose 类规则缺 why（%d prose / %d why）" % (n_prose, n_why)
    return None


def check_baseline_liveness() -> str | None:
    """[21] 豁免账本活性：baseline 里的文件必须仍存在（幽灵豁免回收）。"""
    p = ROOT / "tools" / "_discipline_baseline.json"
    if not p.exists():
        return None
    import json
    data = json.loads(io.open(p, encoding="utf-8").read())
    ghosts = []
    for cat, files in data.items():
        if isinstance(files, dict):
            ghosts += ["%s:%s" % (cat, k) for k in files
                       if not (ROOT / k).exists()]
    if ghosts:
        return "baseline 幽灵豁免（文件已删）：%s" % ", ".join(ghosts[:3])
    return None


CHECKS_EXT = {
    13: ("layering 分层无环", check_layering),
    14: ("no_shim_duplication 转发壳", check_no_shim_duplication),
    15: ("dead_code 死代码回归", check_dead_code_regression),
    16: ("knob_coverage 旋钮依据", check_knob_coverage),
    17: ("no_import_side_effects", check_no_import_side_effects),
    18: ("docs_budget 文档预算", check_docs_budget),
    19: ("metrics_coverage 指标注册表", check_metrics_coverage),
    20: ("rules_ladder 规则执行梯", check_rules_ladder),
    21: ("baseline_liveness 豁免活性", check_baseline_liveness),
}

if __name__ == "__main__":
    bad = 0
    for k, (name, fn) in sorted(CHECKS_EXT.items()):
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            r = "检查本身异常 %r" % (e,)
        tag = "✓" if r is None else "✗"
        print("%s [%2d] %s%s" % (tag, k, name, "" if r is None else " —— " + r))
        bad += r is not None
    sys.exit(1 if bad else 0)
