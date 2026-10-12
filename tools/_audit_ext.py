"""结构化审计扩展（按名注册，Phase C 制度重构后独立入口）。

每项失败即违规。旧 _probe_discipline_audit（26 项合流入口）随旧制度退役，
本文件保留下列仍有效的检查并改接新事实源（registry/）：
  layering / no_shim_duplication / dead_code / knob_coverage /
  no_import_side_effects / metrics_coverage / rules_schema /
  probe_hooks_resolve / render_consistency / decord_pin
消亡项（目标已删或被新守卫取代）：docs_budget（预算制废）、baseline_liveness
（豁免账本废）、no_deprecated_shim_usage（shims 已物理删除）。
引用检查项一律按名——编号会重排，名字不会。
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def _product_files():
    pkg = ROOT / "video_ocr_engine"
    return [p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts]


def check_layering() -> str | None:
    """[14] 模块级导入图无环（P0-1b：v1 本就是 DAG，测试守住）。"""
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
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module in names:
                deps.add(n.module)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name in names:
                        deps.add(a.name)
        graph[name] = deps
    seen, stack = set(), []

    def dfs(n):
        if n in stack:
            cycle = " → ".join(stack[stack.index(n) :] + [n])
            return cycle
        if n in seen:
            return None
        seen.add(n)
        stack.append(n)
        for d in graph.get(n, ()):
            r = dfs(d)
            if r:
                return r
        stack.pop()
        return None

    for n in graph:
        r = dfs(n)
        if r:
            return "模块级导入环：%s" % r
    return None


def check_no_shim_duplication() -> str | None:
    """[15] 转发壳/重复符号回归守卫（S2 已清除的名字不得复活）。"""
    banned = (
        "_otsu_from_hist",
        "_otsu_median_threshold",
        "_preprocess_standard",
        "_text_sep_gray",
        "_gray_mean_abs_diff",
        "similar_binary",
    )
    for p in (ROOT / "video_utils.py", ROOT / "video_ocr_engine" / "_helpers.py"):
        if not p.exists():
            continue
        t = io.open(p, encoding="utf-8").read()
        for b in banned:
            if re.search(r"def %s\b" % b, t):
                return "%s 重新定义了已清除的转发壳 %s" % (p.name, b)
    return None


def check_dead_code_regression() -> str | None:
    """[16] 已删死代码不得复活（TrtEngine.execute*/四个键集常量）。"""
    trt = ROOT / "ocr_trt.py"
    if trt.exists():
        t = io.open(trt, encoding="utf-8").read()
        for name in ("def execute(", "def execute_device("):
            if name in t:
                return "ocr_trt 复活了零调用方法 %s" % name[4:-1]
    ec = ROOT / "engine_config.py"
    if ec.exists():
        t = io.open(ec, encoding="utf-8").read()
        for name in (
            "DECODE_BACKEND_KEYS",
            "DECODE_BACKEND_LABELS",
            "OCR_BACKEND_KEYS",
            "OCR_BACKEND_LABELS",
        ):
            if re.search(r"^%s\s*[:=]" % name, t, re.M):
                return "engine_config 复活了零引用常量 %s" % name
    return None


def check_knob_coverage() -> str | None:
    """[17] 每个旋钮有依据锚点；注册表不超上限。"""
    from video_ocr_engine.config import KNOBS
    from video_ocr_engine.config.registry import _REGISTRY_CAP

    if len(KNOBS.knobs) > _REGISTRY_CAP:
        return "旋钮注册表 %d 超上限 %d" % (len(KNOBS.knobs), _REGISTRY_CAP)
    for k in KNOBS.knobs:
        if not k.rationale_id:
            return "旋钮 %s 缺依据锚点（rationale_id）" % k.name
    return None


def check_no_import_side_effects() -> str | None:
    """[18] 产品模块 import 期不得改写 os.environ（D7）。"""
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
            if "environ[" in src or "environ.setdefault" in src or "putenv" in src:
                if "add_dll_directory" in src or "getenv" in src:
                    continue  # 读取与 DLL 注册（gpu_setup 有 _gpu_initialized 守卫的调用期路径）
                return "%s import 期改写进程环境：%s" % (p.name, src[:60])
    return None


def check_metrics_coverage() -> str | None:
    """[20] 指标注册表：上限 + v1 现役键不丢。

    v1 清单更新（2026-09-17）：pipeline.producer 已出清（全仓零产出点，
    死键只稀释报告），以同代现役键 pipeline.consumer 顶替守卫位。
    """
    from video_ocr_engine.domain.metrics import METRICS, METRIC_CAP

    if len(METRICS.names()) > METRIC_CAP:
        return "指标注册表超 %d 上限" % METRIC_CAP
    for name in (
        "pipeline.decode",
        "pipeline.ocr",
        "pipeline.ocr_tail",
        "pipeline.consumer",
        "pipeline.q_get_wait",
    ):
        if name not in METRICS:
            return "v1 现役指标 %s 从注册表丢失" % name
    return None


def check_rules_schema() -> str | None:
    """规则注册表 schema-lite（R-registry-schema 的先行版；全量版 Phase D）。

    enforcement ∈ {ci, local-gate, advisory}；非 advisory 须有 executor；
    advisory ≤10（宪章不变量）；每条必有 prevents（举证义务）。
    """
    import yaml

    p = ROOT / "registry" / "rules.yaml"
    if not p.exists():
        return "registry/rules.yaml 缺失"
    entries = yaml.safe_load(io.open(p, encoding="utf-8").read()) or []
    rules = [e for e in entries if isinstance(e, dict) and "id" in e]
    if not rules:
        return "rules.yaml 无规则条目"
    n_adv = 0
    for e in rules:
        if e.get("enforcement") not in ("ci", "local-gate", "advisory"):
            return "%s enforcement 非法：%r" % (e["id"], e.get("enforcement"))
        if e.get("enforcement") != "advisory" and not e.get("executor"):
            return "%s 缺 executor（无执行体的不是规则，见宪章）" % e["id"]
        if not e.get("prevents"):
            return "%s 缺 prevents（举证义务）" % e["id"]
        if e.get("enforcement") == "advisory":
            n_adv += 1
    if n_adv > 10:
        return "advisory %d 条超 10 席上限（自动化或删除）" % n_adv
    return None


def check_probe_hooks_resolve() -> str | None:
    """[24] 探针打桩点表全部可解析（L1）。

    探针靠 monkeypatch 私有符号取数，重构会让点位失效。L1 把全部点位收敛到
    `tools/_probe_hooks.py:POINTS` —— 于是"哪些点位坏了"只需在本项报出来，
    而不是靠逐个探针跑来发现（2026-09-13 实测：S9 模块化一次打破 4 个探针）。
    """
    import sys as _sys

    _sys.path.insert(0, str(HERE))
    try:
        from _probe_hooks import hooks
    except Exception as e:  # noqa: BLE001
        return "tools/_probe_hooks.py 不可导入：%r" % (e,)
    bad = hooks.audit()
    if bad:
        return "打桩点失效 %d 项（改 POINTS 表即可，探针不必动）：%s" % (
            len(bad),
            "；".join(bad[:2]),
        )
    return None


def check_render_consistency() -> str | None:
    """[25] 渲染产物一致性 + 结论路径活性（折入统一审计入口，2026-10-10）。

    此前 render --check 是独立第三条命令，容易被忘（纪律段只列了两个
    审计入口）；折进来后全量审计一次跑齐。render 顺带校验结论字段里的
    仓库内全路径存在性（C-32 教训：包化后旧路径标签静默存活两周+）。
    """
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "render.py"), "--check"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=180,
    )
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = [ln for ln in out.splitlines() if ln.strip()][-1:] or ["（无输出）"]
        return "render --check 未过（重渲染：python tools/render.py --write）：%s" % tail[0][:110]
    return None


def check_decord_pin() -> str | None:
    """两仓版本口径：registry/environment.yaml 钉版须含实际安装版本。

    2026-10-10 治理改革轮新增（当时对账 DEPENDENCIES.md 的 decord 行，
    该行曾写 0.8.5 而本机已装 0.9.2）；Phase C 制度重构后事实源迁
    registry/environment.yaml。CI 不装 decord：import 失败按环境事实
    跳过（"装了但不一致"才是缺陷）。发版时的全量清扫见 conclusions_sweep。
    """
    try:
        import decord  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — 环境事实
        return None
    ver = decord.__version__
    import yaml

    env = (
        yaml.safe_load(io.open(ROOT / "registry" / "environment.yaml", encoding="utf-8").read())
        or {}
    )
    pin = str((env.get("dependencies") or {}).get("decord_pin", ""))
    if not pin:
        return "registry/environment.yaml 缺 dependencies.decord_pin"
    if ver != pin:
        return (
            "environment.yaml 钉版 %s ≠ 本机安装 %s——fork 升级当轮须同步"
            "钉版（发版清扫：tools/conclusions_sweep.py --strict）" % (pin, ver)
        )
    return None


CHECKS_EXT = {
    "dead_code_regression": ("dead_code 死代码回归", check_dead_code_regression),
    "decord_pin": ("decord_pin 两仓版本口径", check_decord_pin),
    "knob_coverage": ("knob_coverage 旋钮依据", check_knob_coverage),
    "layering": ("layering 分层无环", check_layering),
    "metrics_coverage": ("metrics_coverage 指标注册表", check_metrics_coverage),
    "no_import_side_effects": ("no_import_side_effects", check_no_import_side_effects),
    "no_shim_duplication": ("no_shim_duplication 转发壳", check_no_shim_duplication),
    "probe_hooks_resolve": ("probe_hooks_resolve 打桩点表", check_probe_hooks_resolve),
    "render_consistency": ("render_consistency 渲染一致", check_render_consistency),
    "rules_schema": ("rules_schema 规则注册表", check_rules_schema),
}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    bad = 0
    for k, (name, fn) in sorted(CHECKS_EXT.items()):
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            r = "检查本身异常 %r" % (e,)
        tag = "✓" if r is None else "✗"
        print("%s %s %s%s" % (tag, name, "", "" if r is None else " —— " + r))
        bad += r is not None
    sys.exit(1 if bad else 0)
