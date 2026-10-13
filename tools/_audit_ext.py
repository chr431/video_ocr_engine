"""结构化审计（按名注册；每项失败即违规，事实源 registry/）。

检查项：layering / no_shim_duplication / dead_code / knob_coverage /
no_import_side_effects / metrics_coverage / rules_schema /
probe_hooks_resolve / render_consistency / decord_pin /
deprecation_due / dependency_floors / gate_manifest / injection_budget。
未设检查：docs_budget / baseline_liveness / deprecated_shim_usage——
对应制度对象已不存在。引用检查项一律按名（编号会重排，名字不会）。
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
    """模块级导入图无环（测试守住）。"""

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
    seen: set[str] = set()
    stack: list[str] = []

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
    """转发壳/重复符号回归守卫（已清除的名字不得复活）。"""
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
    """已删死代码不得复活（TrtEngine.execute*/四个键集常量）。"""
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
    """每个旋钮有依据锚点；注册表不超上限。"""
    from video_ocr_engine.config import KNOBS
    from video_ocr_engine.config.registry import _REGISTRY_CAP

    if len(KNOBS.knobs) > _REGISTRY_CAP:
        return "旋钮注册表 %d 超上限 %d" % (len(KNOBS.knobs), _REGISTRY_CAP)
    for k in KNOBS.knobs:
        if not k.rationale_id:
            return "旋钮 %s 缺依据锚点（rationale_id）" % k.name
    return None


def check_no_import_side_effects() -> str | None:
    """产品模块 import 期不得改写 os.environ。"""
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
    """指标注册表：上限 + 关键现役键不丢。

    守卫位取现役键；无产出点的键（如 pipeline.producer）不作守卫位——
    死键只稀释报告。
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
    """规则注册表 schema-lite（R-registry-schema）。

    enforcement ∈ {ci, local-gate, advisory}；非 advisory 须有 executor；
    advisory ≤10（宪章不变量）；每条必有 prevents（举证义务）。
    """
    import yaml  # type: ignore[import-untyped]

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
    """探针打桩点表全部可解析。

    探针靠 monkeypatch 私有符号取数，重构会让点位失效。全部点位收敛到
    `tools/_probe_hooks.py:POINTS`——"哪些点位坏了"由本项报出，不必逐个
    探针跑才发现。
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
    """渲染产物一致性 + 结论路径活性。

    render --check 折入本审计一次跑齐（不必单独跑）；render 顺带校验结论
    字段里的仓库内全路径存在性（C-32）。
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

    CI 不装 decord：import 失败按环境事实跳过（"装了但不一致"才是缺陷）。
    发版时的全量清扫见 conclusions_sweep。
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


def check_deprecation_due() -> str | None:
    """弃用到期即红（R-deprecation-due）：remove_in ≤ 当前版本而符号仍活着。

    符号活性按 kind 判：env → config 旋钮注册表 env 名；api → import 链取属性。
    已移除的到期项自然通过。
    """
    import yaml

    deps = (
        yaml.safe_load(io.open(ROOT / "registry" / "deprecations.yaml", encoding="utf-8").read())
        or []
    )
    if not deps:
        return None
    from video_ocr_engine.config.constants import __version__ as ver

    def vt(s: str) -> tuple[int, ...]:
        return tuple(int(x) for x in s.split("."))

    for d in deps:
        m = re.search(r"(\d+\.\d+\.\d+)", str(d.get("remove_in", "")))
        if not m or vt(ver) < vt(m.group(1)):
            continue
        name = str(d.get("name", ""))
        alive = False
        if d.get("kind") == "env":
            from video_ocr_engine.config import KNOBS

            alive = any(k.env == name for k in KNOBS.knobs)
        else:
            mod, _, attr = name.rpartition(".")
            try:
                import importlib

                alive = bool(attr) and hasattr(importlib.import_module(mod), attr)
            except Exception:  # noqa: BLE001 — 模块不存在=已删
                alive = False
        if alive:
            return "弃用到期未删：%s（remove_in %s ≤ 当前 %s）" % (name, m.group(1), ver)
    return None


def check_gate_manifest() -> str | None:
    """门禁 manifest 时效：engine_version=当前版本、decord=钉版。

    manifest 由 tools/version_advice.py gate-manifest 跑金标后落盘并随仓提交；
    bump 后未重跑金标即在此报红（本机门禁的异地对账底线）。
    """
    import yaml

    p = ROOT / "artifacts" / "gates" / "golden.yaml"
    if not p.is_file():
        return "artifacts/gates/golden.yaml 缺失（python tools/version_advice.py gate-manifest）"
    m = yaml.safe_load(io.open(p, encoding="utf-8").read()) or {}
    from video_ocr_engine.config.constants import __version__ as ver

    if str(m.get("engine_version")) != ver:
        return "金标 manifest 版本滞后：%s ≠ 当前 %s（bump 后须重跑 gate-manifest）" % (
            m.get("engine_version"),
            ver,
        )
    try:
        import decord  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — CI 无 decord=环境事实
        return None
    env = (
        yaml.safe_load(io.open(ROOT / "registry" / "environment.yaml", encoding="utf-8").read())
        or {}
    )
    pin = str((env.get("dependencies") or {}).get("decord_pin", ""))
    if str(m.get("decord")) != decord.__version__ or decord.__version__ != pin:
        return "金标 manifest decord 口径不一致：manifest=%s 本机=%s 钉版=%s" % (
            m.get("decord"),
            decord.__version__,
            pin,
        )
    return None


def check_injection_budget() -> str | None:
    """注入核与宪章预算（宪章不变量 5）。

    AGENTS ≤6144B（6KB）、charter ≤2048B。触碰=制度失效警报：查什么内容
    泄漏进注入核，禁止上调数值（上调须走宪章程序并附失效复盘）。
    """
    charter = ROOT / "charter.md"
    agents = ROOT / "AGENTS.md"
    if charter.stat().st_size > 2048:
        return "charter.md %dB 超 2048（手写制度文本应极小）" % charter.stat().st_size
    if agents.stat().st_size > 6144:
        return (
            "AGENTS.md %dB 超 6144——失效警报：查注入内容泄漏（不上调数值）" % agents.stat().st_size
        )
    return None


def check_dependency_floors() -> str | None:
    """外部依赖能力 floors（R-dependency-floors）：钉版 ≥ 每个 floor。

    产品依赖的 fork 最低版本唯一住 registry/environment.yaml；注释不写
    "需 fork ≥x.y.z"（那会与钉版构成双事实源）。pin < floor=配置矛盾
    （钉版低于产品所需）→ 红；floor 高于 pin 也是矛盾（floor 声明了不
    存在的能力）→ 红。
    """
    import yaml

    env = (
        yaml.safe_load(io.open(ROOT / "registry" / "environment.yaml", encoding="utf-8").read())
        or {}
    )
    pin = str((env.get("dependencies") or {}).get("decord_pin", ""))
    floors = env.get("dependency_floors") or {}
    if not pin:
        return "registry/environment.yaml 缺 dependencies.decord_pin"
    if not floors:
        return "registry/environment.yaml 缺 dependency_floors（R-dependency-floors）"

    def vt(s: str) -> tuple[int, ...]:
        return tuple(int(x) for x in s.split("."))

    pin_v = vt(pin)
    for name, floor in sorted(floors.items()):
        fv = vt(str(floor))
        if fv > pin_v:
            return "floor %s=%s 高于钉版 %s（声明了钉版不具备的能力）" % (name, floor, pin)
    return None


CHECKS_EXT = {
    "dead_code_regression": ("dead_code 死代码回归", check_dead_code_regression),
    "decord_pin": ("decord_pin 两仓版本口径", check_decord_pin),
    "dependency_floors": ("dependency_floors 依赖能力 floors", check_dependency_floors),
    "deprecation_due": ("deprecation_due 弃用到期", check_deprecation_due),
    "gate_manifest": ("gate_manifest 门禁时效", check_gate_manifest),
    "injection_budget": ("injection_budget 注入预算", check_injection_budget),
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
    import io as _io

    if isinstance(sys.stdout, _io.TextIOWrapper):
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
