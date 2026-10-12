"""统一渲染器——registry/ 与产品配置到全部视图的唯一通道（R-render-consistency）。

视图（永不手写；--check 校验一致性 + 裸 CR + 结论路径活性 + 证据锚点活性）：
  docs/CONCLUSIONS.md   ← registry/conclusions.yaml
  docs/KNOBS.md         ← video_ocr_engine.config.KNOBS（旋钮事实源=产品代码）
  registry/knobs.yaml   ← 同上（机器可读镜像）
  tools/INDEX.md        ← registry/tools.yaml（standing 全列 + 在树 probe）

旧 knowledge/render.py 的职责并入本文件（Phase C，制度重构）；conclusions 的
evidence 为 commit 锚点结构，路径活性在锚点提交上验证（git cat-file），与
"档案归 git"语义一致——树里不存在的 docs/log 路径不算悬空。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
VIEWS = {
    "conclusions": ROOT / "docs" / "CONCLUSIONS.md",
    "knobs_md": ROOT / "docs" / "KNOBS.md",
    "knobs_yaml": REGISTRY / "knobs.yaml",
    "index": ROOT / "tools" / "INDEX.md",
}
_PATH_TOKEN = re.compile(
    r"((?:video_ocr_engine|docs|tools|tests|registry)/[\w./-]+\.(?:py|md|yaml))"
)


def _load(path: Path) -> list:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def _cell(text: str) -> str:
    """markdown 表格单元格：管道转义。"""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _evidence_cell(items: list) -> str:
    parts = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        if "commit" in it and "path" in it:
            parts.append(
                "%s:%s%s"
                % (it["commit"], it["path"], ("#" + str(it["anchor"])) if it.get("anchor") else "")
            )
        elif "commit" in it and "note" in it:
            parts.append("%s（%s）" % (it["note"], it["commit"]))
        elif "commit" in it:
            parts.append(str(it["commit"]))
        elif "path" in it:
            parts.append(str(it["path"]))
        elif "note" in it:
            parts.append(str(it["note"]))
    return "；".join(parts)


def render_conclusions_md(entries: list) -> str:
    lines = [
        "# 现役结论索引（渲染产物；事实源 registry/conclusions.yaml）",
        "",
        "> 本文件由 tools/render.py 渲染，人不得手写（--check 校验）。",
        "> superseded/dead 全文在 git 档案（archive/pre-rework 锚点，",
        "> `git show <commit>:docs/log/2026-09-10-conclusions-history.md` 取回）。",
        "",
        "| ID | 结论 | 前提/边界 | 复评触发 | 证据（commit 锚点） |",
        "|----|------|-----------|----------|------|",
    ]
    for e in entries:
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                e.get("id", ""),
                _cell(e.get("conclusion", "")),
                _cell(e.get("premises", "")),
                _cell(e.get("revisit", "")),
                _cell(_evidence_cell(e.get("evidence", []))),
            )
        )
    lines += ["", "（复评/清扫：python tools/conclusions_sweep.py --strict，发版必跑。）", ""]
    return "\n".join(lines)


def _knobs_registry():
    sys.path.insert(0, str(ROOT))
    from video_ocr_engine.config import KNOBS

    return KNOBS


def render_knobs_md() -> str:
    lines = [
        "# 旋钮注册表（渲染产物：config/knobs.py → 本表）",
        "",
        "| 旋钮 | env | 类型 | 默认 | 依据锚点 |",
        "|---|---|---|---|---|",
    ]
    for k in _knobs_registry().knobs:
        lines.append(
            "| `%s` | `%s` | %s | `%s` | %s |" % (k.name, k.env, k.type, k.default, k.rationale_id)
        )
    lines += ["", "（语义备注与解析细节见 config/knobs.py 各 Knob 的 note 字段。）", ""]
    return "\n".join(lines)


def render_knobs_yaml() -> str:
    lines = ["# 旋钮注册表镜像（渲染产物；事实源 config/knobs.py）", ""]
    for k in _knobs_registry().knobs:
        lines += [
            "- name: %s" % k.name,
            "  env: %s" % k.env,
            "  type: %s" % k.type,
            "  default: %r" % (k.default,),
            "  rationale: %s" % k.rationale_id,
            "  note: %s" % k.note.replace("\n", " "),
            "",
        ]
    return "\n".join(lines)


def render_index_md(tools: list) -> str:
    standing = [t for t in tools if t.get("tier") == "standing"]
    probes = [t for t in tools if t.get("tier") == "probe"]
    lines = [
        "# tools/ 索引（渲染产物；事实源 registry/tools.yaml）",
        "",
        "> 分级判据与生命周期见 registry/tools.yaml 头注（裁决 18）。",
        "> 历史探针不在树（git 档案，archive/pre-rework 锚点起可溯）。",
        "",
        "## 常备工具（standing，质量面=生产代码）",
        "",
        "| 文件 | 用途 |",
        "|---|---|",
    ]
    for t in sorted(standing, key=lambda x: x["id"]):
        lines.append("| `%s` | %s |" % (t["id"], _cell(t.get("why", ""))))
    lines += ["", "## 在树探针（probe，近两轮；超出即出树）", "", "| 文件 | 说明 |", "|---|---|"]
    for t in sorted(probes, key=lambda x: x["id"]):
        note = "%s（last %s）" % (t.get("why", ""), t.get("last_touched", "?"))
        lines.append("| `tools/probes/%s` | %s |" % (t["id"], _cell(note)))
    lines += [
        "",
        "（探针删除协议：结论数字先进 benchmarks.yaml → git rm；数字不丢，档案可溯。）",
        "",
    ]
    return "\n".join(lines)


def _anchor_exists(commit: str, path: str) -> bool:
    r = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}:{path}"], capture_output=True
    )
    return r.returncode == 0


def path_liveness(entries: list) -> list[str]:
    """结论字段里的仓库内全路径活性：裸 path 须在树；commit 锚点须在锚点提交可取。"""
    bad: list[str] = []
    for e in entries:
        eid = e.get("id", "?")
        for field in ("conclusion", "premises", "revisit"):
            for m in _PATH_TOKEN.finditer(str(e.get(field, ""))):
                rel = m.group(1)
                if "*" not in rel and not (ROOT / rel).exists():
                    bad.append("%s(%s): %s 不在树" % (eid, field, rel))
        for it in e.get("evidence", []) if isinstance(e.get("evidence"), list) else []:
            if not isinstance(it, dict):
                continue
            if "commit" in it and "path" in it and not _anchor_exists(it["commit"], it["path"]):
                bad.append("%s(evidence): %s:%s 锚点取不到" % (eid, it["commit"], it["path"]))
            if "path" in it and "commit" not in it and not (ROOT / it["path"]).exists():
                bad.append("%s(evidence): %s 不在树" % (eid, it["path"]))
    return bad


def bare_cr(rel: str) -> list[str]:
    """视图裸 CR=0（\\r 不跟 \\n）——编码事故防线（旧 _doc_guards 职能迁此）。"""
    path = ROOT / rel
    if not path.is_file():
        return [f"{rel}: 视图缺失"]
    n = len(re.findall(rb"\r(?!\n)", path.read_bytes()))
    return [] if n == 0 else [f"{rel}: {n} 处裸 CR"]


def generate() -> dict[str, str]:
    return {
        "conclusions": render_conclusions_md(_load(REGISTRY / "conclusions.yaml")),
        "knobs_md": render_knobs_md(),
        "knobs_yaml": render_knobs_yaml(),
        "index": render_index_md(_load(REGISTRY / "tools.yaml")),
    }


def check() -> list[str]:
    errors: list[str] = []
    want = generate()
    for key, path in VIEWS.items():
        errors.extend(bare_cr(path.relative_to(ROOT).as_posix()))
        if not path.is_file():
            errors.append(f"{path.relative_to(ROOT)}: 视图缺失（--write 生成）")
            continue
        if want[key].strip() != path.read_text(encoding="utf-8").strip():
            errors.append(f"{path.relative_to(ROOT)}: 与源不一致（--write 重渲染）")
    errors.extend(path_liveness(_load(REGISTRY / "conclusions.yaml")))
    return errors


def write() -> int:
    for key, path in VIEWS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generate()[key] + "\n", encoding="utf-8", newline="\n")
    print("rendered: " + ", ".join(str(p.relative_to(ROOT)) for p in VIEWS.values()))
    return 0


def _utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass  # 非 TextIOWrapper（重定向）时无需重配


if __name__ == "__main__":
    _utf8_stdout()
    errs = check()
    if "--write" in sys.argv:
        raise SystemExit(write())
    if errs:
        print("render check：%d 处违规" % len(errs))
        for e in errs:
            print("  " + e)
        raise SystemExit(1)
    print("render check：四视图一致 + 路径/锚点活性 + 裸 CR ✓")
