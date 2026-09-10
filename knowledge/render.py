"""knowledge/ 结构化知识库渲染器（S2-文档-2 首版，v2 §8.1/§14.2）。

yaml 是唯一事实源；docs/CONCLUSIONS.md / docs/KNOBS.md 为渲染产物
（render.py --check 校验一致性，防手写漂移——v1 的 INDEX.md 之病）。
预算（tests/knowledge/test_budgets.py 守护）：conclusions ≤ 8 KB、
knobs ≤ 24 KB；AGENTS.md ≤ 12 KB。
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOW = ROOT / "knowledge"


def load_conclusions() -> list[dict]:
    """解析 conclusions.yaml（手写极简解析：每条以 '- id:' 起）。"""
    text = (KNOW / "conclusions.yaml").read_text(encoding="utf-8")
    entries, cur = [], None
    for line in text.split("\n"):
        m = re.match(r"- id: (\S+)", line)
        if m:
            cur = {"id": m.group(1)}
            entries.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"  (conclusion|status|premises|revisit|evidence|replaced_by): (.*)", line)
        if m:
            cur[m.group(1)] = m.group(2).strip()
    return entries


def render_conclusions_md(entries: list[dict]) -> str:
    act = [e for e in entries if e.get("status") == "active"]
    sup = [e for e in entries if e.get("status") == "superseded"]
    lines = ["# 现役结论索引（L1，唯一规范性结论地）",
             "",
             "> 本文件由 knowledge/render.py 从 knowledge/conclusions.yaml 渲染",
             "> （人不得手写；--check 校验一致性）。状态取值：active / superseded",
             "> （被取代，只留指针）/ dead（已降级 docs/log 历史）。规则与完整",
             "> 说明见 yaml 头部注释。",
             "",
             "| ID | 结论 | 前提/边界 | 复评触发 | 证据 |",
             "|----|------|-----------|----------|------|"]
    for e in act:
        lines.append("| %s | %s | %s | %s | %s |" % (
            e["id"], e.get("conclusion", ""), e.get("premises", ""),
            e.get("revisit", ""), e.get("evidence", "")))
    if sup:
        lines += ["", "## 已取代（指针）", ""]
        for e in sup:
            lines.append("- %s → %s" % (e["id"], e.get("replaced_by", "?")))
    lines += ["", "（dead 条目已整体降级 docs/log/，含复评触发，显式检索可达。）", ""]
    return "\n".join(lines)


def _registry():
    import sys
    sys.path.insert(0, str(ROOT))
    from video_ocr_engine.config import KNOBS
    return KNOBS


def render_knobs_md() -> str:
    lines = ["# 旋钮注册表（渲染产物：config/knobs.py → 本表）",
             "", "| 旋钮 | env | 类型 | 默认 | 依据锚点 |", "|---|---|---|---|---|"]
    for k in _registry().knobs:
        lines.append("| `%s` | `%s` | %s | `%s` | %s |" % (
            k.name, k.env, k.type, k.default, k.rationale_id))
    lines += ["", "（语义备注与解析细节见 config/knobs.py 各 Knob 的 note 字段；"
              "v1 依据全文在 engine_config.py 注释，锚点 `ec:<行>`。）", ""]
    return "\n".join(lines)


def render_knobs_yaml() -> str:
    lines = ["# 旋钮注册表镜像（渲染产物；事实源 config/knobs.py）", ""]
    for k in _registry().knobs:
        lines += ["- name: %s" % k.name,
                  "  env: %s" % k.env,
                  "  type: %s" % k.type,
                  "  default: %r" % (k.default,),
                  "  rationale: %s" % k.rationale_id,
                  "  note: %s" % k.note.replace("\n", " "),
                  ""]
    return "\n".join(lines)


def check() -> int:
    bad = 0
    entries = load_conclusions()
    want = render_conclusions_md(entries)
    have = (ROOT / "docs" / "CONCLUSIONS.md").read_text(encoding="utf-8")
    if want.strip() != have.strip():
        bad += 1
        print("✗ docs/CONCLUSIONS.md 与 conclusions.yaml 不一致（重渲染："
              "python knowledge/render.py --write）")
    else:
        print("✓ CONCLUSIONS.md 一致")
    for path, gen, label in (
            (ROOT / "docs" / "KNOBS.md", render_knobs_md, "KNOBS.md"),
            (KNOW / "knobs.yaml", render_knobs_yaml, "knobs.yaml")):
        if not path.exists() or gen().strip() != path.read_text(encoding="utf-8").strip():
            bad += 1
            print("✗ %s 不一致" % label)
        else:
            print("✓ %s 一致" % label)
    return bad


def write() -> int:
    entries = load_conclusions()
    (ROOT / "docs" / "CONCLUSIONS.md").write_text(
        render_conclusions_md(entries), encoding="utf-8", newline="\n")
    (ROOT / "docs" / "KNOBS.md").write_text(
        render_knobs_md(), encoding="utf-8", newline="\n")
    (KNOW / "knobs.yaml").write_text(
        render_knobs_yaml(), encoding="utf-8", newline="\n")
    print("rendered CONCLUSIONS.md / KNOBS.md / knobs.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(write() if "--write" in sys.argv else check())
