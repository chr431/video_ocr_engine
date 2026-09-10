"""一次性转换器：CONCLUSIONS.md 表 → knowledge/conclusions.yaml（S2-文档-2）。

active/superseded 留在 yaml（superseded 只留指针，Q7 R1）；dead 整条降级
docs/log/2026-09-10-conclusions-history.md（保留复评触发）。转换后
CONCLUSIONS.md 由 render.py 渲染接管。
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_table(md: str):
    rows = []
    for line in md.split("\n"):
        if not line.startswith("| C-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6:
            rows.append(dict(zip(
                ("id", "conclusion", "status_raw", "premises", "revisit", "evidence"), cells)))
    return rows


def norm_status(raw: str) -> str:
    if raw.startswith("active"):
        return "active"
    if raw.startswith("superseded"):
        m = re.search(r"superseded\((C-\d+)\)", raw)
        return "superseded:" + (m.group(1) if m else "?")
    return "dead"


def main() -> int:
    src = (ROOT / "docs" / "CONCLUSIONS.md").read_text(encoding="utf-8")
    rows = parse_table(src)
    assert len(rows) == 33, len(rows)
    y_lines = ["# 活动结论（唯一事实源；docs/CONCLUSIONS.md 为渲染产物）",
               "# 规则（Q7 裁决）：每条含 status/premises/revisit/evidence；",
               "# superseded 只留指针；dead 整体降级 docs/log/（保留复评触发）。",
               ""]
    h_lines = ["# 结论历史（dead/superseded 全文，自 CONCLUSIONS.md 降级；",
               "# append-only，永不注入，显式检索可达——Q7 R3）", ""]
    n_act = n_sup = n_dead = 0
    for r in rows:
        st = norm_status(r["status_raw"])
        if st == "active":
            n_act += 1
            y_lines += ["- id: %s" % r["id"],
                        "  conclusion: %s" % r["conclusion"].replace("|", "\\|"),
                        "  status: active",
                        "  premises: %s" % r["premises"].replace("|", "\\|"),
                        "  revisit: %s" % r["revisit"].replace("|", "\\|"),
                        "  evidence: %s" % r["evidence"].replace("|", "\\|"),
                        ""]
        elif st.startswith("superseded"):
            n_sup += 1
            y_lines += ["- id: %s" % r["id"],
                        "  status: superseded",
                        "  replaced_by: %s" % st.split(":")[1],
                        ""]
            h_lines += ["## %s（superseded → %s）" % (r["id"], st.split(":")[1]),
                        "- 结论：%s" % r["conclusion"],
                        "- 前提：%s" % r["premises"],
                        "- 复评触发：%s" % r["revisit"],
                        "- 证据：%s" % r["evidence"], ""]
        else:
            n_dead += 1
            h_lines += ["## %s（dead）" % r["id"],
                        "- 结论：%s" % r["conclusion"],
                        "- 前提：%s" % r["premises"],
                        "- 复评触发：%s" % r["revisit"],
                        "- 证据：%s" % r["evidence"], ""]
    (ROOT / "knowledge" / "conclusions.yaml").write_text(
        "\n".join(y_lines), encoding="utf-8", newline="\n")
    (ROOT / "docs" / "log" / "2026-09-10-conclusions-history.md").write_text(
        "\n".join(h_lines), encoding="utf-8", newline="\n")
    print("active %d / superseded %d / dead %d" % (n_act, n_sup, n_dead))
    return 0


if __name__ == "__main__":
    sys.exit(main())
