"""PreToolUse(Read) 守卫 —— 拦下对大文档的无 offset 整读（N2，2026-10-10）。

背景：误整读一次 PERFORMANCE.md ≈ 30k tokens（≈12 倍注入成本）。纪律
写在 AGENTS.md 里靠自觉守不住，本脚本由 ZCode PreToolUse hook 调用
（.zcode/config.json），把"整读大文档"变成一次带指引的软拦截。

协议（ZCode hook，见 diagnosing-hooks 技能/schema）：
  · stdin 收 JSON：{"tool_name": "Read", "tool_input": {file_path, offset?, limit?}}；
  · exit 0 + 空输出 = 放行；exit 2 = 拦截，stderr 作为理由回给 agent。

判据（三个条件同时满足才拦，宁纵勿枉——本脚本自身故障一律放行）：
  1. 目标是本仓库内的 .md（ROOT 由本文件位置推导，父目录会话不受影响）；
  2. 文件 > 16 KB（阈值下的小文档整读无妨；knowledge/conclusions.md 是
     L1 索引，本就设计为整读，显式豁免）；
  3. Read 未带 offset/limit（带了 = 已经在分段读，放行）。

拦下后给三条出路：_doc_section.py --toc/--find/单章，或带 offset 重读。
手工运行（stdin 是 tty）直接自检退出，绝不挂起。
"""
from __future__ import annotations

import json
import os
import sys

try:
    if sys.stdin.isatty():
        print("（手工自检：hook 正常。生产调用由 ZCode 经 stdin 传 JSON。）")
        raise SystemExit(0)
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input") or {}
    raw_path = str(tool_input.get("file_path") or "")
    has_range = tool_input.get("offset") is not None or tool_input.get("limit") is not None
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    norm = os.path.normpath(raw_path)
    try:
        in_repo = bool(norm) and os.path.commonpath([ROOT, norm]) == ROOT
    except ValueError:                             # 不同盘符等：非本仓库文件
        in_repo = False
    exempt = norm.endswith(os.path.join("knowledge", "conclusions.md"))

    size = os.path.getsize(norm) if (in_repo and norm.endswith(".md") and os.path.isfile(norm)) else 0
    if in_repo and not exempt and size > 16 * 1024 and not has_range:
        rel = os.path.relpath(norm, ROOT).replace("\\", "/")
        sys.stderr.write(
            "⛔ 整读拦截：%s（%s B）——大文档整读一次≈数万 tokens。\n"
            "  改用其一：\n"
            "  1) python tools/_doc_section.py --toc %s        # 目录+token 数\n"
            "  2) python tools/_doc_section.py --find <关键词>  # 跨文档定位\n"
            "  3) python tools/_doc_section.py %s <章节号>      # 只读该章\n"
            "  4) 确需原文：Read 带 offset/limit 分段读（本守卫放行）。\n"
            % (rel, "{:,}".format(size), rel, rel))
        raise SystemExit(2)
except SystemExit:
    raise
except Exception:                                  # noqa: BLE001 — 守卫自身故障一律放行
    pass  # fail-open：守卫脚本绝不能因为自己的 bug 卡住会话（exit 0 放行）
raise SystemExit(0)
