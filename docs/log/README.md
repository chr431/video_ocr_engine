# docs/log/ —— 实验叙事目录（L3，冷）

自 2026-09-08 起，**新一轮实验的过程叙事写在这里**（一轮一个文件，
命名 `YYYY-MM-DD-主题.md`），`docs/log/PERFORMANCE.md` 冻结增长（只修错、
不追加新章节）。

规则（由 `tests/test_docs_hygiene.py` 部分守护）：

1. 本目录**只放叙事与中间数据**（怎么测的、中间翻了什么案、原始数字）。
2. **结论一行进 `docs/CONCLUSIONS.md`**（带状态/前提/复评触发），本文不得
   复述结论。
3. 本目录**禁止规范性语句**（勿 / 必须 / 不要再用等禁令词与状态断言）——
   禁令与状态只住 `docs/CONCLUSIONS.md`，写在叙事里没有效力。
4. 永不注入、**勿整读**；用 `python tools/_doc_section.py --toc <文件>` 定位。
5. 证据文件（探针 log/CSV）放 `tools/`，本目录只放分析文字与指针。
