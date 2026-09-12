"""金标漂移的定性核对：**段数变了，但唯一文本集是否保持/改善**。

`record.py --verify` 只比哈希，对「引擎分段语义正当变更」这类漂移给不出
"对不对"的判据。铁律 2 的正确性门禁 = 段数 + **唯一文本集**，所以重录
之前必须先量这一项：新向量若丢文本（新集 ⊄ 旧集 ∪ 新增合理值）就是真
退化，不得重录。

用法：python tools/_probe_golden_drift.py            # 全 28 用例
      python tools/_probe_golden_drift.py --case A-  # 子串过滤
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "golden"))
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="")
    args = ap.parse_args()

    import record                                    # noqa: E402

    rows = []
    lost_total = gained_total = 0
    for case in record.MATRIX:
        cid = case["id"]
        if args.case and args.case not in cid:
            continue
        old = json.loads(
            (ROOT / "tests" / "golden" / ("case-" + cid) /
             "stage-ocr.json").read_text(encoding="utf-8"))
        old_segs = old["segments"]
        old_texts = {s[3] for s in old_segs if s[3]}

        _calib, new = record.summarize(case)          # 复用录制口径（vid→路径映射同源）
        new_segs = new["segments"]
        new_texts = {s[3] for s in new_segs if s[3]}

        lost = old_texts - new_texts
        gained = new_texts - old_texts
        lost_total += len(lost)
        gained_total += len(gained)
        rows.append((cid, len(old_segs), len(new_segs),
                     len(old_texts), len(new_texts), sorted(lost)[:4],
                     sorted(gained)[:4]))
        print("%-22s 段 %5d→%5d  唯一文本 %4d→%4d  丢 %2d 增 %2d %s" % (
            cid, len(old_segs), len(new_segs), len(old_texts),
            len(new_texts), len(lost), len(gained),
            ("  丢:%s" % sorted(lost)[:4]) if lost else ""))
    print("合计：丢 %d / 增 %d" % (lost_total, gained_total))
    return 0 if lost_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
