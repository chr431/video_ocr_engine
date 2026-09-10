"""金标 manifest 完整性（无 GPU 可跑，CI 门禁）。

manifest.yaml 记录的期望哈希必须与盘上 case 文件逐字一致——防"改了向量
忘了 manifest"或反之。录制/复核的真实对账在 `record.py --verify`（须本机
GPU），本测试只锁"清单与文件不漂移"。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent


def _parse_manifest():
    text = (GOLDEN / "manifest.yaml").read_text(encoding="utf-8")
    cases, cur = [], None
    for line in text.splitlines():
        m = re.match(r"\s+- id: (\S+)", line)
        if m:
            cur = {"id": m.group(1)}
            cases.append(cur)
            continue
        if cur is not None:
            m = re.match(r"\s+sha: \[([0-9a-f]+), ([0-9a-f]+)\]", line)
            if m:
                cur["sha"] = (m.group(1), m.group(2))
    return cases


def test_manifest_exists_and_complete():
    cases = _parse_manifest()
    assert len(cases) == 28, "S0 基线应为 28 用例，实际 %d" % len(cases)
    assert all("sha" in c for c in cases)


def test_case_files_match_manifest():
    for c in _parse_manifest():
        d = GOLDEN / ("case-%s" % c["id"])
        for fname, want in zip(("stage-calib.json", "stage-ocr.json"), c["sha"]):
            got = hashlib.sha256((d / fname).read_bytes()).hexdigest()[:16]
            assert got == want, "%s/%s 哈希漂移（manifest=%s 盘上=%s）——重录须走 record.py" % (
                c["id"], fname, want, got)
