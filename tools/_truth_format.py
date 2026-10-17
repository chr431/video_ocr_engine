"""真值数据格式：本项目自有的原生格式（主）+ 与 RaceVideoToLog(RVTL) 的兼容 shim。

为什么要自有格式：RVTL 的 csv 头是"给人看的日志行"——逗号分隔的 key=value、
roi 内联四个数、字段缺失合法。这类格式对机器不友好，且**失败是静默的**：实测把
头部改成空格分隔后，既有消费方读不到 frame_start，于是拿错的帧窗口去对齐，
表现为"匹配率 5.8%"的假失败——真值没错、代码没错，错在格式耦合。故：

- **原生格式 truth.v1（主，本模块的默认）**：JSONL。首行 meta
  （schema/video/roi/窗口/来源/协议），其后每帧一条记录，每帧显式带 `state`：
  - `read`     直读（K 次独立读一致得到的值）
  - `inferred` 由连续性/物理等先验补出（**不是读出来的**）
  - `unknown`  不知道（看不清、读数分歧）
  `state` 是这套格式存在的核心理由：C-95 证明"读出来的"与"补出来的"必须分开记录，
  否则"制造"的读数会被当成"读"的读数（先例= test4_truth 的 12 个"混合段"）。
- **RVTL shim（兼容层）**：读写 RVTL 形状的 csv，供既有语料与 `e2e_smoke --truth`
  消费。RVTL 的格式怪癖只允许住在这个模块里，产品/工具别处不得再解析它的头。

原生记录（每帧一行 JSON，字段固定以便下游按 key 取）：
    {"frame": 363, "value": 257, "state": "read", "reads": [257, 257], "engine": 257}
`unknown` 时 value 为 null；`inferred` 时 value 有值，但必须带 `source`（谁补的）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "truth.v1"
STATES = ("read", "inferred", "unknown")
UNKNOWN_TOKEN = "?"


# ────────────────────────── 原生格式（主） ──────────────────────────


def build_meta(
    video: str,
    roi: tuple[int, int, int, int],
    frame_start: int,
    frame_end: int | None,
    blind: bool = True,
    reads: int = 0,
    source: str = "",
) -> dict:
    return {
        "schema": SCHEMA,
        "video": Path(str(video)).name,
        "roi": list(roi),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "protocol": {"blind": blind, "reads": reads},
        "source": source,
    }


def record(
    frame: int,
    value: int | None,
    state: str,
    reads: list[int] | None = None,
    engine: int | None = None,
    source: str | None = None,
) -> dict:
    """一条帧记录；state 非法或 inferred 缺 source 即报错（不留模糊态）。"""
    if state not in STATES:
        raise ValueError("非法 state：%r（允许 %s）" % (state, "/".join(STATES)))
    if state == "unknown" and value is not None:
        raise ValueError("state=unknown 不得带 value（不知道就是不知道）")
    if state != "unknown" and value is None:
        raise ValueError("state=%s 必须带 value" % state)
    if state == "inferred" and not source:
        raise ValueError("state=inferred 必须写明 source（谁把它补出来的）")
    rec: dict[str, Any] = {"frame": int(frame), "value": value, "state": state}
    if reads:
        rec["reads"] = list(reads)
    if engine is not None:
        rec["engine"] = engine
    if source:
        rec["source"] = source
    return rec


def write_native(path: Path, meta: dict, records: list[dict]) -> int:
    """写 JSONL：首行 meta，其后每帧一行。返回写入的帧数。"""
    if meta.get("schema") != SCHEMA:
        raise ValueError("meta.schema 必须是 %s" % SCHEMA)
    lines = [json.dumps({"kind": "meta", **meta}, ensure_ascii=False, sort_keys=True)]
    for r in sorted(records, key=lambda x: x["frame"]):
        lines.append(json.dumps({"kind": "frame", **r}, ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(records)


def read_native(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    recs: list[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        if obj.get("kind") == "meta":
            meta = obj
        elif obj.get("kind") == "frame":
            obj.pop("kind", None)
            recs.append(obj)
        else:
            raise ValueError("原生格式行缺 kind 字段：%r" % ln[:80])
    if meta.get("schema") != SCHEMA:
        raise ValueError("原生格式 meta.schema 异常：%r" % meta.get("schema"))
    return meta, recs


def counts(records: list[dict]) -> dict[str, int]:
    out = {s: 0 for s in STATES}
    for r in records:
        out[r["state"]] = out.get(r["state"], 0) + 1
    return out


# ────────────────────────── RVTL shim（兼容层） ──────────────────────────

_RVTL_ROI = re.compile(r"roi=(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
_RVTL_INT = re.compile(r"([a-z_]+)=(\d+)")


def parse_rvlt_header(text: str) -> dict:
    """读 RVTL 头（'#' 行）。容忍：字段缺失、roi 内联四数、多余列、BOM。

    只做"读得出来"的最低保证；缺 video/roi 即报错（不放默认值蒙混）。
    """
    head: dict = {}
    for ln in text.splitlines():
        if not ln.startswith("#"):
            if ln.strip():
                break
            continue
        m = re.search(r"video=([^\s,]+)", ln)
        if m:
            head["video"] = m.group(1)
        m = _RVTL_ROI.search(ln)
        if m:
            head["roi"] = tuple(int(g) for g in m.groups())
        for key, val in _RVTL_INT.findall(ln):
            if key == "roi":
                continue  # roi 是四元组（上面按内联四数解），标量扫描不得覆盖它
            head[key] = int(val)
        m = re.search(r"codec=([^\s,]+)", ln)
        if m:
            head["codec"] = m.group(1)
    for req in ("video", "roi"):
        if req not in head:
            raise ValueError("RVTL 头缺 %s" % req)
    head.setdefault("frame_start", 0)
    return head


def read_rvlt(path: Path) -> tuple[dict, list[tuple[int, int | None]], list[str]]:
    """读 RVTL 数据行 → [(帧, 值|None)]；无法解析的行原样回传（不静默丢）。

    列语义按列数判定，这是 RVTL 格式里最容易静默出错的地方：
    - RVTL 原生行 `frame,time,text,conf` → **值在第 3 列**（第 2 列是秒数）
    - 本 shim 自己的导出 `frame,value` → 值在第 2 列
    取错列不会报错，只会让真值集合变空/错位（曾经发生过），故此处显式判列数。
    """
    text = path.read_text(encoding="utf-8-sig")
    head = parse_rvlt_header(text)
    rows: list[tuple[int, int | None]] = []
    bad: list[str] = []
    for ln in text.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            bad.append(ln)
            continue
        raw = parts[2] if len(parts) >= 3 else parts[1]
        rows.append((int(parts[0]), int(raw) if raw.isdigit() else None))
    return head, rows, bad


def write_rvlt(path: Path, meta: dict, records: list[dict], allow_inferred: bool = False) -> int:
    """写 RVTL 兼容 csv，供既有消费方（语料比对 / e2e_smoke --truth）使用。

    **默认拒绝把 inferred 当读数写出**——兼容层不得成为"制造读数"的出口；
    确有需要时须显式 allow_inferred=True，并在头部记 inferred 计数。
    """
    roi = meta["roi"]
    inferred = [r for r in records if r["state"] == "inferred"]
    if inferred and not allow_inferred:
        raise ValueError(
            "拒绝写出 %d 条 inferred 记录（它们不是读数）；确需写出请显式 allow_inferred=True"
            % len(inferred)
        )
    head = [
        "# 视觉真值（本项目 truth.v1 的 RVTL 兼容导出；人不得手改）",
        "# video=%s, roi=%d,%d,%d,%d, frame_start=%s, frame_end=%s"
        % (
            meta.get("video", ""),
            roi[0],
            roi[1],
            roi[2],
            roi[3],
            meta.get("frame_start"),
            meta.get("frame_end") or "",
        ),
        "# schema=%s, source=%s, inferred=%d" % (SCHEMA, meta.get("source", ""), len(inferred)),
    ]
    lines = []
    for r in sorted(records, key=lambda x: x["frame"]):
        lines.append(
            "%d,%s" % (r["frame"], r["value"] if r["value"] is not None else UNKNOWN_TOKEN)
        )
    path.write_text("\n".join(head + lines) + "\n", encoding="utf-8")
    return len(records)
