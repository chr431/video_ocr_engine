"""自有真值格式（truth.v1）与 RVTL 兼容 shim 的门禁（tools/_truth_format.py）。

这套格式存在的理由就是"失败不能静默"，所以门禁也照这个方向钉：
1. 原生格式必须无损往返，且逐帧 state（read/inferred/unknown）不得被压平——
   C-95 证明"读出来的"与"补出来的"混同会制造假真值（test4_truth 的 12 个混合段）。
2. RVTL 头只做"读得出来"的最低保证：容忍字段缺失/内联 roi/多余列/BOM，
   但缺 video/roi 必须报错（不许拿默认值蒙混）。
3. 兼容层不得成为"制造读数"的出口：inferred 记录默认拒绝写出。
"""

from __future__ import annotations

import sys

import pytest
from _paths import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import _truth_format as tf

RVTL_OK = """# RaceVideoToLog v2.7.0
# video_hash=abc, video=test5.mp4, fps=59.767, codec=h264
# roi=843,993,948,1025, format=km/h, frame_start=362, frame_end=7585
# max_speed=400.0, max_accel=40.0
362,0.00,257,21
363,1.19,257,21
364,2.39,?
"""


# ────────────────────────── 原生格式 ──────────────────────────


def test_native_round_trip_preserves_state(tmp_path):
    """往返无损：三种 state 都必须原样回来（尤其 unknown 不得变成 0 或空）。"""
    meta = tf.build_meta("test5.mp4", (843, 993, 948, 1025), 362, 7585, blind=True, reads=2)
    recs = [
        tf.record(362, 257, "read", reads=[257, 257], engine=257),
        tf.record(363, None, "unknown", reads=[257, 258], engine=257),
        tf.record(364, 259, "inferred", source="相邻段插值"),
    ]
    p = tmp_path / "t.jsonl"
    assert tf.write_native(p, meta, recs) == 3
    meta2, recs2 = tf.read_native(p)
    assert meta2["schema"] == tf.SCHEMA and meta2["roi"] == [843, 993, 948, 1025]
    assert [r["state"] for r in recs2] == ["read", "unknown", "inferred"]
    assert recs2[1]["value"] is None and recs2[1]["reads"] == [257, 258]
    assert recs2[2]["source"] == "相邻段插值"
    assert tf.counts(recs2) == {"read": 1, "inferred": 1, "unknown": 1}


def test_record_rejects_ambiguous_states():
    """不留模糊态：unknown 带值、read 无值、inferred 无来源，全部拒绝。"""
    with pytest.raises(ValueError):
        tf.record(1, 5, "unknown")
    with pytest.raises(ValueError):
        tf.record(1, None, "read")
    with pytest.raises(ValueError):
        tf.record(1, 5, "inferred")
    with pytest.raises(ValueError):
        tf.record(1, 5, "guessed")


def test_native_write_requires_schema(tmp_path):
    """meta 缺 schema 或版本不符即报错（格式契约不得靠约定）。"""
    with pytest.raises(ValueError):
        tf.write_native(tmp_path / "x.jsonl", {"video": "a.mp4"}, [])


def test_native_read_rejects_unknown_line(tmp_path):
    """行缺 kind 即报错，避免"看不懂的行"被当成空数据吞掉。"""
    p = tmp_path / "bad.jsonl"
    p.write_text('{"kind": "meta", "schema": "truth.v1"}\n{"frame": 1, "value": 2}\n')
    with pytest.raises(ValueError):
        tf.read_native(p)


# ────────────────────────── RVTL shim ──────────────────────────


def test_rvlt_header_is_parsed_tolerantly(tmp_path):
    """容忍真实 RVTL 头的怪癖：roi 内联四数、多余列、BOM、字段顺序任意。"""
    p = tmp_path / "t.csv"
    p.write_text(RVTL_OK, encoding="utf-8-sig")
    head, rows, bad = tf.read_rvlt(p)
    assert head["video"] == "test5.mp4"
    assert head["roi"] == (843, 993, 948, 1025)
    assert (head["frame_start"], head["frame_end"]) == (362, 7585)
    assert head["codec"] == "h264" and head["fps"] == 59
    assert rows == [(362, 257), (363, 257), (364, None)]
    assert bad == []


def test_rvlt_header_missing_required_fields_raises():
    """缺 roi/video 必须报错——不许拿默认值蒙混过去。"""
    with pytest.raises(ValueError):
        tf.parse_rvlt_header("# 什么都没有\n1,2,3,0\n")
    with pytest.raises(ValueError):
        tf.parse_rvlt_header("# video=a.mp4\n1,2,3,0\n")


def test_rvlt_read_reports_bad_lines(tmp_path):
    """无法解析的数据行必须回传，而不是被静默丢掉。"""
    p = tmp_path / "t.csv"
    p.write_text(RVTL_OK.rstrip("\n") + "\n365;3.5;260;21\n", encoding="utf-8")
    _head, rows, bad = tf.read_rvlt(p)
    assert len(rows) == 3 and bad == ["365;3.5;260;21"]


def test_rvlt_export_refuses_inferred_by_default(tmp_path):
    """兼容层不得成为"制造读数"的出口：inferred 默认拒绝写出。"""
    meta = tf.build_meta("test5.mp4", (1, 2, 3, 4), 0, 10)
    recs = [tf.record(1, 5, "read"), tf.record(2, 6, "inferred", source="插值")]
    with pytest.raises(ValueError):
        tf.write_rvlt(tmp_path / "o.csv", meta, recs)
    assert tf.write_rvlt(tmp_path / "o.csv", meta, recs, allow_inferred=True) == 2
    text = (tmp_path / "o.csv").read_text(encoding="utf-8")
    assert "inferred=1" in text, "显式放行时必须在头部留痕"


def test_rvlt_export_round_trips_through_shim(tmp_path):
    """导出的 RVTL csv 必须能被 shim 读回（否则兼容层自欺）。"""
    meta = tf.build_meta("test5.mp4", (843, 993, 948, 1025), 362, 400)
    recs = [tf.record(362, 257, "read"), tf.record(363, None, "unknown")]
    out = tmp_path / "o.csv"
    tf.write_rvlt(out, meta, recs)
    head, rows, bad = tf.read_rvlt(out)
    assert bad == []
    assert head["roi"] == (843, 993, 948, 1025)
    assert head["frame_start"] == 362
    assert rows == [(362, 257), (363, None)]
