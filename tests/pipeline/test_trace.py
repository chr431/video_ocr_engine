"""P4 实验性 trace/timeline（VOE_TRACE_FILE）的守卫。

两条红线：
1. **关闭零成本（结构性）**：旋钮默认空 → extractor 不建 recorder，
   `_prof_end` 尾部只多一次 is-not-None 判断。该判断的运行期成本由
   `tests/config/test_telemetry_cost.py` 的 std 档 ≤2ms/run 守卫背书
   （trace 代码在场但关闭，守卫原样通过）。
2. **开启成本宽上限**（用户裁决放宽，非 std 预算）：~2.3k 事件的记录
   必须毫秒级；上限取防病理回退的宽松值。
"""
from __future__ import annotations

import json
import time

import pytest

from video_ocr_engine.domain.trace import TraceRecorder


def test_record_and_dump_roundtrip(tmp_path):
    tr = TraceRecorder()
    tr.record("decode.batch", 100.0, 100.5)
    tr.record_run(99.9, 1.2)
    out = tmp_path / "t.json"
    info = tr.dump(str(out), meta={"backend": "x"}, wall=1.2)
    assert info["n_events"] == 2
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["schema"] == 1 and d["n_events"] == 2
    # 按 t0 排序 + 相对时间轴（首事件 t0=0）
    assert [e[0] for e in d["events"]] == ["pipeline.run", "decode.batch"]
    assert d["events"][0][1] == pytest.approx(0.0)
    assert d["events"][1][2] - d["events"][1][1] == pytest.approx(0.5)


def test_multithread_buckets_merge(tmp_path):
    import threading
    tr = TraceRecorder()
    t = threading.Thread(target=lambda: tr.record("ocr.infer", 10.0, 10.1))
    t.start()
    t.join()
    tr.record("decode.batch", 5.0, 5.2)
    out = tmp_path / "m.json"
    info = tr.dump(str(out), wall=1.0)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert info["n_events"] == 2
    assert d["events"][0][0] == "decode.batch"    # t0 排序跨线程桶


def test_record_cost_loose_cap():
    """开启成本：2300 事件（真实 run 量级）的记录须 ~1ms 级。

    上限 50ms = 实测的数十倍，只为抓"误把昂贵调用塞进 record"的病理
    回退——本路径的严格成本不归 std 档 2ms 预算管（用户裁决放宽）。
    """
    cost = 0.0
    for _ in range(4):                       # 首轮预热，取末轮
        tr = TraceRecorder()
        t0 = time.perf_counter()
        for i in range(2300):
            tr.record("pipeline.consume_feed", t0, t0 + 0.0001)
        cost = time.perf_counter() - t0
    assert cost < 0.05, "2300 事件记录成本 %.1fms（应 ~1ms 级）" % (cost * 1e3)


def test_knob_defaults_off(monkeypatch):
    """旋钮默认空（关闭）；diag 链路把它带进 RunConfig。"""
    monkeypatch.delenv("VOE_TRACE_FILE", raising=False)
    from video_ocr_engine.config.resolve import resolve
    rc = resolve()
    assert rc.diag_trace_file == ""
    from video_ocr_engine.config.knobs import KNOBS
    assert "diag.trace_file" in {k.name for k in KNOBS.knobs}
