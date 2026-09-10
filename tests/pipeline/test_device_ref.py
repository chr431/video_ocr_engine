"""S9-3 载荷单测：DeviceRef 三态语义 + SegmentTask/InferBatch/OcrResult。"""
from __future__ import annotations

from video_ocr_engine.gpu.frame_ref import DeviceRef
from video_ocr_engine.pipeline.ocr_stage import (
    InferBatch, OcrResult, SegmentTask)


def test_device_ref_three_states():
    # 未裁切：span = 全宽
    r = DeviceRef(ptr=1234, h=48, w=320, owner="buf")
    assert r.deferred is True          # crop_w 未定 = 待裁（或本就不裁）
    assert r.span == (0, 320)
    assert r.x_off is None and r.crop_w is None

    # 延后裁切标记：sharp 就位、裁切未定
    mark = DeviceRef(ptr=1, h=48, w=320, owner="b", sharp=12.5)
    assert mark.deferred is True and mark.sharp == 12.5

    # 已裁切：span = 裁切区间
    cut = r.with_crop(40, 200)
    assert cut.deferred is False
    assert cut.span == (40, 200)
    assert cut.ptr == r.ptr and cut.owner == r.owner   # 引用保持
    assert cut.sharp is None                            # 标记态丢弃


def test_device_ref_with_crop_from_mark():
    mark = DeviceRef(ptr=7, h=48, w=320, owner="b", sharp=3.0)
    cut = mark.with_crop(0, 320)
    assert cut.deferred is False and cut.span == (0, 320)


def test_segment_task_and_infer_batch_fields():
    t = SegmentTask(idx=3, rep=100, crop=None, dev="D", frac=0.5)
    assert (t.idx, t.rep, t.crop, t.dev, t.frac) == (3, 100, None, "D", 0.5)
    b = InferBatch([1], [2], None, [0.1], force_aspect=1.5, infos=["i"])
    assert b.infos == ["i"] and b.force_aspect == 1.5 and b.procs is None
    h = InferBatch([1], [2], ["p"], [0.1])
    assert h.procs == ["p"] and h.infos is None


def test_ocr_result_namedtuple_keeps_indexing():
    r = OcrResult("text", 0.9, 42)
    assert r[0] == "text" and r[1] == 0.9 and r[2] == 42   # 旧索引消费兼容
    assert r.text == "text" and r.conf == 0.9 and r.rep == 42
