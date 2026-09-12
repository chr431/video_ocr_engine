"""路径普查：用 std 档 RunReport 给每条路径做时间去向与背压分诊。

**这是测量，不是 A/B**：单臂各跑一次取报告。任何"A 比 B 快 N%"的结论仍须走
`tools/bench.py ab`（交错测量，见 AGENTS.md 的 7.7% 漂移警告）；本探针只回答
"这条路径的时间花在哪、瓶颈是谁、背压是持续饥饿还是偶发长停顿"。

2026-09-13 起可用：`pipeline.q_{get_wait,put_block}_{n,max}` 进了 std 档，
故不需要 full 档也不需要额外探针即可分诊（总时长只说明堵了多久；次数与
单次最长才区分"持续细碎饥饿"与"个别长停顿"）。

用法：
    python tools/_probe_path_survey.py                 # 默认 5 条路径 × 2000 帧
    SURVEY_FRAMES=8000 python tools/_probe_path_survey.py
    SURVEY_VIDEO=test6 python tools/_probe_path_survey.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_ocr_engine import FieldExtractor  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))
_ROIS = {
    "test5": (843, 993, 948, 1025),
    "test6": (841, 994, 949, 1026),
    "test": (841, 994, 949, 1026),
}

CONFIGS = [
    ("cpu/cpu", dict(decode_backend="cpu", ocr_backend="cpu")),
    ("nvdec/cpu", dict(decode_backend="nvdec", ocr_backend="cpu")),
    ("nvdec/trt", dict(decode_backend="nvdec", ocr_backend="tensorrt")),
    ("hybrid/trt", dict(decode_backend="hybrid", ocr_backend="tensorrt")),
    ("hybrid/cpu", dict(decode_backend="hybrid", ocr_backend="cpu")),
]

HDR = ("路径           wall   backend          decode  consumer    ocr"
       " | get_wait 总/次/最长      put_block 总/次/最长  | 段/uniq")


def _row(label: str, rep: dict, segs: int, uniq: int) -> str:
    sp = rep.get("spans", {})
    g = rep.get("gauges", {})
    c = rep.get("counters", {})

    def s(k):
        return (sp.get(k) or {}).get("sum", 0.0)

    return ("%-13s %6.3f  %-15s %6.3f  %7.3f %6.3f | %5.3f %4s/%.4f  "
            "%5.3f %4s/%.4f | %d/%d" % (
                label, rep.get("wall_s", 0.0),
                (rep.get("pipeline") or {}).get("backend", "?"),
                s("pipeline.decode"), s("pipeline.consumer"), s("pipeline.ocr"),
                g.get("pipeline.q_get_wait", 0.0),
                c.get("pipeline.q_get_wait_n", 0),
                g.get("pipeline.q_get_wait_max", 0.0),
                g.get("pipeline.q_put_block", 0.0),
                c.get("pipeline.q_put_block_n", 0),
                g.get("pipeline.q_put_block_max", 0.0),
                segs, uniq))


def main() -> int:
    name = os.environ.get("SURVEY_VIDEO", "test5")
    frames = int(os.environ.get("SURVEY_FRAMES", "2000"))
    video = str(_VIDEO_DIR / (name + ".mp4"))
    roi = _ROIS.get(name, _ROIS["test"])
    print("视频 %s  ROI %s  帧数 %d" % (video, roi, frames))
    print(HDR)
    print("-" * len(HDR))
    base = None
    for label, kw in CONFIGS:
        try:
            ex = FieldExtractor(video, roi, frame_end=frames, **kw)
            r = ex.extract()
        except Exception as e:                          # noqa: BLE001
            print("%-13s 失败: %r" % (label, e))
            continue
        rep = r.meta.get("report") or {}
        uniq = len({x.text for x in r.segments if x.text})
        print(_row(label, rep, len(r.segments), uniq))
        if base is None:
            base = (len(r.segments), uniq)
        elif (len(r.segments), uniq) != base:
            print("    ⚠️ 段数/唯一文本与首条不一致 —— 该路径结果已变，勿只看耗时")
    print("\n读法：get_wait = 消费者等数据（大 = 解码供给不足）；"
          "put_block = 生产者被压（大 = 下游消费不足）。\n"
          "      最长/总量 占比高 → 个别长停顿；占比低而总量大 → 全程持续饥饿。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
