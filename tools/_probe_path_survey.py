"""路径普查：用 std 档 RunReport 给每条路径做时间去向与背压分诊。

**这是测量，不是 A/B**：单臂各跑一次取报告。任何"A 比 B 快 N%"的结论仍须走
`tools/bench.py ab`（交错测量，见 AGENTS.md 的 7.7% 漂移警告）；本探针只回答
"这条路径的时间花在哪、瓶颈是谁、背压是持续饥饿还是偶发长停顿"。

两个维度：
  · **解码/OCR 组合**（6）：cpu / nvdec / hybrid × cpu / trt —— 含 cpu+TRT
    （2026-09-13 补：上一轮遗漏；这是"软解供给 + 最强消费"的组合，
     用来分离"解码供给不足"与"OCR 消费不足"两种瓶颈）
  · **编码**（同内容族）：`test6` 三编码对照（av1=原片 / h264 / hevc），
    因为解码速率是 codec 强相关的（C-04：h264 CPU 快 ~2.9×、AV1 反转慢）。

用法：
    python tools/_probe_path_survey.py                       # 三编码 × 6 组合
    SURVEY_FRAMES=8000 python tools/_probe_path_survey.py
    SURVEY_VIDS=test6_h264,test6_hevc python tools/_probe_path_survey.py
    SURVEY_CONFIGS=cpu/trt,nvdec/trt python tools/_probe_path_survey.py
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
ROI_TEST6 = (841, 994, 949, 1026)
ROI_TEST5 = (843, 993, 948, 1025)

#: 视频键 → (文件名, ROI)。test6 三件套为**同内容**重编码，编码对照须用它们。
VIDS = {
    "test5": ("test5.mp4", ROI_TEST5),            # h264，独立内容
    "test6_av1": ("test6.mp4", ROI_TEST6),        # av1（原片）
    "test6_h264": ("test6_h264.mp4", ROI_TEST6),
    "test6_hevc": ("test6_hevc.mp4", ROI_TEST6),
}
#: 默认跑同内容三编码族（控制变量：内容一致，只变编码）
DEFAULT_VIDS = ("test6_h264", "test6_hevc", "test6_av1")

CONFIGS = [
    ("cpu/cpu", dict(decode_backend="cpu", ocr_backend="cpu")),
    ("cpu/trt", dict(decode_backend="cpu", ocr_backend="tensorrt")),
    ("nvdec/cpu", dict(decode_backend="nvdec", ocr_backend="cpu")),
    ("nvdec/trt", dict(decode_backend="nvdec", ocr_backend="tensorrt")),
    ("hybrid/cpu", dict(decode_backend="hybrid", ocr_backend="cpu")),
    ("hybrid/trt", dict(decode_backend="hybrid", ocr_backend="tensorrt")),
]

HDR = ("组合           wall   backend          decode  consumer    ocr"
       " | get_wait 总/次/最长      put_block 总/次/最长 | 段/uniq")


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


def _pick(env_key: str, all_keys, default):
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return list(default)
    want = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [w for w in want if w not in all_keys]
    if bad:
        print("未知 %s: %s（可选：%s）" % (env_key, bad, ", ".join(all_keys)))
        sys.exit(2)
    return want


def main() -> int:
    frames = int(os.environ.get("SURVEY_FRAMES", "2000"))
    vid_keys = _pick("SURVEY_VIDS", VIDS, DEFAULT_VIDS)
    combo_names = _pick("SURVEY_CONFIGS", [c[0] for c in CONFIGS],
                        [c[0] for c in CONFIGS])
    combos = [c for c in CONFIGS if c[0] in combo_names]
    print("帧数 %d｜编码 %s｜组合 %s"
          % (frames, "/".join(vid_keys), "/".join(combo_names)))
    summary = []
    for vk in vid_keys:
        fname, roi = VIDS[vk]
        print("\n=== %s（%s）ROI %s ===" % (vk, fname, roi))
        print(HDR)
        print("-" * len(HDR))
        base = None
        for label, kw in combos:
            try:
                ex = FieldExtractor(str(_VIDEO_DIR / fname), roi,
                                    frame_end=frames, **kw)
                r = ex.extract()
            except Exception as e:                      # noqa: BLE001
                print("%-13s 失败: %r" % (label, e))
                continue
            rep = r.meta.get("report") or {}
            uniq = len({x.text for x in r.segments if x.text})
            print(_row(label, rep, len(r.segments), uniq))
            summary.append((vk, label, rep.get("wall_s", 0.0),
                            (rep.get("spans", {}).get("pipeline.decode")
                             or {}).get("sum", 0.0),
                            (rep.get("spans", {}).get("pipeline.ocr")
                             or {}).get("sum", 0.0)))
            if base is None:
                base = (len(r.segments), uniq)
            elif (len(r.segments), uniq) != base:
                print("    ⚠️ 段数/唯一文本与首条不一致 —— 该组合结果已变")
    print("\n=== 汇总（wall / decode / ocr，秒）===")
    for vk, label, wall, dec, ocr in summary:
        print("  %-12s %-13s wall=%.3f  decode=%.3f  ocr=%.3f" %
              (vk, label, wall, dec, ocr))
    print("\n读法：get_wait = 消费者等数据（大 = 解码供给不足）；"
          "put_block = 生产者被压（大 = 下游消费不足）。\n"
          "      最长/总量 占比高 → 个别长停顿；占比低而总量大 → 全程持续饥饿。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
