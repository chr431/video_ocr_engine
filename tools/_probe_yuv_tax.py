"""yuv 输出税归因探针（DESIGN-REVIEW 后续：yuv vs gray 墙钟差的机理）。

问题：rep_crop_format="yuv" 比 "gray" 慢 26~59%（3000帧窗口实测），但内部链
恒为灰度、每段只多一张 rep crop 的 D2H —— 数据量不足以解释。归因三问：
  1) 税在 fork（decord GPU reader 的 NVDEC→输出格式转换）还是引擎
     （luma_nv12 提取 / keep_crops D2H）？—— Stage A(纯 fork) vs Stage C(引擎) 对比。
  2) 税按"每解码帧"还是"每输出帧"计？—— stride8 vs stride1 的税差
     （stride8 只输出 1/8 的帧；若两档税接近 ⇒ 每解码帧收税）。
  3) 同步/延迟主导还是带宽主导？—— ROI 尺寸缩放（带宽税随 ROI 变大，
     逐帧 sync/API 税近似恒定）。

方法：全部单跑串行，同进程内交错重复取 min；Stage A 不调 asnumpy
（设备驻留，模拟 GPU 管线消费）；Stage B 加 asnumpy（宿主交付）。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decord import VideoReader, gpu
import os
_VIDEO_DIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))


VID = str(_VIDEO_DIR / "test5.mp4")
WINDOW = (362, 3362)                      # 3000 源帧
BATCH = 64                                # = config.GPU_PIPELINE_DECODE_BATCH

ROIS = {
    "small(106x33)": (843, 993, 948, 1025),
    "mid(601x151)": (600, 800, 1200, 950),
    "large(1601x601)": (200, 400, 1800, 1000),
}
FMTS = ("gray", "yuv420")


def frames_for(stride):
    return list(range(WINDOW[0], WINDOW[1], stride))


def run_fork(fmt, stride, roi, asnumpy=False):
    """纯 fork 供给率：get_batch 消费帧列表，不碰引擎。返回墙钟秒。"""
    roi_kw = {"roi": (roi[0], roi[1], roi[2] + 1, roi[3] + 1)}
    vr = VideoReader(VID, ctx=gpu(0), output_format=fmt, **roi_kw)
    frames = frames_for(stride)
    last = None
    t0 = time.perf_counter()
    for s in range(0, len(frames), BATCH):
        last = vr.get_batch(frames[s:s + BATCH], roi=roi_kw["roi"])
        if asnumpy:
            last.asnumpy()
    dt = time.perf_counter() - t0
    del last
    del vr
    return dt


def stage_ab():
    print("== Stage A/B：纯 fork 供给率（无引擎；A=设备驻留，B=+asnumpy）==")
    print(f"{'roi':>16} {'stride':>6} {'fmt':>7} {'A墙钟s':>8} {'fps':>7} "
          f"{'B墙钟s':>8}")
    results = {}
    for roi_name, roi in ROIS.items():
        for stride in (8, 1):
            for asnumpy, tag in ((False, "A"), (True, "B")):
                # 交错两轮取 min（消除顺序漂移）
                vals = {}
                for rep in (0, 1):
                    for fmt in FMTS:
                        dt = run_fork(fmt, stride, roi, asnumpy)
                        vals.setdefault(fmt, []).append(dt)
                for fmt in FMTS:
                    dt = min(vals[fmt])
                    results[(roi_name, stride, fmt, tag)] = dt
                    n_out = len(frames_for(stride))
                    print(f"{roi_name:>16} {stride:>6} {fmt:>7} "
                          f"{dt:>8.3f} {n_out/dt:>7.0f} "
                          f"{'' if tag == 'A' else '':>8}")
    print("-- 税分解（min-of-2，秒）--")
    for roi_name in ROIS:
        for stride in (8, 1):
            t_ax = results[(roi_name, stride, "yuv420", "A")]
            t_ag = results[(roi_name, stride, "gray", "A")]
            t_bx = results[(roi_name, stride, "yuv420", "B")]
            t_bg = results[(roi_name, stride, "gray", "B")]
            print(f"{roi_name:>16} s{stride:<5} "
                  f"税A(yuv-gray)={t_ax-t_ag:+.3f}  "
                  f"税B={t_bx-t_bg:+.3f}")
    return results


def run_engine(fmt):
    from video_ocr_engine import FieldExtractor
    roi = ROIS["small(106x33)"]
    ex = FieldExtractor(VID, roi, frame_start=WINDOW[0],
                        frame_end=WINDOW[1], sample_stride=8,
                        rep_crop_format=fmt)
    t0 = time.perf_counter()
    ex.extract()
    wall = time.perf_counter() - t0
    return wall, ex


def stage_c():
    print("\n== Stage C：引擎级交错 A/B（首轮 gray 顺带暖机；各 3 轮）==")
    vals = {}
    details = {}
    order = ("gray", "yuv", "gray", "yuv", "yuv", "gray")
    for i, fmt in enumerate(order):
        wall, ex = run_engine(fmt)
        vals.setdefault(fmt, []).append(wall)
        details.setdefault(fmt, []).append(
            f"run{i}({fmt})={wall:.2f}s decode={ex.timing.get('decode'):.2f}s "
            f"q_get_wait={ex.profile.get('ocr', {}).get('q_get_wait', 0):.2f}s "
            f"段={ex._n_segments}")
        print(f"  run{i} {fmt:>4}: 墙钟={wall:.2f}s "
              f"decode={ex.timing.get('decode'):.2f}s "
              f"q_get_wait={ex.profile.get('ocr', {}).get('q_get_wait', 0):.2f}s "
              f"engine_init={ex.profile.get('ocr', {}).get('engine_init', 0)*1000:.0f}ms")
    for fmt in ("gray", "yuv"):
        print(f"  {fmt}: {sorted(round(v, 3) for v in vals[fmt])}")
    g = min(vals["gray"]); x = min(vals["yuv"])
    print(f"  >> 引擎税(yuv-gray, warm min) = {x-g:+.3f}s"
          f"  （对照 Stage A small/s8 fork 税 ≈ 0.000s）")


if __name__ == "__main__":
    stage_ab()
    stage_c()
