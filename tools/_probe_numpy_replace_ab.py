"""全量 numpy→cv2 替换的引擎级交错 A/B（分臂，回答"全换能否缩减墙钟"）。

命题（用户）：把所有用 numpy 的地方均换成更高效的库，看墙钟是否缩减。

内核 µbench（`_probe_numpy_all_kernels.py`，生产真实形状）已给出候选表：

| 热点 | 位置 | 候选 | 速度 | 逐位 |
|---|---|---|---|---|
| `_np_resize` | **消费者**（ocr_stage） | cv2.resize | **−94~95%** | NO(3e-05) |
| `_cluster_win3` | **生产者**（feed + 合并判定） | cv2.boxFilter | **−34%** | **yes** |
| `_segments_similar` diff | **生产者** | cv2.absdiff+mean+count | **−57%** | **yes** |
| `_text_sep_binary` | **生产者** | cv2.compare | **−29%** | **yes** |
| `sharp_std` | 生产者 | cv2.meanStdDev | −77% | NO(1e-14) |
| `gamma_pow` | 消费者 | cv2.pow | +29%（更慢） | NO |

**关键区别（为什么本轮与上轮结论可能不同）**：
  · resize 在**消费者**线程，而消费者有 ~1.46s `q_get_wait` 空等 →
    省下的时间被空等吸收（上轮实测墙钟仅 −0.26%）。
  · `_cluster_win3` / `_segments_similar` 在**生产者**线程，是 h264-cpu
    的关键路径（producer 2.05s ≈ wall 2.27s），且**逐位一致**——
    理论上应 1:1 兑现。

分臂设计：
  · **A** = 全 numpy 基线
  · **B** = 生产者侧 cv2（三个逐位一致的内核）
  · **C** = B + 消费者侧 cv2 resize（= 用户所说的"全部换掉"）

判据：交错 A/B/A/B/C + 独立子进程 + 热池 + 段数/文本 sha 门禁
（本机可分辨下限 0.484%，A/A p95）。

用法：
  python tools/_probe_numpy_replace_ab.py [--config h264-cpu] \
      [--frames 3000] [--repeat 3]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VIDS = {"h264-cpu": ("test5.mp4", (843, 993, 948, 1025), "cpu", "cpu"),
        "h264-gpu": ("test5.mp4", (843, 993, 948, 1025), "nvdec", "tensorrt"),
        "hevc-cpu": ("test6_hevc.mp4", (841, 994, 949, 1026), "cpu", "cpu")}

#: 生产者侧 cv2 替换。**必须逐位一致**（否则金标/段数会漂）。
#: 三处打桩点：
#:   ① `segmentation._cluster_win3`（feed 与合并判定都走模块全局）
#:   ② `extractor._text_sep_binary`（模块级 from-import 绑定）
#:   ③ `FieldExtractor._segments_similar`（diff 数学内联在方法里，整法替换）
PRODUCER_PATCH = r'''
def _install_producer_cv2():
    import cv2
    import numpy as np
    import video_ocr_engine.domain.segmentation as seg
    import video_ocr_engine.extractor as ex_mod

    def _box3(s, ddepth):
        padded = cv2.copyMakeBorder(s, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        w3 = cv2.boxFilter(padded, ddepth=ddepth, ksize=(3, 3),
                           normalize=False, borderType=cv2.BORDER_ISOLATED)
        return w3[1:-1, 1:-1]

    def cv_cluster_win3(diff):
        if not diff.any():
            return 0.0
        s = (diff.view(np.uint8) if diff.flags.c_contiguous
             else diff.astype(np.uint8))
        return float(_box3(s, cv2.CV_16U).max())

    def cv_text_sep_binary(gray, th):
        g8 = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
        return cv2.compare(g8, int(th), cv2.CMP_GT).astype(np.float32)

    seg._cluster_win3 = cv_cluster_win3
    ex_mod._text_sep_binary = cv_text_sep_binary

    def cv_segments_similar(self, a, b):
        # 与 extractor._segments_similar 同逻辑，只换 numpy 内核
        _text_mode = self._merge_effective_mode()
        if _text_mode == 'binary':
            a = cv_text_sep_binary(a, self._bin_thresh)
            b = cv_text_sep_binary(b, self._bin_thresh)
        if a is None or b is None or a.shape != b.shape:
            return False
        ai = a.astype(np.int16)
        bi = b.astype(np.int16)
        d = cv2.absdiff(ai, bi)
        mean = float(cv2.mean(d)[0])
        changed = int(cv2.countNonZero(cv2.compare(d, 10, cv2.CMP_GT)))
        _dense = seg.dense_gate_hit(cv_cluster_win3(d > 10),
                                    self._merge_dense_gate)
        return seg.similar_decision(mean, changed,
                                    self._merge_similar_threshold,
                                    self._merge_max_changed_pixels,
                                    dense=_dense)

    ex_mod.FieldExtractor._segments_similar = cv_segments_similar
'''

#: 消费者侧 cv2 resize（与上轮 `_probe_preproc_ab.py` 同实现）。
CONSUMER_PATCH = r'''
def _install_consumer_cv2():
    import cv2
    import numpy as np
    import video_ocr_engine.domain.video_utils as vu
    import video_ocr_engine.domain.segmentation as seg
    import video_ocr_engine.ocr.native as nat

    def _cv2_resize(img, new_w, new_h):
        src_h, src_w = img.shape[:2]
        if new_w == src_w and new_h == src_h:
            return img.astype(np.float32)
        f = np.ascontiguousarray(img.astype(np.float32))
        out = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 3 and out.ndim == 2:
            out = out[..., None]
        return out

    for mod in (vu, seg, nat):
        if hasattr(mod, "_np_resize"):
            mod._np_resize = _cv2_resize
'''

WORKER = r'''
import hashlib, json, os, sys, time
sys.path.insert(0, os.environ["PROBE_ROOT"])
import numpy as np

ARM = os.environ["PROBE_ARM"]

''' + PRODUCER_PATCH + "\n" + CONSUMER_PATCH + r'''

if ARM in ("B", "C"):
    _install_producer_cv2()
if ARM == "C":
    _install_consumer_cv2()

from video_ocr_engine import FieldExtractor

vid = os.path.join(os.environ["RACELOG_VIDEO_DIR"], os.environ["PROBE_VIDEO"])
roi = tuple(int(x) for x in os.environ["PROBE_ROI"].split(","))
outs = []
for i in range(2):                      # 第 1 轮冷、第 2 轮热（生产口径）
    ex = FieldExtractor(vid, roi, frame_start=0,
                        frame_end=int(os.environ["PROBE_FRAMES"]),
                        decode_backend=os.environ["PROBE_DEC"],
                        ocr_backend=os.environ["PROBE_OCR"], keep_crops=False)
    t = time.perf_counter()
    r = ex.extract()
    wall = time.perf_counter() - t
    rep = r.meta.get("report") or {}
    sp, ga = rep.get("spans", {}), rep.get("gauges", {})
    texts = sorted({s.text for s in r.segments})
    # 逐段文本+置信度指纹（比唯一文本集更严：能抓到置信度漂移）
    fp = hashlib.sha256(
        "\n".join("%s|%.5f" % (s.text, s.confidence) for s in r.segments
                  ).encode("utf-8")).hexdigest()[:16]
    outs.append({
        "wall": wall, "segs": len(r.segments), "texts": len(texts),
        "fp": fp,
        "producer": (sp.get("pipeline.consumer") or {}).get("sum"),
        "preproc": (sp.get("ocr.preprocess") or {}).get("sum"),
        "preproc_resize": (sp.get("ocr.preproc_resize") or {}).get("sum"),
        "infer": (sp.get("ocr.infer") or {}).get("sum"),
        "q_get_wait": ga.get("pipeline.q_get_wait"),
        "q_put_block": ga.get("pipeline.q_put_block"),
    })
print("PROBE_JSON " + json.dumps(outs))
'''


def one(arm: str, cfg: str, frames: int, roi_override: str = "") -> list:
    vid, roi, dec, ocr = VIDS[cfg]
    if roi_override:
        roi = tuple(int(x) for x in roi_override.split(","))
    env = dict(os.environ)
    env.update({"PROBE_ROOT": str(ROOT), "PROBE_ARM": arm,
                "PROBE_VIDEO": vid, "PROBE_ROI": ",".join(map(str, roi)),
                "PROBE_DEC": dec, "PROBE_OCR": ocr,
                "PROBE_FRAMES": str(frames),
                "RACELOG_VIDEO_DIR": env.get("RACELOG_VIDEO_DIR",
                                             r"D:\Videos\racelog_test")})
    p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    raise SystemExit("worker(%s/%s) 失败：\n%s\n%s"
                     % (arm, cfg, p.stdout[-1500:], p.stderr[-2000:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="h264-cpu")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=4.0)
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--roi", default="",
                    help="覆盖 ROI（宽 ROI 场景：numpy 成本随面积涨，"
                         "而解码成本不变 → 检验关键路径是否易主）")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cfgs = args.config.split(",")
    rows: dict = {c: {a: [] for a in arms} for c in cfgs}
    for i in range(args.repeat):
        for cfg in cfgs:
            for arm in arms:
                out = one(arm, cfg, args.frames, args.roi)
                rows[cfg][arm].append(out)
                h = out[-1]
                print("  [%d] %-10s %-2s hot=%.4f segs=%d fp=%s "
                      "producer=%.3f resize=%.3f wait=%.3f"
                      % (i, cfg, arm, h["wall"], h["segs"], h["fp"],
                         h["producer"] or -1, h["preproc_resize"] or -1,
                         h["q_get_wait"] or -1))
            time.sleep(args.cooldown)

    summary = {}
    print("\n%-10s %-4s %10s %10s %9s  %-8s %s"
          % ("config", "arm", "hot wall", "producer", "Δwall", "符号", "门禁"))
    for cfg in cfgs:
        base = statistics.median([rows[cfg]["A"][i][1]["wall"]
                                  for i in range(args.repeat)])
        base_segs = rows[cfg]["A"][-1][1]["segs"]
        base_fp = rows[cfg]["A"][-1][1]["fp"]
        summary[cfg] = {"A_wall": base, "arms": {}}
        for arm in arms:
            ws = [rows[cfg][arm][i][1]["wall"] for i in range(args.repeat)]
            m = statistics.median(ws)
            signs = ""
            if arm != "A":
                signs = "".join(
                    "+" if rows[cfg][arm][i][1]["wall"]
                    > rows[cfg]["A"][i][1]["wall"] else "-"
                    for i in range(args.repeat))
            segs_ok = all(rows[cfg][arm][i][1]["segs"] == base_segs
                          for i in range(args.repeat))
            fp_ok = all(rows[cfg][arm][i][1]["fp"] == base_fp
                        for i in range(args.repeat))
            gate = ("段数OK" if segs_ok else "段数**FAIL**") + "/" + \
                   ("指纹OK" if fp_ok else "指纹**FAIL**")
            prod = statistics.median([rows[cfg][arm][i][1]["producer"] or 0
                                      for i in range(args.repeat)])
            print("%-10s %-4s %10.4f %10.4f %+8.2f%%  %-8s %s"
                  % (cfg if arm == arms[0] else "", arm, m, prod,
                     (m - base) / base * 100 if arm != "A" else 0.0,
                     signs or "-", gate))
            summary[cfg]["arms"][arm] = {
                "wall": m, "producer": prod,
                "delta_pct": (m - base) / base * 100 if arm != "A" else 0.0,
                "signs": signs, "segs_ok": segs_ok, "fp_ok": fp_ok}

    out = ROOT / "bench" / "numpy_replace_ab.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "raw": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n→ %s" % out)
    print("注：本机可分辨下限 |Δ|p95 = 0.484%（A/A 标定，2026-09-13）")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
