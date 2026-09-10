"""S0 金标向量录制器（v2 ARCHITECTURE.md §8.5 / §11 S0）。

对固定视频窗口跑 v1 全链提取，把**阶段输出摘要**（段结构 / 文本 / 置信度 /
校准阈值 / 降级原因 + 哈希）落成机器可读 JSON——不存像素。这些向量是 v2
strangler 迁移（S3/S4/S5）的逐位对账参照系。

用法：
  python tests/golden/record.py                # 录制全部用例
  python tests/golden/record.py --case A2      # 只录指定用例
  python tests/golden/record.py --verify       # 重跑全部用例并与已录向量逐位比对
  python tests/golden/record.py --verify --case A2

比对口径：timing 仅供参考不比对；其余字段（段结构 / 文本 / 置信度全精度 /
阈值 / 降级 / crop sha）必须逐位一致。分歧处理遵循 §8.5：默认当 bug。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

VIDS = {
    "test5": r"D:\Videos\racelog_test\test5.mp4",        # h264, 7761 帧
    "test6_av1": r"D:\Videos\racelog_test\test6.mp4",    # av1, 23970 帧
    "test6_h264": r"D:\Videos\racelog_test\test6_h264.mp4",
    "test6_hevc": r"D:\Videos\racelog_test\test6_hevc.mp4",
}
ROI = {"test5": (843, 993, 948, 1025), "test6": (841, 994, 949, 1026)}
WIN = (0, 3000)

# ── 覆盖矩阵（先做全：解码×管线×OCR 网格 + 编码族 + 格式/标志位）──────
def _mk(cid, vid, *, env=None, **kw):
    base = dict(video=vid, roi=ROI["test5" if vid == "test5" else "test6"],
                frame_start=WIN[0], frame_end=WIN[1], keep_crops=False)
    base.update(kw)
    return {"id": cid, "kwargs": base, "env": env or {}}


MATRIX = []
# A：test5 上 decode × pipeline × ocr 全网格（12）
for dec in ("cpu", "nvdec", "hybrid"):
    for pipe in (0, 1):
        for ocr in ("cpu", "tensorrt"):  # S3-1 勘误:"onnx"非合法值,v1 静默当 TRT(B3 活案例)
            MATRIX.append(_mk("A-%s-p%d-%s" % (dec, pipe, ocr), "test5",
                              decode_backend=dec, ocr_backend=ocr,
                              env={"GPU_PIPELINE": str(pipe)}))
# B：格式与标志位（基底 = nvdec + GPU 管线 + TRT）
for fmt in ("gray", "yuv"):
    for kc in (True, False):
        MATRIX.append(_mk("B-fmt-%s-kc%d" % (fmt, int(kc)), "test5",
                          decode_backend="nvdec", ocr_backend="tensorrt",
                          rep_crop_format=fmt, keep_crops=kc,
                          env={"GPU_PIPELINE": "1"}))
MATRIX.append(_mk("B-stride8", "test5", decode_backend="nvdec",
                  ocr_backend="tensorrt", sample_stride=8,
                  env={"GPU_PIPELINE": "1"}))
# B-aspect：force_aspect 是宽高比（w/h，>0 压窄内容），输入宽 = 48×ratio。
# S0 发现#1：ratio=320 时输入宽 15360 超出 TRT profile 上限 2048，且 v1 报错
# 路径有缺陷（setInputShape 报错后崩在 reshape ValueError，见 tests/golden/FINDINGS.md）。
MATRIX.append(_mk("B-aspect4", "test5", decode_backend="nvdec",
                  ocr_backend="tensorrt", force_aspect=4.0,
                  env={"GPU_PIPELINE": "1"}))
MATRIX.append(_mk("B-nomerge", "test5", decode_backend="nvdec",
                  ocr_backend="tensorrt", merge_similar=False,
                  env={"GPU_PIPELINE": "1"}))
# C：test6 同内容三编码族 × 解码后端（9）
for vid in ("test6_av1", "test6_h264", "test6_hevc"):
    for dec in ("cpu", "nvdec", "hybrid"):
        MATRIX.append(_mk("C-%s-%s" % (vid.split("_")[1], dec), vid,
                          decode_backend=dec, ocr_backend="tensorrt",
                          env={"GPU_PIPELINE": "1"}))

TIMING_ONLY = ("timing",)  # verify 时跳过的字段（非确定性）


def sha_video(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    env = {"commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
        capture_output=True, text=True).stdout.strip()}
    # S6（F-11）：**把 OCR 引擎产物指纹记进 manifest**。TRT 每次重建的
    # tactic 选择不保证一致 → 同一 profile 的两次构建也会让置信度在第 4 位
    # 漂移（文本/段结构不变）。不记指纹就无法区分"代码回归"与"引擎重建"。
    try:
        import hashlib
        from video_ocr_engine.ocr.trt import TrtEngine
        for cand in TrtEngine._engine_candidates("small"):
            if cand.exists():
                h = hashlib.sha256(cand.read_bytes()).hexdigest()[:16]
                env["ocr_engine"] = "%s@%s" % (cand.name, h)
                break
        else:
            env["ocr_engine"] = None
    except Exception:  # noqa: BLE001 无 TRT/无引擎时只记 None，不阻塞录制
        env["ocr_engine"] = None
    try:
        import decord
        env["decord"] = decord.__version__
    except Exception:
        env["decord"] = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10).stdout.strip()
        env["gpu"], env["driver"] = [x.strip() for x in out.split(",")]
    except Exception:
        env["gpu"] = env["driver"] = None
    env["python"] = sys.version.split()[0]
    return env


def summarize(case: dict) -> tuple[dict, dict]:
    """跑一次提取，返回 (stage-calib, stage-ocr) 两份摘要。

    v1 的 env 是调用期读取（README:236-238），因此按用例临时改写再复原。
    """
    from video_ocr_engine import FieldExtractor
    saved = {k: os.environ.get(k) for k in case["env"]}
    os.environ.update(case["env"])
    try:
        kw = dict(case["kwargs"])
        vid = kw.pop("video")
        roi = kw.pop("roi")
        ex = FieldExtractor(VIDS[vid], roi, **kw)
        t0 = time.perf_counter()
        r = ex.extract()
        wall = time.perf_counter() - t0
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    segs = [[s.start, s.end, s.rep_frame, s.text, s.confidence]
            for s in r.segments]
    crop_sha = None
    if case["kwargs"].get("keep_crops"):
        crop_sha = [hashlib.sha256(
            (s.rep_crop.tobytes() if s.rep_crop is not None else b"")
        ).hexdigest()[:16] for s in r.segments]

    struct = hashlib.sha256(
        json.dumps([[s[0], s[1], s[2]] for s in segs]).encode()).hexdigest()
    text = hashlib.sha256("".join(s[3] for s in segs).encode()).hexdigest()

    calib = {"bin_thresh": getattr(ex, "_bin_thresh", None),
             "degraded_reason": r.meta.get("degraded_reason"),
             "backend": r.meta.get("backend"),
             "ocr_backend": r.meta.get("ocr_backend"),
             "codec": r.meta.get("codec"),
             "color_range": r.meta.get("color_range"),
             "meta_keys": sorted(r.meta.keys())}
    ocr = {"n_segments": len(segs), "segments": segs,
           "seg_struct_sha": struct, "text_sha": text,
           "crop_sha": crop_sha,
           "timing": dict(r.timing), "wall": round(wall, 3)}
    return calib, ocr


def write_case(case: dict, calib: dict, ocr: dict) -> None:
    d = GOLDEN / ("case-%s" % case["id"])
    d.mkdir(exist_ok=True)
    (d / "stage-calib.json").write_text(
        json.dumps(calib, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (d / "stage-ocr.json").write_text(
        json.dumps(ocr, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")


def load_case(case: dict) -> tuple[dict, dict]:
    d = GOLDEN / ("case-%s" % case["id"])
    return (json.loads((d / "stage-calib.json").read_text(encoding="utf-8")),
            json.loads((d / "stage-ocr.json").read_text(encoding="utf-8")))


def verify_case(case: dict) -> list[str]:
    calib_new, ocr_new = summarize(case)
    calib_old, ocr_old = load_case(case)
    diffs = []
    for name, new, old in (("calib", calib_new, calib_old),
                           ("ocr", ocr_new, ocr_old)):
        for k in new:
            if k in TIMING_ONLY or k == "wall":
                continue
            if new[k] != old.get(k):
                diffs.append("%s.%s" % (name, k))
    return diffs


def write_manifest(env: dict, vids_sha: dict) -> None:
    lines = ["# S0 金标向量清单——录制基准与期望哈希（由 record.py 生成，人不得手改）",
             "environment:"]
    for k, v in env.items():
        lines.append("  %s: %s" % (k, json.dumps(v, ensure_ascii=False)))
    lines.append("videos:")
    for vid, sha in vids_sha.items():
        lines.append("  %s: %s" % (vid, sha))
    lines.append("cases:")
    for c in MATRIX:
        d = GOLDEN / ("case-%s" % c["id"])
        hs = []
        for f in ("stage-calib.json", "stage-ocr.json"):
            hs.append(hashlib.sha256(
                (d / f).read_bytes()).hexdigest()[:16])
        lines.append("  - id: %s" % c["id"])
        lines.append("    video: %s" % c["kwargs"]["video"])
        lines.append("    window: [%d, %d]" % (WIN[0], WIN[1]))
        lines.append("    env: %s" % json.dumps(c["env"]))
        lines.append("    kwargs: %s" % json.dumps(
            {k: v for k, v in c["kwargs"].items()
             if k not in ("video", "roi", "frame_start", "frame_end")},
            ensure_ascii=False))
        lines.append("    sha: [%s, %s]" % (hs[0], hs[1]))
    (GOLDEN / "manifest.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="只处理指定 id（子串匹配）")
    ap.add_argument("--verify", action="store_true",
                    help="重跑并与已录向量比对（S0 门禁：哈希一致）")
    args = ap.parse_args()
    cases = [c for c in MATRIX if not args.case or args.case in c["id"]]

    if args.verify:
        bad = 0
        for c in cases:
            diffs = verify_case(c)
            tag = "OK " if not diffs else "DIFF"
            print("%s %s%s" % (tag, c["id"],
                               "" if not diffs else " -> " + ",".join(diffs)))
            bad += bool(diffs)
        print("verify: %d/%d 一致" % (len(cases) - bad, len(cases)))
        return 1 if bad else 0

    env = environment()
    vids_sha = {vid: sha_video(p) for vid, p in VIDS.items()}
    for c in cases:
        t0 = time.perf_counter()
        calib, ocr = summarize(c)
        write_case(c, calib, ocr)
        print("REC %s  %d 段  %.1fs  degraded=%s" % (
            c["id"], ocr["n_segments"], time.perf_counter() - t0,
            calib["degraded_reason"]))
    if not args.case:
        write_manifest(env, vids_sha)
        print("manifest.yaml 已更新（%d 用例）" % len(MATRIX))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
