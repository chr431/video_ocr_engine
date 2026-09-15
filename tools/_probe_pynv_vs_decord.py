"""PyNvVideoCodec（NVIDIA 第一方 NVDEC 绑定）vs decord fork 的正面对决。

动机：本轮 `_probe_binding.py` 实测三条 GPU 路径**全部解码绑定**
（OCR 推理换成零成本，墙钟仅 +0.1~0.2%）——所以"换依赖"只可能在解码侧
成立。而 PyNvVideoCodec 是 NVIDIA 官方维护的 NVDEC Python 绑定，是最有
资格挑战 decord fork 的候选（§4 P2-3 评估过 PyAV/torchaudio/OpenCV，
但那是 2026-08，PyNvVideoCodec 不在当时的表里）。

**公平口径**：
  · 两者都解**同一窗口**（3000 帧），都取 NVDEC 硬件路径；
  · PyNvVideoCodec 走 `SimpleDecoder`（裸解码，不含 ROI）；
  · 输出取回（`.numpy()`）与不取回（纯解码）分别计时——区分
    「解码能力」与「取回开销」；
  · decord 对照 = 本轮 `_probe_decode_ceiling.py` 的 NVDEC 口径。

**已知结构性差异（必须同表报告）**：PyNvVideoCodec 是**通用解码器**，
没有 ROI-first、没有 gray 输出、没有 NVDEC∥CPU hybrid。
所以即使它的裸解码更快，也不能直接换算成引擎收益。

用法：
  python tools/_probe_pynv_vs_decord.py [--video test5.mp4] [--frames 3000]
      [--h264] [--reps 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]


def pynv_decode(video: str, frames: int, gpu_id: int, reps: int,
                fetch: bool) -> dict:
    """PyNvVideoCodec SimpleDecoder：裸解码（+可选取回）。"""
    import PyNvVideoCodec as nvc
    ts = []
    meta = {}
    for _ in range(reps):
        dec = nvc.CreateSimpleDecoder(
            enc_file_path=video, gpuid=gpu_id, usecpuinputbuffer=False)
        got = 0
        t0 = time.perf_counter()
        while got < frames:
            try:
                pkt = dec.DecodeNextPacket()
            except Exception:  # noqa: BLE001  # EOF：上游用异常表示流结束
                break
            if pkt is None:
                break
            if fetch:
                f = dec.GetFrame(pkt)
                if f is None:
                    break
                _ = f.numpy()
            got += 1
        ts.append(time.perf_counter() - t0)
        try:
            del dec
        except Exception:  # noqa: BLE001
            pass  # 析构失败不影响已计时的结果，可忽略
    best = min(ts)
    return {"wall_s": round(best, 4), "frames": got,
            "fps": round(got / best, 1) if got else 0.0,
            "nvc_version": getattr(nvc, "__version__", "?")}


def pynv_decode_extractor(video: str, frames: int, gpu_id: int,
                          reps: int) -> dict:
    """用官方推荐的高层 API（ThreadedDecoder / CreateDecoder）作对照。"""
    import PyNvVideoCodec as nvc
    ts = []
    got = 0
    for _ in range(reps):
        try:
            dec = nvc.CreateDecoder(
                gpuid=gpu_id, codec="h264", usedevicememory=True,
                output_pix_fmt="nv12", threads=4, decode_mode="device",
                crop=None, verbose=False, gpu_direct=True,
                **{"input": video}) if False else None
        except Exception as e:  # noqa: BLE001
            return {"error": "%s: %s" % (type(e).__name__, e)}
        if dec is None:
            return {"error": "CreateDecoder 签名不匹配（跳过高层口径）"}
    return {"skipped": True, "reason": "高层 API 签名未定，仅报裸解码口径",
            "reps": reps, "got": got, "ts": ts}


def decord_nvdec(video: str, roi, frames: int, reps: int) -> dict:
    """decord fork 生产口径：NVDEC + ROI + gray。"""
    from decord import VideoReader, gpu
    ts = []
    for _ in range(reps):
        vr = VideoReader(video, ctx=gpu(0), output_format="gray",
                         width=roi[2] - roi[0], height=roi[3] - roi[1],
                         roi=roi)
        n = min(frames, len(vr))
        t0 = time.perf_counter()
        for s in range(0, n, 64):
            vr.get_batch(list(range(s, min(s + 64, n)))).asnumpy()
        ts.append(time.perf_counter() - t0)
        del vr
    best = min(ts)
    return {"wall_s": round(best, 4), "fps": round(n / best, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5.mp4")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    roi = tuple(int(x) for x in args.roi.split(","))
    video = str(Path(os.environ.get("RACELOG_VIDEO_DIR",
                                    r"D:\Videos\racelog_test")) / args.video)
    res: dict = {"video": args.video, "frames": args.frames, "roi": list(roi)}

    print("视频 %s  窗口 %d 帧" % (args.video, args.frames))

    print("\n① decord fork（现役，NVDEC + ROI + gray）")
    try:
        d = decord_nvdec(video, roi, args.frames, args.reps)
        print("   %8.4fs  %8.1f fps" % (d["wall_s"], d["fps"]))
        res["decord"] = d
    except Exception as e:  # noqa: BLE001
        print("   失败：%s: %s" % (type(e).__name__, e))
        res["decord"] = {"error": str(e)}

    print("\n② PyNvVideoCodec 裸解码（不取回像素）")
    try:
        p0 = pynv_decode(video, args.frames, args.gpu, args.reps, fetch=False)
        print("   %8.4fs  %8.1f fps  (%d 帧, v%s)"
              % (p0["wall_s"], p0["fps"], p0["frames"], p0["nvc_version"]))
        res["pynv_nofetch"] = p0
    except Exception as e:  # noqa: BLE001
        print("   失败：%s: %s" % (type(e).__name__, e))
        res["pynv_nofetch"] = {"error": str(e)}

    print("\n③ PyNvVideoCodec 裸解码 + 取回（.numpy()，全帧）")
    try:
        p1 = pynv_decode(video, args.frames, args.gpu, args.reps, fetch=True)
        print("   %8.4fs  %8.1f fps  (%d 帧)"
              % (p1["wall_s"], p1["fps"], p1["frames"]))
        res["pynv_fetch"] = p1
    except Exception as e:  # noqa: BLE001
        print("   失败：%s: %s" % (type(e).__name__, e))
        res["pynv_fetch"] = {"error": str(e)}

    # 结构性对照
    print("\n④ 结构性差异（PyNvVideoCodec 缺什么）")
    gaps = ["ROI-first 解码（本机 1.75×，见 _probe_decode_ceiling.py）",
            "gray 单通道输出（省 2/3 拷贝）",
            "NVDEC∥CPU hybrid 双解码（C-01/C-46）",
            "YUV420 keep_crops 直通"]
    for g in gaps:
        print("   ✗ %s" % g)
    res["structural_gaps"] = gaps

    out = ROOT / "bench" / "pynv_vs_decord.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
