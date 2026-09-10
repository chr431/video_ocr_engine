"""D1 调查：gpu/host 两管线段数不一致（1083 vs 1042）的帧级原语对比。

同一 NVDEC reader 同一批帧上，逐项对比：
  1. luma 提取：宿主 _nv12_batch_luma_full vs GPU extract_luma（逐位）
  2. sharp：宿主 g.std(axis=(1,2)) vs GPU analyze sums[:,0]（float 容差）
  3. cluster(win3)：宿主 _cluster_win3(bin) vs GPU sums[:,1]（整数）
  4. 相似合并判定：宿主 _segments_similar 语义链（binary mean/chg） vs
     GPU compare_pair（同帧对）

用法：python tools/_probe_d1_prim_diff.py [--video test5] [--batch 64]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VIDS = {
    "test5": (r"D:\Videos\racelog_test\test5.mp4", (843, 993, 948, 1025)),
    "test6": (r"D:\Videos\racelog_test\test6.mp4", (841, 994, 949, 1026)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5", choices=sorted(VIDS))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--th", type=int, default=116)
    args = ap.parse_args()

    from decord import VideoReader, gpu
    from video_ocr_engine._helpers import _ndarray_device_ptr
    from video_ocr_engine._gpu_kernels import GpuFrameAnalyzer
    from video_utils import _nv12_batch_luma_full
    from segmentation import _cluster_win3, similar_decision
    from cuda.bindings import runtime as cudart

    path, roi = VIDS[args.video]
    roi_hw = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)
    vr = VideoReader(path, ctx=gpu(0), output_format="yuv420", roi=roi_hw)
    vr.seek(0)

    an = GpuFrameAnalyzer()
    limited = True  # color_range=0 → limited（两管线同判据）

    # ── 1+2+3：批级原语 ──
    nds = vr.get_batch(list(range(0, args.batch)))
    crops = nds.asnumpy()
    base, shape = _ndarray_device_ptr(nds)
    rows, W = shape[1], shape[2]
    H = rows * 2 // 3
    B = len(crops)

    g_host = _nv12_batch_luma_full(crops, 0)
    ptr = an.extract_luma(base, B, H, W, limited)
    g_gpu = np.empty((B, H, W), dtype=np.uint8)
    cudart.cudaMemcpy(g_gpu.ctypes.data, ptr, B * H * W,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    same_luma = np.array_equal(g_host, g_gpu)
    ndiff = int((g_host != g_gpu).sum()) if not same_luma else 0
    print(f"[1] luma 逐位一致: {same_luma}"
          + ("" if same_luma else f"（差异像素 {ndiff}/{g_host.size}，"
             f"最大差 {int(np.abs(g_host.astype(int)-g_gpu.astype(int)).max())}）"))

    th = args.th
    sharp_host = g_host.astype(np.float32).std(axis=(1, 2))
    bin_host = g_host > th
    win3_host = np.fromiter(
        (_cluster_win3(bin_host[k] != bin_host[max(k - 1, 0)])
         for k in range(B)), dtype=np.float64, count=B)
    # 宿主管线的 win3 输入 = 本帧 bin 与前帧 bin 的异或（状态机语义见 feed）
    prev_buf = an._ensure_prev(B * H * W)
    # fill prev：prev 帧序列 = 第 k-1 帧（首批 prev=本帧自身，与两管线校准
    # 语义一致——此处仅对比原语，首批从第 1 帧起有意义的 diff）
    from video_ocr_engine.gpu.device import _gpu_fill_prev
    _gpu_fill_prev(an, prev_buf, ptr, B, H * W, ptr)
    sums = an.analyze_batch(ptr, prev_buf, B, H, W, float(th))
    sharp_gpu = np.asarray(sums)[:, 0].astype(np.float64)
    clu_gpu = np.asarray(sums)[:, 1].astype(np.float64)
    d = np.abs(sharp_host.astype(np.float64) - sharp_gpu)
    rel = d / np.maximum(sharp_host.astype(np.float64), 1e-9)
    print(f"[2] sharp: 最大绝对差 {d.max():.6f} 最大相对差 {rel.max():.2e} "
          f"（容差 1e-3 内算一致）")
    print(f"[3] cluster: 不一致帧数 {int((win3_host[1:] != clu_gpu[1:]).sum())}"
          f"/{B - 1}（跳过第 0 帧无前帧）")
    if not same_luma or rel.max() > 1e-3:
        bad = np.where((win3_host[1:] != clu_gpu[1:]))[0][:5] + 1
        print(f"    首几个不一致帧号(批内): {bad.tolist()}")

    # ── 4：合并判定语义（宿主 binary 路径 vs GPU compare_pair）──
    a, b = 10, 11  # 任取同批相邻两帧作代表帧对
    ga, gb = g_host[a], g_host[b]
    ba, bb = ga > th, gb > th
    from segmentation import _text_sep_binary
    xa, xb = _text_sep_binary(ga, th), _text_sep_binary(gb, th)
    diff = xa != xb
    n = ga.size
    mad = float(diff.sum()) / n
    chg = int((ba != bb).sum())
    host_verdict = similar_decision(mad, chg, 3.0, int(0.01 * n))
    ya = an.luma_into(int(ptr + a * H * W), an._ensure_prev(H * W), H, W,
                      limited)
    # luma_into 无返回值，手动给 dst
    dst = an._ensure_prev(H * W)
    an.luma_into(int(ptr + a * H * W), dst, H, W, limited)
    dst2 = an._ensure_prev(2 * H * W) if False else None
    mad2, chg2 = an.compare_pair(
        int(ptr + a * H * W), int(ptr + b * H * W), H, W, float(th), 1)
    mean2 = mad2 / n
    gpu_verdict = similar_decision(mean2, int(chg2), 3.0, int(0.01 * n))
    print(f"[4] merge 判定: host(mad={mad:.6f},chg={chg})={host_verdict} "
          f"gpu(mean={mean2:.6f},chg={int(chg2)})={gpu_verdict} "
          f"→ {'一致' if host_verdict == gpu_verdict else '**不一致**'}")
    vr.close()


if __name__ == "__main__":
    main()
