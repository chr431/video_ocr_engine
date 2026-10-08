"""hard-window 位级对照矩阵（C-57 根治门禁，2026-09-28 发布轮）。

对三编码 × start×win 网格逐格跑三种读法（gray+ROI，引擎口径）：
  A. hybrid + set_decode_window（引擎时序：设窗 → seek_accurate → 读）
  B. hybrid 无窗（同范围）
  C. nvdec（gpu ctx）无窗（基线）
断言三者帧 md5 逐位一致 + A 的 hybrid_stats win_subs==0（窗替补哨兵
零触发）。任何一格不一致即缺陷（退出码非 0）——晚起点格（start=5000）
是 C-57 缺陷形态（修复前：尾帧 EOF 容错替补，md5 ≠ 基线 + win_subs>0）。

用法：
  python tools/_probe_window_matrix.py            # 全矩阵（3 码 × 4 格）
  python tools/_probe_window_matrix.py --quick    # 只跑 test5（h264）
  python tools/_probe_window_matrix.py --repeat 5 # 连续 5 轮（发布门禁
                                                  # 口径；单轮 12/12 在
                                                  # 2026-10-08 被证明可被
                                                  # 概率竞态 lucky-pass）
环境：RACELOG_VIDEO_DIR（默认 D:/Videos/racelog_test）；
DECORD_LIBRARY_PATH 指向待验 DLL（默认 D:/Repo/decord/build-dev）。
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_probe import video_dir  # noqa: E402  环境事实源（纪律检查 1）

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("DECORD_LIBRARY_PATH", "D:/Repo/decord/build-dev")

from decord import VideoReader, gpu, hybrid  # noqa: E402

VIDS = [
    ("test5", "h264", (843, 993, 948, 1025)),
    ("test6_hevc", "hevc", (841, 994, 949, 1026)),
    ("test6", "av1", (841, 994, 949, 1026)),
]
GRID = [(0, 1000), (0, 3000), (5000, 1000), (5000, 3000)]  # (start, win)


def read_hash(path, roi, start, n, ctx, use_win):
    vr = VideoReader(path, ctx=ctx, output_format="gray", roi=roi,
                     num_threads=32)
    if use_win:
        vr.set_decode_window(n)
    vr.seek_accurate(start)
    h = hashlib.md5()
    got = 0
    while got < n:
        e = min(got + 64, n)
        b = vr.get_batch(list(range(start + got, start + e)))
        h.update(b.asnumpy().tobytes())
        got += b.shape[0]
        del b
    st = vr.hybrid_stats() or {}
    vr.close()
    return h.hexdigest()[:16], got, st.get("win_subs")


def main():
    quick = "--quick" in sys.argv
    repeat = 1
    if "--repeat" in sys.argv:
        repeat = int(sys.argv[sys.argv.index("--repeat") + 1])
    vdir = video_dir()
    fails = 0
    for rep in range(repeat):
        if repeat > 1:
            print(f"── repeat {rep + 1}/{repeat} ──", flush=True)
        fails += run_matrix(quick, vdir)
    print(f"matrix: {'ALL PASS' if fails == 0 else f'{fails} 格失败'}"
          f"{' × ' + str(repeat) if repeat > 1 else ''}")
    return 1 if fails else 0


def run_matrix(quick, vdir):
    fails = 0
    for name, codec, roi in VIDS:
        if quick and name != "test5":
            continue
        path = os.path.join(vdir, f"{name}.mp4")
        if not os.path.isfile(path):
            print(f"SKIP {name}（无视频）")
            continue
        for start, win in GRID:
            # 越界格钳制到文件尾（顺带覆盖尾窗形态）：引擎契约本就
            # 保证 frame_end ≤ len(vr) 且设窗仅当 span < len
            vr_len = len(VideoReader(path))
            n_eff = min(win, vr_len - start)
            if n_eff <= 0:
                continue
            h_win, got_w, subs = read_hash(path, roi, start, n_eff,
                                           hybrid(0), True)
            h_now, got_n, _ = read_hash(path, roi, start, n_eff,
                                        hybrid(0), False)
            h_nv, got_v, _ = read_hash(path, roi, start, n_eff,
                                       gpu(0), False)
            ok = (h_win == h_now == h_nv and got_w == got_n == got_v == n_eff
                  and subs == 0)
            if not ok:
                fails += 1
            print(f"{'OK ' if ok else 'FAIL'} {codec:4s} start={start:5d}"
                  f" win={n_eff:4d}  hybrid+win={h_win} hybrid={h_now}"
                  f" nvdec={h_nv} got={got_w}/{got_n}/{got_v}"
                  f" win_subs={subs}")
    return fails


if __name__ == "__main__":
    sys.exit(main())
