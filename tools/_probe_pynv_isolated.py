"""PyNvVideoCodec 裸解码产能（独立进程口径，API 已按 2.2.3 校准）。

背景：`_probe_pynv_vs_decord.py` 发现 **decord 与 PyNvVideoCodec 无法共存
于同一进程**——只要先 `import decord`，PyNv 就
`ImportError: DLL load failed while importing _PyNvVideoCodec: 找不到指定的程序`
（先 import PyNv 则正常）。这是 DLL 符号冲突，不是我们的代码问题。

所以本探针把 PyNv 放到**独立子进程**量裸解码产能——**对它最有利的公平口径**
（独占进程、无 decord 干扰），再与本轮 `_probe_decode_ceiling.py` 的
decord NVDEC 口径对照。

API 校准（三个坑，实测得出）：
  ① 模块级 `nvc.CreateSimpleDecoder` 被同名 pybind 类**遮蔽**，直接调用报
     "incompatible function arguments"；必须用**包装类** `nvc.SimpleDecoder`。
  ② 包装类的方法名不是 `DecodeNextPacket`，而是
     `get_batch_frames` / `get_batch_frames_by_index` / `seek_to_index`。
  ③ CUDA 13 起 cudart 在 `bin\\x64\\`；PyNv 仅在 `CUDA_PATH` 存在时才
     `os.add_dll_directory`，本机默认为空 → 必须显式设置。

用法：
  python tools/_probe_pynv_isolated.py [--video test5.mp4] [--frames 3000]
      [--roi 843,993,948,1025] [--reps 2]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKER = r'''
import json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import PyNvVideoCodec as nvc

video = os.environ["PROBE_VIDEO"]
frames = int(os.environ["PROBE_FRAMES"])
reps = int(os.environ["PROBE_REPS"])
fetch = os.environ["PROBE_FETCH"] == "1"

ts, got, err, meta = [], 0, "", {}
for _ in range(reps):
    try:
        dec = nvc.SimpleDecoder(video, 0)
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e); break
    try:
        meta = dec.get_stream_metadata()
    except Exception:
        meta = {}
    got = 0
    try:
        dec.seek_to_index(0)
    except Exception:
        pass
    t0 = time.perf_counter()
    # 官方推荐取帧路径：get_batch_frames_by_index（批量索引，语义最接近
    # decord 的 get_batch(list(range(...)))）
    idx = 0
    while got < frames:
        n = min(64, frames - got)
        try:
            batch = dec.get_batch_frames_by_index(list(range(idx, idx + n)))
        except Exception as e:
            err = "%s: %s" % (type(e).__name__, e); break
        if not batch:
            break
        if fetch:
            for f in batch:
                _ = f.numpy()
        got += len(batch)
        idx += len(batch)
    ts.append(time.perf_counter() - t0)
    try:
        del dec
    except Exception:
        pass

best = min(ts) if ts else 0.0
print("PROBE_JSON " + json.dumps({
    "wall_s": round(best, 4), "frames": got,
    "fps": round(got / best, 1) if best and got else 0.0,
    "version": getattr(nvc, "__version__", "?"), "error": err,
    "meta_keys": list(meta)[:8] if isinstance(meta, dict) else []}))
'''


def find_cuda_path() -> str:
    """显式探测 CUDA Toolkit 目录（PyNv 依赖它注入 DLL 搜索路径）。

    ⚠️ CUDA 13 起 cudart 在 `bin\\x64\\`（不是 `bin\\`）。
    """
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if base.is_dir():
        cands = sorted([p for p in base.iterdir() if p.is_dir()],
                       key=lambda p: p.name, reverse=True)
        for c in cands:
            for sub in (c / "bin" / "x64", c / "bin"):
                if any(sub.glob("cudart64_*.dll")):
                    return str(c)
    return os.environ.get("CUDA_PATH", "")


def run(fetch: bool, video: str, frames: int, reps: int, cuda_path: str) -> dict:
    env = dict(os.environ)
    env.update({"PROBE_VIDEO": video, "PROBE_FRAMES": str(frames),
                "PROBE_REPS": str(reps), "PROBE_FETCH": "1" if fetch else "0"})
    if cuda_path:
        env["CUDA_PATH"] = cuda_path
    p = subprocess.run([sys.executable, "-c", WORKER], env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    return {"error": "worker 失败: %s" % (p.stderr[-700:] or p.stdout[-700:])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test5.mp4")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    video = str(Path(os.environ.get("RACELOG_VIDEO_DIR",
                                    r"D:\Videos\racelog_test")) / args.video)
    cuda_path = find_cuda_path()
    res: dict = {"video": args.video, "frames": args.frames,
                 "cuda_path": cuda_path}
    print("视频 %s  窗口 %d 帧" % (args.video, args.frames))
    print("CUDA_PATH = %s" % (cuda_path or "(未找到 → PyNv 大概率 import 失败)"))

    for tag, fetch in (("批量取帧（不 .numpy()）", False),
                       ("批量取帧 + .numpy()", True)):
        r = run(fetch, video, args.frames, args.reps, cuda_path)
        res["fetch" if fetch else "nofetch"] = r
        if r.get("error"):
            print("\n%-26s 失败：%s" % (tag, r["error"][:300]))
        else:
            print("\n%-26s %8.4fs  %8.1f fps（%d 帧，v%s）"
                  % (tag, r["wall_s"], r["fps"], r["frames"], r["version"]))
    print("\n⚠️ 口径差异（决定「能不能换」的不是这组数字）：")
    for g in ("全帧输出（无 ROI-first；本机 ROI 值 1.75×）",
              "无 gray 单通道输出（NV12/NATIVE 起步）",
              "无 NVDEC∥CPU hybrid 双解码（C-01/C-46）",
              "与 decord 同进程 DLL 冲突（见文件头）"):
        print("   ✗ %s" % g)

    out = ROOT / "bench" / "pynv_isolated.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
