"""CPU+ONNX（无 NVIDIA 显卡）路径：墙钟验证 + 线程预算调参。

## 为什么有这个探针
刚落地的「CPU 软解线程按 OCR 落点分档」只改了 **OCR 在 GPU（TRT）** 那一档；
OCR 在 CPU（ONNX）按设计保持原样。但分档判据是 `_ocr_on_gpu()` =
`ocr_backend != 'cpu'`，它**只看配置不看 TRT 是否真的可用**——
`ocr_backend='auto'` 且 TRT 初始化失败回退 ONNX 时，判据仍为 True，
解码会拿到 8~32 线程去和 ORT 抢核。这条路径必须实测。

## 三组实验
  G2-1 `decode=cpu, ocr=cpu`（无显卡用户的规范路径）：HEAD vs HEAD~1
  G2-2 `decode=cpu, ocr=auto` + `CUDA_VISIBLE_DEVICES=-1`（auto 回退 ONNX
       的风险路径）：HEAD vs HEAD~1
  G2-3 线程预算网格：DECODE_THREADS × OCR_THREADS × OCR_INSTANCES

正确性门：段数 + 唯一文本集合必须与基线逐位一致（文本内容本身不因线程数
变化，变了说明 ROI/取帧错位）。

用法：
  python tools/_probe_cpu_onnx.py --video X --roi a,b,c,d [--frames N]
      [--stride S] [--reps 3] [--grid] [--baseline-rev HEAD~1]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

WORKER = r"""
import os, sys, time, json
# 必须插 cwd 而非固定 ROOT：A/B 时子进程 cwd = 旧版本 worktree，
# 只有插 cwd 才能真的跑旧代码（插 ROOT 会静默比较两次新代码）。
sys.path.insert(0, os.getcwd())
os.environ['ENGINE_PROFILE'] = '1'
_aff = os.environ.get('PROBE_AFFINITY', '')
if _aff:
    import psutil
    psutil.Process().cpu_affinity(list(range(int(_aff))))
path, roi_s, nframes, stride, dbe, obe = sys.argv[1:7]
roi = tuple(int(x) for x in roi_s.split(','))
from video_ocr_engine import FieldExtractor
t_init = time.perf_counter()
ex = FieldExtractor(path, roi, frame_end=int(nframes), sample_stride=int(stride),
                    decode_backend=dbe, ocr_backend=obe)
t0 = time.perf_counter()
r = ex.extract()
wall = time.perf_counter() - t0
texts = sorted({s.text for s in r.segments if s.text})
prof = ex.profile.get('producer', {})
ocr = ex.profile.get('ocr', {})
print(json.dumps({
    'wall': round(wall, 3),
    'segs': len(r.segments),
    'uniq': len(texts),
    'sample': texts[:40],
    'timing': {k: round(v, 3) for k, v in ex.timing.items()},
    'producer': {k: round(v, 3) for k, v in
                 sorted(prof.items(), key=lambda kv: -kv[1])[:5]},
    'ocr': {k: round(v, 3) for k, v in
            sorted(ocr.items(), key=lambda kv: -kv[1])[:5]},
    'ocr_backend': r.meta.get('ocr_backend'),
    'backend': r.meta.get('backend'),
    # 勿用 config.env_int：A/B 时旧版本 engine_config 没有该函数会直接崩，
    # 导致"旧版本全部 ERR"的假象（首轮实测即踩此坑）。
    'decode_threads_env': int(os.environ.get('DECODE_THREADS', '0') or 0),
}))
"""


def run(video, roi, nframes, stride, dbe, obe, env=None, reps=1, cwd=None):
    e = dict(os.environ)
    e.pop("DECODE_THREADS", None)
    e.pop("OCR_THREADS", None)
    e.pop("OCR_INSTANCES", None)
    if env:
        e.update(env)
    worker = WORKER
    best = None
    for _ in range(reps):
        p = subprocess.run(
            [PY, "-c", worker, video, roi, str(nframes), str(stride), dbe, obe],
            capture_output=True, text=True, env=e, cwd=cwd or str(ROOT))
        out = (p.stdout or "").strip().splitlines()
        if p.returncode != 0 or not out:
            return {"err": (p.stderr or "").strip()[-400:]}
        d = json.loads(out[-1])
        if best is None or d["wall"] < best["wall"]:
            best = d
    return best


def make_worktree(rev: str) -> Path | None:
    """把 rev 检出到临时 worktree，用于 HEAD vs 旧版本 A/B。"""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="wt_")) / "old"
    p = subprocess.run(["git", "-C", str(ROOT), "worktree", "add",
                        "--detach", str(d), rev],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  worktree 失败：{p.stderr.strip()[-200:]}")
        return None
    # ocr_engines/ 被 gitignore（TRT 引擎缓存），worktree 里没有 → 拷过去，
    # 否则旧版本的 TRT 用例会走冷构建（污染 A/B 计时）。
    import shutil
    src = ROOT / "ocr_engines"
    if src.is_dir():
        shutil.copytree(src, d / "ocr_engines", dirs_exist_ok=True)
    return d


def drop_worktree(d: Path | None) -> None:
    if not d:
        return
    subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force",
                    str(d)], capture_output=True, text=True)


def show(tag: str, r: dict, ref: dict | None = None) -> None:
    if "err" in r:
        print(f"  {tag:44s} ERR {r['err'][:160]}")
        return
    ok = ""
    if ref and "err" not in ref:
        same = (r["segs"] == ref["segs"] and r["uniq"] == ref["uniq"]
                and r["sample"] == ref["sample"])
        ok = f"  一致={'OK' if same else 'DIFF!'}"
    print(f"  {tag:44s} {r['wall']:7.3f}s  段={r['segs']:4d} "
          f"唯一={r['uniq']:4d}  ocr={r.get('ocr_backend')}{ok}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test5.mp4")
    ap.add_argument("--roi", default="843,993,948,1025")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baseline-rev", default="HEAD~1")
    ap.add_argument("--grid", action="store_true", help="跑 G2-3 线程网格")
    ap.add_argument("--dcd-sweep", default="",
                    help="只扫解码线程数，逗号分隔，如 8,12,16,24,32")
    ap.add_argument("--ocr-threads", type=int, default=0)
    ap.add_argument("--affinity", type=int, default=0,
                    help="绑到前 N 个逻辑核（模拟弱 CPU / 少核机）")
    ap.add_argument("--skip-ab", action="store_true", help="跳过 HEAD vs 旧版 A/B")
    a = ap.parse_args()

    roi = ",".join(str(x) for x in (int(x) for x in a.roi.split(",")))
    print(f"=== CPU+ONNX 路径验证：{Path(a.video).name} "
          f"{a.frames}帧 stride={a.stride} reps={a.reps} ===")

    report: dict = {"video": a.video, "frames": a.frames,
                    "stride": a.stride, "reps": a.reps, "groups": {}}

    # 基线指纹（正确性门）：CPU+ONNX 全默认
    print("\n[基线指纹] decode=cpu ocr=cpu（HEAD）")
    base = run(a.video, roi, a.frames, a.stride, "cpu", "cpu", reps=a.reps)
    show("cpu+onnx (HEAD)", base)
    if "err" in base:
        return 2
    report["groups"]["baseline"] = base

    # ── G2-1 / G2-2：HEAD vs 旧版本 ──
    if not a.skip_ab:
        wt = make_worktree(a.baseline_rev)
        print(f"\n[G2 A/B] 旧版本 = {a.baseline_rev}"
              f"{'' if wt else '（worktree 不可用，跳过）'}")
        cases = [
            ("G2-1 cpu+onnx   (ocr=cpu)", "cpu", "cpu", {}),
            ("G2-2 auto→ONNX 回退 (CUDA 不可见)", "cpu", "auto",
             {"CUDA_VISIBLE_DEVICES": "-1"}),
            ("G2-2r TRT 对照 (CUDA 可见)", "cpu", "auto", {}),
        ]
        for tag, dbe, obe, env in cases:
            head = run(a.video, roi, a.frames, a.stride, dbe, obe,
                       env=env, reps=a.reps)
            old = (run(a.video, roi, a.frames, a.stride, dbe, obe,
                       env=env, reps=a.reps, cwd=str(wt)) if wt else
                   {"err": "no worktree"})
            print(f"\n  {tag}")
            show("    HEAD", head, base)
            show(f"    {a.baseline_rev}", old, base)
            if "err" not in head and "err" not in old:
                d = head["wall"] - old["wall"]
                pct = d / old["wall"] * 100 if old["wall"] else 0
                print(f"    Δ = {d:+.3f}s ({pct:+.1f}%)  "
                      f"{'变慢' if d > 0.05 else '变快' if d < -0.05 else '持平'}")
            report["groups"][tag] = {"head": head, "old": old}
        drop_worktree(wt)

    # ── G2-3：线程预算网格 ──
    if a.dcd_sweep:
        print("\n[G2-3s] 解码线程聚焦扫描（decode=cpu ocr=cpu）")
        results = []
        for dcd in (int(x) for x in a.dcd_sweep.split(",") if x.strip()):
            env = {"DECODE_THREADS": str(dcd)}
            if a.ocr_threads:
                env["OCR_THREADS"] = str(a.ocr_threads)
            if a.affinity:
                env["PROBE_AFFINITY"] = str(a.affinity)
            r = run(a.video, roi, a.frames, a.stride, "cpu", "cpu",
                    env=env, reps=a.reps)
            show(f"dcd={dcd}", r, base)
            results.append({"env": env, **r})
        report["groups"]["dcd_sweep"] = results
        best = min((r for r in results if "err" not in r),
                   key=lambda r: r["wall"], default=None)
        if best:
            print(f"\n  最优：{best['env']} → {best['wall']:.3f}s "
                  f"（基线 {base['wall']:.3f}s，"
                  f"{(best['wall']/base['wall']-1)*100:+.1f}%）")

    if a.grid:
        print("\n[G2-3] 线程预算网格（decode=cpu ocr=cpu）")
        grid = []
        for dcd in (0, 8, 12, 16, 24):
            for ocrt in (0, 8, 16):
                for inst in ("", "0", "1"):
                    if dcd == 0 and ocrt == 0 and inst == "":
                        continue
                    grid.append((dcd, ocrt, inst))
        # 精简：先扫解码线程（OCR 默认），再扫 OCR 线程（解码默认），
        # 最后交叉几组
        grid = [
            (0, 0, ""), (8, 0, ""), (12, 0, ""), (16, 0, ""), (24, 0, ""),
            (0, 8, ""), (0, 16, ""), (0, 24, ""),
            (0, 0, "0"), (0, 0, "1"),
            (12, 8, ""), (12, 16, ""), (16, 8, ""), (16, 16, "0"),
        ]
        results = []
        for dcd, ocrt, inst in grid:
            env = {}
            if dcd:
                env["DECODE_THREADS"] = str(dcd)
            if ocrt:
                env["OCR_THREADS"] = str(ocrt)
            if inst:
                env["OCR_INSTANCES"] = inst
            r = run(a.video, roi, a.frames, a.stride, "cpu", "cpu",
                    env=env, reps=a.reps)
            tag = (f"dcd={dcd or 'auto'} ocrT={ocrt or 'auto'} "
                   f"inst={inst or 'auto'}")
            show(tag, r, base)
            results.append({"env": env, **r})
        report["groups"]["grid"] = results
        best = min((r for r in results if "err" not in r),
                   key=lambda r: r["wall"], default=None)
        if best:
            print(f"\n  最优：{best['env']} → {best['wall']:.3f}s "
                  f"（基线 {base['wall']:.3f}s，"
                  f"{(best['wall']/base['wall']-1)*100:+.1f}%）")

    out = Path(__file__).with_name("_probe_cpu_onnx.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n明细落盘：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
