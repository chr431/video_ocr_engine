"""性能基线/对比驱动探针（多轮深度优化会话用，证据链工具）。

覆盖矩阵：视频 × 解码后端(nvdec/cpu/hybrid) × 管线(GPU 全驻留 / 宿主)。
每配置 runs 次取中位；输出 wall + timing 分相(decode/ocr/ocr_tail) +
ENGINE_PROFILE 生产者相位 + 段数 + 唯一文本集 sha（跨配置一致性门禁）。

用法：
  python tools/_probe_perf_baseline.py --out tools/_ab_perf/baseline.json
      [--videos test6,test5] [--frames 3000] [--runs 2]
      [--label baseline]   # JSON 里标注本次测量身份

JSON 结构：{label, ts, machine, results:{video:{cfg:{...}}}, texts:{video:sha_ref}}
比较两份 JSON：python tools/_probe_perf_baseline.py --compare A.json --out B.json
（--out 为基线 A，--compare 为对比侧 B）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEOS = {
    # av1 精确 seek 慢 → 全部用 frames 窗口从 0 开始，避免 seek 口径
    "test6": dict(path=r"D:\Videos\racelog_test\test6.mp4",
                  roi=(841, 994, 949, 1026), codec="av1"),
    "test5": dict(path=r"D:\Videos\racelog_test\test5.mp4",
                  roi=(843, 993, 948, 1025), codec="h264"),
}
BACKENDS = ("nvdec", "cpu", "hybrid")
PIPELINES = {"gpu": {}, "host": {"GPU_PIPELINE": "0"}}


def _check_idle(max_pct=20.0):
    import psutil
    busy = psutil.cpu_percent(interval=1.5)
    return busy, busy <= max_pct


def _set_envs(envs):
    old = {}
    for k, v in envs.items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    return old


def _restore_envs(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run_cfg(video, roi, frames, backend, envs, runs):
    from video_ocr_engine import FieldExtractor
    old = _set_envs({"ENGINE_PROFILE": "1", **envs})
    walls, timings, profiles, segs_n, texts, metas = [], [], [], [], set(), []
    try:
        # 预热跑（不计入统计）：TRT context 首跑 / decord 首开有冷启动成本
        # （实测 run1 2.59s vs run2 1.81s），混入中位会污染代表值。
        wx = FieldExtractor(video, roi, frame_end=min(frames, 600),
                            decode_backend=backend, ocr_backend="auto",
                            keep_frames=True)
        wx.extract()
        del wx
        for i in range(runs):
            ex = FieldExtractor(video, roi, frame_end=frames,
                                decode_backend=backend, ocr_backend="auto",
                                keep_frames=True)
            t0 = time.perf_counter()
            res = ex.extract()
            wall = time.perf_counter() - t0
            walls.append(wall)
            timings.append(dict(res.timing))
            profiles.append({g: dict(d) for g, d in ex.profile.items()})
            segs_n.append(len(res.segments))
            for s in res.segments:
                if s.text:
                    texts.add(s.text)
            metas.append({"backend": res.meta["backend"],
                          "ocr": res.meta["ocr_backend"],
                          "degraded": res.meta["degraded_reason"]})
            print(f"    run{i+1}: {wall:.3f}s segs={len(res.segments)} "
                  f"timing={ {k: round(v, 3) for k, v in res.timing.items()} } "
                  f"meta={metas[-1]}", flush=True)
    finally:
        _restore_envs(old)
    # 代表值取最快一次的 timing（2 次运行 median=均值不在列表里；
    # timing 分相与 wall 必须来自同一次运行才有拆解意义）
    med_i = walls.index(min(walls))
    return dict(walls=[round(w, 4) for w in walls],
                wall_med=round(statistics.median(walls), 4),
                segs=segs_n, texts_n=len(texts),
                texts_sha=hashlib.sha1(
                    "\n".join(sorted(texts)).encode("utf-8")).hexdigest()[:12],
                timing_med={k: round(v, 4)
                            for k, v in timings[med_i].items()},
                profile=profiles[med_i], meta=metas[med_i])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", default="test6,test5")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--compare", default="")
    args = ap.parse_args()

    if args.compare:
        _compare(args.compare, args.out)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    refs = {}
    for vname in args.videos.split(","):
        v = VIDEOS[vname]
        results[vname] = {}
        print(f"== {vname} ({v['codec']}) roi={v['roi']} ==", flush=True)
        for pname, envs in PIPELINES.items():
            for bk in BACKENDS:
                cfg = f"{pname}/{bk}"
                busy, ok = _check_idle()
                if not ok:
                    print(f"  ⚠️ 机器不空闲({busy:.0f}%)，仍继续（记录在案）",
                          flush=True)
                print(f"  [{cfg}] idle={busy:.1f}%", flush=True)
                results[vname][cfg] = run_cfg(
                    v["path"], v["roi"], args.frames, bk, envs, args.runs)
        refs[vname] = results[vname]["gpu/nvdec"]["texts_sha"]
    doc = dict(label=args.label or out.stem, ts=time.strftime("%F %T"),
               frames=args.frames, runs=args.runs, results=results,
               refs=refs)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n→ {out}")

    # 摘要
    print(f"\n{'video':6} {'cfg':14} {'wall':>8} {'decode':>8} {'ocr':>7} "
          f"{'tail':>7} {'segs':>5} {'texts':>5} sha")
    for vname, cfgs in results.items():
        for cfg, d in cfgs.items():
            t = d["timing_med"]
            print(f"{vname:6} {cfg:14} {d['wall_med']:8.3f} "
                  f"{t.get('decode', 0):8.3f} {t.get('ocr', 0):7.3f} "
                  f"{t.get('ocr_tail', 0):7.3f} {d['segs'][0]:5} "
                  f"{d['texts_n']:5} {d['texts_sha']}"
                  f"{'  ⚠️文本漂移' if d['texts_sha'] != refs[vname] else ''}")


def _compare(a_path, b_path):
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))
    print(f"A={a.get('label')} ({a.get('ts')})  B={b.get('label')} "
          f"({b.get('ts')})")
    print(f"{'video':6} {'cfg':14} {'A wall':>8} {'B wall':>8} {'Δ':>8} "
          f"{'Δ%':>7}  文本")
    for vname in a["results"]:
        for cfg in a["results"][vname]:
            da = a["results"][vname][cfg]
            db = b["results"].get(vname, {}).get(cfg)
            if not db:
                print(f"{vname:6} {cfg:14} {'-':>8} （B 缺）")
                continue
            wa, wb = da["wall_med"], db["wall_med"]
            d = wb - wa
            # 文本一致性：B 的各配置 sha 都要对齐 B 的 gpu/nvdec 参照
            ok_b = db["texts_sha"] == b["refs"].get(vname)
            print(f"{vname:6} {cfg:14} {wa:8.3f} {wb:8.3f} {d:+8.3f} "
                  f"{d / wa * 100:+6.1f}%  "
                  f"{'✓' if ok_b else '⚠️漂移(' + db['texts_sha'] + ')'}")


if __name__ == "__main__":
    main()
