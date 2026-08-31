"""IO 预算探针：量化一次提取中「磁盘 IO」与「PCIe / 主机传输」的确切字节数。

背景：批量多实例并发（README「多实例并发」表）里 2×NVDEC 只有 ~1.1×、
NVDEC∥CPU 有 ~1.4×，曾推测与 "IO 竞争" 有关。本探针不做事先归因，只把
各部分的 IO 量测出来，分三层：

  1. **进程逻辑读** `Process.io_counters().read_bytes`
     解码器/模型加载发起的读请求总量（**含页缓存命中**）。
  2. **物理磁盘读** `psutil.disk_io_counters()`（Windows 走
     IOCTL_DISK_PERFORMANCE，是真正落到磁盘的部分，系统级）。
     ⇒ 逻辑读 − 物理读 ≈ 页缓存命中量。两者谁大决定"磁盘是不是瓶颈"。
  3. **PCIe / 主机传输** hook `cudaMemcpy` / `cudaMemcpyAsync`，按
     cudaMemcpyKind 累加 H2D / D2H / D2D 字节数。

阶段切分：hook `FieldExtractor._prof_end`（ENGINE_PROFILE 的分相出口），
每一次分相结束时采一次快照，相邻快照相减即该阶段的增量。

冷 / 热：同一个 worker 进程内连跑多轮，第 1 轮大概率冷（页缓存未建立），
后续轮命中缓存 —— 冷热的差值就是"真实磁盘 IO"的上界。

用法：
  # 单实例（默认 2 轮：冷 + 热）
  python tools/_probe_io_budget.py --video X --roi a,b,c,d --backend auto

  # 两实例并发（两个不同视频，各起一个子进程以便分离进程级 IO）
  python tools/_probe_io_budget.py --video X --roi A --video2 Y --roi2 B \\
      --pairs auto,cpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:                                  # 中文输出在 GBK 控制台会 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PY = sys.executable
MB = 1024.0 * 1024.0

# cudaMemcpyKind 的整型值（cuda runtime API 固定）
_KIND = {0: "H2H", 1: "H2D", 2: "D2H", 3: "D2D", 4: "default"}


# ═══════════════════════ worker 侧 ═══════════════════════

class _Counter:
    """进程级 IO + PCIe 传输计数器（worker 内单例）。"""

    def __init__(self) -> None:
        import psutil
        self._psutil = psutil
        self._proc = psutil.Process()
        self.xfer = {"H2H": 0, "H2D": 0, "D2H": 0, "D2D": 0, "default": 0,
                     "?": 0}
        self.marks: list[tuple[str, dict]] = []
        self.t0 = time.perf_counter()
        self.base = self.snap()

    def snap(self) -> dict:
        c = self._proc.io_counters()
        return {"read_bytes": c.read_bytes, "read_count": c.read_count,
                "write_bytes": c.write_bytes,
                **{k: v for k, v in self.xfer.items()},
                "wall": time.perf_counter() - self.t0}

    def mark(self, name: str) -> None:
        """记一个分相点（带线程名）。

        分相增量按「同一线程内相邻两点之差」计算 —— 生产者线程（解码）与
        OCR 线程（推理）的标记是交错的，若按全局顺序相减会把对方的读量
        算到自己头上。
        """
        self.marks.append((_thread_name(), name, self.snap()))

    def reset(self) -> None:
        for k in self.xfer:
            self.xfer[k] = 0
        self.marks.clear()
        # 必须先重置 t0 再取 base，否则 base["wall"] 带着上一轮的秒数，
        # 新一轮首个标记会算出负的区间耗时（实测出现过 t=-8.96s）。
        self.t0 = time.perf_counter()
        self.base = self.snap()


_C: _Counter | None = None


def _thread_name() -> str:
    import threading
    t = threading.current_thread()
    return "Main" if t.name == "MainThread" else t.name


def _install_decode_hook() -> bool:
    """包装解码器的 get_batch —— 视频文件读取的唯一起点。

    不套这层的话，解码读到的字节会因为没有更近的标记点而被算进紧随其后
    的 `producer.q_put_block`（区间增量按相邻标记相减），归因会误导。
    """
    ok = False
    try:
        import decord.video_reader as vr_mod
        cls = getattr(vr_mod, "VideoReader", None)
        if cls is not None and hasattr(cls, "get_batch"):
            orig = cls.get_batch
            if not getattr(orig, "_io_probe", False):
                def f(self, *a, __orig=orig, **kw):
                    r = __orig(self, *a, **kw)
                    _C.mark("decode.get_batch")
                    return r
                f._io_probe = True
                cls.get_batch = f
            ok = True
    except Exception:
        pass
    try:
        import hybrid_decode as hd
        cls = getattr(hd, "HybridDecoder", None)
        if cls is not None and hasattr(cls, "get_batch"):
            orig = cls.get_batch
            if not getattr(orig, "_io_probe", False):
                def g(self, *a, __orig=orig, **kw):
                    r = __orig(self, *a, **kw)
                    _C.mark("hybrid.get_batch")
                    return r
                g._io_probe = True
                cls.get_batch = g
            ok = True
    except Exception:
        pass
    return ok


def _install_cuda_hooks() -> bool:
    """hook cudaMemcpy / cudaMemcpyAsync，按方向累加字节数。"""
    try:
        from cuda.bindings import runtime as cudart
    except Exception:
        return False
    for name in ("cudaMemcpy", "cudaMemcpyAsync"):
        orig = getattr(cudart, name, None)
        if orig is None:
            continue

        def _wrap(orig=orig):
            def f(*a, **kw):
                try:
                    count = a[2] if len(a) >= 3 else kw.get("count", 0)
                    kind = a[3] if len(a) >= 4 else kw.get("kind")
                    key = _KIND.get(int(kind), "?")
                    _C.xfer[key] = _C.xfer.get(key, 0) + int(count or 0)
                except Exception:
                    pass
                return orig(*a, **kw)
            return f
        setattr(cudart, name, _wrap())
    return True


def _install_prof_hook() -> None:
    """hook _prof_end：每个分相结束时采一次 IO 快照。"""
    import video_ocr_engine.extractor as ex_mod
    orig = ex_mod.FieldExtractor._prof_end

    def _hook(self, group: str, key: str, t0: float):
        try:
            _C.marks.append((_thread_name(), f"{group}.{key}", _C.snap()))
        except Exception:
            pass
        return orig(self, group, key, t0)
    ex_mod.FieldExtractor._prof_end = _hook


def _part_decode_only(args) -> dict:
    """隔离分量 1：只解码不 OCR —— 量「视频文件读取」的 IO。

    进程级 IO 计数器是全局的，多线程分相区间会重复计入同一份读量（实测
    同一 199MB 被 4 个线程的区间各吃掉一次）。所以分量必须**隔离跑**：
    这一段只做解码，读到的字节就是视频文件的读量。
    """
    import engine_config as _cfg
    from video_ocr_engine import FieldExtractor
    roi = tuple(int(x) for x in args.roi.split(","))
    ex = FieldExtractor(args.video, roi,
                        frame_start=args.frame_start,
                        frame_end=args.frame_start + args.frames,
                        decode_backend=args.backend, keep_crops=False)
    _C.reset()
    t0 = time.perf_counter()
    vr = ex._open_vr()
    t_open = time.perf_counter() - t0
    snap_open = _C.snap()
    roi4 = (roi[0], roi[1], roi[2] + 1, roi[3] + 1)
    stride = int(getattr(ex, "_sample_stride", 1) or 1)
    # 真实 extract 会把 end 夹到 len(vr)；这里必须同样夹，否则越界
    n_total = len(vr)
    frames = [f for f in range(args.frame_start,
                               args.frame_start + args.frames, stride)
              if f < n_total]
    batch = int(getattr(_cfg, "GPU_PIPELINE_DECODE_BATCH", 64) or 64)
    t1 = time.perf_counter()
    for i in range(0, len(frames), batch):
        vr.get_batch(frames[i:i + batch], roi=roi4)
    t_loop = time.perf_counter() - t1
    end = _C.snap()
    return {"part": "decode_only", "backend": args.backend,
            "n_frames": len(frames), "batch": batch,
            "t_open": t_open, "t_loop": t_loop,
            "read_open": snap_open["read_bytes"] - _C.base["read_bytes"],
            "read_loop": end["read_bytes"] - snap_open["read_bytes"],
            "read_total": end["read_bytes"] - _C.base["read_bytes"],
            "read_count": end["read_count"] - _C.base["read_count"],
            "H2D": end["H2D"] - _C.base["H2D"],
            "D2H": end["D2H"] - _C.base["D2H"],
            "D2D": end["D2D"] - _C.base["D2D"]}


def _part_engine_only(args) -> dict:
    """隔离分量 2：只加载 OCR 引擎 —— 量「模型/引擎加载」的 IO。"""
    from ocr_native import acquire_ocr_engine
    _C.reset()
    t0 = time.perf_counter()
    eng = acquire_ocr_engine(args.variant, "tensorrt", fill_width=args.fill_width)
    t_load = time.perf_counter() - t0
    end = _C.snap()
    return {"part": "engine_only", "variant": args.variant,
            "fill_width": args.fill_width, "t_load": t_load,
            "read_total": end["read_bytes"] - _C.base["read_bytes"],
            "read_count": end["read_count"] - _C.base["read_count"],
            "H2D": end["H2D"] - _C.base["H2D"],
            "D2H": end["D2H"] - _C.base["D2H"],
            "D2D": end["D2D"] - _C.base["D2D"]}


def _worker(args) -> int:
    global _C
    _C = _Counter()
    _install_cuda_hooks()
    _install_prof_hook()

    if args.part == "decode-only":
        print("@@JSON@@" + json.dumps(
            {"tag": "decode-only", "backend": args.backend,
             "video": os.path.basename(args.video),
             "parts": [_part_decode_only(args)]}, ensure_ascii=False))
        return 0
    if args.part == "engine-only":
        print("@@JSON@@" + json.dumps(
            {"tag": "engine-only", "backend": args.backend,
             "video": os.path.basename(args.video),
             "parts": [_part_engine_only(args)]}, ensure_ascii=False))
        return 0

    from video_ocr_engine import FieldExtractor
    roi = tuple(int(x) for x in args.roi.split(","))
    hook_ok = _install_decode_hook()
    rounds = []
    for i in range(args.rounds):
        # 每轮前后各采一次**系统级**物理磁盘计数：轮1 冷（页缓存未建立）、
        # 轮2 起命中缓存，两者之差即"真实落盘量"。该计数器是系统级的，会
        # 混入本机其它进程 IO，只在机器空闲时判读。
        disk0 = _sys_disk()
        _C.reset()
        ex = FieldExtractor(args.video, roi,
                            frame_start=args.frame_start,
                            frame_end=args.frame_start + args.frames,
                            decode_backend=args.backend,
                            ocr_backend=args.ocr_backend,
                            keep_crops=False)
        t0 = time.perf_counter()
        res = ex.extract()
        wall = time.perf_counter() - t0
        end = _C.snap()
        disk1 = _sys_disk()
        texts = [s.text for s in res.segments if s.text]

        # 分相聚合：按线程链相邻标记相减，同名阶段累加（q_put_block 之类
        # 每帧一次）。"read" 是「自同线程上一个标记以来的进程读增量」。
        agg: dict[str, dict] = {}
        prev: dict[str, dict] = {}
        for th, name, snap in _C.marks:
            key = f"{th}|{name}"
            p = prev.get(th, _C.base)
            d = agg.setdefault(
                key, {"thread": th, "name": name, "read": 0, "count": 0,
                      "H2D": 0, "D2H": 0, "D2D": 0, "wall": 0.0, "n": 0})
            d["read"] += max(0, snap["read_bytes"] - p["read_bytes"])
            d["count"] += max(0, snap["read_count"] - p["read_count"])
            d["H2D"] += max(0, snap["H2D"] - p["H2D"])
            d["D2H"] += max(0, snap["D2H"] - p["D2H"])
            d["D2D"] += max(0, snap["D2D"] - p["D2D"])
            d["wall"] += snap["wall"] - p["wall"]
            d["n"] += 1
            prev[th] = snap

        rounds.append({
            "round": i + 1,
            "wall": wall,
            "segs": len(res.segments),
            "uniq": len(set(texts)),
            "read_bytes": end["read_bytes"] - _C.base["read_bytes"],
            "read_count": end["read_count"] - _C.base["read_count"],
            "write_bytes": end["write_bytes"] - _C.base["write_bytes"],
            "H2D": end["H2D"] - _C.base["H2D"],
            "D2H": end["D2H"] - _C.base["D2H"],
            "D2D": end["D2D"] - _C.base["D2D"],
            "phys_bytes": disk1["read_bytes"] - disk0["read_bytes"],
            "phys_count": disk1["read_count"] - disk0["read_count"],
            # 注：psutil 的 read_time / busy_time 在本机不可用（实测累计
            # 585ms 且不随 140MB 读变化），不能当"等盘时间"。判断磁盘开销
            # 只能用冷/热墙钟差：轮1 冷（落盘）、轮2 起命中缓存。
            "stages": agg,
        })
        # 上一轮的后台线程可能在 extract() 返回后仍调 _prof_end / get_batch
        # （跨轮污染分相增量）：留一段静默期让它们收尾，再进入下一轮。
        time.sleep(0.3)
    print("@@JSON@@" + json.dumps(
        {"tag": args.tag, "backend": args.backend,
         "ocr_backend": args.ocr_backend, "decode_hook": hook_ok,
         "video": os.path.basename(args.video), "rounds": rounds},
        ensure_ascii=False))
    return 0


# ═══════════════════════ 父进程侧 ═══════════════════════

def _sys_disk():
    import psutil
    d = psutil.disk_io_counters()
    return {"read_bytes": d.read_bytes, "read_count": d.read_count,
            "read_time": getattr(d, "read_time", 0),
            "busy_time": getattr(d, "busy_time", 0)}


def _disk_bench(path: str, rounds_n: int = 3) -> dict:
    """磁盘/计数器校准：纯顺序读同一文件多轮，看逻辑读与物理读的差。

    用途有两个：
      1. 校准 `psutil.disk_io_counters()` 到底统不统计页缓存命中 —— 若第
         2 轮起物理读仍在增加（逻辑读必然增加），说明该计数器含缓存读，
         不能拿它当"真正落盘量"；
      2. 量出本机顺序读吞吐（MB/s），用于估算解码的磁盘占用率。
    """
    import psutil
    proc = psutil.Process()
    out = []
    chunk = 8 << 20
    for i in range(rounds_n):
        l0 = proc.io_counters().read_bytes
        d0 = _sys_disk()
        t0 = time.perf_counter()
        n = 0
        with open(path, "rb") as f:
            while True:
                if not f.read(chunk):
                    break
                n += chunk
        wall = time.perf_counter() - t0
        l1 = proc.io_counters().read_bytes
        d1 = _sys_disk()
        out.append({
            "round": i + 1, "wall": wall,
            "logic_mb": (l1 - l0) / MB,
            "phys_mb": (d1["read_bytes"] - d0["read_bytes"]) / MB,
            "read_time_ms": d1["read_time"] - d0["read_time"],
        })
    return {"file": os.path.basename(path), "size_mb": n / MB, "rounds": out}


def _show_bench(b: dict) -> None:
    print(f"\n===== 磁盘/计数器校准（纯顺序读 {b['file']}，"
          f"{b['size_mb']:.0f}MB × {len(b['rounds'])} 轮）=====")
    print("  判读：逻辑读每轮都应≈文件大小；若物理读也每轮≈文件大小，"
          "说明 psutil 的物理读计数含页缓存命中，不能当真实落盘量。")
    for r in b["rounds"]:
        rt = r["read_time_ms"]
        thr = (r["phys_mb"] / (rt / 1000.0)) if rt > 0 else float("nan")
        print(f"  轮{r['round']}: 墙钟={r['wall']:.3f}s 逻辑读={r['logic_mb']:.1f}MB "
              f"物理读={r['phys_mb']:.1f}MB 服务时间={rt}ms "
              f"→ 吞吐≈{thr:.0f}MB/s")


def _run_workers(jobs: list[list[str]]) -> tuple[list[dict], dict, float]:
    """并发起 worker 子进程，返回 (结果列表, 系统磁盘 delta, 总墙钟)。"""
    d0 = _sys_disk()
    t0 = time.perf_counter()
    procs = [subprocess.Popen([PY, __file__, "--worker", *j],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True,
                              encoding="utf-8", errors="replace")
             for j in jobs]
    outs = [p.communicate() for p in procs]
    wall = time.perf_counter() - t0
    d1 = _sys_disk()
    delta = {k: d1[k] - d0[k] for k in d0}
    res = []
    for (out, err), p in zip(outs, procs):
        line = [l for l in out.splitlines() if l.startswith("@@JSON@@")]
        if not line:
            print(f"[worker 失败 rc={p.returncode}]\n{err[:2000]}")
            continue
        res.append(json.loads(line[0][len("@@JSON@@"):]))
    return res, delta, wall


def _fmt(b: float) -> str:
    return f"{b / MB:9.1f}MB"


def _show(res: list[dict], delta: dict, wall: float, title: str) -> None:
    print(f"\n===== {title} =====")
    print(f"  总墙钟 {wall:.3f}s | 系统物理磁盘读 {_fmt(delta['read_bytes'])} "
          f"/ {delta['read_count']} 次")
    for r in res:
        for rd in r["rounds"]:
            cold = "冷" if rd["round"] == 1 else "热"
            print(f"  [{r['tag']}] {cold}轮{r['round'] if False else rd['round']}: "
                  f"墙钟={rd['wall']:.3f}s 段={rd['segs']} 唯一={rd['uniq']} | "
                  f"逻辑读={_fmt(rd['read_bytes'])} ({rd['read_count']}次) | "
                  f"落盘={_fmt(rd.get('phys_bytes', 0))} "
                  f"({rd.get('phys_count', 0)}次) | "
                  f"H2D={_fmt(rd['H2D'])} D2H={_fmt(rd['D2H'])} "
                  f"D2D={_fmt(rd['D2D'])}")
            print("        —— 分相（读=自同线程上一标记以来的增量）——")
            for _, d in sorted(rd["stages"].items(),
                               key=lambda kv: -kv[1]["read"])[:8]:
                print(f"        {d['thread'][:10]:10s} {d['name']:22s} "
                      f"读={_fmt(d['read'])} "
                      f"H2D={_fmt(d['H2D'])} D2H={_fmt(d['D2H'])} "
                      f"t={d['wall']:7.3f}s n={d['n']}")


def _concurrency(args, job) -> int:
    """并发收益实测：单独跑 A、单独跑 B、再一起跑，算 (tA+tB)/tAB。

    同时报告三种情形下的**落盘量**与**逻辑读**，用来判"并发退化是不是
    磁盘争抢造成的"：若并发时落盘量翻倍而墙钟也翻倍，才是磁盘瓶颈；若
    落盘量几乎不变（页缓存命中）而墙钟仍退化，瓶颈在别处。
    """
    print(f"\n########## 并发收益实测（各 {args.rounds} 轮）##########")
    b1, b2 = [x.strip() for x in args.pairs.split(",")]
    v2 = args.video2 or args.video
    r2 = args.roi2 or args.roi
    ja = job(args.video, args.roi, b1, args.frame_start, "A:" + b1)
    jb = job(v2, r2, b2, args.frame_start2, "B:" + b2)

    solo = {}
    for name, j in (("A", ja), ("B", jb)):
        res, delta, wall = _run_workers([j])
        solo[name] = (wall, delta, res)
        rd = res[0]["rounds"][-1]
        print(f"  单独 {name} ({res[0]['tag']}): 末轮墙钟={rd['wall']:.3f}s "
              f"总墙钟={wall:.3f}s 段={rd['segs']} 唯一={rd['uniq']} "
              f"逻辑读={_fmt(rd['read_bytes'])} 落盘={_fmt(delta['read_bytes'])}")

    res, delta, wall = _run_workers([ja, jb])
    print(f"  并发 A∥B: 总墙钟={wall:.3f}s 落盘={_fmt(delta['read_bytes'])} "
          f"/ {delta['read_count']} 次")
    for r in res:
        rd = r["rounds"][-1]
        print(f"        {r['tag']}: 末轮墙钟={rd['wall']:.3f}s 段={rd['segs']} "
              f"唯一={rd['uniq']} 逻辑读={_fmt(rd['read_bytes'])} "
              f"落盘={_fmt(rd.get('phys_bytes', 0))}")

    wa = solo["A"][0]
    wb = solo["B"][0]
    # 用"末轮墙钟"（稳态、缓存已热）算加速比，排除首轮冷启动差异
    ta = solo["A"][2][0]["rounds"][-1]["wall"]
    tb = solo["B"][2][0]["rounds"][-1]["wall"]
    tA = min(r["rounds"][-1]["wall"] for r in res
             if r["tag"].startswith("A:"))
    tB = min(r["rounds"][-1]["wall"] for r in res
             if r["tag"].startswith("B:"))
    print(f"\n  加速比（末轮稳态墙钟）: (tA+tB)/tAB = ({ta:.3f}+{tb:.3f})"
          f"/{max(tA, tB):.3f} = {(ta + tb) / max(tA, tB):.3f}×")
    print(f"  加速比（进程总墙钟）  : (tA+tB)/tAB = ({wa:.3f}+{wb:.3f})"
          f"/{wall:.3f} = {(wa + wb) / wall:.3f}×")
    print(f"  并发落盘 {_fmt(delta['read_bytes'])} vs 单独之和 "
          f"{_fmt(solo['A'][1]['read_bytes'] + solo['B'][1]['read_bytes'])}")
    return 0


def _show_parts(res: list[dict], delta: dict, wall: float, title: str) -> None:
    print(f"\n===== {title} =====")
    print(f"  总墙钟 {wall:.3f}s | 系统物理磁盘读 {_fmt(delta['read_bytes'])} "
          f"/ {delta['read_count']} 次")
    for r in res:
        for p in r.get("parts", []):
            if p["part"] == "decode_only":
                print(f"  [{r['tag']}] 解码 {p['n_frames']} 帧 batch={p['batch']}: "
                      f"打开={p['t_open']:.3f}s 遍历={p['t_loop']:.3f}s | "
                      f"打开读={_fmt(p['read_open'])} 遍历读={_fmt(p['read_loop'])} "
                      f"合计={_fmt(p['read_total'])} ({p['read_count']}次)")
            else:
                print(f"  [{r['tag']}] 引擎 {p['variant']} fw={p['fill_width']}: "
                      f"加载={p['t_load']:.3f}s | 读={_fmt(p['read_total'])} "
                      f"({p['read_count']}次)")
            print(f"        PCIe: H2D={_fmt(p['H2D'])} D2H={_fmt(p['D2H'])} "
                  f"D2D={_fmt(p['D2D'])}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--video2", help="并发时第二个实例的视频")
    ap.add_argument("--roi2", help="并发时第二个实例的 ROI")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--ocr-backend", default="auto")
    ap.add_argument("--pairs", help="并发配置，如 auto,cpu（逗号分隔两个后端）")
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-start2", type=int, default=0)
    ap.add_argument("--frames", type=int, default=8000)
    ap.add_argument("--rounds", type=int, default=2,
                    help="worker 内轮数（1=冷，后续=热）")
    ap.add_argument("--tag", default="w")
    ap.add_argument("--disk-bench", action="store_true",
                    help="只跑磁盘/计数器校准（读 --video 3 轮），不做提取")
    ap.add_argument("--part", default="extract",
                    choices=["extract", "decode-only", "engine-only"],
                    help="隔离分量：只跑解码 / 只加载引擎 / 完整提取")
    ap.add_argument("--concurrency", action="store_true",
                    help="配合 --pairs：先单独跑 A、B 再并发跑，算加速比")
    ap.add_argument("--variant", default="v6_small")
    ap.add_argument("--fill-width", type=int, default=0)
    args = ap.parse_args()

    if args.worker:
        return _worker(args)

    if args.disk_bench:
        _show_bench(_disk_bench(args.video))
        return 0

    common = ["--frames", str(args.frames), "--rounds", str(args.rounds),
              "--ocr-backend", args.ocr_backend, "--part", args.part,
              "--variant", args.variant, "--fill-width", str(args.fill_width)]

    def job(video, roi, backend, fstart, tag):
        return ["--video", video, "--roi", roi, "--backend", backend,
                "--frame-start", str(fstart), "--tag", tag, *common]

    if args.pairs:
        if args.concurrency:
            return _concurrency(args, job)
        b1, b2 = [x.strip() for x in args.pairs.split(",")]
        v2 = args.video2 or args.video
        r2 = args.roi2 or args.roi
        jobs = [job(args.video, args.roi, b1, args.frame_start, "A:" + b1),
                job(v2, r2, b2, args.frame_start2, "B:" + b2)]
        res, delta, wall = _run_workers(jobs)
        (_show_parts if args.part != "extract" else _show)(
            res, delta, wall, f"并发 {b1} ∥ {b2}")
        return 0

    jobs = [job(args.video, args.roi, args.backend, args.frame_start,
                args.backend)]
    res, delta, wall = _run_workers(jobs)
    (_show_parts if args.part != "extract" else _show)(
        res, delta, wall, f"单实例 {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
