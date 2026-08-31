"""内存带宽探针：把「TRT/ONNX 混配退化 = 内存吞吐量争用」这个**推断**变成测量值。

## 背景

`CLAUDE.md`「探针定位的损耗来源」第 1 条断言混配退化的真因是内存子系统争抢，
关键证据是"8 进程纯内存流拷贝（**~100GB/s**）让 TRT 10.26ms/段"。其中
**~100GB/s 是对探针负载的标签、不是测量值**，"占满 DRAM"是推断。
本机（AMD Zen4 7945HX）PDH 无 DRAM 带宽计数器、AMD uProf 需装驱动，所以改用
**校准式测量**：不读硬件计数器，而是量"负载跑着的时候还剩多少带宽"。

## ⚠️ 关键陷阱：单进程 numpy 有 ~55 GB/s 的自身上限

实测（512MB/数组，copy kernel）：

| 配置 | 聚合带宽 |
| --- | --- |
| 1 进程 × 1 / 8 / 16 / 32 线程 | 49.4 / 56.4 / 54.8 / **55.6** GB/s |
| 2 进程 × 16 线程 | **107.7** GB/s |
| 4 进程 × 16 线程 | **122.6** GB/s |

**单进程加线程完全无效**（1 线程就到 49），必须多进程才能打满。
所以探针侧必须用 `--procs ≥2`，否则测到的"上限"是探针自己的天花板。

## 方法

1. kernel = `np.copyto(c, a)`（1 读 1 写，口径 2N）与 `np.add(a,b,out=c)`
   （2 读 1 写，口径 3N）双口径交叉验证，数组 512MB ≫ 64MB L3。
2. `--mode calib`：无负载下 N 进程 × T 线程 → **B_max**（机器可达带宽）。
3. `--mode bw`：拉起负载（trt / onnx / mixed）循环跑提取，待其进入稳态后
   用同样的 N×T 探针测 **B_with**。
   **负载消耗 ≈ B_max − B_with**（内存系统按时间片分享）。
4. `--mode wall`：只跑负载、不跑探针，测负载自身中位迭代墙钟（反向对照）。
5. `--mode selfcheck`：用已知带宽的合成负载验证差值法是否成立。

## 用法

```
python tools/_probe_mem_bw.py --mode calib --procs 2 --threads 16
python tools/_probe_mem_bw.py --mode bw --work trt
python tools/_probe_mem_bw.py --mode bw --work onnx
python tools/_probe_mem_bw.py --mode bw --work mixed
python tools/_probe_mem_bw.py --mode wall --work mixed --secs 45
python tools/_probe_mem_bw.py --mode selfcheck --load-procs 1
```
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:                                  # 中文输出在 GBK 控制台会 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PY = sys.executable
SCALE = os.path.join(HERE, "_probe_mp_scale.py")
GB = float(1 << 30)


# ═══════════════════════ 探针侧（多进程）═══════════════════════

def _probe_once(procs: int, threads: int, total_mb: float,
                kernel: str, window: float = 6.0) -> tuple[float, list[float]]:
    """在同一时间窗口内起 procs 个带宽进程，返回 (聚合 GB/s, 各进程 GB/s)。

    同步分两步，缺一不可（两个坑都实测踩过）：

    1. 收齐所有进程的 READY（分配+预热完成）后，通过 stdin 广播一个**未来的
       绝对时刻** `START <epoch>`，子进程忙等到该时刻才开窗。只做 READY 不
       够 —— 父进程是**串行** readline 的，进程数一多，第 1 个进程的窗口在
       父进程还在等第 N 个 READY 时就跑完了，各进程实际错开，聚合虚高
       （实测 24 进程读出 137 GB/s，而本机理论峰值只有 96）。
    2. 聚合带宽 = **Σbytes ÷ 所有窗口的并集跨度**，而不是 Σ(各进程自带带宽)。
       后者在部分重叠时静默虚高；并集口径不会。
    """
    ps = [subprocess.Popen(
        [PY, SCALE, "--threads", str(threads), "--total-mb", str(total_mb),
         "--kernel", kernel, "--window", str(window), "--ready"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
        for _ in range(procs)]
    for p in ps:                       # 等所有进程都完成分配/预热
        p.stdout.readline()
    deadline = time.time() + 1.5       # 留出广播到达所有子进程的余量
    for p in ps:
        p.stdin.write(f"START {deadline:.6f}\n")
        p.stdin.flush()
        p.stdin.close()
    outs = [p.communicate() for p in ps]

    vals: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for (o, e), p in zip(outs, ps):
        try:
            toks = (o or "").strip().splitlines()[-1].split()
            b, s, t = float(toks[0]), float(toks[1]), float(toks[2])
            vals.append(b / GB / (t - s))
            lo.append(s)
            hi.append(t)
        except Exception:
            print(f"  [探针进程失败 rc={p.returncode}] "
                  f"{(e or '').strip()[:300]}")
    if not vals:
        return 0.0, []
    span = max(hi) - min(lo)           # 并集跨度，不是任一单进程窗口
    # vals 已是 GiB/s，乘各自窗口还原成 GiB 后再除以并集跨度；这里**不能再除
    # 一次 GB**（第一版就多除了一次，聚合值变成 5e-08）。
    agg = sum(v * (hi[i] - lo[i]) for i, v in enumerate(vals)) / span
    return agg, vals


def _load_spawn(procs: int, threads: int, total_mb: float, kernel: str,
                secs: float) -> list[subprocess.Popen]:
    """拉起 procs 个**纯内存压力**进程（不参与统计，只负责占带宽）。"""
    return [subprocess.Popen(
        [PY, SCALE, "--threads", str(threads), "--total-mb", str(total_mb),
         "--kernel", kernel, "--loop", str(secs)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(procs)]


def _measure(procs: int, threads: int, total_mb: float, kernel: str,
             repeats: int = 2, window: float = 6.0) -> float:
    """取多次最大值：最小值受调度抖动影响，最大值最接近真实可达带宽。"""
    best = 0.0
    for _ in range(repeats):
        best = max(best, _probe_once(procs, threads, total_mb, kernel,
                                     window)[0])
    return best


# ═══════════════════════ 负载侧 ═══════════════════════

def _spawn_workloads(args, kind: str) -> tuple[list[subprocess.Popen],
                                               list[str]]:
    """返回 (进程列表, 各自的停止文件路径)。

    ⚠️ **Windows 陷阱**：`Popen.terminate()` 走的是 `TerminateProcess`，
    子进程被**瞬间杀死**，Python 的 `finally` / atexit 一行都不会执行 ——
    于是依赖 finally 打印 `@@JSON@@` 的 worker 永远交不出统计（第一版就是
    这么静默失败的：跑完 40s 一个迭代都没报）。
    正确做法是让子进程**优雅退出**：传一个停止文件路径，父进程建文件通知
    它收尾，`terminate()` 只作为超时兜底。
    """
    # job = (ocr_backend, video, decode_backend)
    # 注意 `decode_backend=auto` **不区分 OCR _backend，一律尝试 NVDEC**，
    # 所以 `mixed` 其实是两条流水线各开一个 NVDEC 会话。最早期的「完全互补」
    # 双流水线是 CPU 解码+ONNX ∥ NVDEC+TRT，对应下面的 `mixed_cpu`。
    if kind == "mixed":
        jobs = [("tensorrt", args.video, args.decode_backend),
                ("cpu", args.video2, args.decode_backend)]
    elif kind == "trt":
        jobs = [("tensorrt", args.video, args.decode_backend)]
    elif kind == "onnx":
        jobs = [("cpu", args.video, args.decode_backend)]
    elif kind == "trt2":                       # 双 TRT（对照：非混配）
        jobs = [("tensorrt", args.video, args.decode_backend),
                ("tensorrt", args.video2, args.decode_backend)]
    elif kind == "mixed_cpu":                  # ★ 早期互补设计：CPU解+ONNX ∥ NVDEC+TRT
        jobs = [("tensorrt", args.video, "nvdec"),
                ("cpu", args.video2, "cpu")]
    elif kind == "trt2_cpu":                   # 对照：把 NVDEC 降到 1 个，GPU 上下文仍 2 个
        jobs = [("tensorrt", args.video, "nvdec"),
                ("tensorrt", args.video2, "cpu")]
    elif kind == "onnx_cpu":                   # CPU 解码 + ONNX 单跑
        jobs = [("cpu", args.video, "cpu")]
    elif kind == "trt_cpu":                    # CPU 解码 + TRT 单跑
        jobs = [("tensorrt", args.video, "cpu")]
    else:
        return [], []
    ps, stops = [], []
    for i, (b, v, db) in enumerate(jobs):
        stop = os.path.join(HERE, f"_probe_stop_{os.getpid()}_{i}.flag")
        if os.path.exists(stop):
            os.remove(stop)
        stops.append(stop)
        ps.append(subprocess.Popen(
            [PY, __file__, "--mode", "work", "--video", v, "--roi", args.roi,
             "--frame-start", str(args.frame_start), "--frames",
             str(args.frames), "--ocr-backend", b,
             "--decode-backend", db, "--stride", str(args.stride),
             "--stop-file", stop],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace"))
    return ps, stops


def _harvest(ps: list[subprocess.Popen], stops: list[str]) -> list[dict]:
    """通知所有 worker 收尾，再收统计。"""
    out = []
    for s in stops:                            # 先全部通知，让它们并行收尾
        with open(s, "w") as f:
            f.write("stop")
    for p, s in zip(ps, stops):
        try:
            txt = p.communicate(timeout=180)[0] or ""
        except Exception:
            p.kill()
            txt = p.communicate()[0] or ""
        finally:
            try:
                os.remove(s)
            except OSError:
                pass
        for line in (txt or "").splitlines():
            if line.startswith("@@JSON@@"):
                out.append(json.loads(line[len("@@JSON@@"):]))
    return out


# ═══════════════════════ 模式 ═══════════════════════

def _mode_calib(args) -> int:
    print(f"===== B_max：无负载下的机器可达带宽（{args.procs} 进程 × "
          f"{args.threads} 线程，数组 {args.total_mb:.0f}MB，窗口 "
          f"{args.window:.0f}s）=====")
    res = {}
    for k in args.kernels:
        agg, per = _probe_once(args.procs, args.threads, args.total_mb, k,
                               args.window)
        print(f"  {k:6s}: 聚合 {agg:6.1f} GB/s  各进程 {[round(v, 1) for v in per]}")
        res[k] = agg
    print(f"\n  B_max 以 copy 口径为准 = {res.get('copy', 0):.1f} GB/s")
    print("@@JSON@@" + json.dumps({"mode": "calib", "procs": args.procs,
                                  "threads": args.threads, "bw": res}))
    return 0


def _mode_bw(args) -> int:
    kind = args.work
    print(f"===== 负载 [{kind}] 下的剩余带宽 =====")
    procs, stops = _spawn_workloads(args, kind)
    if procs:
        print(f"  已拉起 {len(procs)} 个提取进程，等待进入稳态 {args.warmup}s …")
        time.sleep(args.warmup)

    res = {}
    try:
        for k in args.kernels:
            agg, per = _probe_once(args.procs, args.threads, args.total_mb,
                                   k, args.window)
            print(f"  {k:6s}: 聚合 {agg:6.1f} GB/s  "
                  f"各进程 {[round(v, 1) for v in per]}")
            res[k] = agg
    finally:
        ws = _harvest(procs, stops)

    for w in ws:
        print(f"  [负载 {w['ocr_backend']}/解={w.get('used_decode', '?')}] "
              f"迭代 {w['iters']} 次，"
              f"中位墙钟 {w['wall_median']:.3f}s（最快 {w['wall_min']:.3f}s），"
              f"段数 {w['segs']}，唯一 {w.get('uniq', '?')}")
    print("@@JSON@@" + json.dumps(
        {"mode": "bw", "work": kind, "procs": args.procs,
         "threads": args.threads, "bw": res, "workers": ws}))
    return 0


def _mode_wall(args) -> int:
    print(f"===== 负载 [{args.work}] 单独跑（无探针干扰）=====")
    procs, stops = _spawn_workloads(args, args.work)
    if not procs:
        print("  --work 必须是 trt/onnx/mixed/trt2/mixed_cpu/trt2_cpu/onnx_cpu")
        return 2
    time.sleep(args.secs)
    for w in _harvest(procs, stops):
        print(f"  [{w['ocr_backend']}/解={w.get('used_decode', '?')}] "
              f"迭代 {w['iters']} 次，"
              f"中位墙钟 {w['wall_median']:.3f}s（最快 {w['wall_min']:.3f}s），"
              f"段数 {w['segs']}，唯一 {w.get('uniq', '?')}")
        print("@@JSON@@" + json.dumps(w))
    return 0


def _mode_selfcheck(args) -> int:
    """用已知带宽的合成负载验证「B_max − B_with = 负载消耗」是否成立。

    负载 = L 个进程 × T 线程满速跑同一个 kernel。它的真实带宽可以**单独测**
    （= 无探针干扰时 L 进程的聚合带宽）。把它和探针同时跑，看
    B_load_actual + B_with 是否 ≈ B_max；若明显超出，说明"带宽"这个量在
    本机不能被两个负载线性分享，差值法不可用。
    """
    L, T = args.load_procs, args.load_threads
    print("===== 差值法自检 =====")
    print(f"  合成负载：{L} 进程 × {T} 线程满速 {args.load_kernel}")

    print("\n  [1/3] 负载单独跑的真实带宽 …")
    load_alone = _probe_once(L, T, args.total_mb, args.load_kernel)[0]
    print(f"        B_load(单独) = {load_alone:.1f} GB/s")

    print("\n  [2/3] 探针单独跑（B_max）…")
    bmax = _measure(args.procs, args.threads, args.total_mb, args.load_kernel,
                    repeats=2)
    print(f"        B_max = {bmax:.1f} GB/s")

    print("\n  [3/3] 负载与探针同时跑 …")
    loaders = [subprocess.Popen(
        [PY, SCALE, "--threads", str(T), "--total-mb", str(args.total_mb),
         "--kernel", args.load_kernel, "--loop", str(args.loop_secs)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(L)]
    time.sleep(1.0)
    try:
        bwith = _measure(args.procs, args.threads, args.total_mb,
                         args.load_kernel, repeats=2)
    finally:
        for p in loaders:
            p.terminate()
        for p in loaders:
            try:
                p.wait(timeout=20)
            except Exception:
                p.kill()

    print(f"\n  结果：")
    print(f"    B_max                 = {bmax:6.1f} GB/s")
    print(f"    B_load(单独)          = {load_alone:6.1f} GB/s")
    print(f"    B_with(负载跑着时)    = {bwith:6.1f} GB/s")
    print(f"    差值法推出的负载消耗  = {bmax - bwith:6.1f} GB/s")
    err = (bmax - bwith) - load_alone
    print(f"    与真实值之差          = {err:+6.1f} GB/s "
          f"（{100.0 * err / max(load_alone, 1e-9):+.0f}%）")
    print(f"    B_with + B_load(单独) = {bwith + load_alone:6.1f} GB/s "
          f"（vs B_max {bmax:.1f}，比值 "
          f"{(bwith + load_alone) / max(bmax, 1e-9):.2f}）")
    print("\n  判读：若「差值法推出的消耗」与「B_load(单独)」同量级，差值法可用；"
          "若 B_with + B_load 明显超过 B_max，说明本机带宽不能被两个负载"
          "线性分享（单进程有各自的天花板），差值法只可用于**同口径横向比较**。")
    print("@@JSON@@" + json.dumps(
        {"mode": "selfcheck", "bmax": bmax, "load_alone": load_alone,
         "bwith": bwith}))
    return 0


def _mode_ts(args) -> int:
    """瞬时带宽时间序列：判断负载是「平滑吃带宽」还是「突发式吃带宽」。

    为什么需要它：`bw` 模式给的是 6 秒窗口的**平均**剩余带宽，会把突发负载
    平滑掉。ONNX 推理是批式的，平均 11 GB/s 完全可能是「峰值 40 GB/s、占空
    比 25%」。若真如此，用平均带宽去比对剂量-反应曲线就低估了它的杀伤力。
    本模式把窗口切成 0.2s 小片，逐片算探针拿到的带宽 —— 探针恒定满速，
    其**下跌部分**即同时段被负载抢走的量。
    """
    kind = args.work
    BIN = 0.25
    print(f"===== 瞬时带宽时间序列（负载 [{kind}]，窗口 {args.window:.0f}s，"
          f"切片 {args.slice:.2f}s）=====")
    procs, stops = _spawn_workloads(args, kind)
    if procs:
        time.sleep(args.warmup)

    ps = [subprocess.Popen(
        [PY, SCALE, "--threads", str(args.threads), "--total-mb",
         str(args.total_mb), "--kernel", args.kernels[0], "--window",
         str(args.window), "--ready", "--timeseries", "--slice",
         str(args.slice)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
        for _ in range(args.procs)]
    for p in ps:
        p.stdout.readline()
    deadline = time.time() + 1.5
    for p in ps:
        p.stdin.write(f"START {deadline:.6f}\n")
        p.stdin.flush()
        p.stdin.close()
    outs = [p.communicate()[0] or "" for p in ps]
    ws = _harvest(procs, stops)

    # 每个进程的切片 → 按中点归入 0.25s 桶
    perproc: list[dict[int, list[float]]] = []
    for o in outs:
        buckets: dict[int, list[float]] = {}
        for line in (o or "").splitlines():
            if not line.startswith("TS "):
                continue
            _, s, e, b = line.split()
            dur = float(e) - float(s)
            if dur <= 0:
                continue
            k = int(float(s) / BIN)
            buckets.setdefault(k, [0.0, 0.0])
            buckets[k][0] += float(b)
            buckets[k][1] += dur
        perproc.append({k: v[0] / v[1] / GB for k, v in buckets.items()})

    allk = sorted(set().union(*[set(d) for d in perproc])) if perproc else []
    series = []
    for k in allk:                     # 只统计所有进程都有数据的桶
        vals = [d[k] for d in perproc if k in d]
        if len(vals) == len(perproc):
            series.append(sum(vals))
    series.sort()
    n = len(series)
    if n:
        q = lambda p: series[min(n - 1, int(p * n))]          # noqa: E731
        print(f"  有效切片 {n} 个（{n * BIN:.1f}s 覆盖）")
        print(f"  瞬时聚合带宽 GB/s：min {series[0]:5.1f} | p05 {q(.05):5.1f} | "
              f"中位 {q(.50):5.1f} | p95 {q(.95):5.1f} | max {series[-1]:5.1f}")
        print(f"  被负载抢走（B_max {args.bmax:.1f} − 瞬时）："
              f"中位 {args.bmax - q(.50):5.1f} GB/s，"
              f"最猛 {args.bmax - series[0]:5.1f} GB/s")
        lo = sum(1 for v in series if v < args.bmax - 25) / n
        print(f"  瞬时带宽掉到 B_max−25 以下的时长占比：{100 * lo:.0f}%")
    for w in ws:
        print(f"  [负载 {w['ocr_backend']}/解={w.get('used_decode', '?')}] "
              f"迭代 {w['iters']} 次，"
              f"中位墙钟 {w['wall_median']:.3f}s，段数 {w['segs']}")
    print("@@JSON@@" + json.dumps(
        {"mode": "ts", "work": kind, "n": n,
         "series": [round(v, 1) for v in series]}))
    return 0


def _mode_work(args) -> int:
    """循环跑提取，供 bw/wall/dose 模式拉起。

    退出条件 = 出现 `--stop-file` 指定的文件（见 `_spawn_workloads` 里关于
    Windows `terminate()` 的说明）。每轮迭代结束后检查一次，因此停止延迟
    最多一个迭代；统计在 `finally` 里输出。
    """
    from video_ocr_engine import FieldExtractor
    roi = tuple(int(x) for x in args.roi.split(","))
    walls: list[float] = []
    segs = uniq = 0
    used_dec = "?"                             # 见下方 finally：异常路径也要有值
    try:
        while not (args.stop_file and os.path.exists(args.stop_file)):
            ex = FieldExtractor(args.video, roi,
                                frame_start=args.frame_start,
                                frame_end=args.frame_start + args.frames,
                                decode_backend=args.decode_backend,
                                ocr_backend=args.ocr_backend,
                                sample_stride=args.stride,
                                keep_crops=False)
            t0 = time.perf_counter()
            res = ex.extract()
            walls.append(time.perf_counter() - t0)
            segs = len(res.segments)
            uniq = len({s.text for s in res.segments if s.text})
            # 必须上报**实际**解码后端：nvdec 在打不开时会静默回退 CPU，
            # 若不看这一项，`mixed_cpu` 会退化成 `mixed` 而无人察觉。
            used_dec = str(getattr(ex, "_backend", "?"))
    except BaseException:
        pass
    finally:
        if walls:
            walls.sort()
            print("@@JSON@@" + json.dumps(
                {"ocr_backend": args.ocr_backend, "iters": len(walls),
                 "decode_backend": args.decode_backend,
                 "used_decode": used_dec,
                 "wall_median": walls[len(walls) // 2], "wall_min": walls[0],
                 "wall_max": walls[-1], "segs": segs, "uniq": uniq}))
    return 0


# ═══════════════════════ 判别性实验 ═══════════════════════

def _cpu_pct() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=0.8)
    except Exception:
        return float("nan")


def _hog_spawn(n: int, args) -> list[subprocess.Popen]:
    """拉起 n 个纯压力进程（不参与带宽统计，只负责占用 DRAM / CPU）。"""
    if n <= 0:
        return []
    return _load_spawn(n, args.hog_threads, args.hog_mb, args.hog_kernel,
                       args.loop_secs)


def _hogs_kill(hogs: list[subprocess.Popen]) -> None:
    for p in hogs:
        p.terminate()
    for p in hogs:
        try:
            p.wait(timeout=20)
        except Exception:
            p.kill()


def _mode_bwith(args) -> int:
    """不同 hog 档位下的**剩余带宽** B_with(n)，配 B_max 即可反推 hog 消耗。"""
    print(f"===== 剩余带宽 vs hog 档位（hog={args.hog_mb:.0f}MB/"
          f"{args.hog_kernel}，{args.hog_threads} 线程）=====")
    rows = []
    for n in args.hog_procs:
        hogs = _hog_spawn(n, args)
        time.sleep(2.0)
        try:
            # CPU 必须**在 hog 还活着时**采样。第一版放在 finally 之后采，
            # 那时 hog 已被 kill，读数恒为 ~5%，完全没意义。
            cpu = _cpu_pct()
            agg, per = _probe_once(args.procs, args.threads, args.total_mb,
                                   args.kernels[0], args.window)
        finally:
            _hogs_kill(hogs)
        print(f"  hog {n:3d} 进程: 探针拿到 {agg:6.1f} GB/s  "
              f"(各进程 {[round(v, 1) for v in per]})  系统CPU≈{cpu:.0f}%")
        rows.append({"n": n, "with": agg, "cpu": cpu})
    print("@@JSON@@" + json.dumps({"mode": "bwith", "hog_mb": args.hog_mb,
                                   "rows": rows}))
    return 0


def _mode_dose(args) -> int:
    """剂量-反应：给 OCR 负载加 n 个 hog，看负载墙钟怎么变。

    **这是判别"内存带宽争用" vs "CPU 核争用"的关键实验**：
    用 `--hog-mb 512`（DRAM 常驻）和 `--hog-mb 0.5`（L2 常驻、DRAM 流量≈0）
    各跑一遍，两者的**进程数/线程数完全相同、CPU 占用相近**，唯一差别是
    DRAM 流量。若 DRAM 档把负载打惨而 L2 档几乎无影响 → 带宽是因；
    若两档伤害相当 → 只是核争用。

    ⚠️ `--hog-mb` 是**每个数组**的大小，进程会分配 3 个。要做成 L2 常驻，
    必须让 `3 × hog_mb × 进程数 < 64MB L3`，且每线程工作集
    `2 × hog_mb / 线程数 ≲ 512KB < 1MB/核 的 L2`。第一版用了 4MB，
    8 进程就是 96MB 工作集，**溢出 L3 进了 DRAM**，于是"对照组"自己也在
    吃带宽（实测压掉 19.6 GB/s），对照失效。
    """
    label = "DRAM" if args.hog_mb >= 128 else (
        "L3" if args.hog_mb >= 16 else "L2")
    print(f"===== 剂量-反应：{args.work} 负载 + n 个 {label} 常驻 hog "
          f"({args.hog_mb:.0f}MB × {args.hog_threads} 线程) =====")
    rows = []
    for n in args.hog_procs:
        hogs = _hog_spawn(n, args)
        time.sleep(2.0)
        wl, wstops = _spawn_workloads(args, args.work)
        time.sleep(args.secs)
        cpu = _cpu_pct()          # 负载和 hog 都还在跑的时候采样
        ws = _harvest(wl, wstops)
        _hogs_kill(hogs)
        if not ws:
            print(f"  hog {n:3d}: 负载没跑出迭代")
            rows.append({"n": n, "wall": None})
            continue
        w = ws[0]
        print(f"  hog {n:3d} 进程: 中位墙钟 {w['wall_median']:7.3f}s "
              f"(最快 {w['wall_min']:7.3f}s, {w['iters']:3d} 次, "
              f"段数 {w['segs']}, 唯一 {w.get('uniq', '?')})  "
              f"系统CPU≈{cpu:.0f}%")
        rows.append({"n": n, "wall": w["wall_median"],
                     "wall_min": w["wall_min"], "iters": w["iters"],
                     "segs": w["segs"], "uniq": w.get("uniq"), "cpu": cpu})
    base = next((r["wall"] for r in rows if r["n"] == 0 and r["wall"]), None)
    if base:
        print(f"\n  相对无 hog 基线的退化（基线 {base:.3f}s）：")
        for r in rows:
            if r["wall"]:
                print(f"    hog {r['n']:3d}: {r['wall'] / base:5.2f}×  "
                      f"(+{100 * (r['wall'] / base - 1):5.1f}%)")
    print("@@JSON@@" + json.dumps({"mode": "dose", "work": args.work,
                                   "hog_mb": args.hog_mb,
                                   "hog_threads": args.hog_threads,
                                   "rows": rows}))
    return 0


# ═══════════════════════ CLI ═══════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="calib",
                    choices=["calib", "bw", "wall", "selfcheck", "work",
                             "bwith", "dose", "ts"])
    ap.add_argument("--work", default="none",
                    help="负载：none|trt|onnx|mixed|trt2|mixed_cpu|trt2_cpu|"
                         "onnx_cpu（*_cpu = 该条流水线强制 CPU 解码）")
    ap.add_argument("--procs", type=int, default=2,
                    help="探针进程数（必须 ≥2，单进程有 ~55GB/s 天花板）")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--total-mb", type=float, default=512.0)
    ap.add_argument("--kernels", nargs="+", default=["copy", "triad"])
    ap.add_argument("--window", type=float, default=6.0,
                    help="每个探针进程的施压窗口长度（秒）")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--secs", type=float, default=45.0)
    # selfcheck
    ap.add_argument("--load-procs", type=int, default=1)
    ap.add_argument("--load-threads", type=int, default=16)
    ap.add_argument("--load-kernel", default="copy", choices=["copy", "triad", "sum", "scale"])
    ap.add_argument("--loop-secs", type=float, default=60.0)
    ap.add_argument("--bmax", type=float, default=55.8,
                    help="本机 copy 口径带宽上限，用于换算「被抢走多少」")
    ap.add_argument("--slice", type=float, default=0.2,
                    help="ts 模式的切片长度（秒）")
    # hog（bwith / dose）
    ap.add_argument("--hog-procs", type=int, nargs="+", default=[0, 1, 2, 4],
                    help="hog 进程数档位（含 0 作为基线）")
    ap.add_argument("--hog-threads", type=int, default=4)
    ap.add_argument("--hog-mb", type=float, default=512.0,
                    help="hog 工作集：512=DRAM 常驻 / 64=L3 / 4=L2")
    ap.add_argument("--hog-kernel", default="copy", choices=["copy", "triad", "sum", "scale"])
    # worker
    ap.add_argument("--video", default=r"D:\Videos\racelog_test\test6.mp4")
    ap.add_argument("--video2", default=r"D:\Videos\racelog_test\test5.mp4")
    ap.add_argument("--roi", default="841,994,949,1026")
    ap.add_argument("--frame-start", type=int, default=139)
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--stride", type=int, default=1,
                    help="sample_stride；README 口径为 30000 帧 + stride 8")
    ap.add_argument("--decode-backend", default="auto")
    ap.add_argument("--ocr-backend", default="tensorrt")
    ap.add_argument("--stop-file", default="",
                    help="work 模式：该文件一出现就优雅收尾并输出统计")
    args = ap.parse_args()

    if args.mode == "calib":
        return _mode_calib(args)
    if args.mode == "bw":
        return _mode_bw(args)
    if args.mode == "wall":
        return _mode_wall(args)
    if args.mode == "selfcheck":
        return _mode_selfcheck(args)
    if args.mode == "ts":
        return _mode_ts(args)
    if args.mode == "bwith":
        return _mode_bwith(args)
    if args.mode == "dose":
        return _mode_dose(args)
    return _mode_work(args)


if __name__ == "__main__":
    raise SystemExit(main())
