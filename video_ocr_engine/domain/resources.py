"""§8.6 r5 资源层：L1 相位边界差分 + L2 NVML 峰值因子采样。

两层成本结构不同，故意分开：

* **L1（std 档即开）——纯 stdlib + 已加载的 cuda 绑定，零新依赖（D7）**：
  进程 CPU 时间用 `time.process_time()`（含全部线程，Windows 下即
  GetProcessTimes，实测 ~0.05µs）；RSS 与磁盘读写走 kernel32/psapi 的
  ctypes 调用（实测 1–3µs）；线程数用 `threading.active_count()`。
  → **一次边界 ≈ 5µs**，每 run 8 个边界 ≈ 40µs，对 PI-15 无感。

  两条**实测踩过的坑**（写死在此，防止回退）：
  1. **不要用 `psutil.Process.threads()` / `num_threads()`**：本机实测
     **43ms / 5.1ms 每次**（Windows 逐线程开句柄），调一次就吃满 PI-15 的
     0.1% 预算（`_probe_run_setup_cost` 同批数据）。
  2. **不要在宿主路径上探 CUDA**：`cudaMemGetInfo` 本身 0.9µs，但**首次
     调用会隐式建主上下文（100ms 级）**。故只在该进程**已经**导入过
     cuda 绑定时才采 VRAM（GPU 运行必然已导入），否则记为不可用。

* **L2（full 档专属）**：设备侧峰值因子必须独立采样线程（NVML）。产品路径
  平时不建线程（B6）——线程只在显式 `start()`/`stop()` 之间存在。

诚实性口径：全部读数都是**本机自测**、跨机不可比；PCIe 本机不可直读，只能
由 counter 字节数 ÷ 相位墙钟**推算**，属 L3 推导区（本模块不产该字段）。
"""
from __future__ import annotations

import sys
import threading
import time

__all__ = ["ResourceProbe", "NvmlSampler", "nvml_handle", "gpu_identity"]

_WIN = sys.platform == "win32"

#: 进程级唯一 NVML 会话。None=未试，False=不可用，(nvml, handle)=可用。
#: L2 采样与环境指纹共用它：`nvmlInit_v2` 本机实测 ~20ms，每档各开一次纯属
#: 浪费；且采样线程收尾再 `nvmlShutdown` 会与并发的指纹读取互相踩引用计数。
_NVML: dict = {"state": None, "why": ""}


def nvml_handle():
    """返回 (nvml, device_handle)；不可用抛 OSError（原因记进 `_NVML["why"]`）。"""
    st = _NVML
    if st["state"] is False:
        raise OSError(st["why"] or "NVML 不可用")
    if st["state"]:
        return st["state"]
    import ctypes
    try:
        nvml = ctypes.CDLL("nvml.dll")
        if int(nvml.nvmlInit_v2()) != 0:
            raise OSError("nvmlInit_v2 rc!=0")
        h = ctypes.c_void_p()
        if int(nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h))) != 0:
            raise OSError("nvmlDeviceGetHandleByIndex_v2 rc!=0")
    except Exception as e:  # noqa: BLE001 一次性判定，不重复尝试
        st["state"], st["why"] = False, repr(e)[:120]
        raise OSError(st["why"])
    st["state"] = (nvml, h)
    return st["state"]


def gpu_identity():
    """(gpu_name, driver_version) 或 None——用 NVML 取代 nvidia-smi 子进程。

    本机实测：nvidia-smi 子进程 **43.1ms** vs NVML 首次全程 **20.8ms**，
    且免进程创建（部分环境会拦子进程/被杀软扫描）。指纹在 `environment()`
    里进程级缓存，但它是 **run 结束时的报告组装路径**上的成本，能省一半
    就省一半；两者都失败则记 `unavailable`（不用 None 冒充真值）。
    """
    try:
        nvml, h = nvml_handle()
    except OSError:
        return None
    try:
        import ctypes
        name = ctypes.create_string_buffer(96)
        if int(nvml.nvmlDeviceGetName(h, name, 96)) != 0:
            return None
        drv = ctypes.create_string_buffer(80)
        if int(nvml.nvmlSystemGetDriverVersion(drv, 80)) != 0:
            return None
        return (name.value.decode("utf-8", "replace"),
                drv.value.decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 老驱动缺入口 → 让调用方回退 nvidia-smi
        return None


class _HostCounters:
    """Windows 进程级资源计数器（ctypes，无第三方依赖）。

    每个来源可用与否**逐字记进 `sources`**，报告里必须能区分"真读到"与
    "该来源在本机不可用"——不允许用 None 冒充 0。
    """

    def __init__(self) -> None:
        self.sources: dict = {"cpu": "time.process_time"}
        self._get_rss = None
        self._get_io = None
        if not _WIN:
            self.sources.update(rss="unavailable:非 Windows",
                                disk="unavailable:非 Windows")
            return
        import ctypes
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

            class _IO(ctypes.Structure):
                _fields_ = [("read_ops", ctypes.c_ulonglong),
                            ("write_ops", ctypes.c_ulonglong),
                            ("other_ops", ctypes.c_ulonglong),
                            ("read_bytes", ctypes.c_ulonglong),
                            ("write_bytes", ctypes.c_ulonglong),
                            ("other_bytes", ctypes.c_ulonglong)]

            io = _IO()
            handle = ctypes.c_void_p(-1)   # GetCurrentProcess() 伪句柄

            def get_io(_h=handle, _io=io, _f=kernel32.GetProcessIoCounters):
                if _f(_h, ctypes.byref(_io)):
                    return int(_io.read_bytes), int(_io.write_bytes)
                return None

            self._get_io = get_io
            self.sources["disk"] = "kernel32.GetProcessIoCounters"
        except Exception as e:  # noqa: BLE001
            self.sources["disk"] = "unavailable:%s" % repr(e)[:60]
        try:
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_uint32),
                            ("page_faults", ctypes.c_uint32),
                            ("peak_ws", ctypes.c_size_t),
                            ("ws", ctypes.c_size_t),
                            ("_pad", ctypes.c_size_t * 6)]

            pmc = _PMC()
            pmc.cb = ctypes.sizeof(pmc)

            def get_rss(_p=psapi.GetProcessMemoryInfo, _h=handle, _m=pmc,
                        _cb=ctypes.c_uint(ctypes.sizeof(pmc))):
                if _p(_h, ctypes.byref(_m), _cb):
                    return int(_m.ws)
                return None

            self._get_rss = get_rss
            self.sources["rss"] = "psapi.GetProcessMemoryInfo"
        except Exception as e:  # noqa: BLE001
            self.sources["rss"] = "unavailable:%s" % repr(e)[:60]

    def sample(self) -> dict:
        """一次边界采样：全部字段都是 None 或真值（不可用≠0）。"""
        s = {"t": time.perf_counter(), "cpu": time.process_time(),
             "rss": None, "rb": None, "wb": None, "vram": None,
             "threads": threading.active_count()}
        if self._get_rss is not None:
            s["rss"] = self._get_rss()
        if self._get_io is not None:
            r = self._get_io()
            if r:
                s["rb"], s["wb"] = r
        v = _vram_bytes()
        if v:
            s["vram"] = v
        return s


_VRAM_STATE: dict = {"ok": False, "why": "not-probed"}


def _vram_bytes():
    """(used, total) 字节，或 None。

    只在**本进程已导入** cuda 绑定时尝试（`sys.modules` 查表 0.05µs，绝不
    触发 import），否则 `cudaMemGetInfo` 会**隐式建主上下文（100ms 级）**；
    宿主路径因此天然为 None。失败不置永久标记——GPU run 的首个边界可能
    早于 CUDA 初始化，必须每个边界再试一次（`cuCtxGetCurrent` ~0.2µs）。
    """
    mods = sys.modules
    if "cuda.bindings.driver" not in mods or "cuda.bindings.runtime" not in mods:
        _VRAM_STATE["why"] = "unavailable:cuda 绑定未加载（宿主路径）"
        return None
    try:
        from cuda.bindings import driver as cud
        from cuda.bindings import runtime as cudart
        rc, ctx = cud.cuCtxGetCurrent()
        if int(rc) != 0 or not ctx:
            _VRAM_STATE["why"] = "unavailable:无当前 CUDA 上下文"
            return None
        r, free, total = cudart.cudaMemGetInfo()
        if int(r) != 0:
            _VRAM_STATE["why"] = "unavailable:cudaMemGetInfo rc=%s" % r
            return None
        _VRAM_STATE.update(ok=True, why="cudaMemGetInfo")
        return int(total) - int(free), int(total)
    except Exception as e:  # noqa: BLE001
        _VRAM_STATE["why"] = "unavailable:%s" % repr(e)[:60]
        return None


class ResourceProbe:
    """L1 采样器：`checkpoint(phase)` 在粗相位边界调用，`per_phase()` 出差分。

    边界差分把"绝对读数"变成"这段时间花了多少"——每相位平均并行核数
    （Δcpu/Δwall）、RSS/VRAM 增量、磁盘读写速率，正是 §8.6 要的形态。
    """

    _MAX_PHASES = 32   # 防御性上限：粗相位数远小于此（不是 per-span）

    def __init__(self) -> None:
        self._counters = _HostCounters()
        self._rows: list = []
        self._lock = threading.Lock()

    @property
    def sources(self) -> dict:
        s = dict(self._counters.sources)
        s["vram"] = _VRAM_STATE["why"]      # 真值来源或不可用原因（不猜）
        s["threads"] = "threading.active_count"
        return s

    def checkpoint(self, phase: str) -> None:
        try:
            sample = self._counters.sample()
        except Exception:  # noqa: BLE001 遥测绝不让 run 失败
            return
        with self._lock:
            if len(self._rows) >= self._MAX_PHASES:
                return
            self._rows.append((phase, sample))

    def per_phase(self) -> dict:
        with self._lock:
            rows = list(self._rows)
        out: dict = {}
        for (a, sa), (_b, sb) in zip(rows, rows[1:]):
            name = _b if _b not in out else "%s→%s" % (a, _b)
            dt = float(sb["t"] - sa["t"])
            if dt <= 0:
                continue
            row: dict = {"wall": round(dt, 4), "threads": sb["threads"]}
            dcpu = float(sb["cpu"] - sa["cpu"])
            row["cores_avg"] = round(max(0.0, dcpu) / dt, 2)
            if sa["rss"] is not None and sb["rss"] is not None:
                row["rss_delta_mib"] = round(
                    (sb["rss"] - sa["rss"]) / 1048576.0, 1)
            if sa["rb"] is not None and sb["rb"] is not None:
                row["disk_read_mbps"] = round(
                    max(0, sb["rb"] - sa["rb"]) / dt / 1048576.0, 2)
                row["disk_write_mbps"] = round(
                    max(0, sb["wb"] - sa["wb"]) / dt / 1048576.0, 2)
            if sb["vram"] is not None:
                row["vram_used_mib"] = round(sb["vram"][0] / 1048576.0, 1)
            out[name] = row
        if rows:
            first = rows[0][1]
            base = {"cpu_total_s": round(first["cpu"], 3)}
            if first["rss"] is not None:
                base["rss_mib"] = round(first["rss"] / 1048576.0, 1)
            if first["vram"] is not None:
                base["vram_used_mib"] = round(first["vram"][0] / 1048576.0, 1)
            out["_at_first_checkpoint"] = base
        out["checkpoints"] = [r[0] for r in rows]
        return out


class NvmlSampler:
    """L2：NVML 低采样率设备侧探针（full 档专属；ctypes，不新增依赖）。

    只在显式 `start()`/`stop()` 之间存在；失败逐字记录降级链进
    `sources`（报告必须能区分 NVML 直读与"不可用"）。
    """

    def __init__(self, interval_s: float = 0.2, max_points: int = 600) -> None:
        self._interval = max(0.05, float(interval_s))
        self._max = int(max_points)
        self._pts: list = []       # (t, gpu%, decode%, vram_used_mib)
        self._fails = 0            # 采样失败/全空读数次数（摘要里可见，不静默）
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self._err = ""
        self.sources = "not-started"

    def _open(self):
        """共用进程级 NVML 会话（见 `nvml_handle`；不 shutdown，进程退出回收）。"""
        return nvml_handle()

    def start(self) -> None:
        import ctypes
        try:
            nvml, h = self._open()
        except Exception as e:  # noqa: BLE001 驱动无 NVML → 整体不可用
            self._err = repr(e)[:160]
            self.sources = "nvml_unavailable"
            return

        class _Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        class _Mem(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong),
                        ("used", ctypes.c_ulonglong)]

        util, mem = _Util(), _Mem()
        dec, dur = ctypes.c_uint(), ctypes.c_uint()
        rates = nvml.nvmlDeviceGetUtilizationRates
        dutil = nvml.nvmlDeviceGetDecoderUtilization
        minfo = nvml.nvmlDeviceGetMemoryInfo

        def loop():
            # 每 tick 独立 try（单次 NVML 调用失败只计 _fails，不杀线程）；
            # NVML 会话是进程级的（环境指纹可能还在用），故此处不 shutdown。
            while not self._stop.wait(self._interval):
                row = [time.perf_counter(), None, None, None]
                try:
                    if int(rates(h, ctypes.byref(util))) == 0:
                        row[1] = int(util.gpu)
                    # 第 4 参 pUUID 传 NULL（本机单卡，按索引取设备）
                    if int(dutil(h, ctypes.byref(dec),
                                 ctypes.byref(dur), None)) == 0:
                        row[2] = int(dec.value)
                    if int(minfo(h, ctypes.byref(mem))) == 0:
                        row[3] = round(int(mem.used) / 1048576.0, 1)
                except Exception:  # noqa: BLE001 单次调用失败计入 _fails
                    self._fails += 1
                    continue
                if any(v is not None for v in row[1:]):
                    self._pts.append(tuple(row))
                    if len(self._pts) > self._max:
                        del self._pts[0]      # 环形：只留最近 N 点
                else:
                    self._fails += 1          # 全空读数计入，摘要里可见

        self.sources = "nvml"
        self._th = threading.Thread(target=loop, daemon=True,
                                    name="nvml-sampler")
        self._th.start()

    def stop(self) -> dict | None:
        """关停带超时；**丢弃最后一帧不完整间隔**（半帧不可信）。"""
        if self._th is not None:
            self._stop.set()
            self._th.join(timeout=2.0)
            self._th = None
        pts = list(self._pts)
        fails = self._fails
        if len(pts) < 3:
            return {"sources": self.sources, "error": self._err or None,
                    "sample_failures": fails, "n": len(pts)}
        pts = pts[:-1]

        def summary(i):
            v = sorted(p[i] for p in pts if p[i] is not None)
            if not v:
                return None
            def q(f):
                return v[min(len(v) - 1, int(f * (len(v) - 1)))]
            return {"min": q(0.0), "p50": q(0.5), "p99": q(0.99),
                    "max": q(1.0), "n": len(v)}

        return {"sources": self.sources, "error": self._err or None,
                "interval_s": self._interval, "n": len(pts),
                "sample_failures": fails,
                "gpu_util_pct": summary(1), "nvdec_util_pct": summary(2),
                "vram_used_mib": summary(3)}
