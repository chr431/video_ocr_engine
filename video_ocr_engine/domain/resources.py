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

__all__ = ["ResourceProbe", "NvmlSampler", "nvml_handle", "gpu_identity",
           "register_thread", "thread_ledger"]

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


# ── v8 线程周期账本（full 档）：引擎自有线程的占空归因 ────────────────
#
# QueryProcessCycleTime 是全进程无标签求和（C-63 的结构性盲区：并发环节
# 不可分）。引擎自有线程（consumer/ocr/infer*/producer/drain）经
# OpenThread 按 TID 打开句柄后可逐线程 QueryThreadCycleTime——被换下
# 等待的线程不积累周期，单线程 duty = Δcycles/(f_run×Δwall) 即占空比。
# 外来线程群（decord 解码池/OpenVINO TBB）无自报面，残差=进程总量−
# Σ已注册线程（跨线程归因的下界，进一步细分须 fork 侧报数或 ETW）。
_THREAD_LEDGER: "ThreadLedger | None" = None
_THREAD_LEDGER_TRIED = False


class ThreadLedger:
    """进程级「名字→线程句柄」表 + 周期采样。

    注册发生在创建点（只记 ident，零系统调用）；句柄**惰性打开**——
    首次采样（full 档 run 的相位边界）才 OpenThread，std/off 档零成本
    （AGENTS：遥测新增只进 full 档）。线程已死时 OpenThread/查询失败
    → 该名字从表里除名（缺席≠0）。TID 复用风险：引擎线程生命周期=run
    生命周期且每 run 重新 register（同名覆盖 ident+句柄），风险窗只在于
    不重新 register 的持久线程——现役集合无此形态。
    """

    def __init__(self) -> None:
        import ctypes
        # 私有 WinDLL 实例 + 显式原型：windll 是进程级共享缓存，函数对象
        # 上的 argtypes 会被各使用方互相覆盖；句柄原型必须显式（与本文件
        # GetCurrentProcess 伪句柄截断坑同族——HANDLE 走 c_void_p）。
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.OpenThread.restype = ctypes.c_void_p
        self._k32.OpenThread.argtypes = (ctypes.c_uint32, ctypes.c_int,
                                         ctypes.c_uint32)
        self._k32.QueryThreadCycleTime.restype = ctypes.c_int
        self._k32.QueryThreadCycleTime.argtypes = (ctypes.c_void_p,
                                                   ctypes.c_void_p)
        self._idents: dict = {}
        self._handles: dict = {}
        self._lock = threading.Lock()

    def register(self, name: str, ident: int) -> None:
        """线程创建点登记/刷新（同名覆盖——跨 run 换代线程）。"""
        with self._lock:
            if self._idents.get(name) != ident:
                self._idents[name] = ident
                self._handles.pop(name, None)   # 旧句柄作废

    def sample(self) -> dict:
        """名字 → 周期数；打不开/查不到的名字静默除名（返回里缺席）。"""
        out: dict = {}
        import ctypes
        with self._lock:
            items = list(self._idents.items())
            handles = self._handles
        for name, ident in items:
            h = handles.get(name)
            if h is None:
                # 权限怪癖（2026-10-09 本机 Win11 实证矩阵）：按 0x0400/
                # 0x1000 打开成功但 QueryThreadCycleTime 报 ACCESS_DENIED=5，
                # 唯 THREAD_ALL_ACCESS=0x1FFFFF 句柄可查（自家进程线程恒可
                # 全权打开）；受限环境全权失败时回退试 0x0400，再失败即缺席。
                h = (self._k32.OpenThread(0x1FFFFF, False, ident)
                     or self._k32.OpenThread(0x0400, False, ident))
                if not h:
                    continue   # 线程已死/权限不足：缺席（≠0）
                with self._lock:
                    handles[name] = h
            cyc = ctypes.c_ulonglong()
            if self._k32.QueryThreadCycleTime(h, ctypes.byref(cyc)):
                out[name] = int(cyc.value)
            else:
                with self._lock:
                    handles.pop(name, None)
        return out

    def sample_all(self) -> dict:
        """TID → 周期数：进程**全部**线程（含外来池：decord/OMP/TBB）。

        Toolhelp32 线程快照枚举 + 逐线程开句柄查询后即关（不缓存——
        外来线程生命周期不由我们管，TID 复用风险靠"只在差分里用个体、
        结论只下在簇级"消化）。成本 ≈ 快照 ~1ms + 每线程 ~µs（full 档
        每相位边界一次）。命名线程的值与 `sample()` 同源可互校。
        """
        import ctypes
        from ctypes import wintypes as _wt

        class _TE32(ctypes.Structure):
            _fields_ = [("dwSize", _wt.DWORD), ("cntUsage", _wt.DWORD),
                        ("th32ThreadID", _wt.DWORD),
                        ("th32OwnerProcessID", _wt.DWORD),
                        ("tpBasePri", ctypes.c_long),
                        ("tpDeltaPri", ctypes.c_long),
                        ("dwFlags", _wt.DWORD)]

        k32 = self._k32
        k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        k32.CreateToolhelp32Snapshot.argtypes = (_wt.DWORD, _wt.DWORD)
        k32.Thread32First.restype = ctypes.c_int
        k32.Thread32First.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        k32.Thread32Next.restype = ctypes.c_int
        k32.Thread32Next.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        k32.CloseHandle.restype = ctypes.c_int
        k32.CloseHandle.argtypes = (ctypes.c_void_p,)
        TH32CS_SNAPTHREAD = 0x4
        pid = k32.GetCurrentProcessId()
        snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if not snap or snap == ctypes.c_void_p(-1).value:
            return {}
        tids: list = []
        try:
            e = _TE32()
            e.dwSize = ctypes.sizeof(_TE32)
            it = ctypes.byref(e)
            ok = k32.Thread32First(snap, it)
            while ok:
                if e.th32OwnerProcessID == pid:
                    tids.append(int(e.th32ThreadID))
                ok = k32.Thread32Next(snap, it)
        finally:
            k32.CloseHandle(snap)
        out: dict = {}
        cyc = ctypes.c_ulonglong()
        for tid in tids:
            h = k32.OpenThread(0x1FFFFF, False, tid)
            if not h:
                continue
            if k32.QueryThreadCycleTime(h, ctypes.byref(cyc)):
                out[tid] = int(cyc.value)
            k32.CloseHandle(h)
        return out


def thread_ledger() -> "ThreadLedger | None":
    """进程级单例；非 Windows 恒 None（register_thread 同步降级为 no-op）。"""
    global _THREAD_LEDGER, _THREAD_LEDGER_TRIED
    if not _WIN:
        return None
    if _THREAD_LEDGER is None and not _THREAD_LEDGER_TRIED:
        _THREAD_LEDGER_TRIED = True
        _THREAD_LEDGER = ThreadLedger()
    return _THREAD_LEDGER


def register_thread(name: str, t=None) -> None:
    """线程登记的统一入口（非 Windows/无 ident 静默 no-op）。

    t=None 时登记当前线程（consumer=调用方线程用这种）。
    """
    led = thread_ledger()
    if led is None:
        return
    try:
        ident = t.ident if t is not None else threading.current_thread().ident
    except Exception:  # noqa: BLE001 线程对象尚未 start：无从登记
        return
    if ident:
        led.register(name, int(ident))


class _HostCounters:
    """Windows 进程级资源计数器（ctypes，无第三方依赖）。

    每个来源可用与否**逐字记进 `sources`**，报告里必须能区分"真读到"与
    "该来源在本机不可用"——不允许用 None 冒充 0。
    """

    def __init__(self) -> None:
        self.sources: dict = {"cpu": "time.process_time"}
        self._get_rss = None
        self._get_io = None
        self._get_cycles = None
        if not _WIN:
            self.sources.update(rss="unavailable:非 Windows",
                                disk="unavailable:非 Windows",
                                cycles="unavailable:非 Windows")
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
        try:
            # P2a（2026-09-17 §8.4 落地）：process_time 有 15.625ms（1/64s）
            # tick 量化，短相位（<0.5s）的 dcpu 只有 1~2 tick，cores_avg 是
            # ±49%~100% 量化伪影；cycle 计数分辨率 ≈ 单周期且语义恰是
            # "CPU 时间"（sleep 不增、多线程求和，bench/cycle_quant.json）。
            qpc = kernel32.QueryProcessCycleTime
            qpc.restype = ctypes.c_int
            cyc = ctypes.c_ulonglong()
            # 伪句柄 -1 直传：GetCurrentProcess 默认 restype 是 c_int，64 位
            # 伪句柄被截断后调用静默失败（§8.5 实测坑，绕开而非修复）
            hproc = ctypes.c_void_p(-1)

            def get_cycles(_f=qpc, _h=hproc, _c=cyc):
                if _f(_h, ctypes.byref(_c)):
                    return int(_c.value)
                return None

            self._get_cycles = get_cycles
            self.sources["cycles"] = "kernel32.QueryProcessCycleTime"
        except Exception as e:  # noqa: BLE001
            self.sources["cycles"] = "unavailable:%s" % repr(e)[:60]

    def sample(self) -> dict:
        """一次边界采样：全部字段都是 None 或真值（不可用≠0）。

        `sm_mhz`（W3）：只在 NVML 会话**已经存在**时读（`nvml_handle` 的
        缓存态判据），绝不因采样而初始化 NVML——GPU 路径必然已初始化
        （环境指纹/采样器），宿主纯 CPU 路径天然为 None。这是"这一相位
        跑在什么时钟上"的直接读数（C-39 边界：NVDEC% 不能当占空比，
        时钟/throttle 位才是有效信号）。
        """
        s = {"t": time.perf_counter(), "cpu": time.process_time(),
             "rss": None, "rb": None, "wb": None, "vram": None,
             "cycles": None, "sm_mhz": None, "thr": None,
             "threads": threading.active_count()}
        if _NVML["state"]:                     # 已初始化才读（无则零成本）
            try:
                import ctypes
                nvml, h = _NVML["state"]
                c = ctypes.c_uint()
                if int(nvml.nvmlDeviceGetClockInfo(h, 1, ctypes.byref(c))) == 0:
                    s["sm_mhz"] = int(c.value)
            except Exception:
                pass  # 单次时钟读数失败：留 None（不可用≠0），不上抛
        if self._get_cycles is not None:
            s["cycles"] = self._get_cycles()
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
        s["sm_mhz"] = ("nvmlDeviceGetClockInfo(SM)；仅当 NVML 会话已存在"
                       "（GPU 路径）——宿主纯 CPU 路径为 None，不主动初始化")
        return s

    def checkpoint(self, phase: str, threads: bool = False) -> None:
        try:
            sample = self._counters.sample()
            # v8 线程账本（full 档专属：threads 门由 Metrics.detailed 控制）；
            # 空采样不覆盖（无注册线程的 run 保持缺席语义，≠空账本冒充）
            if threads:
                led = thread_ledger()
                if led is not None:
                    _ts = led.sample()
                    if _ts:
                        sample["thr"] = _ts
                    # v10 全线程快照：含外来池（decord/OMP/TBB）逐 TID 周期
                    # ——争用定位的外来簇归因面（L2，2026-10-09 争用定位轮）
                    _ta = led.sample_all()
                    if _ta:
                        sample["thr_all"] = _ta
        except Exception:  # noqa: BLE001 遥测绝不让 run 失败
            return
        with self._lock:
            if len(self._rows) >= self._MAX_PHASES:
                return
            self._rows.append((phase, sample))

    def per_phase(self) -> dict:
        with self._lock:
            rows = list(self._rows)
        # 自校准频率（P2a）：f_run = 全程 Δcycles / 全程 Δprocess_time，
        # 即"每 CPU 秒的 cycle 数"。全程墙钟秒级、tick 数上百上千，15.625ms
        # 量化占比可忽略（bench/cycle_quant.json：双窗口一致到 0.017%）。
        # cores_avg_cycles = (Δcycles/Δwall) / f_run —— cycle 口径的平均核数，
        # 无 tick 量化伪影。限界：各相位实际频率不同时按全程均值换算有偏
        # （turbo 单核相位会偏低估），但短相位下远比 ±1 tick 的量化误差可信。
        f_run = 0.0
        if len(rows) >= 2:
            fa, fb = rows[0][1], rows[-1][1]
            dcpu_all = float(fb["cpu"] - fa["cpu"])
            if dcpu_all > 0 and fa.get("cycles") is not None \
                    and fb.get("cycles") is not None:
                f_run = (fb["cycles"] - fa["cycles"]) / dcpu_all
        out: dict = {}
        # v10.1 出生删失修复：解码池线程常在相位中段才出生（hybrid 的
        # CPU 臂池取决于校准走了哪条路），边界交集会把它们整体剔除
        # （h264 hybrid 外来簇 0.3M/帧=删失假象 vs 纯臂 13.6M）。改为
        # 携带 seen 表：本区间新出生（此前从未见过）的 TID 基线记 0。
        seen_tids: set = set()
        for (a, sa), (_b, sb) in zip(rows, rows[1:]):
            name = _b if _b not in out else "%s→%s" % (a, _b)
            dt = float(sb["t"] - sa["t"])
            if dt <= 0:
                continue
            row: dict = {"wall": round(dt, 4), "threads": sb["threads"]}
            dcpu = float(sb["cpu"] - sa["cpu"])
            ca, cb = sa.get("cycles"), sb.get("cycles")
            if ca is not None and cb is not None:
                # v7 周期账本（C-63）：原始 cycles 差分本体——A/B 判据用。
                # 与 cores_avg_cycles 不同，不除墙钟、不乘频率换算，跨臂
                # 配对可比；SMT 争用带 ±16% 均值漂移（C-63 前提：交错仍需）。
                row["cycles"] = max(0, cb - ca)
                if f_run > 0:
                    row["cores_avg_cycles"] = round(
                        max(0, cb - ca) / dt / f_run, 2)
            else:
                # v9 精简（2026-10-09 监测层级轮）：cores_avg（process_time
                # tick 口径，15.6ms 量化伪影）只在 cycles 来源缺席（非
                # Windows）时作为**唯一回退**发出；在场时不再发——与
                # cores_avg_cycles 并存=两个"平均核数"打架，读者分不清
                # 权威（P2a 引入 cycle 口径即为取代它）。
                row["cores_avg"] = round(max(0.0, dcpu) / dt, 2)
            # v8 线程账本：逐线程周期差分 + 占空比（duty = Δcycles/
            # (f_run×Δwall)，被换下等待的线程不积累周期——即忙碌份额）。
            ta, tb = sa.get("thr"), sb.get("thr")
            if isinstance(ta, dict) and isinstance(tb, dict):
                thr = {k: max(0, tb[k] - ta[k])
                       for k in set(ta) & set(tb)}
                if thr:
                    row["thr"] = thr
                    if f_run > 0:
                        row["thr_duty"] = {k: round(v / (f_run * dt), 3)
                                           for k, v in thr.items()}
            # v10 外来线程簇：全线程快照差分 − 命名线程（decord/OMP/TBB
            # 池逐 TID；零增量剔除；个体只用于差分、结论下在簇级——
            # 外来线程 TID 复用不可控，probe 侧按旋钮差分聚簇命名）。
            aa_, ab_ = sa.get("thr_all"), sb.get("thr_all")
            if isinstance(aa_, dict) and isinstance(ab_, dict):
                # 命名线程以 TID 反查排除（ledger idents：name→TID）
                led = thread_ledger()
                name2tid = {}
                if led is not None:
                    with led._lock:      # noqa: SLF001 - 同模块内部访问
                        name2tid = dict(led._idents)
                tids_named = set(name2tid.values())
                foreign = {}
                for k in set(ab_):
                    if k in tids_named:
                        continue
                    base = aa_.get(k)
                    if base is None:
                        if k in seen_tids:
                            continue   # 早前存在但区间起点缺席（重生/漏采）
                        base = 0      # 本区间新出生：基线 0
                    d = ab_[k] - base
                    if d > 0:
                        foreign[str(k)] = d
                seen_tids |= set(ab_)
                if foreign:
                    row["thr_foreign"] = foreign
                    row["thr_foreign_n"] = len(foreign)
            # W3：相位**终点**的 SM 时钟（GPU 路径相位差分才有；起点读数
            # 用于人工对齐——时钟爬坡期一个相位内前后差上千 MHz 是常态）
            if sb.get("sm_mhz") is not None:
                row["sm_mhz_at_end"] = sb["sm_mhz"]
            if sa.get("sm_mhz") is not None:
                row["sm_mhz_at_start"] = sa["sm_mhz"]
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
            # v7：全程周期账本 = 末边界 − 首边界（覆盖 calibrate+decode+ocr；
            # 首边界在 open 相位结束处）。两臂同代码路径下配对可比。
            fa_c, fb_c = rows[0][1].get("cycles"), rows[-1][1].get("cycles")
            if fa_c is not None and fb_c is not None:
                out["cycles_e2e"] = {
                    "total": max(0, fb_c - fa_c),
                    "span": "%s..%s" % (rows[0][0], rows[-1][0])}
                # v8：线程级全程账本 = 逐相位差分求和。不能直接用首末两次
                # 采样相减——进程级账本跨 run 存活，首采样可能持有上一轮
                # 已死线程的陈旧句柄（同名不同线程），差值无意义（实测
                # 'ocr':0 伪影，2026-10-09）。
                thr_sum: dict = {}
                for row in out.values():
                    if isinstance(row, dict) and isinstance(row.get("thr"),
                                                            dict):
                        for k, v in row["thr"].items():
                            thr_sum[k] = thr_sum.get(k, 0) + v
                if thr_sum:
                    out["cycles_e2e"]["threads"] = thr_sum
                # v10：外来簇全程合计（跨相位求和会把中途生灭的线程
                # 各自的贡献累上——比首末直减更符合"池总开销"语义）
                foreign_sum = sum(
                    v for row in out.values()
                    if isinstance(row, dict)
                    for v in (row.get("thr_foreign") or {}).values())
                if foreign_sum:
                    out["cycles_e2e"]["threads_foreign"] = foreign_sum
        out["checkpoints"] = [r[0] for r in rows]
        return out


class NvmlSampler:
    """L2：NVML 低采样率设备侧探针（full 档专属；ctypes，不新增依赖）。

    只在显式 `start()`/`stop()` 之间存在；失败逐字记录降级链进
    `sources`（报告必须能区分 NVML 直读与"不可用"）。
    """

    #: nvmlClocksThrottleReasons 位定义（nvml.h；名称 snake_case 记账）。
    #: GpuIdle / ApplicationsClocksSetting / SyncBoost 属常态、非降速；
    #: 其余任一出现 = 设备在主动压时钟（热/功率/硬件保护）——「忙而慢」
    #: 的直接证据（NVDEC% 是时间加权占用率不是吞吐率，掉频时照样 100%）。
    THROTTLE_BITS: tuple = (
        ("gpu_idle", 0x1),
        ("apps_clocks_setting", 0x2),
        ("sw_power_cap", 0x4),
        ("hw_slowdown", 0x8),
        ("sync_boost", 0x10),
        ("sw_thermal_slowdown", 0x20),
        ("hw_thermal_slowdown", 0x40),
        ("hw_power_brake_slowdown", 0x80),
        ("display_clock_setting", 0x100),
    )

    def __init__(self, interval_s: float = 0.2, max_points: int = 600) -> None:
        self._interval = max(0.05, float(interval_s))
        self._max = int(max_points)
        # (t, gpu%, decode%, vram_mib, sm_mhz, mem_mhz, vid_mhz, throttle_mask)
        self._pts: list = []
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
        sm, mclk, vclk = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        reasons = ctypes.c_ulonglong()
        rates = nvml.nvmlDeviceGetUtilizationRates
        dutil = nvml.nvmlDeviceGetDecoderUtilization
        minfo = nvml.nvmlDeviceGetMemoryInfo
        # nvmlClockType: GRAPHICS=0 / SM=1 / MEM=2 / VIDEO=3。NVDEC 工作负载
        # 跟随 VIDEO 域（部分驱动回落 SM），两域都记。
        clock = nvml.nvmlDeviceGetClockInfo
        throttle = getattr(nvml, "nvmlDeviceGetCurrentClocksThrottleReasons",
                           None)

        def loop():
            # 每 tick 独立 try（单次 NVML 调用失败只计 _fails，不杀线程）；
            # NVML 会话是进程级的（环境指纹可能还在用），故此处不 shutdown。
            while not self._stop.wait(self._interval):
                row = [time.perf_counter(), None, None, None,
                       None, None, None, None]
                try:
                    if int(rates(h, ctypes.byref(util))) == 0:
                        row[1] = int(util.gpu)
                    # 第 4 参 pUUID 传 NULL（本机单卡，按索引取设备）
                    if int(dutil(h, ctypes.byref(dec),
                                 ctypes.byref(dur), None)) == 0:
                        row[2] = int(dec.value)
                    if int(minfo(h, ctypes.byref(mem))) == 0:
                        row[3] = round(int(mem.used) / 1048576.0, 1)
                    if int(clock(h, 1, ctypes.byref(sm))) == 0:
                        row[4] = int(sm.value)
                    if int(clock(h, 2, ctypes.byref(mclk))) == 0:
                        row[5] = int(mclk.value)
                    if int(clock(h, 3, ctypes.byref(vclk))) == 0:
                        row[6] = int(vclk.value)
                    if throttle is not None and int(
                            throttle(h, ctypes.byref(reasons))) == 0:
                        row[7] = int(reasons.value)
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

    def points(self) -> list:
        """原始时间点列的只读拷贝（P4 trace 用；stop() 后仍可取）。

        行结构：(t, gpu%, nvdec%, vram_mib, sm_mhz, mem_mhz, vid_mhz,
        throttle_mask)——聚合摘要之外唯一的带时间戳序列。
        """
        return list(self._pts)

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

        out = {"sources": self.sources, "error": self._err or None,
               "interval_s": self._interval, "n": len(pts),
               "sample_failures": fails,
               "gpu_util_pct": summary(1), "nvdec_util_pct": summary(2),
               "vram_used_mib": summary(3),
               "sm_clock_mhz": summary(4), "mem_clock_mhz": summary(5),
               "video_clock_mhz": summary(6)}
        # 热降/功率位：按原因记 tick 数；ticks_throttled 只计「非常态」原因
        # （GpuIdle/AppsClocksSetting/SyncBoost 除外）。与 NVDEC% 联合判读：
        # 占用率高 + 时钟贴上限 + 无原因位 = 忙且健康；占用率高 + 原因位
        # 频发 = 忙而慢（降速）；占用率低 = 闲置（归因看 hybrid-stats）。
        rc = {k: 0 for k, _ in self.THROTTLE_BITS}
        n_throttle = 0
        for p in pts:
            mask = p[7]
            if mask is None:
                continue
            hit = False
            for k, bit in self.THROTTLE_BITS:
                if mask & bit:
                    rc[k] += 1
                    if k not in ("gpu_idle", "apps_clocks_setting",
                                 "sync_boost"):
                        hit = True
            n_throttle += hit
        out["throttle"] = {"ticks_throttled": n_throttle, "ticks": len(pts),
                           "reason_ticks": {k: v for k, v in rc.items() if v}}
        return out
