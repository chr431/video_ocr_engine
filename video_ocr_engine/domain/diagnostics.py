"""在线自诊断：挂死 / 崩溃时**也能**拿到现场（不依赖额外探针）。

问题（本轮实测确认）：RunReport 只在 `extract()` 正常返回时组装——
**挂死或硬崩溃时什么都不输出**。历史代价：§16 host 路径 hybrid+ONNX 环死锁
只能靠 fork 侧 `[hybrid-stats]` 打印 + 一次性探针定位；decord 析构 UAF 崩溃
同样无现场。

本模块给出两条互补通道，共用**一个**守护线程与**一个**输出前缀：

1. `RunJournal`（崩溃日志）：append-only JSONL 里程碑（相位、批/段/chunk 计数、
   队列深度）。先进内存环形缓冲，由守护线程按 `flush_interval_s` 落盘。
   硬崩溃后文件尾部 = 最后到达的里程碑 → 可归因到具体相位。
2. `StallWatchdog`（停顿看门狗）：守护线程每 `poll_s` 读一次单调进展计数；
   若 `now - last_progress > stall_s` → 落 ΔJSON（每线程相位与停滞时长、
   队列深度、指标快照、Python 栈、decord hybrid stats 若可用），可选
   faulthandler 原生栈 / 中止进程。

成本纪律（PI-15）：
- 心跳 = 一次 int 自增 + 一次 `threading.local` 属性赋值（`tick`）。
- 深度探针**只在停顿发生时**才调用（`qsize()` 不进稳态路径）。
- 默认关闭（铁律 10）；`auto` 档随既有 `VOE_REPORT_FILE` opt-in 开启，
  不产生新的文件副作用。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

#: 环形缓冲上限：崩溃日志在内存里最多攒这么多条里程碑（约 200 KB 级）
JOURNAL_MAXLEN = 4096
#: 看门狗轮询间隔（秒）——空转成本 = 一次整数比较
DEFAULT_POLL_S = 0.5
#: 无进展多久算停顿（秒）
DEFAULT_STALL_S = 30.0
#: 崩溃日志落盘间隔（秒）
DEFAULT_FLUSH_S = 0.25


class Progress:
    """单调进展心跳：`tick` 由时序脊柱调用，看门狗只读。"""

    __slots__ = ("_seq", "_local")

    def __init__(self) -> None:
        self._seq = 0
        self._local = threading.local()

    def tick(self, phase: str, detail: str = "") -> int:
        """记录一次进展；返回当前序号。热路径：一次自增 + 一次属性赋值。"""
        self._seq += 1
        self._local.phase = phase
        self._local.detail = detail
        self._local.at = time.monotonic()
        return self._seq

    @property
    def seq(self) -> int:
        return self._seq

def _stack_of(frames: dict, tid: int) -> list:
    fr = frames.get(tid)
    if fr is None:
        return []
    return [ln.rstrip("\n") for ln in traceback.format_stack(fr)[-6:]]


class RunJournal:
    """append-only 里程碑日志（崩溃后定位最后一个到达点）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._buf: deque = deque(maxlen=JOURNAL_MAXLEN)
        self._lock = threading.Lock()
        self._n_written = 0
        self._n_dropped = 0
        self._last_err: str = ""   # 2026-09-19 审查轮：此前只在 except
                                   # 分支赋值，report() 读它会 AttributeError
        self._fh = None

    def add(self, event: str, **fields) -> None:
        rec = {"t": round(time.monotonic(), 4), "wall": time.time(),
               "event": event}
        rec.update(fields)
        with self._lock:
            if len(self._buf) == self._buf.maxlen:
                self._n_dropped += 1
            self._buf.append(rec)

    def flush(self) -> int:
        """把缓冲落盘（守护线程按间隔调用；异常一律吞掉并计数）。"""
        with self._lock:
            if not self._buf:
                return 0
            recs = list(self._buf)
            self._buf.clear()
        try:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a", encoding="utf-8",
                                newline="\n", buffering=1)
            for r in recs:
                self._fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception as e:                      # noqa: BLE001
            self._n_dropped += len(recs)
            self._last_err = repr(e)
            return 0
        self._n_written += len(recs)
        return len(recs)

    def close(self) -> None:
        self.flush()
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:                       # noqa: BLE001
                pass  # 关闭失败无需上抛：进程即将回收 fd
            self._fh = None

    def report(self) -> dict:
        # last_error（2026-09-19 审查轮）：_last_err 此前只写不读——落盘
        # 持续失败（磁盘满/权限）时报告里只有计数，失败原因永远不可见。
        return {"path": str(self.path), "written": self._n_written,
                "dropped": self._n_dropped,
                "last_error": self._last_err or None}


class StallWatchdog:
    """停顿看门狗：无进展超过阈值即落现场快照。"""

    def __init__(self, progress: Progress, out_prefix: str | Path, *,
                 stall_s: float = DEFAULT_STALL_S,
                 poll_s: float = DEFAULT_POLL_S,
                 flush_s: float = DEFAULT_FLUSH_S,
                 journal: RunJournal | None = None,
                 snapshot=None, native_stacks: bool = False,
                 abort: bool = False) -> None:
        self._p = progress
        self._prefix = Path(out_prefix)
        self._stall_s = float(stall_s)
        self._poll_s = float(poll_s)
        self._flush_s = float(flush_s)
        self._journal = journal
        self._snapshot = snapshot          # 返回指标快照的可调用对象
        self._native = bool(native_stacks)
        self._abort = bool(abort)
        self._depths: dict = {}            # 只在停顿/落盘时调用，不进稳态
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self._stalls = 0
        self._last_dump: str | None = None
        self._stall_reports: list = []

    def watch_depth(self, label: str, fn) -> None:
        self._depths[label] = fn

    def ensure_started(self) -> None:
        """惰性启动（首次进展时）：从未跑过 run 就不起线程、不建文件。"""
        if self._th is not None:
            return
        if self._native:
            import faulthandler
            try:
                fh = open(str(self._prefix) + ".faulthandler.txt", "a")
                faulthandler.enable(file=fh, all_threads=True)
            except Exception:                       # noqa: BLE001
                pass  # 原生栈是尽力而为：拿不到也不影响 Python 侧通道
        self._th = threading.Thread(target=self._loop, name="voe-watchdog",
                                    daemon=True)
        self._th.start()

    def _loop(self) -> None:
        last_seq = -1
        last_change = time.monotonic()
        t_flush = time.monotonic()
        while not self._stop.wait(self._poll_s):
            now = time.monotonic()
            seq = self._p.seq
            if seq != last_seq:
                last_seq, last_change = seq, now
            if self._journal is not None and (now - t_flush) >= self._flush_s:
                self._journal.flush()
                t_flush = now
            if (now - last_change) >= self._stall_s:
                self.dump_stall(now - last_change)
                last_change = now      # 每次停顿只落一次，避免刷盘风暴

    def dump_stall(self, age: float) -> dict:
        """落一份停顿现场（JSON）并返回该 dict；自身绝不抛异常。"""
        self._stalls += 1
        payload = {
            "kind": "stall",
            "age_s": round(float(age), 3),
            "stall_index": self._stalls,
            "stall_threshold_s": self._stall_s,
            "progress_seq": self._p.seq,
            "pid": os.getpid(),
            "threads": self._thread_dump(),
            "depths": self._depth_dump(),
        }
        if self._snapshot is not None:
            try:
                payload["metrics"] = self._snapshot()
            except Exception as e:                  # noqa: BLE001
                payload["metrics_error"] = repr(e)
        try:
            payload["hybrid_stats"] = self._hybrid_stats()
        except Exception:                           # noqa: BLE001
            pass  # fork 侧统计为可选增强，缺失不影响现场价值
        if self._journal is not None:
            self._journal.add("stall", age_s=round(float(age), 3),
                              stall_index=self._stalls)
            self._journal.flush()
            payload["journal"] = self._journal.report()
        path = "%s.stall-%d.json" % (self._prefix, self._stalls)
        try:
            tmp = path + ".tmp"
            Path(tmp).write_text(json.dumps(payload, ensure_ascii=False,
                                            indent=1, default=repr),
                                 encoding="utf-8", newline="\n")
            os.replace(tmp, path)
            self._last_dump = path
        except Exception as e:                      # noqa: BLE001
            payload["write_error"] = repr(e)
        self._stall_reports.append({"age_s": payload["age_s"], "path": path})
        if self._abort:
            os._exit(70)                            # 显式请求时才中止进程
        return payload

    def _thread_dump(self) -> dict:
        frames = sys._current_frames()
        out = {}
        for t in threading.enumerate():
            if t.name == "voe-watchdog":
                continue
            out[t.name] = {"daemon": t.daemon, "alive": t.is_alive(),
                           "stack": _stack_of(frames, t.ident)}
        return out

    def _depth_dump(self) -> dict:
        out = {}
        for label, fn in self._depths.items():
            try:
                out[label] = fn()
            except Exception as e:                  # noqa: BLE001
                out[label] = "err:%r" % (e,)
        return out

    @staticmethod
    def _hybrid_stats() -> dict:
        """fork 侧 [hybrid-stats] 计数（E1 草案：fork 暴露 native 快照）。

        取不到就返回空——本模块不因缺它而降级。
        """
        import decord
        fn = getattr(decord, "hybrid_stats", None)
        return dict(fn()) if callable(fn) else {}

    def stop(self) -> dict:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)
            self._th = None
        if self._journal is not None:
            self._journal.close()
        return self.report()

    def report(self) -> dict:
        out = {"armed": True, "stall_s": self._stall_s,
               "polls": int(self._stalls), "stalls": self._stalls}
        if self._stall_reports:
            out["dumps"] = list(self._stall_reports)
        if self._last_dump:
            out["last_dump"] = self._last_dump
        if self._journal is not None:
            out["journal"] = self._journal.report()
        return out


class Diagnostics:
    """进展心跳 + 崩溃日志 + 停顿看门狗的组合门面（管线只认这一个对象）。"""

    armed = True

    def __init__(self, progress: Progress, watchdog: StallWatchdog,
                 journal: RunJournal) -> None:
        self.progress = progress
        self.watchdog = watchdog
        self.journal = journal

    def tick(self, phase: str, detail: str = "") -> None:
        """时序脊柱每次收尾调用：自增 + local 赋值 + 一次环形缓冲入队。

        **每次都写崩溃日志**——否则文件里只剩 run_start 一行，崩溃后无从
        归因（本模块首版就是这个缺陷，已由端到端实测抓出）。
        """
        if self.watchdog._th is None:
            self.watchdog.ensure_started()
        self.progress.tick(phase, detail)
        self.journal.add(phase, **({"detail": detail} if detail else {}))

    def watch_depth(self, label: str, fn) -> None:
        self.watchdog.watch_depth(label, fn)

    def stop(self) -> dict:
        return self.watchdog.stop()

    def report(self) -> dict:
        return self.watchdog.report()


def open_diagnostics(report_file: str | None, *, metrics=None,
                     stall_s: float = DEFAULT_STALL_S) -> object:
    """按既有 opt-in 开启诊断；未 opt-in 返回 `NULL_DIAG`（零成本）。

    设计：**不新增环境旋钮**。诊断会写文件（副作用），故挂在既有
    `VOE_REPORT_FILE` 这个已注册的显式 opt-in 上——要报告文件的人，
    顺带得到同前缀的 `.journal.jsonl` 与 `.stall-N.json`。
    停顿阈值取模块常量（`DEFAULT_STALL_S`），v1 不做旋钮化。
    """
    if not report_file:
        return NULL_DIAG
    base = str(report_file)
    progress = Progress()
    journal = RunJournal(base + ".journal.jsonl")
    snap = None
    if metrics is not None:
        # 只读快照（drain=False，2026-09-19 审查轮）：看门狗在 run 中途
        # 落盘停顿现场，此前用 drain 语义的 snapshot 会把已累计的桶全部
        # 吃掉——停顿之后的最终 RunReport 只剩停顿后的样本，PI-3/PI-10
        # 等阈值指标偏小 → 门禁静默放行。
        snap = lambda: metrics.snapshot(drain=False)
    wd = StallWatchdog(progress, base + ".diag", stall_s=stall_s,
                       journal=journal, snapshot=snap)
    diag = Diagnostics(progress, wd, journal)
    journal.add("run_start", pid=os.getpid())
    return diag


class NullDiagnostics:
    """诊断关闭（默认）：`tick` 是一次方法调用 + 立即返回。"""

    armed = False
    progress = None

    def tick(self, phase: str, detail: str = "") -> None:
        return None

    def watch_depth(self, label: str, fn) -> None:
        return None

    def stop(self) -> dict:
        return {}

    def report(self) -> dict:
        return {}


NULL_DIAG = NullDiagnostics()
