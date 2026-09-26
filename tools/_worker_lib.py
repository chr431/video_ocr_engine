"""探针子进程 WORKER 公共件（P5+P6，2026-09-20 稳健性轮）。

三个痛点（均有实证）：

1. WORKER 以 ``python -c`` 执行、``-c`` 下无 ``__file__`` → 每个探针自己
   拼 PROBE_ROOT 注入与 utf-8 reconfigure（2026-09-20 裁切轮在参数序上
   踩过一次）——``worker_prelude`` 统一；
2. 两轮探针叠加跑污染测量（2026-09-20 奖金池轮事故，数据作废重测）——
   ``acquire_probe_lock`` 探针级互斥；
3. 父进程被杀后孤儿 worker 继续烧 CPU（同上事故）——``watchdog_prelude``
   让 worker 每 5s 查父 pid，父死即 ``os._exit``（psutil 探测，跨平台；
   Windows 上 os.kill(pid,0) 会真杀进程，不可用）。

用法::

    from _worker_lib import (acquire_probe_lock, run_worker,
                              watchdog_prelude, worker_prelude)

    WORKER = worker_prelude + watchdog_prelude + r'''
    import json, time
    ...  # 业务体：sys.argv[1:] = 调用方参数；末行 print(json.dumps(...))
    '''

    with acquire_probe_lock("my_probe"):
        d = run_worker(WORKER, [video, roi, n], env_extra={"A": "1"})
        if "err" in d: ...
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_LOCK_DIR = ROOT / "bench"          # gitignored（registry 同目录）

worker_prelude = r'''
import os, sys
sys.path.insert(0, os.environ["PROBE_ROOT"])
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # 无交互 stdout（嵌入式/null）时 reconfigure 缺席，无需处理
'''

watchdog_prelude = r'''
def _voe_die_with_parent():
    import time
    try:
        import psutil
    except ImportError:
        return               # 无 psutil 则放弃看门狗（测量不受影响）
    ppid = int(os.environ.get("PROBE_PPID", "0"))
    if not ppid:
        return
    while True:
        time.sleep(5.0)
        if not psutil.pid_exists(ppid):
            os._exit(3)      # 父进程已死：立即退出防孤儿烧 CPU
import threading
threading.Thread(target=_voe_die_with_parent, daemon=True).start()
'''


def run_worker(worker_src: str, args, env_extra: dict | None = None,
               timeout: float | None = None, env_clear=()) -> dict:
    """跑一个 WORKER 子进程，返回其末行 JSON（失败返回 {'err': ...}）。

    env_clear：先从父环境剔除的键（A/B 臂切换时防父进程残留旋钮渗透）。
    """
    e = dict(os.environ)
    for k in env_clear:
        e.pop(k, None)
    e["PROBE_ROOT"] = str(ROOT)
    e["PROBE_PPID"] = str(os.getpid())
    if env_extra:
        e.update(env_extra)
    p = subprocess.run(
        [sys.executable, "-c", worker_src, *[str(a) for a in args]],
        capture_output=True, text=True, env=e, timeout=timeout)
    out = (p.stdout or "").strip().splitlines()
    if p.returncode != 0 or not out:
        return {"err": "exit=%s: %s" % (p.returncode,
                                        (p.stderr or "").strip()[-400:])}
    try:
        return json.loads(out[-1])
    except json.JSONDecodeError:
        return {"err": "stdout 尾行非 JSON：%s" % out[-1][:200]}


@contextmanager
def acquire_probe_lock(name: str):
    """同名探针互斥（防两实例叠加跑污染测量）。

    锁 = ``bench/probe-<name>.lock``（内容=持有者 pid）。pid 已死视为
    陈旧锁直接接管；pid 复用的误报由提示语引导人工清锁。
    """
    import psutil
    _LOCK_DIR.mkdir(exist_ok=True)
    lock = _LOCK_DIR / ("probe-%s.lock" % name)
    if lock.exists():
        try:
            pid = int((lock.read_text(encoding="utf-8") or "0").strip())
        except ValueError:
            pid = 0
        if pid and psutil.pid_exists(pid):
            raise SystemExit(
                "另一 %r 实例在跑（pid=%d，锁 %s）；确认非本人测量后"
                "删除锁文件或等待其结束。" % (name, pid, lock))
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass  # 锁已被外部清理
