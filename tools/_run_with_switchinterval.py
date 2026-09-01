"""临时 runner：在指定 sys.setswitchinterval 下跑 bench_hybrid。

用途：验证 GIL 争抢是否是 hybrid 里 CPU 生产者掉速的原因 —— 若瓶颈是
"等 GIL"，调小 switch interval 应改善；若不是则无变化。

用法：
    python tools/_run_with_switchinterval.py <interval> --video X --roi A,B,C,D ...
      （其余参数原样透传给 bench_hybrid.main）
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: _run_with_switchinterval.py <interval> [bench_hybrid 参数...]")
        return 1
    interval = float(sys.argv[1])
    sys.setswitchinterval(interval)
    del sys.argv[1]                      # 剥掉 interval，其余透传
    sys.argv[0] = "bench_hybrid.py"

    from bench_hybrid import main as bench_main
    return int(bench_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
