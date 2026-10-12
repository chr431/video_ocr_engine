"""两级标准化测试入口 —— 日常 / 发版各跑什么，一条命令（2026-10-10）。

动机
----
测试面分散在 pytest、守卫组、金标 verify、发版清扫、版本/KPI 多处，"现在该跑哪
些"靠记纪律段；且金标 full（72s）与 smoke（6.7s）的分层此前只写在
AGENTS 里。本脚本把两级合规固化成一条命令，每步带计时与结果汇总：

    python tools/check.py               # 日常（≈36s）：迭代期默认
    python tools/check.py --tier release  # 发版/合并（≈2min，含全量金标+清扫）

分级依据（实测，本机 2026-10-10，见 docs/log/2026-10-10-测试减负实验轮.md）：
  pytest 29.3s；audit --since 秒级；金标 smoke 6.7s / full 71.9s；
  window matrix（fork 门禁，decord 侧）41.9s×5；run_fast 七套件 346s。
  telemetry-check / ab 不进两级链（只在改遥测/做 A/B 时按 rules.yaml 跑）。

降级：无真值视频环境（CI/新机）自动跳过金标步并标注 SKIP；步骤失败
即非 0 退出，汇总表给全量结果。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")


def _has_videos() -> bool:
    try:
        from env_probe import video_dir

        return os.path.isfile(os.path.join(video_dir(), "test5.mp4"))
    except Exception:  # noqa: BLE001 — 环境事实
        return False


def _golden_case_count() -> str:
    """金标用例数动态取（防标签自漂移——用例增删不用改本文件）。"""
    try:
        import glob as _glob

        n = len(_glob.glob(os.path.join(ROOT, "tests", "golden", "case-*")))
        return str(n) if n else "?"
    except OSError:
        return "?"


STEPS = {
    "pytest": ("pytest -q（全量单测）", [sys.executable, "-m", "pytest", "tests/", "-q"]),
    "refs": ("引用解析（substrate）", [sys.executable, "tools/refs.py", "check"]),
    "render": ("渲染一致性（六视图）", [sys.executable, "tools/render.py", "--check"]),
    "ext": ("结构化审计（按名注册）", [sys.executable, "tools/_audit_ext.py"]),
    "golden_smoke": (
        "金标 smoke（3 用例）",
        [sys.executable, "tests/golden/record.py", "--verify", "--tier", "smoke"],
    ),
    "golden_full": (
        "金标 full（%s 用例）" % _golden_case_count(),
        [sys.executable, "tests/golden/record.py", "--verify"],
    ),
    "sweep": (
        "发版清扫（结论钉版清单）",
        [sys.executable, "tools/conclusions_sweep.py", "--strict"],
    ),
    "version": (
        "版本纪律（档位判据 verify）",
        [sys.executable, "tools/version_advice.py", "verify"],
    ),
    "health": (
        "制度健康度（KPI 对账）",
        [sys.executable, "tools/health.py", "--check"],
    ),
}
DAILY = ["pytest", "refs", "render", "ext", "golden_smoke"]
RELEASE = ["pytest", "refs", "render", "ext", "golden_full", "sweep", "version", "health"]


def run_step(name: str, cmd: list, timeout: int) -> tuple[str, float, str]:
    t0 = time.perf_counter()
    r = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    ok = "✓" if r.returncode == 0 else "✗"
    return ok, time.perf_counter() - t0, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser(description="两级标准化测试入口（日常/发版）")
    ap.add_argument("--tier", default="daily", choices=("daily", "release"))
    args = ap.parse_args()

    plan = list(DAILY)
    if args.tier == "release":
        plan = list(RELEASE)
    has_vid = _has_videos()

    print("=" * 72)
    print("check --tier %s（视频语料：%s）" % (args.tier, "有" if has_vid else "无→跳过金标"))
    print("=" * 72)
    results, failed = [], []
    for name in plan:
        label, cmd = STEPS[name]
        if name.startswith("golden") and not has_vid:
            results.append((name, label, "SKIP", 0.0))
            continue
        ok, dt, out = run_step(name, cmd, timeout=900)
        results.append((name, label, ok, dt))
        if ok == "✗":
            failed.append(name)
            print(out[-1200:])
        elif name == "sweep" and out.strip():
            # 清扫清单即使通过也可见：落后钉版/触发词命中是发版前的决策输入，
            # 只在失败时打印会让"通过但带清单"的输出被吞（2026-10-12 审查轮）。
            print(out.rstrip())
    print("\n汇总：")
    for name, label, ok, dt in results:
        print("  %-2s %-28s %6.1fs" % (ok, label, dt))
    skipped = [name for name, _, ok, _ in results if ok == "SKIP"]
    if failed:
        print("\n✗ 失败：%s" % ", ".join(failed))
        return 1
    if skipped:
        # SKIP 不冒充通过：金标未验证这一事实必须出现在汇总末行（2026-10-12）。
        print("\n△ %s 链部分通过（跳过：%s——金标未验证）" % (args.tier, ", ".join(skipped)))
        return 0
    print("\n✓ %s 链全部通过（总 %.1fs）" % (args.tier, sum(d for _, _, _, d in results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
