"""制度健康度（KPI）计算。

compute → registry/kpi.yaml（数据事实源）；docs/HEALTH.md 由 render.py 渲染。
--check 重算比对（release 链步骤）；口径：

  injection_bytes   注入核字节（AGENTS.md ≤6144，宪章不变量 5——触碰=警报）
  charter_bytes     宪章 ≤2048（手写制度文本的唯一处）
  gov_corpus_bytes  树内治理语料（charter+registry+视图+当轮 log+artifacts）
  gov_commit_share  全史治理类提交占比（pathspec 同旧测法，可与旧 70% 对照）
  pure_gov_share    纯治理轮占比（该提交不触产品/测试代码）
  body_p90          提交正文长度 p90（adv-commit-slim ≤600）
  dangling_refs     refs check 违规数（目标 0；直接复用 refs.check 结果）
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
KPI = REGISTRY / "kpi.yaml"

# 语料口径（KPI 树内治理语料）：文档与数据，不含 L2 执行体源码（那是代码不是语料）；
# 度量产物（kpi.yaml/HEALTH.md）不计入——避免 KPI 自指循环。
GOV_CORPUS = [
    "charter.md",
    "registry/",
    "AGENTS.md",
    "docs/CONCLUSIONS.md",
    "docs/KNOBS.md",
    "docs/log/",
    "tools/INDEX.md",
]
# 提交占比口径（与旧制度 70% 实测可比）：语料 + 制度执行体改动均算治理提交。
GOV_PATHSPEC = GOV_CORPUS + [
    "docs/HEALTH.md",
    "registry/kpi.yaml",
    "tools/refs.py",
    "tools/render.py",
    "tools/_audit_ext.py",
    "tools/check.py",
    "tools/conclusions_sweep.py",
    "tools/version_advice.py",
    "tools/health.py",
]
PRODUCT_PATHSPEC = ["video_ocr_engine/", "tests/", "tools/bench.py", "tools/_measure_lib.py"]


def _utf8_stdout() -> None:
    import io as _io

    if isinstance(sys.stdout, _io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")


def _git(*args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8"
    )
    return r.stdout


def _dir_bytes(paths: list[str]) -> int:
    total = 0
    for rel in paths:
        p = ROOT / rel
        if p.is_file():
            total += p.stat().st_size
        elif p.is_dir():
            total += sum(q.stat().st_size for q in p.rglob("*") if q.is_file())
    kpi = REGISTRY / "kpi.yaml"
    if kpi.is_file():
        total -= kpi.stat().st_size  # 度量产物不计（自指防护；registry/ 内）
    return total


def _commit_shares() -> tuple[float, float]:
    total = [h for h in _git("log", "--format=%H").splitlines() if h.strip()]
    gov = set(_git("log", "--format=%H", "--", *GOV_PATHSPEC).splitlines())
    prod = set(_git("log", "--format=%H", "--", *PRODUCT_PATHSPEC).splitlines())
    pure_gov = gov - prod
    n = max(len(total), 1)
    return 100.0 * len(gov) / n, 100.0 * len(pure_gov) / n


def _body_p90() -> int:
    # 粗口径：全部提交正文行的 p90（逐行而非逐提交，量级指示足够）
    raw = _git("log", "--format=%B").splitlines()
    lens = [len(line) for line in raw] or [0]
    lens.sort()
    return lens[int(len(lens) * 0.9) - 1] if lens else 0


def compute() -> dict:
    import time as _t

    sys.path.insert(0, str(ROOT / "tools"))
    import refs

    dangling = len(refs.check())
    gov_share, pure_share = _commit_shares()
    return {
        "generated_at": _t.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": [
            {
                "name": "注入核字节（≤6144=宪章不变量）",
                "value": (ROOT / "AGENTS.md").stat().st_size,
                "target": "≤6144",
            },
            {
                "name": "宪章字节（手写制度文本唯一处）",
                "value": (ROOT / "charter.md").stat().st_size,
                "target": "≤2048",
            },
            {
                "name": "树内治理语料（字节）",
                "value": _dir_bytes(GOV_CORPUS),
                "target": "≤100KB（benchmarks 归档后）",
            },
            {"name": "治理类提交占比（全史）", "value": round(gov_share, 1), "target": "≤25%"},
            {"name": "纯治理轮占比（全史）", "value": round(pure_share, 1), "target": "≤15%"},
            {"name": "提交信息行 p90（字符）", "value": _body_p90(), "target": "≤600"},
            {"name": "悬空引用（refs）", "value": dangling, "target": "0"},
        ],
    }


def write() -> int:
    KPI.parent.mkdir(parents=True, exist_ok=True)
    KPI.write_text(
        yaml.safe_dump(compute(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    print("kpi.yaml 已生成（重渲染：python tools/render.py --write）")
    return 0


def check() -> int:
    stored = yaml.safe_load(io.open(KPI, encoding="utf-8").read())
    fresh = compute()

    # 时点敏感指标（shares/generated_at）不比对，只比对状态类
    # 时点敏感指标（随每次提交漂移）不比对：占比类 + p90
    VOLATILE = ("治理类提交占比（全史）", "纯治理轮占比（全史）", "提交信息行 p90（字符）")

    def stable(m: dict) -> list:
        return [r for r in m["metrics"] if r["name"] not in VOLATILE]

    if stable(stored) != stable(fresh):
        print(
            "✗ kpi.yaml 与当前实态不符（python tools/health.py && python tools/render.py --write）"
        )
        return 1
    print("✓ kpi.yaml 与实态一致")
    return 0


def main(argv: list[str]) -> int:
    _utf8_stdout()
    if "--check" in argv:
        return check()
    return write()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
