"""跨代码版本的同会话交错 A/B（通用件；2026-09-20 P1 驱动合一轮引入）。

## 为什么存在
``bench ab`` 交错的是**同码两配置**；重构中性验证需要交错的是**同配置
两版本代码**。硬件漂移（GPU 热降，同码连跑可差 7.7%，C-51）禁止
「新代码 vs 注册表过往条目」的直接比对——必须同会话、同轮配对。

## 用法
    python tools/_probe_worktree_ab.py --configs h264-hybrid,h264-cpu,host:h264-cpu

- A 臂 = HEAD 的临时 worktree（旧代码，gitignored 的 ocr_engines/ 会
  拷入防 TRT 冷构建污染计时）；B 臂 = 当前工作树（新代码，含未提交改动）。
- ``host:`` 前缀 = 该配置追加 ``--ocr-backend cpu``（GPU 门控关 → 宿主
  管线路径）；无前缀走该配置默认 OCR 后端。
- 每臂子进程 = ``bench.py run --rounds 2``（自带 GPU 时钟门禁），取本轮
  wall 最小值；臂序逐轮轮转（偶数轮 old→new，奇数轮 new→old）。
- 判据（同 bench ab 口径）：同轮配对差分均值 + 符号多数；
  |均值| > --hard 且 B 慢占多数 → 退出码 1。
- 段数同轮比对是免费的正确性旁证（重构中性 ⇒ 每轮段数必须相等）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_worktree(rev: str) -> Path | None:
    """把 rev 检出到临时 worktree（模式同 _probe_cpu_onnx）。"""
    import tempfile
    import shutil
    d = Path(tempfile.mkdtemp(prefix="wt_ab_")) / "old"
    p = subprocess.run(["git", "-C", str(ROOT), "worktree", "add",
                        "--detach", str(d), rev],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(f"worktree 失败：{p.stderr.strip()[-200:]}")
        return None
    # ocr_engines/ 被 gitignore（TRT 引擎缓存），worktree 里没有 → 拷过去，
    # 否则旧版本的 TRT 用例会走冷构建（污染 A/B 计时）。
    src = ROOT / "ocr_engines"
    if src.is_dir():
        shutil.copytree(src, d / "ocr_engines", dirs_exist_ok=True)
    return d


def drop_worktree(d: Path | None) -> None:
    if not d:
        return
    subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force",
                    str(d)], capture_output=True, text=True)


def _registry_len(cwd: Path) -> int:
    f = cwd / "bench" / "registry.jsonl"
    if not f.exists():
        return 0
    with f.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def run_arm(cwd: Path, config: str, ocr_backend: str, label: str) -> dict:
    """一臂一次 = bench run（rounds=2，取 wall 最小）；返回本轮结果。"""
    cmd = [sys.executable, "tools/bench.py", "run", "--config", config,
           "--rounds", "2", "--label", label]
    if ocr_backend:
        cmd += ["--ocr-backend", ocr_backend]
    n0 = _registry_len(cwd)
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        return {"err": f"exit={p.returncode}: {p.stderr.strip()[-200:]}"}
    f = cwd / "bench" / "registry.jsonl"
    with f.open(encoding="utf-8") as fh:
        lines = fh.readlines()[n0:]
    recs = [json.loads(l) for l in lines]
    if not recs:
        return {"err": "registry 无新条目"}
    best = min(recs, key=lambda r: r["wall"])
    return {"wall": best["wall"], "n_segments": best.get("n_segments")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="h264-hybrid,h264-cpu,host:h264-cpu",
                    help="逗号分隔；host: 前缀 = 追加 --ocr-backend cpu")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--hard", type=float, default=1.5,
                    help="配对差分均值 %% 判决阈值")
    ap.add_argument("--baseline-rev", default="HEAD")
    args = ap.parse_args()

    specs = []
    for tok in args.configs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("host:"):
            specs.append((tok[5:], "cpu"))
        else:
            specs.append((tok, ""))

    old = make_worktree(args.baseline_rev)
    if old is None:
        return 2
    fail = 0
    try:
        for config, ocr in specs:
            tag = f"host:{config}" if ocr else config
            print(f"\n== {tag}（A=旧 {args.baseline_rev} / B=新工作树，"
                  f"{args.rounds} 轮交错）==")
            olds, news = [], []
            for i in range(args.rounds):
                order = (("old", old), ("new", ROOT))
                if i % 2:
                    order = order[::-1]        # 臂序轮转：奇数轮 B 先跑
                for arm, cwd in order:
                    r = run_arm(cwd, config, ocr, f"wt-ab-{arm}-{tag}")
                    time.sleep(args.cooldown)
                    if "err" in r:
                        print(f"  r{i} {arm}: ERR {r['err']}")
                        (olds if arm == "old" else news).append(None)
                        continue
                    print(f"  r{i} {arm}: {r['wall']:7.3f}s  段={r['n_segments']}")
                    (olds if arm == "old" else news).append(r)
            pairs = [(a, b) for a, b in zip(olds, news)
                     if a and b and "err" not in a and "err" not in b]
            if len(pairs) < 2:
                print(f"  ✗ 有效配对不足（{len(pairs)}）")
                fail += 1
                continue
            seg_bad = sum(1 for a, b in pairs
                          if a["n_segments"] != b["n_segments"])
            ds = [(b["wall"] / a["wall"] - 1.0) * 100.0 for a, b in pairs]
            mean = sum(ds) / len(ds)
            var = sum((d - mean) ** 2 for d in ds) / (len(ds) - 1)
            se = var ** 0.5 / len(ds) ** 0.5
            slower = sum(1 for d in ds if d > 0)
            print(f"  配对差分（B-A）%：均值 {mean:+.3f}  SE {se:.3f}  "
                  f"逐轮 {['%+.2f' % d for d in ds]}")
            print(f"  B 慢 {slower}/{len(ds)}；段数不等轮 {seg_bad}/{len(pairs)}"
                  f"（重构中性要求 0）")
            if seg_bad:
                fail += 1
            if abs(mean) > args.hard and slower > len(ds) / 2:
                print(f"  ✗ B 慢 {mean:+.2f}% 超阈值 {args.hard}% 且符号多数")
                fail += 1
            else:
                print(f"  ✓ 判定：{'中性' if abs(mean) <= args.hard else '均值超阈但符号不多数'}")
    finally:
        drop_worktree(old)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
