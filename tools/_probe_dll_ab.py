"""_probe_dll_ab.py —— 换 DLL 的交错 A/B（两个 decord 构建之间的对照）。

为什么单独一个探针
------------------
`tools/bench.py ab` 是**同进程内**跑两臂（为 PI-15 的插桩成本判据设计），
而换 DLL 必须**每臂一个新进程**（DLL 一旦加载不可替换）。本探针补的就是
这个缺口：把"手工换 DLL + md5 校验 + 前后对照"固化成可复现的交错实验。

口径（与 bench ab 一致的部分照抄）
--------------------------------
- 每臂独立子进程；**逐轮交错**，并**每轮轮转先后**（拉丁方，消位置效应）。
- 统计量：同轮**配对差分**的**均值** + SE；判失败/判效应另需**符号多数一致**
  （≥70%）。可分辨下限取 PI-15 标定值（`bench.py` 的 `PI15_LIMITS["std_pct"]`）。
- 冷/热分开打印：`--inner` 轮内取均值，第 1 轮为冷启动。

用法
----
把两个候选 DLL 放到**同一个含 FFmpeg 运行库的目录**（如
`D:\\Repo\\decord\\build-081fix`），命名任意（如 `decord.dll.varA`/`.varB`）：

    python tools/_probe_dll_ab.py --dll-a <dir>\\decord.dll.varA \\
        --dll-b <dir>\\decord.dll.varB --config hevc-hybrid --window 3000 \\
        --rounds 6 --inner 2

探针会把所选变体拷成 `<dir>\\decord.dll`（每次拷贝后校验 md5），并在子进程里以
`DECORD_LIBRARY_PATH=<dir>` 运行引擎 —— 因此**别在实验期间跑别的 hybrid 任务**。
结束时恢复为 `--restore`（缺省 = 最初那份 `decord.dll`）。

⚠️ **本探针的噪声带比 `bench ab` 大一个量级（实测 sd 4.6% vs 0.5%）**——因为每臂
都是**冷启动的新进程**（DLL 必须重新加载）。2026-09-13 实测（hevc-hybrid /
3000 帧 / rounds=4 / inner=2）：配对差分 −3.16 −2.89 +5.90 +3.50 → 均值 +0.837%、
sd 4.567%、SE 2.283%、符号 2/4 ⇒ 判"噪声内"。含义：
- 判 **≥5%** 的效应：4 轮足够（符号一致性即可）；
- 判 **1% 级**效应：需 **≥20 轮**（或加大 `--inner`）；
- 判"默认路径未被改动"这类**等价性**主张：只需看是否落在噪声内（本探针正合适）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import bench  # noqa: E402  （复用 CONFIGS/VIDS/ROI 与 _round，避免第二套口径）

OUT = ROOT / "bench" / "dll_ab.json"

WORKER = r'''
import json, os, sys, time
sys.path.insert(0, os.environ["PROBE_TOOLS"])
sys.path.insert(0, os.environ["PROBE_ROOT"])
import bench
cfg, window, ocr, keep, telemetry, dec, fw = sys.argv[1:8]
r = bench._round(cfg, int(window), telemetry, keep == "1", ocr,
                 decode_override=dec, fill_width=(int(fw) if fw != "-" else None))
print(json.dumps(r))
'''


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def run_arm(dll: Path, target: Path, cfg, window, ocr, keep, telemetry,
            decode, fill_width) -> dict:
    """换 DLL → 跑一次 extract（子进程）→ 返回 {wall, ...}。"""
    import shutil
    shutil.copy2(dll, target)
    want = md5(dll)
    got = md5(target)
    assert want == got, "DLL 拷贝校验失败：%s vs %s" % (want, got)
    env = dict(os.environ)
    env["PROBE_TOOLS"] = str(TOOLS)
    env["PROBE_ROOT"] = str(ROOT)
    env["DECORD_LIBRARY_PATH"] = str(target.parent)
    p = subprocess.run(
        [sys.executable, "-c", WORKER, cfg, str(window), ocr,
         "1" if keep else "0", telemetry, decode or "",
         "-" if fill_width is None else str(fill_width)],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError("臂失败(rc=%d): %s" % (p.returncode,
                                                 (p.stderr or "")[-400:]))
    r = json.loads(p.stdout.strip().splitlines()[-1])
    r["dll_md5"] = want
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll-a", required=True)
    ap.add_argument("--dll-b", required=True)
    ap.add_argument("--target", default="", help="被替换的 decord.dll（缺省=A 同目录下的 decord.dll）")
    ap.add_argument("--restore", default="", help="实验结束后恢复的 DLL（缺省=实验开始时的 decord.dll）")
    ap.add_argument("--keep-last", action="store_true",
                    help="结束后不恢复，保留最后一个臂的 DLL（默认会恢复）")
    ap.add_argument("--config", default="hevc-hybrid")
    ap.add_argument("--window", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--ocr-backend", default="tensorrt")
    ap.add_argument("--keep-crops", action="store_true")
    ap.add_argument("--telemetry", default="std")
    ap.add_argument("--decode", default="", help="覆盖 decode_backend")
    ap.add_argument("--fill-width", type=int, default=None)
    args = ap.parse_args()

    a, b = Path(args.dll_a), Path(args.dll_b)
    target = Path(args.target) if args.target else a.parent / "decord.dll"
    backup = Path(args.restore) if args.restore else None
    for p in (a, b, target):
        assert p.exists(), "缺文件：%s" % p
    assert a.parent == b.parent == target.parent, \
        "两个变体与 decord.dll 必须在同一目录（FFmpeg 运行库也在这儿）"

    started = md5(target)
    # 实验开始时把 target 另存一份，结束时默认恢复 —— 否则会**留下最后一个臂的
    # DLL**（实测踩过：goldens 复核时误验了旧 DLL）。`--keep-last` 可关闭。
    pre = Path(str(target) + ".pre-ab")
    import shutil as _sh
    _sh.copy2(target, pre)
    print("DLL A: %s  md5=%s" % (a.name, md5(a)))
    print("DLL B: %s  md5=%s" % (b.name, md5(b)))
    print("目标 : %s（实验前 md5=%s，已另存 %s）" % (target, started, pre.name))
    print("配置 : %s window=%d rounds=%d inner=%d ocr=%s decode=%s"
          % (args.config, args.window, args.rounds, args.inner,
             args.ocr_backend, args.decode or "-"))

    per = {"A": [], "B": []}
    try:
        for i in range(args.rounds):
            order = ("A", "B") if i % 2 == 0 else ("B", "A")   # 轮转先后
            for arm in order:
                dll = a if arm == "A" else b
                ws = []
                for _ in range(args.inner):
                    r = run_arm(dll, target, args.config, args.window,
                                args.ocr_backend, args.keep_crops,
                                args.telemetry, args.decode, args.fill_width)
                    ws.append(r["wall"])
                w = statistics.fmean(ws)
                per[arm].append(w)
                print("  [r%02d] %s  %.4fs%s" % (
                    i, arm, w, "" if args.inner == 1 else
                    "  （%d 次均值）" % args.inner))
                if args.cooldown and i < args.rounds - 1:
                    import time as _t
                    _t.sleep(args.cooldown)
    finally:
        import shutil as _sh2
        if args.keep_last:
            print("--keep-last：目标保持为最后一次实验用的 DLL（%s）" % md5(target))
        else:
            src = backup if backup is not None else pre
            _sh2.copy2(src, target)
            print("已恢复 %s（md5=%s）" % (src.name, md5(target)))

    d = [(bb - aa) / aa * 100.0 for aa, bb in zip(per["A"], per["B"])]
    n = len(d)
    mean = statistics.fmean(d)
    sd = statistics.stdev(d) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n else 0.0
    pos = sum(1 for x in d if x > 0)
    need = max(2, int(round(n * 0.70 + 0.5)))
    signs = pos >= need or (n - pos) >= need
    floor = bench.PI15_LIMITS["std_pct"]
    resolved = abs(mean) > floor
    print("\n配对差分（B 相对 A，%%）：%s" % " ".join("%+.2f" % x for x in d))
    print("  A min %.4fs / B min %.4fs" % (min(per["A"]), min(per["B"])))
    print("  均值 %+.3f%%  中位 %+.3f%%  sd %.3f%%  SE %.3f%%  符号 %d正/%d负（需 %d 同向）"
          % (mean, statistics.median(d), sd, se, pos, n - pos, need))
    if not resolved:
        print("判定：**不可判定**（|均值| %.3f%% ≤ 本机可分辨下限 %.2f%%）" % (abs(mean), floor))
    elif not signs:
        print("判定：**噪声内**（超下限但符号不一致 %d/%d）" % (pos, n))
    else:
        who = "B 更快" if mean < 0 else "A 更快"
        print("判定：**真效应** —— %s（%.2f%%，符号一致 %d/%d）"
              % (who, abs(mean), max(pos, n - pos), n))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"a": str(a), "b": str(b), "target": str(target),
                   "md5_a": md5(a), "md5_b": md5(b), "started_md5": started,
                   "config": args.config, "window": args.window,
                   "rounds": args.rounds, "inner": args.inner,
                   "walls": per, "diffs": d, "mean": mean, "se": se,
                   "signs_pos": pos, "resolved": resolved, "signs": signs},
                  f, ensure_ascii=False, indent=1)
    print("落盘", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
