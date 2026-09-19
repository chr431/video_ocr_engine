"""hybrid 发布门禁（§18 硬化）：把 §16/§17 的手工验证矩阵固化成一键脚本。

背景：死锁类 bug 两次都只在引擎口径暴露（§10.2、§16.1），且时序敏感——
零散手跑守不住，必须矩阵化重复。门禁分六步（全部子进程 + 硬超时，
退出码非 0 = 有 FAIL）：

  1 gold      金标 28 用例（tests/golden/record.py --verify）
  2 fast      三码 × hybrid+TRT（hybrid_gpu 设备路径）×2：完成 + 段数对表
  3 slow      三码 + bf16 酷刑流 × hybrid+ONNX（宿主路径）：完成 + 段数对表
  4 stress    压测 harness 三码 hybrid_gpu 全片 ×2 trials：bad=0
  5 corrupt   损坏码流三例（faststart 截断 60%/90% + 中段坏字节）× 双路径：
              完成或干净报错均可，超时/崩溃 = FAIL（验证护栏+EOF 冲刷兜底）
  6 ablation  KICK_OFF=1（禁 kick）+ h264+cpu：**期望优雅降级**（段数
              8340 不变、不挂死）。⚠️ 2026-09-19 修正：旧说明写"期望超时
              挂死（反馈清偿承重）"——该机制已随 fef3c4b 重设计删除
              （kick_guard_ 默认 0，克隆路径只由 KICK_BURST>0 启用），
              旧判据测的东西已不存在。现判据 = 无 kick 回退路径未被破坏。

用法：
  python tools/_probe_release_gate.py                     # 用已装 wheel
  python tools/_probe_release_gate.py --fork D:/Repo/decord/build-081fix
  python tools/_probe_release_gate.py --only fast,slow    # 跑子集
  python tools/_probe_release_gate.py --keep-corrupt      # 保留损坏流产物

⚠️ 段数期望表是**内容锚点**：引擎分段语义正当变更（如代表帧选择改动）会
改段数——改引擎后先单跑确认新段数再更新表；fork/dll 变更不得动段数。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

_VDIR = Path(os.environ.get("RACELOG_VIDEO_DIR", r"D:\Videos\racelog_test"))
_CORRUPT_DIR = ROOT / "bench" / "corrupt"
from env_probe import ffmpeg_bin  # 事实源（纪律：不写绝对路径）
_FFMPEG = str(ffmpeg_bin("ffmpeg"))

# (文件名, 期望段数)。段数锚点说明见模块 docstring。
# 2026-09-12 重锚：segment.merge_dense_gate 默认开后四例统一为 8340
# （重锚前 8241/8243/8241/8242，编码与重排深度间有 ±2 差异——那差异来自
# **噪声合并**，门挡掉后段结构对编码不敏感）。重锚依据：零文本丢失 +
# 逐帧准确率净 +226 帧，见 docs/log/2026-09-12-准确项.md。
E2E_CASES = {
    "test6_h264.mp4": 8340,
    "test6_hevc.mp4": 8340,
    "test6.mp4": 8340,
    "test6_h264_bf16.mp4": 8340,   # §17 酷刑流（重排深度 17）
}

_RE_E2E = re.compile(
    r"E2E (\S+) (\S+) wall=([\d.]+)s segs=(\d+) gpu_pipeline_mode=(\S+)")


def sh(args: list[str], timeout: float, env: dict | None = None,
       cwd: Path | None = None) -> tuple[int, str, str, bool]:
    """跑子进程，返回 (rc, out, err, timed_out)。"""
    try:
        p = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, env=env, cwd=cwd)
        return p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        def _s(x):
            return x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")
        return 124, _s(e.stdout), _s(e.stderr), True


def gate_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.pop("DECORD_HYBRID_DEBUG", None)
    env.pop("DECORD_HYBRID_KICK_BURST", None)
    env.pop("DECORD_HYBRID_STATS", None)
    if extra:
        env.update(extra)
    return env


def run_e2e(vid: str, backend: str, ocr: str, env: dict,
            timeout: float) -> tuple[bool, str]:
    """单次引擎 e2e。返回 (pass, 摘要)。"""
    rc, out, err, to = sh(
        [sys.executable, str(ROOT / "tools" / "_probe_e2e_mode.py"),
         vid, backend, ocr], timeout, env=env)
    if to:
        return False, "TIMEOUT"
    m = _RE_E2E.search(out)
    if not m:
        crash = "Traceback" not in err
        return False, ("CRASH(rc=%d)" % rc) if crash else "ERR(rc=%d)" % rc
    wall, segs = float(m.group(3)), int(m.group(4))
    exp = E2E_CASES.get(Path(vid).name)
    seg_ok = (exp is None or segs == exp)
    # stall 取证默认开后，健康运行不应出现 [pop-stall]（A2 的零误报门）
    stall = err.count("[pop-stall]")
    ok = rc == 0 and seg_ok and stall == 0
    note = "segs=%d%s wall=%.1fs" % (segs, "" if seg_ok else "≠%d" % exp, wall)
    if stall:
        note += " ⚠stall×%d" % stall
    return ok, note


def gen_corrupt() -> list[Path]:
    """生成损坏码流（faststart 重封装使 moov 前置，截断/坏字节后仍可开）。

    产物放 bench/corrupt/（git 忽略），名字含 test6 保证 ROI 约定命中。
    """
    if not os.path.exists(_FFMPEG):
        alt = shutil.which("ffmpeg")
        if not alt:
            raise RuntimeError("ffmpeg 不可用：%s" % _FFMPEG)
    ff = _FFMPEG if os.path.exists(_FFMPEG) else shutil.which("ffmpeg")
    _CORRUPT_DIR.mkdir(parents=True, exist_ok=True)
    src = _VDIR / "test6_h264.mp4"
    base = _CORRUPT_DIR / "test6_gate_base.mp4"
    rc, _, err, _ = sh([ff, "-y", "-i", str(src), "-c", "copy",
                        "-movflags", "+faststart", str(base)], 120)
    if rc != 0:
        raise RuntimeError("faststart 重封装失败: %s" % err[-200:])
    raw = base.read_bytes()
    out: list[Path] = []
    for name, frac in (("test6_gate_trunc60.mp4", 0.6),
                       ("test6_gate_trunc90.mp4", 0.9)):
        p = _CORRUPT_DIR / name
        p.write_bytes(raw[: int(len(raw) * frac)])
        out.append(p)
    mid = bytearray(raw)
    off = int(len(mid) * 0.55)
    for i in range(off, min(off + 4096, len(mid))):
        mid[i] ^= 0xFF
    p = _CORRUPT_DIR / "test6_gate_midcorrupt.mp4"
    p.write_bytes(bytes(mid))
    out.append(p)
    base.unlink()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", default=os.environ.get("DECORD_FORK_BUILD", ""),
                    help="指向 fork 构建树（不给了用已装 wheel）")
    ap.add_argument("--only", default="",
                    help="逗号分隔子集：gold,fast,slow,stress,corrupt,ablation")
    ap.add_argument("--timeout-e2e", type=float, default=300.0)
    ap.add_argument("--timeout-ablation", type=float, default=120.0,
                    help="消融判别的挂死等待（健康跑 ~25s 会先完成 → FAIL）")
    ap.add_argument("--keep-corrupt", action="store_true")
    args = ap.parse_args()

    env = gate_env()
    if args.fork:
        env["DECORD_LIBRARY_PATH"] = args.fork
    dll = env.get("DECORD_LIBRARY_PATH", "已装 wheel")
    steps = set(args.only.split(",")) if args.only else {
        "gold", "fast", "slow", "stress", "corrupt", "ablation"}
    results: list[tuple[str, str, bool]] = []
    t_all = time.perf_counter()

    import decord
    print("门禁环境: decord %s（dll: %s）" % (decord.__version__, dll))

    if "gold" in steps:
        t0 = time.perf_counter()
        rc, out, err, to = sh(
            [sys.executable, str(ROOT / "tests" / "golden" / "record.py"),
             "--verify"], 1800, env=env)
        m = re.search(r"verify: (\d+)/(\d+) 一致", out)
        ok = bool(m) and m.group(1) == m.group(2) and not to
        results.append(("gold", m.group(0) if m else "无输出(rc=%d,to=%s)" % (rc, to), ok))
        print("  [gold] %s (%.0fs)" % (results[-1][1], time.perf_counter() - t0))

    if "fast" in steps:
        for rep in (1, 2):
            for vid in ("test6_h264.mp4", "test6_hevc.mp4", "test6.mp4"):
                ok, note = run_e2e(vid, "hybrid", "tensorrt", env, args.timeout_e2e)
                results.append(("fast", "%s %s" % (vid, note), ok))
                print("  [fast %d] %s %s" % (rep, vid, note))

    if "slow" in steps:
        for vid in E2E_CASES:
            ok, note = run_e2e(vid, "hybrid", "cpu", env, args.timeout_e2e)
            results.append(("slow", "%s %s" % (vid, note), ok))
            print("  [slow] %s %s" % (vid, note))

    if "stress" in steps:
        t0 = time.perf_counter()
        rc, out, err, to = sh(
            [sys.executable, str(ROOT / "tools" / "_probe_stress_harness.py"),
             "--cases", "hevc-gpu,av1-gpu,h264lg-gpu",
             "--trials", "2", "--nt", "32", "--frames", "99999",
             "--timeout", "180"], 1800, env=env)
        ok = ("bad=0" in out) and not to
        tail = (out.strip().splitlines() or ["?"])[-1]
        results.append(("stress", "%s (%.0fs)" % (tail, time.perf_counter() - t0), ok))
        print("  [stress] %s" % results[-1][1])

    if "corrupt" in steps:
        try:
            files = gen_corrupt()
        except RuntimeError as e:
            results.append(("corrupt", "生成失败: %s" % e, False))
            files = []
        for p in files:
            for backend, ocr in (("hybrid", "tensorrt"), ("hybrid", "cpu")):
                rc, out, err, to = sh(
                    [sys.executable, str(ROOT / "tools" / "_probe_e2e_mode.py"),
                     str(p), backend, ocr], args.timeout_e2e, env=env)
                if to:
                    ok, note = False, "TIMEOUT"
                elif rc == 0 and _RE_E2E.search(out):
                    ok, note = True, _RE_E2E.search(out).group(0)
                elif "Traceback" in err:
                    ok, note = True, "干净报错(rc=%d)" % rc
                else:
                    ok, note = False, "CRASH(rc=%d)" % rc
                results.append(("corrupt", "%s %s/%s %s" % (
                    p.name, backend, ocr, note), ok))
                print("  [corrupt] %s" % results[-1][1])
        if not args.keep_corrupt and _CORRUPT_DIR.exists():
            shutil.rmtree(_CORRUPT_DIR, ignore_errors=True)

    if "ablation" in steps:
        # 2026-09-14（C-46）复测：泵内流序 kick 在新架构里是**纯优化而非
        # 承重**——DECORD_HYBRID_KICK_OFF=1 全禁后，离场侧 DPB 尾帧由其
        # 下一个被供 GOP 的 IDR 天然冲刷、经 stash/expected 正确交付
        #（实测 h264+cpu 段数 8340 不变、wall 持平）。旧版"期望挂死"的
        # 债务克隆承重判别随机制删除而作废。本步改为**优雅降级守护**：
        # 禁 kick 必须仍以正确段数完成——若退化成挂死/缺帧，说明无 kick
        # 回退路径被破坏（回归到"kick 是唯一活路"的脆弱形态）。
        rc, out, err, to = sh(
            [sys.executable, str(ROOT / "tools" / "_probe_e2e_mode.py"),
             "test6_h264.mp4", "hybrid", "cpu"],
            args.timeout_e2e,
            env=gate_env({"DECORD_HYBRID_KICK_OFF": "1",
                          **({"DECORD_LIBRARY_PATH": args.fork}
                             if args.fork else {})}))
        m = _RE_E2E.search(out or "")
        segs = int(m.group(4)) if m else -1
        ok = (not to) and rc == 0 and segs == E2E_CASES["test6_h264.mp4"]
        note = ("优雅降级✓ segs=%d" % segs if ok else
                ("挂死(无kick回退被破坏)" if to else "segs=%d≠8340" % segs))
        results.append(("ablation", note, ok))
        print("  [ablation] %s" % note)

    fails = [r for r in results if not r[2]]
    print("\n════ 发布门禁：%d/%d 通过（%.0fs）════" % (
        len(results) - len(fails), len(results), time.perf_counter() - t_all))
    for step, note, ok in fails:
        print("  ✗ [%s] %s" % (step, note))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
