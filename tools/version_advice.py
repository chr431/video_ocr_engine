"""版本纪律执行器（R-version-tag / R-version-advice；宪章不变量 3）。

子命令：
  dump            导出公开 API 面 → artifacts/api_surface.json（随发版提交，
                  作为下一版对照基线）
  advice          对比上一 v-tag 的 API 面与金标 manifest → 档位建议
  verify          实际版本增量 vs 建议档位一致（零逃生）
  snapshot-bench  gzip bench/registry.jsonl → bench/snapshots/v<ver>.jsonl.gz
                  （树内保留近两版，更早出树——与探针/log 同一生命周期）
  gate-manifest   跑金标 full 并落 artifacts/gates/golden.yaml（版本+钉版+指纹+
                  退出码+用时；CI 校验其版本时效，见 _audit_ext gate_manifest）

判据（宪章 3）：公开 API 面 diff 非空 或 金标 manifest diff 非空 → minor；
否则 patch。档位是计算结果而非临场裁量。
"""

from __future__ import annotations

import gzip
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
GOLDEN = ROOT / "tests" / "golden" / "bench_baseline.json"


def _utf8_stdout() -> None:
    import io

    for s in (sys.stdout, sys.stderr):
        if isinstance(s, io.TextIOWrapper):
            s.reconfigure(encoding="utf-8")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8"
    )


def last_vtag() -> str | None:
    r = _git("describe", "--tags", "--abbrev=0", "--match", "v*")
    return r.stdout.strip() if r.returncode == 0 else None


def _git_show(ref: str, path: str) -> str | None:
    r = _git("show", f"{ref}:{path}")
    return r.stdout if r.returncode == 0 else None


def api_surface() -> dict:
    """公开 API 面：顶层公共符号 → {kind, signature}（确定性排序）。"""
    sys.path.insert(0, str(ROOT))
    import video_ocr_engine as voe

    out: dict[str, dict] = {}
    for name in sorted(dir(voe)):
        if name.startswith("_"):
            continue
        obj = getattr(voe, name)
        kind = (
            "module"
            if inspect.ismodule(obj)
            else ("class" if inspect.isclass(obj) else ("func" if callable(obj) else "other"))
        )
        entry: dict = {"kind": kind}
        if kind in ("class", "func"):
            try:
                entry["signature"] = str(inspect.signature(obj))
            except (TypeError, ValueError):
                entry["signature"] = "<unavailable>"
        out[name] = entry
    return {"version": getattr(voe, "__version__", "?"), "symbols": out}


def verdict(api_diff: bool, golden_diff: bool) -> str:
    """档位判据（宪章 3，零裁量）。"""
    return "minor" if (api_diff or golden_diff) else "patch"


def _diff_keys(cur: dict, prev: dict) -> list[str]:
    return sorted(k for k in set(cur) | set(prev) if cur.get(k) != prev.get(k))


def advice() -> int:
    tag = last_vtag()
    cur = json.loads((ARTIFACTS / "api_surface.json").read_text(encoding="utf-8"))
    if tag is None:
        print("无 v-tag 可对照（首版前）：dump 建立基线即可")
        return 0
    prev_txt = _git_show(tag, "artifacts/api_surface.json")
    if prev_txt is None:
        print(
            f"基线未立：{tag} 无 artifacts/api_surface.json——本轮 dump 并随发版提交，"
            "下一版起 advice 生效"
        )
        return 0
    prev = json.loads(prev_txt)
    api_keys = _diff_keys(cur["symbols"], prev["symbols"])
    golden_cur = GOLDEN.read_text(encoding="utf-8") if GOLDEN.is_file() else ""
    golden_prev = _git_show(tag, "tests/golden/bench_baseline.json") or ""
    golden_diff = golden_cur.strip() != golden_prev.strip()
    v = verdict(bool(api_keys), golden_diff)
    print(f"advice：{v}")
    if api_keys:
        print("  API 面变化：")
        for k in api_keys[:10]:
            print(
                f"    {k}: {prev['symbols'].get(k, '（新增）')} → {cur['symbols'].get(k, '（删除）')}"
            )
    if golden_diff:
        print("  金标 manifest 有 diff（行为/产物变更）")
    return 0


def _ver_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split("."))


def verify() -> int:
    """实际版本增量 vs 建议档位（零逃生）。无版本变化时要求两 diff 皆空。"""
    tag = last_vtag()
    sys.path.insert(0, str(ROOT))
    from video_ocr_engine.config.constants import __version__ as cur_ver

    if tag is None:
        print("无 v-tag：跳过（首版前）")
        return 0
    prev_ver = tag.lstrip("v")
    cur_map, prev_map = _ver_tuple(cur_ver), _ver_tuple(prev_ver)
    same = cur_map == prev_map
    minor_bump = cur_map[:2] != prev_map[:2]
    patch_bump = (not same) and (not minor_bump)
    cur = json.loads((ARTIFACTS / "api_surface.json").read_text(encoding="utf-8"))
    prev_txt = _git_show(tag, "artifacts/api_surface.json")
    api_diff = False
    golden_diff = False
    if prev_txt is not None:
        prev = json.loads(prev_txt)
        api_diff = bool(_diff_keys(cur["symbols"], prev["symbols"]))
        golden_cur = GOLDEN.read_text(encoding="utf-8") if GOLDEN.is_file() else ""
        golden_diff = (
            golden_cur.strip() != (_git_show(tag, "tests/golden/bench_baseline.json") or "").strip()
        )
    v = verdict(api_diff, golden_diff)
    if same:
        if api_diff or golden_diff:
            print(f"✗ 版本未动（{cur_ver}）但存在 diff——应当 bump {v}")
            return 1
        print(f"✓ {cur_ver}：无版本变化且无 diff（未发版状态）")
        return 0
    actual = "minor" if minor_bump else ("patch" if patch_bump else "?")
    if f"v{cur_ver}" not in {tg.strip() for tg in _git("tag", "--list").stdout.splitlines()}:
        print(f"✗ 版本 {cur_ver} 已 bump 但同名 tag 缺失（R-version-tag）")
        return 1
    if actual != v:
        print(
            f"✗ 乱档：实际 {actual}，判据建议 {v}（API diff={api_diff}，金标 diff={golden_diff}）"
        )
        return 1
    print(f"✓ {prev_ver} → {cur_ver}：{actual}，与判据一致")
    return 0


def snapshot_bench(keep: int = 2) -> int:
    import video_ocr_engine.config.constants as constants

    snap_dir = ROOT / "bench" / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    src = ROOT / "bench" / "registry.jsonl"
    dest = snap_dir / f"v{constants.__version__}.jsonl.gz"
    if src.is_file():
        with open(src, "rb") as f_in, gzip.open(dest, "wb") as f_out:
            f_out.write(f_in.read())
        print(f"快照：{dest.relative_to(ROOT)}（{dest.stat().st_size // 1024}KB）")
    old = sorted(
        snap_dir.glob("v*.jsonl.gz"), key=lambda p: _ver_tuple(p.stem[1:].removesuffix(".jsonl"))
    )
    for p in old[:-keep]:
        p.unlink()
        print(f"出树（git 历史保留）：{p.name}")
    return 0


def gate_manifest() -> int:
    """跑金标 full 并落 manifest——CI 据此对账版本时效（_audit_ext gate_manifest）。"""
    import video_ocr_engine.config.constants as constants

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cases = sorted((ROOT / "tests" / "golden").glob("case-*"))
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "golden" / "record.py"), "--verify"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1200,
    )
    dt = time.perf_counter() - t0
    decord_ver: str | None = None
    try:
        import decord as _decord

        decord_ver = _decord.__version__
    except Exception:  # noqa: S110 — CI 无 decord=环境事实
        pass
    tail = [ln for ln in (r.stdout or "").splitlines() if ln.strip()][-1:]
    manifest = {
        "kind": "golden",
        "engine_version": constants.__version__,
        "decord": decord_ver,
        "cases": len(cases),
        "exit": r.returncode,
        "last_line": tail[0] if tail else "",
        "seconds": round(dt, 1),
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "freshness 判据=engine_version/decord 与当前一致（_audit_ext gate_manifest）",
    }
    out = ARTIFACTS / "gates" / "golden.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    import yaml  # type: ignore[import-untyped]

    out.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    print(f"manifest：{out.relative_to(ROOT)}（exit={r.returncode}，{dt:.1f}s）")
    return r.returncode


def main(argv: list[str]) -> int:
    _utf8_stdout()
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "dump":
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "api_surface.json").write_text(
            json.dumps(api_surface(), ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        print("dump：artifacts/api_surface.json")
        return 0
    if cmd == "advice":
        return advice()
    if cmd == "verify":
        return verify()
    if cmd == "snapshot-bench":
        return snapshot_bench()
    if cmd == "gate-manifest":
        return gate_manifest()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
