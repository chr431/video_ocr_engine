"""发版清扫 —— 铁律 9「迁移清扫」的脚本化（2026-10-10 治理改革轮）。

动机
----
铁律 9 要求"依赖升级/大迁移落地时 grep 结论前提列、逐条复核"——此前是
纯人肉动作，0.21.0 清扫前积了 13 条 superseded 没翻状态。本脚本把"哪些
结论的前提钉了旧版本/哪些复评触发命中当前事件"变成一条命令的清单，
发版（bump 版本）与 fork 换代时跑。

判据（只列清单，不自动翻状态——翻状态要人看过证据）：
  1. 结论 premises/revisit 里的 X.Y.Z 版本钉，落后于当前引擎或 decord
     实际版本 → 复评候选（钉版=该结论的适用边界，边界已越过须复核）；
  2. 与当前版本**相等**的钉版 → 提示为"当轮锚点"（正常，无需动作）；
  3. 无版本钉但 revisit 含「换代/换卡/换机」类触发词 → 信息性列出。

用法
----
    python tools/conclusions_sweep.py            # 清单（退出码恒 0，供人判）
    python tools/conclusions_sweep.py --strict   # 存在落后钉版时退出 1（CI/发版钩子用）

版本事实源：引擎 = video_ocr_engine/config/constants.__version__；
decord = import decord（装不到按环境事实跳过该列，与审计 decord_pin 同语义）。
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys

import io as _io

if isinstance(sys.stdout, _io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# 钉版=显式命名空间（fork/decord/wheel→decord；引擎/engine→engine）
_PIN = re.compile(r"(fork|decord|wheel|引擎|engine)\s*v?(\d+\.\d+\.\d+)")


def _ver_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split("."))


def _load_entries() -> list[dict]:
    text = io.open(os.path.join(ROOT, "registry", "conclusions.yaml"), encoding="utf-8").read()
    entries, cur = [], None
    for line in text.split("\n"):
        m = re.match(r"- id: (\S+)", line)
        if m:
            cur = {"id": m.group(1)}
            entries.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"  (conclusion|status|premises|revisit): (.*)", line)
        if m:
            # yaml 渲染值可能带单引号包裹，钉版扫描前剥掉
            cur[m.group(1)] = m.group(2).strip().strip("'")
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="存在落后钉版时退出 1（发版钩子用）")
    args = ap.parse_args()

    import video_ocr_engine.config.constants as constants

    engine_ver = constants.__version__
    decord_ver = None
    try:
        import decord

        decord_ver = decord.__version__
    except Exception:  # noqa: BLE001,S110 — decord 未装=环境事实（CI 只装引擎），该列跳过、不判失败
        pass  # decord 未装=环境事实（CI 只装引擎），该列跳过、不判失败

    print("=" * 70)
    print(
        "发版清扫：结论钉版 vs 当前版本（引擎 %s，decord %s）"
        % (engine_ver, decord_ver or "未安装")
    )
    print("=" * 70)

    stale, current_anchor, trigger_info = [], [], []
    for e in _load_entries():
        if e.get("status") != "active":
            continue
        blob = "%s | %s" % (e.get("premises", ""), e.get("revisit", ""))
        # 钉版=显式命名空间前缀（fork/decord/wheel → decord 类；引擎/engine →
        # engine 类）。裸 x.y.z 不算钉——2026-10-12 清账轮实测：重钉注记里的
        # 时代数字被上下文词启发式误判成钉版（与宪章引用规则同一哲学：
        # 数字必须带命名空间，否则不可判定）。单测见 tests/registry/test_sweep_pins.py。
        for m in _PIN.finditer(blob):
            pin = m.group(2)
            kind = "decord" if m.group(1) in ("fork", "decord", "wheel") else "engine"
            cur = {"engine": engine_ver, "decord": decord_ver}[kind]
            if not cur:
                continue
            if _ver_tuple(pin) < _ver_tuple(cur):
                stale.append((e["id"], pin, kind, cur))
            elif pin == cur:
                current_anchor.append((e["id"], pin, kind))
        if not _PIN.search(blob) and re.search(r"换代|换卡|换机|驱动变更", e.get("revisit", "")):
            trigger_info.append((e["id"], e.get("revisit", "")[:40]))

    if stale:
        print("\n复评候选（前提钉版已落后于当前版本——翻状态前先看证据）：")
        for cid, pin, kind, cur in stale:
            print("  %-6s 钉 %-8s < 当前 %s %s" % (cid, pin, kind, cur))
    else:
        print("\n复评候选：无（没有落后于当前版本的钉版）")
    if current_anchor:
        print(
            "\n当轮锚点（钉版 == 当前版本，正常，无需动作）：%s"
            % ", ".join("%s(%s)" % (c, p) for c, p, _ in current_anchor)
        )
    if trigger_info:
        print("\n触发词命中（换代/换卡类，信息性）：")
        for cid, t in trigger_info:
            print("  %-6s %s" % (cid, t))

    print(
        "\n处理：确认失效→状态翻 superseded/dead 并出源（全文随出源提交进"
        " git 档案，检索经 commit 锚点）；仍成立→钉版改为当前版本或补注"
        "前提。两仓口径（registry/environment.yaml 钉版）由审计项"
        " decord_pin 独立守护。"
    )
    if stale and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
