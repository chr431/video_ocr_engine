"""视觉真值获取链（确定性部分）——把"人工校对真值"换成"视觉子代理直读 + 一致面放行"。

背景：新增测试用例时最大的成本是真值。此前的做法是用同族 OCR 模型当"真值生成器"，
失败的原因是结构性的——同族模型共享同样的系统性混淆，错得一样且**不会报错**，因此
无法自证。本链改用视觉子代理读代表帧：实测 3317 个真实裁剪上它错读 0（旧模型）或
6（更便宜的模型，0.18%），且在像素确实无法判定处**拒答**而不是猜。

分工（本工具只做确定性部分，唯一需要外部的一步是子代理读图）：

    sheets  视频 + ROI（取自 truth 头）→ 引擎分段代表帧 → 3x 裁剪 → 3 列 × 8 行样张
    plan    打印分片方案与逐片调度契约（协议单一来源，不靠人记提示词）
    〔子代理〕Read 样张 → Write 每片一个 CSV（image-analyst 的落盘契约）
    harvest 收数：宽容解析 + 顺序无关映射 + 格式守卫 + 三方比对 + 一致面分析
    emit    产出真值：原生 truth.v1（JSONL，逐帧带 state）+ 可选 RVTL 兼容 csv

数据格式：本项目的原生格式是 truth.v1（见 tools/_truth_format.py）；RVTL 的 csv 头
只作为**外部兼容格式**经 shim 读写，其格式怪癖不得渗进本工具或下游。逐帧 state
（read / inferred / unknown）是格式的核心——"读出来的"与"补出来的"必须分开记。

两条实测换来的硬约束：

1. **解析宽容，但不许静默丢行**。实测有子代理把标签连同 `f` 前缀写进文件
   （`f03532,157`）；丢掉的行若被当成"拒答"计数，格式故障就会伪装成未知面，
   把错误藏起来。故无法解析的行一律上抛。
2. **不依赖子代理的顺序与标签**。实测同一批样张，有的按列优先读、有的按行优先；
   帧号前缀也会被漏抄（`f01625` 抄成 `625`）。故映射按"精确标签 → 唯一后缀 →
   位置兜底"三级对齐；未映射项单独计数，绝不混入"错"。

自动放行判据：真值只在视觉与引擎**读数一致**的帧上自动放行，分歧帧与任一方拒答
的帧进审查面。实测 3317 帧上「一致且都错」为 0 例，故自动放行面错误率为 0，
审查面 = 3.3–3.6%——这就是把人工面从"全部"压到"分歧处"的那一刀。

用法：
  # 已有真值头（复用其视频/ROI/帧范围）
  python tools/truth_build.py --truth-csv <race truth csv> sheets [--stride 4]
  python tools/truth_build.py --truth-csv <...> plan [--sheets-per-agent 4]
  python tools/truth_build.py --truth-csv <...> harvest --run <第一遍> [--run <第二遍> ...]
  python tools/truth_build.py --truth-csv <...> emit --run <目录>... --out <csv>

  # 全新视频（尚无真值）：显式给素材与 ROI 起链
  python tools/truth_build.py --case <名> --video <路径> --roi x1,y1,x2,y2 \
      --frame-start N --frame-end M sheets

协议要点（默认即盲读；`--no-blind` 只用于对照实验）：
  · 样张全局乱序、只标乱序 id，`key.json`（id→帧）不进子代理调度面
  · 调度文案不得给出来源/单位/位数/顺序等上下文，也不得暗示可跨格推断
  · 判决读两遍（两个独立子代理）取"两读一致 ∧ 与引擎一致"为放行面，其余进审查面

退出码：0=通过；2=参数/输入错误；3=格式守卫失败（存在无法解析的行）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import _truth_format as tf  # noqa: E402  自有真值格式 + RVTL 兼容 shim

SHEET_COLS = 3
SHEET_ROWS = 8
CROP_SCALE = 3
UNKNOWN = "?"


# ────────────────────────── truth 头解析 ──────────────────────────


def parse_truth_header(path: Path) -> dict:
    """读 RVTL 头（薄封装：格式知识住在 tools/_truth_format.py 的 shim 里）。"""
    return tf.parse_rvlt_header(path.read_text(encoding="utf-8-sig"))


def case_dir(work: Path, args: argparse.Namespace) -> Path:
    """用例工作目录名：优先 truth 文件名（`test5_ref.csv` → `test5_ref`），否则用 --case。"""
    if getattr(args, "truth_csv", None) is not None:
        return work / args.truth_csv.stem
    if not getattr(args, "case", None):
        raise SystemExit("需要 --truth-csv 或 --case 之一来命名用例")
    return work / args.case


def resolve_source(args: argparse.Namespace) -> dict:
    """取样张来源：已有真值头（复用其视频/ROI/帧范围）或新视频显式参数。

    新视频尚无真值时走显式参数——这是本链对"全新素材"的入口。
    """
    if getattr(args, "truth_csv", None) is not None:
        head = parse_truth_header(args.truth_csv)
        return {
            "video": head["video"],
            "roi": head["roi"],
            "frame_start": head["frame_start"],
            "frame_end": head.get("frame_end"),
        }
    if not (args.video and args.roi):
        raise SystemExit("新视频需给出 --video 与 --roi（x1,y1,x2,y2）")
    roi = tuple(int(v) for v in str(args.roi).split(","))
    if len(roi) != 4:
        raise SystemExit("--roi 需为 x1,y1,x2,y2 四个整数")
    return {
        "video": args.video,
        "roi": roi,
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
    }


# ────────────────────────── 样张生成 ──────────────────────────


def cmd_sheets(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image, ImageDraw

    from env_probe import video_dir  # 外部路径唯一出口

    head = resolve_source(args)
    video = Path(head["video"])
    if not video.is_absolute():
        video = video_dir() / head["video"]
    if not video.exists():
        print("找不到视频：%s" % video)
        return 2
    out = case_dir(args.work, args)
    sheets_dir = out / "sheets"
    crops_dir = out / "crops"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    from video_ocr_engine import FieldExtractor

    ex = FieldExtractor(
        str(video),
        head["roi"],
        frame_start=head["frame_start"],
        frame_end=head.get("frame_end") or head["frame_start"],
        keep_crops=False,
    )
    res = ex.extract()
    segs = list(res.segments)
    reps = [s.rep_frame for s in segs][:: args.stride]
    print("分段 %d 个；取代表帧 %d 个（stride=%d）" % (len(segs), len(reps), args.stride))
    (out / "engine.csv").write_text(
        "\n".join("%d,%s" % (s.rep_frame, s.text or "") for s in segs) + "\n", encoding="utf-8"
    )

    # 盲读协议（默认）：全局打乱 + 只给乱序 id。领域上下文与时间顺序都会抬高读数，
    # 见 C-95（带上下文 0.18% 错读 vs 盲读 0.75%），故默认采用诚实口径；
    # 乱序 id → 帧号的对应表落在 key.json，**不得进入子代理的调度面**。
    order = list(reps)
    if args.blind:
        import random

        random.Random(args.seed).shuffle(order)
    ids = list(range(1, len(order) + 1))
    label_of = {i: (("%05d" % i) if args.blind else ("f%05d" % f)) for i, f in zip(ids, order)}
    key = {str(i): f for i, f in zip(ids, order)}

    import decord

    x1, y1, x2, y2 = head["roi"]
    vr = decord.VideoReader(str(video), ctx=decord.cpu(0))
    id_of = {f: i for i, f in zip(ids, order)}
    for i in range(0, len(order), 64):
        idx = order[i : i + 64]
        frames = vr.get_batch(idx).asnumpy()
        for f, frame in zip(idx, frames):
            crop = Image.fromarray(np.asarray(frame[y1 : y2 + 1, x1 : x2 + 1]))
            crop = crop.resize((crop.width * CROP_SCALE, crop.height * CROP_SCALE), Image.LANCZOS)
            crop.save(crops_dir / ("c%05d.png" % id_of[f]))

    font = _load_font(22)
    font_s = _load_font(16)
    per = SHEET_COLS * SHEET_ROWS
    w, h = Image.open(crops_dir / ("c%05d.png" % ids[0])).size
    manifest: dict[str, list[int]] = {}
    for i in range(0, len(ids), per):
        chunk = ids[i : i + per]
        name = "%s_s%02d" % (args.case_prefix, i // per + 1)
        sheet = Image.new(
            "RGB", (SHEET_COLS * (w + 100), 34 + SHEET_ROWS * (h + 8)), (255, 255, 255)
        )
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (6, 6),
            "%s - the small text at the RIGHT of each crop is its id" % name,
            fill=(0, 0, 0),
            font=font_s,
        )
        for j, cid in enumerate(chunk):
            x = (j // SHEET_ROWS) * (w + 100)
            y = 34 + (j % SHEET_ROWS) * (h + 8)
            sheet.paste(Image.open(crops_dir / ("c%05d.png" % cid)), (x, y))
            draw.text((x + w + 6, y + h // 2 - 12), label_of[cid], fill=(0, 0, 0), font=font)
        sheet.save(sheets_dir / (name + ".png"))
        manifest[name] = chunk
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(
        "样张 %d 张 → %s（blind=%s seed=%d；含 manifest.json/key.json/engine.csv；裁剪 %dx%d）"
        % (len(manifest), sheets_dir, args.blind, args.seed, w, h)
    )
    return 0


def _load_font(size: int):
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ────────────────────────── 收数 ──────────────────────────


def parse_shard(path: Path) -> tuple[list[tuple[int, str]], list[str]]:
    """解析一个分片 CSV。

    宽容：标签可带 `f` 前缀与前导零。但**不静默丢行**——不可解析的行原样返回，
    由调用方判为格式故障。把坏行丢掉会让它伪装成"拒答"，把错误藏进未知面。
    """
    rows: list[tuple[int, str]] = []
    bad: list[str] = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.strip():
            continue
        m = re.fullmatch(r"\s*f?0*(\d+)\s*,\s*(\d+|\?)\s*", ln)
        if m:
            rows.append((int(m.group(1)), m.group(2)))
        else:
            bad.append(ln)
    return rows, bad


def sheet_range(name: str) -> tuple[int, int] | None:
    """从分片文件名 `<case>_s<first>_<last>.csv` 取出样张编号区间（纯数字）。"""
    m = re.search(r"_s(\d+)_(\d+)\.csv$", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def sheet_no(name: str) -> int:
    """样张名 `<case>_s<NN>` 的编号。"""
    m = re.search(r"_s(\d+)$", name)
    if m is None:
        raise ValueError("样张名不含 _s<NN>：%s" % name)
    return int(m.group(1))


def assemble(manifest: dict[str, list[int]], run_dir: Path) -> tuple[dict[int, str], dict]:
    """把各分片汇总为 帧→读数。映射与子代理的读序/标签无关。"""
    stat = dict(shards=0, rows=0, bad_lines=0, exact=0, suffix=0, pos=0, dup=0, unmapped=0)
    names = list(manifest)
    vision: dict[int, str] = {}
    for shard in sorted(run_dir.glob("*.csv")):
        rng = sheet_range(shard.name)
        if rng is None:
            continue
        first, last = rng
        idx = {sheet_no(sn): k for k, sn in enumerate(names)}
        i, j = idx.get(first), idx.get(last)
        if i is None or j is None:
            print("  跳过（区间不在 manifest 内）：%s" % shard.name)
            continue
        expect = [f for sn in names[i : j + 1] for f in manifest[sn]]
        got, bad = parse_shard(shard)
        stat["shards"] += 1
        stat["rows"] += len(got)
        stat["bad_lines"] += len(bad)
        if bad:
            print("  !! %s 有 %d 行无法解析，例：%r" % (shard.name, len(bad), bad[:2]))
        if len(got) != len(expect):
            print("  !! %s 行数 %d != 预期 %d" % (shard.name, len(got), len(expect)))
        pool, resolved = set(expect), {}
        for lab, val in got:  # 一级：精确标签
            if lab in pool:
                resolved[lab] = val
                pool.discard(lab)
                stat["exact"] += 1
        still = []
        for lab, val in [(a, b) for a, b in got if a not in resolved]:
            cand = [e for e in pool if str(e).endswith(str(lab))]  # 二级：唯一后缀
            if len(cand) == 1:
                resolved[cand[0]] = val
                pool.discard(cand[0])
                stat["suffix"] += 1
            else:
                still.append((lab, val))
        rest = sorted(pool)  # 三级：位置兜底
        stat["pos"] += len(still)
        stat["unmapped"] += max(0, len(still) - len(rest))
        for e, (_lab, val) in zip(rest, still):
            resolved[e] = val
        stat["dup"] += sum(1 for e in resolved if e in vision)
        vision.update(resolved)
    return vision, stat


def _num(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def load_engine(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    out = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.strip():
            continue
        f, _, v = ln.partition(",")
        if f.isdigit():
            out[int(f)] = v
    return out


def score(
    vision: dict[int, str], engine: dict[int, str], truth: dict[int, int], reps: list[int]
) -> dict[str, Any]:
    """三方比对 + 一致面分析（自动放行面的错误率就是相关误差率）。"""
    res: dict[str, Any] = dict(
        n=len(reps),
        agree=0,
        agree_bad=0,
        split=0,
        v_unk=0,
        x_missing=0,
        v_ex=0,
        v_near=0,
        v_bad=[],
        e_bad=[],
        e_n=0,
        fixable=0,
    )
    for f in reps:
        t = truth.get(f)
        if t is None:
            continue
        v_raw, e_raw = vision.get(f), engine.get(f)
        v = None if v_raw in (None, UNKNOWN) else _num(v_raw)
        e = _num(e_raw)
        # 计数语义：视觉拒答（v_unk）与"引擎没有可交叉的读数"（x_missing）是两件事，
        # 混在一起会把"读了但无处交叉"冒充成"没读出来"。
        if v is None:
            res["v_unk"] += 1
        if e is None:
            res["x_missing"] += 1
        if v is None or e is None:
            if e is not None and abs(e - t) > 1:
                res["e_bad"].append((f, e, t))
                res["e_n"] += 1
            continue
        if v == e:
            res["agree"] += 1
            if v != t:
                res["agree_bad"] += 1
        else:
            res["split"] += 1
            if e != t:
                res["e_bad"].append((f, e, t))
                res["e_n"] += 1
                res["fixable"] += 1
        if abs(v - t) == 0:
            res["v_ex"] += 1
        elif abs(v - t) == 1:
            res["v_near"] += 1
        else:
            res["v_bad"].append((f, v, t))
    if not engine:
        res["agree"] = res["split"] = -1  # 未提供引擎读数：不做一致面分析
    return res


# ────────────────────────── 子命令 ──────────────────────────


def _load_case(args: argparse.Namespace):
    """载入用例的 manifest/key/engine；返回 (case_dir, manifest, id→帧, engine)。"""
    out = case_dir(args.work, args)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    keyp = out / "key.json"
    key = json.loads(keyp.read_text(encoding="utf-8")) if keyp.exists() else {}
    return out, man, {int(k): v for k, v in key.items()}, load_engine(out / "engine.csv")


def _reads_to_frames(man: dict, key: dict[int, int], run: Path) -> tuple[dict[int, str], dict]:
    """一个分片目录 → {帧: 读数}。

    manifest 的格子是 id（盲读）或帧号（非盲读）；有 key 时按 id→帧 还原。
    还原不经过任何顺序假设，故子代理的读序与标签转录错误都不影响映射。
    """
    by_id, stat = assemble(man, run)
    if not key:
        return {i: v for i, v in by_id.items()}, stat
    return {key[i]: v for i, v in by_id.items() if i in key}, stat


def cmd_plan(args: argparse.Namespace) -> int:
    out, man, key, _ = _load_case(args)
    names = list(man)
    per = args.sheets_per_agent
    print(
        "用例：%s   样张 %d 张，裁剪 %d 个%s"
        % (
            out.name,
            len(names),
            sum(len(v) for v in man.values()),
            "（盲读：乱序 id，key.json 不入调度面）" if key else "",
        )
    )
    print("分片计划（每代理 %d 张 = %d 个裁剪）：" % (per, per * SHEET_COLS * SHEET_ROWS))
    for k in range(0, len(names), per):
        grp = names[k : k + per]
        print(
            "  %s_s%02d_%02d.csv   %s..%s   %d 行"
            % (
                out.name,
                sheet_no(grp[0]),
                sheet_no(grp[-1]),
                grp[0],
                grp[-1],
                sum(len(man[g]) for g in grp),
            )
        )
    print("\n调度契约（每个分片一个子代理；协议单一来源，勿临时改写）：")
    print("  · 只列出该分片的样张绝对路径；输出=上面那个 CSV（UTF-8 无 BOM、无表头、`id,值`）")
    print("  · 明确禁止读其它文件（尤其 engine.csv / key.json / 真值 CSV）——否则实验作废")
    print("  · 看不清写 `%s`，不要猜；读序=每张图内先第一列 8 个，再第二列，再第三列" % UNKNOWN)
    print("  · **不要**给出素材来源、单位、字体、位数、顺序等上下文，也不要暗示可跨格推断——")
    print("    实测上下文会把错读压掉约 3/4（C-95），那是拿连续性替读者补答案，不是读数")
    print("  · 回复只要一行摘要（路径+条数+异常），数据只走文件")
    print("  · 结论按两次独立盲读 + 与引擎一致的闸门读取（harvest 支持多 --run）")
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    out, man, key, engine = _load_case(args)
    runs = list(args.run)
    reps = [key[i] for sn in man for i in man[sn]] if key else [f for sn in man for f in man[sn]]
    truth = _load_race_truth(args.truth_csv) if args.truth_csv else {}

    reads, stats = [], []
    for r in runs:
        v, st = _reads_to_frames(man, key, r)
        reads.append(v)
        stats.append(st)
    total_bad = sum(s["bad_lines"] for s in stats)
    for r, st in zip(runs, stats):
        print(
            "  %s：分片 %d 行 %d（无法解析 %d）映射 精确 %d 后缀 %d 位置兜底 %d 未映射 %d"
            % (
                r,
                st["shards"],
                st["rows"],
                st["bad_lines"],
                st["exact"],
                st["suffix"],
                st["pos"],
                st["unmapped"],
            )
        )

    if len(reads) == 1:
        res = score(reads[0], engine, truth, reps)
        _report_single(res, out)
    else:
        res = _report_stack(reads, engine, truth, reps)
    if total_bad:
        print("格式守卫：失败——存在无法解析的行，先修复再采信上面的数字")
        return 3
    return 0


def _report_single(res: dict, out: Path) -> None:
    den = res["v_ex"] + res["v_near"] + len(res["v_bad"]) + res["v_unk"]
    print(
        "视觉（单读）：读对 %d (%6.2f%%)  差1 %d  错读 %d  拒答 %d (%5.2f%%)"
        % (
            res["v_ex"],
            100 * res["v_ex"] / max(1, den),
            res["v_near"],
            len(res["v_bad"]),
            res["v_unk"],
            100 * res["v_unk"] / max(1, den),
        )
    )
    if res["v_bad"]:
        print("  错读明细（帧,读数,真值）:", res["v_bad"][:15])
    if res["agree"] >= 0:
        print(
            "  一致面 %d (%.2f%%)「一致且都错」%d；分歧 %d；拒答 %d；引擎无读数 %d"
            % (
                res["agree"],
                100 * res["agree"] / max(1, res["n"]),
                res["agree_bad"],
                res["split"],
                res["v_unk"],
                res["x_missing"],
            )
        )
    else:
        print("（未找到 %s，跳过一致面分析）" % (out / "engine.csv"))


def _report_stack(reads: list[dict[int, str]], engine: dict, truth: dict, reps: list) -> dict:
    """K 次独立读 + 引擎的闸门栈（实测口径：C-95/blind_repeat）。"""
    n = acc = acc_bad = split = abst = review = 0
    bad = []
    for f in reps:
        vals = [_num(v.get(f)) for v in reads]
        e = _num(engine.get(f))
        t = truth.get(f)
        n += 1
        if any(v is None for v in vals):
            abst += 1
        elif len(set(vals)) > 1:
            split += 1
        if vals[0] is not None and len(set(vals)) == 1 and vals[0] == e:
            acc += 1
            if t is not None and vals[0] != t:
                acc_bad += 1
                bad.append((f, vals[0], t))
        else:
            review += 1
    print("K=%d 次独立读 + 引擎一致性闸门：" % len(reads))
    print(
        "  放行 %d (%.2f%%)，其中错 %d (%.4f%%)；审查面 %d (%.2f%%) = 拒答 %d + 分歧 %d + 引擎不符 %d"
        % (
            acc,
            100 * acc / max(1, n),
            acc_bad,
            100 * acc_bad / max(1, acc),
            review,
            100 * review / max(1, n),
            abst,
            split,
            review - abst - split + min(abst, split),
        )
    )
    if bad:
        print("  闸门残余错读（两读一致 ∧ 引擎一致 但错）:", bad[:12])
    return {
        "agree": acc,
        "agree_bad": acc_bad,
        "split": split,
        "v_unk": abst,
        "x_missing": 0,
        "v_ex": 0,
        "v_near": 0,
        "v_bad": [],
        "n": n,
        "e_n": 0,
        "fixable": 0,
    }


def cmd_emit(args: argparse.Namespace) -> int:
    out, man, key, engine = _load_case(args)
    reads = [_reads_to_frames(man, key, r)[0] for r in args.run]
    reps = [key[i] for sn in man for i in man[sn]] if key else [f for sn in man for f in man[sn]]
    # 头部必须与 Race 真值文件同形（逗号分隔的 key=value，roi 内联四数）——
    # 实测：换一种写法（空格分隔）会让既有消费方（e2e_smoke --truth）静默读不到
    # frame_start，于是拿错误的帧窗口去对齐，表现为"全部不匹配"的假失败。
    if args.truth_csv is not None:
        head = parse_truth_header(args.truth_csv)
        video, roi = head["video"], head["roi"]
        fs, fe = head["frame_start"], head.get("frame_end")
    else:
        if not args.roi:
            raise SystemExit("emit 需要 --truth-csv 或 --roi/--video 来写头部")
        video = args.video or out.name
        roi = tuple(int(v) for v in str(args.roi).split(","))
        fs, fe = args.frame_start, args.frame_end
    records, known = [], 0
    for f in sorted(reps):
        vals = [v for v in (_num(x.get(f)) for x in reads) if v is not None]
        eng = _num(engine.get(f)) if engine else None
        if not vals:
            records.append(tf.record(f, None, "unknown", engine=eng))
        elif len(set(vals)) == 1:
            records.append(tf.record(f, vals[0], "read", reads=vals, engine=eng))
            known += 1
        else:  # 多读分歧：不投票取值，只留读数痕迹供审查
            records.append(tf.record(f, None, "unknown", reads=vals, engine=eng))
    meta = tf.build_meta(
        video,
        tuple(roi),
        fs,
        fe if isinstance(fe, int) else None,
        blind=bool(key),
        reads=len(reads),
        source=",".join(Path(r).name for r in args.run),
    )
    tf.write_native(args.out, meta, records)
    unknown = len(records) - known
    print(
        "写出 %s（原生 %s）：%d 帧（read %d / unknown %d）"
        % (args.out, tf.SCHEMA, len(records), known, unknown)
    )
    if args.rvlt is not None:
        tf.write_rvlt(args.rvlt, meta, records)
        print("  兼容导出 %s（RVTL 形状，供既有消费方与语料比对）" % args.rvlt)
    return 0


def _load_race_truth(path: Path) -> dict[int, int]:
    """读 RVTL 真值的数据行（经 shim）。"""
    _head, rows, _bad = tf.read_rvlt(path)
    return {f: v for f, v in rows if v is not None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="视觉真值获取链（确定性部分）")
    ap.add_argument(
        "--truth-csv",
        type=Path,
        default=None,
        help="已有的 Race 真值 CSV（提供视频/ROI/帧范围；新视频无真值时改用 --video/--roi）",
    )
    ap.add_argument("--case", default=None, help="用例名（无 --truth-csv 时用于命名工作目录）")
    ap.add_argument("--video", default=None, help="新视频（可绝对路径；相对则按视频目录解析）")
    ap.add_argument("--roi", default=None, help="x1,y1,x2,y2（含端点，与 truth 头同义）")
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument(
        "--work",
        type=Path,
        default=None,
        help="工作目录（默认 <语料同级>/vision_truth_build——不写进语料目录，也不进仓库工作树）",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sheets", help="生成代表帧裁剪与样张（需视频/引擎）")
    p.add_argument("--stride", type=int, default=1, help="分段抽样步长")
    p.add_argument("--case-prefix", default=None, help="样张名前缀（默认取用例名）")
    p.add_argument(
        "--no-blind",
        dest="blind",
        action="store_false",
        help="关闭盲读：按时间顺序排列并标注真实帧号（仅用于对照实验，不用于建真值）",
    )
    p.add_argument("--seed", type=int, default=20261016, help="盲读打乱种子（可复现）")
    p.set_defaults(func=cmd_sheets, blind=True)

    p = sub.add_parser("plan", help="打印分片方案与调度契约")
    p.add_argument("--sheets-per-agent", type=int, default=4)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("harvest", help="收数并比对（模型无关；多个 --run = K 次独立读 + 闸门栈）")
    p.add_argument(
        "--run", type=Path, action="append", required=True, help="分片 CSV 目录，可给多次"
    )
    p.set_defaults(func=cmd_harvest)

    p = sub.add_parser("emit", help="产出原生真值（truth.v1 JSONL）+ 可选 RVTL 兼容导出")
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True, help="原生格式输出（truth.v1 JSONL）")
    p.add_argument(
        "--rvlt",
        type=Path,
        default=None,
        help="额外导出 RVTL 形状 csv（兼容既有语料与 e2e_smoke --truth）",
    )
    p.set_defaults(func=cmd_emit)

    args = ap.parse_args(argv)
    if args.work is None:
        # 真值/样张是外部语料的派生物：与语料**同级**而不是写进语料目录
        # （语料目录是受保护的手工校对资产），同时也不写进仓库工作树。
        from env_probe import video_dir

        args.work = video_dir().parent / "vision_truth_build"
    if getattr(args, "case_prefix", None) is None:
        args.case_prefix = case_dir(args.work, args).name
    if args.frame_end is None:
        args.frame_end = (
            parse_truth_header(args.truth_csv).get("frame_end") if args.truth_csv else None
        )
    return int(args.func(args))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
