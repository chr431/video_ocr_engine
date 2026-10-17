"""视觉真值获取链的门禁（tools/truth_build.py）。

这一链取代的是"人工逐帧校对真值"，所以它自己出错的代价是**无声污染真值**——
比没有真值更坏。故本模块钉死三件事：

1. 解析宽容但**不静默丢行**：坏行必须上抛（否则格式故障会伪装成"拒答"，
   把错误藏进未知面里）。守卫必须能失败——负例见 test_bad_line_is_not_swallowed。
2. 映射与子代理的读序/标签无关：实测同一批样张有按列优先读、有按行优先读，
   标签也会被漏抄前缀（f01625 → 625）。
3. 计数的语义不能混：视觉拒答 ≠ 引擎没有可交叉的读数 ≠ 错读。

全部合成数据，不需要视频/GPU（R-pytest-offline）。
"""

from __future__ import annotations

import json
import sys

import pytest
from _paths import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import truth_build as tb  # tools/ 惯例：无包结构，路径注入后直名导入


# ────────────────────────── 解析：宽容 + 不静默 ──────────────────────────


def test_parse_shard_accepts_f_prefix_and_leading_zeros(tmp_path):
    """实测有子代理把标签连同 f 前缀原样写入（f03532,157）——必须照样读进来。"""
    p = tmp_path / "s.csv"
    p.write_text("f03532,157\n3536,158\n003537,159\n3538,?\n\n", encoding="utf-8")
    rows, bad = tb.parse_shard(p)
    assert bad == []
    assert rows == [(3532, "157"), (3536, "158"), (3537, "159"), (3538, "?")]
    assert tb.parse_shard(p)[0][3][1] == tb.UNKNOWN  # 拒答标记原样保留


def test_bad_line_is_not_swallowed(tmp_path):
    """守卫负测：坏行必须出现在 bad 里（而不是被丢掉、遂被当成"拒答"）。"""
    p = tmp_path / "s.csv"
    p.write_text("3532,157\n3536: 158\n3537,159\n", encoding="utf-8")
    rows, bad = tb.parse_shard(p)
    assert rows == [(3532, "157"), (3537, "159")]
    assert bad == ["3536: 158"], "坏行被吞掉了——格式故障会伪装成未知面"


def test_sheet_range_and_sheet_no():
    assert tb.sheet_range("test5_ref_s401_404.csv") == (401, 404)
    assert tb.sheet_range("note.txt") is None
    assert tb.sheet_no("test5_ref_s401") == 401
    with pytest.raises(ValueError):
        tb.sheet_no("badname")


# ────────────────────────── 映射：与读序/标签无关 ──────────────────────────


def _manifest(n_sheets: int = 2) -> dict[str, list[int]]:
    return {"case_s%02d" % i: [100 * i + k for k in range(24)] for i in range(1, n_sheets + 1)}


def _write(run_dir, name, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text("\n".join("%s,%s" % r for r in rows) + "\n", encoding="utf-8")


def test_assemble_handles_both_reading_orders(tmp_path):
    """列优先与行优先两种读序都要映射正确（实测两种都出现过）。"""
    man = _manifest(2)
    cells = man["case_s01"]
    col_major = [(f, f % 100) for f in cells]  # 顺序 = manifest 顺序
    row_major = [
        (cells[c * 8 + r], cells[c * 8 + r] % 100) for r in range(8) for c in range(3)
    ]  # 视觉行优先
    d1, d2 = tmp_path / "a", tmp_path / "b"
    _write(d1, "case_s01_01.csv", col_major)
    _write(d2, "case_s01_01.csv", row_major)
    v1, s1 = tb.assemble(man, d1)
    v2, s2 = tb.assemble(man, d2)
    assert v1 == v2, "读序不同导致映射不同"
    assert s1["unmapped"] == s2["unmapped"] == 0
    assert all(v1[f] == str(f % 100) for f in cells)


def test_assemble_recovers_truncated_labels(tmp_path):
    """f01625 被抄成 625 时，靠"预期帧号以标签结尾"唯一匹配救回来。"""
    man = {"case_s01": [1620 + k for k in range(24)]}
    rows = [(f - 1000, f % 100) for f in man["case_s01"]]  # 丢掉了千位
    d = tmp_path / "run"
    _write(d, "case_s01_01.csv", rows)
    vision, stat = tb.assemble(man, d)
    assert stat["suffix"] == 24 and stat["unmapped"] == 0
    assert vision[1625] == "25"


def test_sheets_covering_same_frames_twice_is_reported(tmp_path):
    """重复映射要计数（两个分片声明同一段样张时不能静默覆盖）。"""
    man = _manifest(1)
    d = tmp_path / "run"
    _write(d, "case_s01_01.csv", [(f, f % 100) for f in man["case_s01"]])
    _write(d, "other_s01_01.csv", [(f, f % 100) for f in man["case_s01"]])
    _, stat = tb.assemble(man, d)
    assert stat["dup"] > 0


# ────────────────────────── 比对：计数语义 ──────────────────────────


def test_score_separates_abstention_from_missing_engine():
    """视觉拒答 94 与"引擎没有读数"13 是两回事，混算会把审查面算错。"""
    reps = [1, 2, 3]
    vision = {1: "10", 2: "?", 3: "30"}
    engine = {1: "10", 2: "", 3: "30"}  # 2 号引擎没读出（非数值），1/3 号正常
    truth = {1: 10, 2: 20, 3: 30}
    res = tb.score(vision, engine, truth, reps)
    assert res["v_unk"] == 1 and res["x_missing"] == 1
    assert res["agree"] == 2 and res["v_ex"] == 2


def test_score_agreement_face_can_fail():
    """守卫必须能失败：双方一致且都错时 agree_bad 必须为 1（否则不是守卫）。"""
    reps = [1, 2]
    truth = {1: 10, 2: 20}
    good = tb.score({1: "10", 2: "20"}, {1: "10", 2: "20"}, truth, reps)
    assert good["agree"] == 2 and good["agree_bad"] == 0
    bad = tb.score({1: "10", 2: "99"}, {1: "10", 2: "99"}, truth, reps)
    assert bad["agree"] == 2 and bad["agree_bad"] == 1, "一起错却没被识别"


def test_score_without_engine_skips_agreement_analysis():
    """没有引擎读数时不得假装做过一致面分析。"""
    res = tb.score({1: "10"}, {}, {1: 10}, [1])
    assert res["agree"] == -1 and res["split"] == -1


# ────────────────────────── 头解析与产出 ──────────────────────────


def test_parse_truth_header_reads_single_source(tmp_path):
    """视频/ROI/帧范围只从真值头取，不另设配置（避免第二份事实源）。"""
    p = tmp_path / "x_truth.csv"
    p.write_text(
        "# RaceVideoToLog v2.7.0\n"
        "# video_hash=abc, video=test5.mp4, fps=59.767, codec=h264\n"
        "# roi=843,993,948,1025, format=km/h, frame_start=362, frame_end=7585\n"
        "362,0.00,257,21\n",
        encoding="utf-8",
    )
    head = tb.parse_truth_header(p)
    assert head["video"] == "test5.mp4"
    assert head["roi"] == (843, 993, 948, 1025)
    assert (head["frame_start"], head["frame_end"]) == (362, 7585)


def test_parse_truth_header_rejects_incomplete(tmp_path):
    """缺 video/roi 时必须报错，不能拿默认值往下跑。"""
    p = tmp_path / "bad.csv"
    p.write_text("# 没有头信息\n1,2,3,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tb.parse_truth_header(p)


def test_emit_writes_native_schema_and_mirrors_to_rvlt(tmp_path):
    """主产物是原生 truth.v1（逐帧 state）；RVTL 兼容导出只是镜像，未知写 ?。"""
    import _truth_format as tf

    case = tmp_path / "case_truth"
    (case / "run").mkdir(parents=True)
    man = {"case_truth_s01": [10, 11, 12]}
    (case / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (case / "run" / "case_truth_s01_01.csv").write_text("10,7\n11,?\n12,9\n", encoding="utf-8")
    out, rvlt = case / "truth.jsonl", case / "truth.csv"
    a = _args(
        tmp_path,
        case,
        [case / "run"],
        out,
        rvlt=rvlt,
        video="v.mp4",
        roi="1,2,3,4",
        frame_start=0,
        frame_end=99,
    )
    assert tb.cmd_emit(a) == 0
    meta, recs = tf.read_native(out)
    assert meta["schema"] == tf.SCHEMA and meta["video"] == "v.mp4"
    assert [(r["frame"], r["value"], r["state"]) for r in recs] == [
        (10, 7, "read"),
        (11, None, "unknown"),
        (12, 9, "read"),
    ]
    body = [ln for ln in rvlt.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    assert body == ["10,7", "11,?", "12,9"]


# ────────────────────────── 盲读映射与闸门栈 ──────────────────────────


def _args(work, case, runs, out=None, **kw):
    """构造 cmd_* 需要的命名空间（避免各测试重复样板）。"""
    from types import SimpleNamespace

    base = dict(
        work=work,
        case=case.name,
        truth_csv=None,
        video=None,
        roi=None,
        frame_start=0,
        frame_end=None,
        run=list(runs),
        out=out,
        rvlt=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_blind_ids_map_back_to_frames(tmp_path):
    """盲读的乱序 id 必须经 key 还原到帧号；映射不依赖子代理的读序。"""
    man = {"case_s01": [4, 1, 3, 2]}  # 格子顺序 = 乱序 id 顺序
    key = {"1": 900, "2": 901, "3": 902, "4": 903}
    d = tmp_path / "run"
    d.mkdir()
    (d / "case_s01_01.csv").write_text("4,10\n1,11\n3,12\n2,13\n", encoding="utf-8")
    got, _ = tb._reads_to_frames(man, {int(k): v for k, v in key.items()}, d)
    assert got == {900: "11", 901: "13", 902: "12", 903: "10"}


def test_gate_stack_accepts_only_agreement_and_flags_rest():
    """闸门栈：两读一致 ∧ 引擎一致才放行；分歧/拒答/引擎不符全部进审查面。"""
    reps = [1, 2, 3, 4]
    truth = {1: 50, 2: 60, 3: 70, 4: 80}
    r1 = {1: "50", 2: "60", 3: "70", 4: "?"}
    r2 = {1: "50", 2: "99", 3: "70", 4: "80"}
    engine = {1: "50", 2: "60", 3: "71", 4: "80"}
    res = tb._report_stack([r1, r2], engine, truth, reps)
    assert res["agree"] == 1, "只有帧 1 满足两读一致 ∧ 引擎一致"
    assert res["agree_bad"] == 0
    assert res["split"] == 1 and res["v_unk"] == 1


def test_gate_stack_can_fail():
    """守卫必须能失败：两读一致 ∧ 引擎一致但三方都错时必须计入（这是 C-95 的实际残余）。"""
    reps = [1]
    r1, r2, engine = {1: "180"}, {1: "180"}, {1: "180"}
    res = tb._report_stack([r1, r2], engine, {1: 100}, reps)
    assert res["agree"] == 1 and res["agree_bad"] == 1, "共谋错读未被识别"


def test_emit_refuses_to_vote_away_disagreement(tmp_path):
    """多读分歧不得投票取值——只能记未知（分歧值本身没有任何依据）。"""
    case = tmp_path / "case_truth"
    (case / "run1").mkdir(parents=True)
    (case / "run2").mkdir(parents=True)
    (case / "manifest.json").write_text(json.dumps({"case_truth_s01": [1, 2]}), encoding="utf-8")
    (case / "run1" / "case_truth_s01_01.csv").write_text("1,10\n2,20\n", encoding="utf-8")
    (case / "run2" / "case_truth_s01_01.csv").write_text("1,10\n2,21\n", encoding="utf-8")
    out, rvlt = case / "t.jsonl", case / "t.csv"
    a = _args(
        tmp_path, case, [case / "run1", case / "run2"], out, rvlt=rvlt, video="v.mp4", roi="1,2,3,4"
    )
    assert tb.cmd_emit(a) == 0
    import _truth_format as tf

    _meta, recs = tf.read_native(out)
    assert [(r["value"], r["state"]) for r in recs] == [(10, "read"), (None, "unknown")]
    assert recs[1]["reads"] == [20, 21], "分歧必须留下读数痕迹供审查"
    body = [ln for ln in rvlt.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    assert body == ["1,10", "2,?"]
