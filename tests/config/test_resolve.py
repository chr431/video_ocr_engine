"""S1 门禁测试：env 契约 + resolve() 与 v1 读取点逐位等价（v2 §11 S1）。

等价性手法（文档 S1"验证手法"）：对随机 (env, 参数) 组合，断言
resolve() 的解析结果 == v1 engine_config.env_* 解析器在同一 env 下的
返回值——两套解析器任何一侧漂移都会在此被抓到。
"""
from __future__ import annotations

import pytest

import engine_config as config
from video_ocr_engine.config import KNOBS, RunConfig, resolve

ENV_NAMES = KNOBS.env_names()


# ── env 契约：18 个 v1 旋钮名 + v2 新增 ─────────────────────────────
def test_env_contract_covers_all_v1_knobs():
    v1_names = {getattr(config, n) for n in dir(config) if n.endswith("_ENV")}
    assert len(v1_names) == 18
    assert set(ENV_NAMES) >= v1_names          # v1 全部迁入，一个不丢
    # v2 新增仅两个：r5 telemetry 三档 + S6-0 报告 sidecar（§8.6 N-3）
    assert set(ENV_NAMES) - v1_names == {"VOE_TELEMETRY", "VOE_REPORT_FILE"}


def test_knob_registry_shape():
    assert len(KNOBS.knobs) == 20              # 18 v1 + telemetry + report_file
    for k in KNOBS.knobs:
        assert k.rationale_id, k.name          # 每个旋钮有依据锚点（D5）


# ── 优先级：显式参数 > env > 默认（Q5/r3）；VOE_ENV_WINS 逃生门 ──────
def test_priority_param_beats_env_beats_default():
    rc = resolve(env={"OCR_GAMMA": "2.5"},
                 overrides={"ocr.gamma": 2.2})
    assert rc.ocr_gamma == 2.2
    rc = resolve(env={"OCR_GAMMA": "2.5"})
    assert rc.ocr_gamma == 2.5
    rc = resolve(env={})
    assert rc.ocr_gamma == 2.0


def test_override_none_means_not_given():
    rc = resolve(env={"OCR_GAMMA": "2.5"}, overrides={"ocr.gamma": None})
    assert rc.ocr_gamma == 2.5                 # None 不锁定（v1 参数默认 None 语义）


def test_env_wins_escape_hatch_restores_v1_semantics():
    with pytest.warns(DeprecationWarning):
        rc = resolve(env={"VOE_ENV_WINS": "1", "OCR_GAMMA": "2.5"},
                     overrides={"ocr.gamma": 2.2})
    assert rc.ocr_gamma == 2.5                 # v1：env 盖过参数


# ── 解析语义：与 v1 engine_config.env_* 逐位等价（随机对账）─────────
BOOL_KNOBS = [k for k in KNOBS.knobs if k.type == "bool"]
INT_KNOBS = [k for k in KNOBS.knobs if k.type == "int"]
FLOAT_KNOBS = [k for k in KNOBS.knobs if k.type == "float"]


INT_SAMPLES = ["", "  ", "x", "1.5", "-3", "0", " 7 ", "999999"]
FLOAT_SAMPLES = ["", "x", "2.5", "0", "-1e-1", " 3 "]


@pytest.mark.parametrize("raw", INT_SAMPLES)
def test_int_parse_matches_v1(raw, monkeypatch):
    for k in INT_KNOBS:
        monkeypatch.setenv(k.env, raw)
        got = resolve().get(k.name)            # env 缺省 → 读 os.environ
        want = config.env_int(k.env, k.default)
        assert got == want, (k.name, raw, got, want)
        monkeypatch.delenv(k.env, raising=False)


@pytest.mark.parametrize("raw", FLOAT_SAMPLES)
def test_float_parse_matches_v1(raw, monkeypatch):
    for k in FLOAT_KNOBS:
        monkeypatch.setenv(k.env, raw)
        got = resolve().get(k.name)
        want = config.env_float(k.env, k.default)
        assert got == want, (k.name, raw, got, want)
        monkeypatch.delenv(k.env, raising=False)


@pytest.mark.parametrize("raw", ["1", "TRUE", " yes ", "On", "0", "FALSE",
                                 " no ", "Off", "maybe", ""])
def test_bool_parse_matches_v1(raw, monkeypatch):
    for k in BOOL_KNOBS:
        monkeypatch.setenv(k.env, raw)
        got = resolve().get(k.name)
        want = config.env_bool(k.env, k.default)
        assert got == want, (k.name, raw, got, want)
        monkeypatch.delenv(k.env, raising=False)


def test_gpu_pipeline_tri_state():
    assert resolve(env={}).pipeline_gpu is None          # 未设 → 规则
    assert resolve(env={"GPU_PIPELINE": "0"}).pipeline_gpu is False
    assert resolve(env={"GPU_PIPELINE": "1"}).pipeline_gpu is True
    # v1 细节：设置了但非法 → False（显式关），≠ 未设（_gpu_pipeline.py:566-570）
    assert resolve(env={"GPU_PIPELINE": "banana"}).pipeline_gpu is False


def test_text_sep_merge_raw_string_semantics():
    assert resolve(env={}).segment_text_sep_merge == "binary"
    assert resolve(env={"TEXT_SEP_MERGE": "contrast"}).segment_text_sep_merge \
        == "contrast"                     # 原样透传（校验是 S3 B3 的事）
    rc = resolve(env={"TEXT_SEP_MERGE": "x"}, overrides={"segment.text_sep_merge": "y"})
    assert rc.segment_text_sep_merge == "y"


# ── config_digest（§6.1：两次同配置必同 digest；改一档必变）──────────
def test_config_digest_stable_and_sensitive():
    a = resolve(env={"OCR_GAMMA": "2.0", "OCR_THREADS": "4"})
    b = resolve(env={"OCR_GAMMA": "2.0", "OCR_THREADS": "4"})
    assert a.config_digest == b.config_digest and len(a.config_digest) == 16
    c = resolve(env={"OCR_GAMMA": "2.0", "OCR_THREADS": "5"})
    assert c.config_digest != a.config_digest


def test_runconfig_is_frozen():
    rc = resolve(env={})
    with pytest.raises(Exception):
        rc.ocr_gamma = 9.0


def test_runconfig_defaults_match_registry():
    rc = resolve(env={})
    for k in KNOBS.knobs:
        assert rc.get(k.name) == k.default, k.name
