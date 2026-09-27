"""解码器契约协商测试（R4 跨仓契约机器化，2026-09-28）。

两层：
1. 协商逻辑（假 decord 模块，CI 无 decord 也可跑）：无契约面 → None
   回退；版本新于已知 → 告警；能力缺失 → DecoderContractError 显式拒绝
   （且 hybrid 拒绝不被降级 try 吞掉——放 try 内会静默回退纯 GPU）。
2. 对账（需真 decord）：tests/golden/decoder_contract.yaml 的 DC 假设 ↔
   fork features() 键集一一对应；fork 无契约面（上游原版/旧 wheel）时
   skip——对账义务只在契约面存在后生效。
"""
from __future__ import annotations

import logging
import sys
import types

import pytest

from video_ocr_engine.decode.contract import (
    KNOWN_CONTRACT_VERSION, DecoderContractError, FeatureView, probe_contract,
)

# DC 假设 → fork features() 键（tests/golden/decoder_contract.yaml 规范侧；
# fork python/decord/_contract.py 实现侧）。DC-09 是引擎侧持有引用的生命
# 周期约定（owner 显式化属引擎 FrameBuffer 设计），无 fork 查询面，不进表。
DC_KEYS = {
    'DC-01': ('roi_first',),
    'DC-02': ('next_roi_stream',),
    'DC-03': ('hybrid_ctx', 'hybrid_gpu_ctx'),
    'DC-04': ('yuv420_packed_nv12',),
    'DC-05': ('gray_output',),
    'DC-06': ('stride_fast',),
    'DC-07': ('batch_stream',),
    'DC-08': ('get_color_range', 'get_codec'),
    'DC-09': (),   # 引擎侧约定，无 fork 查询面（见 docstring）
    'DC-10': ('device_ptr_layout',),
}


# ── 协商逻辑（假 decord）──────────────────────────────────────────────
def _install_fake_decord(monkeypatch, *, version=None, feats=None):
    """注入假 decord 包；version=None 模拟无契约面（上游/旧 wheel）。"""
    dec = types.ModuleType('decord')
    vr_mod = types.ModuleType('decord.video_reader')
    if version is not None:
        dec.CONTRACT_VERSION = version
        dec.features = lambda: dict(feats or {})
    monkeypatch.setitem(sys.modules, 'decord', dec)
    monkeypatch.setitem(sys.modules, 'decord.video_reader', vr_mod)
    return dec


def test_probe_no_contract_surface(monkeypatch):
    _install_fake_decord(monkeypatch, version=None)
    assert probe_contract() is None


def test_probe_contract_present():
    view = FeatureView(1, {'roi_first': True})
    assert view.contract_version == 1
    assert view.has('roi_first') and not view.has('hybrid_ctx')


def test_probe_newer_version_warns(monkeypatch, caplog):
    _install_fake_decord(monkeypatch, version=KNOWN_CONTRACT_VERSION + 5,
                         feats={'roi_first': True})
    with caplog.at_level(logging.WARNING):
        view = probe_contract()
    assert view is not None and view.contract_version > KNOWN_CONTRACT_VERSION
    assert any('新于引擎已知' in r.message for r in caplog.records)


def test_probe_features_failure_is_no_contract(monkeypatch):
    dec = types.ModuleType('decord')
    dec.CONTRACT_VERSION = 1
    dec.features = lambda: 1 / 0
    monkeypatch.setitem(sys.modules, 'decord', dec)
    with pytest.raises(ZeroDivisionError):
        dec.features()   # 直接调用确实抛
    assert probe_contract() is None   # 协商层吞异常按无契约面


def test_require_missing_raises():
    view = FeatureView(1, {'roi_first': True})
    with pytest.raises(DecoderContractError, match='DC-03'):
        view.require('hybrid_ctx', 'DC-03', 'hybrid ctx')


# ── 引擎接线（hybrid 拒绝不被降级 try 吞掉）───────────────────────────
def _fake_decord_open_stack(monkeypatch, feats):
    """假 decord：gpu/cpu/hybrid ctx 可开，VideoReader 返回假 reader。"""
    dec = types.ModuleType('decord')
    dec.CONTRACT_VERSION = 1
    dec.features = lambda: dict(feats)
    ctx = lambda i: ('ctx', i)   # noqa: E731 - 假 ctx 工厂
    dec.cpu = ctx
    dec.gpu = ctx
    dec.hybrid = ctx
    dec.hybrid_gpu = ctx

    vr_mod = types.ModuleType('decord.video_reader')
    vr_mod._CAPI_VideoReaderSetRoi = object()

    class _VR:
        def __init__(self, *a, **k):
            self.len = 100

        def __len__(self):
            return 100

        def get_codec(self):
            return 'h264'

        def get_color_range(self):
            return 0

    dec.VideoReader = _VR
    monkeypatch.setitem(sys.modules, 'decord', dec)
    monkeypatch.setitem(sys.modules, 'decord.video_reader', vr_mod)
    return dec


def test_hybrid_missing_capability_raises_not_degrades(monkeypatch):
    """契约面说 hybrid_ctx 缺 → 显式拒绝；不得静默回退纯 GPU。"""
    _fake_decord_open_stack(monkeypatch, feats={'roi_first': True})
    from video_ocr_engine.decode.decord_source import DecordFrameSource
    degraded: list = []
    src = DecordFrameSource('v.mp4', (0, 0, 10, 10), 'hybrid', False,
                            degraded, ocr_on_gpu_fn=lambda: False)
    with pytest.raises(DecoderContractError, match='DC-03'):
        src.open()
    assert not any('回退纯 GPU' in d for d in degraded)


def test_hybrid_contract_ok_opens(monkeypatch):
    feats = {'roi_first': True, 'hybrid_ctx': True,
             'hard_decode_window': False}
    _fake_decord_open_stack(monkeypatch, feats=feats)
    from video_ocr_engine.decode.decord_source import DecordFrameSource
    degraded: list = []
    src = DecordFrameSource('v.mp4', (0, 0, 10, 10), 'hybrid', False,
                            degraded, ocr_on_gpu_fn=lambda: False)
    vr = src.open()
    assert src.backend_label == 'decord/hybrid'
    assert vr is not None
    # 硬窗能力缺（契约面说没有）→ 记降级而不是裸 getattr 探测
    assert any('硬窗' in d for d in degraded)


def test_roi_guard_contract_path(monkeypatch):
    """契约面在且 roi_first 缺 → 构造期显式拒绝（DC-01）。"""
    _fake_decord_open_stack(monkeypatch, feats={'roi_first': False})
    from video_ocr_engine.decode.decord_source import ensure_roi_capable_decoder
    with pytest.raises(DecoderContractError, match='DC-01'):
        ensure_roi_capable_decoder()


# ── 对账（需真 decord 契约面）─────────────────────────────────────────
def test_fork_features_match_contract_yaml():
    """decoder_contract.yaml 的 DC 假设 ↔ fork features() 一一对账。"""
    import yaml

    import decord
    if not hasattr(decord, 'features'):
        pytest.skip('decord 无契约面（上游原版/旧 fork）：对账义务未生效')

    root = __file__.replace('\\', '/').rsplit('/tests/', 1)[0]
    with open(f'{root}/tests/golden/decoder_contract.yaml', encoding='utf-8') as f:
        spec = yaml.safe_load(f)
    feats = decord.features()

    missing_keys = []
    for a in spec['assumptions']:
        for key in DC_KEYS[a['id']]:
            if not feats.get(key):
                missing_keys.append(f"{a['id']}:{key}")
    assert not missing_keys, f'fork 契约面缺能力: {missing_keys}'

    # hybrid_stats 键集：报告 v6 穿透面的超集断言（fork 可加键不可删）
    stats_keys = feats.get('hybrid_stats_keys') or ()
    for must in ('frames_c', 'frames_g', 'force_eof', 'cache_peak_mb',
                 'window_frames'):
        assert must in stats_keys, f'hybrid_stats_keys 缺 {must}'

    assert decord.CONTRACT_VERSION <= KNOWN_CONTRACT_VERSION, (
        'fork 契约版本新于引擎已知——升级 KNOWN_CONTRACT_VERSION 并复核键集')
