"""解码器契约协商（R4 跨仓契约机器化，2026-09-28）。

decord fork 自 0.8.5 起暴露契约面（`decord.CONTRACT_VERSION` +
`decord.features()`），把引擎侧 decoder_contract.yaml（DC-01..10）的
隐式假设变成可对账的机器面。本模块是引擎侧的协商入口：

- `probe_contract()`：读取契约面；无契约面（上游原版 decord / 旧 fork
  wheel）返回 None——这不是错误，引擎回退既有结构性探测（hasattr /
  try-open 降级链），行为与 R4 之前逐位一致。
- 契约面存在且版本**新于**引擎已知：告警不拒绝（新 fork + 旧引擎的
  正常组合，唯一诚实动作是让人看见）。
- `FeatureView.require()`：能力缺失时抛 `DecoderContractError`（带
  DC 编号与原因）——替代「深处崩溃」，调用点只在引擎本就无法降级
  的路径（hybrid ctx 缺失是配置错误而非运行环境波动，不该静默换臂）。

对账测试（tests/decode/test_contract.py）以 tests/golden/
decoder_contract.yaml 为规范侧，fork features() 为实现侧，两侧漂移
即红。键→DC 映射见该测试的 `DC_KEYS`。
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: 引擎已知的最新的 fork 契约面版本（fork `_contract.CONTRACT_VERSION`）
KNOWN_CONTRACT_VERSION = 1


class DecoderContractError(RuntimeError):
    """解码器契约能力缺失（显式拒绝，替代深处崩溃）。"""


class FeatureView:
    """契约面快照（真子集语义：未知键忽略，已知键强校验）。"""

    __slots__ = ('contract_version', '_features')

    def __init__(self, contract_version: int, features: Mapping[str, Any]):
        self.contract_version = int(contract_version)
        self._features = dict(features)

    def has(self, key: str) -> bool:
        return bool(self._features.get(key))

    def get(self, key: str) -> Any:
        return self._features.get(key)

    def require(self, key: str, dc: str, usage: str) -> None:
        """能力缺失即抛 DecoderContractError（带 DC 编号与后果说明）。"""
        if not self.has(key):
            raise DecoderContractError(
                f'decord fork 契约面缺失能力 {key!r}（{dc}，{usage}）；'
                f'当前 CONTRACT_VERSION={self.contract_version}，'
                '请升级 chr431/decord fork 或更换 decode_backend')

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return (f'FeatureView(v{self.contract_version}, '
                f'{len(self._features)} keys)')


def probe_contract() -> FeatureView | None:
    """读取 decord 契约面；不存在则 None（上游原版/旧 fork，合法状态）。

    版本新于引擎已知时告警（新 fork + 旧引擎组合，不拒绝）。
    """
    import decord  # 延迟导入：构造期不依赖 decord 的约定保持不变
    version = getattr(decord, 'CONTRACT_VERSION', None)
    features_fn = getattr(decord, 'features', None)
    if version is None or not callable(features_fn):
        return None
    try:
        feats = features_fn()
    except Exception:  # noqa: BLE001 - 契约面自身故障按无契约处理
        logger.warning('decord.features() 调用失败，按无契约面处理',
                       exc_info=True)
        return None
    view = FeatureView(version, feats if isinstance(feats, Mapping) else {})
    if view.contract_version > KNOWN_CONTRACT_VERSION:
        logger.warning(
            'decord 契约面版本 %d 新于引擎已知 %d：新 fork 配旧引擎，'
            '行为以结构性探测为准；升级引擎后本告警消失',
            view.contract_version, KNOWN_CONTRACT_VERSION)
    return view
