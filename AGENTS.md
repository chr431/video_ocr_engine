# AGENTS.md — 注入核（过渡版，Phase E 起由渲染生成）

> **制度已重构（2026-10-12，rework/institution）**：宪章=`charter.md`，全部规则/结论/
> 工具分级的事实源=`registry/`（人读视图由 `tools/render.py` 渲染），引用查询=
> `python tools/refs.py resolve <C-NN / R-名 / decord@x.y.z>`。本文件 Phase E 起改为
> charter+registry 的渲染产物，当前为过渡手写版（只含指针，不含事实）。

## 铁律与规则

见 `registry/rules.yaml`（35 条：ci 18 / local-gate 7 / advisory 9；每条含执行体与所防
事故）。最高约束见 `charter.md`（七条不变量）。

## 现役架构

链路与各阶段入口见 `registry/architecture.yaml`（唯一事实源）。要点：解码 → 像素分段 →
代表帧 → OCR → 相似段合并；hybrid 双解码已由 decord fork 原生实现（引擎只透传）。

## 环境与命令

本机与外部事实（python/硬件/decord 钉版/测试视频目录）见 `registry/environment.yaml`。

- 合规链：`python tools/check.py`（daily）/ `--tier release`（发版：金标 full+清扫 --strict）
- 性能 A/B：`python tools/bench.py ab`（交错+时钟门禁；阈值先 `--aa` 标定；hybrid 对照臂=
  目标码最快纯解码臂，C-53）
- 引用查询：`python tools/refs.py resolve C-53`
- 视图重渲染：`python tools/render.py --write`
- 环境异常：`python tools/env_doctor.py`；改 decord C++ 后 `rebuild_dev.bat`
- 金标：`python tests/golden/record.py --verify [--tier smoke]`

## 探针与工具

standing（常备，质量面=生产代码）住 `tools/`；probe（一次性取证）住 `tools/probes/`，
近两轮出树。清单见 `tools/INDEX.md`（渲染产物）。历史探针/大档案/旧 log 一律在 git
（`archive/pre-rework` tag 锚点起），不在工作树。

## 提交纪律

结论进 `registry/conclusions.yaml`（带 commit 锚点 evidence），叙事进当轮 log
（`docs/log/YYYY-MM-DD-主题.md`，发版出树），提交信息=结果一句话+指针（≤600 字符）。
版本 bump 判据与 tag 对账见宪章不变量 3。
