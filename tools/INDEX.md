# tools/ 索引（渲染产物；事实源 registry/tools.yaml）

> 分级判据与生命周期见 registry/tools.yaml 头注。
> 历史探针不在树（git 档案，archive/pre-rework 锚点起可溯）。

## 常备工具（standing，质量面=生产代码）

| 文件 | 用途 |
|---|---|
| `_audit_ext.py` | 结构化审计（按名注册）：layering/knob/hooks/render/decord_pin/rules_schema 等 |
| `_measure_lib.py` | 测量原语库（VIDS/ROI/capture_shas/paired_ab） |
| `_probe_hooks.py` | R-probe-hooks-points 执行体（POINTS 打桩表） |
| `_probe_run_setup_cost.py` | 启动成本探针（tests/ 引用） |
| `_probe_stress_harness.py` | 压测基建（adv-stress-harness 规格） |
| `_probe_window_matrix.py` | R-window-gate 执行体（矩阵门禁，默认 --repeat 5） |
| `_run_with_switchinterval.py` | 竞态复现基建（worker 型） |
| `_truth_format.py` | 自有真值格式 truth.v1（JSONL，逐帧 state=read/inferred/unknown）+ RVTL 兼容 shim（唯一允许知道 RVTL 头的模块） |
| `_worker_lib.py` | 子进程 worker 公共库 |
| `bench.py` | R-ab-interleave/R-threshold-resolvable 执行体；账本 registry.jsonl + 标定账本 aa_calibration.jsonl；bulk（多文件批量视图）/matrix（旋钮交互屏） |
| `bench_hybrid.py` | hybrid 批量基准（bench 家族） |
| `check.py` | R-check-tiers 执行体（两级合规链） |
| `conclusions_sweep.py` | R-conclusions-sweep 执行体（发版清扫 --strict） |
| `e2e_smoke.py` | 端到端冒烟 / 真值验证（真实视频） |
| `env_doctor.py` | 环境体检/部署/审计（R-path-no-hardcode 执行体） |
| `env_probe.py` | ffmpeg/decord 解析统一入口（外部依赖路径唯一出口） |
| `health.py` | KPI 计算→registry/kpi.yaml（自我约束执行体；HEALTH.md 视图源） |
| `probe_decode_rates.py` | 各后端解码速率基准（bench 家族） |
| `refs.py` | R-refs-resolve 执行体（引用解析器） |
| `render.py` | R-render-consistency 执行体（七视图统一渲染） |
| `truth_build.py` | 视觉真值获取链（代表帧→样张→收数/一致面→产出真值；守卫 tests/test_truth_build.py，含"坏行不得静默丢弃"负测） |
| `version_advice.py` | R-version-tag/R-version-advice 执行体（API 面 dump/档位判据/verify/金标 manifest/bench 快照） |

## 在树探针（probe，近两轮；超出即出树）

| 文件 | 说明 |
|---|---|
| `tools/probes/_probe_c68_gap.py` | 近两轮（0.25.x/0.26.x）在用取证（last 2026-10-11） |
| `tools/probes/_probe_hybrid_acceptance.py` | 近两轮（0.25.x/0.26.x）在用取证（last 2026-10-11） |
| `tools/probes/_probe_hybrid_supply.py` | 近两轮（0.25.x/0.26.x）在用取证（last 2026-10-11） |
| `tools/probes/_probe_ov_version_ab.py` | 近两轮（0.25.x/0.26.x）在用取证（last 2026-10-11） |
| `tools/probes/_probe_quiet_suite.py` | 近两轮（0.25.x/0.26.x）在用取证（last 2026-10-11） |

（探针删除协议：结论数字已由 conclusions.yaml 的 commit 锚点承载 → git rm；数字不丢。）

