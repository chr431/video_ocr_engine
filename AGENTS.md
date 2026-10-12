# AGENTS.md — 注入核（渲染产物；事实源 charter.md + registry/）

> 本文件由 tools/render.py 渲染，人不得手写；引用查询 `python tools/refs.py resolve <token>`。

> 修订须附取证并逐条裁决，无冷静期。本文件之外不存在手写制度文本。

## 精神（一句话版）
单源：事实只写一遍，其余皆视图。可执行：无执行机制的不是规则（advisory≤10）。
档案归 git：工作树只放现役物，一切对象有退出。确定性：重复判断预先裁决。
锚点：本文件不变，其余皆可流。举证：规则注明所防事故，守卫必须能失败。

## 不变量
1. 分层：L0 本文件｜L1 registry/（唯一事实源）｜L2 执行（CI+本机门禁，harness 无关）｜L3 视图（全渲染，永不手写）｜L4 git（唯一档案）。
2. 引用：跨文件只用注册名（C-NN / R-名 / decord@x.y.z / fork@C-NN / archive/<tag>）；禁裸 §N、禁裸 0.x；校验=refs check。
3. 版本：仅授予独立发布产物、外部钉版、数据契约三类。bump 判据：API 面或金标 diff 非空→minor，否则 patch；bump 必须同名 tag，CI 对账。
4. 生命周期：规则随执行体生死；结论 superseded 次版出源；探针留近两轮（轮=发版）；log 发版出树；视图随源生灭。
5. 注入核 ≤6KB：长期宽松上限；触碰=制度失效警报，查内容泄漏，不上调数值。
6. 守卫带负测；守卫的守卫=负测，两层封顶。
7. 本机事实住 registry/environment.yaml，不住规则与宪章文本。

## 现役架构（源 registry/architecture.yaml）

链路：解码 → 像素分段 → 代表帧 → OCR → 相似段合并

- 解码 → `video_ocr_engine/decode/decord_source.py`：decord.VideoReader.get_batch 唯一入口；hybrid 走 decord 原生 ctx（引擎
- 分段 → `video_ocr_engine/domain/segmentation.py`
- 编排引擎 → `video_ocr_engine/pipeline/engine.py`：SegmentEngine 唯一入口；EngineInputs 显式 run 契约；生命周期单出处 _driver.p
- 宿主后端 → `video_ocr_engine/pipeline/host_backend.py`
- GPU 后端 → `video_ocr_engine/pipeline/gpu_backend.py`：gray+NVDEC+TRT 默认；设备侧拆分 gpu/ 包（pools/streams/device）
- OCR 会话 → `video_ocr_engine/pipeline/ocr_stage.py`：SessionSpec 契约；CPU 引擎=OpenVINO 唯一（C-48）
- OCR 调度/引擎池 → `video_ocr_engine/ocr/native.py`：池上限 _POOL_MAX_PER_KEY=4 / _POOL_MAX_TOTAL=16，key=(model,typ
- TRT → `video_ocr_engine/ocr/trt.py`
- 配置常量/旋钮 → `video_ocr_engine/config/constants.py`
- 运行报告 → `video_ocr_engine/pipeline/report.py`：RunReport schema v10（REPORT_VERSION 单源于代码；指标注册表 domain/metr
- 分相打桩 → `video_ocr_engine/domain/prof.py`：ProfSpine 单一计时脊柱；门面 _prof_end 薄委托
- 性能 A/B → `tools/bench.py`：run/diff/show/ab/telemetry-check；账本 bench/registry.jsonl（滚动

- 并行维度唯一：decode_backend="hybrid" 的 CPU+NVDEC 双解码，已由 decord fork 原生实现；无 dual pipeline、无 DUAL_* 环境变量（历史提及均为旧档案）
- 引擎内部恒单通道灰度：decord 只输出 yuv420（keep_crops 且 rep_crop_format="yuv"）或 gray
- 相似段合并在分离图上进行，默认 binary（TEXT_SEP_MERGE=binary|off，稠密簇门见 C-38）
- GPU 管线门控：gray + decode∈{auto,nvdec,cpu} + ocr≠cpu + NVDEC/TRT 可用时自动启用；GPU+ONNX 无净收益（C-09）
- auto 恒 NVDEC 优先=刻意决策（C-08）；批量吞吐=逐文件顺序跑（C-53，pool 已删）
- 遥测两档（D30）：off=发布默认；full=调试/bench 默认；std 别名已拒收
- fp16 已产品化（TRT_FP16=1，opt-in）；CUDA Graph 无收益已移除

## 环境要点（源 registry/environment.yaml）

- decord@0.9.8（钉版对账=审计 decord_pin）；视频目录经 RACELOG_VIDEO_DIR（默认 D:/Videos/racelog_test）
- 测试视频一律 sample_stride=1；中文输出须 UTF-8 重配（详见 environment）

## 工具与门禁（源 registry/tools.yaml + rules.yaml）

- standing 20 件（质量面=生产代码）+ probes 5 个在树（近两轮）；清单=tools/INDEX.md
- 合规链 `python tools/check.py`（daily / --tier release）；A/B `python tools/bench.py ab`；
  金标 `python tests/golden/record.py --verify`；清扫 `python tools/conclusions_sweep.py --strict`；
  版本 `python tools/version_advice.py advice|verify`；性能测量启动前自动过 R-perf-quiet 负载门禁。

## 结论

- active 48 条（源 registry/conclusions.yaml）；表=docs/CONCLUSIONS.md；死档经 git 档案取回。

