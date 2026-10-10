# 依赖与运行环境（video_ocr_engine）

> 迁移自 RaceVideoToLog/DEPENDENCIES.md，只保留引擎识别链相关依赖与性能笔记。
> 版本号以 2026-08 实测为准；`pyproject.toml` 中的下限约束是兼容基线。

## 核心依赖

| 包 | 当前版本 | 来源 | 说明 |
| --- | --- | --- | --- |
| numpy | 2.x | PyPI | 预处理/信号计算，纯 numpy 无 scipy |
| openvino | **2026.3.x（pyproject 上界 `<2026.4`）**；本机 2026.3.1 | PyPI | **CPU OCR 默认后端**（2026-09-14 起）：模型级 2.1× 于 ORT + argmax 全一致（log 2026-09-14-OpenVINO模型级A-B/集成轮）；wheel ~76MB，仅 CPU 插件。**冻结分发可剪至 73MB**（删 NPU/GPU/异构前端等，配方与等价性实证见 log 2026-09-14-OpenVINO冻结瘦身）。⚠️ **2026.4.0 发布当天（2026-09-28）在 windows runner `_infer` 原生崩溃**（同版本上午绿/下午红 = VM 异构或间歇崩溃；本机未评估）——上界钉 2026.3 系，换代评估按 C-48 复评触发另起一轮 |
| psutil | 6+ | PyPI | 物理核数探测 / RSS 采样（缺失时降级） |
| decord | **0.8.5（已发布 [v0.8.5](https://github.com/chr431/decord/releases/tag/v0.8.5)；本机已装回 cp313 wheel，master 领先见下节）** | chr431/decord release | NVDEC 硬解 + CPU 软解；**PyPI 官方版不支持** `next_roi` / ROI-first / GPU gray / YUV420 / `sample_stride` 等差步长快速路径；fork 自 0.8.2 起发布 cp39–cp314 全版本 wheel，`pip install <wheel>` 即用 |
| cuda-python | 13.x | PyPI | TRT 执行 + decord GPU DLL 注册 |
| tensorrt_*_bindings | 11.x | PyPI | TensorRT thin binding（~1MB）；运行 DLL 从系统 PATH 加载 |

> `tensorrt` 元包与 `tensorrt_*_libs`（~2.2GB DLL）被有意排除。运行时从
> NVIDIA 官网安装的 CUDA Toolkit / TensorRT 的 `bin` 目录加载 DLL。

## 开发依赖（dev extras，2026-09-20 稳健性轮起与 CI lint job 同源）

| 包 | 当前版本 | 说明 |
| --- | --- | --- |
| pytest | 8+/9+ | 测试 |
| ruff | 0.16.6 | 精选集 E4/E7/E9/F（pyproject `[tool.ruff.lint]`）；产品代码零违规基线；风格类规则不启用（% 格式化与长中文注释是本仓风格） |
| mypy | 2.3.1 | **domain/ 严格试点**（`files` + `follow_imports=skip` 边界）；无 stubs 第三方（decord/cuda/psutil/tensorrt）按 Any 容忍 |

## GPU 加速（运行时，不打包）

| 组件 | 来源 | 说明 |
| --- | --- | --- |
| CUDA Toolkit 13.x | NVIDIA 官网 | cudart/cublas 等 DLL，需在 PATH |
| TensorRT | NVIDIA 官网 | nvinfer DLL，需在 PATH；首次运行自动构建引擎缓存到 `ocr_engines/` |

`gpu_setup.ensure_gpu_initialized()` 会扫描 PATH 并注册 DLL 目录，同时把找到的
目录前置到 `os.environ["PATH"]`（`tensorrt` 的 `find_lib()` 只搜 PATH）。

## 已知问题与注意

### decord（自建 fork，pip wheel 安装）
- **0.9.2（2026-10-08 已发布：tag v0.9.2 + GitHub Release cp39–cp314
  wheel 矩阵 + win64-gpu.zip；本机 cp313 wheel 已装）= 0.8.5 + p3
  合并 + 硬窗架构重做**（收口轮，log 2026-10-08 收口与窗口架构重做；
  C-62）：0.8.6 候选（p3 双缺陷修复）在发布门禁复跑中暴露健康格
  hevc(0,3000) 单帧静默替补（同 dev dll 2/12 轮、失败内容确定、
  观测即消失——五轮窗口修复史的结构性复发），裁决**重做窗口架构**
  而非继续打补丁：①窗=绝对帧区间 [T,T+n)（pump 按 GOP 判交，
  删 reader seek_prefix_ 跨类算术）②drain marker 载荷分离
  （0=EOF/1=WINDOW_END，`range_end_pushed_` 与 `eof_pushed_` 分家）
  ③SessionState 会话对象化（ResetRouting=整体重建，僵尸 kick 类
  缺陷结构性不可达）④窗模式禁替补（FetchCachedFrame 窗激活恒
  false——静默换帧不可达，win_subs 结构性恒 0，缺帧 rewind 后
  DumpState+FATAL）。契约面不变（`window_seek_safe`/
  `hard_decode_window` 语义维持，新 stats 键 window_lo/hi=兼容新增，
  无 CONTRACT_VERSION bump；15 键）。fork 侧窗口覆盖从零到一：
  run_fast 第七套件 window + smoke 窗格 + `_kick_roulette` 参数化；
  引擎矩阵探针升级 `--repeat`（单轮 12/12 被 flake 证明可 lucky-pass，
  发布门禁=连续 5 轮）。
  **验证（wheel 态，发布工件）**：窗口矩阵 12/12×5 全绿 + 金标
  33/33（D 组晚起点窗用例走真窗路径）+ smoke 含窗格；dev 态另过
  七套件/stream×10/发布门禁 19/19（段数锚 8340）/dll_ab
  （redo 较 0.8.5 wheel +1.57%，<5% 门禁，构建配置混杂留档）/
  布局轮盘 11 布局 0 命中（含旧架构必坏位 shift=48）。
- **0.8.5（2026-09-28 已发布：tag v0.8.5 + GitHub Release
  cp39–cp314 wheel 矩阵 + win64-gpu.zip；本机 cp313 wheel 已装回）**：
  0.8.5 = 0.8.4 + 夜间重构轮五提交 + 发布轮两提交——EOF 预算在途宽限
  （av1 顺序读停滞 FATAL 根治，见 log 2026-09-28 R3R5 夜间轮·续章）+ 三条静默路径
  取证硬化 + GPU-only 解包（USE_CUDA 不可关，构建仍零 Toolkit）+
  hybrid TU 拆分（src/video/hybrid/ 四文件）+ 契约面
  （`decord.CONTRACT_VERSION`/`features()`，引擎 decode/contract.py
  协商，tests/decode/test_contract.py 对账）+ CMakePresets + 契约
  键集补 `gpu_arm_stall`（f071146 冻结检测计数，晚于契约面提交
  771d5e0 落地，发布轮补齐声明集）+ fork README 契约面文档化/
  消费者中立化。
  **验证（wheel 态，2026-09-28 发布轮）**：金标 28/28 逐位一致 +
  引擎 276 测试（契约对账真跑，skip 消失）+ fork 六套件过 wheel DLL
  （gpu/formats/md5/stream/stride/lockstep 全绿）。本地构建 wheel
  md5 `a6b0ee44…`（63,947,411 B；发布产物以 Release 页为准）。
  **gpu_arm_stall 监视哨已接线**：报告 `hybrid` 段全量穿透 + 引擎
  `_driver` 对 >0 记 WARNING（CU 臂启动竞态冻结被自愈重掷的次数；
  fork 侧为统计根治——真值生产中持续增长时按 fork tools/ 取证链
  续钻，log 2026-09-28 R3R5 夜间轮·续章）。
  **C-57 saga 存档**：哨兵（win_subs）/空真加固/EOF 网 v2（0.8.5 后
  四提交，已入 0.9.2）→ 前缀+僵尸 kick 修复（p3 分支，2026-09-28
  夜间轮根治，已合 0.9.2）→ 2026-10-08 复发 flake → 重做（C-62）。
  归因未定论：post-p3 构建 2/12 失败、pre-p3/no-prefix/0.8.5 对照
  臂干净，但每臂不同二进制布局（本仓已知布局翻转此类竞态）且未
  交错——指向性证据而非定论（叙事留档）。
- **0.8.4（2026-09-19 已发布，tag v0.8.4，cp39–cp314 + win64-gpu.zip）**：
  v0.8.4 = v0.8.3 + 冻结前清理（cuMemcpyPeerAsync 符号修复 / 死代码
  −14.6k 行 / nvml.h 手写替代 795KB→45 行）+ 性能参考纯解码口径
  （详见 log 2026-09-19-decord发布与引擎清理）。
  （2026-09-20 审计修复轮曾处 dev 部署态 `a2deb3c2`——错误槽
  per-instance / H2D 积攒窗默认关 / update_version.py argv 化，
  已并入上述 0.8.5 dev 链。）
- **0.8.3（2026-09-12 已发布，tag v0.8.3，cp39–cp314 + win64-gpu.zip；
  CI cp313 wheel 抽验：金标 28/28 + 宿主挂死用例干净 + TRT 段数一致）**：
  v0.8.3 = 0.8.2 + 十一个提交（73e5540 析构 UAF 修复 /
  d94d92e hybrid GPU 池深按 ROI 重算 / 3b96c6f hybrid chunk 预路由 / 93a5ce1 /
  **99b8785 FORCE_SIDE 诊断臂死锁修复** / **75b8602 hybrid 非打印 stats 层 +
  sustained 产能估计器（实验性 opt-in）** / **4ccf889 sustained 产能口径转默认
  + EWMA 旧口径删除 + kick 突发治倾斜计划死锁** / **281a738 上载批大小消融
  旋钮 + kick 承重判别** / **02c91e6→**b67f9eb** 宿主慢消费者环死锁修复——反馈式
  清偿（债务快照+克隆至清偿，EOF 冲刷兜底）取代定长突发**），本地提交 486d2f6 已 bump 版本号（三源一致）。
  - **本地 cp313 wheel 已构建并安装**本机构建 sha16 `8e6f17a5`；CI 发布产物
    sha16 `f6cc1a48`，抽验同绿）。**wheel 口径验证全绿**（§15-§17）：金标 28/28 ×2（重录整矩阵同序——局部
    `--case` 重录会录在冷池态，见 FINDINGS F-8 注记）、挂死用例干净、
    引擎 e2e 三码全片段数与 dev dll 一致、压测 9/9、decode-only 漂移带内。
  - ⚠️ **§16 死锁事故**：hybrid 宿主路径 + ONNX 慢消费者（h264lg B 金字塔
    深重排）曾 2/2 必现挂死——突发被在途窗口截断 + KICK_BURST=5 不够
    冲开 DPB。修复见 `knowledge/benchmarks.yaml:hybrid_host_deadlock_fix`。
    另 §10.2 的 kick=0 判别当日不可复现（时序敏感），替换证据 = kicks
    激活计数 + wheel≡dev 全对位；历史 4/4 对照保留。
  - **sustained（滑窗持续产能）现为唯一 CPU 产能口径**（4ccf889 起）：
    `DECORD_CPU_RATE_SUSTAINED` env 已删除。机制/数字见
    `knowledge/benchmarks.yaml` 的 `hybrid_cpu_rate_ratchet`。
    kick 清偿（**反馈式**，b67f9eb 起：克隆至离场侧债务清偿，护栏 64 防损坏流，
    EOF 冲刷兜底；bf16 重排 17>16 酷刑流 3/3 实证）治 open-GOP DPB 队头尾帧
    死锁；`DECORD_HYBRID_KICK_BURST`：0 = 消融关（单包 kick），>0 = 护栏值。
  - 引擎 GPU 管线全片 e2e（配对 3 遍）：hevc −24%（且 hybrid 首次显著
    胜纯 NVDEC −23%）、h264 −4.7%、av1 持平；金标 28/28 逐位一致。
  - 开发 dll md5 `2fd49ea5`（build-dev，2026-09-12 replan 回滚后从
    HEAD 281a738 重建——MSVC 非确定性构建产物，md5 每次重建会变，以
    行为判别为准：回滚后构建的 stats 行无 `replans=` 字段）。
- **h264 NVDEC 是本机硬件天花板，非软件问题**（2026-09-10 实测）：同内容
  三编码 NVDEC 982(h264)/2103(hevc)/1749(av1) fps；ffmpeg 自带 cuvid 同比
  （16.5x vs 38.7x）。h264 解码选峰值应走 CPU 软解/显式 hybrid。
- **0.8.2 起改用 release wheel + pip 安装**（不再 editable 源码导入、不再
  手工部署 DLL）：wheel 自带 `decord.dll` 与 FFmpeg 63 运行库（包根目录），
  `pip install decord-0.8.2-cp313-cp313-win_amd64.whl` 即用；0.8.2 已含
  cuMemcpy2D_v2 修复（fork 7ef70f5）。当前 dll md5 6597eea6。
- **本地开发 dll（2026-09-10，含未发布修复）**：`build-dev/decord.dll`
  含 VideoReader 析构顺序 UAF 修复（hybrid_gpu 帧 Deleter 摸已析构池 →
  av1 hybrid close 偶发/必现崩溃；fork 本地提交 73e5540，**未推送/未发
  wheel**）。
- **⚠️ 本地 dll 已于 2026-09-13 重建（含盲阶段路由修正）**：fork 提交
  `f946d72`（净 +5/−1 行）——`ChooseSide()` 盲阶段由"无条件采样 CPU"改为
  "只采样一个 CPU chunk，其后回 GPU 直供"，避免连续两块压慢腿串行化队头。
  新 md5 `6367c5fa…`（旧 `2f4c9a11…`，备份 `decord.dll.bak-prefix-0913-1324`）。
  **该 DLL 已过金标 28/28**（含 C-hevc-hybrid）。重建方式（不经 cmd.exe）：
  在 PS 工具里直接配 `PATH`/`INCLUDE`/`LIB` 后跑 `ninja`（见
  `docs/log/2026-09-13-hybrid解码率与理论并联和.md`「续六」）。
  ⚠️ 该提交**未推送、未发上游 wheel** —— 与本文件"本地开发 dll"同政策。
  ✅ **产品默认路径已含该修正（2026-09-13 重建 wheel 并安装）**：
  `dist/decord-0.8.3-cp313-cp313-win_amd64.whl`（63,925,990 B）→
  `pip install --force-reinstall --no-deps`。包内 `decord.dll` md5
  `4116fb4a…` → **`af2a652d…`**（含 `f946d72`：盲阶段只采样一个 CPU chunk）。
  - **验证**：金标 `record.py --verify` **28/28**（**不设** `DECORD_LIBRARY_PATH`）；
    路由行为随补丁改变 —— `DECORD_HYBRID_DEBUG=1` 下 `hybrid-plan` 建立点
    由 `k0=3` 变为 **`k0=4`**（这是判定"安装的 DLL 确含补丁"的判据）。
  - ⚠️ **顺带变更了 FFmpeg 运行库来源**：包内 `avcodec-63.dll` md5
    `e32261c7…` → **`a8deaa57…`**。原 wheel 用的是 `D:/Repo/decord-release-dl/…`
    树（**该目录已不存在**），本次用现存且 CI 脚本引用的
    `D:/Software/ffmpeg-n9.0-latest-win64-gpl-shared-9.0`（与 `rebuild_dev.bat`
    同源）。两者同为 avcodec-63（FFmpeg 9），金标 28/28 已复验。
  - 回滚：`dist/prepatch/decord-0.8.3-cp313-cp313-win_amd64.whl`（改名前为
    09-12 13:47 那份），`pip install --force-reinstall --no-deps` 即可退回。
- **⚠️ 2026-09-13 晚：本地 dll 与 wheel 再重建（fork `4cebeef`，并联缺口收口
  C-45）**：ROI 银行解耦（queue 按 ROI 字节重算 + raw 按全帧 768MB 解耦）+
  计划滑动视界（`DECORD_HYBRID_PLAN_HORIZON`，默认 4096，0=关；
  **REPLAN_PCT/GAP 已删除**）。fork 全片 hevc +12.8% / h264 +14.8% / av1
  +3.0%（对两解码器并联和 77/68/88%→87/78/90%）；引擎干净对 A/B：hevc
  −4.8%（28 轮 23/28）、av1 −2.9%（7/8）、h264 平价。金标 28/28。
  - wheel：`dist/decord-0.8.3-cp313-cp313-win_amd64.whl`（63,926,361 B），
    包内 `decord/decord.dll` md5 **`27349408…`**；已 `pip install` 覆盖。
    **判据**：不设 `DECORD_LIBRARY_PATH` 跑 `DECORD_HYBRID_STATS=1`，
    `[hybrid-stats] plan` 行含 `horizon=4096 rebuilds=N`。
  - dev dll：`build-dev/decord.dll` = `bc7c27a1…`（同源 4cebeef 干净
    重建；MSVC 非确定性 → md5 与 wheel 内不同属正常）。
  - **✅ 2026-09-14：GOP 派工架构合入并发布（C-46）**：wheel 重建安装，
    包内 `decord/decord.dll` md5 **`6ec5b1ea…`**（判据：stats 行含
    `cache_peak=/late=/strag=`）。三码引擎 A/B hevc −6.29%（6/6）/h264
    −4.78%（6/6）/av1 平价；18/18 段数恒定；金标 28/28×2。回滚 =
    `dist/prepatch/` 旧 wheel 或 fork `4cebeef`。
  - **构建陷阱（本轮两事故）**：本机 ninja 对 `video_reader.cc.obj` 无头依赖
    （`ninja -C build-dev -t deps` 显示 `#deps 0`）——改
    `hybrid_threaded_decoder.h`/`ffmpeg/threaded_decoder.h` 的**类布局**后
    增量构建不重编 video_reader：①成员移位→构造期访问崩溃；②尾加成员
    +回退源码→`NeedsPackets` 等内联函数按错布局读成员=**静默行为损坏**
    （A/B 判定被污染一轮）。**规矩：改这两个头后 `rm` 掉
    `video_reader.cc.obj`（或 touch 源文件）再构建；换 DLL 的 A/B 两臂都
    要全对象干净重建**。历史 82c62e3f 经数字比对确认为良性 stale
    （尾加成员不移位），既往结论不受影响。
  - 重建方式（**不经 cmd.exe**）：在 PS 工具里配 MSVC `PATH`/`INCLUDE`/`LIB`
    ＋ `FFMPEG_DIR`，再以**系统解释器绝对路径**跑
    `python -m pip wheel . --no-deps --no-build-isolation -w dist`。
    ⚠️ 坑：`PATH` 上的 `python` 是受管运行时（**未装 scikit-build-core**）→
    必须显式用 `c:/Users/eric chen/AppData/Local/Programs/Python/python313/python.exe`。开发态用 `DECORD_LIBRARY_PATH=D:\Repo\decord\build-dev`
  切换；重建 `rebuild_dev.bat`（FFMPEG_DIR=D:/Software/ffmpeg-n9.0-...）。
  pip wheel 0.8.2 与引擎当前代码兼容（已验证 sha 逐位一致），仅缺该崩溃
  修复。诊断构建 `configure_asan.bat`（ASAN）。
- **v0.8.1 起硬性要求 FFmpeg 9（avcodec-63）**：FFmpeg 7/8 支持已删除
  （跨版本关键帧索引与 seek 落点正确性 bug，fork 内 D4 决策）。wheel 已
  打包 FFmpeg9 运行库，无外部依赖。
- 升级实测（0.8.2 wheel vs 0.8.1 本地构建，NT=24 顺序 av1）：1143 vs
  1143 fps，原生性能零差异；seek 成本本身不变（同步前向解码
  ~2.6ms/帧×关键帧距离，两代一致），但 **FFmpeg9 dav1d 帧线程扩展性
  大幅改善**（NT16 797fps → NT24 1164fps），引擎 av1 线程策略已随之
  调整（见 `extractor._decode_num_threads`，C-31）。
- 无 NVIDIA GPU 时自动回退 CPU 软解；强制 CPU 用 `decode_backend="cpu"`
  构造参数（`DECORD_FORCE_CPU` env 已于 0.9.2 删除）。
- `sample_stride>1` 的等差步长快速路径需要 fork ≥v0.7.12；旧版退化为逐索引
  seek，仍正确但更慢。
- `DECORD_SKIP_LOOP_FILTER` 透传（关去块滤波，可选的速度/准确率取舍旋钮）
  需要 fork ≥v0.7.13；旧版忽略该 env。

### TensorRT
- 11.3.0.99 评估（2026-09-15，log TRT113评估）：推理持平/无收益，留 11.2；
  升级时必须删引擎缓存（MVC 兼容加载会稀释成假数字）。
- `find_lib()` 只搜 `os.environ["PATH"]`，不认 `os.add_dll_directory()`；
  `TrtEngine` 初始化前会调用 `gpu_setup.ensure_gpu_initialized()` 更新 PATH。
- 首次构建 FP32 引擎约 1 分钟；FP16 构建慢 2.2 倍且推理无提升，不推荐。
- TRT 引擎与构建版本不兼容（10 产物无法被 11 加载）；加载失败会自动删除重建，
  不会静默回退 ONNX。
- **GPTuner（Global Performance Tuner）Windows 不可用**：`config.all_build_routes`
  在 Windows 返回空，调优只能走 Linux 或默认路线。

## 模型资产

- `assets/ocr_models/PP-OCRv6_rec_small.onnx`（**fp16 权重版，2026-09-14 起唯一分发模型**：与 TRT fp16 路径共用；六片 47134 帧真值代价 ±1 帧/片、方向随机，log 2026-09-14-OV读fp16模型评估；onnx/onnxconverter_common 依赖全路径归零）
- `assets/ocr_models/ppocrv6_dict.txt`

`ocr_native._models_dir()` / `ocr_trt._models_dir()` 支持：
1. frozen：`_MEIPASS/ocr_models`
2. 源码树：`<repo>/assets/ocr_models`
3. wheel 安装：`sys.prefix/assets/ocr_models`（data-files 布局）

## 检查更新

```bash
pip list --outdated
```

升级流程建议：
1. 升级单个包；
2. `python -m pytest tests/ -v`；
3. 用真实视频跑一次端到端（至少 CPU+CPU 与 GPU+TRT 各一次）；
4. 对比逐帧文本/置信度指纹，确认无读数漂移。

#### fork 本地开发构建（指针式——命令事实源在 fork 仓，勿在此复制）

**单一事实源**：`decord/CMakePresets.json`（dev/asan/release-check 三
preset）+ `decord/rebuild_dev.bat`（构建+部署+md5 校验一体）+
`decord/make_wheel.bat`（发 wheel）。本节只记事实源的用法与已知坑，
具体命令以脚本为准（0.17.0 审计轮：本节曾内联 2026-09-17 时代的
SDK 组装/手工 DLL 替换配方，连续三轮漂移成误导——去重为指针）：

- **dev 构建+部署**：fork 仓跑 `rebuild_dev.bat`（vcvars + `cmake
  --preset dev` + `--build` + 部署进 site-packages + env_doctor md5
  校验；会 `taskkill python`，先停本地测试）。
- **只构建不部署**：`build-dev\_build_dev.bat`（构建进 `build-dev/`，
  产物经 `DECORD_LIBRARY_PATH=build-dev` 切换；无缓存时自动先配置）。
- **FFMPEG_DIR**：BtbN `D:/Software/ffmpeg-n9.0-latest-win64-gpl-shared-9.0`
  （与 make_wheel.bat / release.yml CI 同源）。0.8.1 起**零 Toolkit**：
  驱动 API 动态加载，导入表无 nvcuvid 属正常（验证看 .obj）。
- **发 wheel**：本地 `make_wheel.bat`；正式发布走 GitHub Actions →
  Release 工作流（自动 tag + cp39–cp314 矩阵 + zip）。
- **git-bash 直调 .bat 陷阱**：引号转义会废掉 vcvars——脚本落盘后
  `cmd //c` 跑。MSVC 升级后旧 build 目录 CMake 缓存指向已删编译器：
  清 `CMakeCache.txt`+`CMakeFiles/` 重配。
- fork 遥测穿透 = `vr.hybrid_stats()`（`HybridStatsProbe` 线协议，
  report v6 `hybrid` 段）。
