# 依赖与运行环境（video_ocr_engine）

> 迁移自 RaceVideoToLog/DEPENDENCIES.md，只保留引擎识别链相关依赖与性能笔记。
> 版本号以 2026-08 实测为准；`pyproject.toml` 中的下限约束是兼容基线。

## 核心依赖

| 包 | 当前版本 | 来源 | 说明 |
| --- | --- | --- | --- |
| numpy | 2.x | PyPI | 预处理/信号计算，纯 numpy 无 scipy |
| onnxruntime | 1.29.x | PyPI | CPU OCR 后端；1.28 含 protobuf CVE 修复；1.29.0 实测升级安全、性能持平 |
| psutil | 6+ | PyPI | 物理核数探测 / RSS 采样（缺失时降级） |
| decord | **0.8.3（已发布 [v0.8.3](https://github.com/chr431/decord/releases/tag/v0.8.3)，含 §16/§17 死锁修复）** | chr431/decord release | NVDEC 硬解 + CPU 软解；**PyPI 官方版不支持** `next_roi` / ROI-first / GPU gray / YUV420 / `sample_stride` 等差步长快速路径；fork 自 0.8.2 起发布 cp39–cp314 全版本 wheel，`pip install <wheel>` 即用 |
| cuda-python | 13.x | PyPI | TRT 执行 + decord GPU DLL 注册 |
| tensorrt_*_bindings | 11.x | PyPI | TensorRT thin binding（~1MB）；运行 DLL 从系统 PATH 加载 |

> `tensorrt` 元包与 `tensorrt_*_libs`（~2.2GB DLL）被有意排除。运行时从
> NVIDIA 官网安装的 CUDA Toolkit / TensorRT 的 `bin` 目录加载 DLL。

## GPU 加速（运行时，不打包）

| 组件 | 来源 | 说明 |
| --- | --- | --- |
| CUDA Toolkit 13.x | NVIDIA 官网 | cudart/cublas 等 DLL，需在 PATH |
| TensorRT | NVIDIA 官网 | nvinfer DLL，需在 PATH；首次运行自动构建引擎缓存到 `ocr_engines/` |

`gpu_setup.ensure_gpu_initialized()` 会扫描 PATH 并注册 DLL 目录，同时把找到的
目录前置到 `os.environ["PATH"]`（`tensorrt` 的 `find_lib()` 只搜 PATH）。

## 已知问题与注意

### decord（自建 fork，pip wheel 安装）
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
  - 开发 dll md5 `2fd49ea5`（build-081fix，2026-09-12 replan 回滚后从
    HEAD 281a738 重建——MSVC 非确定性构建产物，md5 每次重建会变，以
    行为判别为准：回滚后构建的 stats 行无 `replans=` 字段）。
- **h264 NVDEC 是本机硬件天花板，非软件问题**（2026-09-10 实测）：同内容
  三编码 NVDEC 982(h264)/2103(hevc)/1749(av1) fps；ffmpeg 自带 cuvid 同比
  （16.5x vs 38.7x）。h264 解码选峰值应走 CPU 软解/显式 hybrid。
- **0.8.2 起改用 release wheel + pip 安装**（不再 editable 源码导入、不再
  手工部署 DLL）：wheel 自带 `decord.dll` 与 FFmpeg 63 运行库（包根目录），
  `pip install decord-0.8.2-cp313-cp313-win_amd64.whl` 即用；0.8.2 已含
  cuMemcpy2D_v2 修复（fork 7ef70f5）。当前 dll md5 6597eea6。
- **本地开发 dll（2026-09-10，含未发布修复）**：`build-081fix/decord.dll`
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
  - dev dll：`build-081fix/decord.dll` = `bc7c27a1…`（同源 4cebeef 干净
    重建；MSVC 非确定性 → md5 与 wheel 内不同属正常）。
  - **构建陷阱（本轮两事故）**：本机 ninja 对 `video_reader.cc.obj` 无头依赖
    （`ninja -C build-081fix -t deps` 显示 `#deps 0`）——改
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
    必须显式用 `c:/Users/eric chen/AppData/Local/Programs/Python/python313/python.exe`。开发态用 `DECORD_LIBRARY_PATH=D:\Repo\decord\build-081fix`
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
  构造参数（`DECORD_FORCE_CPU` env 已于 0.9.0 删除）。
- `sample_stride>1` 的等差步长快速路径需要 fork ≥v0.7.12；旧版退化为逐索引
  seek，仍正确但更慢。
- `DECORD_SKIP_LOOP_FILTER` 透传（关去块滤波，可选的速度/准确率取舍旋钮）
  需要 fork ≥v0.7.13；旧版忽略该 env。

### onnxruntime
- TRT/CUDA provider DLL 不通过 ORT provider 使用；TRT 由 `ocr_trt.TrtEngine`
  直接调用。
- 1.29 新增参数（`ORT_INTRA/INTER_OP_NUM_THREADS`、parallel 执行、spin off）
  实测无收益，未启用。

### TensorRT
- `find_lib()` 只搜 `os.environ["PATH"]`，不认 `os.add_dll_directory()`；
  `TrtEngine` 初始化前会调用 `gpu_setup.ensure_gpu_initialized()` 更新 PATH。
- 首次构建 FP32 引擎约 1 分钟；FP16 构建慢 2.2 倍且推理无提升，不推荐。
- TRT 引擎与构建版本不兼容（10 产物无法被 11 加载）；加载失败会自动删除重建，
  不会静默回退 ONNX。
- **GPTuner（Global Performance Tuner）Windows 不可用**：`config.all_build_routes`
  在 Windows 返回空，调优只能走 Linux 或默认路线。

## 模型资产

- `assets/ocr_models/PP-OCRv6_rec_small.onnx`
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
