# 依赖与运行环境（video_ocr_engine）

> 迁移自 RaceVideoToLog/DEPENDENCIES.md，只保留引擎识别链相关依赖与性能笔记。
> 版本号以 2026-08 实测为准；`pyproject.toml` 中的下限约束是兼容基线。

## 核心依赖

| 包 | 当前版本 | 来源 | 说明 |
| --- | --- | --- | --- |
| numpy | 2.x | PyPI | 预处理/信号计算，纯 numpy 无 scipy |
| onnxruntime | 1.29.x | PyPI | CPU OCR 后端；1.28 含 protobuf CVE 修复；1.29.0 实测升级安全、性能持平 |
| psutil | 6+ | PyPI | 物理核数探测 / RSS 采样（缺失时降级） |
| decord | **0.8.3（本地 wheel 已装，GitHub 发布待 dispatch，§15）** | chr431/decord release | NVDEC 硬解 + CPU 软解；**PyPI 官方版不支持** `next_roi` / ROI-first / GPU gray / YUV420 / `sample_stride` 等差步长快速路径；fork 自 0.8.2 起发布 cp39–cp314 全版本 wheel，`pip install <wheel>` 即用 |
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
- **0.8.3（2026-09-12 发布验证轮完成 + §16 死锁修复追加，GitHub 发布待用户
  dispatch）**：fork master 领先 0.8.2 十个提交（73e5540 析构 UAF 修复 /
  d94d92e hybrid GPU 池深按 ROI 重算 / 3b96c6f hybrid chunk 预路由 / 93a5ce1 /
  **99b8785 FORCE_SIDE 诊断臂死锁修复** / **75b8602 hybrid 非打印 stats 层 +
  sustained 产能估计器（实验性 opt-in）** / **4ccf889 sustained 产能口径转默认
  + EWMA 旧口径删除 + kick 突发治倾斜计划死锁** / **281a738 上载批大小消融
  旋钮 + kick 承重判别** / **02c91e6→**b67f9eb** 宿主慢消费者环死锁修复——反馈式
  清偿（债务快照+克隆至清偿，EOF 冲刷兜底）取代定长突发**），本地提交 486d2f6 已 bump 版本号（三源一致）。
  - **本地 cp313 wheel 已构建并安装**（含 b67f9eb，sha256 `8e6f17a5…`，
    64MB dll 闭包完整；0.8.2 wheel 在 fork `dist/` 可回滚）。**wheel 口径
    验证全绿**（§15/§16）：金标 28/28 ×2（重录整矩阵同序——局部
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
  wheel**）。开发态用 `DECORD_LIBRARY_PATH=D:\Repo\decord\build-081fix`
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
