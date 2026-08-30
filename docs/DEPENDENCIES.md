# 依赖与运行环境（video_ocr_engine）

> 迁移自 RaceVideoToLog/DEPENDENCIES.md，只保留引擎识别链相关依赖与性能笔记。
> 版本号以 2026-08 实测为准；`pyproject.toml` 中的下限约束是兼容基线。

## 核心依赖

| 包 | 当前版本 | 来源 | 说明 |
| --- | --- | --- | --- |
| numpy | 2.x | PyPI | 预处理/信号计算，纯 numpy 无 scipy |
| onnxruntime | 1.29.x | PyPI | CPU OCR 后端；1.28 含 protobuf CVE 修复；1.29.0 实测升级安全、性能持平 |
| psutil | 6+ | PyPI | 物理核数探测 / RSS 采样（缺失时降级） |
| decord | **自建 fork** | chr431/decord | NVDEC 硬解 + CPU 软解；**PyPI 版不支持** `next_roi` / ROI-first / GPU gray / YUV420 / `sample_stride` 等差步长快速路径 |
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

### decord（自建 fork）
- 必须使用 `chr431/decord` release 构建，不能使用 PyPI 版。
- 需与对应 FFmpeg DLL 同目录（Windows）。
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
