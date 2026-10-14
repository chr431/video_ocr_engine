# 依赖与运行环境（渲染产物；事实源=pyproject + registry/environment.yaml）

> 本文件由 tools/render.py 渲染，人不得手写；fork 版本史经 git 档案
> 取回，换代教训见 docs/CONCLUSIONS.md。

## Python 依赖（约束=pyproject；本机口径=environment.yaml）

| pyproject 约束 | 本机 |
|---|---|
| `numpy>=2` | 2.x |
| `openvino>=2026.3,<2026.5` | 2026.4.1 |
| `psutil>=6` | 6+ |

## 本机外部件（不在 PyPI 依赖；事实源=environment.yaml）

| 件 | 口径 | 说明 |
|---|---|---|
| decord（fork） | decord@0.9.8 | chr431/decord GitHub Release（cp39–cp314 wheel；本地构建走 fork 仓 rebuild_dev.bat）；NVDEC 硬解 + CPU 软解 |
| cuda-python | 13.x | TRT 执行 + decord GPU DLL 注册 |
| TensorRT | 11.2 | 首次运行自动建引擎缓存到 ocr_engines/ |
| FFmpeg | 9（avcodec-63） | wheel 已打包运行库 |

## 开发依赖（钉版=pyproject dev extras；与 CI lint job 同源）

- `pytest>=8`
- `ruff==0.16.6`
- `mypy==2.3.1`
- `pyyaml`

## fork 能力 floors（CI 校验 pin ≥ 每个 floor）

- decord_stride_fastpath ≥ 0.7.12
- decord_env_passthrough ≥ 0.7.13
- decord_hybrid_native ≥ 0.7.15
- decord_setroi_deepbank ≥ 0.8.5

### GPU 运行时

- CUDA Toolkit 与 TensorRT 的 DLL 从系统 PATH 加载（不打包）；gpu_setup.ensure_gpu_initialized() 扫描 PATH 注册 DLL 目录并前置 PATH（tensorrt 的 find_lib() 只搜 PATH）
- TRT 引擎与构建版本不兼容（旧产物不可加载）；加载失败自动删缓存重建，不静默回退；FP32 首建约 1 分钟，FP16 慢 2.2× 且推理无提升；GPTuner 在 Windows 不可用

### decord 操作事实

- 无 NVIDIA GPU 自动回退 CPU 软解；强制 CPU 用 decode_backend="cpu" 构造参数
- h264 NVDEC 是本机硬件天花板（test6 全片 23970 帧引擎口径三码 NVDEC 942/2074/1724 fps，2026-10-14 r84 复测）——h264 峰值走 CPU 软解或显式 hybrid
- fork 能力 floors（步长快速路径/env 透传/hybrid 原生/SetRoi 深银行）唯一住本表 dependency_floors

### 模型资产

- assets/ocr_models/PP-OCRv6_rec_small.onnx（fp16 权重版，唯一分发模型）+ ppocrv6_dict.txt；_models_dir() 支持 frozen（_MEIPASS）/源码树/wheel（data-files）三布局

### fork 本地开发构建（命令事实源在 fork 仓）

- 构建命令的事实源在 fork 仓（CMakePresets.json + rebuild_dev.bat 构建即部署含 md5 校验 + make_wheel.bat 发 wheel）；rebuild_dev.bat 会 taskkill python，先停本地测试
- git-bash 直调 .bat 的引号转义会废 vcvars——脚本落盘后经 cmd //c 跑；MSVC 升级后清 CMakeCache.txt + CMakeFiles/ 重配
- 改 hybrid_threaded_decoder.h / ffmpeg/threaded_decoder.h 的类布局后必须删 video_reader.cc.obj 再构建（增量构建不重编=静默行为损坏，两起事故）；换 DLL 的 A/B 两臂都要全对象干净重建

## 升级流程

1. 升级单个包 → 2. `python -m pytest tests/ -v` → 3. 真实视频端到端
（CPU+CPU 与 GPU+TRT 各一）→ 4. 对比逐帧文本/置信度指纹确认无读数漂移。

