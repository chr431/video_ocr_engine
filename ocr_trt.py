"""TensorRT 引擎构建 / 缓存 / 执行（OcrEngine 的 GPU 后端）。

与 ONNX 路径共享 OCR 预处理与 CTC 后处理；本模块只负责 TRT 特有逻辑：
引擎候选查找（模型目录 → 程序目录缓存 → 旧 LOCALAPPDATA 只读回退）、
反序列化校验、本地构建缓存与 CUDA 显存执行。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

import engine_config as config
log = logging.getLogger(__name__)


def _models_dir() -> Path:
    """模型资产目录（源码 / wheel 安装 / frozen 多路径兼容）。

    与 ocr_native._models_dir 保持相同查找顺序：源码树 → site-packages
    候选 → sys.prefix data-files 安装位置。
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", "")) / "ocr_models"
    here = Path(__file__).resolve().parent
    candidates = [
        here / "assets" / "ocr_models",
        here.parent / "assets" / "ocr_models",
        Path(sys.prefix) / "assets" / "ocr_models",
    ]
    marker = "PP-OCRv6_rec_small.onnx"
    for p in candidates:
        if (p / marker).is_file():
            return p
    return candidates[0]


class TrtEngine:
    """反序列化 TRT 引擎 + 执行上下文 + 输入/输出显存缓冲复用。"""

    def __init__(self, models: Path, size: str,
                 progress_cb=None) -> None:
        from gpu_setup import ensure_gpu_initialized
        # tensorrt_bindings.find_lib() 只搜 os.environ["PATH"]：
        # 首次使用前注册 CUDA/TensorRT DLL 目录（幂等）。
        ensure_gpu_initialized()

        self._progress_cb = progress_cb
        self.engine_path: Path | None = None

        # 逐个候选尝试加载：已存在的引擎可能是 TRT 版本/GPU 架构不匹配的
        # 陈旧产物。加载失败 → 删除（可写目录），尝试下一个候选；
        # 全部失败才进入重建（构建到本目录缓存，可写）。
        for cand in self._engine_candidates(size):
            if not cand.exists():
                continue
            try:
                self._load(cand)
                self.engine_path = cand
                log.info("TensorRT 引擎已加载: %s", self.engine_path)
                break
            except Exception as e:
                log.warning("TensorRT 引擎 %s 加载失败 (%s)，删除并尝试下一个候选",
                            cand.name, e)
                try:
                    cand.unlink(missing_ok=True)
                except OSError:
                    pass  # 只读目录（打包 EXE 内）删不掉，保留无害

        if self.engine_path is None:
            self.engine_path = self._engine_candidates(size)[1]  # 本目录缓存（可写）
            if self._progress_cb:
                self._progress_cb("TensorRT 引擎不存在，开始本地构建（首次运行，约 2 分钟）...")
            log.info("TensorRT 引擎不存在，开始本地构建（首次运行，约几分钟）...")
            self._build(models, size, self.engine_path)
            log.info("TensorRT 引擎已构建: %s", self.engine_path)
            if self._progress_cb:
                self._progress_cb("TensorRT 引擎构建完成")
            self._load(self.engine_path)

        # 输入/输出 device buffer 状态（host 侧不再保留中间 staging 数组）
        self._dev_in: int | None = None
        self._dev_out: int | None = None
        self._out_nbytes = 0
        self._stream = None  # CUDA stream：异步 HtoD/execute/DtoH 流水线用

    @staticmethod
    def _engine_candidates(size: str) -> list[Path]:
        """engine 查找顺序：模型目录（本机构建）→ 本目录缓存 → 旧 LOCALAPPDATA。

        - [0] 模型目录（打包只读，通常不存在）
        - [1] 本目录缓存（可写，构建目标 —— 免安装设计，不写 %LOCALAPPDATA%）
        - [2] 旧版本（≤v2.13）LOCALAPPDATA 缓存（只读兼容：已发布版本用户
          首次运行新版本可复用，避免重建；不写入）
        """
        name = (f"multi_PP-OCRv6_rec_{size}_{config.TRT_ENGINE_SM}"
                f"_fp32_tf32unset.engine")
        # 实验钩子：RVTOL_TRT_BATCH_PROFILE=N 用 batch=N 的 TRT profile 引擎
        # （独立缓存文件名 _pbN，不污染默认 batch 引擎）
        _pb = os.environ.get("RVTOL_TRT_BATCH_PROFILE")
        if _pb and _pb.isdigit():
            name = (f"multi_PP-OCRv6_rec_{size}_{config.TRT_ENGINE_SM}"
                    f"_fp32_tf32unset_pb{_pb}.engine")
        cands = [_models_dir() / "models" / name]
        cands.append(config.app_data_dir() / "ocr_engines" / name)
        legacy = (Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
                  / "RaceVideoToLog" / "ocr_engines" / name)
        cands.append(legacy)
        return cands

    def _load(self, engine_path: Path) -> None:
        """反序列化引擎并读取 profile 元数据；失败抛异常（由调用方决定重建/回退）。

        反序列化失败场景：TRT 版本升级后旧产物（序列化版本号不匹配）、
        GPU 架构不匹配（如 sm89 引擎换到 sm80 卡）。
        """
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:  # type: ignore[attr-defined]
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()  # type: ignore[attr-defined]
        in_name = self.engine.get_tensor_name(0)
        out_name = self.engine.get_tensor_name(1)
        prof_in = self.engine.get_tensor_profile_shape(in_name, 0)
        self.in_name = in_name
        self.out_name = out_name
        self.max_batch = int(prof_in[2][0])  # profile 的 batch 上限（如 6）
        self.max_in_shape = tuple(int(v) for v in prof_in[2])
        self._last_in_shape: tuple | None = None
        self._out_shape: tuple | None = None

    def _build(self, models: Path, size: str, engine_path: Path) -> None:
        """从 ONNX 构建 TRT 引擎（沿用 rapidocr 的 rec profile 配置）。"""
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]
        builder = trt.Builder(logger)  # type: ignore[attr-defined]
        # TRT 11 移除了 EXPLICIT_BATCH（隐式 batch 自 10 起已删，显式为默认），
        # getattr 回退保持 10/11 双兼容；TRT 11 下 flags=0 语义即显式 batch。
        try:
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # type: ignore[attr-defined]
        except AttributeError:
            flags = 0
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)  # type: ignore[attr-defined]
        onnx_path = models / f"PP-OCRv6_rec_{size}.onnx"
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError(f"ONNX 解析失败: {onnx_path}")
        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,  # type: ignore[attr-defined]
            config.TRT_WORKSPACE_BYTES)
        profile = builder.create_optimization_profile()
        _pb = os.environ.get("RVTOL_TRT_BATCH_PROFILE")
        opt_b = int(_pb) if (_pb and _pb.isdigit()) else config.TRT_PROFILE_BATCH
        h = config.OCR_TARGET_H
        profile.set_shape(
            network.get_input(0).name,
            min=(1, 3, h, config.TRT_PROFILE_MIN_W),
            opt=(opt_b, 3, h, config.TRT_PROFILE_OPT_W),
            max=(opt_b, 3, h, config.TRT_PROFILE_MAX_W))
        builder_config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, builder_config)
        if serialized is None:
            raise RuntimeError("TRT engine 构建失败")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)

    def _prepare_shape(self, shape: tuple) -> tuple:
        """更新 TRT context 输入 shape（幂等），返回输出 shape。"""
        if self._last_in_shape != shape:
            self.context.set_input_shape(self.in_name, shape)
            self._last_in_shape = shape
            self._out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        return self._out_shape

    def _ensure_stream(self) -> int:
        """创建专用 CUDA stream（TRT async 必须用非默认流）。"""
        if self._stream is None:
            from cuda.bindings import runtime as cudart  # type: ignore[import-not-found]
            _err, self._stream = cudart.cudaStreamCreate()
        return self._stream

    def synchronize(self) -> None:
        """等待当前 CUDA stream 上的所有 async 操作完成。"""
        if self._stream is None:
            return
        from cuda.bindings import runtime as cudart  # type: ignore[import-not-found]
        cudart.cudaStreamSynchronize(self._stream)

    def execute_async(self, x: np.ndarray,
                      out_host: "np.ndarray | None" = None) -> np.ndarray:
        """异步执行一批输入（batch ≤ max_batch），不等待完成。

        调用方必须在读取 out_host/host_out 前调用 synchronize()。
        out_host 提供时直接把 DtoH 结果写入该 float32 连续数组，避免每次
        额外分配 host_out 并在 _infer_locked 中再 concatenate。
        """
        from cuda.bindings import runtime as cudart  # type: ignore[import-not-found]
        stream = self._ensure_stream()
        out_shape = self._prepare_shape(x.shape)
        # 输入 device buffer：按 max profile 形状预分配并复用。
        # 直接以当前批的 numpy 内存作为 HtoD 源，去掉旧 host_in staging 拷贝：
        # 预处理结果本来就是 host float32 连续数组，无需再平铺进一块固定 host buffer。
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x)
        if self._dev_in is None:
            size_in = int(np.prod(self.max_in_shape)) * 4
            _, self._dev_in = cudart.cudaMalloc(size_in)
        dev_in = self._dev_in
        cudart.cudaMemcpyAsync(
            dev_in, x.ctypes.data, x.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        # 输出 device buffer 按需增长复用（cudaMalloc 每次 ~ms，避免每片分配）
        out_nbytes = int(np.prod(out_shape)) * 4
        if self._dev_out is None or out_nbytes > self._out_nbytes:
            if self._dev_out is not None:
                cudart.cudaFree(self._dev_out)
            _, self._dev_out = cudart.cudaMalloc(out_nbytes)
            self._out_nbytes = out_nbytes
        dev_out = self._dev_out
        # execute_async_v3 需要显式设置输入/输出 tensor 地址
        self.context.set_tensor_address(self.in_name, dev_in)
        self.context.set_tensor_address(self.out_name, dev_out)
        self.context.execute_async_v3(stream)
        if out_host is not None:
            if (not out_host.flags.c_contiguous
                    or out_host.dtype != np.float32
                    or out_host.nbytes < out_nbytes):
                raise ValueError("out_host 必须是足够大的 float32 连续数组")
            cudart.cudaMemcpyAsync(
                out_host.ctypes.data, dev_out, out_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
            return out_host
        host_out = np.empty(out_shape, dtype=np.float32)
        cudart.cudaMemcpyAsync(
            host_out.ctypes.data, dev_out, out_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        return host_out

    def execute(self, x: np.ndarray, out_host: "np.ndarray | None" = None) -> np.ndarray:
        """同步执行一批输入（batch ≤ max_batch），复用输入/输出 buffer。"""
        result = self.execute_async(x, out_host)
        self.synchronize()
        return result
