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


class GpuPreprocessor:
    """TRT 路径的 GPU 预处理：把已 48 高的 float32 HWC 图直接变成模型输入。

    只在 GPU 上完成最后一层 transpose + normalize + pad：
    - 输入：list[float32 ndarray] (H, W_i, C)，已由 _preprocess_standard 缩到 48 高
    - 输出：device float32 tensor (B, C, H, W_out)，可被 TrtEngine 直接执行
    该路径避免在 host 上构造完整 padded batch，并让 DtoH 前始终是显存数据。
    """
    _KERNEL = r'''
extern "C" __global__ void prep(
    const float* __restrict__ raw,
    const int* __restrict__ widths,
    float* __restrict__ out,
    int B, int H, int W_out, int C) {
    int total = B * C * H * W_out;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int b = i / (C * H * W_out);
    int rem = i % (C * H * W_out);
    int c = rem / (H * W_out);
    int rem2 = rem % (H * W_out);
    int y = rem2 / W_out;
    int x = rem2 % W_out;
    int w = widths[b];
    if (x < w) {
        int src = ((b * H + y) * w + x) * C + c;
        float v = raw[src];
        out[i] = (v / 255.0f - 0.5f) / 0.5f;
    } else {
        out[i] = 0.0f;
    }
}

extern "C" __global__ void prep_gray_raw(
    const unsigned char* __restrict__ raw,
    float* __restrict__ out,
    int B, int src_h, int src_w,
    int dst_h, int dst_w, int content_w, float gamma) {
    int total = B * 3 * dst_h * dst_w;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int b = i / (3 * dst_h * dst_w);
    int rem = i % (3 * dst_h * dst_w);
    int c = rem / (dst_h * dst_w);
    int rem2 = rem % (dst_h * dst_w);
    int y = rem2 / dst_w;
    int x = rem2 % dst_w;
    if (x >= content_w) { out[i] = 0.0f; return; }
    double sx = (x + 0.5) * ((double)src_w / (double)content_w) - 0.5;
    double sy = (y + 0.5) * ((double)src_h / (double)dst_h) - 0.5;
    sx = fmax(0.0, fmin(sx, (double)(src_w - 1)));
    sy = fmax(0.0, fmin(sy, (double)(src_h - 1)));
    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    double wx = sx - x0, wy = sy - y0;
    const unsigned char* base = raw + ((size_t)b * src_h * src_w);
    double v00 = base[y0 * src_w + x0];
    double v10 = base[y0 * src_w + x1];
    double v01 = base[y1 * src_w + x0];
    double v11 = base[y1 * src_w + x1];
    double g = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 +
               (1 - wx) * wy * v01 + wx * wy * v11;
    double g2 = 255.0 * pow(g / 255.0, (double)gamma);
    out[i] = (g2 / 255.0f - 0.5f) / 0.5f;
}
'''

    def __init__(self) -> None:
        import numpy as np  # noqa: F401
        from cuda.core import (Buffer, Device, LaunchConfig, Program,
                               ProgramOptions, launch)
        self._dev = Device()
        self._dev.set_current()
        self._prog = Program(
            self._KERNEL, code_type="c++",
            options=ProgramOptions(std="c++11",
                                   arch=f"sm_{self._dev.arch}"))
        self._mod = self._prog.compile(
            "cubin", name_expressions=("prep", "prep_gray_raw"))
        self._kernel = self._mod.get_kernel("prep")
        self._kernel_raw = self._mod.get_kernel("prep_gray_raw")
        self._launch_cls = LaunchConfig
        self._launch = launch
        self._buffer_cls = Buffer
        from cuda.bindings import runtime as cudart
        _err, self._stream = cudart.cudaStreamCreate()
        self._raw_size = 0
        self._raw_dev = None
        self._raw_buf = None
        self._out_size = 0
        self._out_dev = None
        self._out_buf = None
        self._width_size = 0
        self._width_dev = None
        self._width_buf = None

    def _ensure_raw(self, nbytes: int) -> int:
        from cuda.bindings import runtime as cudart
        if self._raw_size < nbytes:
            if self._raw_dev is not None:
                cudart.cudaFree(self._raw_dev)
            _err, self._raw_dev = cudart.cudaMalloc(nbytes)
            self._raw_buf = self._buffer_cls.from_handle(self._raw_dev, nbytes)
            self._raw_size = nbytes
        return self._raw_dev

    def _ensure_width(self, nbytes: int) -> int:
        from cuda.bindings import runtime as cudart
        if self._width_size < nbytes:
            if self._width_dev is not None:
                cudart.cudaFree(self._width_dev)
            _err, self._width_dev = cudart.cudaMalloc(nbytes)
            self._width_buf = self._buffer_cls.from_handle(self._width_dev, nbytes)
            self._width_size = nbytes
        return self._width_dev

    def _ensure_out(self, nbytes: int) -> int:
        from cuda.bindings import runtime as cudart
        if self._out_size < nbytes:
            if self._out_dev is not None:
                cudart.cudaFree(self._out_dev)
            _err, self._out_dev = cudart.cudaMalloc(nbytes)
            self._out_buf = self._buffer_cls.from_handle(self._out_dev, nbytes)
            self._out_size = nbytes
        return self._out_dev

    def process_gray_raw(self, infos: list, out_width: int):
        """处理 decord GPU 灰度帧：D2D 聚批 → GPU resize+gamma+normalize+pad。

        infos: [(dev_ptr, src_h, src_w, owner), ...]，frame 已位于显存。
        返回 (device_ptr, output_shape)。调用方必须保持 owner/本对象存活。
        """
        import numpy as np
        from cuda.bindings import runtime as cudart
        B = len(infos)
        src_h = int(infos[0][1])
        src_w = int(infos[0][2])
        dst_h = int(config.OCR_TARGET_H)
        content_w = max(1, int(src_w * dst_h / src_h))
        dst_w = int(out_width)
        if content_w > dst_w:
            dst_w = content_w
        raw_nbytes = B * src_h * src_w
        raw_dev = self._ensure_raw(raw_nbytes)
        for i, (src_dev, _sh, _sw, _owner) in enumerate(infos):
            cudart.cudaMemcpyAsync(
                raw_dev + i * src_h * src_w,
                int(src_dev), src_h * src_w,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                self._stream)
        out_nbytes = B * 3 * dst_h * dst_w * 4
        out_dev = self._ensure_out(out_nbytes)
        gamma = float(config.OCR_GAMMA)
        _env_g = os.environ.get("RVTOL_OCR_GAMMA")
        if _env_g:
            try:
                gamma = float(_env_g)
            except ValueError:
                pass
        shape = (B, 3, dst_h, dst_w)
        total = int(np.prod(shape))
        block = 256
        grid = (total + block - 1) // block
        self._launch(
            self._stream,
            self._launch_cls(grid=grid, block=block),
            self._kernel_raw,
            self._raw_buf, self._out_buf,
            np.int32(B), np.int32(src_h), np.int32(src_w),
            np.int32(dst_h), np.int32(dst_w), np.int32(content_w),
            np.float32(gamma))
        return int(out_dev), shape

    def process(self, images: list, out_width: int):
        """返回 (device_ptr, output_shape)。调用方必须保持本对象存活。"""
        import numpy as np
        from cuda.bindings import runtime as cudart
        B = len(images)
        H = int(images[0].shape[0])
        C = int(images[0].shape[2])
        # 平铺每个图的真实像素，只有实际宽度数据上 GPU；pad 由 kernel 补 0
        raw = np.concatenate(
            [im.reshape(-1) for im in images]).astype(np.float32, copy=False)
        widths = np.array([int(im.shape[1]) for im in images], dtype=np.int32)

        raw_dev = self._ensure_raw(raw.nbytes)
        width_dev = self._ensure_width(widths.nbytes)
        out_nbytes = B * C * H * out_width * 4
        out_dev = self._ensure_out(out_nbytes)

        cudart.cudaMemcpyAsync(
            raw_dev, raw.ctypes.data, raw.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)
        cudart.cudaMemcpyAsync(
            width_dev, widths.ctypes.data, widths.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)

        shape = (B, C, H, out_width)
        total = int(np.prod(shape))
        block = 256
        grid = (total + block - 1) // block
        self._launch(
            self._stream,
            self._launch_cls(grid=grid, block=block),
            self._kernel,
            self._raw_buf, self._width_buf, self._out_buf,
            np.int32(B), np.int32(H), np.int32(out_width), np.int32(C))
        return int(out_dev), shape


class GpuFrameAnalyzer:
    """GPU 帧分析：sharp(std) + 相邻帧 3x3 聚类变化分，直接把标量回传 host。

    当前用于实验性 GPU 分段路径：只把每帧的 (sharp, cluster_score) 小数组
    D2H，不再把整帧 ROI 灰度拷贝回 host。
    """
    _KERNEL = r'''
extern "C" __global__ void analyze_gray(
    const unsigned char* __restrict__ raw,
    const unsigned char* __restrict__ prev,
    float* __restrict__ summary,
    int B, int H, int W, float th) {
    int b = blockIdx.x;
    if (b >= B) return;
    const unsigned char* cur = raw + (size_t)b * H * W;
    const unsigned char* pre = prev + (size_t)b * H * W;
    double sum = 0.0, sum2 = 0.0;
    int maxc = 0;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int v = cur[y * W + x];
            sum += v; sum2 += (double)v * v;
            int s = 0;
            for (int dy = -1; dy <= 1; dy++) {
                int yy = y + dy;
                if (yy < 0 || yy >= H) continue;
                for (int dx = -1; dx <= 1; dx++) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= W) continue;
                    int cv = cur[yy * W + xx];
                    int pv = pre[yy * W + xx];
                    if ((cv > th) != (pv > th)) s++;
                }
            }
            if (s > maxc) maxc = s;
        }
    }
    double mean = sum / (double)(H * W);
    double var = sum2 / (double)(H * W) - mean * mean;
    summary[b * 2] = (float)(var > 0.0 ? sqrt(var) : 0.0);
    summary[b * 2 + 1] = (float)maxc;
}
'''

    def __init__(self) -> None:
        import numpy as np  # noqa: F401
        from cuda.core import Device, Program, ProgramOptions
        self._dev = Device()
        self._dev.set_current()
        self._prog = Program(
            self._KERNEL, code_type="c++",
            options=ProgramOptions(std="c++11",
                                   arch=f"sm_{self._dev.arch}"))
        self._mod = self._prog.compile("cubin", name_expressions=("analyze_gray",))
        self._kernel = self._mod.get_kernel("analyze_gray")
        from cuda.bindings import runtime as cudart
        _err, self._stream = cudart.cudaStreamCreate()
        self._summary_size = 0
        self._summary_dev = None
        self._prev_size = 0
        self._prev_dev = None

    def _ensure_prev(self, nbytes: int) -> int:
        from cuda.bindings import runtime as cudart
        if self._prev_size < nbytes:
            if self._prev_dev is not None:
                cudart.cudaFree(self._prev_dev)
            _err, self._prev_dev = cudart.cudaMalloc(nbytes)
            self._prev_size = nbytes
        return self._prev_dev

    def analyze_batch(self, raw_ptr: int, prev_ptr: int, B: int,
                      H: int, W: int, th: float) -> "np.ndarray":
        """一次 kernel 分析 B 帧；prev_ptr 必须是已准备好的 B 帧前帧缓冲。"""
        import numpy as np
        from cuda.bindings import runtime as cudart
        from cuda.core import Buffer, LaunchConfig, launch
        nbytes = B * 2 * 4
        if self._summary_size < nbytes:
            if self._summary_dev is not None:
                cudart.cudaFree(self._summary_dev)
            _err, self._summary_dev = cudart.cudaMalloc(nbytes)
            self._summary_size = nbytes
        buf = Buffer.from_handle(self._summary_dev, nbytes)
        launch(self._stream, LaunchConfig(grid=B, block=1), self._kernel,
               Buffer.from_handle(raw_ptr, B * H * W),
               Buffer.from_handle(prev_ptr, B * H * W),
               buf, np.int32(B), np.int32(H), np.int32(W), np.float32(th))
        cudart.cudaStreamSynchronize(self._stream)
        out = np.empty((B, 2), dtype=np.float32)
        cudart.cudaMemcpy(
            out.ctypes.data, self._summary_dev, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return out

    def analyze(self, raw_ptr: int, prev_ptr: int, B: int, H: int, W: int,
                th: float) -> "np.ndarray":
        """返回 (B,2) float32 host 数组：每帧 (sharp, cluster_score)。"""
        import numpy as np
        from cuda.bindings import runtime as cudart
        from cuda.core import Buffer, LaunchConfig, launch
        nbytes = B * 2 * 4
        if self._summary_size < nbytes:
            if self._summary_dev is not None:
                cudart.cudaFree(self._summary_dev)
            _err, self._summary_dev = cudart.cudaMalloc(nbytes)
            self._summary_size = nbytes
        buf = Buffer.from_handle(self._summary_dev, nbytes)
        launch(self._stream, LaunchConfig(grid=B, block=1), self._kernel,
               Buffer.from_handle(raw_ptr, B * H * W),
               Buffer.from_handle(prev_ptr, B * H * W),
               buf, np.int32(B), np.int32(H), np.int32(W), np.float32(th))
        cudart.cudaStreamSynchronize(self._stream)
        out = np.empty((B, 2), dtype=np.float32)
        cudart.cudaMemcpy(
            out.ctypes.data, self._summary_dev, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return out


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

    def execute_device_async(self, dev_input: int, shape: tuple,
                             out_host: "np.ndarray | None" = None) -> np.ndarray:
        """异步执行已位于显存的输入（GPU 预处理结果），省去 HtoD。

        dev_input 必须是当前 stream 上有效的 device 指针；调用方须在读取
        out_host 前 synchronize()。
        """
        from cuda.bindings import runtime as cudart  # type: ignore[import-not-found]
        stream = self._ensure_stream()
        out_shape = self._prepare_shape(shape)
        out_nbytes = int(np.prod(out_shape)) * 4
        if self._dev_out is None or out_nbytes > self._out_nbytes:
            if self._dev_out is not None:
                cudart.cudaFree(self._dev_out)
            _, self._dev_out = cudart.cudaMalloc(out_nbytes)
            self._out_nbytes = out_nbytes
        dev_out = self._dev_out
        self.context.set_tensor_address(self.in_name, dev_input)
        self.context.set_tensor_address(self.out_name, dev_out)
        self.context.execute_async_v3(stream)
        if out_host is not None:
            if (not out_host.flags.c_contiguous or out_host.dtype != np.float32
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

    def execute_device(self, dev_input: int, shape: tuple,
                       out_host: "np.ndarray | None" = None) -> np.ndarray:
        """同步执行显存输入。"""
        result = self.execute_device_async(dev_input, shape, out_host)
        self.synchronize()
        return result

    def execute(self, x: np.ndarray, out_host: "np.ndarray | None" = None) -> np.ndarray:
        """同步执行一批输入（batch ≤ max_batch），复用输入/输出 buffer。"""
        result = self.execute_async(x, out_host)
        self.synchronize()
        return result
