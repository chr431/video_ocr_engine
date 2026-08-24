"""GPU 内核类（从 ocr_trt.py 拆出）：显存上的预处理 / 归约 / 帧分析。

GpuPreprocessor（resize+gamma+normalize+pad 内核）、GpuOutputReducer
（vocab 维 argmax 归约）、GpuFrameAnalyzer（sharp/cluster 逐帧分析 +
逐帧直方图 + merge_similar 差异标量）。TrtEngine 保留在 ocr_trt.py；
ocr_trt 顶部 re-export 本模块三个类，保持旧导入路径兼容。
"""
from __future__ import annotations

import os
import time

import numpy as np

import engine_config as config

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
        _env_g = os.environ.get("OCR_GAMMA")
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


class GpuOutputReducer:
    """TRT 输出的 GPU 侧 vocab 维归约：(B,S,C) float32 → (B,S) 索引+概率。

    DtoH 数据量从 B*S*C*4 字节降到 B*S*8 字节（宽 ROI 下数千倍），使 TRT
    推理输出不落 RAM——显存全驻留路径的最后一环。并列取首个，与
    numpy.argmax 语义一致。
    """

    _KERNEL = r'''
extern "C" __global__ void argmax_last(
    const float* __restrict__ preds,
    int* __restrict__ idx_out,
    float* __restrict__ prob_out,
    long long total_rows, int C) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_rows) return;
    const float* row = preds + (size_t)i * C;
    float best = row[0];
    int bi = 0;
    for (int c = 1; c < C; ++c) {
        float v = row[c];
        if (v > best) { best = v; bi = c; }
    }
    idx_out[i] = bi;
    prob_out[i] = best;
}
'''

    def __init__(self, stream: int | None = None) -> None:
        import numpy as np  # noqa: F401
        from cuda.core import (Buffer, Device, LaunchConfig, Program,
                               ProgramOptions, launch)
        self._dev = Device()
        self._dev.set_current()
        self._prog = Program(
            self._KERNEL, code_type="c++",
            options=ProgramOptions(std="c++11",
                                   arch=f"sm_{self._dev.arch}"))
        self._mod = self._prog.compile("cubin", name_expressions=("argmax_last",))
        self._kernel = self._mod.get_kernel("argmax_last")
        self._launch_cls = LaunchConfig
        self._launch = launch
        self._buffer_cls = Buffer
        from cuda.bindings import runtime as cudart
        self._stream = stream
        if self._stream is None:
            _err, self._stream = cudart.cudaStreamCreate()
        self._idx_dev = None
        self._idx_size = 0
        self._prob_dev = None

    def reduce(self, preds_dev: int, out_shape: tuple):
        """对显存中的 (B,S,C) 输出做归约，回传 (idx int32[B*S], prob f32[B*S])。"""
        import numpy as np
        from cuda.bindings import runtime as cudart
        rows = int(np.prod(out_shape[:-1], dtype=np.int64))
        cdim = int(out_shape[-1])
        nbytes_idx = rows * 4
        if self._idx_size < nbytes_idx:
            if self._idx_dev is not None:
                cudart.cudaFree(self._idx_dev)
            _err, self._idx_dev = cudart.cudaMalloc(nbytes_idx)
            self._idx_size = nbytes_idx
            self._prob_dev = None
        if self._prob_dev is None:
            _err, self._prob_dev = cudart.cudaMalloc(nbytes_idx)
        idx_buf = self._buffer_cls.from_handle(self._idx_dev, nbytes_idx)
        prob_buf = self._buffer_cls.from_handle(self._prob_dev, nbytes_idx)
        block = 256
        grid = (rows + block - 1) // block
        self._launch(self._stream,
                     self._launch_cls(grid=grid, block=block),
                     self._kernel,
                     self._buffer_cls.from_handle(int(preds_dev),
                                                  rows * cdim * 4),
                     idx_buf, prob_buf,
                     np.int64(rows), np.int32(cdim))
        idx = np.empty(rows, dtype=np.int32)
        prob = np.empty(rows, dtype=np.float32)
        cudart.cudaMemcpy(idx.ctypes.data, self._idx_dev, nbytes_idx,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        cudart.cudaMemcpy(prob.ctypes.data, self._prob_dev, nbytes_idx,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return idx, prob


class GpuFrameAnalyzer:
    """GPU 帧分析：sharp(std) + 相邻帧 3x3 聚类变化分，直接把标量回传 host。

    GPU 分段路径专用：只把每帧的 (sharp, cluster_score) 小数组 D2H，
    不再把整帧 ROI 灰度拷贝回 host。analyze_gray 为 block=一帧 的并行
    归约实现；sim_pair 计算两帧在二值化/原始域的差异标量（merge_similar
    判定用），同样只回传标量。
    """
    _KERNEL = r'''
extern "C" __global__ void analyze_gray(
    const unsigned char* __restrict__ raw,
    const unsigned char* __restrict__ prev,
    double* __restrict__ summary,
    int B, int H, int W, float th) {
    // block = 一帧：256 线程分片扫描 + shared 归约（旧版每帧单线程串行
    // 扫 H*W*9，是 GPU 分段路径的主要瓶颈）。cluster 为整数计数，逐位一致；
    // sharp 用 int64 精确累加像素值与平方和（加法顺序无关），方差实数精确
    // ——与宿主"严格大于保先者"的代表帧选择语义对齐；浮点归约顺序差异
    // 曾导致近平局选帧漂移（批量时间戳 ±1s 偏移）。
    __shared__ unsigned long long s_sum[256];
    __shared__ unsigned long long s_sum2[256];
    __shared__ int s_max[256];
    int b = blockIdx.x;
    if (b >= B) return;
    const unsigned char* cur = raw + (size_t)b * H * W;
    const unsigned char* pre = prev + (size_t)b * H * W;
    int n = H * W;
    int t = threadIdx.x;
    unsigned long long sum = 0, sum2 = 0;
    int maxc = 0;
    for (int p = t; p < n; p += 256) {
        int y = p / W;
        int x = p - y * W;
        unsigned long long v = cur[p];
        sum += v; sum2 += v * v;
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
    s_sum[t] = sum; s_sum2[t] = sum2; s_max[t] = maxc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) {
            s_sum[t] += s_sum[t + s];
            s_sum2[t] += s_sum2[t + s];
            if (s_max[t + s] > s_max[t]) s_max[t] = s_max[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        double mean = s_sum[0] / (double)n;
        double var = s_sum2[0] / (double)n - mean * mean;
        summary[b * 2] = var > 0.0 ? sqrt(var) : 0.0;
        summary[b * 2 + 1] = (double)s_max[0];
    }
}

extern "C" __global__ void sim_pair(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ b,
    double* __restrict__ out,
    int n, int th, int use_bin) {
    // merge_similar 判定的差异标量：out[0]=MAD 累加和，out[1]=显著变化数。
    // use_bin=1 按二值化域（|0-255|差 ⇔ 阈值穿越），否则按原始灰度域。
    // 与宿主 _segments_similar（binary text_sep / raw）语义一一对应。
    __shared__ double s_mad[256];
    __shared__ unsigned long long s_chg[256];
    int t = threadIdx.x;
    double mad = 0.0;
    unsigned long long chg = 0;
    for (int p = t; p < n; p += 256) {
        if (use_bin) {
            int d = ((a[p] > th) != (b[p] > th)) ? 1 : 0;
            mad += d;
            chg += d;
        } else {
            int d = abs((int)a[p] - (int)b[p]);
            mad += d;
            chg += d > 10 ? 1 : 0;
        }
    }
    s_mad[t] = mad; s_chg[t] = chg;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) {
            s_mad[t] += s_mad[t + s];
            s_chg[t] += s_chg[t + s];
        }
        __syncthreads();
    }
    if (t == 0) { out[0] = s_mad[0]; out[1] = (double)s_chg[0]; }
}

extern "C" __global__ void hist_gray_perframe(
    const unsigned char* __restrict__ raw,
    int* __restrict__ hists,   // (B, 256)
    int HxW) {
    // block = 一帧；线程内串行累加私有计数后一次性 atomicAdd，减少竞争
    __shared__ int priv[256];
    int b = blockIdx.x;
    if (threadIdx.x < 256) priv[threadIdx.x] = 0;
    __syncthreads();
    const unsigned char* cur = raw + (size_t)b * HxW;
    for (int i = threadIdx.x; i < HxW; i += blockDim.x)
        atomicAdd(&priv[cur[i]], 1);
    __syncthreads();
    if (threadIdx.x < 256 && priv[threadIdx.x] > 0)
        atomicAdd(&hists[b * 256 + threadIdx.x], priv[threadIdx.x]);
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
        self._mod = self._prog.compile(
            "cubin",
            name_expressions=("analyze_gray",
                              "hist_gray_perframe", "sim_pair"))
        self._kernel = self._mod.get_kernel("analyze_gray")
        self._kernel_hist_pf = self._mod.get_kernel("hist_gray_perframe")
        self._kernel_sim = self._mod.get_kernel("sim_pair")
        from cuda.bindings import runtime as cudart
        _err, self._stream = cudart.cudaStreamCreate()
        self._summary_size = 0
        self._summary_dev = None
        self._prev_size = 0
        self._prev_dev = None
        self._histpf_size = 0
        self._histpf_dev = None

    def _ensure_prev(self, nbytes: int) -> int:
        from cuda.bindings import runtime as cudart
        if self._prev_size < nbytes:
            if self._prev_dev is not None:
                cudart.cudaFree(self._prev_dev)
            _err, self._prev_dev = cudart.cudaMalloc(nbytes)
            self._prev_size = nbytes
        return self._prev_dev

    def histograms_perframe(self, raw_ptr: int, B: int,
                            H: int, W: int) -> "np.ndarray":
        """逐帧直方图：(B,256) int32 回传 host（B×1KB），供宿主复刻
        '逐帧 Otsu 取中位数' 校准语义——与单流水线阈值行为完全一致，
        且校准帧不落 RAM（仅 50KB 标量表）。"""
        import numpy as np
        from cuda.bindings import runtime as cudart
        from cuda.core import Buffer, LaunchConfig, launch
        nbytes = B * 256 * 4
        if self._histpf_size < nbytes:
            if self._histpf_dev is not None:
                cudart.cudaFree(self._histpf_dev)
            _err, self._histpf_dev = cudart.cudaMalloc(nbytes)
            self._histpf_size = nbytes
        cudart.cudaMemsetAsync(self._histpf_dev, 0, nbytes, self._stream)
        buf = Buffer.from_handle(self._histpf_dev, nbytes)
        launch(self._stream, LaunchConfig(grid=B, block=256),
               self._kernel_hist_pf,
               Buffer.from_handle(raw_ptr, B * H * W), buf, np.int32(H * W))
        cudart.cudaStreamSynchronize(self._stream)
        hists = np.empty((B, 256), dtype=np.int32)
        cudart.cudaMemcpy(
            hists.ctypes.data, self._histpf_dev, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return hists

    def analyze_batch(self, raw_ptr: int, prev_ptr: int, B: int,
                      H: int, W: int, th: float) -> "np.ndarray":
        """一次 kernel 分析 B 帧；prev_ptr 必须是已准备好的 B 帧前帧缓冲。"""
        import numpy as np
        from cuda.bindings import runtime as cudart
        from cuda.core import Buffer, LaunchConfig, launch
        nbytes = B * 2 * 8
        if self._summary_size < nbytes:
            if self._summary_dev is not None:
                cudart.cudaFree(self._summary_dev)
            _err, self._summary_dev = cudart.cudaMalloc(nbytes)
            self._summary_size = nbytes
        buf = Buffer.from_handle(self._summary_dev, nbytes)
        launch(self._stream, LaunchConfig(grid=B, block=256), self._kernel,
               Buffer.from_handle(raw_ptr, B * H * W),
               Buffer.from_handle(prev_ptr, B * H * W),
               buf, np.int32(B), np.int32(H), np.int32(W), np.float32(th))
        cudart.cudaStreamSynchronize(self._stream)
        out = np.empty((B, 2), dtype=np.float64)
        cudart.cudaMemcpy(
            out.ctypes.data, self._summary_dev, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return out

    def compare_pair(self, a_ptr: int, b_ptr: int, H: int, W: int,
                     th: int, use_bin: bool) -> "tuple[float, int]":
        """两帧差异标量（merge_similar 判定）：(mad_sum, changed_count)。

        use_bin=True 按二值化域：mad_sum = 穿越阈值像素数（宿主换算
        MAD = 255*mad_sum/n），changed = 同一计数（|0-255|>10 恒真）。
        use_bin=False 按原始灰度域：mad_sum = |a-b| 整数和，changed =
        count(|a-b|>10)。与宿主 _segments_similar 的两个条件一一对应。
        """
        import numpy as np
        from cuda.bindings import runtime as cudart
        from cuda.core import Buffer, LaunchConfig, launch
        if getattr(self, "_sim_dev", None) is None:
            _err, self._sim_dev = cudart.cudaMalloc(2 * 8)
        n = H * W
        out_buf = Buffer.from_handle(self._sim_dev, 2 * 8)
        launch(self._stream, LaunchConfig(grid=1, block=256),
               self._kernel_sim,
               Buffer.from_handle(a_ptr, n),
               Buffer.from_handle(b_ptr, n),
               out_buf, np.int32(n), np.int32(th),
               np.int32(1 if use_bin else 0))
        cudart.cudaStreamSynchronize(self._stream)
        out = np.empty(2, dtype=np.float64)
        cudart.cudaMemcpy(out.ctypes.data, self._sim_dev, 2 * 8,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return float(out[0]), int(out[1])


