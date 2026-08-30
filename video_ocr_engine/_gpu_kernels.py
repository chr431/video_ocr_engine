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

# NVRTC 编译缓存（DESIGN-REVIEW B5）：编译产物（Program/Module）进程级共享
# ——无状态、线程安全（kernel launch 只读）；Buffer/stream 等可变状态仍归
# 各实例。没有这层缓存时，每次 extract 新建 GpuFrameAnalyzer/GpuPreprocessor
# 都会重付一次 NVRTC 编译（长进程批量的固定开销）。
_KERNEL_MODULE_CACHE: dict = {}


def _compile_module(src: str, name_expressions: tuple):
    """按 (arch, src) 缓存编译 cubin 模块；返回共享 Module。"""
    from cuda.core import Device, Program, ProgramOptions
    dev = Device()
    dev.set_current()
    key = (getattr(dev, "arch", "?"), src)
    mod = _KERNEL_MODULE_CACHE.get(key)
    if mod is None:
        prog = Program(src, code_type="c++",
                       options=ProgramOptions(std="c++11",
                                              arch=f"sm_{dev.arch}"))
        mod = prog.compile("cubin",
                           name_expressions=list(name_expressions))
        _KERNEL_MODULE_CACHE[key] = mod
    return mod

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
    const long long* __restrict__ bases,
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
        // bases[b] = 图 b 在拼接缓冲中的元素基址（宽度不齐时不能假设等宽）
        long long src = bases[b] + (long long)y * w * C + (long long)x * C + c;
        float v = raw[src];
        out[i] = (v / 255.0f - 0.5f) / 0.5f;
    } else {
        out[i] = 0.0f;
    }
}

extern "C" __global__ void prep_gray_raw(
    const unsigned char* __restrict__ raw,
    const int* __restrict__ xoffs,
    const int* __restrict__ crop_ws,
    const int* __restrict__ content_ws,
    float* __restrict__ out,
    int B, int src_h, int src_w,
    int dst_h, int dst_w, float gamma) {
    // 逐项宽度裁切（P0-4 GPU 直通）：xoffs[b]/crop_ws[b] = 该帧参与 OCR 的
    // 源列区间 [xoff, xoff+crop_w)，content_ws[b] = 裁后按 dst_h/src_h 缩放
    // 的目标内容宽（host 侧按宿主 _preprocess_standard 同一 int 截断式计算）。
    // 未裁切项 xoff=0、crop_w=src_w → 与旧全宽内核逐位一致。
    int total = B * 3 * dst_h * dst_w;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int b = i / (3 * dst_h * dst_w);
    int rem = i % (3 * dst_h * dst_w);
    int c = rem / (dst_h * dst_w);
    int rem2 = rem % (dst_h * dst_w);
    int y = rem2 / dst_w;
    int x = rem2 % dst_w;
    int xoff = xoffs[b];
    int cw = crop_ws[b];
    int content_w = content_ws[b];
    if (x >= content_w) { out[i] = 0.0f; return; }
    double sx = (double)xoff + (x + 0.5) * ((double)cw / (double)content_w) - 0.5;
    double sy = (y + 0.5) * ((double)src_h / (double)dst_h) - 0.5;
    sx = fmax((double)xoff, fmin(sx, (double)(xoff + cw - 1)));
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
        from cuda.core import Buffer, Device, LaunchConfig, launch
        self._dev = Device()
        self._dev.set_current()
        self._mod = _compile_module(self._KERNEL, ("prep", "prep_gray_raw"))
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
        self._bases_size = 0
        self._bases_dev = None
        self._bases_buf = None
        self._i32_size = 0
        self._i32_dev = None

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

    def _ensure_bases(self, nbytes: int) -> int:
        from cuda.bindings import runtime as cudart
        if self._bases_size < nbytes:
            if self._bases_dev is not None:
                cudart.cudaFree(self._bases_dev)
            _err, self._bases_dev = cudart.cudaMalloc(nbytes)
            self._bases_buf = self._buffer_cls.from_handle(self._bases_dev, nbytes)
            self._bases_size = nbytes
        return self._bases_dev

    def _ensure_out(self, nbytes: int) -> int:
        from cuda.bindings import runtime as cudart
        if self._out_size < nbytes:
            if self._out_dev is not None:
                cudart.cudaFree(self._out_dev)
            _err, self._out_dev = cudart.cudaMalloc(nbytes)
            self._out_buf = self._buffer_cls.from_handle(self._out_dev, nbytes)
            self._out_size = nbytes
        return self._out_dev

    def _ensure_i32x3(self, nbytes: int):
        """一块 B*3*4 的 int32 设备缓冲（prep_gray_raw 的 xoffs/crop_ws/
        content_ws 三数组连续排布，单次 H2D 上载）。返回 (dev, buf)。"""
        from cuda.bindings import runtime as cudart
        if self._i32_size < nbytes:
            if self._i32_dev is not None:
                cudart.cudaFree(self._i32_dev)
            _err, self._i32_dev = cudart.cudaMalloc(nbytes)
            self._i32_size = nbytes
        return self._i32_dev, self._buffer_cls.from_handle(self._i32_dev,
                                                           nbytes)

    def release(self) -> None:
        """释放全部设备缓冲（DESIGN-REVIEW C5）。重复调用安全；再次使用
        时各 _ensure_* 按需重建。"""
        from cuda.bindings import runtime as cudart
        for attr in ("_raw_dev", "_out_dev", "_width_dev", "_bases_dev",
                     "_i32_dev"):
            ptr = getattr(self, attr, None)
            if ptr:
                try:
                    cudart.cudaFree(ptr)
                except Exception:
                    pass
            setattr(self, attr, None)
        self._raw_size = self._out_size = self._width_size = 0
        self._bases_size = self._i32_size = 0

    def process_gray_raw(self, infos: list, out_width: int,
                         force_aspect: float = 0.0):
        """处理 GPU 灰度帧：D2D 聚批 → GPU resize+gamma+normalize+pad。

        infos: [(dev_ptr, src_h, src_w, owner), ...]，frame 已位于显存；
        也接受 6 元组 (dev_ptr, src_h, src_w, owner, x_off, crop_w) ——
        该帧只把源列区间 [x_off, x_off+crop_w) 参与缩放（P0-4 宽度自适应
        裁切的 GPU 直通；区间由 GPU col_ink + 宿主余量规则给出）。
        force_aspect > 0：强制 OCR 输入宽 = OCR_TARGET_H × force_aspect
        （内容整体拉伸到该宽度，与宿主 _preprocess_standard 的 force_aspect
        语义一致；边界吸附用 round），此时忽略逐项裁切区间（与宿主
        _crop_to_content 的 force_aspect 跳过语义一致）。
        返回 (device_ptr, output_shape)。调用方必须保持 owner/本对象存活。
        """
        import numpy as np
        from cuda.bindings import runtime as cudart
        B = len(infos)
        src_h = int(infos[0][1])
        src_w = int(infos[0][2])
        dst_h = int(config.OCR_TARGET_H)
        xoffs = np.zeros(B, dtype=np.int32)
        crop_ws = np.full(B, src_w, dtype=np.int32)
        content_ws = np.empty(B, dtype=np.int32)
        if force_aspect and force_aspect > 0:
            # 顺序 ⑦「先定比例、后裁」在 GPU 上的等价写法。
            #
            # 朴素做法是把裁后区间**拉伸**到 forced_w（= 顺序 ⑥），这会改变
            # 内容宽高比 → 畸变。实测（生产口径 代表帧+tol=1 误读）：
            #   ① 不裁 7 / ⑥ 先裁再定比例 9 / **⑦ 先定比例再裁 0**（test5）
            #   ① 17 / ⑥ 5 / **⑦ 0**（test6）
            #
            # ⑦ 的等价实现：整幅的缩放比例 s = forced_w / src_w，裁后区间
            # [xoff, xoff+cwid) 按 **同一个 s** 缩放 → content_w = cwid*s。
            # 内核里 sx 用 cw/content_w 反算源坐标，等效缩放正是 src_w/forced_w，
            # 与"整幅先缩到 forced_w 再裁"逐点一致。
            forced_w = max(1, int(round(dst_h * force_aspect)))
            for i, t in enumerate(infos):
                if len(t) >= 6:
                    xo, cwid = int(t[4]), int(t[5])
                else:
                    xo, cwid = 0, src_w
                xoffs[i] = xo
                crop_ws[i] = cwid
                content_ws[i] = max(1, int(cwid * forced_w / src_w))
        else:
            for i, t in enumerate(infos):
                if len(t) >= 6:
                    xo, cwid = int(t[4]), int(t[5])
                else:
                    xo, cwid = 0, src_w
                xoffs[i] = xo
                crop_ws[i] = cwid
                # 与宿主 _preprocess_standard 的 new_w 同式（int 截断），
                # 未裁切项与旧全宽内核的 content_w 计算逐位一致。
                content_ws[i] = max(1, int(cwid * dst_h / src_h))
        dst_w = int(out_width)
        if int(content_ws.max()) > dst_w:
            dst_w = int(content_ws.max())
        raw_nbytes = B * src_h * src_w
        raw_dev = self._ensure_raw(raw_nbytes)
        for i, t in enumerate(infos):
            cudart.cudaMemcpyAsync(
                raw_dev + i * src_h * src_w,
                int(t[0]), src_h * src_w,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                self._stream)
        out_nbytes = B * 3 * dst_h * dst_w * 4
        out_dev = self._ensure_out(out_nbytes)
        gamma = config.env_float(config.OCR_GAMMA_ENV, float(config.OCR_GAMMA))
        shape = (B, 3, dst_h, dst_w)
        total = int(np.prod(shape))
        block = 256
        grid = (total + block - 1) // block
        i32 = np.concatenate([xoffs, crop_ws, content_ws])
        i32_dev, i32_buf = self._ensure_i32x3(i32.nbytes)
        cudart.cudaMemcpyAsync(
            i32_dev, i32.ctypes.data, i32.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)
        bnb = B * 4
        self._launch(
            self._stream,
            self._launch_cls(grid=grid, block=block),
            self._kernel_raw,
            self._raw_buf,
            self._buffer_cls.from_handle(i32_dev, bnb),
            self._buffer_cls.from_handle(i32_dev + bnb, bnb),
            self._buffer_cls.from_handle(i32_dev + 2 * bnb, bnb),
            self._out_buf,
            np.int32(B), np.int32(src_h), np.int32(src_w),
            np.int32(dst_h), np.int32(dst_w),
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
        # 各图在拼接缓冲中的元素基址（批内宽度可能不齐；kernel 按
        # widths[b] 索引，没有基址表会对非首图错位/越界）。
        bases = np.zeros(B, dtype=np.int64)
        if B > 1:
            bases[1:] = np.cumsum(widths.astype(np.int64) * H * C)[:-1]

        raw_dev = self._ensure_raw(raw.nbytes)
        width_dev = self._ensure_width(widths.nbytes)
        bases_dev = self._ensure_bases(bases.nbytes)
        out_nbytes = B * C * H * out_width * 4
        out_dev = self._ensure_out(out_nbytes)

        cudart.cudaMemcpyAsync(
            raw_dev, raw.ctypes.data, raw.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)
        cudart.cudaMemcpyAsync(
            width_dev, widths.ctypes.data, widths.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)
        cudart.cudaMemcpyAsync(
            bases_dev, bases.ctypes.data, bases.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)

        shape = (B, C, H, out_width)
        total = int(np.prod(shape))
        block = 256
        grid = (total + block - 1) // block
        self._launch(
            self._stream,
            self._launch_cls(grid=grid, block=block),
            self._kernel,
            self._raw_buf, self._width_buf, self._bases_buf, self._out_buf,
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
        from cuda.core import Buffer, Device, LaunchConfig, launch
        self._dev = Device()
        self._dev.set_current()
        self._mod = _compile_module(self._KERNEL, ("argmax_last",))
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

    def release(self) -> None:
        """释放设备缓冲（DESIGN-REVIEW C5）。重复调用安全。"""
        from cuda.bindings import runtime as cudart
        for attr in ("_idx_dev", "_prob_dev"):
            ptr = getattr(self, attr, None)
            if ptr:
                try:
                    cudart.cudaFree(ptr)
                except Exception:
                    pass
            setattr(self, attr, None)
        self._idx_size = 0

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
    归约实现。
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

extern "C" __global__ void col_ink(
    const unsigned char* __restrict__ raw,
    int* __restrict__ range,   // [首列, 末列]，kernel 直接写出（无需初始化）
    int H, int W, int th) {
    // rep 帧的「有墨迹列范围」（P0-4 GPU 直通裁切的判据输入）：每列
    // g > th 的像素数 ≥ 2 才算有效列（抗孤立噪点），与宿主
    // _crop_to_content 的列判据一致。单 block 256 线程跨列分片 +
    // shared 归约；无合格列时 range = (INT_MAX, -1)（host 判 first>last
    // → None），故无需预先初始化或原子操作。
    __shared__ int s_first[256];
    __shared__ int s_last[256];
    int t = threadIdx.x;
    int first = 0x7fffffff, last = -1;
    for (int x = t; x < W; x += 256) {
        const unsigned char* col = raw + x;
        int cnt = 0;
        for (int y = 0; y < H; ++y)
            cnt += (col[(size_t)y * W] > th) ? 1 : 0;
        if (cnt >= 2) {
            if (x < first) first = x;
            last = x;   // x 递增扫描 → last 即最新合格列
        }
    }
    s_first[t] = first;
    s_last[t] = last;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) {
            if (s_first[t + s] < s_first[t]) s_first[t] = s_first[t + s];
            if (s_last[t + s] > s_last[t]) s_last[t] = s_last[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        range[0] = s_first[0];
        range[1] = s_last[0];
    }
}

extern "C" __global__ void sim_pair(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ b,
    double* __restrict__ out,
    int n, int th, int use_bin) {
    // merge_similar 判定的差异标量：out[0]=MAD 累加和，out[1]=显著变化数。
    // use_bin=1 按二值化域（阈值穿越 ⇔ |0-255| 差），否则按原始灰度域。
    // 与宿主 _segments_similar（binary text_sep / raw）语义一一对应：
    // 整数精确累加（与 numpy 的 float32 均值仅差末位舍入，见 _similar_device）。
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

extern "C" __global__ void luma_nv12(
    const unsigned char* __restrict__ src,   // packed NV12: (B, H+ceil(H/2), W)
    unsigned char* __restrict__ out,         // (B, H, W) 灰度 Y
    int B, int H, int W, int limited) {
    // 与宿主 _nv12_luma_full 逐位一致：limited/tv 时
    // (y-16)*(255/219) → floor(x+0.5) → clip 0..255；full/pc 原样。
    long long total = (long long)B * H * W;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((long long)i >= total) return;
    int b = (int)((long long)i / ((long long)H * W));
    int rem = (int)((long long)i % ((long long)H * W));
    long long rows = (long long)H + (H + 1) / 2;
    const unsigned char* base = src + ((long long)b * rows + rem / W) * W
        + (rem % W);
    int v = *base;
    if (limited) {
        float val = (float)(v - 16) * (255.0f / 219.0f);
        float f = floorf(val + 0.5f);
        if (f < 0.0f) f = 0.0f;
        else if (f > 255.0f) f = 255.0f;
        out[i] = (unsigned char)f;
    } else {
        out[i] = (unsigned char)v;
    }
}
'''

    def __init__(self) -> None:
        from cuda.core import Device
        self._dev = Device()
        self._dev.set_current()
        self._mod = _compile_module(
            self._KERNEL,
            ("analyze_gray", "hist_gray_perframe", "luma_nv12",
             "sim_pair", "col_ink"))
        self._kernel = self._mod.get_kernel("analyze_gray")
        self._kernel_hist_pf = self._mod.get_kernel("hist_gray_perframe")
        self._kernel_luma = self._mod.get_kernel("luma_nv12")
        self._kernel_sim = self._mod.get_kernel("sim_pair")
        self._kernel_ink = self._mod.get_kernel("col_ink")
        from cuda.bindings import runtime as cudart
        _err, self._stream = cudart.cudaStreamCreate()
        self._summary_size = 0
        self._summary_dev = None
        self._prev_size = 0
        self._prev_dev = None
        self._histpf_size = 0
        self._histpf_dev = None
        self._luma_size = 0
        self._luma_dev = None
        self._range_dev = None

    def release(self) -> None:
        """释放全部设备缓冲（DESIGN-REVIEW C5）。重复调用安全；再次使用
        时各 _ensure_* / content_range / compare_pair 按需重建。"""
        from cuda.bindings import runtime as cudart
        for attr in ("_prev_dev", "_histpf_dev", "_summary_dev",
                     "_luma_dev", "_range_dev", "_sim_dev"):
            ptr = getattr(self, attr, None)
            if ptr:
                try:
                    cudart.cudaFree(ptr)
                except Exception:
                    pass
            setattr(self, attr, None)
        self._prev_size = self._histpf_size = self._summary_size = 0
        self._luma_size = 0

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
        hists = np.empty((B, 256), dtype=np.int32)
        cudart.cudaMemcpyAsync(
            hists.ctypes.data, self._histpf_dev, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream)
        cudart.cudaStreamSynchronize(self._stream)
        return hists

    def content_range(self, raw_ptr: int, H: int, W: int,
                      th: int) -> "tuple[int, int] | None":
        """rep 帧的「有墨迹列范围」(first, last)：每列 g>th 计数 ≥2 的
        首/末列，判据与宿主 _crop_to_content 一致（P0-4 GPU 直通裁切）。

        DtoH 仅 8 字节；无合格列（全空帧）返回 None。调用方必须保证
        raw_ptr 上的帧数据在本次同步前有效（owner 存活）。
        """
        import numpy as np
        from cuda.bindings import runtime as cudart
        from cuda.core import Buffer, LaunchConfig, launch
        if self._range_dev is None:
            _err, self._range_dev = cudart.cudaMalloc(2 * 4)
        launch(self._stream, LaunchConfig(grid=1, block=256),
               self._kernel_ink,
               Buffer.from_handle(int(raw_ptr), H * W),
               Buffer.from_handle(self._range_dev, 2 * 4),
               np.int32(H), np.int32(W), np.int32(int(th)))
        out = np.empty(2, dtype=np.int32)
        cudart.cudaMemcpyAsync(
            out.ctypes.data, self._range_dev, 2 * 4,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream)
        cudart.cudaStreamSynchronize(self._stream)
        if int(out[0]) > int(out[1]):
            return None
        return int(out[0]), int(out[1])

    def luma_into(self, src_ptr: int, dst_ptr: int, H: int, W: int,
                  limited: bool, B: int = 1) -> None:
        """packed NV12 → 灰度 Y，写入指定 dst（B 帧连续）。

        src: (B, H+ceil(H/2), W) packed；dst: (B, H, W)。与宿主
        _nv12_luma_full 逐位一致。调用方负责 dst 的生命周期/对齐。
        """
        import numpy as np
        from cuda.bindings import runtime as cudart
        from cuda.core import Buffer, LaunchConfig, launch
        nbytes = B * H * W
        block = 256
        grid = (nbytes + block - 1) // block
        launch(self._stream, LaunchConfig(grid=grid, block=block),
               self._kernel_luma,
               Buffer.from_handle(src_ptr, B * (H + (H + 1) // 2) * W),
               Buffer.from_handle(dst_ptr, nbytes),
               np.int32(B), np.int32(H), np.int32(W),
               np.int32(1 if limited else 0))

    def extract_luma(self, src_ptr: int, B: int, H: int, W: int,
                     limited: bool) -> int:
        """packed NV12 → 灰度 Y 平面（D2D），与宿主 _nv12_luma_full 逐位一致。

        src: (B, H+ceil(H/2), W) packed NV12 device 指针；返回 Y 缓冲
        (B, H, W) device 指针。缓冲在本对象内自适应复用——同一批的后续
        kernel 完成前不被覆盖（同一批内安全；跨批重新调用会覆盖内容，
        前批引用须在覆盖前消费完）。
        """
        from cuda.bindings import runtime as cudart
        nbytes = B * H * W
        if self._luma_size < nbytes:
            if self._luma_dev is not None:
                cudart.cudaFree(self._luma_dev)
            _err, self._luma_dev = cudart.cudaMalloc(nbytes)
            self._luma_size = nbytes
        self.luma_into(src_ptr, self._luma_dev, H, W, limited, B)
        return self._luma_dev

    def compare_pair(self, a_ptr: int, b_ptr: int, H: int, W: int,
                     th: int, use_bin: bool) -> "tuple[int, int]":
        """两帧差异标量（merge_similar 判定）：(mad_sum, changed_count)。

        use_bin=True 按二值化域：mad_sum = 阈值穿越像素数（宿主换算
        MAD = 255*mad_sum/n），changed = 同一计数（|0-255|>10 恒真）。
        use_bin=False 按原始灰度域：mad_sum = |a-b| 整数和，changed =
        count(|a-b|>10)。整数精确累加（double 归约，值域 < 2^53），
        与宿主 _segments_similar 的两个条件一一对应。
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
        out = np.empty(2, dtype=np.float64)
        cudart.cudaMemcpyAsync(
            out.ctypes.data, self._sim_dev, 2 * 8,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream)
        cudart.cudaStreamSynchronize(self._stream)
        return int(out[0]), int(out[1])

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
        out = np.empty((B, 2), dtype=np.float64)
        cudart.cudaMemcpyAsync(
            out.ctypes.data, self._summary_dev, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream)
        cudart.cudaStreamSynchronize(self._stream)
        return out


