"""原生 OCR 识别引擎 — 绕过 rapidocr，直接使用 ONNX Runtime / TensorRT。

与 rapidocr 的 TextRecognizer 输出逐字节对齐（预处理/CTC 后处理复刻）：
- 预处理: resize 到 48 高 + (x/255 - 0.5) / 0.5 归一化 + pad 到 batch 最大宽
- 推理:   ONNX (onnxruntime, 动态 batch) / TensorRT（引擎实现见 ocr_trt.py）
- 后处理: argmax(axis=2) + max(axis=2) + CTC 去重 + blank(0) 过滤 + 字符映射
- 字符表: assets/ocr_models/ppocrv6_dict.txt（18708 字符 + 末尾空格 + 开头 blank = 18710）

输出对象携带 .txts / .scores（文本与置信度，供上层应用做数值解析/纠错）。
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np

import engine_config as config
from ocr_trt import TrtEngine

# ONNX CPU 性能优化：避免 OpenMP 线程忙等，降低 CPU 空转。
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

log = logging.getLogger(__name__)


def _models_dir() -> Path:
    """模型资产目录（源码 / wheel 安装 / frozen 多路径兼容）。

    查找顺序：
      1. frozen: PyInstaller _MEIPASS/ocr_models
      2. 源码树: <repo>/assets/ocr_models（也兼容未来包内 assets）
      3. site-packages/assets/ocr_models（若改为 package-data 布局）
      4. sys.prefix/assets/ocr_models（当前 pyproject data-files 安装位置）
    返回第一个包含模型文件的目录；都找不到时返回源码树候选，
    让后续打开文件时给出自然的 FileNotFoundError。
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", "")) / "ocr_models"
    here = Path(__file__).resolve().parent
    candidates = [
        here / "assets" / "ocr_models",          # 源码 / 包内资源
        here.parent / "assets" / "ocr_models",   # 模块在 video_ocr_engine/ 包内时
        Path(sys.prefix) / "assets" / "ocr_models",  # data-files 安装位置
    ]
    marker = "PP-OCRv6_rec_small.onnx"
    for p in candidates:
        if (p / marker).is_file():
            return p
    return candidates[0]


def cpu_physical_cores() -> int:
    """物理核数（psutil 缺失时用逻辑核/2 估算，最小 2）。"""
    try:
        import psutil  # type: ignore[import-not-found]
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        physical = None
    if not physical:
        physical = max(2, (os.cpu_count() or 8) // 2)
    return max(2, int(physical))


def auto_ocr_thread_count() -> int:
    """OCR 推理线程预算：全部物理核。

    实测（16C32T，decord v0.7.9 + onnxruntime，test5 7223 帧）：
    解码走 NVDEC 卸载 / FFmpeg 帧线程 + filter auto（只占 SMT 份额），
    OCR 全物理核满负荷正收益；超物理核（超线程）不再提升。CPU 与 GPU
    解码后端统一使用同一预算。
    """
    return cpu_physical_cores()


class RecOut:
    """兼容 extract_speed_value 的输出对象（txts/scores）。"""

    __slots__ = ("txts", "scores")

    def __init__(self, txt: str, score: float) -> None:
        self.txts = (txt,)
        self.scores = [float(score)]


class OcrEngine:
    """PP-OCRv6 rec 原生引擎（ONNX / TensorRT 双后端）。

    Args:
        variant: "v6_small"（唯一模型，v2.13 起）
        engine_type: "onnxruntime" | "tensorrt"
        progress_cb: 构建引擎等耗时阶段的进度消息回调 (str)。
        fill_width: OCR 输入 pad 宽度下限（px）。0 = 用 config 默认。速度窄图
            对宽 pad 更准，用户可调（GUI 160-320）。
        num_threads: ONNX 推理线程数。None = OCR_THREADS env →
            默认物理核/2（仅直接构造 OcrEngine 时）；生产管线会显式传入
            auto_ocr_thread_count()（全物理核），因此 CPU/GPU 解码后端统一。
    """

    def __init__(self, variant: str = "v6_small",
                 engine_type: str = "onnxruntime",
                 progress_cb: "callable | None" = None,
                 fill_width: int = 0,
                 num_threads: int | None = None) -> None:
        self._variant = variant
        self._progress_cb = progress_cb
        self._fill_width = fill_width
        self._num_threads = num_threads
        self._lock = threading.Lock()
        size = variant.replace("v6_", "")
        models = _models_dir()

        # ── 字符表（与 rapidocr CTCLabelDecode.get_character 一致）──
        dict_name = "ppocrv6_dict.txt"
        with open(models / dict_name, "rb") as f:
            chars = [ln.decode("utf-8").strip("\n").strip("\r\n")
                     for ln in f.readlines()]
        chars.append(" ")          # 末尾插入空格
        chars.insert(0, "blank")   # 开头插入 blank（CTC 空白，索引 0）
        self._chars = chars

        # ── 模型 ──
        self._trt: TrtEngine | None = None
        self._gpu_pre = None  # TRT GPU 预处理（懒加载）
        # 显存全驻留 CTC：TRT 输出在 GPU 完成 vocab 维 argmax/max，输出
        # 不落 RAM。仅在 call_gpu_raw（GPU 分段管线）路径生效，因此默认
        # 开启是安全的；GPU_CTC=0 显式关闭。
        self._gpu_ctc_mode = (engine_type == "tensorrt" and
                              config.env_bool(config.GPU_CTC_ENV, default=True))
        if engine_type == "tensorrt":
            try:
                self._trt = TrtEngine(models, size, progress_cb=self._progress_cb)
            except Exception as e:
                log.warning("TensorRT 引擎不可用 (%s)，回退 ONNX 后端。", e)
                self._init_onnx(models, size)
        else:
            self._init_onnx(models, size)

    @property
    def backend_name(self) -> str:
        """实际推理后端：'tensorrt' 或 'onnxruntime'（CSV 头/日志使用）。"""
        return "tensorrt" if self._trt is not None else "onnxruntime"

    # ═══════════════ ONNX 后端 ═══════════════

    def _init_onnx(self, models: Path, size: str) -> None:
        import onnxruntime as ort
        # 直接构造 OcrEngine（未传 num_threads）时默认物理核/2：避免 ONNX
        # 推理占满全部逻辑核并与解码器抢核。生产管线 SegmentPipeline.
        # _ocr_num_threads 显式传 auto_ocr_thread_count()（全物理核），
        # 所以 CPU/GPU 解码场景统一走显式线程预算，不走此默认。
        so = ort.SessionOptions()
        physical = cpu_physical_cores()
        n = max(2, physical // 2)
        # 优先级：显式 num_threads > env 钩子 > 默认物理核/2
        if self._num_threads:
            n = max(1, int(self._num_threads))
        else:
            _env_t = os.environ.get(config.OCR_THREADS_ENV)
            if _env_t:
                n = max(1, int(_env_t))
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 2
        model_path = models / f"PP-OCRv6_rec_{size}.onnx"
        self._session = ort.InferenceSession(
            str(model_path), sess_options=so,
            providers=["CPUExecutionProvider"])

    # ═══════════════ 预处理（复刻 rapidocr resize_norm_img）═══════════════

    @staticmethod
    def _resize_norm(img: np.ndarray, max_wh_ratio: float,
                     height: int = config.OCR_TARGET_H) -> np.ndarray:
        """resize 到固定高 + (x/255-0.5)/0.5 归一化 + pad 到 batch 最大宽。

        输入已是目标高度 float32（pipeline._preprocess_standard 输出）时跳过
        _np_resize —— 其等尺寸路径的 astype 拷贝是无谓开销。数值路径不变
        （省略的是同一 float32 数据的整块拷贝），逐位一致。
        """
        from video_utils import _np_resize
        img_width = int(height * max_wh_ratio)
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(height * ratio))
        if resized_w == w and h == height and img.dtype == np.float32:
            resized = img
        else:
            resized = _np_resize(img, resized_w, height)
        resized = resized.transpose((2, 0, 1)) / 255
        resized = (resized - 0.5) / 0.5
        pad = np.empty((3, height, img_width), dtype=np.float32)
        pad[:, :, :resized_w] = resized
        if resized_w < img_width:
            # np.empty 未初始化：尾部必须显式置 0（原 np.zeros 语义）
            pad[:, :, resized_w:] = 0.0
        return pad

    # ═══════════════ 推理 ═══════════════

    def _infer(self, batch_np: np.ndarray) -> np.ndarray:
        # 整段持锁：TRT 路径的 ctx/buffers 是实例共享可变状态，预热线程与
        # 主 OCR 线程必须串行（GPU 单上下文本就不能并行，串行不损失吞吐）。
        with self._lock:
            return self._infer_locked(batch_np)

    def _infer_locked(self, batch_np: np.ndarray) -> np.ndarray:
        if self._trt is not None:
            first_batch = batch_np[:min(len(batch_np), self._trt.max_batch)]
            out_shape = self._trt._prepare_shape(first_batch.shape)
            # 预分配整批输出并让每个子批直接写进对应切片，免去逐批 host_out
            # 分配和最后的 concatenate 拷贝。
            preds = np.empty(
                (len(batch_np),) + tuple(out_shape[1:]), dtype=np.float32)
            # 所有子批全部异步 enqueue 到同一 CUDA stream，最后只同步一次；
            # 消除每个子批一次 host-GPU 往返同步，让 GPU 连续执行。
            for i in range(0, len(batch_np), self._trt.max_batch):
                n = min(self._trt.max_batch, len(batch_np) - i)
                self._trt.execute_async(
                    batch_np[i:i + n], out_host=preds[i:i + n])
            self._trt.synchronize()
            return preds
        # ONNX 动态 batch 无上限：超大输入会让中间激活内存爆炸
        # （MaxPool bad allocation）。分片限制单批帧数，输出形状不变。
        # 16（原 64）为历史最优：小片更快且 ORT arena 峰值更低
        # （64: 920MB vs 16: 300MB，(3,48,320) small 模型 992 帧实测）。
        onnx_max = config.OCR_ONNX_CHUNK
        if len(batch_np) <= onnx_max:
            return np.asarray(self._session.run(None, {"x": batch_np})[0],
                              dtype=np.float32)
        outs = []
        for i in range(0, len(batch_np), onnx_max):
            outs.append(np.asarray(
                self._session.run(None, {"x": batch_np[i:i + onnx_max]})[0],
                dtype=np.float32))
        return np.concatenate(outs, axis=0)

    # ═══════════════ 后处理（复刻 CTCLabelDecode）═══════════════

    def _ctc_decode(self, pred: np.ndarray) -> RecOut:
        """单帧 (seq, 6906) → (文本, 置信度)。

        CTC：argmax → 相邻去重 → 移除 blank(0) → 字符映射。
        置信度 = 选中帧概率均值（round 5，与 rapidocr 一致）。
        """
        idx = pred.argmax(axis=1)
        prob = pred.max(axis=1)
        keep = np.ones(len(idx), dtype=bool)
        keep[1:] = idx[1:] != idx[:-1]
        keep &= idx != 0  # blank
        if keep.any():
            text = "".join(self._chars[i] for i in idx[keep])
            # 与 rapidocr 一致：每帧概率先 round(5) 再取均值，最后 round(5)
            confs = [round(float(p), 5) for p in prob[keep]]
            conf = round(float(np.mean(confs)), 5)
        else:
            text, conf = "", 0.0
        return RecOut(text, conf)

    def _ctc_decode_batch(self, preds: np.ndarray) -> list:
        """批 CTC decode：(B, seq, C) → list[RecOut]。

        分块归约：整批 argmax 在 C=6906 时产生 (B, seq) int64，~1000 帧一次
        归约峰值 ~2.2GB（Windows 堆不归还 → RSS 保持高位）。分块后峰值
        ~150MB。逐行归约与整批数值一致。
        """
        out: list = []
        for s0 in range(0, len(preds), config.OCR_CTC_CHUNK):
            chunk = preds[s0:s0 + config.OCR_CTC_CHUNK]
            idx = chunk.argmax(axis=2)  # (B, seq) int64
            prob = chunk.max(axis=2)
            keep = np.ones_like(idx, dtype=bool)
            keep[:, 1:] = idx[:, 1:] != idx[:, :-1]
            keep &= idx != 0  # blank
            for b in range(len(chunk)):
                kb = keep[b]
                if kb.any():
                    text = "".join(self._chars[i] for i in idx[b][kb])
                    # 与 rapidocr 一致：每帧概率先 round(5) 再取均值，最后 round(5)
                    confs = [round(float(p), 5) for p in prob[b][kb]]
                    conf = round(float(np.mean(confs)), 5)
                else:
                    text, conf = "", 0.0
                out.append(RecOut(text, conf))
        return out

    # ═══════════════ 批处理入口 ═══════════════

    def __call__(self, img_list: list) -> list:
        """批识别：与 rapidocr text_rec 同语义，按输入顺序返回结果。"""
        if not img_list:
            return []
        heights = [im.shape[0] for im in img_list]
        h0 = heights[0]
        # 按宽度排序（rapidocr 的加速策略；结果映射回原顺序）
        order = np.argsort([im.shape[1] for im in img_list])
        # pad 宽度 = max(批内最大宽高比, 本模型下限/OCR_TARGET_H)。速度数字
        # 是窄图（48 高后 78-160 宽），不设下限会让 GPU 白算过多宽度；
        # v6_small 对输入宽度敏感（窄图误读升高），必须有下限。
        # 优先级：用户 fill_width > env OCR_PAD_SMALL >
        # config.OCR_PAD_WIDTH_MIN_BY_MODEL。
        if self._fill_width > 0:
            _floor = self._fill_width
        else:
            _floor = config.OCR_PAD_WIDTH_MIN_BY_MODEL.get(
                self._variant, config.OCR_PAD_WIDTH_MIN)
            _env = os.environ.get(config.OCR_PAD_SMALL_ENV)
            if _env and _env.isdigit():
                _floor = int(_env)
        max_wh = max(_floor / config.OCR_TARGET_H,
                     *(float(im.shape[1]) / im.shape[0] for im in img_list))
        if self._trt is not None:
            # GPU 预处理路径：直接生成显存模型输入，省去 host batch 构造
            return self._call_trt_gpu(img_list, max_wh, h0)
        batch_np = np.stack([self._resize_norm(img_list[i], max_wh, h0)
                             for i in order])
        preds = self._infer(batch_np)
        results: list = [None] * len(img_list)
        # 批向量化 decode：argmax/max/keep 一次归约（与逐帧 _ctc_decode
        # 数值相同 —— 同一归约按行应用）；text 拼接保持逐帧
        if preds.ndim == 3:
            batch_results = self._ctc_decode_batch(preds)
            for k, idx in enumerate(order):
                results[idx] = batch_results[k]
        else:
            for k, idx in enumerate(order):
                results[idx] = self._ctc_decode(preds[k])
        return results

    # ═══════════════ TRT GPU 预处理专用路径 ═══════════════

    def _get_gpu_pre(self):
        """创建 GpuPreprocessor 并与 TrtEngine 共享同一 CUDA stream。"""
        from ocr_trt import GpuPreprocessor
        if self._gpu_pre is None:
            self._gpu_pre = GpuPreprocessor()
            self._gpu_pre._stream = self._trt._ensure_stream()
        return self._gpu_pre

    def _call_trt_gpu(self, img_list: list, max_wh: float,
                      h0: int) -> list:
        """TRT：GPU 端 transpose+normalize+pad → device tensor → 推理。"""
        self._get_gpu_pre()
        out_width = int(h0 * max_wh)
        dev_ptr, shape = self._gpu_pre.process(img_list, out_width)
        preds = self._infer_trt_device(dev_ptr, shape)
        results: list = []
        if preds.ndim == 3:
            results = self._ctc_decode_batch(preds)
        else:
            for k in range(len(img_list)):
                results.append(self._ctc_decode(preds[k]))
        return results

    def call_gpu_raw(self, infos: list) -> list:
        """TRT：直接消费 decord GPU NDArray（灰度 raw），跳过 D2H 代表帧拷贝。

        infos: [(dev_ptr, src_h, src_w, owner), ...]
        GPU_CTC=1 时进一步在 GPU 上完成 vocab 维 argmax/max，输出
        不落 RAM（DtoH 仅 B*seq*8 字节），宿主 CTC 直接吃小数组。
        """
        self._get_gpu_pre()
        src_h = int(infos[0][1])
        src_w = int(infos[0][2])
        if self._fill_width > 0:
            _floor = self._fill_width
        else:
            _floor = config.OCR_PAD_WIDTH_MIN_BY_MODEL.get(
                self._variant, config.OCR_PAD_WIDTH_MIN)
            _env = os.environ.get(config.OCR_PAD_SMALL_ENV)
            if _env and _env.isdigit():
                _floor = int(_env)
        max_wh = max(_floor / config.OCR_TARGET_H,
                     float(src_w) / float(src_h))
        out_width = int(config.OCR_TARGET_H * max_wh)
        dev_ptr, shape = self._gpu_pre.process_gray_raw(infos, out_width)
        if getattr(self, "_gpu_ctc_mode", False):
            idx2d, prob2d = self._trt.execute_device_argmax(dev_ptr, shape)
            return self._ctc_from_idxprob(idx2d, prob2d)
        preds = self._infer_trt_device(dev_ptr, shape)
        results = []
        if preds.ndim == 3:
            results = self._ctc_decode_batch(preds)
        else:
            for k in range(len(preds)):
                results.append(self._ctc_decode(preds[k]))
        return results

    def _ctc_from_idxprob(self, idx: "np.ndarray",
                          prob: "np.ndarray") -> list:
        """从 GPU 归约后的 (B,S) 索引/概率做 CTC 解码（与
        _ctc_decode_batch 语义一致：相邻去重 → 去 blank → 字符映射，
        置信度 = 保留位置概率均值 round5）。"""
        import numpy as np
        out: list = []
        for b in range(len(idx)):
            ib = idx[b]
            pb = prob[b]
            keep = np.ones(len(ib), dtype=bool)
            keep[1:] = ib[1:] != ib[:-1]
            keep &= ib != 0  # blank
            if keep.any():
                text = "".join(self._chars[int(i)] for i in ib[keep])
                confs = [round(float(p), 5) for p in pb[keep]]
                conf = round(float(np.mean(confs)), 5)
            else:
                text, conf = "", 0.0
            out.append(RecOut(text, conf))
        return out

    def _infer_trt_device(self, dev_ptr: int, shape: tuple) -> np.ndarray:
        """对已位于显存的整批输入执行 TRT，整批只同步一次。"""
        B = int(shape[0])
        max_batch = self._trt.max_batch
        first_shape = (min(B, max_batch),) + tuple(shape[1:])
        out_shape = self._trt._prepare_shape(first_shape)
        preds = np.empty(
            (B,) + tuple(out_shape[1:]), dtype=np.float32)
        elem_floats = int(np.prod(shape[1:]))
        for i in range(0, B, max_batch):
            n = min(max_batch, B - i)
            off = i * elem_floats * 4
            self._trt.execute_device_async(
                dev_ptr + off, (n,) + tuple(shape[1:]),
                out_host=preds[i:i + n])
        self._trt.synchronize()
        return preds
