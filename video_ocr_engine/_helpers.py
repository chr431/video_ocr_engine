"""引擎级独立工具函数（从 extractor.py 拆出，无类依赖）。

_ocr_batch_size / _ndarray_device_ptr / _otsu_from_hist / _gray_mean_abs_diff。
extractor 与各 mixin 直接引用；为保持外部兼容，extractor 仍 re-export。
"""
import os as _os

import numpy as np

import engine_config as config

def _ocr_batch_size() -> int:
    _env = _os.environ.get("OCR_BATCH")
    if _env and _env.isdigit():
        return max(1, int(_env))
    return config.OCR_BATCH_SIZE


def _ndarray_device_ptr(nd):
    """从 decord GPU NDArray DLPack 解析 device 数据基址。

    返回 (base_ptr:int, shape:tuple[int,...])。调用方必须保持 nd 存活。
    """
    import ctypes
    cap = nd.to_dlpack()
    _get = ctypes.pythonapi.PyCapsule_GetPointer
    _get.restype = ctypes.c_void_p
    _get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = _get(cap, b"dltensor")

    class _DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int32),
                    ("device_id", ctypes.c_int32)]

    class _DLDataType(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                    ("lanes", ctypes.c_uint16)]

    class _DLTensor(ctypes.Structure):
        _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice),
                    ("ndim", ctypes.c_int32), ("dtype", _DLDataType),
                    ("shape", ctypes.POINTER(ctypes.c_int64)),
                    ("strides", ctypes.POINTER(ctypes.c_int64)),
                    ("byte_offset", ctypes.c_uint64)]

    t = ctypes.cast(ptr, ctypes.POINTER(_DLTensor)).contents
    shape = tuple(int(t.shape[i]) for i in range(t.ndim))
    return int(t.data), shape


def _otsu_from_hist(hist) -> int:
    """从 256-bin 直方图算 Otsu 阈值（与 segmentation._otsu 等价）。"""
    hist = np.asarray(hist, dtype=np.int64)
    total = int(hist.sum())
    if total <= 0:
        return config.OTSU_FALLBACK_THRESH
    st = float((np.arange(256) * hist).sum())
    sb = 0.0
    wb = 0
    best = config.OTSU_FALLBACK_THRESH
    vmax = -1.0
    for t in range(256):
        wb += int(hist[t])
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += t * int(hist[t])
        mb = sb / wb
        mf = (st - sb) / wf
        vb = wb * wf * (mb - mf) ** 2
        if vb > vmax:
            vmax = vb
            best = t
    return best


def _gray_mean_abs_diff(a, b) -> float:
    """两帧分段灰度 ROI 的平均绝对差；形状不一致时视为不相似。"""
    if a is None or b is None:
        return float("inf")
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
