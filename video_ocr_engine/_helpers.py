"""引擎级独立工具函数（从 extractor.py 拆出，无类依赖）。

_ocr_batch_size / _ndarray_device_ptr / 进度映射 / _read_fps_from_vr。
（S2：_otsu_from_hist/_otsu_median_threshold/_gray_mean_abs_diff 已删——
前两者是转发壳，实现唯一出处 segmentation；后者全仓零产品调用点。）
"""
from video_ocr_engine.config import constants as config

def _ocr_batch_size() -> int:
    return max(1, config.env_int(config.OCR_BATCH_ENV, config.OCR_BATCH_SIZE))


# 进度百分比映射（解码 3→58，OCR 58→86；与旧版进度条语义一致）。两条流水线
# 路径（单 / GPU 全驻留）各自写一遍 `3+frac*55` / `58+frac*28` 是
# 可维护性债务：调进度区间需同步多处。此处收敛唯一出处，各路径只传 frac。
def _decode_progress_pct(frac: float) -> float:
    """解码/分段阶段进度百分比：3 + frac*55（frac∈[0,1]，输出 [3, 58]）。"""
    return 3.0 + max(0.0, min(1.0, frac)) * 55.0


def _ocr_progress_pct(frac: float) -> float:
    """OCR 阶段进度百分比：58 + frac*28（frac∈[0,1]，输出 [58, 86]）。"""
    return 58.0 + max(0.0, min(1.0, frac)) * 28.0


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


def _read_fps_from_vr(vr):
    """从 decord reader 读平均帧率（get_avg_fps 优先，get_fps 兜底）。

    都拿不到（或均为非正）返回 None，调用方按 DEFAULT_FPS_FALLBACK 兜底。
    """
    for m in ('get_avg_fps', 'get_fps'):
        fn = getattr(vr, m, None)
        if fn is None:
            continue
        try:
            v = float(fn())
        except Exception:
            continue
        if v and v > 0:
            return v
    return None
