"""GPU DLL 搜索路径注册。

从 PATH 扫描 CUDA / cuDNN / TensorRT 目录并注册到 Windows DLL 搜索路径。
OcrEngine 初始化 TensorRT 前调用 ensure_gpu_initialized()；旧的后端选择
（select_backend/get_engine_type/get_setup_advice）已废弃：OcrEngine 自身
按 engine_type 直接初始化并在失败时回退 ONNX。
"""
from __future__ import annotations
import logging
import os as _os

logger = logging.getLogger("video_ocr_engine.gpu_setup")

# ═══════════════════ 内部状态 ═══════════════════
_gpu_initialized: bool = False
_dll_dir_cookies: list = []  # 保持 os.add_dll_directory() 返回值存活


def ensure_gpu_initialized() -> None:
    """延迟初始化 GPU：首次调用时扫描并加载 CUDA/cuDNN/TensorRT DLL。"""
    global _gpu_initialized
    if not _gpu_initialized:
        _gpu_initialized = True
        _register_gpu_dlls()


def _register_gpu_dlls() -> None:
    """扫描 PATH 中的 CUDA / cuDNN / TensorRT DLL 目录并注册到搜索路径。

    用户只需将对应 bin 目录加入 PATH，无需特定安装位置。
    例如：C:\\Program Files\\NVIDIA\\TensorRT-11.x\\bin
    """
    # DLL 特征文件名（用于识别目录类型）
    _TRT_MARKERS = ("nvinfer",)
    _CUDA_MARKERS = ("cudart64_", "cudart32_", "cublas64_")
    _CUDNN_MARKERS = ("cudnn64_", "cudnn_ops64_")

    _found_trt: list[str] = []
    _found_cuda: list[str] = []
    _found_cudnn: list[str] = []
    _other_dirs: list[str] = []

    # ── 扫描 PATH 中所有目录 ──
    # Windows: os.environ 只在进程启动时读取合并后的 PATH。
    # 注册表修改后未注销重登的会话中，需直接读注册表补充。
    _path_raw = _os.environ.get("PATH", "")
    if _os.name == "nt":
        try:
            import winreg as _wr
            _reg_paths: list[str] = []
            for _hive, _subkey in [(_wr.HKEY_CURRENT_USER, "Environment"),
                                    (_wr.HKEY_LOCAL_MACHINE,
                                        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")]:
                try:
                    _k = _wr.OpenKey(_hive, _subkey, 0, _wr.KEY_READ)
                    _val, _ = _wr.QueryValueEx(_k, "Path")
                    _wr.CloseKey(_k)
                    _reg_paths.extend(_val.split(";"))
                except OSError:
                    pass
            # 追加注册表中独有的条目（去重）
            _env_set = {_os.path.normpath(p) for p in _path_raw.split(_os.pathsep) if p.strip()}
            _reg_extra = [p for p in _reg_paths
                            if p.strip() and _os.path.normpath(p) not in _env_set]
            if _reg_extra:
                _path_raw += _os.pathsep + _os.pathsep.join(_reg_extra)
                logger.info("PATH 补充注册表条目: %d 个", len(_reg_extra))
        except Exception:
            pass

    _seen: set[str] = set()
    _path_entries = _path_raw.split(_os.pathsep)
    # 截断过长的 PATH（仅日志用）
    _path_preview = _path_raw[:500] + ("..." if len(_path_raw) > 500 else "")
    logger.info("PATH 扫描: %d 个条目, 前500字符: %s", len(_path_entries), _path_preview)
    for _entry in _path_entries:
        _entry = _os.path.normpath(_entry.strip())
        if not _entry or _entry in _seen:
            continue
        _seen.add(_entry)
        if not _os.path.isdir(_entry):
            continue

        # 检查目录中的 DLL 类型
        try:
            _contents = _os.listdir(_entry)
        except OSError:
            continue

        _lower_contents = [f.lower() for f in _contents]
        if any(f.startswith(m) for f in _lower_contents for m in _TRT_MARKERS):
            _found_trt.append(_entry)
        elif any(f.startswith(m) for f in _lower_contents for m in _CUDNN_MARKERS):
            _found_cudnn.append(_entry)
        elif any(f.startswith(m) for f in _lower_contents for m in _CUDA_MARKERS):
            _found_cuda.append(_entry)
        elif any(f.endswith(".dll") for f in _lower_contents):
            _other_dirs.append(_entry)

    # ── 注册 DLL 搜索目录 ──
    # os.add_dll_directory() 用于 Windows 原生 LoadLibrary，
    # 但 tensorrt 包的 find_lib() 只搜 PATH，所以也要更新 os.environ。
    global _dll_dir_cookies
    _dll_dir_cookies.clear()
    _registered = 0
    _path_new: list[str] = []
    for _label, _dirs in [("CUDA", _found_cuda), ("cuDNN", _found_cudnn),
                            ("TensorRT", _found_trt), ("DLL", _other_dirs)]:
        if _dirs:
            logger.info("%s: %s", _label, ", ".join(_dirs[:3]))
        for _d in _dirs:
            try:
                _dll_dir_cookies.append(_os.add_dll_directory(_d))
                _path_new.append(_d)
                _registered += 1
            except (AttributeError, OSError):
                pass

    # 将找到的目录前置到 PATH（tensorrt 包依赖此路径）
    if _path_new:
        _existing = _os.environ.get("PATH", "")
        _os.environ["PATH"] = _os.pathsep.join(_path_new) + \
            (_os.pathsep + _existing if _existing else "")

    if not _found_trt:
        logger.info("TensorRT DLL 未在 PATH 中找到 (搜索了 %d 个目录)", len(_path_entries))
    logger.info("GPU DLL 搜索路径注册: %d 个目录 (TRT:%d CUDA:%d)",
        _registered, len(_found_trt), len(_found_cuda))
