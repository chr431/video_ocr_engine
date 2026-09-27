"""引擎策略层（R1，2026-09-27）：解码/OCR 后端选择的纯函数集合。

自 extractor 门面下放的**无状态决策**（输入只有配置与后端名，输出可表
驱动单测）：
  - merge_effective_mode：相似段合并的分离模式归一
  - ocr_on_gpu / ocr_engine_type：OCR 落点与执行引擎类型
  - gpu_pipeline_enabled：GPU 全驻留管线门控（GPU_PIPELINE 三态）
  - ocr_num_threads：OCR 推理线程预算（少核分核）
  - decode_num_threads 的 re-export：事实源在 config.decode_caliber

判定依据的实测注释随实现一并迁入（勿据直觉改判，见 docs/CONCLUSIONS.md
对应条目）。带状态的决策（codec 探测后的线程档位重开）留在
decode/decord_source.py（打开流程的一部分）；CPU 软解线程档位的事实源
在 config.decode_caliber（单一出处，探针与引擎同源派生）。
"""
from __future__ import annotations

from video_ocr_engine.config import constants as config


def merge_effective_mode(text_sep: str | None) -> str:
    """merge_similar 使用的分离模式：'binary' | ''（原始灰度比较）。

    contrast 模式已移除（实验证实无净收益，0.9.0 清理）；未知值归一
    到引擎默认 binary。
    """
    _m = (text_sep or '').strip().lower()
    if _m in ('2', 'binary'):
        return 'binary'
    if _m in ('off', ''):
        return ''
    return 'binary'   # contrast/未知值 → 引擎默认 binary


def ocr_on_gpu(ocr_backend: str | None) -> bool:
    """OCR 推理是否卸载到 GPU（TensorRT）。

    为 True 时 host CPU 在解码阶段基本空闲（TRT 只占少量提交线程），
    解码可以放宽线程预算（见 decode_num_threads）。仅按配置判断，
    不表示 TRT 一定可用（不可用时 OcrEngine 内部回退 OpenVINO，此时
    解码线程偏多只是轻微过订阅，实测不劣化）。
    """
    return (ocr_backend or 'auto').lower() != 'cpu'


def ocr_engine_type(ocr_backend: str | None) -> str:
    """OCR 推理后端：auto/tensorrt → tensorrt（失败回退 CPU 路径），cpu → openvino。

    hybrid（2026-09-20 hybrid ocr 轮，opt-in）→ 'hybrid'：TRT 设备
    车道 + OpenVINO CPU 车道各一引擎（ocr_stage.acquire_engines 取双
    擎、共享 infer_q 工作窃取）。收益面 = OCR-bound 场景（当前实测
    仅 batch_test 稠密字幕 stride=1 类负载，infer 忙时 92% wall）；
    decode-bound 的常规负载零收益（L3 本就空等）。
    """
    _b = (ocr_backend or 'auto').lower()
    if _b == 'cpu':
        return 'onnxruntime'
    if _b == 'hybrid':
        return 'hybrid'
    return 'tensorrt'


def gpu_pipeline_enabled(rc, decode_backend: str | None,
                         ocr_backend: str | None,
                         video_path: str) -> bool:
    """GPU 全驻留零拷贝管线：NVDEC 直通或 CPU 解码 + H2D（P1-3）。

    默认（GPU_PIPELINE 未设置）启用条件（全部满足）：
    - decode_backend ∈ {auto, nvdec, cpu, hybrid}：auto/nvdec 走 NVDEC
      设备指针直通（NVDEC 打开失败时回退 CPU 解码分支）；cpu 显式
      选择 CPU 软解 + H2D 进 GPU 分段/OCR（P1-3 解耦——CPU 解码的
      墙钟收益与零拷贝 OCR 不再互斥）；hybrid 走 CPU 分支消费
      hybrid 解码器交付的宿主数组（§8.3：双解码收益 + 零拷贝 OCR
      叠加，原互斥门控已移除）。
    - TensorRT 可用且 ocr_backend ≠ cpu —— 全程 raw 才有净收益
      （GPU 分段+ONNX 实测无优势，默认走宿主管线，配置面更简）
    - cuda-python（cuda.core / cuda.bindings）可导入

    env GPU_PIPELINE：'0' 显式关闭；'1' 强制尝试（跳过 TRT 要求，
    允许 GPU 分段+ONNX 等实验组合）；不设置 = 上述默认规则。
    rc.pipeline_gpu 为构造期冻结的三态（None=规则 / falsy 关 /
    truthy 强制）。
    """
    from ..gpu import device as _gp
    # 经模块属性解析:tests/探针 patch gpu.device.nvdec_available
    # 等模块级名字(§10.4 patch 点),函数级导入保持该间接性
    _cuda = _gp._cuda_python_available
    _val = rc.pipeline_gpu
    if _val is not None:
        if not _val:
            return False
        forced = True
    else:
        forced = False
    backend = (decode_backend or 'auto').lower()
    if backend not in ('auto', 'nvdec', 'cpu', 'hybrid'):
        return False
    if not _cuda():
        return False
    if not forced:
        if (ocr_backend or 'auto').lower() == 'cpu':
            return False
        if not _gp.tensorrt_available():
            return False
    if backend == 'cpu':
        # CPU 解码分支不依赖 NVDEC：跳过 nvdec 探测（避免无谓的
        # GPU reader 试开；TRT 可用性已由上方门控确认）。
        return True
    return _gp.nvdec_available(str(video_path))


def ocr_num_threads(rc, codec: str, backend_label: str) -> int:
    """OCR 推理线程预算：OCR_THREADS env 钩子优先，否则全物理核；
    CPU 软解且物理核 ≤ 8 时与解码显式分核（cores//2，防过订阅）。

    解码（NVDEC 全卸载 / CPU 下 FFmpeg 帧线程 2 + filter auto 只占
    SMT 份额）不抢物理核，OCR 吃满全部物理核；CPU 软解在少核机上
    FFmpeg 帧线程与 OCR 争抢（实测 4 核 ocrT=2 28.0s vs 全核 33.1s、
    8 核 ocrT=4 17.8s vs 20.7s），分核更优；核数多时（16）分核反而
    差 → 保持全核。显式参数传入引擎，不污染全局 env。
    """
    from video_ocr_engine.ocr.native import auto_ocr_thread_count
    _env = rc.ocr_threads
    if _env:
        return max(1, _env)
    cores = auto_ocr_thread_count()
    if codec == 'av1' and backend_label.startswith('decord/CPU'):
        return max(2, cores // 2)
    if backend_label.startswith('decord/CPU') and cores <= config.CPU_CORES_SPLIT_THRESHOLD:
        return max(2, cores // 2)
    return cores
