"""全部旋钮声明（v1 engine_config 18 个 env 旋钮的等价物 + r5 telemetry）。

默认值与解析语义**逐项对齐 v1 实测**（附录 C 全表 / engine_config.py:28-64）：
  bool —— 真集 1/true/yes/on、假集 0/false/no/off（strip+lower），其余回退默认；
  int/float —— 缺省/空/非法回退默认；
  str  —— 原样（TEXT_SEP_MERGE 在 v1 是裸 os.environ.get，未知值由消费端回落 binary）。

特殊语义（note 字段如实记录）：
  decode.num_threads / decode.hybrid_cpu_threads / ocr.threads —— 0 = 自动
  （派生计算留在消费端，resolve 只产出原始值，与 v1 读取点逐位等价）；
  pipeline.gpu —— 三态：未设 None（规则判定）/ falsy False / **非法值也 False**
  （v1 _gpu_pipeline.py:566-570：经 env_bool(default=False)，非法≠未设）；
  ocr.pad_small —— 0 = 不生效，链路 0→fill_width→模型下限（消费端 ocr_native）。
"""
from __future__ import annotations

from .registry import Knob, Registry

_ALL = ("cpu", "gpu")

KNOBS = Registry(knobs=(
    # ── decode ──
    Knob("decode.num_threads", "DECODE_THREADS_ENV", "DECODE_THREADS", "int", 0,
         "decode", _ALL, "ec:96-130", note="0=自动分档（codec×stride 公式在消费端）"),
    Knob("decode.hybrid_cpu_threads", "HYBRID_CPU_THREADS_ENV", "HYBRID_CPU_THREADS",
         "int", 0, "decode", _ALL, "ec:87-95",
         note="0=自动 clamp [HYBRID_CPU_THREADS_AUTO_MIN, MAX]=[8,16]"),
    # ── ocr ──
    Knob("ocr.threads", "OCR_THREADS_ENV", "OCR_THREADS", "int", 0,
         "ocr", _ALL, "ec:68", note="0=全部物理核（消费端派生）"),
    Knob("ocr.batch", "OCR_BATCH_ENV", "OCR_BATCH", "int", 16,
         "ocr", _ALL, "ec:366", note="v1 默认取 OCR_BATCH_SIZE=16"),
    Knob("ocr.gamma", "OCR_GAMMA_ENV", "OCR_GAMMA", "float", 2.0,
         "ocr", _ALL, "ec:216-219"),
    Knob("ocr.pad_small", "OCR_PAD_SMALL_ENV", "OCR_PAD_SMALL", "int", 0,
         "ocr", _ALL, "ec:235-266", note="0=不生效；>fill_width 时抬升 pad 下限"),
    Knob("ocr.roi_autocrop", "OCR_ROI_AUTOCROP_ENV", "OCR_ROI_AUTOCROP", "bool",
         True, "ocr", _ALL, "ec:271-311"),
    Knob("ocr.roi_autocrop_margin", "OCR_ROI_AUTOCROP_MARGIN_ENV",
         "OCR_ROI_AUTOCROP_MARGIN", "int", 10, "ocr", _ALL, "ec:312-336"),
    Knob("ocr.roi_autocrop_min_gain", "OCR_ROI_AUTOCROP_MIN_GAIN_ENV",
         "OCR_ROI_AUTOCROP_MIN_GAIN", "int", 10, "ocr", _ALL, "ec:359"),
    Knob("ocr.reorder_window", "OCR_REORDER_WINDOW_ENV", "OCR_REORDER_WINDOW",
         "int", 64, "ocr", _ALL, "ec:360", note="v1 消费端 max(1,·)；0 也抬到 1"),
    Knob("ocr.instances", "OCR_INSTANCES_ENV", "OCR_INSTANCES", "bool", True,
         "ocr", ("cpu",), "ec:77", note="仅 ONNX 双实例路径"),
    Knob("ocr.gpu_ctc", "GPU_CTC_ENV", "GPU_CTC", "bool", True,
         "ocr", ("gpu",), "ec:83", note="仅 TRT 输出归约"),
    # ── segment ──
    Knob("segment.text_sep_merge", "TEXT_SEP_MERGE_ENV", "TEXT_SEP_MERGE", "str",
         "binary", "segment", _ALL, "ec:229",
         note="v1 裸 get 非布尔；未知值消费端回落 binary（B3 将改构造期校验）"),
    # ── pipeline ──
    Knob("pipeline.gpu", "GPU_PIPELINE_ENV", "GPU_PIPELINE", "bool|none", None,
         "pipeline", ("gpu",), "ec:81",
         note="三态：未设 None=规则 / falsy 关 / truthy 强制 / 非法=关（非 None）"),
    Knob("pipeline.gpu_stream", "GPU_PIPELINE_STREAM_ENV", "GPU_PIPELINE_STREAM",
         "bool", False, "pipeline", ("gpu",), "ec:82", note="C-10 判零收益，opt-in"),
    # ── diag ──
    Knob("diag.profile", "ENGINE_PROFILE_ENV", "ENGINE_PROFILE", "bool", False,
         "diag", _ALL, "ec:84", note="v1 读于构造期；v2 统一构造期一次（§10.2）"),
    Knob("diag.subprobe", "TRT_SUBPROBE_ENV", "TRT_SUBPROBE", "bool", False,
         "diag", ("gpu",), "ec:85", note="v1 读于 import 期；v2 统一构造期一次（§10.2）"),
    Knob("diag.bounds_debug", "DEBUG_BOUNDS_ENV", "DEBUG_BOUNDS", "bool", False,
         "diag", _ALL, "ec:86"),
    Knob("diag.telemetry", "VOE_TELEMETRY", "VOE_TELEMETRY", "str", "std",
         "diag", _ALL, "v2:8.6-r5", note="off/std/full；off=NullMetrics 一键关闭（PI-15）"),
))
