"""全部旋钮声明。

默认值与解析语义：
  bool —— 真集 1/true/yes/on、假集 0/false/no/off（strip+lower），其余回退默认；
  int/float —— 缺省/空/非法回退默认；
  str  —— 原样（TEXT_SEP_MERGE 未知值由消费端回落 binary）。

特殊语义（note 字段如实记录）：
  decode.num_threads / decode.hybrid_cpu_threads / ocr.threads —— 0 = 自动
  （派生计算留在消费端，resolve 只产出原始值）；
  pipeline.gpu —— 三态：未设 None（规则判定）/ falsy False / **非法值也 False**
  （经 env_bool(default=False)，非法≠未设）；
  ocr.pad_small —— 0 = 不生效，链路 0→fill_width→模型下限（消费端 ocr/native.py）。
"""

from __future__ import annotations

from .registry import Knob, Registry

_ALL = ("cpu", "gpu")

KNOBS = Registry(
    knobs=(
        # ── decode ──
        Knob(
            "decode.num_threads",
            "DECODE_THREADS_ENV",
            "DECODE_THREADS",
            "int",
            0,
            "decode",
            _ALL,
            "config/constants.py:DECODE_THREADS_ENV",
            note="0=自动分档（codec×stride 公式在消费端）",
        ),
        Knob(
            "decode.hybrid_cpu_threads",
            "HYBRID_CPU_THREADS_ENV",
            "HYBRID_CPU_THREADS",
            "int",
            0,
            "decode",
            _ALL,
            "config/constants.py:HYBRID_CPU_THREADS_ENV",
            note="0=自动：与 cpu 后端同一 codec 感知策略（_decode_num_threads）",
        ),
        # ── ocr ──
        Knob(
            "ocr.threads",
            "OCR_THREADS_ENV",
            "OCR_THREADS",
            "int",
            0,
            "ocr",
            _ALL,
            "config/constants.py:OCR_THREADS_ENV",
            note="0=全部物理核（消费端派生）",
        ),
        Knob(
            "ocr.batch",
            "OCR_BATCH_ENV",
            "OCR_BATCH",
            "int",
            16,
            "ocr",
            _ALL,
            "config/constants.py:OCR_BATCH_ENV",
            note="默认取 OCR_BATCH_SIZE=16",
        ),
        Knob(
            "ocr.gamma",
            "OCR_GAMMA_ENV",
            "OCR_GAMMA",
            "float",
            2.0,
            "ocr",
            _ALL,
            "config/constants.py:OCR_GAMMA_ENV",
        ),
        Knob(
            "ocr.pad_small",
            "OCR_PAD_SMALL_ENV",
            "OCR_PAD_SMALL",
            "int",
            0,
            "ocr",
            _ALL,
            "config/constants.py:OCR_PAD_SMALL_ENV",
            note="0=不生效；>fill_width 时抬升 pad 下限",
        ),
        Knob(
            "ocr.roi_autocrop",
            "OCR_ROI_AUTOCROP_ENV",
            "OCR_ROI_AUTOCROP",
            "bool",
            True,
            "ocr",
            _ALL,
            "config/constants.py:OCR_ROI_AUTOCROP_ENV",
        ),
        Knob(
            "ocr.roi_autocrop_margin",
            "OCR_ROI_AUTOCROP_MARGIN_ENV",
            "OCR_ROI_AUTOCROP_MARGIN",
            "int",
            10,
            "ocr",
            _ALL,
            "config/constants.py:OCR_ROI_AUTOCROP_MARGIN_ENV",
        ),
        Knob(
            "ocr.roi_autocrop_min_gain",
            "OCR_ROI_AUTOCROP_MIN_GAIN_ENV",
            "OCR_ROI_AUTOCROP_MIN_GAIN",
            "int",
            10,
            "ocr",
            _ALL,
            "config/constants.py:OCR_ROI_AUTOCROP_MIN_GAIN_ENV",
        ),
        Knob(
            "ocr.reorder_window",
            "OCR_REORDER_WINDOW_ENV",
            "OCR_REORDER_WINDOW",
            "int",
            64,
            "ocr",
            _ALL,
            "config/constants.py:OCR_REORDER_WINDOW_ENV",
            note="消费端 max(1,·)；0 也抬到 1",
        ),
        Knob(
            "ocr.instances",
            "OCR_INSTANCES_ENV",
            "OCR_INSTANCES",
            "bool",
            True,
            "ocr",
            ("cpu",),
            "config/constants.py:OCR_INSTANCES_ENV",
            note="仅 ONNX 双实例路径",
        ),
        Knob(
            "ocr.gpu_ctc",
            "GPU_CTC_ENV",
            "GPU_CTC",
            "bool",
            True,
            "ocr",
            ("gpu",),
            "config/constants.py:GPU_CTC_ENV",
            note="仅 TRT 输出归约",
        ),
        Knob(
            "ocr.trt_fp16",
            "TRT_FP16_ENV",
            "TRT_FP16",
            "bool",
            False,
            "ocr",
            ("gpu",),
            "ocr/trt.py:_fp16_on",
            note="fp16 ONNX（keep_io_types）+ STRONGLY_TYPED 引擎（层内 half、"
            "IO fp32）；准确率门禁 archive/pre-rework:tools/_probe_acc_ab.py --knob TRT_FP16=0,1",
            env_live_only=True,
        ),  # env-live-only：消费者调用期直读 env；digest 收 resolve 期 env 实值；override 拒收
        Knob(
            "ocr.trt_defer_sync",
            "TRT_DEFER_SYNC_ENV",
            "TRT_DEFER_SYNC",
            "bool",
            False,
            "ocr",
            ("gpu",),
            "ocr/trt.py 延迟收集环",
            note="raw 直通批深度 2 延迟收集（提交不等待、结果滞后一批、"
            "提交序交付）；治 launch 裸奔（运行期 SM ~40%）；"
            "门禁=文本 sha 逐位+段数+bench ab",
            env_live_only=True,
        ),  # env-live-only：消费者调用期直读 env；digest 收 resolve 期 env 实值；override 拒收
        # ── segment ──
        Knob(
            "segment.text_sep_merge",
            "TEXT_SEP_MERGE_ENV",
            "TEXT_SEP_MERGE",
            "str",
            "binary",
            "segment",
            _ALL,
            "config/constants.py:TEXT_SEP_MERGE_ENV",
            note="裸 get 非布尔；未知值消费端回落 binary",
        ),
        Knob(
            "segment.merge_dense_gate",
            "SEG_MERGE_DENSE_GATE_ENV",
            "SEG_MERGE_DENSE_GATE",
            "int",
            5,
            "segment",
            _ALL,
            "C-38",
            note="0=关；>0=差异图最大 3×3 窗口和（win3）阈值：≥ 此值判「内容已"
            "变」恒不合并。默认 = SEG_C（与断段判据共用同一 win3 定义，两处"
            "自洽）；判据/实测见 C-38",
        ),
        # ── pipeline ──
        Knob(
            "pipeline.gpu",
            "GPU_PIPELINE_ENV",
            "GPU_PIPELINE",
            "bool|none",
            None,
            "pipeline",
            ("gpu",),
            "config/constants.py:GPU_PIPELINE_ENV",
            note="三态：未设 None=规则 / falsy 关 / truthy 强制 / 非法=关（非 None）",
        ),
        Knob(
            "pipeline.gpu_stream",
            "GPU_PIPELINE_STREAM_ENV",
            "GPU_PIPELINE_STREAM",
            "bool",
            False,
            "pipeline",
            ("gpu",),
            "config/constants.py:GPU_PIPELINE_STREAM_ENV",
            note="C-10 判零收益，opt-in",
            env_live_only=True,
        ),  # env-live-only：消费者调用期直读 env；digest 收 resolve 期 env 实值；override 拒收
        Knob(
            "pipeline.gpu_drainer",
            "GPU_PIPELINE_DRAINER_ENV",
            "GPU_PIPELINE_DRAINER",
            "bool",
            False,
            "pipeline",
            ("gpu",),
            "gpu/streams.py:_batch_iter",
            note="层4 排空线程（get_batch 独占线程预取深 4 批）；实测零收益"
            "默认关（引擎税=线程争用而非排空阻塞，同 C-10 判例）",
            env_live_only=True,
        ),  # env-live-only：消费者调用期直读 env；digest 收 resolve 期 env 实值；override 拒收
        # ── diag ──
        Knob(
            "diag.profile",
            "ENGINE_PROFILE_ENV",
            "ENGINE_PROFILE",
            "bool",
            False,
            "diag",
            _ALL,
            "config/constants.py:ENGINE_PROFILE_ENV",
            note="构造期一次读取（resolve 冻结）",
        ),
        Knob(
            "diag.subprobe",
            "TRT_SUBPROBE_ENV",
            "TRT_SUBPROBE",
            "bool",
            False,
            "diag",
            ("gpu",),
            "config/constants.py:TRT_SUBPROBE_ENV",
            note="TRT_SUBPROBE 在 trt.py 导入期读一次（进程级；诊断档，不做调用期活读）",
            env_live_only=True,
        ),  # env-live-only：消费者调用期直读 env；digest 收 resolve 期 env 实值；override 拒收
        Knob(
            "diag.bounds_debug",
            "DEBUG_BOUNDS_ENV",
            "DEBUG_BOUNDS",
            "bool",
            False,
            "diag",
            _ALL,
            "config/constants.py:DEBUG_BOUNDS_ENV",
            env_live_only=True,
        ),  # env-live-only：消费者调用期直读 env；digest 收 resolve 期 env 实值；override 拒收
        Knob(
            "diag.telemetry",
            "VOE_TELEMETRY",
            "VOE_TELEMETRY",
            "str",
            "off",
            "diag",
            _ALL,
            "diag.telemetry",
            note="off/full 两档（发布=off 全关，调试=full）；std=受谴责别名→full",
        ),
        Knob(
            "diag.report_file",
            "VOE_REPORT_FILE",
            "VOE_REPORT_FILE",
            "str",
            "",
            "diag",
            _ALL,
            "pipeline/report.py:RunReport",
            note="RunReport 细档 JSON sidecar 输出路径；空=不写（写文件是副作用，须显式 opt-in）",
        ),
        Knob(
            "diag.trace_file",
            "VOE_TRACE_FILE",
            "VOE_TRACE_FILE",
            "str",
            "",
            "diag",
            _ALL,
            "domain/trace.py",
            note="实验性全事件时间线 JSON 路径；空=关（关闭零成本：_prof_end "
            "一次 is-not-None 判断）；需 VOE_TELEMETRY!=off；开启时自带 "
            "NVML 0.2s 点列",
        ),
    )
)
