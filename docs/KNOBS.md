# 旋钮注册表（渲染产物：config/knobs.py → 本表）

| 旋钮 | env | 类型 | 默认 | 依据锚点 |
|---|---|---|---|---|
| `decode.num_threads` | `DECODE_THREADS` | int | `0` | config/constants.py:DECODE_THREADS_ENV |
| `decode.hybrid_cpu_threads` | `HYBRID_CPU_THREADS` | int | `0` | config/constants.py:HYBRID_CPU_THREADS_ENV |
| `ocr.threads` | `OCR_THREADS` | int | `0` | config/constants.py:OCR_THREADS_ENV |
| `ocr.batch` | `OCR_BATCH` | int | `16` | config/constants.py:OCR_BATCH_ENV |
| `ocr.gamma` | `OCR_GAMMA` | float | `2.0` | config/constants.py:OCR_GAMMA_ENV |
| `ocr.pad_small` | `OCR_PAD_SMALL` | int | `0` | config/constants.py:OCR_PAD_SMALL_ENV |
| `ocr.roi_autocrop` | `OCR_ROI_AUTOCROP` | bool | `True` | config/constants.py:OCR_ROI_AUTOCROP_ENV |
| `ocr.roi_autocrop_margin` | `OCR_ROI_AUTOCROP_MARGIN` | int | `10` | config/constants.py:OCR_ROI_AUTOCROP_MARGIN_ENV |
| `ocr.roi_autocrop_min_gain` | `OCR_ROI_AUTOCROP_MIN_GAIN` | int | `10` | config/constants.py:OCR_ROI_AUTOCROP_MIN_GAIN_ENV |
| `ocr.reorder_window` | `OCR_REORDER_WINDOW` | int | `64` | config/constants.py:OCR_REORDER_WINDOW_ENV |
| `ocr.instances` | `OCR_INSTANCES` | bool | `True` | config/constants.py:OCR_INSTANCES_ENV |
| `ocr.gpu_ctc` | `GPU_CTC` | bool | `True` | config/constants.py:GPU_CTC_ENV |
| `ocr.trt_fp16` | `TRT_FP16` | bool | `False` | ocr/trt.py:_fp16_on |
| `ocr.trt_defer_sync` | `TRT_DEFER_SYNC` | bool | `False` | ocr/trt.py 延迟收集环 |
| `segment.text_sep_merge` | `TEXT_SEP_MERGE` | str | `binary` | config/constants.py:TEXT_SEP_MERGE_ENV |
| `segment.merge_dense_gate` | `SEG_MERGE_DENSE_GATE` | int | `5` | C-38 |
| `pipeline.gpu` | `GPU_PIPELINE` | bool|none | `None` | config/constants.py:GPU_PIPELINE_ENV |
| `pipeline.gpu_stream` | `GPU_PIPELINE_STREAM` | bool | `False` | config/constants.py:GPU_PIPELINE_STREAM_ENV |
| `pipeline.gpu_drainer` | `GPU_PIPELINE_DRAINER` | bool | `False` | gpu/streams.py:_batch_iter |
| `diag.profile` | `ENGINE_PROFILE` | bool | `False` | config/constants.py:ENGINE_PROFILE_ENV |
| `diag.subprobe` | `TRT_SUBPROBE` | bool | `False` | config/constants.py:TRT_SUBPROBE_ENV |
| `diag.bounds_debug` | `DEBUG_BOUNDS` | bool | `False` | config/constants.py:DEBUG_BOUNDS_ENV |
| `diag.telemetry` | `VOE_TELEMETRY` | str | `off` | diag.telemetry |
| `diag.report_file` | `VOE_REPORT_FILE` | str | `` | pipeline/report.py:RunReport |
| `diag.trace_file` | `VOE_TRACE_FILE` | str | `` | domain/trace.py |

（语义备注与解析细节见 config/knobs.py 各 Knob 的 note 字段。）

