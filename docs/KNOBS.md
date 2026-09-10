# 旋钮注册表（渲染产物：config/knobs.py → 本表）

| 旋钮 | env | 类型 | 默认 | 依据锚点 |
|---|---|---|---|---|
| `decode.num_threads` | `DECODE_THREADS` | int | `0` | ec:96-130 |
| `decode.hybrid_cpu_threads` | `HYBRID_CPU_THREADS` | int | `0` | ec:87-95 |
| `ocr.threads` | `OCR_THREADS` | int | `0` | ec:68 |
| `ocr.batch` | `OCR_BATCH` | int | `16` | ec:366 |
| `ocr.gamma` | `OCR_GAMMA` | float | `2.0` | ec:216-219 |
| `ocr.pad_small` | `OCR_PAD_SMALL` | int | `0` | ec:235-266 |
| `ocr.roi_autocrop` | `OCR_ROI_AUTOCROP` | bool | `True` | ec:271-311 |
| `ocr.roi_autocrop_margin` | `OCR_ROI_AUTOCROP_MARGIN` | int | `10` | ec:312-336 |
| `ocr.roi_autocrop_min_gain` | `OCR_ROI_AUTOCROP_MIN_GAIN` | int | `10` | ec:359 |
| `ocr.reorder_window` | `OCR_REORDER_WINDOW` | int | `64` | ec:360 |
| `ocr.instances` | `OCR_INSTANCES` | bool | `True` | ec:77 |
| `ocr.gpu_ctc` | `GPU_CTC` | bool | `True` | ec:83 |
| `segment.text_sep_merge` | `TEXT_SEP_MERGE` | str | `binary` | ec:229 |
| `pipeline.gpu` | `GPU_PIPELINE` | bool|none | `None` | ec:81 |
| `pipeline.gpu_stream` | `GPU_PIPELINE_STREAM` | bool | `False` | ec:82 |
| `diag.profile` | `ENGINE_PROFILE` | bool | `False` | ec:84 |
| `diag.subprobe` | `TRT_SUBPROBE` | bool | `False` | ec:85 |
| `diag.bounds_debug` | `DEBUG_BOUNDS` | bool | `False` | ec:86 |
| `diag.telemetry` | `VOE_TELEMETRY` | str | `std` | v2:8.6-r5 |
| `diag.report_file` | `VOE_REPORT_FILE` | str | `` | v2:8.6-N3 |

（语义备注与解析细节见 config/knobs.py 各 Knob 的 note 字段；v1 依据全文在 engine_config.py 注释，锚点 `ec:<行>`。）
