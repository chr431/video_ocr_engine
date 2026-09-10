# S0 录制期发现（v1 行为缺陷候选，供 v2 §13.3 清单增补）

录制本身就是对 v1 的一次全矩阵巡检。以下为 2026-09-10 录制轮抓到的问题：

## F-1 · `force_aspect` 越界值 → 报错路径崩溃（B-aspect 用例，ratio=320）

- **现象**：`force_aspect=320`（被误当宽度传入；实际语义是宽高比 w/h）时，
  OCR 输入宽 = 48×320 = 15360，超出 TRT 引擎 profile 上限
  `[1,3,48,32]..[6,3,48,2048]`。
- **v1 缺陷不在"拒绝执行"而在"怎么拒绝"**：TRT `setInputShape` 已经报
  `API Usage Error`，但错误未在 `execute_device_argmax` 干净上抛，而是让
  `idx_all.reshape(B, seq)` 以 `ValueError: cannot reshape array of size 84
  into shape (18,4)` 崩溃——真实根因（宽度越界）被 reshape 异常掩盖。
- **v2 处置建议**：构造期校验 `force_aspect` 合法域（宽高比 > 0，且
  `OCR_TARGET_H × ratio ≤ 引擎 max_input_w`，违反即 `ValueError`——对齐 B3
  的"未知值构造期失败"原则）；设备执行路径的引擎错误必须原样上抛。

## F-2 · `meta` 键数实测 9（v2 文档 §10.1 写"10 个键"）

- `extractor.py:283-296` 组装的 meta 恰为 9 键：backend / ocr_backend /
  codec / n_segments / engine_version / color_range / rep_crop_format /
  degraded_reason / params。已在 v2 ARCHITECTURE.md 附录 D 记勘误 D-15。
