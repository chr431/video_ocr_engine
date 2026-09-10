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

## F-4 · 宿主外提引入"活读时序耦合"回归，被金标当场抓获（2026-09-10 S3-3a）

- **现象**：驱动迁入 `pipeline/host_backend.py` 后，162 个单测全绿，但金标
  `--verify` 显示 6 个宿主路径用例段数/文本全面漂移（GPU 路径不漂）。
- **根因**：v1 驱动校准后**立即**写 `self._bin_thresh`，而 `_segments_similar`
  在流式期间**活读**该属性；新门面把回写延迟到 run 结束 → 流式期间阈值为 0
  → binary 分离图全错 → 合并判定失真。这正是 P0-1 描述的隐式时序契约，
  单测（无真视频）无法覆盖。
- **修复**：`HostRunSpec.on_bin_thresh` 回调，驱动校准后即时回写。
- **启示**：金标门禁在"全部单测绿"的情况下抓住了行为回归——S3 手术期间
  任何一步都不许跳过 `--verify`。

## F-2 · `meta` 键数实测 9（v2 文档 §10.1 写"10 个键"）

- `extractor.py:283-296` 组装的 meta 恰为 9 键：backend / ocr_backend /
  codec / n_segments / engine_version / color_range / rep_crop_format /
  degraded_reason / params。已在 v2 ARCHITECTURE.md 附录 D 记勘误 D-15。

## F-3 · B3 修复当场抓获 S0 矩阵自身的一个真实踩坑（2026-09-10 S3-1）

- S0 录制矩阵曾写 `ocr_backend="onnx"`——v1 语义把它静默当 **tensorrt** 跑
  （P0-3 列举的陷阱类别），导致 6 个"onnx"用例实为 TRT 重复、ONNX 零覆盖。
- B3（构造期校验）落地后 `--verify` 立即失败并暴露此错。矩阵已改为
  `ocr_backend="cpu"` 并重录：**真 ONNX 与 TRT 在 test5 上文本 sha 逐位相同、
  段结构相同、置信度差在第 4 位小数**（0.99463 vs 0.99455）——OCR 后端
  文本级等价首次获得金标背书。

