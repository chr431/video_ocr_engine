# S0 录制期发现（v1 行为缺陷候选，供 v2 §13.3 清单增补）

录制本身就是对 v1 的一次全矩阵巡检。以下为 2026-09-10 录制轮抓到的问题：

## F-8 · 金标置信度的"冷池/热池"敏感性（2026-09-10 S6 轮；非回归）

- **现象**：单用例、单进程（冷池）跑 `C-h264-nvdec` / `B-fmt-gray-kc1` 时，
  个别段的置信度在第 4 位小数漂移（0.82868 vs 0.82785、0.95512 vs 0.95528），
  段结构与文本 sha **逐位不变**。同进程第二次（热池）跑即完全一致。
- **根因**：`raw_ready` 在 emit 时才置位（单 TRT 引擎就绪 → 代表帧可留显存
  走 GPU raw 直通）。冷池 TRT 反序列化 0.31s 内 emit 的前若干段走**宿主
  numpy 预处理**，热池（0.07s）则走 **CUDA 预处理**——两者是 F-7 已记录的
  浮点敏感类（numpy↔CUDA 差异经 TRT 放大到少数段）。
- **验证**：`git stash` 掉 S6 全部改动后现象依旧（同一进程、同一断点），
  确认与改动无关；`tools/_probe_golden_diff.py <case>` 同进程连跑两次可复现
  "冷≠录、热=录"。
- **口径结论**：金标 `--verify` 必须按**一次进程跑完整矩阵**（脚本默认行为，
  池随首个用例转热）；单元格式的 `--case` 单跑不构成门禁。这是门禁使用口径，
  不是豁免——段数 / 文本 sha 仍然逐位硬门禁。

## F-10 · TRT profile batch 6→18：**文本逐位不变、置信度第 4 位漂移**（2026-09-10 S6）

- **动机**：`ocr.infer` 的每子批提交开销（launch + 归约 kernel + D2H + 形状
  sync）与 profile batch 成反比。同批 1116 张微基准：batch 6（186 子批）
  0.663s → batch 18（62 子批）0.557s（`tools/_probe_trt_maxbatch.py`）。
- **端到端交错 A/B**（`tools/_probe_engine_ab.py`，同一份代码只换引擎产物）：
  h264-cpu 热轮 **−5.85%**（符号 3/3），`ocr.infer` 0.828→0.555s（**−33%**），
  子批 181→61；h264-gpu（解码受限）+0.55%（OCR 相位 −24.6% 被解码掩盖）。
- **对账口径**：重跑全矩阵 `--verify` → 22 个 TRT 用例全部
  **段结构 sha 与文本 sha 逐位相同**，仅**置信度**在第 4 位小数漂移
  （如 0.82773 vs 0.82785；381/1109 段）。这属于 §8.5 的可白名单类别
  （"置信度这类浮点敏感输出"；段数/文本 sha 永不豁免，本处未违反）。
- **真值口径**：`e2e_smoke --verify`（test5，3000 帧，`sample_stride=1`）
  匹配率 **99.9%**——与旧引擎同（文本没变，准确率必然同）。
- **处置**：按 §10.2 显式声明（迁移指南 §4.1）+ 金标重录；引擎文件名加
  `_b{batch}` 标记，避免"改 profile 被旧缓存静默掩盖"（代价：首次重建
  68s，一次性）。

## F-11 · 金标与**引擎产物**强绑定：同 profile 重建也会让置信度漂移（2026-09-10 S6）

- **现象**：`_b18` 引擎因一次瞬态 CUDA 失败被重建（同日 00:42）后，重跑
  `--verify` 22 个 TRT 用例全部 DIFF——仍是**段结构 sha / 文本 sha 逐位相同、
  仅置信度第 4 位小数漂移**（531/1109 段）。
- **根因**：TRT 构建期的 tactic 选择不保证跨构建一致（无 timing cache 时）。
  同一份 ONNX、同一个 profile，两次构建的数值路径可以不同。
- **处置**：manifest 记录**引擎指纹**（`ocr_engine: 文件名@sha256[:16]`），
  使"代码回归"与"引擎重建"可区分；金标按当前产物重录（28/28 复验通过）。
- **口径**：金标向量 = **代码 × 视频 × GPU × 引擎产物** 的函数；换机/重建
  按 §8.5 以 manifest 为准重录，不得静默容忍。

## F-9 · `meta['report']` 增补使 `calib.meta_keys` 全矩阵变化（2026-09-10 S6-0）

- §8.6 N-3 要求 RunReport 从 `meta['report']` 透出（telemetry≠off，只增不改）。
  `record.py` 的 calib 摘要含 `meta_keys`，因此 28 例全部出现**且仅出现**
  `calib.meta_keys` 一处差异（其余字段逐位一致）。
- 处置：按 §10.2"有意变更需显式声明"重录金标（本文件与该轮 commit 一同记录
  录制基准），并在 CHANGELOG 声明。重录前的对账证据：28 例中除 meta_keys
  外零差异。

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

## F-6 · GPU 外提第三处时序耦合 + 顺带发现 v1 隐匿缺陷（2026-09-10 S3-3c）

- **现象**：GPU 驱动迁入 `pipeline/gpu_backend.py` 后 27/28——仅 B-stride8
  段结构相同但 289/339 个 rep_frame 漂移；spy 显示新旧 sharp/cluster 数值
  **完全相同**、仅帧号序列不同（旧 401,402… 连续 / 新 408,416… 网格）。
- **根因（F-6）**：旧代码在 `open_vr()` **之后**用 `self._backend` 判
  `on_gpu`（NVDEC 设备直通 vs CPU 解码+H2D 分支）；迁移门面在 spec 构造期
  （open 之前）求值——B1 重置后标签为空 → 恒走 CPU 分支。stride=1 时两条
  流输出恰好逐位相同（故 27 例全过），stride>1 才暴露。
- **修复**：`on_gpu` 判定移回 `run_gpu_pipeline` 内部、open 之后。
- **顺带发现（v1 隐匿缺陷候选 B8）**：NVDEC 帧流主体批 yield `f0 + k`
  （连续帧号），`sample_stride>1` 时 rep_frame 落在采样网格之外（如
  stride8 下出现 401）。段边界取自 frames 列表故正确；仅代表帧号受影响。
  金标已冻结该行为（S3 逐位一致原则）；修复属行为变更，留待 §10.2 流程
  与 S5+ 裁决。

## F-5 · 会话契约化抓获第二处活读时序耦合（2026-09-10 S3-3b）

- **现象**：OcrSession 改吃 SessionSpec 后，GPU 路径 `--verify` 崩于
  `KeyError: 0`（results 空）；宿主路径正常。
- **根因**：GPU 驱动在会话启动**之后**才置 `self._gpu_pipeline_mode = True`
  （`_gpu_pipeline.py` 原 :767），旧会话靠 flush 时 `getattr(ex, ...)`
  **活读**兜底；spec 构造期冻结 → raw 直通判死 → dev 项永不产出结果。
- **修复**：置位提前到驱动开头（会话启动前）——驱动本就知道自己是谁，
  这是正确的语义顺序；该标志唯一消费方就是会话。
- **F-4/F-5 同类**：金标两次抓住"活读属性被冻结后时序暴露"的回归，
  证明 spec 化方向正确（把隐式时序契约变成显式顺序契约）。

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

