# video_ocr_engine 设计/使用逻辑问题分析报告

> **范围**：静态代码阅读 + 文档（README / PERFORMANCE / PERFORMANCE-ROADMAP /
> ARCHIVE / DEPENDENCIES / engine_config 注释）对照；**未运行代码、未跑真机**。
> 性能类结论引用仓库自带实测记录，未复验。
> **产出**：只做分析报告（用户 2026-06 确认），不修改代码。
>
> **二轮复核（2026-08-30）**：一轮全部条目对当前代码（v0.9.2，commit 6aed92f）
> 逐条复核——行号已刷新核对，仅 C3 的细节描述有误已修正（见该条）；本轮通读
> 全部核心模块后**新增 10 条**（A6 / B5 / C5-C10 / D7-D9），并重排优先级表。
> **四组覆盖**：API/参数设计、架构/模块边界、正确性/资源与并发风险、使用体验/文档。
> 证据位置均为当前仓库文件路径 + 行号。

**总体判断**：性能导向极强、工程纪律很好的代码（"逐位一致"门禁、真值验证、
实验结论归档），但"性能优先"的决策在 **API 层与使用层付出系统性代价**：
三套参数入口互相遮蔽、库 import 即改全局环境、接口冒充、注释即契约、
文档分裂成五份。二轮复核补充两个系统性主题：

1. **近三轮功能变更已造成成片文档漂移**（stride 解禁、§8.3 hybrid×GPU 管线
   合并、pad 224 回退）：README / engine_config / DEPENDENCIES 中至少 5 处
   现役描述与代码矛盾，其中一处是同页自相矛盾（README 环境变量表列着
   0.9.0 已删除的钩子）。该仓库的"结论归档"纪律覆盖了 docs/，没覆盖 README。
2. **单发 extract + 无显式资源生命周期** 与 README 官方推荐的"长进程
   多视频批量"用法存在设计缺口：OCR 引擎/内核/设备缓冲跨视频零复用、
   显存只增不减、错误路径泄漏线程——单次提取都没问题，长进程累积。

---

## A 组：API / 参数 / 默认值设计

### A1. 同一调参旋钮有多套入口，优先级不一致且文档没写全

- **证据**：
  - `OCR_PAD_SMALL`（env）**压过**构造参数 `fill_width`：`ocr_native.py:324-331`
    （注释里明说"env 是调参覆盖语义，必须能盖过构造参数"）。
  - `TEXT_SEP_MERGE`（env）**压过** `merge_text_sep`（构造）：`extractor.py:185-196`。
  - `OCR_GAMMA`（env）是**唯一**入口：构造层无参数承接，只在
    `video_utils.py:182-183` 被读。
  - 另有一批**仅 env 单入口**的旋钮无构造参数承接：
    `OCR_ROI_AUTOCROP / _MARGIN / _MIN_GAIN`、`OCR_REORDER_WINDOW`
    （`extractor.py:132-145`）、`OCR_INSTANCES`（`_host_pipeline.py:392-394`）、
    `GPU_PIPELINE`、全部 `HYBRID_*`（`extractor.py:337-339`）。
- **影响**：README 声明"构造参数已覆盖绝大多数用法；环境变量仅在批量调优/
  诊断时使用"（`README.md:186`），但用户构造参数设 `fill_width=320` 时可能被
  残留的 `OCR_PAD_SMALL` 静默覆盖；autocrop 全家只能靠 env——
  **调优现场无法判断哪个入口在生效，也无法给单实例单独设 autocrop**。
- **建议**：README 增加"优先级表"（env > 构造参数 > 常量）与"仅 env 入口"清单；
  或构造时打印生效值；autocrop 旋钮提升为构造参数。

### A2. `decode_backend="auto"` 在用户最常见场景（h264 多核）不是最优

- **证据**：README 自己写"强多核 CPU + h264 可手动选 cpu 获得更高吞吐"
  （`README.md:74-77`）；实测 h264 上 CPU+TRT 比 NVDEC+TRT 快约 2×
  （test5 全片 8.112s vs 4.466s，`PERFORMANCE-ROADMAP.md:148-153`）；
  `auto` 默认"优先 NVDEC"（`engine_config.py:176`）。P0-3"自动选型"已决策不做
  （`PERFORMANCE-ROADMAP.md:305-314`，理由：静态判据不可靠且判错代价成倍）。
- **影响**：默认用户拿不到最优性能；README 没解释"为什么默认不这么做"
  （决策记录在开发文档，用户不可见）。
- **建议**：README 补"为什么 auto 不自动选 CPU"段（已有部分，可补全 P0-3 理由）。

### A3. `frame_end=0` 的兼容语义只活在代码注释里

- **证据**：验证逻辑允许 `frame_end=0`（`extractor.py:179-183`，
  `tests/test_extractor_params.py:26`）；运行语义 = 到末尾（`0 or total`，
  `extractor.py:534, 240`）；但 README 只说 `frame_end=None` 表示到末尾
  （`README.md:54`）。
- **影响**：用户不知道 0 是合法值；维护者不知道 None/0 是双入口。
- **建议**：README 补一句；或纳入下次破坏版本清理（删掉 0 兼容）。

### A4. 超界 `frame_end` 被静默截断而非报错

- **证据**：`end = min(self._frame_end or total, total)`
  （`extractor.py:534, 240`）；构造校验只查 `frame_end > frame_start`
  （`extractor.py:179-183`），不查视频总长。
- **影响**：用户传 `frame_end=999999` 静默处理到片尾，无法发现参数错误
  （对照：`frame_start` 超界会报"帧区间为空"，start/end 语义不对称）。
- **建议**：打开解码器后校验并 `ValueError`（或至少 warning + 文档声明截断语义）。

### A5. `force_aspect` 与 `fill_width` 强耦合，构造期无任何提示

- **证据**：`engine_config.py:188-200` 注释明说"取值依赖 force_aspect：
  >0 时越大越准，=0 时偏小更佳"；2026-08-29 刚因此翻过一次车
  （160→224 回退，`engine_config.py:226-252`）。
- **影响**：只调 `fill_width` 不调 `force_aspect` 的用户可能拿到相反方向的次优值；
  正确组合完全靠注释传承。
- **建议**：`fill_width` 与 `force_aspect` 组合异常时告警；或把组合写进构造文档必读。

### A6.【新增】同类 env 的读取时机不一致：有的构造期一次性"烘焙"，有的每次调用读取

- **证据**：
  - **构造期烘焙**（构造后改 env 不生效）：`OCR_ROI_AUTOCROP / _MARGIN /
    _MIN_GAIN / OCR_REORDER_WINDOW` 在 `__init__` 读入
    （`extractor.py:132-145`）。
  - **调用期读取**（构造后改 env 生效）：`OCR_PAD_SMALL` 每批读
    （`ocr_native.py:324`）、`OCR_GAMMA` 每次预处理读（`video_utils.py:182-183`）、
    `OCR_BATCH` 每批读（`_helpers.py:10-11`）、`TEXT_SEP_MERGE` 每次合并判定读
    （`extractor.py:189-191`）、`GPU_PIPELINE` 每次 extract 读
    （`_gpu_pipeline.py:180`）、`OCR_THREADS` 每次 OCR 会话读
    （`extractor.py:498`）。
- **影响**：同是"env 覆盖"，行为不同：`FieldExtractor(...)` 构造完再
  `os.environ["OCR_ROI_AUTOCROP"]="0"` 无效，改 `OCR_PAD_SMALL` 却有效。
  没有任何文档说明哪些 env 在什么时刻生效——调参现场最容易踩的时间陷阱。
- **建议**：统一为一种时机（建议全部调用期读取，与"env 是调参覆盖"语义一致），
  或在 README env 表中加"生效时机"一列。

---

## B 组：架构 / 模块边界

### B1. `extractor.py` 同时是骨架、解码器工厂、线程预算中心

- **证据**：
  - `_open_vr`（`extractor.py:270-347`）：后端选路 / NVDEC 回退 / hybrid 包装 /
    AV1 特判 / num_threads 计算。
  - `_decode_num_threads`（`extractor.py:369-425`）与 `_ocr_num_threads`
    （`extractor.py:487-506`）**反向 import** `ocr_native.auto_ocr_thread_count`——
    引擎核心依赖 OCR 模块的线程策略。
- **影响**：接入新解码后端/新编码时改动集中在引擎核心文件，回归面大。
- **建议**：拆 `DecoderFactory` / `ThreadBudget` 独立模块（仓库已有按逻辑拆分先例）。

### B2. 顶层模块与包存在双向依赖

- **证据**：`video_ocr_engine` 包 import 顶层（`engine_config` / `segmentation` /
  `video_utils` / `ocr_native` / `ocr_trt`）；`ocr_trt.py:20` 又 re-export
  `video_ocr_engine._gpu_kernels`——依赖图成环（能加载，因 `_gpu_kernels`
  无反向依赖）。
- **影响**：pip 安装（`py-modules` 平铺）与 submodule/sys.path 挂载
  （README 推荐的两种方式，`README.md:15-29`）对顶层模块可用性要求不同，
  wheel 布局一变就碎；公共 helper（如 `nv12_to_rgb`）不在包导出面，
  用户必须 `from video_utils import nv12_to_rgb`（顶层模块）。
- **建议**：长期把全部模块收进包内（`video_ocr_engine.*`），顶层只留兼容 shim；
  或明确只支持一种安装方式。

### B3. 宿主与 GPU 两条流水线重复实现，一致性靠测试而非单一实现

- **证据**：校准（`_host_calibrate` vs `histograms_perframe+_otsu_from_hist`）、
  分段状态机（`_host_segment_frames` vs `_run_pipelined_gpu` 主循环，
  `extractor.py:569-591` vs `_gpu_pipeline.py:642-696`）、
  merge 判定（`_segments_similar` vs `_similar_device`）各写一份。
- **影响**：
  - 任何语义修正都要改两处；
  - `_gpu_pipeline.py:576-582` 残留 `contrast` 分支——`contrast` 已在 0.9.0 删除，
    `_merge_effective_mode()` 只会返回 binary/''（`extractor.py:185-196`），
    **该分支是死代码**（`_CpuFrameRef` 与 `_similar_device` 的 docstring
    仍在描述 contrast 流量，注释随代码过期）。
- **建议**：删死分支（低风险）；中期把分段状态机收敛为共享纯函数（GPU 侧只替换
  "每帧标量来源"）。

### B4. `HybridDecoder.seek_accurate` 是空操作，但对调用方伪装成可用接口

- **证据**：`hybrid_decode.py:663-665` 直接 `return`；注释解释"分片定位由
  生产者在片首完成"。当前调用方恰好安全（主流程 `frame_start>0` 的 seek 被吞掉后，
  生产者自己从 `fr[0]` 起解，`extractor.py:536`）。
- **影响**：任何未来新增的"seek 后读取"调用方都会静默拿到错误帧序——
  接口签名存在但语义被掏空（经典"接口冒充"）。
- **建议**：改名（如 `_noop_seek`）或抛 `NotImplementedError` 并让调用方显式跳过。

### B5.【新增】OCR 引擎 / 内核 / 分析器跨视频零复用，且复用入口半途而废

- **证据**：
  - 公共 `extract()` → `_run_pipelined()`（`extractor.py:229, 508-512`），
    每次走 `_start_ocr_session` **新建 OcrEngine**（`_host_pipeline.py:397-399`）
    → TRT 反序列化 + context + 设备缓冲（`ocr_native.py:137`）。
  - 宿主路径留了私有复用参数 `_run_pipelined_host(_ocr_engines)`
    （`extractor.py:514`），但 (1) 公共 `extract()` 不透传；
    (2) **GPU 管线完全丢弃该参数**——`_run_pipelined_gpu(self)` 签名不收
    （`_gpu_pipeline.py:203`），内部硬编码 `_start_ocr_session(None)`
    （`_gpu_pipeline.py:257`），分发处也没传（`extractor.py:511`）。
  - GPU 管线每次 extract 还新建 `GpuFrameAnalyzer`（`_gpu_pipeline.py:266`）、
    每个新 OcrEngine 懒建 `GpuPreprocessor`（`ocr_native.py:354-360`）——
    两者的 `__init__` 都做 **NVRTC 现场编译 cubin**（`_gpu_kernels.py:96-113,
    591-610`），无任何跨实例缓存。
- **影响**：README 官方推荐长进程批量（多线程多实例，`README.md:126-146`；
  上层 GUI 应用 RaceVideoToLog 同为长进程）：每个视频都要重复付
  TRT 反序列化 + 2~3 次 NVRTC 编译 + 全套设备缓冲分配的固定成本，
  且直接加剧 C5 的显存只增不减。
- **建议**：短期补齐"复用入口断头"（GPU 管线接上 `_ocr_engines` 参数）；
  中期提供引擎级复用 API（进程级 OCR 引擎池 / 模块级 cubin 缓存），
  让 `extract()` 默认复用而非每次重建。

---

## C 组：正确性 / 资源与并发风险

### C1. `nvdec_available` 的 `lru_cache` 会缓存瞬态失败

- **证据**：`video_utils.py:212-230`——异常路径 `return False` 是正常返回，
  **会被 lru_cache（maxsize=64）缓存**；`_gpu_pipeline_enabled` 用其结果做门控
  （`_gpu_pipeline.py:201`）。
- **影响**：进程内第一次探测遇瞬态失败（驱动忙/显存压力/并发探测），该视频的
  GPU 管线判定被永久钉死为 False；注释声称"进程内稳定"对**稳定失败**成立，
  对**瞬态失败**不成立。另：`nvdec_available(video_path)` 每次对**新视频**
  都会真实打开一次 reader（见 C10：叠加后最坏开 3 次）。
- **建议**：缓存成功结果、不缓存失败；失败时每次重探。

### C2. GPU 缓冲池的 GC 回收是"注释即契约"，无类型/编译期约束

- **证据**：`_YFrame.__del__` / `_DevBatch.__del__` 引用归零即归还池
  （`_gpu_pipeline.py:47-51, 103-107`），安全前提是"raw OCR 与 sim_pair
  同步读完才可能归零"（`_gpu_pipeline.py:89-93`）；
  `_gpu_kernels.py:707-711` 的 `extract_luma` 缓冲复用同理（"同一批内安全；
  跨批重新调用会覆盖"）。
- **影响**：任何新调用方在异步路径上持有引用即可能悬垂；这类契约散落多处，
  全靠注释传承。
- **建议**：至少集中写一份"设备内存生命周期契约"文档；长期把池对象改为显式
  `release()` 而非 `__del__`（与 C5 同一改造）。

### C3. 宿主路径 yuv 模式的批量灰度缓冲复用是失效代码（实际性能缺陷）

- **证据**：`_host_pipeline.py:105-115`——
  - 初值：yuv 时 `g_buf` 形状算成 `(B, rows, W*2//3)`（把 2//3 乘在**宽度**上；
    NV12 的 Y 平面实际是 `(rows*2//3, W)`）——初值本身就是错的；
  - 复用条件：`g_buf.shape[1:]`（yuv 时两元 `(rows, W*2//3)`）与一元
    `(rows*2//3,)` 比较——**两元 vs 一元永不相等**，yuv 模式永远走
    `_batch_luma` 重新分配；gray 模式比较 `(H, W)==(H, W)` 才正常复用。
- **影响**：默认 `rep_crop_format="yuv"`（`engine_config.py:168`）+ 宿主管线
  （无 GPU / ocr=cpu 场景）下，每批解码都新建灰度数组——注释声称的优化
  实际没生效（正确性无影响，性能/内存有损失，且是"注释与代码不符"的维护陷阱）。
- **建议**：yuv 分支初值改为 `(B, rows*2//3, W)`、复用条件改为与
  `(crops.shape[1]*2//3, crops.shape[2])` 比较，并加一个"复用确实发生"的
  单元测试（现测试只测正确性，测不出"优化失效"）。

### C4. OCR 引擎初始化/推理错误延迟到 `extract()` 末尾才抛，且丢失线程上下文

- **证据**：OCR worker 内异常 append 到 `ocr_err`（`_host_pipeline.py:465-466,
  598-599`），producer 侧 `_put_ocr` 检测后 raise（`_host_pipeline.py:368-376`），
  主流程末尾 `if ocr_err: raise ocr_err[0]`（`extractor.py:601, 707`）。
- **影响**：模型缺失 / TRT 构建失败等用户可修复问题，表现为"extract 跑了一段后
  抛个裸异常"（同一个异常对象跨线程被重复 raise，原始 traceback 与抛出点混在一起），
  没有"OCR worker 失败"的上下文，也没有降级原因。
- **建议**：异常携带来源标记（如 `raise RuntimeError("OCR worker 失败") from e`），
  或走结构化错误类型。

### C5.【新增】设备内存"只增不减"，无任何显式释放路径——长进程批量显存单调增长

- **证据**：
  - 池只回收不释放：`_YFramePool/_DevBatchPool._release` 在池未满时**只 append**，
    满了才 `cudaFree`（`_gpu_pipeline.py:76-84, 148-156`）；extract() 返回后
    池对象连同 `_free` 列表被 GC，`_YFrame.__del__` 只会往垂死的池里 append——
    **已入池的设备块没有任何代码路径 cudaFree**，直到进程退出/CUDA context 销毁。
  - TRT 侧：`TrtEngine._dev_in/_dev_out` 按需增长、无 `__del__`、无 `close()`
    （`ocr_trt.py:99-107, 220-235`）；每 extract 新建 OcrEngine（B5）→
    每视频一套，旧实例 GC 只回收 Python 对象，`cudaMalloc` 的块不动。
  - 内核侧：`GpuPreprocessor/GpuFrameAnalyzer/GpuOutputReducer` 的
    `_ensure_*` 全部只增不缩、无释放（`_gpu_kernels.py:129-179, 621-628`）。
- **影响**：单次提取无感；README 推荐的长进程批量/上层 GUI（B5 场景）下，
  每视频累积数 MB~数十 MB 设备内存（TRT 输出缓冲随输入宽增长，
  2048 宽时可达 ~100MB 量级），显存单调增长直到进程退出。
  引擎没有任何 `close()/release()`，用户想还也还不掉。
- **建议**：给 `FieldExtractor`/OCR 会话补显式生命周期（`close()` 语义或
  上下文管理器），池与 TRT 缓冲在其内统一释放；这与 C2 的"显式 release"
  是同一件事的两面，应一起设计。

### C6.【新增】GPU 管线错误路径上 producer 线程无取消机制——线程与解码器泄漏

- **证据**：`_producer` 循环 `producer_q.put(item)` **无超时、无停止事件**
  （`_gpu_pipeline.py:491-498`）；消费端任何异常（OCR worker 死亡经 `_put_ocr`
  抛出 `_gpu_pipeline.py:635, 706-707`、`_similar_device`/`_d2h_rep` 的 CUDA
  错误等）都会让主线程跳出消费循环，producer 随即**永久阻塞在 put**。
- **影响**：daemon 线程不阻止进程退出，但长进程里每次失败的 extract 泄漏一个
  阻塞线程 + 它引用的 decord GPU reader/NDArray/设备指针；批量场景失败累积。
  宿主管线解码在主线程，无此问题——两条管线错误处理不对称。
- **建议**：producer 加停止事件 + `put(timeout)` 轮询停止位；消费端 finally
  中 set 停止位（与 `HybridDecoder.close()` 已有的 `_stop` 机制对齐）。

### C7.【新增】`cancel_check` 在阻塞点失效——取消延迟无上界

- **证据**：
  - 取消检查只在分段主循环每 100 帧一查（`_host_pipeline.py:205-206`、
    `_gpu_pipeline.py:679-680`）。
  - 两个阻塞点不查取消：`_put` 在 OCR 队列满时无限循环
    （`_host_pipeline.py:368-376`，只查 `ocr_err`）；OCR 引擎初始化期间
    完全没有取消点（`_host_pipeline.py:387-401`，TRT 首次构建可达分钟级）。
  - GPU 管线消费端阻塞在 `producer_q.get()` 时同样不查（`_gpu_pipeline.py:644`）。
- **影响**：OCR 跟不上（慢盘/宽 ROI/大模型构建）时，用户取消请求要等
  当前阻塞解除才生效，最坏等整个引擎构建/队列排空——取消语义"名义存在、
  实际无界"。
- **建议**：`_put` 循环内加 `cancel_check` 检查（抛 `CancelledError` 类）；
  引擎构建的 progress 回调路径顺带检查取消。

### C8.【新增】非 ROI decord 的兼容路径是半成品：只有校准做 ROI 回退切片，主流程静默用整帧

- **证据**：引擎显式兼容无 ROI API 的旧 decord（`extractor.py:283-289`
  `_has_roi_api` 探测，失败则 `roi_kw={}` 全帧输出）；此时——
  - 校准做了兜底：`_host_calibrate` 里 `_crop_is_expected` 不符则手工切片
    （`_host_pipeline.py:60-61, 72-73`）；
  - **主帧流没做**：`_host_frame_stream` 直接 `yield crops[k]`
    （`_host_pipeline.py:101-115, 129-133`），无任何 ROI 检查；
  - GPU 管线只查通道数不查 ROI 尺寸（`_gpu_pipeline.py:275-288, 296-309`）。
- **影响**：装了非 fork decord 的用户（README 虽声明 fork 必需，但兼容代码
  存在即构成可走路径）：分段/OCR 在**整帧**上进行、`roi` 参数被静默忽略、
  校准阈值（ROI 域）与分段数据（整帧域）尺寸错位、merge 因形状不齐恒 False
  ——产出的是"看起来正常"的错误结果，比直接报错糟糕得多。
- **建议**：短期在 `_host_frame_stream`/GPU 校准处补 `_crop_is_expected`
  检查 + 切片兜底（与校准一致）；或干脆检测到无 ROI API 时构造期显式报错
  （README 已声明 fork 必需，半兼容比不兼容更危险）。

### C9.【新增】`nv12_to_rgb` 不接收 color_range：full-range 流的 RGB 预览会偏色，docstring 自相矛盾

- **证据**：`video_utils.py:62-105`——函数恒按 limited 矩阵展开 Y
  （`(y-16)/255`，`video_utils.py:75`），无 color_range 参数；内部灰度路径
  `_nv12_luma_full` 却是按流 range 正确展开的（`video_utils.py:41-51`）。
  docstring 前后矛盾：第 65-66 行说"Y 已由 decoder 按流 range 展开"，
  第 73 行注释又说"decord yuv420 的 Y/U/V 均为原始 8-bit"。
- **影响**：`get_color_range()==1`（full/pc）的片源，`nv12_to_rgb(seg.rep_crop)`
  会再做一次 `-16` 展开 → 预览发灰/偏色（仅影响预览，不影响识别链，
  识别链 Y 展开走 `_nv12_luma_full` 是对的）。
- **建议**：函数加 `color_range: int = 0` 参数（默认值保持现行为），
  修 docstring；`ExtractedSegment` 无法携带 range 信息，可在文档注明
  "默认 limited 语义"。

### C10.【新增】GPU→宿主回退路径重复打开视频，最坏同一视频开 3 次

- **证据**：`_run_pipelined_gpu` 在形状不符时 `return self._run_pipelined_host(None)`
  共 5 处（`_gpu_pipeline.py:276, 286, 299, 304, 309`）；宿主管线随后**再次**
  `_open_vr()`（`extractor.py:529`），而第一次打开的 vr 未关闭即弃
  （普通 decord VR 无 close，只能等 GC）；叠加门控探测
  `nvdec_available` 的真实打开（`video_utils.py:226`），同一视频最坏
  探测 + GPU 开 + 回退重开 = 3 次。
- **影响**：回退是异常路径（旧 fork / 形状不符），但每次都是实打实的
  打开成本 + 解码器句柄悬挂；`decode_backend="nvdec"` 显式指定的用户
  回退时还会拿不到任何"已回退"的结构化信息（D3）。
- **建议**：回退时把已打开的 vr 传给宿主路径复用，或先 close；至少在
  `meta` 里记录回退原因（联动 D3）。

---

## D 组：使用体验 / 文档

### D1. `import` 即改全局环境：`DECORD_SKIP_LOOP_FILTER` 的 setdefault 在 `__init__.py` 顶层

- **证据**：`video_ocr_engine/__init__.py:26`；该开关**改变输出像素**
  （无去块平滑、rep_crop 有块状伪影，README 自己承认，`README.md:202`）。
  同类第二处：`ocr_native.py:26` 的 `OMP_WAIT_POLICY` setdefault
  （影响同进程其他 ONNX/OMP 使用方，程度较轻）。
- **影响**：对"通用文本提取库"（CLAUDE.md 定位）而言，import 库会静默改变
  **同进程内其他 decord 使用方**的解码输出——库的隐式全局副作用，
  与"零领域语义/纯文本输出"定位冲突。
- **建议**：改为构造参数/显式 env（缺省不设），或至少文档加粗"进程级副作用"。

### D2. 进度回调永远不会到 100%

- **证据**：`_decode_progress_pct` 输出 [3,58]、`_ocr_progress_pct` 输出 [58,86]
  （`_helpers.py:17-24`），引擎初始化进度固定 2.5（`_host_pipeline.py:389`）。
- **影响**：GUI 进度条到 86% 后停住；README 未说明"进度上限 86%"。
- **建议**：文档注明口径，或把末尾 14% 分配给"收尾/结果组装"。

### D3. `ExtractionResult.meta` 不含本次实际参数、引擎版本，降级原因也不可见

- **证据**：`meta` 只有 backend/ocr_backend/codec/n_segments
  （`extractor.py:247-250`）——而 `_result_types.py:33` 的注释承诺
  "backend/codec/**引擎版本**等"，版本从未写入；NVDEC/TRT 降级只写
  `logger.warning`（`extractor.py:301`、`ocr_native.py:139`），`auto` 的 GPU
  打开失败甚至静默（`extractor.py:293-302`）。
- **影响**：用户看到 `ocr_backend='onnxruntime'` 无法得知是"没装 TRT"还是
  "引擎加载失败"；批量并发后无法从结果反查配置与版本。
- **建议**：meta 增加 `params`（本次生效的 stride/fill_width/force_aspect/merge
  等）、`engine_version` 与 `degraded_reason`（可选字符串）。

### D4. `rep_crop` 默认是 NV12 二维数组，对非 CV 用户是隐性摩擦

- **证据**：默认 `rep_crop_format="yuv"`（`engine_config.py:168`），
  `ExtractedSegment.rep_crop` 文档只写"外部用 nv12_to_rgb 转 RGB"
  （`_result_types.py:20-22`）；且 helper 在顶层模块 `video_utils`
  而非包导出面（B2）。
- **影响**：`plt.imshow(seg.rep_crop)` 直接花屏；没有挂在结果对象上的
  开箱即用 helper。
- **建议**：`ExtractionResult` 增加 `rep_crop_rgb(seg)` 便捷方法或 `to_rgb`
  helper；文档加"我拿到的是什么形状"的示例。

### D5. `keep_frames=False` 同时清空段级帧号，文档未声明

- **证据**：`segments[i].frames` 变 `()`（`extractor.py:234`）、
  `result.frames` 变 `[]`（`extractor.py:244`）；README 只说"降低内存占用"
  （`README.md:118-119`）。
- **影响**：依赖"段→帧区间"的下游会静默拿到空序列。
- **建议**：README 补一句"段级 frames 一并丢弃，仅保留 start/end/rep_frame"。

### D6. 文档分裂成五份且互相引用，使用者/维护者看到两套真相

- **证据**：性能结论散落 `PERFORMANCE.md` / `PERFORMANCE-ROADMAP.md` /
  `ARCHIVE.md`，且互相"见 §x"；`ARCHIVE.md` 保留大量已删除功能的完整历史
  （对使用者是噪声），其"状态更新"注释自己也已过期（仍写 stride==1 门控，
  `ARCHIVE.md:14-20`）；CLAUDE.md 未在 README 中被引用。
- **影响**：同一结论（如 pad 下限 224、auto 后端、hybrid 门控）在多个文件有
  不同时代版本，容易踩"旧结论当现行"——本轮 D7/D8 的漂移正是这么发生的。
- **建议**：README 增加"文档地图"（哪份是现状、哪份是历史档案）；性能现状
  收敛到单文件，历史全部入 ARCHIVE。

### D7.【新增】README 环境变量表两处失实（含一处同页自相矛盾）

- **证据**：
  - `README.md:197`：`OCR_PAD_SMALL` 写"默认 160"——2026-08-29 已回退为
    224（`engine_config.py:196, 253`），README 未同步。
  - `README.md:194`：环境变量表仍在列 `GPU_PIPELINE_ASYNC`；而**同一份
    README 的 221-224 行**声明它已于 0.9.0 删除，engine_config 中也无此常量
    （代码 grep 为空）——同页自相矛盾的"幽灵旋钮"。
  - `docs/DEPENDENCIES.md`「decord」节仍在推荐 `DECORD_FORCE_CPU=1`——
    同样是 0.9.0 删除项（`README.md:221-225`）。
- **影响**：用户按表调参不生效还以为自己的用法错了；"幽灵钩子"特别损耗
  文档可信度。
- **建议**：修三处引用；顺带在 README 加一条 CI/grep 自检（env 表与
  `engine_config` 的 `*_ENV` 常量对账，防再漂移）。

### D8.【新增】hybrid 语义在三个文档处停在旧门控（stride==1 / 走宿主管线），与代码和测试矛盾

- **证据**：代码现状——stride>1 已解禁（`extractor.py:321-333` 注释明说
  "stride>1 已解禁"；`hybrid_decode.py:541` 步长取采样 stride）、
  hybrid 已并入 GPU 管线（`_gpu_pipeline.py:166-171` 门控含 hybrid；
  `tests/test_gpu_pipeline.py:82-86` 已有对应断言）。但：
  - `README.md:81`："要求 NVDEC 可用、`sample_stride==1`"——已不成立；
  - `README.md:160`："decode_backend∈{auto,nvdec}" + `README.md:176`：
    "`decode_backend="hybrid"` 走宿主"——§8.3 合并后均已不成立
    （README:178-182 只补了 cpu 分支，漏改这两处）；
  - `engine_config.py:88-92`：hybrid 注释仍写"仅 GPU(NVDEC) 可用、stride==1、
    未开 GPU 全驻留管线时生效"——三点全部过期。
- **影响**：用户按文档在 stride>1 时不敢用 hybrid、或以为 hybrid 必然绕开
  GPU 管线；engine_config 是"唯一事实源"（README:227-228），其注释失真
  危害最大。
- **建议**：三处同步；并注意这已是连续第二轮发现"功能变更只改代码 +
  PERFORMANCE/CLAUDE，不改 README/engine_config 注释"——建议把
  "README env 表 + engine_config 注释"列入功能变更的收尾清单。

### D9.【新增】`FieldExtractor` 是单发对象，但结果有双入口且带公开 setter，语义未声明

- **证据**：结果同时挂在返回值与实例上——`extract()` 返回
  `ExtractionResult`，同时写 `self.crops`（`extractor.py:605, 710`）、
  `self.timing`、`self._frames`；`frames` 还有**公开 setter**
  （`extractor.py:252-259`）；二次 `extract()` 会全量重算并覆盖所有实例态，
  无任何保护或说明。
- **影响**：用户把 `ex.crops` 与 `result.segments[i].rep_crop` 混用时，
  二次提取/外部赋值会让两份真相分叉；"跑一次就扔"的单发语义对
  批量复用实例的用户（B5 的目标人群）是隐形地雷。
- **建议**：文档声明单发语义；中期把实例属性降级为内部态（结果只从
  返回值取），`frames` setter 收进私有（先确认 RaceVideoToLog 无依赖）。

---

## 优先级建议

| 级别 | 条目 | 理由 |
|---|---|---|
| **P0（短期，低风险）** | D7/D8 文档失实与漂移（README env 表、hybrid 三处、DEPENDENCIES） | 纯文档修正，成本最低；D7 一处是同页自相矛盾，特别伤可信度 |
| | C3 g_buf yuv 复用失效 | 注释声称的优化实际没生效，修两行 + 补测试 |
| | A1 参数优先级/入口清单文档 | 用户可被残留 env 静默覆盖，先文档化 |
| | C1 nvdec_available 缓存瞬态失败 | 门控被错误钉死，改动小 |
| **P1（中期，设计）** | C5 显存无释放路径 + B5 引擎/内核跨视频复用 | 同一主题（长进程批量生命周期），需要一起设计 close()/复用 API |
| | C6 producer 线程取消 + C7 取消阻塞点 | 并发收口一批做；错误路径泄漏 + 取消无界 |
| | C8 非 ROI decord 静默整帧 | 先决策：补切片兜底，还是明确不支持并报错 |
| | D1 import 全局副作用（skip_loop_filter + OMP） | 需要产品决策（默认开 vs 显式 opt-in） |
| | B4 seek 空操作冒充 / B3 contrast 死分支 | 接口语义掏空；0.9.0 清理漏网 |
| | C10 回退双开视频 | 小改动，顺手做 |
| | B2 双向依赖 / B1 extractor 过重 | 结构性债务，随下一个破坏版本做 |
| **P2（打磨）** | A2-A6、C2、C4、C9、D2-D6、D9 | 体验/文档类，按需排期 |

---

## 未验证项（如实声明）

- C1 的"瞬态失败"概率、C3 的实测性能损失幅度、D3 的降级场景覆盖，
  均需真机/单元验证；本报告未运行任何代码。
- C5 的显存增长速率、C6 的泄漏累积速度，取决于批量规模与输入宽度，
  需长进程实测确认量级；机制本身由代码路径直接可证。
- C8 只影响非 fork decord 用户（README/DEPENDENCIES 已声明 fork 必需），
  实际是否存在这类用户未知；"兼容路径可走"由代码分支可证。
- C9 需要一个 full/pc range 的真实片源复验预览偏色。
- 所有性能结论引用仓库自带实测（`PERFORMANCE.md` / `PERFORMANCE-ROADMAP.md`），
  未在本机复现。

---

## 修复状态（2026-08-30 修复轮收口）

> 本轮按优先级 P0→P1→P2 落地，行为变更一处（D1，见下）；C3 复核时发现
> 一轮描述不完整（gray 分支同样失效），已一并修复。验证：单元测试 93 项
> 全过（含本轮新增 6 项）；真机 e2e 冒烟 7 配置矩阵全 PASS（test5 3000 帧
> stride8，真值匹配率 100%、跨配置唯一文本重合率 100%）；引擎池对象复用
> 与 5 轮提取显存稳定已实测；取消响应实测 3.3s（3s 截止）。

| 条目 | 状态 | 说明 |
|---|---|---|
| A1 参数优先级/入口 | ✅ 文档 | README 新增优先级（env > 构造 > 常量）与仅-env 入口清单 |
| A2 auto 不选 CPU | ✅ 文档 | README 补 P0-3 决策理由 |
| A3 frame_end=0 | ✅ 文档 | README 用法注释声明 0/None 双入口 |
| A4 超界截断 | ✅ 代码+文档 | 两条流水线超界 warning（保留截断语义，不破坏兼容） |
| A5 fill_width×force_aspect | ✅ 文档 | README 用法后加参数组合提示 |
| A6 env 读取时机 | ✅ 代码+文档 | autocrop/reorder 四旋钮改 property 调用期读 env，全表统一 |
| B1/B2 结构债务 | ⏸ 未动 | 需破坏版本，维持原排期 |
| B3 contrast 死分支 | ✅ 代码 | `_similar_device` 分支删除 + 三处注释清理 |
| B4 seek 冒充 | ✅ 代码 | `HybridDecoder.seek_accurate` 抛 `NotImplementedError`；两条流水线对 hybrid 跳过 seek |
| B5 引擎跨视频复用 | ✅ 代码 | GPU 管线接通 `_ocr_engines` 透传 + **进程级 OCR 引擎池**（ocr_native.acquire/checkin，key=(model,type,fill_width,threads)，每 key 上限 4）+ NVRTC 模块进程级缓存 |
| C1 探测缓存瞬态失败 | ✅ 代码+单测 | 成功才缓存（`_nvdec_probe_success` lru_cache，失败抛出不进缓存） |
| C2 注释即契约 | ✅ 由 C5 取代 | 显式 release 路径落地后，悬垂面收敛（池/TRT/内核均有 release） |
| C3 g_buf 复用失效 | ✅ 代码+单测 | **复核修正**：不只 yuv——gray 分支 `shape[:2]` 同样少一维，两种格式复用都从未生效；两分支均已修复并加"复用确实发生"单测 |
| C4 错误上下文丢失 | ✅ 代码 | OCR worker / GPU producer 异常统一 `raise RuntimeError(...) from e` |
| C5 设备内存无释放 | ✅ 代码 | `TrtEngine.release` / `GpuPreprocessor.release` / `GpuOutputReducer.release` / `GpuFrameAnalyzer.release` / 两池 `release_all`；GPU 管线 finally 统一释放（OCR 引擎归池常驻） |
| C6 producer 无取消 | ✅ 代码 | `producer_stop` Event + `put(timeout)` 轮询，finally 置位 |
| C7 取消阻塞点失效 | ✅ 代码 | 宿主 `_put` Full 分支 + GPU 消费端 `get(timeout)` 轮询均查 `cancel_check`（取消契约 = 回调抛异常，已在注释声明） |
| C8 非 ROI decord 静默整帧 | ✅ 代码 | 构造期 `_ensure_roi_capable_decoder`：无 `_CAPI_VideoReaderSetRoi` 直接 ValueError（decord 未安装不拦截，保持构造零依赖）；**用户决策：直接报错** |
| C9 nv12_to_rgb 色域 | ✅ 代码+单测 | 新增 `color_range` 参数（full/pc 矩阵），docstring 矛盾修正 |
| C10 回退双开视频 | ✅ 代码 | `_fallback_to_host`：普通 reader 复用（随机访问无消费状态）；hybrid 必须 close 后重开（分片消费指针已前进，复用会序错位——复核发现的隐藏约束） |
| D1 import 全局副作用 | ✅ 代码+文档 | **行为变更**：`setdefault` 移除，`DECORD_SKIP_LOOP_FILTER` 改显式 opt-in（默认不再关滤波，CPU 软解 HEVC/h264 回到完整去块滤波；e2e 复验真值匹配不受影响）。PERFORMANCE.md §12.5.1 加状态注 |
| D2 进度上限 86% | ✅ 文档 | README 结果节注明口径 |
| D3 meta 缺参数/版本/降级 | ✅ 代码 | meta 新增 `params`（本次生效参数）、`engine_version`、`degraded_reason`（NVDEC 回退/yuv 退化/hybrid 失败/形状回退/TRT→ONNX 五类）、`color_range`、`rep_crop_format` |
| D4 rep_crop 预览摩擦 | ✅ 代码+文档 | `ExtractionResult.rep_crop_rgb(seg)`（按 meta 自动选格式/色域） |
| D5 keep_frames 副作用 | ✅ 文档 | README 声明段级 frames 一并清空 |
| D6 文档分裂 | ✅ 文档 | README 新增文档地图（现状/历史/维护者分层） |
| D7 README env 表失实 | ✅ 文档 | `OCR_PAD_SMALL` 默认 160→224、`GPU_PIPELINE_ASYNC` 幽灵行删除、DEPENDENCIES `DECORD_FORCE_CPU` 清理 |
| D8 hybrid 三处旧门控 | ✅ 文档 | README（stride 解禁 + GPU 管线合并）、engine_config 注释、ARCHIVE 状态注同步 |
| D9 单发语义/结果双入口 | ✅ 文档 | README 声明单发语义与"以返回值为准"；`frames` setter 保留（上层应用兼容） |

**发布备注**：版本号未动（0.9.2）；D1 是唯一用户可见行为变更，建议随
下一个 tag 在变更日志中置顶。
