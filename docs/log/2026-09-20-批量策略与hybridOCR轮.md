# 2026-09-20 批量策略与 hybrid OCR 轮：pool 退役 + 双车道 v0 裁决

> 任务（用户）：①宏伟设想分析——decode 统一到 hybrid+FORCE_SIDE、OCR-bound
> 场景引入双 OCR 车道；②裁决后执行：decode 不动，从取消 pool 开始，
> batch_test 5 集标清剧集（stride=1，预期 OCR-bound）实测 hybrid OCR。

## 1. 设想分析（先量后做）

### 1.1 decode 统一（hybrid+FORCE_SIDE 替代纯臂）——裁决：不做

`tools/_probe_unify_decode.py`（fork 纯解码口径，子进程自调用因 FORCE_SIDE
为进程级 static 缓存，交错 ×3）：

| 对照 | 输出等价（[5000,6000) md5） | 速率 | ctor |
|---|---|---|---|
| hybrid+FORCE_CPU vs cpu（h264/hevc） | **位级一致** | +1.0% / −0.1% | 128~139ms vs 11~25ms |
| hybrid+FORCE_GPU vs gpu（三码） | **位级一致** | −0.6~−0.8% | 平价 |
| hybrid+FORCE_CPU vs cpu（**av1**） | — | **退化 GPU**（1707≈1715fps，纯 cpu 1264） | — |

机制近零成本（h264/hevc 位级一致+速率平价），三个洞否决统一：
①av1×cpu 语义不存在（非 IDR 边界的纯 CPU 分片无法按序交付，fork 刻意
退化，实测确认）——补洞=open-GOP 保序改造（C-57 同级专项）；②无 GPU
机器失去 cpu 后端（hybrid ctor 即初始化 NVDEC）；③每 reader +~110ms
冷税（短窗批量线性损失）。维护账：真正的双管线成本在 host/GPU（OCR 侧），
decode 消费侧早已单一化——统一 decode 省得少、风险集中多（hybrid 缺陷域
从三选一变全量命中）。纯臂保留定位= fallback+基线+无 GPU 支持。

### 1.2 双 OCR 容量账（既有数字推导）

TRT 容量 ≈1913 段/s（hevc-hybrid infer 忙时 4.36s/8340 段）、OV ≈1028
（C-48/C-47 比值 1.86×）、双车道和 2941（**+54%**）；当前 GPU 路径需求
≈890 段/s（decode-bound，L3 空等）——**OCR-bound 场景当前不存在**，
触发器=解码再快 ~2×（多 NVDEC 单元）或 OCR 需求 ~2×（模型/密度/ROI）。
触发后收益上界=OCR 相位 −35%。

## 2. pool vs hybrid 串行（三臂交错，上问实验）

`tools/_probe_pool_vs_serial.py`（同进程引擎池共享、臂序轮转、同轮配对、
三码族全片 ×5 热轮）：

| 臂 | 中位 | vs 串行 | 逐轮配对 |
|---|---:|---:|---|
| serial-hybrid | 28.61s | 基线 | — |
| pool-pair（h264→cpu，hevc/av1→nvdec） | 29.34s | +2.6% | +1.61%，快 1/5 |
| pool-hybrid（2 并发 hybrid reader） | 29.69s | +3.8% | +2.29%，快 1/5 |

段数 3×8340 全对（含并发 hybrid reader 正确性——P4 修复后的首个真实并发
实证）。裁决：**跨视频并发无优势**——C-52 的 −20.6% 基线是全-nvdec 派工
时代；单 NVDEC 上并发只增争用（hevc/av1 的 NVDEC 双会话重叠窗撞 C-01，
h264→cpu 单文件本就比 hybrid 慢 25%）。**pool 0.15 废弃**（run 发
DeprecationWarning，README 批量章改「逐文件 hybrid 顺序跑」示例，
MIGRATION 登记，0.16 删除）。

## 3. hybrid OCR v0：机制成立、架构自败、根因已量化

### 3.1 OCR-bound 前提实测（新三国01，696×424 h264，~73.5k 帧，stride=1）

ROI=[144,398,551,423]（batch_params.txt 口径）：segs=26461（稠密字幕条
stride=1 → 段数≈帧数/2.8），`ocr.infer` 忙时 **37.3s / wall 40.7s = 92%**
——OCR-bound 确认（decode 80% 忙）。TRT 此负载 22.6ms/批 16 ≈ 1.41ms/段。

### 3.2 v0 实现（`ocr_backend="hybrid"`，opt-in）

复用双实例 ONNX 的多引擎骨架：acquire_engines 取 TRT+OV 各一（同池 key
参数），worker 每引擎一线程共享 infer_q（自然工作窃取=库存派工），结果按
idx 收敛。双引擎 → raw_ready 自动 False → GPU 管线走宿主 crop 回退
（ONNX 回退既有机制）。TRT 不可用时双 OV 仍成立（等价双实例，降级透出）。

### 3.3 实测（`tools/_probe_hybrid_ocr.py`，整集交错 ×2）

| 臂 | wall | infer Σ | decode Σ | preprocess Σ |
|---|---:|---:|---:|---:|
| 单 TRT | 38.2 / 38.6s | 35.9 / 36.6 | 31.1 / 32.2 | 0.1 |
| 双车道 | 53.5 / 62.6s | 61.4 / 63.6 | 18.0 / 16.7 | **52.0 / 61.3** |

**+51% 回归**。正确性：段数恒等 26461；文本仅 **3 段不一致**
（seg7605-07「捧/撵」形近字，两车道读法之差，真值未裁）。

### 3.4 根因分解（30k 帧子相位 + 变体）

| 变体 | wall | preprocess | luma | crop | resize |
|---|---:|---:|---:|---:|---:|
| dual+默认 yuv | 19.7s | 17.7 | 0.7 | 1.8 | **15.2** |
| dual+gray_rep ×2 | 18.1 / 19.1 | 17.1 / 17.9 | 0.0 | 2.0 | 15.0 / 15.8 |

凶手=**宿主 numpy resize 1.37ms/段（prep 的 86%）**：v0 为双引擎放弃 raw
设备直通后，串行 worker 的宿主预处理成为新瓶颈（11099 段 × 1.6ms ≈
worker 98% 忙，两条 infer 车道在后面挨饿——infer Σ 24s 只兑出 1.3× 并行）。
luma 无辜（0.7s），gray rep 无效（resize 主导）。

### 3.5 v0.5 设计（留档，未实施）

共享设备预处理：raw_ready 放宽为「有 TRT 引擎即可」（GPU 管线继续交付
设备帧）→ flush 对设备项跑一次 `_gpu_pre.process_gray_raw`（现有 kernel）
→ TRT 车道直接吃设备张量；OV 车道 D2H **已预处理**张量（224×48≈10.7KB/
帧）+ OV 增加 prepped-host 入口（跳过内部 prep）。改动面：native.py 拆
`_gpu_raw_submit` 的 prep/infer 耦合 + ocr_stage flush 分流 + 流/生存期
（prep 环形缓冲的 D2H 事件同步——§16/§17 同族雷区，需专属门禁轮）。
天花板：OCR 相位 →~13s 后墙钟由 decode 主导（~31s busy）≈ **−15~20%**。

## 4. v0.5 实施与实测（设备 prep 共享，同日晚段）

### 4.1 实现

- **OV 引擎**：`OcrEngine.call_prepped(batch_nchw)`——吃已预处理
  (B,3,48,W) float32 张量，跳过内部 `_resize_norm`（v0 回归根因）；
- **车道 prep**：`ocr_lane_prep(eng)` 给 OV 引擎挂**独立 stream** 的
  GpuPreprocessor（TRT 车道复用引擎自身流）——独立流是双车道不串行的
  前提；`checkin_ocr_engine` 归还前 `release_lane_prep` 清理；
- **flush 分流**：`_dual` 判定（2 引擎 = {tensorrt, openvino} 且
  gpu_pipeline_mode）时设备帧也进 raw 通道；
- **raw_ready 语义放宽**：双车道同样置位（设备帧由共享 prep 消费，
  两侧都不需宿主代表帧）——首版遗漏此点导致两条车道全落宿主路径；
- **D2H**：pinned 缓冲 + 同流异步拷贝 + 事件同步（首版用同步
  `cudaMemcpy` = 隐式设备级同步，每批打断 TRT 在途推理，两车道
  infer 忙时 36.9→69.2s）。

### 4.2 实测（整集 26461 段，交错 ×2）

| 臂 | wall | infer Σ | decode Σ | preprocess Σ |
|---|---:|---:|---:|---:|
| 单 TRT | 38.6 / 38.7s | 35.9 / 36.8 | 31.3 / 32.3 | 0.1 |
| 双车道 v0.5 | 37.4 / 38.5s | 66.5 / 68.4 | 32.5 / 33.9 | **0.1** |

**中位 −1.8%**（v0 是 +48%）——回归消除，但**未达预期收益**。正确性：
段数恒等 26461；文本差异仍仅 3 段（同 v0 的「捧/撵」形近字）。

### 4.3 为何没吃到容量（逐车道实测）

| 车道 | 批数 | 批均 | 段均 |
|---|---:|---:|---:|
| TRT（双车道下） | 1176 | 27.06ms | 1.69ms |
| OV（双车道下） | 470 | 64.64ms | 4.04ms |
| TRT 单臂基线 | — | 22.6ms | 1.41ms |
| OV 单臂基线（cpu 路径） | — | — | 7.08ms |

容量账：理论 +54%（TRT 1913 + OV 1028 段/s）→ 实测 **+18%**
（591 + 248 = 839 段/s vs 708）。三个衰减因子：

1. **OV 车道产能只有 TRT 的 1/3**（4.04 vs 1.69ms/段）——工作窃取下
   OV 线程每批耗时长，自然少拿批（470 vs 1176）；容量和的短板效应；
2. **TRT 车道被争用 +20%**（22.6→27.06ms/批）——OV 的 CPU 推理与
   TRT 车道的宿主侧 enqueue/CTC 抢核；
3. **decode 侧同时被挤压**（decode Σ 31.3→32.5~33.9s）——hybrid 解码
   的 CPU 臂与 OV 车道抢同一批物理核（本机 16C32T 已被解码 + 双 OCR
   三层占满）。

净效果：OCR 相位容量提升被 decode 相位劣化对冲，墙钟只落 1.8%。
**触发条件重写**：双 OCR 的收益前提是「OCR 相位容量成为唯一瓶颈且
CPU 侧有余量」——本机 OCR-bound 负载下 CPU 侧没有余量（hybrid 解码
的 CPU 臂已占满），故收益被抹平；在「NVDEC 纯解码（CPU 空闲）+ OCR
需求高」的部署下才会兑现（此时 v0.5 的机制可直接复用）。

## 4b. 交付清单

- pool：`run()` DeprecationWarning + 模块 docstring + README 批量章重写 +
  MIGRATION 0.15 条目 + 钉子（warning/空输入）。
- hybrid OCR：`ocr_backend="hybrid"` 全链（校验/引擎类型/双擎获取/混合
  标签/降级透出）+ 钉子 5 条 + README 参数注记（标实验性/勿生产）。
- 探针 ×3 入库：`_probe_pool_vs_serial.py`、`_probe_unify_decode.py`、
  `_probe_hybrid_ocr.py`。

## 复评触发

- 双 OCR v0.5：OCR-bound 负载要收益必先落设备 prep 共享；落地后用
  `_probe_hybrid_ocr.py` 复测（期望 wall → decode 地板 ~31s）。
- pool 删除：0.16（两版本惯例）。
- 「捧/撵」3 段差异真值未裁——下游若用双车道，注意两引擎读法差。

## 证据

- `tools/_probe_pool_vs_serial.py`（§2）、`tools/_probe_unify_decode.py`（§1.1）、
  `tools/_probe_hybrid_ocr.py`（§3.3）
- batch_test 素材：`D:\Videos\batch_test`（batch_params.txt ROI/步长口径）
- 相关结论：C-53（追加批量注记）、C-52（→superseded）、C-58（hybrid OCR v0）
