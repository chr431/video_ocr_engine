# 性能提升路线图分析（2026-08-28 实测）

本文回答"还能怎么快"，按**已验证收益 → 中等难度 → 底层依赖变更 → C/C++ 重写**
排序。所有数字均为本次在本机（7945HX 16C32T + RTX 4060 Laptop，
decord fork 0.7.12 / onnxruntime 1.29 / TRT）**单跑 A/B 实测**，
不是估算。测量脚本见 `tools/_probe_*.py`。

> ⚠️ **本文修正了 `docs/PERFORMANCE.md` 的三条"已锁定"结论**。
> 那三条是在 **OCR 跑 CPU（ONNX）占满全部物理核** 的时代得出的；
> TRT 成为默认后 CPU 核是空闲的，前提变了，结论失效。详见 §1.1、§1.3。

---

## 0. TL;DR

| 项 | 改动量 | 实测收益 | 状态 |
|---|---|---|---|
| **P0-1** 解码线程数随核数缩放（现役默认上限 8 → 16~32） | 1 个常量 / 1 个 env | **h264 全片 -40%、字幕整集 -50%** | ✅ 已验证 |
| **P0-2** host 输入的 TRT 批也走 GPU argmax 归约 | ~6 行（复用现成 `execute_device_argmax`） | **-8%~-14%**（OCR 侧 -39%），结果逐位一致 | ✅ 已验证 |
| **P0-3** `auto` 后端按「编码 + 核数」选择 | 1 个判定函数 | 让 P0-1 的收益自动生效 | ✅ 数据已备 |
| P1-1 真跳帧解码（`stride>1` 目前**零**解码收益） | 中（改 fork 的 C++） | 潜在 **2~4×**（仅 stride>1） | 未验证 |
| P1-2 hybrid 泛化为 K 路 GOP 并行 | 中 | 1080p h264 已接近 CPU 单路，边际 | 部分验证 |
| P1-3 解耦 GPU OCR 管线与 NVDEC | 中 | 让 P0-1 与零拷贝 OCR 叠加 | 未验证 |
| P2 底层依赖变更（PyAV / 自写解码层） | 大 | 每帧 ~0.15ms 固定开销，小 ROI 有量级空间 | 未验证 |
| P3 **C/C++ 全量重写** | 极大 | **收益上限仅 3%~6% —— 不推荐** | ✅ 已量化否决 |
| P3' **定点下沉**（`_cluster_win3` + 状态机） | 中 | 现有 ROI 3.4%~5.6%；**大 ROI（≥10 万像素）10×** | 已量化 |

**一句话**：先做 P0（一天工作量，h264 场景直接快一倍），再做 P1-1；
**不要做全量 C 重写**——把整条链翻译成 C++ 只能拿回 3~6%，
真正值得下沉的只有 `_cluster_win3` 一个函数，而且只在大 ROI 时才划算。

---

## 1. 重新测量：墙钟到底去哪了

### 1.1 「decode 已到 NVDEC 硬件上限」——只在 NVDEC 路径下成立

现有文档的核心判断是"decode 占墙钟 92~98%，且 NVDEC 已到硬件上限，
所以只能在 OCR 侧抠"。前半句对，**后半句把"NVDEC 的上限"当成了"解码的上限"**。

同一台机器、同一视频（test5，1080p h264，ROI 33×106）：

| 路径 | 吞吐 | ms/帧 |
|---|---|---|
| decord **NVDEC**（ROI gray） | 966 fps | 1.035 |
| ffmpeg **NVDEC** 裸解码 | 845~872 fps | 1.15~1.18 |
| decord **CPU 软解**（现役默认线程） | 1311 fps | 0.763 |
| **ffmpeg CPU 软解 `-threads auto`** | **1938 fps** | 0.516 |
| **decord CPU 软解 @16 线程** | **2247 fps** | 0.445 |
| **decord CPU 软解 @24 线程** | **2632 fps** | 0.380 |

两个结论：

1. **NVDEC 确实到顶了**（decord 甚至比裸 ffmpeg 还快 10%，ROI-first 已经把
   转换开销吃干净了）——在 NVDEC 路径上再优化确实没空间，这条旧结论保留。
2. **但 CPU 软解远没有到顶**：现役默认线程下只有 1311 fps，而 ffmpeg 同机
   可以跑 1938 fps，decord 开到 24 线程能到 **2632 fps（+101%）**。

### 1.2 根因：decord 的 CPU 解码线程数被钉在 8

`decord/src/video/video_reader.cc:237`：

```cpp
if (kDLCPU == ctx_.device_type) {
    dec_ctx->thread_count = nb_thread_decoding_ > 0
        ? nb_thread_decoding_
        : DECORD_FFMPEG_THREAD_COUNT;      // ← 引擎不传时走这里
}
```

`DECORD_FFMPEG_THREAD_COUNT = clamp(hardware_concurrency()/4, 2, 8)`
→ 32 线程机上 = **8**，且可用 env `DECORD_FFMPEG_THREAD_COUNT` 覆盖。

而引擎 `_decode_num_threads()` 在物理核 > 8 时**返回 `None`**（codimension 注释：
"16 核分核反而差"）。于是 CPU 解码长期跑在 8 线程。

**那条"16 核分核反而差"的实验是 `ocr_threads` 与 `decode_threads` 联合调整的**
（8 核实测 `ocrT=4/dcd=4`），它证明的是"把 OCR 线程砍半不划算"，
**不是"解码线程不该多加"**。TRT 默认后 OCR 不吃 CPU 核，这个约束消失了。

实测（端到端，`uniq` = 唯一文本数，用于校验一致性）：

**test5 全片（7223 帧，stride=1）**

| 配置 | 墙钟 | 相对 | 段数 | uniq |
|---|---:|---:|---:|---:|
| NVDEC+TRT（现役默认） | 8.112s | 100% | 2492 | 315 |
| CPU+TRT 默认 8 线程 | 6.452s | 79.5% | 2492 | 315 |
| **CPU+TRT dcdT=16** | **4.875s** | **60.1%** | 2492 | 315 |
| CPU+TRT dcdT=24 | 4.995s | 61.6% | 2492 | 315 |
| CPU+TRT dcdT=32 | 5.085s | 62.7% | 2492 | 315 |
| **CPU+TRT dcdT=16 + P0-2** | **4.466s** | **55.0%** | 2492 | 315 |

**新三国01 整集（73430 源帧，stride=8，标清 696×424）**

| 配置 | 墙钟 | 相对 | 段数 | uniq |
|---|---:|---:|---:|---:|
| NVDEC+TRT（现役默认） | 21.785s | 100% | 1151 | 573 |
| CPU+TRT 默认 8 线程 | 15.897s | 73.0% | 1151 | 573 |
| CPU+TRT dcdT=16 | 11.500s | 52.8% | 1151 | 573 |
| CPU+TRT dcdT=24 | 11.149s | 51.2% | 1151 | 573 |
| **CPU+TRT dcdT=32** | **10.812s** | **49.6%** | 1151 | 573 |

段数 / 唯一文本 / 代表帧在所有配置下**完全一致**，是纯性能差异。

### 1.3 编码与核数的交叉点（决定 P0-3 怎么判）

| 片源 | 编码 | NVDEC+TRT | CPU+TRT 最优 | 结论 |
|---|---|---:|---:|---|
| test5 全片 | h264 1080p | 8.112s | **4.466s** | CPU 大胜 |
| 新三国01 整集 | h264 标清 | 21.785s | **10.812s** | CPU 大胜 |
| test.mp4 | HEVC | **2.101s** | 2.720s | NVDEC 仍优 |
| test6 | AV1 | **2.449s** | 5.865s | NVDEC 大优（CPU 加线程无效） |

**AV1 的 CPU 软解完全不随 FFmpeg 线程数扩展**（8/16/24/32 线程全是 5.8~5.9s），
dav1d 有自己的线程池。这与 fork 里 AV1 特判 `max_frame_delay` 的注释一致。

**弱 CPU 敏感性**（`psutil.cpu_affinity` 绑到前 8 个逻辑核模拟 8 核机）：

| 片源 | NVDEC+TRT | CPU+TRT dcdT=8 | CPU+TRT dcdT=24 |
|---|---:|---:|---:|
| test5 全片 | 8.064s | 7.974s | **7.581s（-6%）** |
| 新三国01 整集 | 21.630s | 17.157s | **14.106s（-35%）** |

**关键：提高解码线程数在任何核数下都不劣化**（最差也是并列最优）。
风险不在"线程给多了"，而在"h264 上该选 CPU 还是 NVDEC"——8 核时 1080p 两者基本打平，
标清仍是 CPU 明显更好。

### 1.4 `sample_stride>1` 对解码是零收益（被低估的结构性问题）

同一视频、同样解码到 `frames` 列表：

| 采样 | 采样帧数 | 耗时 | 有效 fps |
|---|---:|---:|---:|
| stride=1 | 3000 | 3.152s | 952 |
| **stride=8** | **375** | **3.098s** | **121** |

`GetBatch` 等差步长快速路径只是**少交付帧**，中间的帧照样全部解码
（必须顺序解码到采样点）。所以 stride 省下的是**分段与 OCR**，不是解码。
在整集字幕场景（decode 8.9s / 墙钟 10.8s）里，解码仍占 82%——
**这条才是真正卡住 stride 场景上限的地方**，见 P1-1。

### 1.5 Python 分段层的真实成本（C 重写的决策输入）

合成帧跑与 `_host_segment_frames` 完全相同的逐帧逻辑，剥离解码与 OCR
（已按真实切段率生成，非最坏情况）：

| ROI | 面积 | 逐帧成本 | `_cluster_win3` 占比 | 理论 fps 上限 |
|---|---:|---:|---:|---:|
| 106×33（速度数字） | 3.5k px | **34.5 µs** | 16.1 µs（47%） | 29k |
| 407×25（字幕条） | 10k px | **40.6 µs** | 33.3 µs（82%） | 25k |
| 800×200 | 160k px | **1208 µs** | 614 µs（51%） | **828** |
| 1600×600 | 960k px | **6838 µs** | 3575 µs（52%） | **146** |

固定开销约 **25 µs/帧**（循环 + yield + std + 二值化 + 相邻比较），
其余与 ROI **面积线性相关**，`_cluster_win3` 占一半左右。

换算成"占当前优化后墙钟的比例"：

- test5 全片：0.250s / 4.466s = **5.6%**
- 新三国01 整集：0.365s / 10.812s = **3.4%**

→ 现有 ROI 下，把整条链翻译成 C++ **最多拿回 3~6%**。
→ 但 ROI ≥ 10 万像素时，Python 分段本身就是 828 fps 的硬天花板，**此时 C/CUDA 下沉是 10× 级收益**（这也正好印证了 `PERFORMANCE.md` §9 里"GPU 分段只在 ROI ≥10 万像素时可能有净收益"的推测——现在有数据了）。

---

## 2. P0：已验证收益，建议立即落地

### P0-1 解码线程数随核数缩放

**改动**：`FieldExtractor._decode_num_threads()` 不再在核数多时返回 `None`；
改为按后端分档（OCR 在 GPU 时给解码更多核）：

```python
# 建议（示意，阈值需用 §1.3 数据在你的目标机型上复核）
# 逻辑核 hw，物理核 pc：
#   OCR 在 GPU（TRT）→ dcdT = min(32, max(16, hw))          # 16/32 均可，16 更保守
#   OCR 在 CPU（ONNX）→ dcdT = max(4, pc // 2)              # 保留现役分核逻辑
#   AV1 + CPU          → dcdT = max(2, pc // 2)              # 加线程无效，别浪费
```

**收益**：h264 全片 -40%、字幕整集 -50%（§1.2）。
**风险**：低。§1.3 已验证 8 核机上高线程不劣化。唯一注意点是 ONNX 后端要保留分核。
**不需要改 decord**：`num_threads` 显式传入即覆盖 fork 默认。

### P0-2 host 输入的 TRT 批也走 GPU argmax 归约

**现状**：`OcrEngine._call_trt_gpu` 走 `GpuPreprocessor.process()` 后输入已在显存，
但仍调 `_infer_trt_device` → **DtoH 整批 `(B,S,18710)` float32**
（B=16 / S≈80 时 ≈ **95 MB/批**）。只有 `call_gpu_raw`（NVDEC 直通路径）
走 `execute_device_argmax`（DtoH 仅 ~12 KB，约 1300×）。

**改动**：`ocr_native._call_trt_gpu` 增加与 `call_gpu_raw` 相同的分支：

```python
if getattr(self, "_gpu_ctc_mode", False):
    idx2d, prob2d = self._trt.execute_device_argmax(dev_ptr, shape)
    return self._ctc_from_idxprob(idx2d, prob2d)
```

`execute_device_argmax` 已实现且已在生产路径使用，这里只是复用。

**收益**（原型实测）：

| 场景 | 现役 | 原型 | Δ | 一致性 |
|---|---:|---:|---:|---|
| test5 3000 帧 | 2.619s | **2.248s** | **-14.2%** | 前 40 段文本+置信度逐位一致 |
| test5 全片 | 4.874s | **4.466s** | **-8.4%** | 逐位一致 |
| 新三国01 整集 | 7.057s | 7.041s | -0.2% | 逐位一致（`infer` 3.179→1.954s，**-39%**） |

整集那条墙钟没变，是因为**解码仍是主项、OCR 完全被掩盖**（`q_get_wait` 4.9s）——
这正好说明它对 CPU 解码路径是净赚、对解码受限场景零风险。

### P0-3 `auto` 后端按「编码 + 核数」选择

现役 `auto` = 优先 NVDEC。按 §1.3 应改为：

```
codec == h264 且 逻辑核 >= 16            → cpu（配 P0-1 的高线程）
codec == h264 且 8 <= 逻辑核 < 16        → 分辨率判定：<=720p 用 cpu，1080p+ 用 nvdec
codec in (hevc, av1) 或 核数 < 8         → nvdec
显式指定 cpu/nvdec/hybrid                → 尊重用户
```

`hybrid` 在 h264 上仍略优于纯 CPU（decode 1.95s vs 2.15s），但墙钟基本持平
（2.64s vs 2.61s），因为它引入了校准开销与 OCR 尾批；**P0-1 落地后 hybrid 的
边际价值变小**，建议维持实验态，不必急着转正。

---

## 3. P1：中等难度（需要改 decord fork 或引擎结构）

### P1-1 真跳帧解码 —— 当前最大的单点机会

**问题**：§1.4 已证实 `stride>1` 只省分段/OCR，不省解码。
整集字幕场景 decode 8.9s / 墙钟 10.8s，**解码占了 82%**，
而其中至少 7/8 的帧是"解码出来就扔掉"的。

**已有失败记录**（`PERFORMANCE.md` §6，两次尝试均封板）：
手动按 `pict_type` 丢 B 帧 packet、或动态切 `AVDISCARD_NONREF`，
都报 `missing picture in access unit` 并对不上 `seek_accurate` 真值。
根因记录得很准：H.264 High profile 下部分 B 帧本身是参考帧。

**尚未尝试的正确做法**（建议下次走这条路，不要再手动过滤 packet）：

1. **用 FFmpeg 原生的 `AVCodecContext.skip_frame = AVDISCARD_NONREF`**，
   让解码器自己跳过非参考帧的重建（它维护 DPB，不会破坏参考关系），
   而不是在 decord 的 `PushNextFiltered` 里丢 packet。
   这是 FFmpeg 快进/缩略图场景的标准用法，语义由解码器保证。
2. **与 stride 结合的关键约束**：只能在"该帧既非参考帧、也不是采样点"时跳过。
   需要在推送 packet 前知道帧类型 → 可以用**滑动窗口**：
   维护一个深度 = stride 的 packet 队列，只对"确定不需要且不可能是参考帧"的
   packet 置位。做不到严格判断时，退化为"`skip_frame` 常开 + 采样点若被跳过则
   回退到该 GOP 起点重解"——回退概率低（B 帧占比高时偶发）。
3. **保守门控**：只在 `sample_stride >= 4` 且 `codec == h264` 时启用，
   结果必须能通过对齐校验（`tools/e2e_smoke.py --verify` 比 `seek_accurate` 真值）。

**收益量级**：h264 典型 GOP 中非参考 B 帧占 60~75%，
stride=8 时可跳过的帧约 6/8 → **解码有望 2~4×**，
即整集场景 10.8s → 约 4~6s。这是本文所有未验证项里潜力最大的一个。

**风险**：正确性。必须逐帧对齐校验，且默认关闭、env 开关保护。

### P1-2 hybrid 泛化为 K 路 GOP 并行

现役 hybrid 固定两路（1×NVDEC + 1×CPU，动态分界 + 连续扫掠 + 对称接管）。
实测在 h264 上 decode 1.95s 已是全场最低，但墙钟与纯 CPU@16 持平（2.64 vs 2.61s），
因为它额外付出校准（~0.25s）与更大的 OCR 尾批（0.24s vs 0.24s）。

**若要继续投入**，方向是把 2 路泛化为 K 路（K 个 CPU 解码实例按 GOP 分片 +
1 路 NVDEC 兜底），在**多路 CPU 实例**上扩展——但 §1.1 显示单路 CPU@24 线程
已经跑到 2632 fps（1080p），多实例的边际收益有限，**优先级低于 P1-1**。

### P1-3 解耦 GPU OCR 管线与 NVDEC

现役零拷贝管线（`_gpu_pipeline.py`）的门控要求 `decode ∈ {auto, nvdec}`，
因为 raw OCR 需要 decord 的 GPU NDArray 设备指针。
**CPU 解码拿不到设备指针 → 只能走宿主 OCR**（`_preprocess_standard` 在 CPU + DtoH logits）。

而 §1.2 表明 CPU 解码在 h264 上更快，于是出现"快的解码"与"快的 OCR"互斥。

**解法**：把代表帧批量 H2D 后接入现有 `GpuPreprocessor` + `execute_device_argmax`。
代表帧只占采样帧的 12~34%（整集 1151 / 9178），H2D 流量可忽略。
这条路能同时拿到 P0-1 的解码收益与零拷贝管线的 OCR 收益。
P0-2 是这条路的**前半段**（先消掉 DtoH），后半段是 preprocess 上 GPU。

---

## 4. P2：底层依赖变更

### P2-1 decord fork 改动（性价比高于换依赖）

fork 本身是对的（ROI-first 是全局最大的单点优化：test5 上 526→966 fps，+84%）。
建议的改动按性价比排序：

1. **`DECORD_FFMPEG_THREAD_COUNT` 上限从 8 放宽**（P0-1 的 fork 侧配合）——
   引擎显式传 `num_threads` 即可，不必改 fork。
2. **去掉 ROI 输出路径上的每帧同步**（P2-2 详述）。
3. **`get_batch` 支持"仅解码不取回"**：现役已天然支持（§1.1 测过 `asnumpy`
   与否无差异），说明输出侧已无浪费。

### P2-2 自写 C++ 解码层（替换/绕过 decord）

**动机**：§1.1 的两点拟合显示存在约 **0.15 ms/帧的固定开销**，
与分辨率、输出尺寸、批大小、是否取像素**全部无关**：

| 扫描项 | test5 (1080p, NVDEC) | 结论 |
|---|---|---|
| 批大小 B=1 → 1024 | 969 → 965 fps | 无影响（非调用开销） |
| ROI 33×106 → 1077×87 | 965 → 965 fps | 无影响（非输出开销） |
| 是否 `asnumpy()` | 957 → 966 fps | 无影响（非 D2H 开销） |
| `next_roi` vs `get_batch` | 990 vs 966 fps | 无影响 |

即：**每帧 ~1.03 ms 全是解码器内的成本**，其中约 0.15 ms 是与像素无关的固定部分
（最可能是 cuvid `MapVideoFrame`/`UnmapVideoFrame` 的逐帧同步 + NVDEC
逐图提交延迟）。这部分在标清视频上占到 53%（新三国01：0.148/0.277 ms）。

**若自写 C++ 解码层**（FFmpeg C API + CUVID，或直接用 `cuvid` + NvDecoder 样例）：

- 可做的：多帧在 flight（批量 decode、延迟 map）、用 CUDA event 取代逐帧同步、
  输出直接写入引擎自己的池、ROI crop 在 kernel 里做（fork 已做）、
  `sample_stride` 的跳过逻辑直接内建（= P1-1）。
- 代价：需要维护 FFmpeg 版本适配（现役 fork 有 `build-ff7`/`build-ff9` 多套构建，
  说明这块本来就脆）、Windows DLL 分发、PyInstaller 打包。
- **收益上限估算**：0.15 ms/帧的固定开销在 1080p 上占 14%、标清上占 53%。
  若全部消除，标清场景约 -35%（≈ P0-1 已拿到的量级），1080p 约 -10%。

**评估**：**优先级低于 P1-1**。理由：P1-1 拿到的是 2~4×，
而 P2-2 费尽力气只有 10~35%，且维护成本高一个数量级。
建议**先做 P1-1**（它必须动 fork 的 C++，正好顺带验证自写层的必要性）。

### P2-3 解码后端替换评估

| 方案 | 优点 | 致命问题 |
|---|---|---|
| **PyAV**（cffi 绑定 FFmpeg） | 官方 FFmpeg、API 完整、好维护 | **无 ROI-first**：全帧解码 + 全帧转换，test5 实测 966→526 fps（-46%）。除非自己接 `hwdownload` 前的 crop/scale filter，但那又回到写 C++ |
| **torchaudio / nvdecode 绑定** | 有 GPU 解码 | 依赖 PyTorch（+2GB），且同样无 ROI-first |
| **OpenCV cv::cudacodec::VideoReader** | 简单 | 无 ROI-first，且 seek/格式控制弱 |
| **自写 pybind11 解码扩展** | 完全可控 | 见 P2-2，成本大 |

**结论：不换。** ROI-first 是本项目最大的单点收益，目前只有自建 fork 提供。
所有"更标准"的替代方案都会先亏掉 46%。

### P2-4 OCR 侧依赖

- **onnxruntime 保留**：无 TRT 环境的兜底路径，已是 CPU 上的最优配置
  （双实例 + 批 16，`PERFORMANCE.md` §3/§5 已锁定）。
- **不要引入 OpenVINO / DirectML**：OCR 在 TRT 可用时不是瓶颈
  （GPU 路径 ~1041 段/s），瓶颈在解码。
- **值得做的**：P0-2（消 DtoH）+ P1-3（preprocess 上 GPU）。

---

## 5. P3：C/C++ 重写评估（用户明确问到的部分）

### 5.1 结论：全量重写不划算，收益上限 3~6%

由 §1.5 的实测直接得出：

| 场景 | Python 分段层总成本 | 优化后墙钟 | **上限占比** |
|---|---:|---:|---:|
| test5 全片 | 0.250s | 4.466s | **5.6%** |
| 新三国01 整集 | 0.365s | 10.812s | **3.4%** |

把 `_host_frame_stream` + `_host_segment_frames` + `segmentation.py` +
`video_utils` 的灰度/预处理全部改成 C++/pybind11，**理论上最多拿回这 3~6%**，
而且要付出：GIL/线程模型重做、与 numpy 的零拷贝边界、Windows 构建与打包、
四个平台的 wheel、以及"逐位一致"的回归风险（本仓所有优化都以逐位一致为准入门槛）。

**投入产出比不成立。** 旧文档"decode 占 92~98%"如果按字面理解，
会得出"Python 只剩几个百分点"——这个数字反而说明**别重写**。

### 5.2 值得做的是"定点下沉"，只有一个函数

`_cluster_win3` 占 Python 分段成本的 **47%~82%**（§1.5），
且它是纯粹的热点：每帧一次、6 次全帧切片加法、`O(面积)`、无分支。

| ROI | 现成本 | 下沉后（估） | 场景收益 |
|---|---:|---:|---|
| 106×33 | 16.1 µs | ~2 µs | 全片 **-2.7%** |
| 407×25 | 33.3 µs | ~3 µs | 整集 **-2.8%** |
| **800×200** | **614 µs** | ~10 µs | **-60%（828 fps → ~1万 fps）** |
| **1600×600** | **3575 µs** | ~50 µs | **-70%（146 fps → ~2000 fps）** |

**判断标准**：ROI 面积 < 5 万像素 → **不值得**（收益 < 3%）；
ROI ≥ 10 万像素（如整屏字幕、大区域 OCR）→ **值得，且是 10× 级**。

若要做，优先用 **CUDA kernel 而非 C++**：`_gpu_kernels.py` 里的
`GpuFrameAnalyzer` 已经有 analyze kernel 的雏形，接上去比新写一个
C++ 扩展的维护成本低，且大 ROI 场景本来就有 GPU。
`PERFORMANCE.md` §9 的推测（"GPU 分段只在 ROI ≥10 万像素时可能有净收益，
无实测先例，未立项"）现在有了量化支撑——**10 万像素这个分界线是对的**。

### 5.3 如果一定要重写，重写哪里

按 ROI 排序，只重写这一段（约 150 行 C++/CUDA）：

```
for each frame:  gray/std → binarize(th) → prev_xor → cluster_win3 → 边界判定
                 └─────────── 全部合并为一个 kernel / 一个 C++ 循环 ──────────┘
```

保留在 Python 的：解码调度、段管理、OCR 批处理、结果组装
（这些是每**段**一次，不是每**帧**一次，频率低 3~30 倍）。

---

## 6. 建议执行顺序

| 序 | 项 | 工作量 | 累计预期（h264 整集） |
|---|---|---|---|
| 1 | **P0-1** 解码线程数 | 半天 | 21.8s → 10.8s |
| 2 | **P0-2** GPU argmax（host 输入） | 半天 | 10.8s → ~10.5s |
| 3 | **P0-3** auto 后端判定 | 半天 | 收益自动生效，无需用户调参 |
| 4 | 回归：`tools/e2e_smoke.py --verify` 全矩阵 | 半天 | — |
| 5 | **P1-1** 真跳帧解码（改 fork） | 3~5 天 | → 4~6s（未验证） |
| 6 | **P1-3** CPU 解码 + GPU OCR 解耦 | 2~3 天 | 叠加零拷贝 OCR |
| 7 | P2-2 自写 C++ 解码层 | 2 周+ | 先做完 5 再评估 |
| — | ~~P3 全量 C 重写~~ | — | **不做**（上限 3~6%） |

每完成一项，按仓库约定把结论追加到 `docs/PERFORMANCE.md`
（含失败项，避免重复投入），并同步修正本文 §0 的表格。

---

## 7. 附录：测量方法

| 脚本 | 用途 |
|---|---|
| `tools/_probe_ceiling.py` | 纯解码吞吐矩阵（后端 × ROI × stride） |
| `tools/_probe_perframe.py` | 每帧固定开销拆解（批大小/ROI/取像素扫描） |
| `tools/_probe_ffmpeg.py` | 系统 ffmpeg 对照（NVDEC / CPU 线程扫描） |
| `tools/_probe_threads.py` | decord CPU 解码线程数扫描 |
| `tools/_probe_e2e_ab.py` | 端到端后端 × 线程 A/B（含分相剖面） |
| `tools/_probe_hybrid_ab.py` | hybrid × 线程 A/B |
| `tools/_probe_gpu_ctc.py` | P0-2 原型验证（monkeypatch，不改生产代码） |
| `tools/_probe_final.py` | 全片/整集收口 + `--affinity N` 弱 CPU 模拟 |
| `tools/_probe_python_cost.py` | Python 逐帧成本（C 重写决策输入） |

> 以上均为本次分析用的临时探针（`_` 前缀），可按需保留或删除。
> 若要长期保留，建议合并为一个 `tools/bench_decode_threads.py` 回归工具。

**方法学要求**（与 `PERFORMANCE.md` §1 一致，本次全部遵守）：
A/B 单跑串行、短窗口快迭代 + 全片复核、`ENGINE_PROFILE=1` 取分相、
每段数/唯一文本/代表帧一致性作为正确性门槛。
