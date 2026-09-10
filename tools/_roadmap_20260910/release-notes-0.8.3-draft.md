# decord v0.8.3 Release Notes 草稿（本地预备，未发布）

> 位置：审核后作为 GitHub Release 正文；fork 本地提交 486d2f6（版本号 bump）。

## Highlights

- **修复**：VideoReader 析构顺序 use-after-free（73e5540）——hybrid_gpu 输出帧
  的 Deleter 可能摸到已析构的 GPU 缓冲池，av1 hybrid close 偶发/必现
  access violation。ASAN 下 24 reader 序列零内存错误。
- **性能**：hybrid chunk 预路由（3b96c6f）——双侧速率就绪即按能力比一次性
  规划全部 chunk，混跑从交替转并发。av1 6000 帧 decode-only 中位
  2146→2765 fps（vs 理想和损耗 29.5%→8.2%），全片 23970 帧三份额配置
  与 NVDEC 逐位一致。
- **性能**：hybrid GPU 池深按 ROI 帧字节重算（d94d92e）——ROI 模式下池深
  虚小 ~570× 导致 GPU 侧长期断粮；现随 SetRoi 重算（clamp 8192 帧）。
  附实验性 `DECORD_HYBRID_FORCE_SHARE=<0..1>` 静态分账 governor（默认关）。
- **修复**：rebuild_dev.bat FFMPEG_DIR 指向修正（93a5ce1）。

## 升级注意

- 含 0.8.2 全部内容（FFmpeg 9 / avcodec-63 硬性要求不变；cuMemcpy2D_v2
  修复在 wheel 内）。
- hybrid/hybrid_gpu ctx 仍标记 experimental（调度/性能/内存边界可能随版本
  变化，Python 侧 import 时有 UserWarning）。
- Python 版本支持：3.9–3.14（三段式流水线自动构建）。

## 验证清单（发布前）

- [ ] `tools/_probe_hybrid_bitwise.py` 全片逐位一致（引擎仓，三编码）
- [ ] 引擎 sha 门禁：`tools/_probe_perf_baseline.py` 双跑文本集一致
- [ ] av1 hybrid close 压测（此前必现崩溃的路径）×24
- [ ] pip wheel 三编码解码速率与 build-081fix 一致（±5%）
