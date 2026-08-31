"""一次性工具：把 CLAUDE.md 拆成「注入核」+ docs/DECISIONS.md 历史档案。

设计要点
--------
1. 用 `newline=''` 读写（已由 tools/_probe_cr_roundtrip.py 实测 100% 保真），
   **不是二进制模式**。CLAUDE.md 有 955 个 CRLF 行尾、0 个行中裸 CR，
   所以这里保真压力不大，但仍统一用 newline='' 避免 os.linesep 干扰。
2. 切分点固定在 `## 通用约定` 这一行：其上的「文件头 + 状态块」被整份重写，
   其下 L32–L956 原文**逐字不动**迁到 docs/DECISIONS.md。
3. 迁出的原文保持 `###` 层级不变（不重编编号、不改标题），只在首尾加档案头尾。

用法
----
    python tools/_split_claude_md.py            # 预演，打印计划不写盘
    python tools/_split_claude_md.py --apply    # 真正执行
"""

from __future__ import annotations

import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "CLAUDE.md")
DST = os.path.join(ROOT, "docs", "DECISIONS.md")

# 切分锚点：CLAUDE.md 第 32 行
SPLIT_MARK = "\n## 通用约定\n"

CORE = r'''# CLAUDE.md — 开发记录与约定（注入核）

> 本文件在每个会话开头被注入，**只放"现在必须知道的"**。
> **硬上限 12 KB** — 超了就把内容迁到 `docs/DECISIONS.md`，这里只留指针。

## 文档地图（6 份，别再新增）

| 文件 | 性质 | 什么时候读 |
|---|---|---|
| `README.md` | 用户向 API / 用法 | 写调用代码时 |
| `CLAUDE.md`（本文件） | 维护者向**注入核**：铁律 + 现役架构 + 结论指针 | 自动注入 |
| `docs/PERFORMANCE.md` | 现役性能实测（§1–§15, §17, §19–§21） | 动性能相关代码前 |
| `docs/DECISIONS.md` | 每轮决策过程、已删除功能、设计审查结论 | 想问"为什么这么做"时 |
| `docs/ARCHIVE.md` | 归档（PERF §4 / §8 / §16 / §18），**编号保留勿重编** | 只看"为什么不做" |
| `docs/DEPENDENCIES.md` | 依赖版本与已知问题 | 装环境 / 报 bug 时 |

⚠️ **现役规则以本文件为准**。`docs/DECISIONS.md` 是迁出的原文存档，
两者冲突时以本文件为真相（避免"两套真相"，见设计审查 D6）。

## 铁律（先量后做）

1. **先量后做**：任何结论必须带实测数字，禁止凭直觉推断下结论。
2. **正确性门禁** = 段数 + 唯一文本集（或真值准确率），不是"看起来没问题"。
3. **判断 OCR 变好还是变坏，只测"文本有没有变"会得出错误结论** —— 必须
   **按帧对齐真值**测准确率；别用置信度当代理（`羸弱→赢弱` 是退化，
   置信度反而 0.9433→0.9700）。
4. **均值不能替代逐片检查**：5 片均值 ±0.01pp 曾用来支持"无负面影响"，
   第 6 片就翻了案。
5. **真值本身要抽查**：版本、剥零、哨兵、时间基准都可能错
   （见 DECISIONS「P0-6 翻案」）。
6. **别按"看起来旧"删脚本**：`tools/` 的探针是**证据链**，删之前先查引用
   （35/40 被文档引用，另有 5 处跨探针 import 与 69 处文档路径引用）。
7. **探针放 `tools/_probe_*.py`**（下划线前缀 = 调查工具，不随产品发布）。
8. **新结论一律追加到现役章节，不新建文档**；版本号改动必须打同名 git tag。
9. **向后兼容**：新功能默认关闭，除非明确作为新默认；新增遗留面一律先标
   deprecated、两个版本后删除。

## 现役架构

链路：**解码 → 像素分段 → 代表帧 → OCR → 相似段合并**

| 阶段 | 入口 |
|---|---|
| 解码 | `decord.VideoReader.get_batch` / `hybrid_decode.HybridDecoder.get_batch`（**唯一入口**） |
| 分段 | `segmentation.py` |
| 宿主管线 | `video_ocr_engine/_host_pipeline.py` |
| GPU 管线 | `video_ocr_engine/_gpu_pipeline.py`（gray+NVDEC+TRT 时默认） |
| OCR 调度 / 引擎池 | `ocr_native.py`（`acquire_ocr_engine` / `checkin_ocr_engine`） |
| TRT | `ocr_trt.py` + `video_ocr_engine/_gpu_kernels.py` |
| 配置常量 | `engine_config.py`（`GPU_PIPELINE_DECODE_BATCH=64` 等） |
| 分相打桩 | `video_ocr_engine/extractor.py:317 _prof_end` |

**现役并行维度只有一个**：`decode_backend="hybrid"` 的 CPU+NVDEC 双解码生产者
竞争（`hybrid_decode.py`）。**没有 dual pipeline、没有 `DUAL_*` 环境变量、
没有 `_dual_pipeline.py`** —— 历史提及均为旧档案，勿据此调优。

其他现役事实：

- **引擎内部恒为单通道灰度**：decord 只输出 `'yuv420'`（keep_crops 且
  `rep_crop_format="yuv"`）或 `'gray'`；旧 `gray_output`/`yuv_output` 已删除。
- **相似段合并**在分离图上进行，默认 `binary`（`TEXT_SEP_MERGE=binary|off`）。
- **GPU 管线门控**：gray + `decode∈{auto,nvdec,cpu}` + `ocr≠cpu` + NVDEC/TRT
  可用时自动启用；`GPU_PIPELINE=0` 关 / `=1` 强制。GPU+ONNX 组合实测无净收益。
- **OCR 引擎池**：`_POOL_MAX_PER_KEY=4`、`_POOL_MAX_TOTAL=16`，
  key=(model, type, fill_width, threads)。

## 已封板结论（勿重复投入）

| 结论 | 详见 |
|---|---|
| **IO 不是并发退化原因**：磁盘 IO 占单次墙钟 <1%（0.03–0.05s / 5–7s），PCIe 0.01% | PERF §19 |
| **内存带宽数字被高估 1.8×**：本机 B_max 只有 **55.8 GB/s**（非 ~100） | PERF §20 |
| **并发退化真因 = NVDEC 会话数**（单硬件单元串行）。互补设计（CPU 软解+ONNX ∥ NVDEC+TRT）聚合加速 **1.87×**；两条都走 NVDEC 只有 1.01~1.20× | PERF §21 |
| **解码后端必须按编码选**：h264 CPU 快 2.88×，AV1 CPU 慢 2.56×（7.4× 反转）；解码占管线 98%+ | PERF §21 |
| **冷启动 yuv 格式税不存在**，是测量假象；冷轮多出的 ~0.55s 里 0.50–0.53s 是 TRT 引擎构造 | PERF §15 |
| **分段合并**在不误合并约束下已无普适空间 | PERF §14 |
| **GPU 分段 + ONNX OCR 无净收益**，默认门控只放行 NVDEC+TRT | PERF §9 |
| **真跳帧**（丢 `nal_ref_idc==0` 整包）安全，但收益仅 1.03~1.48×（原估 2~4×） | DECISIONS「下一步三目标轮」 |
| **`skip_loop_filter`** 1.11~1.36×，但改变输出像素；默认关闭（opt-in） | DECISIONS「P0-6 翻案」 |
| **pad 224 保持**：160 已回退（生产误读退化），320 已证伪 | PERF §16.2 |
| **裁切余量 10% 优于 0%**；裁切即使省不到算力也能提准确率（旧"守卫"前提是错的） | DECISIONS「第四轮」 |
| **`auto` 后端在 h264 多核不是最优**（CPU+TRT 快约 2×），但静态判据不可靠、判错代价成倍 → 保持现状 | DECISIONS 设计审查 A2 |

## 编辑护栏（docs/PERFORMANCE.md）

该文件含 **934 个裸 `\r`**，其中 **916 个是行中间残留**、18 个是真 CRLF 行尾。

- **不需要二进制模式**。文本模式显式 `newline=''` 或 `newline='\n'` 即
  **100% 保真**（`tools/_probe_cr_roundtrip.py` 实测）。唯一致命的是
  **默认的 `newline=None`**（通用换行翻译，把 916 个行中 CR 全转成 `\n`，
  文件从 3450 行劈到 4366 行）。
- **Edit 工具是安全的**。它只做 CRLF→LF 规范化（丢那 18 个**行尾** CR），
  行中 916 个一个不丢；而 `.gitattributes` 就是 `* text=auto eol=lf`，
  那正是 git 期望的。旧记的"Edit 不安全"**是错的**。
- ⚠️ **git 把该文件判为二进制**（`git ls-files --eol` → `i/-text w/-text`），
  行中裸 CR 让 `text=auto` 自动检测失败 → `git diff` 对它只输出
  `Binary files differ`，**没法 review**，`eol=lf` 也不生效。
  清掉那 916 个行中 CR 即可改判回 text（已验证），但会产生一次性大 diff，
  **建议单独一轮做，别混在别的改动里**。
- 自检：`open(p,'rb').read().count(b'\r')`；拆档后
  PERFORMANCE.md + ARCHIVE.md 之和应为 **934**。

## 环境与命令

- Python：`c:/Users/eric chen/AppData/Local/Programs/Python/python313/python.exe`
  （**PATH 上的 `python` 缺 numpy，不是项目环境**）。pytest / numpy / psutil /
  cuda.bindings / decord 均可用。
- 硬件：RTX 4060（**单 NVDEC 单元**）/ 16 物理核 32 逻辑核 / 2×16GB DDR5-6000
  （实测流式上限 **55.8 GB/s**；WMI 的 `Speed`=5600 是 SPD 标称，
  `ConfiguredClockSpeed`=6000 才是实际值）。
- 测试视频 `D:\Videos\racelog_test\`，真值在 `ground_truth_csv/`
  （头是 `# roi=...`，**必须用正则取四个整数**，按逗号切只能拿到第一个）。
- 中文输出需 `sys.stdout.reconfigure(encoding="utf-8")`（GBK 控制台会崩）。
- `tools/` 子目录下 `sys.path[0]` 是 tools/，探针须 `sys.path.insert(0, 上级目录)`。
'''

DEC_HEADER = r'''# 设计决策档案（video_ocr_engine）

> **本文件是 2026-08-31 对 `CLAUDE.md` 瘦身时迁出的原文存档**，用于回答
> "为什么这样做 / 为什么不做"。
>
> ⚠️ **现役规则以 `CLAUDE.md` 为准**。本文件是**历史原文**，不再维护；
> 若与 `CLAUDE.md` 冲突，一律以 `CLAUDE.md` 为真相。
>
> **迁出原因**：`CLAUDE.md` 会被部分 harness 在会话开头全量注入。它当时已
> 70,764 B / 956 行 / 41,317 字，而其中只有约 3.9 KB 是现役事实 ——
> 让每会话为 67 KB 历史档案付费不合理。本文件按需读取，无预算限制。
>
> **结构**：正文是 CLAUDE.md 原 L32–L956 逐字搬迁，标题与编号未改；
> 章节索引在文末。代码注释里的 `DESIGN-REVIEW Xn` 标记指向本文
> 「设计审查结论」一节。

---
'''


def build_index(body: str) -> str:
    """从迁出的正文中提取 ### 标题，生成文末索引。"""
    lines = []
    n = 0
    for ln in body.split("\n"):
        if ln.startswith("### "):
            n += 1
            title = ln[4:].strip()
            lines.append("%2d. %s" % (n, title))
        elif ln.startswith("## "):
            n += 1
            lines.append("%2d. **%s**" % (n, ln[3:].strip()))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写盘（默认预演）")
    args = ap.parse_args()

    with open(SRC, "r", encoding="utf-8", newline="") as f:
        orig = f.read()

    # 按行切（文件是 CRLF，不能用 b"\n## ...\n" 这种 LF 锚点）
    lines = orig.split("\n")
    hits = [k for k, ln in enumerate(lines) if ln.rstrip("\r") == "## 通用约定"]
    if len(hits) != 1:
        print("❌ 切分锚点 `## 通用约定` 命中 %d 次（期望 1），中止。" % len(hits))
        return 1

    i = hits[0]
    head = "\n".join(lines[:i])
    body = "\n".join(lines[i:])
    # 统一归一到 LF：body 每行本就带 \r，若后面再 replace("\n", crlf) 会变 \r\r\n
    body = body.replace("\r\n", "\n").rstrip("\n") + "\n"

    crlf = "\r\n" if "\r\n" in orig else "\n"

    core = CORE.replace("\n", crlf).rstrip(crlf) + crlf
    dec = (DEC_HEADER.replace("\n", crlf) + crlf
           + body.replace("\n", crlf)
           + crlf
           + ("## 章节索引（迁出时自动生成）%s%s%s" % (crlf, crlf,
              build_index(body).replace("\n", crlf)))
           + crlf)

    print("=" * 66)
    print("CLAUDE.md 拆分计划%s" % ("" if args.apply else "（预演，未写盘）"))
    print("=" * 66)
    print("原文        %7d 字符 / %4d 行" % (len(orig), orig.count("\n")))
    print("  ├ 文件头   %7d 字符（整份重写）" % len(head))
    print("  └ 迁出正文 %7d 字符 / %4d 行（逐字不动）" % (len(body), body.count("\n")))
    print()
    print("CLAUDE.md    %7d → %7d 字符（%+.0f%%）"
          % (len(orig), len(core), 100.0 * (len(core) - len(orig)) / len(orig)))
    print("              %7d B → %7d B（UTF-8）"
          % (len(orig.encode("utf-8")), len(core.encode("utf-8"))))
    print("docs/DECISIONS.md  %7d 字符 / %7d B（UTF-8）"
          % (len(dec), len(dec.encode("utf-8"))))
    print()
    n_sec = body.count("\n### ") + body.count("\n## ")
    print("迁出章节数：%d" % n_sec)
    print("行尾约定：%s" % ("CRLF" if crlf == "\r\n" else "LF"))

    if not args.apply:
        print()
        print("（预演结束，加 --apply 执行）")
        return 0

    with open(SRC, "w", encoding="utf-8", newline="") as f:
        f.write(core)
    with open(DST, "w", encoding="utf-8", newline="") as f:
        f.write(dec)

    # 复核
    o = open(SRC, "rb").read()
    d = open(DST, "rb").read()
    print()
    print("写盘完成，复核：")
    print("  CLAUDE.md        %7d B / %4d 行 / CR %d"
          % (len(o), o.count(b"\n"), o.count(b"\r")))
    print("  DECISIONS.md     %7d B / %4d 行 / CR %d"
          % (len(d), d.count(b"\n"), d.count(b"\r")))
    ok = (d.count(body.replace("\n", crlf).encode("utf-8")) == 1)
    print("  正文逐字包含在 DECISIONS.md 中：%s" % ("是" if ok else "否 ⚠️"))
    if crlf == "\r\n":
        bad = d.count(b"\r\r\n")
        print("  双 CR 残留（应为 0）：%d %s" % (bad, "⚠️" if bad else "✅"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
