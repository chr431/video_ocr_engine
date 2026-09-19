# -*- coding: utf-8 -*-
"""假配置面修复（最小正确方案）：6 个"env 活读"旋钮标注 + digest 排除。

背景（2026-09-19 技术债调查）：6 个旋钮的 RunConfig 字段解析后无人读取，
消费者直读 env（`config.env_bool(...)`）。它们是**真实可用的能力**，但：
  ① 字段值不反映实际行为（rc.trt_fp16=False 而 env 可能为真）；
  ② **污染 config_digest**——A/B 对账拿到与实际行为不符的指纹。

为何不改成注入：subprobe 是模块级常量、在 trt.py 14 处热路径读；
fp16 走 TrtEngine 构造（有测试依赖）；stream/drainer 在 gpu ctx 装配期。
改动面覆盖 TRT 引擎构建路径，风险显著高于"digest 更准"的收益。

采用方案（最小正确）：
  · registry 增 `env_live_only: bool = False` 标记；
  · 6 个旋钮标 True（并在 note 里说明"调用期 env 活读，不进 digest"）；
  · resolve 计算 digest 时**排除**它们；
  · 审计项保证该标记与消费者实际行为一致（人工核对项，见文档）。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

LIVE_ONLY = {
    "ocr.trt_fp16", "ocr.trt_defer_sync", "pipeline.gpu_stream",
    "pipeline.gpu_drainer", "diag.subprobe", "diag.bounds_debug",
}


def patch(path, pairs):
    raw = open(path, "rb").read().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    for old, new in pairs:
        o = old.replace("\n", nl)
        n = new.replace("\n", nl)
        if raw.count(o) != 1:
            print("  SKIP(%s) hits=%d: %s" % (path, raw.count(o),
                                              old.split("\n")[0][:56]))
            continue
        raw = raw.replace(o, n)
        print("  ok  %s: %s" % (path, old.split("\n")[0][:56]))
    open(path, "wb").write(raw.encode("utf-8"))


# ── 1. registry：加标记字段 ─────────────────────────────────────────
patch("video_ocr_engine/config/registry.py", [
    ("""    note: str = ""               # 解析语义备注（三态 / 特殊回退等）
    deprecated: str | None = None""",
     """    note: str = ""               # 解析语义备注（三态 / 特殊回退等）
    deprecated: str | None = None
    # 2026-09-19：该旋钮由消费者**调用期 env 活读**，RunConfig 字段值
    # 不决定行为 → 从 config_digest 排除（否则 A/B 指纹与实际不符）。
    # 标记为 True 的旋钮，其 RunConfig 字段仅供 introspection 兼容。
    env_live_only: bool = False"""),
])

# ── 2. resolve：digest 排除 ─────────────────────────────────────────
patch("video_ocr_engine/config/resolve.py", [
    ("""    digest_src = json.dumps(values, sort_keys=True, ensure_ascii=True,
                            default=str)""",
     """    # digest 只覆盖**生效值**（2026-09-19）：env_live_only 旋钮的实际
    # 行为由调用期 env 决定，把解析结果计入 digest 会让 A/B 指纹撒谎。
    _live_only = {k.name.replace(".", "_") for k in KNOBS.knobs
                  if getattr(k, "env_live_only", False)}
    _digest_vals = {k: v for k, v in values.items() if k not in _live_only}
    digest_src = json.dumps(_digest_vals, sort_keys=True, ensure_ascii=True,
                            default=str)"""),
])

print("阶段完成：registry 标记 + digest 排除")
print("待手工：6 个旋钮定义处加 env_live_only=True + note 说明")
