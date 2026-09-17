"""RunReport —— 一次 run 的唯一观测出口（v2 §8.6 N-3/N-5，S6-0）。

设计要点（逐条对 §8.6）：
- **唯一出口**：所有观测来自注入的 `Metrics`（N-2 单一计时脊柱）+ 编排
  显式字段，不再有"探针伸手进 ex._*"的第三套键名（M-1/M-3）。
- **schema 版本化**：`REPORT_VERSION` 只增不改；演进必须 bump 并留快照测试。
- **PI 守卫绑定命名指标**（N-5）：`health` 是机器判定，红旗可被 `bench`
  直接判失败——PI-10（热池 engine_init）、PI-3（syncs/chunk）在此落地。
- **透出**：`meta['report']`（telemetry≠off）。off 档无此键、不组装报告、
  不起采样器（PI-15 由 bench 三档互比门禁背书）。
- 细档 JSON sidecar 由 `diagnostics.report_file` 显式 opt-in（写文件是副作用）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# schema 版本：**只增不改**；任何结构演进必须 bump + 快照测试。
# v2（S6 续轮）：新增 `resources`（L1 相位边界差分）与 `hardware`（L2 NVML
# 峰值因子，仅 full 档采样过才出现）——都是**新增键**，按 v1 解析的旧读者不受影响。
# v3（2026-09-13）：新增 `diagnostics`（看门狗/崩溃日志/停顿落盘），同样**只加键**。
# v4（2026-09-17 重设计）：新增 `span_relations`（静态父子表 + 路径标记）与
# 派生 span `<parent>_other`（= parent.sum − Σ在场子项 sum），产品化
# "consumer − 已计 span = 缺口"的手工归因算术；spans 表只加键。
# v5（2026-09-17 续）：新增 `histograms`（无界 TOTALS 键的耗时分布，**full
# 档专属**——std 档该段缺席，不是空值冒充）+ `histograms_meta`（分箱自述）。
REPORT_VERSION = 5

#: v4：span 嵌套关系（代码级核对，2026-09-17 勘察）。**子项互不重叠**是
#: 硬约束——嵌套更深的键（如宿主 merge_pair/q_put_block 在 consume_feed
#: 内部）不得重复列入，否则 _other 会被双重扣减。
#: - host：consumer 线程墙钟内串行包含 decode.*（流生成器在 feed 循环内
#:   被拉动）+ 状态机 feed（consume_feed；merge_pair/q_put_block 嵌套于
#:   其中，不单列）。pipeline.decode ≡ pipeline.consumer（同区间双键，
#:   遗留等价键，读表时知悉）。
#: - gpu：pipeline.consumer 不产出（生产者线程成本由 pipeline.decode 表达），
#:   故无 consumer 关系；ocr 两对关系两路径同构。
SPAN_RELATIONS: dict = {
    "host": {
        "pipeline.consumer": ("decode.batch", "decode.luma_batch",
                              "decode.sharp_batch", "decode.binarize_batch",
                              "pipeline.consume_feed"),
        "pipeline.consume_feed": ("segment.merge_pair",
                                  "pipeline.q_put_block"),
        "ocr.infer": ("ocr.ctc_decode",),
        "ocr.preprocess": ("ocr.preproc_luma", "ocr.preproc_autocrop",
                           "ocr.preproc_resize"),
    },
    "gpu": {
        # W2：TRT 三子相位（full 档才有值；std 档缺席记 0，_other 退化为
        # 与 v4 相同口径——关系表静态声明，取值按在场与否）
        "ocr.infer": ("ocr.ctc_decode", "ocr.trt_enq", "ocr.trt_reduce",
                      "ocr.trt_concat"),
        "ocr.preprocess": ("ocr.preproc_luma", "ocr.preproc_autocrop",
                           "ocr.preproc_resize"),
    },
}


def _derive_other_spans(spans: dict, gauges: dict, path: str) -> dict:
    """v4：派生 `<parent>_other`（父 span 中未被已计子项覆盖的缺口）。

    子项取值顺序：spans 表（有界相位）→ TOTALS 汇总 gauge（无界键在
    std 档是 sum gauge）。缺席子项记 0（= 该键本次 run 没花时间）。
    条目只有 {n, sum}——不对缺口伪造 min/p50/max 分位数。
    """
    out: dict = {}
    for parent, children in SPAN_RELATIONS.get(path, {}).items():
        p = spans.get(parent)
        if not p:
            continue
        acc = 0.0
        for c in children:
            v = spans.get(c, {}).get("sum")
            if v is None:
                v = gauges.get(c)
            acc += float(v or 0.0)
        out[parent + "_other"] = {"n": p["n"],
                                  "sum": _num(max(0.0, p["sum"] - acc))}
    return out

#: PI 守卫阈值（§13.2 N-5：散文 → 指标名 + 阈值）
PI_LIMITS = {
    "PI-3": ("ocr.syncs_per_chunk", 3.0, "次/chunk"),
    "PI-10": ("ocr.engine_init", 0.1, "s"),
}

_ENV_CACHE: dict | None = None


def environment() -> dict:
    """录制基准字段（与 S0 manifest 同字段，§8.6 N-3）。

    进程内缓存：GPU/驱动查询要 **20–43ms 一次**（本机 NVML 20.8ms、
    nvidia-smi 子进程 43.1ms），落在报告组装路径上——每 run 重查会直接
    污染 std 档开销（PI-15）。首 run 付一次，之后 ~0。
    """
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return dict(_ENV_CACHE)
    from video_ocr_engine.config import constants as config
    env = {"engine_version": getattr(config, "__version__", None),
           "python": sys.version.split()[0]}
    try:
        import decord
        env["decord"] = decord.__version__
    except Exception:  # noqa: BLE001
        env["decord"] = None
    try:
        import tensorrt
        env["tensorrt"] = tensorrt.__version__
    except Exception:  # noqa: BLE001
        env["tensorrt"] = None
    # GPU/驱动：优先 NVML（本机实测 20.8ms 首调 vs nvidia-smi 子进程 43.1ms，
    # 且不起子进程），失败才回退 nvidia-smi；两者都不行记 unavailable。
    # 这段成本落在**报告组装路径**上（进程级缓存，只付一次），故能省则省。
    from video_ocr_engine.domain.resources import gpu_identity
    ident = gpu_identity()
    if ident:
        env["gpu"], env["driver"], env["gpu_source"] = ident[0], ident[1], "nvml"
    else:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=10).stdout.strip()
            name, _, drv = out.partition(",")
            env["gpu"] = name.strip() or None
            env["driver"] = drv.strip() or None
            env["gpu_source"] = "nvidia-smi" if name.strip() else "unavailable"
        except Exception:  # noqa: BLE001 无驱动工具 → 显式记不可用
            env["gpu"] = env["driver"] = None
            env["gpu_source"] = "unavailable"
    _ENV_CACHE = dict(env)
    return env


def _num(x):
    return round(float(x), 6)


def health(snap: dict, *, counters: dict | None = None) -> dict:
    """PI 阈值的机器判定（N-5）：只判"本次 run 能自证"的两条。

    PI-10 只在**池热**时判（同进程第 2 次及以后；`ocr.engine_reuse>0`）——
    冷启动付 0.31–0.42s 是既有事实（§7.5），不是回归。
    """
    gauges = snap.get("gauges", {})
    counters = counters if counters is not None else snap.get("counters", {})
    out: dict = {}
    v = gauges.get("ocr.engine_init")
    if v is not None:
        hot = counters.get("ocr.engine_reuse", 0) > 0
        out["PI-10"] = {"metric": "ocr.engine_init", "value": _num(v),
                        "limit": "<0.1s(热池)", "hot_pool": hot,
                        "ok": (v < PI_LIMITS["PI-10"][1]) if hot else True}
    v = gauges.get("ocr.syncs_per_chunk")
    if v is not None:
        out["PI-3"] = {"metric": "ocr.syncs_per_chunk", "value": _num(v),
                       "limit": "<=3", "ok": v <= PI_LIMITS["PI-3"][1]}
    return out


def build_report(metrics, *, wall: float, config_digest: str = "",
                 params: dict | None = None, degradations: list | None = None,
                 n_segments: int = 0, backend: str = "",
                 ocr_backend: str = "", extra: dict | None = None,
                 hardware: dict | None = None,
                 diagnostics: dict | None = None,
                 span_path: str = "host") -> dict:
    """组装 RunReport（telemetry=off 时返回 {}）。"""
    if not getattr(metrics, "enabled", False):
        return {}
    snap = metrics.snapshot()
    # 派生指标：填充率 = 内容列 / pad 列（R5 / S6-f 的判据本体）
    _pad = snap["counters"].get("ocr.padded_cols", 0)
    _content = snap["counters"].get("ocr.content_cols", 0)
    if _pad:
        snap["gauges"]["ocr.fill_pct"] = 100.0 * _content / _pad
    spans = {k: {kk: _num(vv) for kk, vv in v.items()}
             for k, v in snap["spans"].items()}
    gauges = {k: _num(v) for k, v in snap["gauges"].items()}
    # v4：缺口派生（只加键；条目形状 {n,sum}，不伪造分位数）
    derived = _derive_other_spans(spans, gauges, span_path)
    spans.update(derived)
    rep = {
        "report_version": REPORT_VERSION,
        "tier": metrics.tier,
        "wall_s": _num(wall),
        "spans": spans,
        "counters": dict(snap["counters"]),
        "gauges": gauges,
        "health": health(snap),
        "environment": environment(),
        "pipeline": {"backend": backend, "ocr_backend": ocr_backend,
                     "n_segments": n_segments,
                     "config_digest": config_digest},
        "degradations": list(degradations or []),
        "span_relations": {
            "path": span_path,
            "parent_children": dict(SPAN_RELATIONS.get(span_path, {})),
            "note": "_other = parent.sum − Σ在场子项 sum（缺席记 0；子项互"
                    "不重叠；host 路径 pipeline.decode ≡ pipeline.consumer"
                    "为遗留同值键）"},
    }
    # §8.6 r5 资源层（**report_version 2 的新增段，只加不改**）：
    # resources = L1 相位边界差分（std+ 即有）；hardware = L2 NVML 峰值因子
    # （仅 full 档采样过才出现；未采样时**不写该键**，而不是写空值冒充）。
    if hasattr(metrics, "resource_report"):
        res = metrics.resource_report()
        if res:
            res = dict(res)
            res["notes"] = ("L1 为进程级差分，OCR 与解码并发时核数互相计入"
                            "（相位平均并行核数的本意）；cores_avg_cycles 为"
                            "cycle 口径（无 process_time 15.625ms tick 量化，"
                            "短相位可信；按全程平均频率换算，变频相位有偏）；"
                            "本机自测口径，跨机不可比；PCIe 本机不可直读，"
                            "只能由 counter 字节 ÷ 相位墙钟推算（L3 推导区，"
                            "本表不产该字段）")
            rep["resources"] = res
    if hardware is not None:
        rep["hardware"] = hardware
    # v5：分布形状（full 档专属；std 档不写该键——缺席 ≠ 空值）
    hists = snap.get("histograms") or {}
    if hists:
        from video_ocr_engine.domain.metrics import (
            HIST_BUCKET_LOWER, HIST_EXP_BIAS, HIST_N_BUCKETS)
        rep["histograms"] = {
            k: {"n": v["n"], "buckets": list(v["buckets"])}
            for k, v in hists.items()}
        rep["histograms_meta"] = {
            "n_buckets": HIST_N_BUCKETS,
            "exp_bias": HIST_EXP_BIAS,
            "bucket_lower_s": [round(x, 8) for x in HIST_BUCKET_LOWER],
            "note": "桶号 = frexp(秒)指数 + %d；桶 i 覆盖 [lower[i], "
                    "lower[i+1])，末桶含上溢；读数=事件次数" % HIST_EXP_BIAS,
        }
    # v3：诊断段（未 arming 时不写该键，而不是写空值冒充）
    if diagnostics:
        rep["diagnostics"] = dict(diagnostics)
    if params:
        rep["params"] = dict(params)
    if extra:
        rep.update(extra)
    return rep


def write_report_file(report: dict, path: str) -> None:
    """细档 JSON sidecar（显式 opt-in；写文件是副作用，默认关闭）。"""
    import json
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                 encoding="utf-8", newline="\n")


def red_flags(report: dict) -> list:
    """报告中的失败项（bench 在线判失败用）。"""
    return [k for k, v in (report.get("health") or {}).items()
            if not v.get("ok", True)]
