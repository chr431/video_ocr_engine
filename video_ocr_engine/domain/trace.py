"""P4 实验性全事件时间线（VOE_TRACE_FILE，默认关）。

定位（2026-09-17 重设计轮，用户裁决纳入实验）：RunReport 是**聚合口径**
（n/sum/分位数，样本即弃）；本模块保留**逐事件时间线** (name, t0, t1,
tid)，供离线重建并发图（跨线程重叠 / 空洞 / 冷热段 / 嵌套父子）。

约束与成本：
- **默认关、关闭零成本**：`extractor._prof_end` 尾部只多一次 `is not None`
  判断（不取时钟不分配——t1 复用已算好的 t0+elapsed）。std 档 ≤2ms/run
  的成本守卫（tests/config/test_telemetry_cost.py）在 trace 代码在场但
  关闭时必须原样通过，即结构性守护。
- **开启成本不受该守卫约束**（用户裁决放宽）：每事件一次 list.append，
  ~2.3k 事件/run；单独测试钉宽松上限（实测×3，防病理回退）。
- 嵌套父子离线重建：同线程区间包含即可，无需父 ID、不改 `_prof_end`
  签名（脊柱不动）。
- NVML：开启时自带 0.2s 采样线程（与 full 档 L2 无关，trace 独立持有），
  点列与 host 事件同用 `time.perf_counter()` 时间轴 → 可对齐。
"""
from __future__ import annotations

import json
import threading
import time

__all__ = ["TraceRecorder"]


class TraceRecorder:
    """线程局部累积 + 一次性合并落盘（与 Metrics 同构的桶模型）。"""

    __slots__ = ("_local", "_buckets", "_lock", "_t0", "_nvml")

    def __init__(self) -> None:
        self._local = threading.local()
        self._buckets: list = []
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._nvml = None

    def record(self, name: str, t0: float, t1: float | None = None) -> None:
        """记一条事件（热路径：一次 list.append；t1 缺省现取）。"""
        b = getattr(self._local, "ev", None)
        if b is None:
            b = self._local.ev = []
            with self._lock:          # 每线程一次（桶登记）
                self._buckets.append(b)
        b.append((name, t0, t1 if t1 is not None else time.perf_counter(),
                  threading.get_ident()))

    def record_run(self, t0: float, wall: float) -> None:
        """顶层 pipeline.run（由 extractor 在收尾记，真实锚点）。"""
        self.record("pipeline.run", t0, t0 + wall)

    def start_hardware(self) -> None:
        """trace 自带 NVML 点列（0.2s tick）；不可用时静默降级为纯 host 线。"""
        try:
            from .resources import NvmlSampler
            s = NvmlSampler(interval_s=0.2)
            s.start()
            self._nvml = s
        except Exception:  # noqa: BLE001 NVML 不可用：host 时间线仍完整
            self._nvml = None

    def dump(self, path: str, *, meta: dict | None = None,
             wall: float = 0.0) -> dict:
        """合并全部线程桶并写 JSON；返回 {n_events, path} 供日志。"""
        from pathlib import Path
        with self._lock:
            buckets, self._buckets = self._buckets, []
        evs = [e for b in buckets for e in b]
        evs.sort(key=lambda e: e[1])
        base = min((e[1] for e in evs), default=self._t0)
        nvml_pts: list = []
        if self._nvml is not None:
            self._nvml.stop()
            nvml_pts = [[round(p[0] - base, 6)] + list(p[1:])
                        for p in self._nvml.points() if p[0] is not None]
        out = {
            "schema": 1,
            "wall_s": round(wall, 6),
            "n_events": len(evs),
            "meta": dict(meta or {}),
            "columns": ["name", "t0", "t1", "tid"],
            # 紧凑行式（无缩进）控制体量：~2.3k 事件 ≈ 数百 KB
            "events": [[n, round(a - base, 6), round(b - base, 6), t]
                       for n, a, b, t in evs],
            "nvml_points": nvml_pts,
            "note": "时间轴相对首事件起点（s，perf_counter）；nvml 行 = "
                    "[t, gpu%, nvdec%, vram_mib, sm_mhz, mem_mhz, vid_mhz, "
                    "throttle_mask]；timing 派生段（pipeline.decode/ocr/"
                    "ocr_tail）的 t1 锚定在报告组装时刻，误差=ocr_tail+"
                    "组装时长；嵌套父子按同线程区间包含重建",
        }
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8",
                     newline="\n")
        return {"n_events": len(evs), "path": str(p)}
