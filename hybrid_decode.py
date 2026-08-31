"""CPU+NVDEC 双解码读取器 v3：速率比例分界 + 两端连续扫掠（对称接管）。

入口：`FieldExtractor(decode_backend="hybrid")`（与 auto/cpu/nvdec 并列的
解码后端选择；无环境变量入口，仅显式参数选择）。

v3 设计（2026-08，探针定位 v2 退化后重写）：

  v2（kfe 共享队列竞争）实测退化根因（HEVC，CPU 慢 4.5×）：
    1. FIFO 竞争 + in-flight 令牌使分片在 GPU/CPU 间严格交替领取；
       消费者按全局帧序取帧 → 慢生产者的每一片都是关键路径串行等待，
       快生产者被令牌限制无法超前；
    2. 交替领取使"连续扫掠免 seek"失效：每个生产者除首片外几乎每片
       seek（GPU ~50-190ms/次、CPU ~35-65ms/次）；
    3. 结果：HEVC hybrid decode 2.4-2.8s 反而比纯 NVDEC 2.0s 慢
       20-40%；h264（CPU 快）时 hybrid 赢但靠 CPU 单端而非并行。

  v3（速率比例分界 + 两端连续扫掠 + 快端接管）：
    1. 采样帧序列仍按关键帧边界切分片（kfe，边界 seek 便宜）；
    2. hybrid_begin 时并行实测两后端顺序解码速率（256 帧 + 16 帧
       warmup 丢弃，双线程）；
    3. 按帧数速率比例把分片切成两段：快端从头连续扫掠前半（0 次 seek），
       慢端 seek 一次到分界片首后连续扫掠后半（1 次 seek）；
       慢端份额夹在 [15%, 45%]，速率比 >1.8x 时只给 1 片试探
       （防校准误差放大）；
    4. 快端接管：快端扫完自己区后逐片接管慢端区未开始片（一次 seek
       连续扫掠）——校准误差自愈；慢端只做自己区、区空即退出（不反向
       接管快端区，避免破坏快端连续扫掠）；
    5. 内存上界：每生产者"已产出未消费"片数 ≤ inflight（默认 2），
       消费者按序排空后才继续产下一片（字幕宽 ROI 防内存暴涨）；
    6. 消费者仍按全局帧序取帧（零改动），交付序不变。

  对外仍是 VideoReader 同形替身：len / get_batch / next_roi / seek_accurate
  / get_*；正确性依赖 v0.7.8+ 双后端 YUV420 逐位一致。
"""
from __future__ import annotations

import bisect
import os
import threading
import time

import numpy as np

import engine_config as config

# ── 探针（诊断）：HYBRID_PROBE=1 时打印各生产者的分片时序与速率；
# HYBRID_PROBE_CSV 为输出 CSV 路径时追加逐片明细。 ──
_HYBRID_PROBE = os.environ.get(config.HYBRID_PROBE_ENV, "0") == "1"
_HYBRID_PROBE_CSV = os.environ.get(config.HYBRID_PROBE_CSV_ENV, "") or None

# 分片粒度上限（HYBRID_MAX_CHUNK_FRAMES>0）：hybrid_begin 生成的分片若
# 超过该帧数则继续按关键帧边界/等分拆小（内存上界 = inflight × 该上限；
# 宽 ROI 字幕整集防单大片 2000+ 帧一次性缓存在 ch['data']）。0 = 不拆
#（兼容 v3 原行为）。env 读取在 extractor 构造 HybridDecoder 时完成。

# close() 等待后台生产者退出的上限（秒）。正常路径下线程响应 _stop 后
# 立即退出，此值只是防挂死兜底：宁可留下一个已停摆的 daemon 线程，也不
# 让 close() 无限期阻塞调用方。
_CLOSE_JOIN_TIMEOUT = 5.0


class _Batch:
    """最小 decord NDArray 兼容壳：asnumpy() / shape。

    注意：帧数据是解码后 asnumpy() 的宿主数组（非 decord GPU NDArray），
    **有意不提供 to_dlpack** —— 调用方不应从这里取 device 指针
    （extractor 已对 HybridDecoder 关闭 dev_info 采集）。
    """

    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape


def _nearest_keyframe_sample(target: int, key_frames: list[int],
                             frames: list[int]) -> int:
    """返回离 target 最近的关键帧，再吸附到最近的采样帧号（保持采样网格）。"""
    if not key_frames or not frames:
        return target
    idx = bisect.bisect_left(key_frames, target)
    cand = [idx - 1, idx]
    cand = [i for i in cand if 0 <= i < len(key_frames)]
    if not cand:
        return target
    key = min((key_frames[i] for i in cand),
              key=lambda k: (abs(k - target), k))
    sidx = bisect.bisect_left(frames, key)
    sc = [sidx - 1, sidx]
    sc = [i for i in sc if 0 <= i < len(frames)]
    if not sc:
        return target
    return min((frames[i] for i in sc),
               key=lambda f: (abs(f - key), f))


def _keyframe_every_chunks(frames: list[int], key_frames: list[int],
                           rest_start: int, last_end: int, stride: int,
                           min_gap: int, max_chunks: int) -> list[tuple[int, int]]:
    """每关键帧一片（kfe）——双解码分片生成。

    按基础最小片间距切；若关键帧过密（mkv 重编码 ~每 30-140 源帧一个
    关键帧）导致片数超过上限，逐步放大间距合并，片数受控在 max_chunks
    以内。边界吸附到最近采样帧（保持全帧覆盖、无缝隙/无重叠；吸附帧离
    关键帧 ≤ stride/2，seek_accurate 仍便宜）。返回覆盖 [rest_start,
    last_end) 的连续切片列表（首片起点=rest_start，末片终点=last_end）。
    """
    _key_list = [k for k in key_frames if rest_start < k < last_end]
    if not _key_list:
        return [(rest_start, last_end)]
    _mg = max(1, int(min_gap))
    _mx = max(1, int(max_chunks))
    _s = max(1, int(stride))
    _big: list[tuple[int, int]] = [(rest_start, last_end)]
    for _iter in range(80):
        _cand: list[tuple[int, int]] = []
        _prev2 = rest_start
        for _k in _key_list:
            _b = _nearest_keyframe_sample(_k, key_frames, frames)
            if (_b - _prev2) // _s >= _mg and _b < last_end:
                _cand.append((_prev2, _b))
                _prev2 = _b
        _cand.append((_prev2, last_end))
        _big = _cand
        if len(_cand) - 1 <= _mx:
            break
        _mg = max(_mg + 1, int(_mg * 1.5))
    return _big


def _split_oversized(specs, frames: list[int], key_frames: list[int],
                     max_frames: int) -> list[tuple[int, int]]:
    """把超过 max_frames 采样帧的片拆小（内存上界 = inflight × max_frames）。

    优先按已有关键帧边界切（seek 便宜）；关键帧不足时按帧数等分，
    保证拆后每片帧数 ≤ max_frames、且覆盖完整无缝隙。max_frames<=0 原样返回。
    """
    if max_frames <= 0 or not specs:
        return specs
    out: list[tuple[int, int]] = []
    _key_list = sorted(k for k in key_frames)
    for a, b in specs:
        fis = [f for f in frames if a <= f < b]
        if len(fis) <= max_frames:
            out.append((a, b))
            continue
        # 候选切点：片内关键帧（吸附到采样帧，去重、排除端点）
        cuts = []
        for k in _key_list:
            if a < k < b:
                s = _nearest_keyframe_sample(k, key_frames, frames)
                if a < s < b and s not in cuts:
                    cuts.append(s)
        cuts.sort()
        # 从片首开始，按"当前片 + 下一关键帧 ≤ 上限"贪心切；关键帧不够时
        # 按帧数等分补足
        seg_start = a
        while True:
            seg_fis = [f for f in frames if seg_start <= f < b]
            if len(seg_fis) <= max_frames:
                out.append((seg_start, b))
                break
            # 找最远的关键帧切点，使左片 ≤ 上限；无则等分
            chosen = None
            for c in cuts:
                if seg_start < c < b:
                    left = [f for f in frames if seg_start <= f < c]
                    if len(left) <= max_frames:
                        chosen = c
            if chosen is None:
                left_fis = seg_fis[:max_frames]
                last = left_fis[-1]
                # 切点必须是采样帧边界（半开区间 [seg_start, chosen)）：
                # chosen = 左片最后采样帧的下一个采样帧。frames 严格递增，
                # 用 index+1 取下一个；已是最后一个采样帧则切到片尾。
                pos = frames.index(last)
                chosen = frames[pos + 1] if pos + 1 < len(frames) else b
                if chosen <= seg_start:
                    chosen = b
            out.append((seg_start, chosen))
            seg_start = chosen
    return out


def _measure_rate(reader, frames: list[int], roi: tuple, n: int,
                  batch: int = 64, warmup: int = 8) -> float:
    """顺序解码测速（帧/s）。调用方负责先 seek 到起始帧。

    warmup 帧先丢弃（解码会话/队列初始化开销会污染首批速率——
    NVDEC 首个 get_batch 可能含 ~50ms 启动成本），再测 n 帧。
    """
    if n <= 1:
        return 0.0
    fr = frames[:n]
    i = 0
    if warmup > 0:
        w = min(warmup, len(fr) - 1)   # 至少保留 1 帧用于测速
        reader.get_batch(fr[:w], roi=roi).asnumpy()
        i = w
    t0 = time.perf_counter()
    measured = 0
    while i < len(fr):
        be = min(i + batch, len(fr))
        reader.get_batch(fr[i:be], roi=roi).asnumpy()
        measured += be - i
        i = be
    dt = time.perf_counter() - t0
    return measured / max(dt, 1e-9)


def _dynamic_split(counts: list[int], rf: float, rs: float, *,
                   slow_is_cpu: bool, max_share: float = 0.45,
                   safety: float = 0.95,
                   discount_cpu: float = 0.45,
                   discount_gpu: float = 0.85) -> int:
    """动态分界（v4 纯函数）：返回快端片数 split_idx（慢端 = [split_idx, n)）。

    约束：慢端生产时间 ≤ 快端生产时间 × safety（慢端不拖尾）；在满足
    约束前提下给慢端尽量多的片（慢端贡献最大化）。慢端稳态速率 = rs ×
    折扣（慢端=CPU 软解 ×0.45 修正缓冲衰减高估；=NVDEC ×0.85）。
    返回 split_idx ∈ [1, n-1]（两端各至少 1 片）。
    """
    n = len(counts)
    if n <= 1:
        return n
    total_fr = sum(counts)
    if total_fr <= 0 or rf <= 0 or rs <= 0:
        return max(1, n - 1)
    disc = discount_cpu if slow_is_cpu else discount_gpu
    rs_eff = rs * disc
    best_split = n            # 快端片数（慢端 = n - best_split）
    for k in range(1, n):     # k = 慢端片数
        slow_fr = sum(counts[n - k:])
        fast_fr = total_fr - slow_fr
        if fast_fr <= 0:
            break
        t_slow = slow_fr / max(rs_eff, 1e-9)
        t_fast = fast_fr / max(rf, 1e-9)
        if t_slow > t_fast * safety:
            break
        if slow_fr / total_fr > max_share:
            break
        best_split = n - k
    return max(1, min(n - 1, best_split))


class HybridDecoder:
    """双解码读取器 v3：速率比例分界 + 两端连续扫掠（对下游透明）。"""

    # 慢端份额上限：45%（防校准把过多给慢端）。下限 = 至少 1 片
    #（按片数动态取，见 hybrid_begin）。
    _SLOW_MAX_SHARE = 0.45

    def __init__(self, ex, gpu_vr, *, max_chunks: int = 16,
                 cpu_threads: int = 0, inflight: int = 2,
                 min_gap: int = 16, calib_frames: int = 0,
                 max_chunk_frames: int = 0):
        # 分片粒度上限：>0 时把超过该帧数的片继续拆小（内存上界 =
        # inflight × max_chunk_frames 帧）。0 = 不拆（兼容 v3）。
        self._max_chunk_frames = max(0, int(max_chunk_frames))
        # 速率校准帧数：0 = 用 env HYBRID_CALIB_FRAMES，缺省 40（弱 CPU
        # 下 256 帧校准 ~0.4s，会吃掉混合解码的收益；40 帧 ~0.06-0.12s
        # 已足够稳定，且 seek 是固定成本与帧数无关；短校准配合稳态折扣
        # HYBRID_SLOW_DISCOUNT 修正 CPU 软解的缓冲衰减高估）。
        calib = config.env_int(
            config.HYBRID_CALIB_FRAMES_ENV,
            config.HYBRID_CALIB_FRAMES_DEFAULT) if calib_frames <= 0 else int(calib_frames)
        self._calib_frames = max(32, calib)
        self._gpu = gpu_vr
        self._ex = ex
        self._roi = (ex._roi[0], ex._roi[1], ex._roi[2] + 1, ex._roi[3] + 1)
        total = len(gpu_vr)
        f0 = int(ex._frame_start or 0)
        f1 = min(int(ex._frame_end or total), total)
        if f1 - f0 < 4:
            raise ValueError('window too short for hybrid decode')
        self._f0, self._f1 = f0, f1
        self._max_chunks = max(2, int(max_chunks))
        self._min_gap = max(1, int(min_gap))
        self._inflight = max(1, int(inflight))
        # 慢端预取上限：慢端解码尾段，消费者要等快端前段消费完才轮到它；
        # inflight=2 时慢端只能提前 2 片，消费者到尾段时慢端才刚起步 →
        # decode 结束更早但 OCR 尾批堆积（ocr_tail 增大，墙钟净亏）。
        # 慢端允许更多预取（默认 4 片）让尾段提前就绪、消费者连续消费。
        self._inflight_slow = max(
            self._inflight,
            config.env_int(config.HYBRID_SLOW_INFLIGHT_ENV,
                           config.HYBRID_SLOW_INFLIGHT_DEFAULT))
        nt_kw = {}
        if cpu_threads and cpu_threads > 0:
            nt_kw['num_threads'] = cpu_threads
        # CPU reader 后台打开：与构造后到 hybrid_begin 之间的工作
        # （fps 读取/帧序列/分片生成）以及 hybrid_begin 里**不依赖 CPU
        # reader 的 GPU 端测速**重叠。实测（2026-08-29，热缓存）打开近零
        # 成本、墙钟持平 —— 路线图"打开 ~0.12s"估算未复现；改动结构性
        # 严格不劣（冷缓存/慢盘首次打开兜底）故保留。打开失败在
        # hybrid_begin 里上抛（校准失败同为 hybrid_begin 的既有失败面；
        # GPU reader 已成功打开的前提下 CPU 打开失败实际不可达）。
        self._cpu = None
        self._cpu_open_err: list = []

        def _open_cpu() -> None:
            try:
                from decord import cpu as _cpu
                self._cpu = self._ex._open_decord_reader(_cpu(0), {},
                                                         **nt_kw)
            except Exception as e:  # noqa: BLE001
                self._cpu_open_err.append(e)

        self._cpu_thread = threading.Thread(target=_open_cpu, daemon=True)
        self._cpu_thread.start()

        self._stop = threading.Event()
        self._err = []
        self._cv = threading.Condition()
        self._chunks = []      # {'fis','data','off','delivered','done','owner','started'}
        self._starts = []      # 每片首帧（bisect 用）
        self._begun = False
        self._closed = False
        self._seq_fi = None
        self._threads = []
        # ── v3 状态 ──
        self._fast_tag = "gpu"
        self._fast_reader = None
        self._slow_reader = None
        self._split_idx = 0
        # 每生产者"已产出未消费"计数（内存上界 = inflight 片）
        self._unconsumed = {"fast": 0, "slow": 0}
        # ── 探针状态 ──
        self._probe = _HYBRID_PROBE
        self._probe_csv = _HYBRID_PROBE_CSV
        self._probe_rows: list = []
        self._pname = {}       # id(reader) -> 'gpu'/'cpu'
        self._probe_lock = threading.Lock()

    # ─────────────── 分片生成与启动（frames 就绪后调用） ───────────────

    def hybrid_begin(self, frames) -> None:
        if getattr(self, '_closed', False):
            raise RuntimeError('HybridDecoder 已关闭')
        if self._begun:
            return
        self._begun = True
        fr = [f for f in frames if self._f0 <= f < self._f1]
        if len(fr) < 4:
            raise ValueError('hybrid: sampled frames too few')
        try:
            keys = [int(k) for k in self._gpu.get_key_indices()]
        except Exception:
            keys = []
        try:
            specs = _keyframe_every_chunks(
                fr, keys, fr[0], fr[-1] + 1,
                max(1, int(getattr(self._ex, '_sample_stride', 1))),
                self._min_gap, self._max_chunks)
        except Exception:
            specs = []
        if len(specs) < 2:
            n = min(self._max_chunks, max(2, len(fr) // 256))
            step = (len(fr) + n - 1) // n
            specs = [(fr[i], fr[min(i + step, len(fr)) - 1] + 1)
                     for i in range(0, len(fr), step)]
        # 分片粒度上限（HYBRID_MAX_CHUNK_FRAMES>0）：超过上限的片继续拆小。
        # 优先按已有关键帧边界切（seek 便宜），否则等分到 ≤ 上限。
        if self._max_chunk_frames > 0:
            specs = self._split_oversized(specs, fr, keys)
        for a, b in specs:
            fis = [f for f in fr if a <= f < b]
            if not fis:
                continue
            self._chunks.append({'fis': fis, 'data': [], 'off': 0,
                                 'delivered': 0, 'all_delivered': False,
                                 'counted': False, 'consumed': False,
                                 'done': False, 'owner': None,
                                 'started': False})
            self._starts.append(fis[0])
        n = len(self._chunks)
        # ── 速率校准（并行测速，双线程） ──
        # 单轮测速。多轮取中位（原 HYBRID_CALIB_ROUNDS）实测净负已删除
        # （3 轮 -21%：~0.68s 成本 > 分界精度收益）。
        #
        # **校准预算按"源帧"计，不按"采样帧"**（stride>1 才正确）：
        # 采样帧数 = 源帧数 / stride，若按采样帧给预算，stride=8 时
        # HYBRID_CALIB_FRAMES=40 要解 40×8=320 源帧，比 stride=1 贵 8 倍
        # （实测 AV1 stride=8 的校准占满 ~0.19s，是 hybrid 启动开销的大头）。
        # 而校准只取两后端速率的**比值**，解同样多的源帧信息量等价 ——
        # 换算回采样帧数即可，stride==1 时与旧行为逐位一致。
        _step = max(1, int(getattr(self._ex, '_sample_stride', 1)))
        _src_budget = max(16, self._calib_frames)
        calib = max(6, min(len(fr), _src_budget // _step))
        # warmup 同样按采样帧折算，避免 stride 大时预热吃掉全部计时样本
        _warm = max(1, min(8, calib // 3))
        rates: dict = {}

        def _calib(tag, reader):
            try:
                reader.seek_accurate(fr[0])
                # 单轮测速：多轮取中位（HYBRID_CALIB_ROUNDS）实测净负
                # （3 轮 -21%：~0.68s 测速成本 > 分界精度收益，
                # 见 docs/PERFORMANCE.md §10.5），0.9.0 删除。
                rates[tag] = _measure_rate(reader, fr, self._roi, calib,
                                           warmup=_warm)
            except Exception as e:
                rates[tag] = 0.0
                with self._cv:
                    self._err.append(e)

        ths = []
        # GPU 测速不依赖 CPU reader → 立即启动，与后台打开 CPU reader 重叠；
        # CPU 测速等打开完成后再跑（见 __init__ 的后台打开注释）。
        t = threading.Thread(target=_calib, args=("gpu", self._gpu),
                             daemon=True)
        t.start()
        ths.append(t)

        def _calib_cpu_when_ready() -> None:
            self._cpu_thread.join()
            if self._cpu_open_err:
                with self._cv:
                    self._err.append(self._cpu_open_err[0])
                return
            _calib("cpu", self._cpu)

        t = threading.Thread(target=_calib_cpu_when_ready, daemon=True)
        t.start()
        ths.append(t)
        for t in ths:
            t.join()
        if self._err:
            raise self._err[0]
        r_gpu = rates.get("gpu", 0.0)
        r_cpu = rates.get("cpu", 0.0)
        if self._probe:
            print(f"[hybrid] calib: gpu={r_gpu:.0f}fps cpu={r_cpu:.0f}fps "
                  f"chunks={n}", flush=True)
        # ── 速率比例分界（快端在前） ──
        if r_gpu >= r_cpu:
            self._fast_tag = "gpu"
            self._fast_reader = self._gpu
            self._slow_reader = self._cpu
            rf, rs = max(r_gpu, 1.0), max(r_cpu, 1.0)
        else:
            self._fast_tag = "cpu"
            self._fast_reader = self._cpu
            self._slow_reader = self._gpu
            rf, rs = max(r_cpu, 1.0), max(r_gpu, 1.0)
        # ── 动态分界（v4）：在"慢端不拖尾"约束下给慢端尽量多的片 ──
        # 理论最优 = 速率比例（两端同时完成 → decode = N/(rf+rs)）；但
        # 并发解码时慢端速率会打折（争抢），比例份额会给慢端过多片 →
        # 慢端拖尾、decode 反被拖慢（实测 HEVC 8 核：比例 25% → 慢端 3 片
        # 1.36s > 快端 1.16s，decode 被拖到 1.36s）。
        # 另：短校准会高估慢端稳态速率（HEVC 软解有缓冲衰减：48 帧测
        # 495fps、384 帧测 205fps，快测高估 2 倍+）→ 需按稳态折扣修正。
        # 慢端 = CPU 软解：×0.45（缓冲衰减，48 帧快测 ≈ 稳态的 2.2 倍）；
        # 慢端 = NVDEC：×0.85（NVDEC 稳态略降）。env HYBRID_SLOW_DISCOUNT
        # 可覆盖。
        counts = [len(ch['fis']) for ch in self._chunks]
        slow_is_cpu = (self._slow_reader is self._cpu)
        default_disc = (config.HYBRID_SLOW_DISCOUNT_DEFAULT_CPU if slow_is_cpu
                        else config.HYBRID_SLOW_DISCOUNT_DEFAULT_GPU)
        slow_disc = config.env_float(config.HYBRID_SLOW_DISCOUNT_ENV,
                                     default_disc)
        self._split_idx = _dynamic_split(
            counts, rf, rs, slow_is_cpu=slow_is_cpu,
            discount_cpu=slow_disc, discount_gpu=slow_disc)
        if self._probe:
            print(f"[hybrid] split: fast={self._fast_tag}->[0,{self._split_idx}) "
                  f"slow->[{self._split_idx},{len(self._chunks)}) "
                  f"rf={rf:.0f} rs={rs:.0f} "
                  f"rs_eff={rs*slow_disc:.0f} disc={slow_disc:.2f}",
                  flush=True)
        self._pname[id(self._gpu)] = "gpu"
        self._pname[id(self._cpu)] = "cpu"
        for tag, reader in ((self._fast_tag, self._fast_reader),
                            ("slow", self._slow_reader)):
            t = threading.Thread(target=self._producer, args=(reader,),
                                 daemon=True)
            t.start()
            self._threads.append(t)
        if self._probe:
            print(f"[hybrid] begin chunks={n} frames={len(fr)}", flush=True)

    # ─────────────── 生产者 ───────────────

    def _zone(self, who: str) -> tuple[int, int]:
        if who == "fast":
            return 0, self._split_idx
        return self._split_idx, len(self._chunks)

    def _take_chunk(self, who: str):
        """取一片。who='fast'/'slow'。

        优先取自己区未认领片（连续扫掠）；快端自己区空后逐片接管慢端区
        未认领片（一次 seek 连续扫掠，校准误差自愈）；慢端区空即退出
        （不反向接管快端区，避免破坏快端连续扫掠）。
        """
        while not self._stop.is_set():
            with self._cv:
                limit = (self._inflight_slow if who == "slow"
                         else self._inflight)
                if self._unconsumed[who] >= limit:
                    self._cv.wait(0.05)
                    continue
                lo, hi = self._zone(who)
                idx = self._next_unclaimed(lo, hi)
                if idx is not None:
                    self._claim(idx, who)
                    return idx
                if who == "slow":
                    # 慢端只做自己区；区空即退出（不反向接管快端区——
                    # 会破坏快端连续扫掠并引入额外 seek）
                    return -1
                # 快端自己区空 → 接管慢端未认领尾段（逐片认领，避免
                # "认领整段但只生产第一片"后 _next_unclaimed 全为
                # started 导致退出的漏片 bug）
                oidx = self._next_unclaimed(self._split_idx, len(self._chunks))
                if oidx is None:
                    return -1   # 全认领完，退出
                self._claim(oidx, who)
                return oidx
        return -1

    def _next_unclaimed(self, lo: int, hi: int):
        for j in range(lo, hi):
            if not self._chunks[j]['started']:
                return j
        return None

    def _claim(self, idx: int, who: str):
        ch = self._chunks[idx]
        ch['owner'] = who
        ch['started'] = True
        ch['claim_t'] = time.perf_counter()
        ch['claim_by'] = who

    def _producer(self, reader):
        fast = (reader is self._fast_reader)
        who = "fast" if fast else "slow"
        prev_end = None
        # 片间"连续扫掠免 seek"依赖 prev_end 与下一片首帧严格相等。
        # 步长必须取采样网格 stride：stride>1 时下一片首帧 = fis[-1]+stride，
        # 用 +1 会导致**每片都 seek**（GPU 50~190ms/次、CPU 35~65ms/次，
        # 16 片 ≈ 1.6s 纯开销，实测 HEVC/AV1 stride=8 反而比纯 NVDEC 慢
        # ~55%，且比任一单端独跑都慢）。与 next_roi 的 +1 → +stride 是
        # 同一类缺陷（stride==1 时两者恒等，故长期未被发现）。
        step = max(1, int(getattr(self._ex, '_sample_stride', 1)))
        while not self._stop.is_set():
            idx = self._take_chunk(who)
            if idx < 0:
                return
            ch = self._chunks[idx]
            fis = ch['fis']
            t_chunk = time.perf_counter()
            t_seek = 0.0
            try:
                if prev_end is None or fis[0] != prev_end:
                    t_s = time.perf_counter()
                    reader.seek_accurate(fis[0])
                    t_seek = time.perf_counter() - t_s
                i = 0
                batch = 64
                while i < len(fis) and not self._stop.is_set():
                    be = min(i + batch, len(fis))
                    chunk = fis[i:be]
                    arr = reader.get_batch(chunk, roi=self._roi).asnumpy()
                    with self._cv:
                        for k, fi in enumerate(chunk):
                            ch['data'].append((fi, arr[k]))
                        self._cv.notify_all()
                    i = be
                with self._cv:
                    ch['done'] = True
                    if not ch.get('all_delivered', False):
                        self._unconsumed[who] += 1
                        ch['counted'] = True
                    self._cv.notify_all()
                prev_end = fis[-1] + step
                t_done = time.perf_counter()
                ch['produce_s'] = t_done - t_chunk
                ch['seek_s'] = t_seek
                if self._probe:
                    with self._probe_lock:
                        self._probe_rows.append(
                            (idx, self._pname.get(id(reader), who),
                             len(fis), fis[0], fis[-1], t_chunk,
                             t_done, t_seek, ch.get('claim_t', 0.0),
                             ch.get('claim_by', '')))
            except Exception as e:  # noqa: BLE001
                self._err.append(e)
                with self._cv:
                    ch['done'] = True
                    self._cv.notify_all()
                return

    # ─────────────── 消费者（主线程） ───────────────

    def _chunk_index(self, fi: int) -> int:
        return max(bisect.bisect_right(self._starts, fi) - 1, 0)

    def _pop_frames(self, fis: list[int]) -> list:
        """批量弹出连续帧：同片内一次锁取尽可能多的帧，减少锁/等待次数。

        帧必须按序（fis 递增且落在同一片）；跨片边界逐片处理。已交付的
        帧会从分片缓存中删除，避免 NumPy 数组一直保留到整个 decoder 销毁。
        """
        out: list = []
        i = 0
        while i < len(fis):
            fi = fis[i]
            ci = self._chunk_index(fi)
            ch = self._chunks[ci]
            stalled = 0
            while True:
                with self._cv:
                    delivered = int(ch.get('delivered', ch.get('off', 0)))
                    if ch['data']:
                        n_take = 0
                        while (i + n_take < len(fis)
                               and n_take < len(ch['data'])
                               and self._chunk_index(fis[i + n_take]) == ci):
                            got, crop = ch['data'][n_take]
                            if got != fis[i + n_take]:
                                raise RuntimeError(
                                    'hybrid 序错位: want=%d got=%d'
                                    % (fis[i + n_take], got))
                            out.append(crop)
                            n_take += 1
                        if n_take:
                            del ch['data'][:n_take]
                            delivered += n_take
                            ch['delivered'] = delivered
                            ch['off'] = 0
                            i += n_take
                            if delivered >= len(ch['fis']):
                                ch['all_delivered'] = True
                                if ch.get('done') and ch.get('counted'):
                                    own = ch.get('owner')
                                    if own and own in self._unconsumed:
                                        self._unconsumed[own] -= 1
                                    ch['counted'] = False
                                ch['consumed'] = True
                            self._cv.notify_all()
                            break
                    if ch.get('done') and delivered < len(ch['fis']):
                        raise RuntimeError('hybrid 片数据不足')
                    if ch.get('done') and delivered >= len(ch['fis']):
                        if ch.get('counted'):
                            own = ch.get('owner')
                            if own and own in self._unconsumed:
                                self._unconsumed[own] -= 1
                            ch['counted'] = False
                        ch['consumed'] = True
                        break
                    if self._err:
                        raise RuntimeError('hybrid 解码失败: %r'
                                           % self._err[:1])
                if self._stop.is_set():
                    raise RuntimeError('hybrid 解码被取消')
                stalled += 1
                if stalled > 6000:   # ~20min 无进展防御
                    raise RuntimeError('hybrid 解码停滞')
                with self._cv:
                    self._cv.wait(0.05)
        return out

    # ─────────────── VideoReader 兼容接口 ───────────────

    def __len__(self):
        return len(self._gpu)

    def get_batch(self, frame_list, roi=None):
        arrs = self._pop_frames(list(frame_list))
        return _Batch(np.stack(arrs))

    def next_roi(self, x1, y1, x2, y2):
        """校准顺序流：与 get_batch 共享同一交付序。

        步长取采样网格 stride（缺省 1）。现役 hybrid 安全门要求 stride==1
        （_open_vr），但此接口的帧号推进必须与采样网格一致，否则放宽安全门
        后校准帧号会错位（曾为隐性缺陷：硬编码 +1）。
        """
        stride = max(1, int(getattr(self._ex, '_sample_stride', 1)))
        if self._seq_fi is None:
            self._seq_fi = self._starts[0] if self._starts else self._f0
        crop = self._pop_frames([self._seq_fi])[0]
        fi = self._seq_fi
        self._seq_fi = fi + stride
        return _Batch(crop)

    def seek_accurate(self, fi: int):
        """不支持外部 seek（接口显式化，DESIGN-REVIEW B4）。

        分片定位由生产者在片首完成，外部 seek 会与预取竞态；旧实现静默
        吞掉调用（接口冒充），现显式报错。引擎两条流水线已改为对 hybrid
        跳过 seek。
        """
        raise NotImplementedError(
            "HybridDecoder 不支持外部 seek：分片定位由生产者在片首完成；"
            "调用方应在 hybrid 模式下跳过 seek_accurate")

    def get_avg_fps(self):
        return self._gpu.get_avg_fps()

    def get_fps(self):
        try:
            return self._gpu.get_fps()
        except Exception:
            return self.get_avg_fps()

    def get_color_range(self):
        try:
            return self._gpu.get_color_range()
        except Exception:
            return 0

    def get_codec(self):
        try:
            return self._gpu.get_codec()
        except Exception:
            return ''

    def get_key_indices(self):
        try:
            return self._gpu.get_key_indices()
        except Exception:
            return []

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        current = threading.current_thread()
        threads = [getattr(self, '_cpu_thread', None),
                   *getattr(self, '_threads', [])]
        # join 带超时：收尾的目的是"不留活线程干扰下一任务"，但如果某个
        # producer 卡死，无超时 join 会把"线程泄漏"升级成"close 永久挂死"，
        # 后者更糟。超时后线程不再等待，名字留在 _unjoined 供诊断。
        unjoined = []
        for t in threads:
            if t is None or t is current:
                continue
            t.join(_CLOSE_JOIN_TIMEOUT)
            if t.is_alive():
                unjoined.append(t)
        self._unjoined = [t.name for t in unjoined]
        self._threads = []
        for reader in (getattr(self, '_cpu', None),
                       getattr(self, '_gpu', None)):
            close = getattr(reader, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
        if self._probe and self._probe_rows:
            rows, self._probe_rows = self._probe_rows, []
            self._dump_probe(rows)

    def _dump_probe(self, rows):
        """打印/落盘逐片时序；close 时调用（消费侧已完成，队列已空）。"""
        rows = sorted(rows)
        by_who: dict = {}
        for idx, who, nf, f0, f1, claim_t, done_t, seek, ct, cb in rows:
            by_who.setdefault(who, []).append(
                (idx, nf, f0, f1, claim_t, done_t, seek, ct, cb))
        print("\n[hybrid probe] 逐片时序 (claim→done)：", flush=True)
        for who in ("gpu", "cpu"):
            if who not in by_who:
                continue
            # lst 元组: (idx,nf,f0,f1,claim_t,done_t,seek,ct,cb)
            lst = by_who[who]
            n_fr = sum(r[1] for r in lst)
            prod = sum(r[5] - r[4] for r in lst)
            n_seek = sum(1 for r in lst if r[6] > 0.005)
            seek_t = sum(r[6] for r in lst)
            print(f"  [{who}] 片={len(lst)} 帧={n_fr} 生产耗时={prod:.3f}s "
                  f"seek次数={n_seek} seek总={seek_t:.3f}s", flush=True)
            for r in lst:
                idx, nf, f0, f1, claim_t, done_t, seek, ct, cb = r
                print(f"    #{idx} claim_by={cb} n={nf} [{f0}..{f1}] "
                      f"claim+{claim_t-ct:.3f}s produce={done_t-claim_t:.3f}s "
                      f"seek={seek:.3f}s", flush=True)
        if self._probe_csv:
            try:
                import csv
                with open(self._probe_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["idx", "producer", "n_frames", "f0", "f1",
                                "claim_t", "done_t", "seek_s", "claim_wait_s",
                                "claim_by"])
                    for idx, who, nf, f0, f1, t0, t1, seek, ct, cb in rows:
                        w.writerow([idx, who, nf, f0, f1,
                                    round(t0, 4), round(t1, 4),
                                    round(seek, 4), round(t0 - ct, 4), cb])
                print(f"[hybrid probe] CSV → {self._probe_csv}", flush=True)
            except Exception as e:
                print(f"[hybrid probe] CSV 写入失败: {e}", flush=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
