"""统一驱动骨架（P1 宿主/GPU 实现统一轮）：两后端同构生命周期的单出处。

v1 的 run_host_pipeline / run_gpu_pipeline 生命周期逐段同构（open →
fps/帧区间 → begin_reading → OCR 会话 → 校准 → 帧流消费 → 收尾），
差异只在「帧从哪来、统计在哪算、代表帧怎么交付」——这正是 v2 §6.3
预留的 lane 注入点（FrameSource/SegmentBackend 端口的驱动侧对偶）。
本骨架收同构段（_run_common 只收了 setup 的纯函数段，这里收完整
生命周期），两后端退化为 lane 策略：calibrate / frame_items / emit /
similar / abort / fallback / stop_consume / after_stream / release
（宿主侧多为 no-op）。算法原语唯一出处仍在 segmentation.py
（C-32 不变）；「P2c 对齐」式的两路径相位集漂移自此结构性消失
（键集一致性由 tests/pipeline/test_phase_key_registry.py 守护）。

计时口径统一（只影响遥测 span、不影响产物）：calibrate span 的 t0
取在 OCR 会话启动**之后**（旧宿主口径；旧 GPU 口径含会话启动）——
两路径自此同义，L1 per_phase 差分可比。
"""
from __future__ import annotations

import logging
import time

from video_ocr_engine.domain.segmentation import SegmentStateMachine

logger = logging.getLogger(__name__)


def run_segment_pipeline(spec, res, lane, ocr_engines=None,
                         preopened_vr=None):
    """资源阶段 → 会话 → 校准 → 帧流消费 → 收尾（宿主/GPU 共用）。

    spec 取两 RunSpec 的公共字段（帧区间/采样/C/回调/指标）；res 由
    调用方构造（HostRunResult / GpuRunResult），差异字段由 lane 的
    fallback/crops 等回填。宿主 lane 的 calibrate 恒 (True, th)，
    fallback 分支只有 GPU lane 会走。
    """
    from ._run_common import begin_reading, compute_frames, ensure_fps
    _MET = spec.metrics
    _t_open = time.perf_counter()
    vr = preopened_vr if preopened_vr is not None else spec.open_vr()
    lane.after_open(vr)
    ensure_fps(spec, vr)
    res.fps = spec.fps_box[0]
    total = len(vr)
    frames = compute_frames(spec, total)
    if not frames:
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛（资源由进程回收）
        raise ValueError(
            f"帧区间为空: frame_start={spec.frame_start}, "
            f"frame_end={spec.frame_end}, total={total}")
    try:
        begin_reading(vr, spec, frames)
    except BaseException:
        logger.debug("begin_reading 失败进入回退清理", exc_info=True)
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛
        raise
    if spec.prof_end is not None:
        spec.prof_end('producer', 'open_and_fps', _t_open)
    # L1 资源边界（§8.6 r5）：粗相位结束处各采一次，差分见 run report。
    # 记的是**进程级**用量，OCR 与解码并发时核数会互相计入——这是"该相位
    # 平均并行核数"的本意（不是单相位隔离）。off 档此调用为空操作。
    _MET.checkpoint('open')
    # OCR 会话提前到校准前启动：worker 线程内构建引擎，与校准并行重叠；
    # 引擎就绪前 emit 自动走 host 回退，语义不变。
    try:
        ocr_session = spec.start_ocr_session(ocr_engines)
    except BaseException:
        logger.debug("OCR 会话启动失败进入清理", exc_info=True)
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛（资源由进程回收）
        raise
    results = ocr_session.results
    ocr_err = ocr_session.err
    ocr_wall = ocr_session.wall

    # ── 校准（宿主 Otsu / GPU 设备侧直方图；False = 形状不符回退宿主）──
    # P2c 卫生：calibrate 的 t0 从 open 收口处起算（不含 open 相位），
    # 与 setup 不重叠双计。
    _t_calib = time.perf_counter()
    try:
        _calib_ok, _th = lane.calibrate(spec, vr, frames, ocr_session)
    except BaseException:
        logger.debug("校准相位异常进入清理", exc_info=True)
        lane.abort(ocr_session)
        try:
            vr.close()
        except Exception:
            pass  # 清理路径：close 失败无需上抛（资源由进程回收）
        raise
    if not _calib_ok:
        # 形状不符等：回退宿主（C10 语义）——会话收尾（worker 归还引擎）、
        # 释放设备侧临时缓冲，但 **reader 不 close**：宿主路径复用已打开
        # 的 reader（get_batch 随机访问无消费状态，免二次打开/解码器悬挂）。
        return lane.fallback(res, ocr_session, ocr_engines, vr)
    res.bin_thresh = _th
    lane.after_calibrate(_th, ocr_session)   # F-4：阈值即时回写等收尾装配
    if spec.prof_end is not None:
        # 键统一为 calib_total（旧 GPU 路径的 gpu_calib_total 与其映射
        # 同一指标 pipeline.calibrate——双键是两驱动各自调用点的历史
        # 残留，P1 合一后单调用点单键）
        spec.prof_end('producer', 'calib_total', _t_calib)
    _MET.checkpoint('calibrate')

    # ── 帧流消费（宿主=线程内生成器；GPU=生产者线程+队列包成生成器）──
    from .._helpers import _decode_progress_pct
    machine = SegmentStateMachine(
        frames, C=spec.C,
        on_emit=lambda seg, rep, frac: lane.emit(seg, rep, frac),
        on_similar=lane.similar,
        on_cancel=spec.cancel,
        on_progress=lambda k, frac: spec.progress(
            f'[{spec.backend_label()}] {lane.progress_verb}: '
            f'{k}/{len(frames)}',
            _decode_progress_pct(frac)),
        debug_tag=lane.debug_tag)
    segs: list = []
    t0 = time.perf_counter()
    try:
        k = 0
        for fi, sharp, feed_kwargs in lane.frame_items(
                spec, vr, frames, ocr_session):
            _t_feed = time.perf_counter()
            machine.feed(k, fi, sharp, **feed_kwargs)
            if spec.prof_end is not None:
                spec.prof_end('producer', 'consume_feed', _t_feed)
            k += 1
        lane.after_stream()
        machine.finish()
        segs = machine.segs
    finally:
        lane.stop_consume()
        _t_consume_end = time.perf_counter()
        res.timing['decode'] = _t_consume_end - t0
        _MET.checkpoint('decode')
        if spec.prof_end is not None:
            spec.prof_end('producer', 'consumer_total', t0)
        try:
            ocr_session.finish()
        except BaseException:
            logger.debug("ocr_session.finish 清理忽略异常", exc_info=True)
        res.timing['ocr_tail'] = time.perf_counter() - _t_consume_end
        # fork 遥测穿透（2026-09-17）：close 前取快照（原子计数器，此后
        # 解码器将销毁）；非 hybrid 解码器方法缺席 → 保持 None。
        _fs = getattr(vr, "hybrid_stats", None)
        if _fs is not None:
            try:
                res.fork_stats = _fs() or None
            except Exception:
                logger.debug("fork 遥测抓取失败忽略", exc_info=True)
        try:
            vr.close()   # hybrid 探针/资源释放：显式停止生产者线程
        except Exception:
            pass  # 清理路径：close 失败无需上抛（资源由进程回收）
        lane.release()
    if ocr_err:
        # C4：补"OCR worker 失败"上下文并保留原始异常链
        raise RuntimeError(f"OCR worker 失败: {ocr_err[0]!r}") from ocr_err[0]
    # PI-15：逐段计数一律在此一次性上报（run 内不再逐段调 Python）
    _MET.counter('segment.segments', len(segs))
    lane.report_counters()
    res.timing['ocr'] = ocr_wall[0]
    _MET.checkpoint('ocr')
    res.frames = frames
    res.segs = segs
    res.n_segments = len(segs)
    res.crops = lane.crops
    res.texts = [results[i][0] for i in range(len(segs))]
    res.confs = [results[i][1] for i in range(len(segs))]
    res.rep_frames = [results[i][2] for i in range(len(segs))]
    return res
