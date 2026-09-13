"""ExtractionPool —— 跨视频互补配对（层5，redesign 分支）。

动机（C-01/C-07/C-08）：单视频内 hybrid 已接近两解码器并联和的物理
上限；**跨视频**配对（h264 走 CPU 软解、hevc/av1 走 NVDEC）没有单视频
混跑的 SMT/LLC 干扰税——两套资源真·独立。批量场景的吞吐上限在此。

调度（实测驱动的两点，`_probe_pool_pairing.py`）：
- **编码感知**：h264→cpu（软解快 1.7~2.9×），hevc/av1→nvdec；
  naive 交替会把 hevc 分给 CPU 慢腿（三件套实测 -0.7% 无增益）。
- **LPT（最长优先）**：按帧数降序派发。3 项 2 工人时最长任务后启
  = makespan 直接变差（实测双峰 27.2/34.0s 的成因）；编码感知 + LPT
  后稳定 ~27.2s（约 -20%）。

线程模型：每视频一线程（decord GIL 释放充分）；max_workers 缺省
min(n,2)——NVDEC 单会话最优（C-01），TRT OCR 共享引擎池。
结果按**传入顺序**返回（LPT 只影响执行序）。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from .. import FieldExtractor

logger = logging.getLogger(__name__)


def _probe_video(path: str) -> tuple:
    """(codec, 帧数)——编码感知派工与 LPT 的依据；失败返回 ('', 0)。"""
    try:
        import decord as _d
        vr = _d.VideoReader(path, ctx=_d.cpu(0))
        codec = str(vr.get_codec() or '').lower()
        n = len(vr)
        vr.close()
        return codec, n
    except Exception:  # noqa: BLE001
        return '', 0


def run(items: list, *, backends: str = 'pair',
        max_workers: int = None) -> list:
    """批量抽取：编码感知互补配对 + LPT 派发。

    items: FieldExtractor(...) 的 kwargs（video 必填；decode_backend
           可留空由策略分配）。
    backends: 'pair'（缺省，编码感知+LPT）| 'nvdec' | 'cpu' |
              显式列表（与 items 等长，逐项指定，不做重排）。
    """
    items = [dict(it) for it in items]
    if isinstance(backends, str):
        if backends == 'pair':
            probe = [_probe_video(it.get('video_path')
                                  or it.get('video', '')) for it in items]
            plan = ['cpu' if c == 'h264' else 'nvdec' for c, _ in probe]
            order = sorted(range(len(items)), key=lambda i: -probe[i][1])
            work = [(order[i], items[order[i]], plan[order[i]])
                    for i in range(len(items))]
        else:
            work = [(i, it, backends) for i, it in enumerate(items)]
    else:
        if len(backends) != len(items):
            raise ValueError('backends 列表必须与 items 等长')
        work = [(i, it, backends[i]) for i, it in enumerate(items)]
    if max_workers is None:
        max_workers = min(len(items), 2)
    results: list = [None] * len(items)
    errors: list = [None] * len(items)
    t0 = time.perf_counter()

    def _work(idx: int, kw: dict, backend: str) -> None:
        kw = dict(kw)
        if 'video' in kw:
            kw['video_path'] = kw.pop('video')   # 探针/调用方用 video 简写
        kw.setdefault('decode_backend', backend)
        st = time.perf_counter()
        try:
            ex = FieldExtractor(**kw)
            results[idx] = ex.extract()
        except Exception as e:  # noqa: BLE001
            errors[idx] = e
            logger.exception('ExtractionPool 第 %d 项失败（backend=%s）',
                             idx, backend)
        finally:
            logger.info('pool item %d backend=%s wall=%.3fs',
                        idx, backend, time.perf_counter() - st)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(lambda w: _work(*w), work))
    logger.info('pool done: %d items wall=%.3fs', len(items),
                time.perf_counter() - t0)
    for e in errors:
        if e is not None:
            raise e
    return results
