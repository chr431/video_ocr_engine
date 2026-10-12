"""hybrid 验收口径（2026-10-11 性能收口轮）：三码全片（23970 帧）引擎 e2e。

分子 = hybrid e2e（extract 墙钟，含校准/管线开销）；分母 = 同会话同口径
纯臂 e2e 之和（cpu 后端 + nvdec 后端）。warmup 1 + 2 reps 中位。
C-76 后基线：h264 93.2% / hevc 94.9% / av1 94.1%（验收线 ≥90%）。
"""
import os, statistics, sys, time, json
sys.path.insert(0,'tools'); sys.path.insert(0,'.')
os.environ['GPU_PIPELINE']='1'
os.environ['DECORD_HYBRID_STATS']='1'
from _measure_lib import ROI, VIDS
from video_ocr_engine import FieldExtractor

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "tools"))
# R-perf-quiet：后台负载门禁——超阈立即退出，不浪费整轮测量
from _measure_lib import BackgroundLoadError, assert_machine_quiet

try:
    assert_machine_quiet()
except BackgroundLoadError as _e:  # noqa: BLE001 — 门禁语义：超阈退出
    print(_e)
    raise SystemExit(2)

def run(vid, backend, reps=2):
    N = 23970
    xs, spans, stats = [], None, []
    for i in range(reps+1):
        ex = FieldExtractor(VIDS[vid], ROI['test6'], frame_start=0, frame_end=None,
                            decode_backend=backend, ocr_backend='tensorrt', sample_stride=1)
        t0=time.perf_counter(); r=ex.extract(); w=time.perf_counter()-t0
        n = r.meta.get('frames_total') or N
        xs.append(n/w)
        spans=(r.meta.get('report') or {}).get('spans') or {}
        if i>0 and backend=='hybrid':
            dec = spans.get('decode') or {}
            stats.append({'n':n,'spans':{k:dec.get(k) for k in ('sum','count') if isinstance(dec,dict)}})
    return statistics.median(xs[1:]), spans, stats

out={}
for code, vid in (('h264','test6_h264'),('hevc','test6_hevc'),('av1','test6_av1')):
    rc,_,_ = run(vid,'cpu'); rg,_,_ = run(vid,'nvdec'); rh,spans,st = run(vid,'hybrid')
    out[code]={'cpu':round(rc),'nvdec':round(rg),'sum':round(rc+rg),'hybrid':round(rh),'pct':round(100*rh/(rc+rg),1)}
    print(code, out[code], flush=True)
print(json.dumps(out,indent=1))
