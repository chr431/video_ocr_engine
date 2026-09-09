"""seek 落点正确性对照：seek_accurate(s)+取帧 的像素 vs 顺序解码到 s 的像素。"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decord import VideoReader, cpu  # noqa: E402

VIDEO = r"D:\Videos\racelog_test\test6.mp4"
ROI = (841, 994, 950, 1027)


def h(arr) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


vr_seq = VideoReader(VIDEO, ctx=cpu(0), num_threads=4, output_format="gray")
keys = list(vr_seq.get_key_indices())
targets = []
for k in keys[1:4]:
    targets += [k, k + 60, k + 120, k + 240]
targets = [min(t, len(vr_seq) - 1) for t in targets]

# 顺序真值
truth = {}
for s in sorted(set(targets)):
    vr_seq.seek_accurate(0) if s == 0 else None
    batch = vr_seq.get_batch(list(range(s, s + 1)), roi=ROI).asnumpy()
    truth[s] = h(batch[0])
print("顺序真值哈希完成（顺序=逐目标小窗口，窗口间跳变走同一内部 seek 路径，"
      "可能引入同样的问题——改用一次性大批量读取重建真值）")

# 更严格真值：一次读 k..k+250 连续块，块内落点绝无 seek
truth2 = {}
for k in keys[1:4]:
    block = vr_seq.get_batch(list(range(k, k + 241)), roi=ROI).asnumpy()
    for d in (0, 60, 120, 240):
        truth2[k + d] = h(block[d])

vr_seek = VideoReader(VIDEO, ctx=cpu(0), num_threads=4, output_format="gray")
ok = bad = 0
for s in targets:
    vr_seek.seek_accurate(s)
    got = h(vr_seek.get_batch([s], roi=ROI).asnumpy()[0])
    exp = truth2[s]
    tag = "OK " if got == exp else "BAD"
    ok, bad = ok + (got == exp), bad + (got != exp)
    print(f"  s={s:6d} seek落点={got} 顺序真值={exp}  {tag}")
print(f"合计: 一致 {ok} / 不一致 {bad}")
