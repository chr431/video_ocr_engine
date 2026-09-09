import sys, time
sys.path.insert(0, ".")
from decord import VideoReader, cpu
V = r"D:\Videos\racelog_test\test6.mp4"; ROI=(841,994,950,1027)
vr = VideoReader(V, ctx=cpu(0), num_threads=24, output_format="gray")
vr.get_batch([0], roi=ROI).asnumpy()  # 预热
best = 0
for _ in range(3):
    t0 = time.perf_counter()
    for i in range(0, 4000, 64):
        vr.get_batch(list(range(i, min(i+64,4000))), roi=ROI).asnumpy()
    dt = time.perf_counter()-t0
    best = max(best, 4000/dt)
print(f"NT=24 顺序 4000 帧 最优 3 轮: {best:.0f} fps")
