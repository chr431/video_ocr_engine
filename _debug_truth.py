import csv
from pathlib import Path

def load(path):
    rows=[]
    with open(path, encoding='utf-8-sig', newline='') as f:
        r=csv.reader(f)
        next(r,None)
        for row in r:
            rows.append(tuple(row))
    return rows

t=load(r'D:\Videos\text_video_test\text_test_truth.csv')
d=load(r'D:\Videos\text_video_test\text_test_subtitles.csv')
print('truth:', [repr(x) for x in t[:3]])
print('default:', [repr(x) for x in d[:3]])
set_t={x[1] for x in t}
print('first default in truth set?', d[0][1] in set_t)
for i,x in enumerate(d[:20]):
    if x[1] not in set_t:
        print('mismatch', i, repr(x[1]))
        break
