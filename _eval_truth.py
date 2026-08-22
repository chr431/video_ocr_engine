import csv, sys
from pathlib import Path

BASE = Path(r'D:\Videos\text_video_test')
TRUTH = BASE / 'text_test_truth.csv'

def load(path):
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            t = row[0].strip()
            text = row[1].strip()
            # normalize time h:m:s or hh:mm:ss
            parts = t.split(':')
            sec = int(parts[-1])
            if len(parts) >= 3:
                sec += int(parts[-2]) * 60 + int(parts[-3]) * 3600
            elif len(parts) == 2:
                sec += int(parts[-2]) * 60
            rows.append((sec, text))
    return rows

truth = load(TRUTH)
default = load(BASE / 'text_test_subtitles.csv')
binary = load(BASE / 'text_test_binary_subtitles.csv')

print('counts truth/default/binary:', len(truth), len(default), len(binary))

def match_by_text(ocr, truth):
    truth_list = [t for _, t in truth]
    ocr_list = [t for _, t in ocr]
    # sequence containment: count truth texts found in OCR order (greedy)
    idx = 0
    matched = []
    for t in truth_list:
        found = None
        for j in range(idx, len(ocr_list)):
            if ocr_list[j] == t:
                found = j
                break
        if found is not None:
            matched.append(t)
            idx = found + 1
        else:
            matched.append(None)
    return matched

for name, ocr in [('default', default), ('binary', binary)]:
    truth_set = {t for _, t in truth}
    ocr_set = {t for _, t in ocr}
    present = len(truth_set & ocr_set)
    missing = sorted(truth_set - ocr_set)
    extra = sorted(ocr_set - truth_set)
    print(f'--- {name} ---')
    print('set present:', present, '/', len(truth_set), 'missing:', len(missing), 'extra:', len(extra))
    print('missing sample:', missing[:10])
    print('extra sample:', extra[:10])

    # row-level exact by timestamp
    truth_by_time = {}
    for sec, text in truth:
        truth_by_time.setdefault(sec, set()).add(text)
    exact = 0
    for sec, text in ocr:
        if text in truth_by_time.get(sec, set()):
            exact += 1
    print('exact time+text rows:', exact, '/', len(ocr))

# Also sequence comparison with tolerance? print exact differences for truth with same texts? not now
