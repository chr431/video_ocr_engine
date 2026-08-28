"""真跳帧解码：整包丢弃非参考帧的可行性与收益（goal 1 决定性实验）。

## 背景
`ffmpeg -skip_frame noref` 只给 1.12x（跳过 49.8% 帧）——FFmpeg 的 noref
路径仍在做熵解码，只省重建。真正的机会是**连包都不送进解码器**。

## H.264 规范依据
`nal_ref_idc == 0` 的图**不会**被任何后续图用作帧间预测参考。
因此整包丢弃它对参考完整性**理论上安全**（与旧 PERFORMANCE.md §6 的
"按 pict_type 丢 B 帧"根本不同：High profile 下 B 帧可能 nal_ref_idc>0
即仍是参考帧，按 pict_type 丢必然破坏 DPB；按 nal_ref_idc 丢则不会）。

本脚本不依赖该理论推断，而是**实测三件事**：
  Q1 非参考帧占比（收益上限）
  Q2 整包丢弃后的真实解码提速
  Q3 安全性：剩余帧是否与全量解码的同帧逐字节一致

## 流程
  1. `ffmpeg -c copy -bsf:v h264_mp4toannexb -f h264` 导出 Annex-B 裸流
  2. Python 解析 NAL：按 first_mb_in_slice==0 切 access unit，读 nal_ref_idc
  3. 生成三份流：全量 / 丢全部非参考帧 / 丢"非参考且非采样点"（stride 网格）
  4. 分别解码为 gray rawvideo，计时 + 逐帧对齐校验

用法：
  python tools/_probe_drop_nonref.py --video X [--aus 4000] [--stride 8] [--reps 3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FFMPEG = r"D:\Software\ffmpeg8\bin\ffmpeg.exe"
FFPROBE = r"D:\Software\ffmpeg8\bin\ffprobe.exe"

SLICE_NON_IDR = 1
SLICE_IDR = 5
SLICE_AUX = 19


# ───────────────────────── Annex-B 解析 ─────────────────────────

def iter_nals(buf: bytes):
    """迭代 Annex-B NAL 单元：yield (start, end, nal_type, nal_ref_idc)。"""
    n = len(buf)
    i = 0
    # 找第一个起始码
    starts = []
    while i < n - 3:
        if buf[i] == 0 and buf[i + 1] == 0 and buf[i + 2] == 1:
            starts.append(i + 3)
            i += 3
        else:
            i += 1
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else n
        # 去掉尾部对齐零字节
        while e > s and buf[e - 1] == 0:
            e -= 1
        if e <= s:
            continue
        hdr = buf[s]
        yield s, e, (hdr & 0x1F), ((hdr >> 5) & 0x03)


def unescape_rbsp(b: bytes) -> bytes:
    """去模拟防止竞争字节 0x03（00 00 03 → 00 00）。"""
    out = bytearray()
    zeros = 0
    i = 0
    n = len(b)
    while i < n:
        c = b[i]
        if zeros >= 2 and c == 3 and i + 1 < n and b[i + 1] <= 3:
            zeros = 0
            i += 1
            continue
        out.append(c)
        zeros = zeros + 1 if c == 0 else 0
        i += 1
    return bytes(out)


class BitReader:
    __slots__ = ("b", "pos")

    def __init__(self, b: bytes):
        self.b = b
        self.pos = 0

    def u(self, n: int) -> int:
        v = 0
        for _ in range(n):
            byte = self.b[self.pos >> 3] if (self.pos >> 3) < len(self.b) else 0
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def ue(self) -> int:
        lz = 0
        while self.u(1) == 0:
            lz += 1
            if lz > 32:
                return 0
        if lz == 0:
            return 0
        return (1 << lz) - 1 + self.u(lz)


def first_mb_in_slice(payload: bytes) -> int:
    """解析 slice header 的 first_mb_in_slice（ue(v)），用于切 access unit。"""
    try:
        rbsp = unescape_rbsp(payload[1:])
        br = BitReader(rbsp)
        return br.ue()
    except Exception:
        return -1


def parse_aus(buf: bytes, max_aus: int = 0):
    """切 access unit：slice NAL 且 first_mb_in_slice==0 → 新 AU。

    返回 [(byte_start, byte_end, nal_ref_idc, is_idr), ...]；
    byte 区间含该 AU 前置的 SPS/PPS/SEI/AUD。
    未识别（解析失败）时按"每个 slice 一个 AU"兜底。
    """
    aus = []
    cur_start = None
    cur_ref = None
    cur_idr = False
    pending_nal = False
    for s, e, ntype, ref_idc in iter_nals(buf):
        is_slice = ntype in (SLICE_NON_IDR, SLICE_IDR)
        if is_slice:
            fmb = first_mb_in_slice(buf[s:e])
            new_au = (fmb == 0 or fmb < 0)
            if new_au and pending_nal:
                aus.append((cur_start, s, cur_ref, cur_idr))
                cur_start, cur_ref, cur_idr, pending_nal = None, None, False, False
            if cur_start is None:
                cur_start = s
            cur_ref = ref_idc
            cur_idr = (ntype == SLICE_IDR)
            pending_nal = True
        else:
            # 非 slice NAL 归属下一个 AU（SEI/SPS/PPS/AUD）
            if pending_nal:
                aus.append((cur_start, s, cur_ref, cur_idr))
                cur_start, cur_ref, cur_idr, pending_nal = None, None, False, False
            if cur_start is None:
                cur_start = s
    if pending_nal and cur_start is not None:
        aus.append((cur_start, len(buf), cur_ref, cur_idr))
    if max_aus and len(aus) > max_aus:
        # 截取到 max_aus 个完整 AU
        end = aus[max_aus - 1][1]
        aus = aus[:max_aus]
        buf = buf[:end]
    return aus, buf


# ───────────────────────── 解码与校验 ─────────────────────────

def decode_time(src: Path, threads: int,
                opts: list[str] | None = None) -> float:
    """`-f null` 纯解码计时（无输出 I/O）。

    注意：早先用 `-f rawvideo` 计时被落盘 I/O 严重污染（1080p×4000 帧 =
    8.3GB），导致丢帧方案的收益被**低估**（丢帧同时少了 I/O）。计时必须走
    null muxer；像素对齐校验才用 rawvideo，且只做一次。
    """
    args = [FFMPEG, "-hide_banner", "-nostats", "-loglevel", "error",
            *(opts or []), "-threads", str(threads), "-i", str(src),
            "-f", "null", "-"]
    t0 = time.perf_counter()
    p = subprocess.run(args, capture_output=True, text=True, errors="replace")
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"decode failed rc={p.returncode}\n"
                           f"{(p.stderr or '')[-800:]}")
    return dt


def decode_gray(src: Path, out: Path, threads: int) -> tuple[float, int, int, int]:
    """解码为 gray rawvideo（仅用于对齐校验，计时不可用——见 decode_time）。"""
    args = [FFMPEG, "-hide_banner", "-nostats", "-loglevel", "error", "-y",
            "-threads", str(threads), "-i", str(src),
            "-pix_fmt", "gray", "-f", "rawvideo", str(out)]
    t0 = time.perf_counter()
    p = subprocess.run(args, capture_output=True, text=True, errors="replace")
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"decode failed rc={p.returncode}\n"
                           f"{(p.stderr or '')[-800:]}")
    sz = out.stat().st_size
    # 从 ffmpeg 拿分辨率
    wh = probe_wh(src)
    w, h = wh
    return dt, (sz // (w * h) if w and h else 0), w, h


def probe_wh(src: Path) -> tuple[int, int]:
    p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height",
                        "-of", "csv=p=0:s=,", str(src)],
                       capture_output=True, text=True, errors="replace")
    try:
        a, b = p.stdout.strip().split(",")
        return int(a), int(b)
    except Exception:
        return 0, 0


def frames(path: Path, w: int, h: int):
    """按帧产出 bytes。"""
    fs = w * h
    with open(path, "rb") as f:
        while True:
            b = f.read(fs)
            if len(b) < fs:
                return
            yield b


def frame_hashes(path: Path, w: int, h: int) -> list[bytes]:
    return [hashlib.blake2b(b, digest_size=16).digest()
            for b in frames(path, w, h)]


def subsequence_match(sub: list[bytes], full: list[bytes]) -> tuple[bool, int]:
    """sub 是否为 full 的**顺序子序列**（逐元素相等）。

    丢帧后输出的帧必须与全量解码中同序号帧逐字节一致；
    允许"跳过"，不允许"错位/内容不同"。
    """
    j = 0
    for h in sub:
        while j < len(full) and full[j] != h:
            j += 1
        if j >= len(full):
            return False, len(sub)
        j += 1
    return True, len(sub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--aus", type=int, default=4000, help="分析前 N 个 access unit")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0,
                    help="单值，或用 --thread-sweep 扫多值")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--keep", action="store_true", help="保留中间产物")
    ap.add_argument("--thread-sweep", default="",
                    help="逗号分隔的线程数列表，如 4,8,16,32；扫出收益-线程曲线")
    ap.add_argument("--extra-opts", default="",
                    help="追加一组解码器选项，空格分隔，如 '-skip_idct noref'")
    a = ap.parse_args()

    th = a.threads if a.threads > 0 else __import__("os").cpu_count() or 8
    sweep = [int(x) for x in a.thread_sweep.split(",") if x.strip()] or [th]
    for si, th in enumerate(sweep):
        rc = _run_one(a, th, si)
        if rc != 0:
            return rc
    return 0


def _run_one(a, th: int, sweep_idx: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dropnr_"))
    try:
        # 1) 导出 Annex-B
        annexb = tmp / "full.h264"
        t0 = time.perf_counter()
        p = subprocess.run(
            [FFMPEG, "-hide_banner", "-nostats", "-loglevel", "error", "-y",
             "-i", a.video, "-c", "copy", "-bsf:v", "h264_mp4toannexb",
             "-f", "h264", str(annexb)],
            capture_output=True, text=True, errors="replace")
        if p.returncode != 0:
            print("Annex-B 导出失败：", (p.stderr or "")[-600:])
            return 2
        print(f"[1] Annex-B 导出 {annexb.stat().st_size/1e6:.1f}MB "
              f"({time.perf_counter()-t0:.2f}s)")

        # 2) 解析 AU
        buf = annexb.read_bytes()
        aus, buf = parse_aus(buf, a.aus)
        total = len(aus)
        nonref = sum(1 for _s, _e, r, _i in aus if r == 0)
        idr = sum(1 for _s, _e, _r, i in aus if i)
        print(f"[2] AU 解析：{total} 个  |  非参考(nal_ref_idc=0) {nonref} "
              f"({nonref/total:.1%})  |  IDR {idr}  |  平均 GOP {total/max(idr,1):.0f}")

        # 3) 生成三份流
        def build(name: str, keep_pred) -> Path:
            parts = []
            for s, e, r, i in aus:
                if keep_pred(r, i):
                    parts.append(buf[s:e])
            out = tmp / f"{name}.h264"
            out.write_bytes(b"\x00\x00\x00\x01".join(
                [b""] + parts) if parts else b"")
            return out

        idx = {id(au): k for k, au in enumerate(aus)}
        counter = [0]

        def keep_all(r, i):
            return True

        def keep_ref(r, i):
            return r != 0

        def keep_ref_or_sample(r, i):
            k = counter[0]
            counter[0] += 1
            return (r != 0) or (k % a.stride == 0)

        variants = {
            "full": (build("full", keep_all), total),
            "drop_nonref": (build("drop_nonref", keep_ref), None),
            f"drop_nonref_stride{a.stride}": (
                build(f"drop_stride{a.stride}", keep_ref_or_sample), None),
        }

        # 4) 计时（-f null，无 I/O）：变体 × 解码器选项矩阵
        optsets = [
            ("plain", []),
            ("noloop_all", ["-skip_loop_filter", "all"]),
            ("noloop_noref", ["-skip_loop_filter", "noref"]),
        ]
        if a.extra_opts:
            optsets.append(("custom", a.extra_opts.split()))
        print(f"\n[3] 解码计时（-f null，threads={th}, reps={a.reps}）")
        base_t = base_n = 0
        report = {"video": a.video, "aus": total, "nonref": nonref,
                  "nonref_pct": nonref / total, "idr": idr,
                  "threads": th, "stride": a.stride, "variants": {}}

        def count_keep(pred) -> int:
            counter[0] = 0
            return sum(1 for _s, _e, r, i in aus if pred(r, i))

        counts = {"full": total,
                  "drop_nonref": count_keep(keep_ref),
                  f"drop_nonref_stride{a.stride}": count_keep(keep_ref_or_sample)}

        for name, (path, _) in variants.items():
            for oname, opts in optsets:
                ts = [decode_time(path, th, opts) for _ in range(a.reps)]
                med = statistics.median(ts)
                n = counts[name]
                if name == "full" and oname == "plain":
                    base_t, base_n = med, n
                speed = base_t / med if med > 0 else 0
                key = f"{name} :: {oname}"
                print(f"  {key:44s} {med:6.3f}s  "
                      f"帧={n:5d} ({n / base_n if base_n else 0:5.1%})  "
                      f"提速={speed:4.2f}x")
                report["variants"][key] = {
                    "median_s": round(med, 3), "frames": n,
                    "kept_pct": n / base_n if base_n else 0,
                    "runs": [round(t, 3) for t in ts],
                    "speedup": round(speed, 3), "opts": opts}

        # 对齐校验与解码选项无关（只要流相同），扫描时只做首轮
        if sweep_idx == 0:
            print(f"\n[4] 像素对齐校验（rawvideo gray，各一次）")
            base_hashes = None
            for name, (path, _) in variants.items():
                o = tmp / f"{name}.gray"
                _dt, n, w, h = decode_gray(path, o, th)
                hs = frame_hashes(o, w, h)
                if name == "full":
                    base_hashes = hs
                    print(f"  {name:26s} 参考帧 {len(hs)}")
                    continue
                ok, matched = subsequence_match(hs, base_hashes)
                print(f"  {name:26s} 帧 {len(hs):5d}  对齐="
                      f"{'OK' if ok else 'FAIL'} ({matched}/{len(hs)})")
                for k in report["variants"]:
                    if k.startswith(name + " ::"):
                        report["variants"][k]["aligned"] = bool(ok)
                        report["variants"][k]["matched"] = matched
                o.unlink(missing_ok=True)

        out = Path(__file__).with_name("_probe_drop_nonref.json")
        old = {}
        if out.exists() and sweep_idx > 0:
            try:
                old = json.loads(out.read_text(encoding="utf-8"))
            except Exception:
                old = {}
        old[f"threads={th}"] = report
        out.write_text(json.dumps(old, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"明细落盘：{out}  (threads={th})")
        if a.keep:
            print(f"中间产物：{tmp}")
        return 0
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
