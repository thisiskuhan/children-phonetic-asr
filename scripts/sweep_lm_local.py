#!/usr/bin/env python3
"""Local CPU-only LM weight sweep using cached log-probs from RunPod."""

import ctypes
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

NPZ_PATH = ROOT / "runs" / "run_16_nst_on_best_of_run_15" / "lm_sweep_logprobs.npz"
LM_PATH = ROOT / "submission" / "src" / "lm_trigram.bin"
SO_PATH = ROOT / "submission" / "src" / "beam_decode.so"
OUT_DIR = ROOT / "data" / "reports"

BLANK = 0
UNK = 1
BEAM_WIDTH = 50
LENGTH_ALPHA = 0.0
BEAM_WORKERS = 8
BATCH_SIZE = 64

LM_WEIGHTS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
              0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]


def ts():
    return time.strftime("%H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    m, n = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


def corpus_per(hyps, refs):
    total_edits = 0
    total_ref = 0
    for h, r in zip(hyps, refs):
        total_edits += edit_distance(h, r)
        total_ref += len(r)
    return total_edits / total_ref if total_ref > 0 else 0.0


class ParallelBeamDecoder:
    def __init__(self, so_path, n_workers, lm_table=None, lm_weight=0.0, unk_id=1):
        self._so_path = so_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._idx = 0
        self._so_copies = []
        self._pool = ThreadPoolExecutor(max_workers=n_workers)
        self._lm_table = lm_table
        self._lm_weight = lm_weight
        self._unk_id = unk_id
        self._use_lm = lm_table is not None and lm_weight > 0.0

    def set_lm_weight(self, w):
        self._lm_weight = w
        self._use_lm = self._lm_table is not None and w > 0.0

    def _get_lib(self):
        if hasattr(self._local, 'lib'):
            return self._local.lib, self._local.has_lm, \
                   self._local.out_buf, self._local.out_len
        with self._lock:
            idx = self._idx
            self._idx += 1
        tmp = f'/tmp/_beam_local_{idx}.so'
        shutil.copy2(self._so_path, tmp)
        self._so_copies.append(tmp)
        lib = ctypes.CDLL(tmp)
        lib.ctc_beam_search.restype = ctypes.c_int
        lib.ctc_beam_search.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        has_lm = False
        try:
            lib.ctc_beam_search_lm.restype = ctypes.c_int
            lib.ctc_beam_search_lm.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_double, ctypes.c_double,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_double, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ]
            has_lm = True
        except AttributeError:
            pass
        out_buf = (ctypes.c_int * 512)()
        out_len = ctypes.c_int(0)
        self._local.lib = lib
        self._local.has_lm = has_lm
        self._local.out_buf = out_buf
        self._local.out_len = out_len
        return lib, has_lm, out_buf, out_len

    def _decode_one(self, args):
        lp, beam_width, blank, length_alpha = args
        lib, has_lm, out_buf, out_len = self._get_lib()
        T, V = lp.shape
        lp_c = np.ascontiguousarray(lp, dtype=np.float32)
        out_len.value = 0

        if self._use_lm and has_lm:
            lm = np.ascontiguousarray(self._lm_table, dtype=np.float32)
            lib.ctc_beam_search_lm(
                lp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                T, V, beam_width, blank,
                length_alpha, 1e9,
                lm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self._lm_weight, self._unk_id,
                out_buf, ctypes.byref(out_len),
            )
        else:
            lib.ctc_beam_search(
                lp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                T, V, beam_width, blank,
                length_alpha, 1e9,
                out_buf, ctypes.byref(out_len),
            )
        return list(out_buf[:out_len.value])

    def decode_batch(self, lp_slices, beam_width, blank, length_alpha):
        args = [(lp, beam_width, blank, length_alpha) for lp in lp_slices]
        futures = [self._pool.submit(self._decode_one, a) for a in args]
        return [f.result() for f in futures]

    def shutdown(self):
        self._pool.shutdown(wait=False)
        for p in self._so_copies:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def main():
    t_start = time.time()

    # ── Load cached log-probs ────────────────────────────────────
    log(f"Loading cached log-probs from {NPZ_PATH}...")
    cached = np.load(str(NPZ_PATH), allow_pickle=True)
    all_lp = list(cached["log_probs"])
    all_ref_ids = [list(r) for r in cached["ref_ids"]]
    log(f"Loaded {len(all_lp)} samples")

    # ── Load LM ──────────────────────────────────────────────────
    assert LM_PATH.exists(), f"LM not found: {LM_PATH}"
    raw = np.fromfile(str(LM_PATH), dtype=np.float32)
    V = round(len(raw) ** (1.0 / 3.0))
    assert V * V * V == len(raw), f"LM size mismatch: {len(raw)} != {V}^3"
    lm_table = raw.reshape(V, V, V)
    log(f"LM loaded: V={V}, {LM_PATH.stat().st_size / 1024:.0f} KB")

    # ── Compile .so if needed ────────────────────────────────────
    if not SO_PATH.exists():
        import subprocess
        c_path = SO_PATH.with_suffix('.c')
        assert c_path.exists(), f"No .so or .c found"
        log("Compiling beam_decode.so...")
        subprocess.check_call(
            ["gcc", "-O3", "-shared", "-fPIC", "-o", str(SO_PATH), str(c_path), "-lm"],
            timeout=30,
        )
        log("Compiled OK")

    # ── Init decoder ─────────────────────────────────────────────
    decoder = ParallelBeamDecoder(
        str(SO_PATH), BEAM_WORKERS,
        lm_table=lm_table, lm_weight=0.0, unk_id=UNK)

    # ── Greedy baseline ──────────────────────────────────────────
    log("Computing greedy baseline...")
    greedy_hyps = []
    for lp in all_lp:
        ids = np.argmax(lp, axis=-1)
        collapsed = []
        prev = -1
        for tok in ids:
            if tok != BLANK and tok != UNK and tok != prev:
                collapsed.append(int(tok))
            prev = tok
        greedy_hyps.append(collapsed)
    greedy_per = corpus_per(greedy_hyps, all_ref_ids)
    log(f"Greedy PER: {greedy_per:.6f}")

    # ── Sweep ────────────────────────────────────────────────────
    results = [{"lm_weight": "greedy", "per": greedy_per, "time_sec": 0}]

    for lm_w in LM_WEIGHTS:
        decoder.set_lm_weight(lm_w)
        log(f"Sweeping LM_WEIGHT={lm_w:.2f}...")
        t_sw = time.time()

        hyps_ids = []
        for b_start in range(0, len(all_lp), BATCH_SIZE):
            batch_lp = all_lp[b_start:b_start + BATCH_SIZE]
            ids_list = decoder.decode_batch(batch_lp, BEAM_WIDTH, BLANK, LENGTH_ALPHA)
            hyps_ids.extend(ids_list)

        sweep_per = corpus_per(hyps_ids, all_ref_ids)
        elapsed = time.time() - t_sw
        results.append({
            "lm_weight": lm_w,
            "per": sweep_per,
            "time_sec": round(elapsed, 1),
        })
        log(f"  LM_WEIGHT={lm_w:.2f} → PER={sweep_per:.6f} ({elapsed:.1f}s)")

    decoder.shutdown()

    # ── Results ──────────────────────────────────────────────────
    beam_results = [r for r in results if r["lm_weight"] != "greedy"]
    best = min(beam_results, key=lambda r: r["per"])
    greedy_r = results[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "sweep_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "val_samples": len(all_lp),
        "beam_width": BEAM_WIDTH,
        "length_alpha": LENGTH_ALPHA,
        "beam_workers": BEAM_WORKERS,
        "lm_path": str(LM_PATH),
        "greedy_per": greedy_r["per"],
        "best_lm_weight": best["lm_weight"],
        "best_beam_per": best["per"],
        "improvement_over_greedy": greedy_r["per"] - best["per"],
        "total_time_sec": round(time.time() - t_start, 1),
        "results": results,
    }

    json_path = OUT_DIR / "lm_weight_sweep.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    txt_path = OUT_DIR / "lm_weight_sweep.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  LM WEIGHT SWEEP RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Val samples: {len(all_lp)} (DS1+DS2 only)\n")
        f.write(f"Beam width: {BEAM_WIDTH}\n")
        f.write(f"Length alpha: {LENGTH_ALPHA}\n\n")
        f.write(f"{'LM Weight':>12} | {'PER':>10} | {'Time':>8} | {'vs Greedy':>10}\n")
        f.write("-" * 50 + "\n")
        for r in results:
            w = r["lm_weight"]
            w_str = f"{w}" if isinstance(w, str) else f"{w:.2f}"
            diff = greedy_r["per"] - r["per"]
            diff_str = f"{diff:+.6f}" if w != "greedy" else "baseline"
            t_str = f"{r['time_sec']:.1f}s" if r["time_sec"] > 0 else "-"
            marker = " ★" if r is best else ""
            f.write(f"{w_str:>12} | {r['per']:.6f}   | {t_str:>8} | {diff_str:>10}{marker}\n")
        f.write("-" * 50 + "\n")
        f.write(f"\n★ BEST: LM_WEIGHT={best['lm_weight']:.2f} → PER={best['per']:.6f}\n")
        f.write(f"  Greedy PER: {greedy_r['per']:.6f}\n")
        f.write(f"  Improvement: {greedy_r['per'] - best['per']:+.6f}\n")
        f.write(f"  Total sweep time: {time.time() - t_start:.0f}s\n")

    log(f"Results: {json_path}")
    log(f"Report:  {txt_path}")
    log("")
    log("=" * 50)
    log(f"  Greedy PER:     {greedy_r['per']:.6f}")
    log(f"  Best beam PER:  {best['per']:.6f} (LM_WEIGHT={best['lm_weight']:.2f})")
    log(f"  Improvement:    {greedy_r['per'] - best['per']:+.6f}")
    log(f"  Total time:     {time.time() - t_start:.0f}s")
    log("=" * 50)


if __name__ == "__main__":
    main()
