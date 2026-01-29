# -*- coding: utf-8 -*-
"""
BCI2000-style AR model, originally in https://doi.org/10.3758/BF03200585
Stieger here: https://doi.org/10.1038/s41597-021-00883-1

- ring buffer for online
- Every 40 ms:
    - Laplacian around C3 and C4
    - 16th order Yule Walker AR estimation
    - PSD via integration in the band we want
    - control is  alpha(C4) - alpha(C3)
    - baseline (first 10 s): collects mean(H), prints BASELINE
    - after baseline: rolling z-score over last 30 s, prints LEFT/RIGHT command

Electrode order (AspireCreate System)
fz, fc3, fc1, fcz, fc2, fc4, C3, C1, Cz, C2, C4, Cp3, Cp1, CpZ, Cp2, Cp4
"""

import time
from collections import deque

import numpy as np
import pylsl

import scipy.signal as sig
from scipy.linalg import toeplitz


ELECTRODES = [
    "Fz", "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C3", "C1", "CZ", "C2", "C4",
    "CP3", "CP1", "CPZ", "CP2", "CP4"
]

UPDATE_SEC = 0.040          # 40 ms
WIN_SEC = 1             # 1 s
AR_ORDER = 16               # Arbitrary tbh, I chose 16th order but i think the original wolpaw paper was 12th? Either way it works. 
ALPHA_LO, ALPHA_HI = 10.5, 13.5  # 3 Hz bin centered at 12 Hz based of Stieger paper
BASELINE_SEC = 10.0 # Normalising the outputs
ZSCORE_SEC = 30.0 # The rolling window for recalc z score
Z_CLIP = 3.0 # anything above 3 we just clip it, we're not doing magnitude = speed but we can do if u want


# Left: around C3: FC3, Cp3, C1, Cz
# Right: around C4: FC4, Cp4, C2, Cz
SURROUND_C3 = ["fc3", "Cp3", "C1", "Cz"]
SURROUND_C4 = ["fc4", "Cp4", "C2", "Cz"]


def yule_walker_ar(x: np.ndarray, order: int):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size
    if n <= order:
        raise ValueError(f"Window too short for AR({order}): n={n}")

    r = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(order + 1)])
    R = toeplitz(r[:-1])
    a = np.linalg.solve(R, r[1:])
    sigma2 = r[0] - np.dot(a, r[1:])
    A = np.concatenate(([1.0], -a))
    return A, float(max(sigma2, 1e-12))


def ar_bandpower(x: np.ndarray, order: int, f_lo: float, f_hi: float, fs: float):
    """
    AR PSD via freq response of sqrt(sigma2)/A(z); integrate over [f_lo, f_hi]. In our case, its centered around 12Hz 3 Hz bin width so 10.5-13.5Hz.
    """
    A, sigma2 = yule_walker_ar(x, order)
    freqs, h = sig.freqz(np.sqrt(sigma2), A, worN=1024, fs=fs)
    psd = np.abs(h) ** 2
    m = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(m):
        return 0.0
    return float(np.trapz(psd[m], freqs[m]))


def resolve_inlet():
    streams = pylsl.resolve_streams(wait_time=2.0)
    if not streams:
        raise SystemExit("No LSL streams found.")

    # Prefer EEG streams if available
    eeg = [s for s in streams if (s.type() or "").lower() == "eeg"]
    chosen = eeg[0] if eeg else streams[0]

    inlet = pylsl.StreamInlet(chosen, max_buflen=60, max_chunklen=0)
    info = inlet.info()
    fs = float(info.nominal_srate())
    if fs <= 0:
        raise SystemExit("Stream nominal_srate() is 0/unknown. This script requires a fixed sampling rate.")

    # Try to map channel labels from stream metadata (if present); else assume incoming order matches ELECTRODES
    labels = []
    try:
        ch = info.desc().child("channels").child("channel")
        while ch and ch.name():
            lab = ch.child_value("label")
            labels.append(lab)
            ch = ch.next_sibling()
    except Exception:
        labels = []

    return inlet, fs, labels


def build_index_map(stream_labels):
    """
    Returns a dict electrode_name -> index in incoming sample vector.
    If stream_labels are present, map by label (case-insensitive).
    Else assume incoming order is exactly ELECTRODES.
    """
    if stream_labels:
        norm = {str(l).strip().lower(): i for i, l in enumerate(stream_labels)}
        idx = {}
        missing = []
        for e in ELECTRODES:
            k = e.strip().lower()
            if k in norm:
                idx[e] = norm[k]
            else:
                missing.append(e)
        if missing:
            raise SystemExit(
                "Stream channel labels did not include required electrodes:\n"
                f"{missing}\n"
                "Fix your stream labels or remove label-based mapping."
            )
        return idx

    # Fallback: assume strict order
    return {e: i for i, e in enumerate(ELECTRODES)}


def spatial_filter(window_2d: np.ndarray, idx: dict):
    """
    window_2d: [n_samp, n_ch]
    Returns:
      c3_filt, c4_filt: [n_samp]
    """
    c3 = window_2d[:, idx["C3"]]
    c4 = window_2d[:, idx["C4"]]

    ref3 = np.stack([window_2d[:, idx[ch]] for ch in SURROUND_C3], axis=1).mean(axis=1)
    ref4 = np.stack([window_2d[:, idx[ch]] for ch in SURROUND_C4], axis=1).mean(axis=1)

    return (c3 - ref3), (c4 - ref4)


def main():
    inlet, fs, stream_labels = resolve_inlet()
    idx = build_index_map(stream_labels)

    n_ch = inlet.info().channel_count()
    if stream_labels:
        print(f"Connected: {inlet.info().name()} ({inlet.info().type()}) @ {fs:.2f} Hz, channels={n_ch} (label-mapped)")
    else:
        print(f"Connected: {inlet.info().name()} ({inlet.info().type()}) @ {fs:.2f} Hz, channels={n_ch} (assuming fixed order)")

    win_n = int(round(WIN_SEC * fs))
    step_n = int(round(UPDATE_SEC * fs))
    if win_n < 8:
        raise SystemExit(f"Window too small: {win_n} samples at fs={fs}.")
    if step_n < 1:
        step_n = 1

    ar_order = min(AR_ORDER, win_n - 1)

    baseline_updates = int(round(BASELINE_SEC / UPDATE_SEC))
    z_hist_len = max(5, int(round(ZSCORE_SEC / UPDATE_SEC)))

    buf = deque(maxlen=win_n)
    diff_hist = deque(maxlen=z_hist_len)
    baseline_buf = []

    last_print = 0.0

    sample_count = 0
    while True:
        s, _ = inlet.pull_sample(timeout=1.0)
        if not s:
            continue

        # Keep only channels we need; still store full vector for indexing
        buf.append(np.asarray(s, dtype=float))
        sample_count += 1

        # Update every step_n samples once we have a full window
        if len(buf) < win_n or (sample_count % step_n) != 0:
            continue

        w = np.stack(buf, axis=0)  # [win_n, n_ch]

        # Feature extraction
        c3f, c4f = spatial_filter(w, idx)
        p3 = ar_bandpower(c3f, ar_order, ALPHA_LO, ALPHA_HI, fs)
        p4 = ar_bandpower(c4f, ar_order, ALPHA_LO, ALPHA_HI, fs)

        diff = float(p4 - p3)  # horizontal control

        # Baseline
        if len(baseline_buf) < baseline_updates:
            baseline_buf.append(diff)
            if time.time() - last_print > 0.5:
                mu = float(np.mean(baseline_buf))
                print(f"BASELINE  n={len(baseline_buf)}/{baseline_updates}  diff={diff:+.3e}  mean={mu:+.3e}")
                last_print = time.time()
            continue

        baseline_mean = float(np.mean(baseline_buf))
        diff_b = diff - baseline_mean

        # Rolling z-score
        diff_hist.append(diff_b)
        arr = np.asarray(diff_hist, dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        eps = 1e-12 + 1e-3 * abs(mu)
        z = 0.0 if sd < eps else (diff_b - mu) / sd
        if Z_CLIP > 0:
            z = float(np.clip(z, -Z_CLIP, Z_CLIP))

        cmd = "RIGHT" if z > 0 else "LEFT"

        # Print every update (40 ms)
        print(
            f"{cmd:5s}  "
            f"z={z:+.3f}  "
            f"diff={diff:+.3e}  "
            f"diff_b={diff_b:+.3e}  "
            f"p3={p3:.3e}  p4={p4:.3e}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
