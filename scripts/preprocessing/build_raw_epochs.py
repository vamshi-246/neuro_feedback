"""Save raw trial waveforms, so a CNN can learn features instead of receiving ours.

Every archive so far stores summaries: band powers, peak amplitudes, coupling values.
Those are numbers we chose to compute, which means anything we did not think to
measure was discarded before any model saw it. A convolutional network needs the
waveform itself, where neighbouring samples are genuinely related and filters can be
learned rather than hand-designed.

Trial admission mirrors build_rich_pilot.py and build_dataset.py exactly, so the
epochs here line up row for row with the rich features and can be fed to one model
side by side.

Two processing choices
----------------------
Downsampling to 250 Hz. The recordings are 1000 Hz, which makes each epoch four
times longer than it needs to be for a 1-45 Hz analysis and multiplies the CNN's
input length for no added information. decimate applies an anti-aliasing filter
first, so nothing inside the bands of interest is lost.

Band-pass 1-45 Hz and baseline correction. This matches the ERP branch and removes
slow drift and mains-range noise. The CNN is being asked to find pain-related shape,
not to rediscover that drift exists.

Storage is float32: 28,452 trials x 4 channels x 375 samples is roughly 170 MB.

Usage (from repo root):
    python scripts/preprocessing/build_raw_epochs.py
"""

import argparse
from collections import Counter
import os
import sys
import time

import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_dataset import (
    DEFAULT_CHANNELS,
    ONSET_TOLERANCE_SAMPLES,
    _atomic_save_npz,
    _normalize_npz_path,
    _require_cache_file,
)
from dataset_registry import REGISTRY, subject_ids
from eeglab_io import load_recording, pick_channels
from feature_extraction import BASELINE_WINDOW_S, bin_rating

TARGET_HZ = 250.0
EPOCH_S = (-0.5, 1.0)        # same span the features are computed over
FILTER_HZ = (1.0, 45.0)


def process_subject_epochs(spec, subject_id, cache_dir, channels):
    stem = f"{subject_id}_{spec.derivative_stage}"
    set_path = os.path.join(cache_dir, spec.dataset_id, f"{stem}.set")
    fdt_path = os.path.join(cache_dir, spec.dataset_id, f"{stem}.fdt")
    _require_cache_file(set_path)
    _require_cache_file(fdt_path)

    rec = load_recording(spec.dataset_id, subject_id, set_path, fdt_path)
    ch_idx = pick_channels(rec.channel_labels, channels)
    expected_onset = spec.pre_stim_s * rec.srate

    admitted, ratings, epochs, laser = [], [], [], []
    rejections = Counter()
    for t in range(rec.data.shape[2]):
        if not rec.trial_ok[t]:
            rejections[str(rec.trial_status[t])] += 1
            continue
        onset = float(rec.onset_samples[t])
        if not np.isfinite(onset) or abs(onset - expected_onset) > ONSET_TOLERANCE_SAMPLES:
            rejections["event_onset_mismatch"] += 1
            continue
        if not str(rec.event_types[t]).strip():
            rejections["missing_event_type"] += 1
            continue
        admitted.append(t)
        ratings.append(rec.ratings[t])
        epochs.append(t + 1)
        laser.append(rec.laser_power[t])

    if not admitted:
        return None

    onset = int(round(expected_onset))
    data = rec.data[np.ix_(ch_idx, np.arange(rec.data.shape[1]), admitted)].astype(np.float64)

    nyq = rec.srate / 2.0
    sos = butter(4, [FILTER_HZ[0] / nyq, FILTER_HZ[1] / nyq], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, data, axis=1)
    b0 = onset + int(round(BASELINE_WINDOW_S[0] * rec.srate))
    b1 = onset + int(round(BASELINE_WINDOW_S[1] * rec.srate))
    filtered = filtered - np.mean(filtered[:, b0:b1, :], axis=1, keepdims=True)

    s0 = onset + int(round(EPOCH_S[0] * rec.srate))
    s1 = onset + int(round(EPOCH_S[1] * rec.srate))
    if not 0 <= s0 < s1 <= filtered.shape[1]:
        raise ValueError(f"{spec.dataset_id}/{subject_id}: epoch window outside recording")
    segment = filtered[:, s0:s1, :]

    factor = int(round(rec.srate / TARGET_HZ))
    if factor > 1:
        segment = decimate(segment, factor, axis=1, ftype="fir", zero_phase=True)

    # (trials, channels, samples) is the layout PyTorch convolutions expect.
    waveforms = np.transpose(segment, (2, 0, 1)).astype(np.float32)
    return (
        waveforms,
        np.asarray(ratings, dtype=np.float64),
        np.asarray(epochs, dtype=np.int64),
        np.asarray(laser, dtype=np.float64),
        float(rec.srate),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=sorted(REGISTRY), default=sorted(REGISTRY))
    ap.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--out", default="outputs/rich_full/raw_epochs_all9.npz")
    args = ap.parse_args()

    args.out = _normalize_npz_path(args.out)
    channels = [c.strip() for c in args.channels]
    expected_samples = int(round((EPOCH_S[1] - EPOCH_S[0]) * TARGET_HZ))
    print(f"Channels: {channels}")
    print(f"Each epoch: {EPOCH_S[0]}..{EPOCH_S[1]}s at {TARGET_HZ:g} Hz "
          f"= {expected_samples} samples\n")

    all_waves, all_ratings, all_epochs, all_laser = [], [], [], []
    all_subject, all_dataset = [], []
    failures = []
    started = time.time()

    for dataset_id in args.datasets:
        spec = REGISTRY[dataset_id]
        for subject_id in subject_ids(spec):
            try:
                result = process_subject_epochs(spec, subject_id, args.cache_dir, channels)
            except Exception as exc:  # noqa: BLE001 -- report all failures, abort at the end
                failures.append((dataset_id, subject_id, type(exc).__name__, str(exc)))
                print(f"[FAIL] {dataset_id}/{subject_id}: {type(exc).__name__}: {exc}")
                continue
            if result is None:
                failures.append((dataset_id, subject_id, "NoAcceptedTrials", "all rejected"))
                continue
            waves, ratings, epochs, laser, srate = result
            if waves.shape[2] != expected_samples:
                raise AssertionError(
                    f"{dataset_id}/{subject_id}: {waves.shape[2]} samples, "
                    f"expected {expected_samples}"
                )
            all_waves.append(waves)
            all_ratings.append(ratings)
            all_epochs.append(epochs)
            all_laser.append(laser)
            all_subject.extend([subject_id] * ratings.size)
            all_dataset.extend([dataset_id] * ratings.size)
        print(f"  {dataset_id} done ({time.time()-started:.0f}s)")

    if not all_waves:
        raise RuntimeError("no subjects produced epochs")

    waveforms = np.concatenate(all_waves, axis=0)
    ratings = np.concatenate(all_ratings, axis=0)
    labels = np.asarray([bin_rating(r) for r in ratings], dtype=np.int64)
    _atomic_save_npz(
        args.out,
        waveforms=waveforms,
        ratings=ratings,
        labels=labels,
        epoch_index=np.concatenate(all_epochs, axis=0),
        laser_power=np.concatenate(all_laser, axis=0),
        subject_id=np.asarray(all_subject, dtype=str),
        dataset_id=np.asarray(all_dataset, dtype=str),
        channel_order=np.asarray(channels, dtype=str),
        sample_rate_hz=np.asarray(TARGET_HZ),
        epoch_window_s=np.asarray(EPOCH_S, dtype=np.float64),
    )
    print(f"\nSaved {waveforms.shape[0]} epochs of shape "
          f"{waveforms.shape[1]}x{waveforms.shape[2]} to {args.out}")
    print(f"Subjects: {len(set(zip(all_dataset, all_subject)))}   Failures: {len(failures)}")
    print(f"Size on disk: {waveforms.nbytes/1e6:.0f} MB   Time: {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
