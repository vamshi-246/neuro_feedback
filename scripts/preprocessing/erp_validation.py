"""Check whether the evoked potential is real in THIS data, and where its peaks are.

Two separate worries motivate this script.

First, circularity.  rich_feature_extraction.py searches for N2 inside a 150-300 ms
window taken from the laser-evoked-potential literature, so an N2 latency inside
150-300 ms is guaranteed by construction and proves nothing.  Measuring the peak
locations from the grand-average waveform, with no search window imposed, gives an
answer the textbook did not dictate.

Second, and more decisive: an evoked potential can be perfectly real and still be
useless here if its size does not vary with reported pain.  The whole reason for
adding these features is the claim that N2-P2 amplitude tracks rated intensity, so
this script measures N2-P2 separately for Low, Moderate, and High trials.  If the
three are indistinguishable, the ERP features will not help and a full 24 GB
rebuild is not worth starting.

Every subject contributes equally to the grand average regardless of trial count,
so a handful of high-trial subjects cannot dominate the waveform.

Peak windows reported here are measured on whichever subjects are loaded.  To use
them as tuned feature-extraction settings rather than as a descriptive check, they
must be re-measured on training subjects only, otherwise validation and test
subjects would influence the feature definition.

Usage (from repo root):
    python scripts/preprocessing/erp_validation.py
    python scripts/preprocessing/erp_validation.py --datasets ds005289 ds005286
"""

import argparse
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_dataset import DEFAULT_CHANNELS, ONSET_TOLERANCE_SAMPLES, _require_cache_file
from dataset_registry import REGISTRY, subject_ids
from eeglab_io import load_recording, pick_channels
from feature_extraction import BASELINE_WINDOW_S, bin_rating
from rich_feature_extraction import ERP_BAND_HZ
from train_lstm import CLASS_NAMES

# Resample every subject's ERP onto one shared time base so datasets recorded at
# different sampling rates can be averaged together.
GRID_START_S = -0.2
GRID_END_S = 0.8
GRID_STEP_S = 0.002  # 500 Hz, finer than any peak structure being measured


def subject_erp(spec, subject_id, cache_dir, channels, grid):
    """Return (channels, len(grid)) mean ERP per class index, plus trial counts."""

    stem = f"{subject_id}_{spec.derivative_stage}"
    set_path = os.path.join(cache_dir, spec.dataset_id, f"{stem}.set")
    fdt_path = os.path.join(cache_dir, spec.dataset_id, f"{stem}.fdt")
    _require_cache_file(set_path)
    _require_cache_file(fdt_path)

    rec = load_recording(spec.dataset_id, subject_id, set_path, fdt_path)
    ch_idx = pick_channels(rec.channel_labels, channels)
    expected_onset = spec.pre_stim_s * rec.srate

    admitted, labels = [], []
    for t in range(rec.data.shape[2]):
        if not rec.trial_ok[t]:
            continue
        onset = float(rec.onset_samples[t])
        if not np.isfinite(onset) or abs(onset - expected_onset) > ONSET_TOLERANCE_SAMPLES:
            continue
        if not str(rec.event_types[t]).strip():
            continue
        admitted.append(t)
        labels.append(bin_rating(rec.ratings[t]))
    if not admitted:
        return None, None

    onset = int(round(expected_onset))
    data = rec.data[np.ix_(ch_idx, np.arange(rec.data.shape[1]), admitted)].astype(np.float64)
    nyq = rec.srate / 2.0
    sos = butter(4, [ERP_BAND_HZ[0] / nyq, ERP_BAND_HZ[1] / nyq], btype="bandpass", output="sos")
    erp = sosfiltfilt(sos, data, axis=1)

    b0 = onset + int(round(BASELINE_WINDOW_S[0] * rec.srate))
    b1 = onset + int(round(BASELINE_WINDOW_S[1] * rec.srate))
    erp = erp - np.mean(erp[:, b0:b1, :], axis=1, keepdims=True)

    source_times = (np.arange(erp.shape[1]) - onset) / rec.srate
    labels = np.asarray(labels, dtype=np.int64)
    per_class = np.full((len(CLASS_NAMES), len(channels), grid.size), np.nan)
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for class_index in range(len(CLASS_NAMES)):
        mask = labels == class_index
        counts[class_index] = int(mask.sum())
        if not mask.any():
            continue
        mean_wave = erp[:, :, mask].mean(axis=2)
        for c in range(len(channels)):
            per_class[class_index, c] = np.interp(grid, source_times, mean_wave[c])
    return per_class, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=sorted(REGISTRY), default=["ds005289"])
    ap.add_argument("--subjects-per-dataset", type=int, default=0)
    ap.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    ap.add_argument("--cache-dir", default="data_cache")
    args = ap.parse_args()

    channels = [c.strip() for c in args.channels]
    grid = np.arange(GRID_START_S, GRID_END_S + GRID_STEP_S / 2, GRID_STEP_S)

    per_subject, per_subject_counts = [], []
    for dataset_id in args.datasets:
        spec = REGISTRY[dataset_id]
        subs = subject_ids(spec)
        if args.subjects_per_dataset > 0:
            subs = subs[: args.subjects_per_dataset]
        for subject_id in subs:
            try:
                waves, counts = subject_erp(spec, subject_id, args.cache_dir, channels, grid)
            except Exception as exc:  # noqa: BLE001 -- a bad subject must not stop the survey
                print(f"[skip] {dataset_id}/{subject_id}: {type(exc).__name__}: {exc}")
                continue
            if waves is None:
                continue
            per_subject.append(waves)
            per_subject_counts.append(counts)
    if not per_subject:
        raise SystemExit("no subjects produced an ERP")

    stack = np.stack(per_subject)                 # (subjects, classes, channels, time)
    counts = np.stack(per_subject_counts)
    print(f"\nSubjects averaged: {stack.shape[0]}   "
          f"trials per class across subjects: {counts.sum(axis=0).tolist()}")
    print("Every subject weighted equally, so high-trial subjects cannot dominate.\n")

    # Collapse classes into one overall waveform, weighting subjects equally.
    with np.errstate(invalid="ignore"):
        overall = np.nanmean(stack, axis=1)       # (subjects, channels, time)
    grand = np.nanmean(overall, axis=0)           # (channels, time)

    print("=" * 74)
    print("MEASURED peaks from the grand-average waveform (no search window imposed)")
    print("=" * 74)
    print(f"{'channel':<10}{'N2 amp':>10}{'N2 lat':>10}{'P2 amp':>10}{'P2 lat':>10}{'N2-P2':>10}")
    post = grid >= 0.05
    measured = {}
    for c, channel in enumerate(channels):
        wave = grand[c]
        n2_i = np.argmin(np.where(post, wave, np.inf))
        after = grid > grid[n2_i]
        p2_i = np.argmax(np.where(after, wave, -np.inf))
        measured[channel] = (grid[n2_i], grid[p2_i], wave[p2_i] - wave[n2_i])
        print(f"{channel:<10}{wave[n2_i]:>9.2f}{1000*grid[n2_i]:>9.0f}ms"
              f"{wave[p2_i]:>9.2f}{1000*grid[p2_i]:>9.0f}ms"
              f"{wave[p2_i]-wave[n2_i]:>10.2f}")

    peak_channel = max(measured, key=lambda ch: measured[ch][2])
    print(f"\n  Largest N2-P2 at: {peak_channel}"
          f"   (a genuine laser-evoked potential is vertex-maximal, i.e. Cz)")
    n2s = [measured[c][0] for c in channels]
    p2s = [measured[c][1] for c in channels]
    print(f"  Measured N2 latencies: {[f'{1000*v:.0f}ms' for v in n2s]}")
    print(f"  Measured P2 latencies: {[f'{1000*v:.0f}ms' for v in p2s]}")
    print(f"  Currently CONFIGURED search windows: N2 150-300ms, P2 after N2 to 550ms")
    print("  If the measured peaks sit outside those windows, the configured windows")
    print("  are wrong for this data and should be widened or re-centred.")

    # ---- The decisive test: does N2-P2 size actually track reported pain? ----
    print("\n" + "=" * 74)
    print("DOES THE EVOKED POTENTIAL TRACK REPORTED PAIN?")
    print("=" * 74)
    print(f"{'channel':<10}" + "".join(f"{name:>14}" for name in CLASS_NAMES) + f"{'High-Low':>12}")
    for c, channel in enumerate(channels):
        row, amps = f"{channel:<10}", []
        for class_index in range(len(CLASS_NAMES)):
            waves = stack[:, class_index, c, :]
            usable = ~np.isnan(waves).any(axis=1)
            if usable.sum() == 0:
                amps.append(np.nan)
                row += f"{'n/a':>14}"
                continue
            wave = waves[usable].mean(axis=0)
            n2_i = np.argmin(np.where(post, wave, np.inf))
            after = grid > grid[n2_i]
            p2_i = np.argmax(np.where(after, wave, -np.inf))
            amplitude = wave[p2_i] - wave[n2_i]
            amps.append(amplitude)
            row += f"{amplitude:>12.2f}uV"
        row += f"{amps[2]-amps[0]:>11.2f}uV" if np.isfinite(amps[2]) and np.isfinite(amps[0]) else f"{'n/a':>12}"
        print(row)

    print(
        "\n  Read this table as the go/no-go for the full rebuild:\n"
        "    High clearly larger than Low  -> the ERP carries pain information;\n"
        "                                     the 24 GB rebuild is justified.\n"
        "    All three about equal         -> N2-P2 does not track rating here, so\n"
        "                                     ERP features will not lift accuracy.\n"
        "  Note this is a group average.  A separation visible here still has to\n"
        "  survive single-trial, cross-subject prediction, which is much harder."
    )


if __name__ == "__main__":
    main()
