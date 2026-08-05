"""
End-to-end: download (or reuse cached) derivative EEG files for a list of
subjects across the 5 chosen datasets, extract the RTL-shaped feature
sequence per trial, and save one unified file.

This is the "standard input contract" step: no matter which of the 9 original
datasets a trial came from, what comes out of this script looks identical --
same 4 channels' worth of alpha/beta/theta power, same 3 time windows, same
0-255 scale, same 3-class label.

Usage (from repo root):
    python scripts/preprocessing/build_dataset.py --subjects-per-dataset 1
"""

import argparse
import os
import urllib.request

import numpy as np

from dataset_registry import REGISTRY, s3_set_url, s3_fdt_url, subject_ids
from eeglab_io import load_recording, pick_channels
from feature_extraction import extract_trial_features, bin_rating

COMMON_CHANNELS = ["Fz", "Cz", "C3", "C4"]


def _download(url: str, dest: str):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        # A failed/interrupted transfer (dropped connection, DNS hiccup, etc.)
        # can leave a truncated file sitting at `dest`. Without this cleanup,
        # the exists-and-nonzero check above would wrongly treat that broken
        # leftover as "already downloaded" on every future run and never
        # retry it. Delete it so the next run re-attempts from scratch.
        if os.path.exists(dest):
            os.remove(dest)
        raise


def process_subject(spec, subject_id: str, cache_dir: str):
    set_path = os.path.join(cache_dir, spec.dataset_id, f"{subject_id}_mark_ica.set")
    fdt_path = os.path.join(cache_dir, spec.dataset_id, f"{subject_id}_mark_ica.fdt")
    _download(s3_set_url(spec, subject_id), set_path)
    _download(s3_fdt_url(spec, subject_id), fdt_path)

    rec = load_recording(spec.dataset_id, subject_id, set_path, fdt_path)
    ch_idx = pick_channels(rec.channel_labels, COMMON_CHANNELS)
    onset_sample = int(round(spec.pre_stim_s * rec.srate))

    n_trials = rec.data.shape[2]
    raw_rows = []
    ratings = []
    for t in range(n_trials):
        if not rec.trial_ok[t]:
            continue
        trial_channels = rec.data[ch_idx, :, t]  # (4, pnts)
        feats = extract_trial_features(trial_channels, rec.srate, onset_sample)  # (3,3)
        raw_rows.append(feats)
        ratings.append(rec.ratings[t])

    return np.array(raw_rows), np.array(ratings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects-per-dataset", type=int, default=1)
    ap.add_argument(
        "--cache-dir",
        default=os.path.join(
            os.environ.get("TEMP", "."), "neuro_feedback_pipeline_cache"
        ),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "pipeline_dev", "pooled_features.npz"
        ),
    )
    args = ap.parse_args()

    all_raw, all_ratings, all_labels, all_subject, all_dataset = [], [], [], [], []

    for ds_id, spec in REGISTRY.items():
        subs = subject_ids(spec)[: args.subjects_per_dataset]
        for subject_id in subs:
            try:
                raw, ratings = process_subject(spec, subject_id, args.cache_dir)
            except Exception as exc:  # noqa: BLE001 -- surface and keep going
                print(f"[SKIP] {ds_id}/{subject_id}: {exc}")
                continue
            print(f"[OK] {ds_id}/{subject_id}: {len(ratings)} rated trials, "
                  f"rating range {ratings.min():.1f}-{ratings.max():.1f}")
            all_raw.append(raw)
            all_ratings.append(ratings)
            all_labels.append(np.array([bin_rating(r) for r in ratings]))
            all_subject += [subject_id] * len(ratings)
            all_dataset += [ds_id] * len(ratings)

    raw_power = np.concatenate(all_raw, axis=0)       # (N, 3, 3)
    ratings = np.concatenate(all_ratings, axis=0)      # (N,)
    labels = np.concatenate(all_labels, axis=0)        # (N,)

    # Deliberately NOT quantized to 0-255 here. Turning raw band power into
    # the RTL's 0-255 scale requires picking a low/high bound, and those
    # bounds must be fit on TRAIN-subject trials only (same discipline as
    # DS005285_LSTM_ARCHITECTURE.md) -- this script doesn't know the
    # train/val/test split, only train_lstm.py does, so the quantization step
    # now happens there instead, after the split.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        raw_power=raw_power,            # (N, 3 time steps, 3 bands: alpha,beta,theta), un-quantized
        ratings=ratings,                # (N,) real 0-10 pain score
        labels=labels,                  # (N,) 0=Low 1=Moderate 2=High
        subject_id=np.array(all_subject),
        dataset_id=np.array(all_dataset),
    )
    print(f"\nSaved {raw_power.shape[0]} trials to {args.out}")
    print("Label distribution:", {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})


if __name__ == "__main__":
    main()
