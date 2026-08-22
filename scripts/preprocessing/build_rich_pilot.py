"""Build rich features for a subset of subjects, to test them before a full rebuild.

The full archive covers 858 recordings and 24 GB.  Rebuilding all of it to test an
unproven feature set would be wasteful, so this script rebuilds a pilot subset and
writes a deliberately simple archive.  The strict schema-1 contract enforced by
train_lstm.py is left completely untouched: nothing here can affect the existing
pipeline, and the decision to widen that contract is only worth making once the
new features have demonstrably earned it.

Trial admission mirrors build_dataset.process_subject exactly -- same rating
requirement, same onset tolerance, same event checks -- so a trial admitted here
is the same trial admitted there.  That is what allows compare_feature_sets.py to
score old and new features on identical rows and attribute any difference to the
features alone.

Reads only local cache files and never touches the network.

Usage (from repo root):
    python scripts/preprocessing/build_rich_pilot.py
    python scripts/preprocessing/build_rich_pilot.py --datasets ds005289 --all-subjects
"""

import argparse
from collections import Counter
import os
import sys
import time

import numpy as np

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
from feature_extraction import bin_rating
from mvar_features import (
    MVAR_DEFAULT_ORDER,
    extract_mvar_features,
    mvar_feature_names,
)
from rich_feature_extraction import (
    RichFeatureError,
    extract_rich_features,
    rich_feature_names,
)

# Four smaller datasets spanning three different EEG systems (ANT, Biosemi, BP).
# Together they are about 2.9 GB instead of 24 GB, while still providing enough
# subjects for an honest subject-level split.
PILOT_DATASETS = ["ds005284", "ds005286", "ds005289", "ds005291"]


def process_subject_rich(
    spec, subject_id, cache_dir, channels,
    include_coupling=True, mvar_order=0,
):
    """Return (features, ratings, epoch_index, qc) for one cached subject."""

    cache_stem = f"{subject_id}_{spec.derivative_stage}"
    set_path = os.path.join(cache_dir, spec.dataset_id, f"{cache_stem}.set")
    fdt_path = os.path.join(cache_dir, spec.dataset_id, f"{cache_stem}.fdt")
    _require_cache_file(set_path)
    _require_cache_file(fdt_path)

    rec = load_recording(spec.dataset_id, subject_id, set_path, fdt_path)
    ch_idx = pick_channels(rec.channel_labels, channels)
    expected_onset = spec.pre_stim_s * rec.srate

    admitted, ratings, epochs = [], [], []
    laser_power, event_type = [], []
    rejections = Counter()
    for t in range(rec.data.shape[2]):
        if not rec.trial_ok[t]:
            rejections[str(rec.trial_status[t])] += 1
            continue
        stored_onset = float(rec.onset_samples[t])
        if not np.isfinite(stored_onset):
            rejections["missing_event_onset"] += 1
            continue
        if abs(stored_onset - expected_onset) > ONSET_TOLERANCE_SAMPLES:
            rejections["event_onset_mismatch"] += 1
            continue
        if not str(rec.event_types[t]).strip():
            rejections["missing_event_type"] += 1
            continue
        admitted.append(t)
        ratings.append(rec.ratings[t])
        epochs.append(t + 1)  # preserve EEGLAB's one-based epoch identity
        # Stimulus intensity alone already outpredicts every EEG feature tested
        # so far, so it has to be carried through as the bar new features must clear.
        laser_power.append(rec.laser_power[t])
        event_type.append(str(rec.event_types[t]))

    qc = {
        "input_trials": int(rec.data.shape[2]),
        "accepted_trials": len(admitted),
        "rejections": dict(sorted(rejections.items())),
        "sampling_rate_hz": float(rec.srate),
    }
    if not admitted:
        return None, None, None, None, None, qc

    # Every admitted trial passed the same onset tolerance, so one shared onset
    # is exact; assert it rather than assume it.
    onset = int(round(expected_onset))
    if abs(onset - expected_onset) > 0.51:
        raise RichFeatureError(
            "onset_not_integral",
            f"expected onset {expected_onset} is not within half a sample of {onset}",
        )

    subset = rec.data[np.ix_(ch_idx, np.arange(rec.data.shape[1]), admitted)]
    features = extract_rich_features(
        subset, rec.srate, onset, include_coupling=include_coupling
    )
    if mvar_order > 0:
        # MVAR columns are appended after every rich column, matching the order
        # the two name lists are concatenated in main().
        features = np.concatenate(
            [features, extract_mvar_features(subset, rec.srate, onset, order=mvar_order)],
            axis=1,
        )
    return (
        features,
        np.asarray(ratings, dtype=np.float64),
        np.asarray(epochs, dtype=np.int64),
        np.asarray(laser_power, dtype=np.float64),
        np.asarray(event_type, dtype=str),
        qc,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=sorted(REGISTRY), default=PILOT_DATASETS)
    ap.add_argument("--subjects-per-dataset", type=int, default=0,
                    help="0 means every registered subject in the selected datasets.")
    ap.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--out", default="outputs/rich_pilot/rich_pilot_features.npz")
    ap.add_argument(
        "--skip-coupling",
        action="store_true",
        help=(
            "Drop channel-pair coupling. Its column count grows with the square of "
            "the channel count, and it scored weakest of all groups on its own."
        ),
    )
    ap.add_argument(
        "--mvar-order",
        type=int,
        default=0,
        help=(
            f"Add multivariate autoregressive features at this model order "
            f"(0 disables; {MVAR_DEFAULT_ORDER} is the tested default). Captures how "
            f"the signal moves and which channel drives which, unlike the symmetric "
            f"coupling features."
        ),
    )
    args = ap.parse_args()

    args.out = _normalize_npz_path(args.out)
    channels = [c.strip() for c in args.channels]
    include_coupling = not args.skip_coupling
    names = rich_feature_names(channels, include_coupling=include_coupling)
    if args.mvar_order > 0:
        names = names + mvar_feature_names(channels, order=args.mvar_order)
    print(f"Datasets: {args.datasets}")
    print(f"Channels: {channels}")
    print(f"Rich features per trial: {len(names)}")
    print(f"Cache: {os.path.abspath(args.cache_dir)}\n")

    all_features, all_ratings, all_epochs = [], [], []
    all_laser, all_event = [], []
    all_subject, all_dataset = [], []
    failures = []
    started = time.time()

    for dataset_id in args.datasets:
        spec = REGISTRY[dataset_id]
        subs = subject_ids(spec)
        if args.subjects_per_dataset > 0:
            subs = subs[: args.subjects_per_dataset]
        for subject_id in subs:
            try:
                features, ratings, epochs, laser, events, qc = process_subject_rich(
                    spec, subject_id, args.cache_dir, channels,
                    include_coupling=include_coupling,
                    mvar_order=args.mvar_order,
                )
            except Exception as exc:  # noqa: BLE001 -- collect every failure, abort later
                failures.append((dataset_id, subject_id, type(exc).__name__, str(exc)))
                print(f"[FAIL] {dataset_id}/{subject_id}: {type(exc).__name__}: {exc}")
                continue
            if features is None:
                failures.append((dataset_id, subject_id, "NoAcceptedTrials", "all trials rejected"))
                print(f"[FAIL] {dataset_id}/{subject_id}: all trials rejected by QC")
                continue
            all_features.append(features)
            all_ratings.append(ratings)
            all_epochs.append(epochs)
            all_laser.append(laser)
            all_event.append(events)
            all_subject.extend([subject_id] * ratings.size)
            all_dataset.extend([dataset_id] * ratings.size)
            print(
                f"[OK] {dataset_id}/{subject_id}: {ratings.size}/{qc['input_trials']} trials "
                f"@ {qc['sampling_rate_hz']:.0f} Hz  ({time.time()-started:.0f}s elapsed)"
            )

    if not all_features:
        raise RuntimeError("no subjects produced features")

    features = np.concatenate(all_features, axis=0)
    ratings = np.concatenate(all_ratings, axis=0)
    epochs = np.concatenate(all_epochs, axis=0)
    labels = np.asarray([bin_rating(r) for r in ratings], dtype=np.int64)
    if features.shape[1] != len(names):
        raise AssertionError(
            f"feature width {features.shape[1]} does not match {len(names)} names"
        )

    _atomic_save_npz(
        args.out,
        features=features,
        feature_names=np.asarray(names, dtype=str),
        ratings=ratings,
        labels=labels,
        epoch_index=epochs,
        laser_power=np.concatenate(all_laser, axis=0),
        event_type=np.concatenate(all_event, axis=0),
        subject_id=np.asarray(all_subject, dtype=str),
        dataset_id=np.asarray(all_dataset, dtype=str),
        channel_order=np.asarray(channels, dtype=str),
    )
    subjects = len(set(zip(all_dataset, all_subject)))
    print(f"\nSaved {features.shape[0]} trials x {features.shape[1]} features to {args.out}")
    print(f"Subjects: {subjects}   Failures: {len(failures)}")
    print("Label distribution:", {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})
    print(f"Total time: {time.time()-started:.0f}s")
    if failures:
        print("\nFailures:")
        for row in failures[:20]:
            print("  ", row)


if __name__ == "__main__":
    main()
