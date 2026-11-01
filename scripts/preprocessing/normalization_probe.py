"""Read-only probe: does removing each person's individual scale make features transfer?

feature_ceiling_check.py established that the saved band-power features separate
TRAINING trials perfectly (99.99%) while transferring almost nothing to held-out
people (~41%).  That is a transferability failure, not a shortage of information,
so the useful question is whether per-subject or per-dataset standardization
recovers signal that survives across people.

Why this could work: relative band power still carries a large person-specific
offset (skull thickness, electrode impedance, individual baseline rhythms) and a
large recording-rig offset (nine datasets, different hardware).  Standardizing
within a subject or a dataset removes that offset and keeps only how a trial
differs from that person's or that rig's own typical trial.

IMPORTANT -- what per-subject standardization assumes at deployment time:
it uses a held-out subject's own FEATURE distribution (never their labels).  In
practice that means a new user must supply a short unlabeled calibration
recording before predictions are made.  That is a real deployment requirement,
not a free improvement, and it must be reported alongside any accuracy gain.
Per-dataset standardization carries the weaker requirement of knowing which rig
produced the recording.

Gradient boosting stands in for the LSTM here only because it trains in seconds
and already matched or beat it; whichever scheme wins is then worth a real LSTM
run.  Nothing is written to disk and the test split is never touched.

Usage (from repo root):
    python scripts/preprocessing/normalization_probe.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    model_feature_view,
    select_dataset_view,
)

MIN_TRIALS_FOR_STATS = 5
STD_FLOOR = 1e-8


def _group_standardize(X, group_keys, *, robust=False):
    """Centre and scale every column inside each group independently.

    Groups with too few trials to estimate a spread are centred only, so a
    noisy single-trial standard deviation cannot manufacture extreme values.
    """

    out = np.array(X, dtype=np.float64, copy=True)
    for key in set(group_keys):
        mask = np.array([k == key for k in group_keys])
        block = out[mask]
        if robust:
            centre = np.median(block, axis=0)
            spread = np.median(np.abs(block - centre), axis=0) * 1.4826
        else:
            centre = block.mean(axis=0)
            spread = block.std(axis=0)
        if block.shape[0] < MIN_TRIALS_FOR_STATS:
            spread = np.ones_like(spread)
        spread = np.where(spread < STD_FLOOR, 1.0, spread)
        out[mask] = (block - centre) / spread
    return out


def build_schemes(X, dataset_id, subject_id):
    subject_keys = list(zip(dataset_id.tolist(), subject_id.tolist()))
    dataset_keys = dataset_id.tolist()
    schemes = {
        "none (current pipeline)": X,
        "per-dataset z-score": _group_standardize(X, dataset_keys),
        "per-subject z-score": _group_standardize(X, subject_keys),
        "per-subject robust (median/MAD)": _group_standardize(X, subject_keys, robust=True),
    }
    schemes["per-subject z, then per-dataset z"] = _group_standardize(
        schemes["per-subject z-score"], dataset_keys
    )
    return schemes


def balanced_sample_weights(labels):
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("training split is missing a class")
    return (labels.size / (len(CLASS_NAMES) * counts))[labels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/all9_full/all9_features.npz")
    ap.add_argument(
        "--split-from-checkpoint",
        default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt",
    )
    ap.add_argument("--feature-mode", choices=("channel-average", "per-channel"),
                    default="per-channel")
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    archive = load_feature_archive(args.data)
    _, training_dataset_ids = select_dataset_view(archive, None)
    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        args.split_from_checkpoint, archive, training_dataset_ids
    )

    dataset_id = archive["dataset_id"].astype(str)
    subject_id = archive["subject_id"].astype(str)
    y = archive["labels"].astype(np.int64)
    train_mask = keys_mask(dataset_id, subject_id, train_keys)
    val_mask = keys_mask(dataset_id, subject_id, val_keys)

    power, feature_order = model_feature_view(archive, args.feature_mode)
    X = np.log1p(power.reshape(power.shape[0], -1))

    trials_per_subject = np.bincount(
        np.unique(
            [f"{d}/{s}" for d, s in zip(dataset_id.tolist(), subject_id.tolist())],
            return_inverse=True,
        )[1]
    )
    print(f"Feature mode: {args.feature_mode}  ({X.shape[1]} columns per trial)")
    print(
        f"Trials per subject: min={trials_per_subject.min()} "
        f"median={int(np.median(trials_per_subject))} max={trials_per_subject.max()}"
    )
    print(f"Train trials: {int(train_mask.sum())}   Validation trials: {int(val_mask.sum())}")
    print(f"Do-nothing balanced-accuracy floor: {1/len(CLASS_NAMES):.4f}\n")

    y_train, y_val = y[train_mask], y[val_mask]
    w_train = balanced_sample_weights(y_train)

    print(f"{'normalization scheme':<34}{'train bal':>11}{'VAL bal':>10}{'val acc':>10}")
    print("-" * 65)
    results = []
    for name, Xs in build_schemes(X, dataset_id, subject_id).items():
        model = HistGradientBoostingClassifier(random_state=args.seed)
        model.fit(Xs[train_mask], y_train, sample_weight=w_train)
        train_bal = balanced_accuracy_score(y_train, model.predict(Xs[train_mask]))
        val_pred = model.predict(Xs[val_mask])
        val_bal = balanced_accuracy_score(y_val, val_pred)
        val_acc = accuracy_score(y_val, val_pred)
        results.append((name, val_bal))
        print(f"{name:<34}{train_bal:>11.4f}{val_bal:>10.4f}{val_acc:>10.4f}")

    best_name, best_val = max(results, key=lambda row: row[1])
    baseline = dict(results)["none (current pipeline)"]
    print(
        f"\nBest scheme: {best_name} at {best_val:.4f} validation balanced accuracy "
        f"({100*(best_val-baseline):+.2f} points vs the current pipeline)."
    )
    print(
        "Reminder: any per-subject scheme requires a short unlabeled calibration\n"
        "recording from each new user at deployment time."
    )


if __name__ == "__main__":
    main()
