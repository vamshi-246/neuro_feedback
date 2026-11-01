"""Ask which TASK the rebuilt features can actually do well, not which feature is best.

Feature work has run its course.  Rebuilding from raw signal with evoked potentials,
five bands, gamma, connectivity and shape descriptors moved balanced accuracy from
about 0.46 to 0.49 on three-class subjective pain.  That is a real gain and it is
nowhere near 90%, so the limit is no longer the feature list.

What has never been tested is whether the question itself is the hard part.  Three
things make the current task unusually punishing, and each can be removed:

  Moderate sits between Low and High and absorbs most of the confusion.  Ratings
  6.9 and 7.0 are the same sensation on opposite sides of a label boundary.

  The target is subjective.  Two people given identical stimuli report different
  numbers, and nothing in the EEG can resolve a disagreement about wording.

  Stimulus energy is objective and already recorded.  Predicting it removes rater
  noise entirely, at the cost of measuring the stimulus rather than the experience.

Every task below uses the same subjects, the same locked split, the same classifier,
and EEG features only.  Laser power is never a feature: for the intensity task it is
the label, and elsewhere including it would confuse "reads pain" with "reads the
laser".  Differences across rows therefore reflect the difficulty of the question.

Balanced accuracy is comparable across rows only after accounting for class count:
chance is 1/3 for a three-class row and 1/2 for a two-class row, so each row prints
its own chance level and its margin above it.

Usage (from repo root):
    python scripts/preprocessing/task_ceiling_check.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    select_dataset_view,
)


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def intensity_tertiles(laser_power, dataset_id):
    """Bin stimulus energy into low/medium/high WITHIN each dataset.

    Every experiment used its own energy range, so a global threshold would encode
    which experiment a trial came from rather than how strong the stimulus was.
    """

    labels = np.full(laser_power.size, -1, dtype=np.int64)
    for dataset in set(dataset_id.tolist()):
        mask = dataset_id == dataset
        values = laser_power[mask]
        low, high = np.quantile(values, [1 / 3, 2 / 3])
        if not low < high:  # too few distinct energies to form three bins
            continue
        binned = np.digitize(values, [low, high])
        labels[mask] = binned
    return labels


def run_task(name, X, y, train_mask, val_mask, seed):
    keep_train = train_mask & (y >= 0)
    keep_val = val_mask & (y >= 0)
    classes = np.unique(y[keep_train])
    if classes.size < 2 or keep_val.sum() == 0:
        print(f"  {name:<42}{'(not enough classes)':>34}")
        return
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X[keep_train], y[keep_train],
              sample_weight=balanced_sample_weights(y[keep_train]))
    pred = model.predict(X[keep_val])
    bal = balanced_accuracy_score(y[keep_val], pred)
    acc = accuracy_score(y[keep_val], pred)
    chance = 1.0 / classes.size
    print(
        f"  {name:<42}{int(keep_val.sum()):>8}{classes.size:>4}"
        f"{chance:>9.3f}{bal:>10.4f}{acc:>9.4f}{100*(bal-chance):>+9.1f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rich", default="outputs/rich_full/rich_all9_features.npz")
    ap.add_argument("--baseline", default="outputs/all9_full/all9_features.npz")
    ap.add_argument(
        "--split-from-checkpoint",
        default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt",
    )
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    rich = np.load(args.rich, allow_pickle=False)
    X = rich["features"]
    ratings = rich["ratings"].astype(np.float64)
    labels3 = rich["labels"].astype(np.int64)
    laser = rich["laser_power"].astype(np.float64)
    dataset_id = rich["dataset_id"].astype(str)
    subject_id = rich["subject_id"].astype(str)

    base_archive = load_feature_archive(args.baseline)
    _, training_dataset_ids = select_dataset_view(base_archive, None)
    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        args.split_from_checkpoint, base_archive, training_dataset_ids
    )
    train_mask = keys_mask(dataset_id, subject_id, train_keys)
    val_mask = keys_mask(dataset_id, subject_id, val_keys)

    print(f"EEG features only: {X.shape[1]} columns   trials {X.shape[0]}")
    print(f"Train {int(train_mask.sum())}   Validation {int(val_mask.sum())}")
    print("Laser power is never used as a feature.\n")

    print(f"  {'task':<42}{'val n':>8}{'cls':>4}{'chance':>9}{'balanced':>10}"
          f"{'acc':>9}{'vs chance':>9}")
    print("  " + "-" * 91)

    run_task("A. Low/Moderate/High (current task)", X, labels3,
             train_mask, val_mask, args.seed)

    # Drop the middle class: keeps the same question but removes the boundary
    # where a 0.1 rating difference flips the answer.
    low_high = np.where(labels3 == 0, 0, np.where(labels3 == 2, 1, -1))
    run_task("B. Low vs High only (drop Moderate)", X, low_high,
             train_mask, val_mask, args.seed)

    # The clinically meaningful split: did this hurt at all?
    pain = np.where(ratings >= 4.0, 1, 0).astype(np.int64)
    run_task("C. No-pain vs pain (rating >= 4)", X, pain,
             train_mask, val_mask, args.seed)

    run_task("D. Severe vs not (rating >= 7)",
             np.asarray(X), np.where(ratings >= 7.0, 1, 0).astype(np.int64),
             train_mask, val_mask, args.seed)

    # Objective target: what the laser actually did, free of rater disagreement.
    run_task("E. Stimulus intensity, 3 levels", X,
             intensity_tertiles(laser, dataset_id), train_mask, val_mask, args.seed)

    intensity = intensity_tertiles(laser, dataset_id)
    run_task("F. Stimulus intensity, lowest vs highest",
             X, np.where(intensity == 0, 0, np.where(intensity == 2, 1, -1)),
             train_mask, val_mask, args.seed)

    print(
        "\n  'vs chance' is the honest comparison across rows: a two-class task\n"
        "  starts at 50%, so a raw 70% there is worth less than it looks next to\n"
        "  a three-class result. Every row is still a held-out-subject result,\n"
        "  meaning the model has never seen the person it is being scored on."
    )


if __name__ == "__main__":
    main()
