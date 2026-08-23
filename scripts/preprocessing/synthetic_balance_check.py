"""Does adding synthetic minority-class trials beat the class weighting we already use?

The pipeline already corrects class imbalance by weighting: Low and High trials count
for more in the loss than the more numerous Moderate ones. Measured earlier, that
correction was worth a lot (0.3976 -> 0.4787 pooled going from no correction to full).

Generating synthetic minority trials targets the same imbalance by a different route,
so the question is whether it adds anything on top of weighting or merely repeats it.
This script measures that instead of arguing it.

Methods compared, all trained on the identical locked split
-----------------------------------------------------------
  none                 no correction at all, the floor
  class weighting      what the pipeline does today
  random oversample    duplicate minority trials until classes match
  SMOTE                new trials interpolated between a minority trial and one of its
                       nearest same-class neighbours, the standard approach
  SMOTE within-subject interpolation restricted to the SAME person's trials. Standard
                       SMOTE will happily blend two different people, inventing a
                       physiology that belongs to nobody, which is precisely the
                       cross-subject transfer the models keep failing at.
  SMOTE + weighting    both corrections together

Synthetic trials are created ONLY inside the training split. Validation stays entirely
real; nothing invented is ever scored.

Reported on dataset-macro balanced accuracy
-------------------------------------------
Pooled scoring rewards a model for recognising which dataset a trial came from, since
the nine datasets have very different class mixes and are partly identifiable from the
signal: dataset identity alone scores 0.4064 pooled with no EEG whatsoever. Dataset
macro scores each dataset separately and then averages, so that shortcut earns nothing.
Pooled is printed alongside for continuity with earlier numbers, but the macro column
is the honest one.

Usage (from repo root):
    python scripts/preprocessing/synthetic_balance_check.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    select_dataset_view,
)

SMOTE_NEIGHBOURS = 5


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def random_oversample(X, y, rng):
    target = np.bincount(y).max()
    parts_X, parts_y = [X], [y]
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        shortfall = target - rows.size
        if shortfall <= 0:
            continue
        picked = rng.choice(rows, size=shortfall, replace=True)
        parts_X.append(X[picked])
        parts_y.append(np.full(shortfall, label, dtype=np.int64))
    return np.vstack(parts_X), np.concatenate(parts_y)


def smote(X, y, rng, groups=None):
    """Interpolate new minority trials toward same-class neighbours.

    When ``groups`` is given, a trial may only be blended with another trial from the
    same group, so a synthetic trial always describes one real person rather than an
    average of two.
    """

    target = np.bincount(y).max()
    parts_X, parts_y = [X], [y]
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        shortfall = target - rows.size
        if shortfall <= 0 or rows.size < 2:
            continue
        made_X = []
        while len(made_X) < shortfall:
            seed_row = rows[rng.integers(rows.size)]
            if groups is None:
                pool = rows
            else:
                pool = rows[groups[rows] == groups[seed_row]]
                if pool.size < 2:
                    continue  # this person has no same-class partner to blend with
            k = min(SMOTE_NEIGHBOURS + 1, pool.size)
            finder = NearestNeighbors(n_neighbors=k).fit(X[pool])
            _, neighbours = finder.kneighbors(X[seed_row].reshape(1, -1))
            choices = [i for i in neighbours[0] if pool[i] != seed_row]
            if not choices:
                continue
            partner = pool[choices[rng.integers(len(choices))]]
            step = rng.random()
            made_X.append(X[seed_row] + step * (X[partner] - X[seed_row]))
        parts_X.append(np.asarray(made_X))
        parts_y.append(np.full(len(made_X), label, dtype=np.int64))
    return np.vstack(parts_X), np.concatenate(parts_y)


def score(model, X_val, y_val, dataset_val):
    prediction = model.predict(X_val)
    pooled = balanced_accuracy_score(y_val, prediction)
    per_dataset = [
        balanced_accuracy_score(y_val[dataset_val == ds], prediction[dataset_val == ds])
        for ds in sorted(set(dataset_val.tolist()))
    ]
    return float(np.mean(per_dataset)), pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rich", default="outputs/rich_full/rich_all9_features.npz")
    ap.add_argument("--baseline", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--split-from-checkpoint",
                    default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt")
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    rich = np.load(args.rich, allow_pickle=False)
    X = rich["features"]
    y = rich["labels"].astype(np.int64)
    dataset_id = rich["dataset_id"].astype(str)
    subject_id = rich["subject_id"].astype(str)

    archive = load_feature_archive(args.baseline)
    _, training_dataset_ids = select_dataset_view(archive, None)
    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        args.split_from_checkpoint, archive, training_dataset_ids
    )
    tr = keys_mask(dataset_id, subject_id, train_keys)
    va = keys_mask(dataset_id, subject_id, val_keys)
    rng = np.random.default_rng(args.seed)

    X_train, y_train = X[tr], y[tr]
    X_val, y_val, dataset_val = X[va], y[va], dataset_id[va]
    subject_codes = np.unique(
        [f"{d}||{s}" for d, s in zip(dataset_id[tr].tolist(), subject_id[tr].tolist())],
        return_inverse=True,
    )[1]

    counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    print(f"Training class counts: "
          + "  ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, counts)))
    print(f"Imbalance ratio (largest / smallest): {counts.max()/counts.min():.2f}")
    print(f"Validation stays 100% real: {int(va.sum())} trials\n")

    print(f"  {'method':<34}{'train rows':>11}{'DATASET-MACRO':>15}{'pooled':>9}")
    print("  " + "-" * 70)
    results = {}
    for name, builder, weighted in [
        ("none (no correction)", None, False),
        ("class weighting (current)", None, True),
        ("random oversample", "random", False),
        ("SMOTE", "smote", False),
        ("SMOTE within-subject", "smote_subject", False),
        ("SMOTE + class weighting", "smote", True),
    ]:
        if builder is None:
            fit_X, fit_y = X_train, y_train
        elif builder == "random":
            fit_X, fit_y = random_oversample(X_train, y_train, rng)
        elif builder == "smote":
            fit_X, fit_y = smote(X_train, y_train, rng)
        else:
            fit_X, fit_y = smote(X_train, y_train, rng, groups=subject_codes)

        weights = balanced_sample_weights(fit_y) if weighted else None
        model = HistGradientBoostingClassifier(random_state=args.seed)
        model.fit(fit_X, fit_y, sample_weight=weights)
        macro, pooled = score(model, X_val, y_val, dataset_val)
        results[name] = macro
        print(f"  {name:<34}{fit_X.shape[0]:>11}{macro:>15.4f}{pooled:>9.4f}")

    print(f"\n  {'chance':<34}{'':>11}{1/len(CLASS_NAMES):>15.4f}"
          f"{1/len(CLASS_NAMES):>9.4f}")
    current = results["class weighting (current)"]
    best_name = max(results, key=results.get)
    print(f"\n  Best: {best_name} at {results[best_name]:.4f} dataset-macro "
          f"({100*(results[best_name]-current):+.2f} points vs what we use today)")
    print(
        "\n  If the synthetic rows land near the weighting row, both are fixing the same\n"
        "  imbalance and there is nothing to gain from doing it twice. A clear win means\n"
        "  the interpolated trials carry something weighting cannot supply."
    )


if __name__ == "__main__":
    main()
