"""Does training on one homogeneous dataset beat training on all nine pooled?

Per-dataset scores reported so far all came from a model trained on all nine
datasets and merely scored separately on each, so they say nothing about training
on one dataset alone. That is the question here.

The comparison is paired. For each dataset the same validation subjects are scored
twice: once by a model trained only on that dataset's training subjects, once by a
model trained on all nine. Identical people, identical trials, identical features,
identical classifier. Only the training data differs, so the difference is
attributable to it.

Two traps this script is built to avoid
---------------------------------------
Cherry-picking. Running nine datasets and quoting the best one is selection after
the fact, and with nine tries a winner appears by luck alone. The summary therefore
leads with the average across all nine and the count of datasets where single-dataset
training won, not with the best row.

False precision. One dataset leaves roughly six validation subjects, and a number
from six people moves a great deal. Every row prints its subject and trial counts,
and rows too small to interpret are marked rather than quietly averaged in.

The pooled model is trained once and reused for every row, exactly as it would be in
deployment: one model serving every recording site.

Usage (from repo root):
    python scripts/preprocessing/single_vs_pooled_check.py
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
    select_dataset_view,
)

MIN_INTERPRETABLE_VAL_TRIALS = 150


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def fit_and_score(X_fit, y_fit, X_score, y_score, seed):
    if len(np.unique(y_fit)) < 2:
        return None
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X_fit, y_fit, sample_weight=balanced_sample_weights(y_fit))
    prediction = model.predict(X_score)
    return (
        balanced_accuracy_score(y_score, prediction),
        accuracy_score(y_score, prediction),
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
    labels = rich["labels"].astype(np.int64)
    dataset_id = rich["dataset_id"].astype(str)
    subject_id = rich["subject_id"].astype(str)

    base_archive = load_feature_archive(args.baseline)
    _, training_dataset_ids = select_dataset_view(base_archive, None)
    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        args.split_from_checkpoint, base_archive, training_dataset_ids
    )
    tr = keys_mask(dataset_id, subject_id, train_keys)
    va = keys_mask(dataset_id, subject_id, val_keys)

    print(f"Features: {X.shape[1]} columns   train {int(tr.sum())}   val {int(va.sum())}")
    print(f"Chance balanced accuracy: {1/len(CLASS_NAMES):.4f}")
    print("Every row scores the SAME validation subjects both ways.\n")

    # One pooled model, trained once, reused for every dataset's validation rows.
    pooled = HistGradientBoostingClassifier(random_state=args.seed)
    pooled.fit(X[tr], labels[tr], sample_weight=balanced_sample_weights(labels[tr]))

    print(f"  {'dataset':<11}{'trainN':>8}{'valN':>7}{'valSub':>7}"
          f"{'single':>9}{'pooled':>9}{'diff':>8}")
    print("  " + "-" * 60)

    differences, small = [], []
    for dataset in sorted(set(dataset_id.tolist())):
        own_train = tr & (dataset_id == dataset)
        own_val = va & (dataset_id == dataset)
        if not own_train.sum() or not own_val.sum():
            continue
        val_subjects = len(set(subject_id[own_val].tolist()))

        single = fit_and_score(X[own_train], labels[own_train],
                               X[own_val], labels[own_val], args.seed)
        pooled_prediction = pooled.predict(X[own_val])
        pooled_balanced = balanced_accuracy_score(labels[own_val], pooled_prediction)
        if single is None:
            print(f"  {dataset:<11}{int(own_train.sum()):>8}{int(own_val.sum()):>7}"
                  f"{val_subjects:>7}{'one class':>9}{pooled_balanced:>9.4f}{'':>8}")
            continue

        difference = single[0] - pooled_balanced
        flag = "" if own_val.sum() >= MIN_INTERPRETABLE_VAL_TRIALS else "  (small)"
        print(f"  {dataset:<11}{int(own_train.sum()):>8}{int(own_val.sum()):>7}"
              f"{val_subjects:>7}{single[0]:>9.4f}{pooled_balanced:>9.4f}"
              f"{difference:>+8.4f}{flag}")
        differences.append(difference)
        if own_val.sum() < MIN_INTERPRETABLE_VAL_TRIALS:
            small.append(dataset)

    # The headline is the average and the win count, never the best single row.
    differences = np.asarray(differences)
    wins = int(np.sum(differences > 0))
    print("\n  " + "-" * 60)
    print(f"  Datasets where single-dataset training won: {wins} of {differences.size}")
    print(f"  Average difference: {100*differences.mean():+.2f} points balanced accuracy")
    print(f"  Median difference:  {100*np.median(differences):+.2f} points")
    if small:
        print(f"  Rows marked (small) have under {MIN_INTERPRETABLE_VAL_TRIALS} "
              f"validation trials: {small}")

    print(
        "\n  Reading this honestly:\n"
        "    average clearly positive -> homogeneous single-dataset training really is\n"
        "                                better, and the pooled approach is costing you.\n"
        "    average near zero or negative -> pooling is doing its job, and a single\n"
        "                                strong dataset was luck rather than a pattern.\n"
        "  One dataset scoring well is not evidence on its own. With nine tries, the\n"
        "  best of them looks good even when nothing real is happening."
    )


if __name__ == "__main__":
    main()
