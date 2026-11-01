"""Feature selection, hyperparameter search, and ensembling, chosen without peeking.

Three things the project has never done, all of which apply to every dataset rather
than a favoured subset:

  feature selection   the comparison table says compact wins -- 48 hand-picked columns
                      (delta + gamma) beat all 235. That grouping was drawn by hand, so
                      a real search should do at least as well and probably better.
  hyperparameter search  every model so far used library defaults. Nothing was tuned.
  ensembling          the LSTM, gradient boosting and CNN all land near 0.44 but make
                      different mistakes, so averaging them usually gains a little.

How choices are made honestly
-----------------------------
Selecting a feature count or a hyperparameter by checking the validation set and
keeping whatever scores highest is how a number gets inflated: with enough candidates,
something wins by luck, and that luck does not repeat. Every decision here is therefore
made by cross-validation INSIDE the training subjects, grouped so a subject never
appears in both halves of a fold. The validation set is scored exactly once at the end,
with the already-chosen settings.

Reported on dataset-macro
-------------------------
Pooled scoring rewards a model for recognising which of the nine datasets a trial came
from, since they have very different class mixes and are partly identifiable from the
signal. Dataset-macro scores each dataset separately then averages, so that shortcut
earns nothing. Pooled is printed alongside for continuity.

Usage (from repo root):
    python scripts/preprocessing/optimize_pipeline.py
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    select_dataset_view,
)

CV_FOLDS = 3
SUBSET_SIZES = (16, 32, 48, 64, 96, 128, 192, None)  # None means every column
HYPERPARAMETERS = [
    {},  # library defaults, what every earlier run used
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300},
    {"max_depth": 3, "learning_rate": 0.1, "l2_regularization": 1.0},
    {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300, "l2_regularization": 1.0},
    {"max_depth": 6, "learning_rate": 0.05, "l2_regularization": 5.0},
    {"max_leaf_nodes": 15, "learning_rate": 0.05, "max_iter": 400,
     "l2_regularization": 1.0, "min_samples_leaf": 40},
    {"max_leaf_nodes": 63, "learning_rate": 0.1, "min_samples_leaf": 100},
]


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def dataset_macro(y_true, y_pred, datasets):
    return float(np.mean([
        balanced_accuracy_score(y_true[datasets == ds], y_pred[datasets == ds])
        for ds in sorted(set(datasets.tolist()))
    ]))


def cross_validate(X, y, groups, datasets, params, seed):
    """Mean dataset-macro across training folds, subjects never split across a fold."""

    scores = []
    for fit_idx, score_idx in GroupKFold(n_splits=CV_FOLDS).split(X, y, groups):
        model = HistGradientBoostingClassifier(random_state=seed, **params)
        model.fit(X[fit_idx], y[fit_idx],
                  sample_weight=balanced_sample_weights(y[fit_idx]))
        scores.append(
            dataset_macro(y[score_idx], model.predict(X[score_idx]), datasets[score_idx])
        )
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rich", default="outputs/rich_full/rich_all9_pac.npz")
    ap.add_argument("--baseline", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--split-from-checkpoint",
                    default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt")
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    rich = np.load(args.rich, allow_pickle=False)
    X = rich["features"]
    names = list(rich["feature_names"].astype(str))
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

    X_train, y_train, ds_train = X[tr], y[tr], dataset_id[tr]
    X_val, y_val, ds_val = X[va], y[va], dataset_id[va]
    groups = np.asarray(
        [f"{d}||{s}" for d, s in zip(dataset_id[tr].tolist(), subject_id[tr].tolist())]
    )

    print(f"Features {X.shape[1]}   train {int(tr.sum())}   validation {int(va.sum())}")
    print(f"All choices made by {CV_FOLDS}-fold subject-grouped CV inside the training"
          f" split.\nValidation is scored once, at the end.\n")
    started = time.time()

    # ---- Stage 1: how many features, ranked without touching validation ----
    ranker = ExtraTreesClassifier(
        n_estimators=300, random_state=args.seed, n_jobs=-1
    ).fit(X_train, y_train, sample_weight=balanced_sample_weights(y_train))
    order = np.argsort(ranker.feature_importances_)[::-1]

    print("Stage 1 - how many features?")
    print(f"  {'columns':>9}{'CV dataset-macro':>20}")
    best_subset, best_subset_score = None, -1.0
    for size in SUBSET_SIZES:
        cols = order if size is None else order[:size]
        score = cross_validate(X_train[:, cols], y_train, groups, ds_train, {}, args.seed)
        label = "all" if size is None else str(size)
        print(f"  {label:>9}{score:>20.4f}")
        if score > best_subset_score:
            best_subset, best_subset_score = cols, score
    chosen = "all" if best_subset.size == X.shape[1] else str(best_subset.size)
    print(f"  -> chose {chosen} columns ({time.time()-started:.0f}s)\n")

    # ---- Stage 2: hyperparameters, on the chosen columns ----
    print("Stage 2 - hyperparameters?")
    best_params, best_param_score = {}, -1.0
    for params in HYPERPARAMETERS:
        score = cross_validate(
            X_train[:, best_subset], y_train, groups, ds_train, params, args.seed
        )
        print(f"  {str(params) if params else 'defaults':<72}{score:>9.4f}")
        if score > best_param_score:
            best_params, best_param_score = params, score
    print(f"  -> chose {best_params if best_params else 'defaults'} "
          f"({time.time()-started:.0f}s)\n")

    # ---- Stage 3: score the choices once, and try an ensemble ----
    print("Stage 3 - final scoring on validation (first and only look)")
    print(f"  {'model':<46}{'DATASET-MACRO':>15}{'pooled':>9}{'acc':>8}")
    print("  " + "-" * 78)

    def report(tag, predictions):
        macro = dataset_macro(y_val, predictions, ds_val)
        print(f"  {tag:<46}{macro:>15.4f}"
              f"{balanced_accuracy_score(y_val, predictions):>9.4f}"
              f"{accuracy_score(y_val, predictions):>8.4f}")
        return macro

    baseline = HistGradientBoostingClassifier(random_state=args.seed)
    baseline.fit(X_train, y_train, sample_weight=balanced_sample_weights(y_train))
    before = report("baseline: all features, defaults", baseline.predict(X_val))

    tuned = HistGradientBoostingClassifier(random_state=args.seed, **best_params)
    tuned.fit(X_train[:, best_subset], y_train,
              sample_weight=balanced_sample_weights(y_train))
    after = report(f"selected {chosen} columns + tuned", tuned.predict(X_val[:, best_subset]))

    # Diverse members, so their mistakes are less likely to coincide.
    members = [
        ("gbdt-tuned", tuned, best_subset),
        ("gbdt-defaults", baseline, np.arange(X.shape[1])),
        ("extra-trees", ExtraTreesClassifier(
            n_estimators=400, random_state=args.seed, n_jobs=-1), best_subset),
        ("logistic", LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=args.seed), best_subset),
    ]
    probabilities = []
    for tag, model, cols in members:
        if tag not in ("gbdt-tuned", "gbdt-defaults"):
            fit_X = X_train[:, cols]
            if tag == "logistic":
                mean, sd = fit_X.mean(0), fit_X.std(0) + 1e-9
                model.fit((fit_X - mean) / sd, y_train,
                          sample_weight=balanced_sample_weights(y_train))
                probabilities.append(model.predict_proba((X_val[:, cols] - mean) / sd))
                continue
            model.fit(fit_X, y_train, sample_weight=balanced_sample_weights(y_train))
        probabilities.append(model.predict_proba(X_val[:, cols]))
    ensemble = report("ensemble of 4 models", np.mean(probabilities, axis=0).argmax(1))

    print(f"\n  Reference, same split: old alpha/beta/theta 0.4146,"
          f" rich features 0.4415, delta+gamma 0.4467")
    print(f"  selection + tuning: {100*(after-before):+.2f} points")
    print(f"  ensembling on top:  {100*(ensemble-after):+.2f} points")
    print(f"  total:              {100*(max(after, ensemble)-before):+.2f} points")
    print(f"\n  Total time {time.time()-started:.0f}s")
    print("\n  Top 15 features the ranker kept:")
    for i in best_subset[:15]:
        print(f"    {names[i]}")


if __name__ == "__main__":
    main()
