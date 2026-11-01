"""Three untested ideas, each applied ON TOP of the current best configuration.

The best model so far is not the default one. optimize_pipeline.py established, by
cross-validation inside the training subjects, that the winning setup is the top 16
features by importance with tuned boosting parameters -- 0.4505 dataset-macro against
0.4429 for all 235 columns at library defaults. Every test here therefore starts from
that configuration rather than from the earlier baseline, so a gain is a gain over the
best we have and not a rediscovery of ground already covered.

Feature ranking is refitted for each specific task using training rows only, because
the columns that separate Moderate from everything else are not necessarily the ones
that separate Low from High.

The three ideas, from a literature summary of what typically lifts three-class EEG pain
models, minus the suggestions this project has already tested and rejected:

1. Hierarchical cascade
   One three-way decision replaced by two binary ones, on the argument that Moderate is
   a transition zone overlapping both neighbours. This project's own numbers make it
   plausible: Low against High alone reaches 72.4% where the three-way reaches 48%. Both
   orderings are tried, since it is not obvious whether to isolate the ambiguous class
   first or last.

2. Band ratios
   Theta over alpha, beta over alpha and the rest. A ratio is invariant to anything that
   scales a whole recording -- electrode contact, skull thickness, amplifier gain --
   which is the between-person variation the transfer failures point at. They are added
   to the candidate pool and the selection is rerun, so the question asked is whether a
   ratio earns a place among the best 16, not merely whether it can be computed.

3. Per-subject label normalisation
   Z-scoring each person's own ratings before binning, so a label means "high for this
   person" rather than "high on the 0-10 scale". This is NOT the per-subject
   normalisation already tested and rejected, which centred the FEATURES and deleted
   between-person differences that carry real signal. This centres the LABELS.
   It answers a different question, so it is reported separately rather than as an
   improvement, and it needs each user's rating distribution -- another calibration
   requirement.

Usage (from repo root):
    python scripts/preprocessing/cascade_and_labels_check.py
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    select_dataset_view,
)

# Both chosen by cross-validation inside the training subjects, in optimize_pipeline.py.
SELECTED_FEATURES = 16
TUNED = {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300,
         "l2_regularization": 1.0}
LOG_POWER = re.compile(r"^(?P<ch>[^:]+):(?P<band>delta|theta|alpha|beta|gamma):"
                       r"w(?P<w>\d+):log_absolute$")


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def best_config(X_train, y_train, seed, k=SELECTED_FEATURES):
    """Rank on training rows, keep the top k, fit with the tuned parameters.

    Returns (model, columns) so the same columns can be applied to validation.
    """

    ranker = ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    ranker.fit(X_train, y_train, sample_weight=balanced_sample_weights(y_train))
    columns = np.argsort(ranker.feature_importances_)[::-1][:k]
    model = HistGradientBoostingClassifier(random_state=seed, **TUNED)
    model.fit(X_train[:, columns], y_train,
              sample_weight=balanced_sample_weights(y_train))
    return model, columns


def band_ratio_columns(X, names):
    """Every within-channel, within-window band ratio. Logs already, so a subtraction."""

    grouped = {}
    for index, name in enumerate(names):
        match = LOG_POWER.match(name)
        if match:
            grouped.setdefault((match.group("ch"), match.group("w")), []).append(
                (match.group("band"), index))
    columns, labels = [], []
    for (channel, window), entries in sorted(grouped.items()):
        entries.sort()
        for i, (band_a, col_a) in enumerate(entries):
            for band_b, col_b in entries[i + 1:]:
                columns.append(X[:, col_a] - X[:, col_b])
                labels.append(f"{channel}:w{window}:{band_a}_over_{band_b}")
    if not columns:
        return np.empty((X.shape[0], 0)), []
    return np.stack(columns, axis=1), labels


def per_subject_z_labels(ratings, subject_keys, cut_quantiles):
    """Bin each person's own z-scored ratings at the population's class rates."""

    z = np.zeros_like(ratings)
    for key in set(subject_keys.tolist()):
        mask = subject_keys == key
        values = ratings[mask]
        spread = values.std()
        # Someone who gave the same rating every time has no within-person contrast;
        # centring alone stops them being rescaled by noise.
        z[mask] = (values - values.mean()) / (spread if spread > 1e-6 else 1.0)
    low, high = np.quantile(z, cut_quantiles)
    return np.where(z < low, 0, np.where(z < high, 1, 2)).astype(np.int64)


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
    ratings = rich["ratings"].astype(np.float64)
    dataset_id = rich["dataset_id"].astype(str)
    subject_id = rich["subject_id"].astype(str)

    archive = load_feature_archive(args.baseline)
    _, training_dataset_ids = select_dataset_view(archive, None)
    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        args.split_from_checkpoint, archive, training_dataset_ids
    )
    tr = keys_mask(dataset_id, subject_id, train_keys)
    va = keys_mask(dataset_id, subject_id, val_keys)
    ds_val = dataset_id[va]
    subject_keys = np.asarray(
        [f"{d}||{s}" for d, s in zip(dataset_id.tolist(), subject_id.tolist())]
    )

    def report(tag, y_true, y_pred, note=""):
        macro = float(np.mean([
            balanced_accuracy_score(y_true[ds_val == ds], y_pred[ds_val == ds])
            for ds in sorted(set(ds_val.tolist()))
        ]))
        print(f"  {tag:<48}{macro:>14.4f}"
              f"{balanced_accuracy_score(y_true, y_pred):>10.4f}"
              f"{accuracy_score(y_true, y_pred):>8.4f}  {note}")
        return macro

    print(f"Starting configuration: top {SELECTED_FEATURES} features + tuned parameters")
    print(f"  {TUNED}")
    print(f"Pool {X.shape[1]} features   train {int(tr.sum())}   "
          f"validation {int(va.sum())}")
    print(f"Chance {1/len(CLASS_NAMES):.4f}\n")
    print(f"  {'approach':<48}{'DATASET-MACRO':>14}{'pooled':>10}{'acc':>8}")
    print("  " + "-" * 88)

    model, columns = best_config(X[tr], y[tr], args.seed)
    current = report("CURRENT BEST: 16 features + tuned",
                     y[va], model.predict(X[va][:, columns]))

    # --- 1. cascades, each stage getting its own selection and the tuned parameters ---
    is_moderate = (y == 1).astype(np.int64)
    edges = y != 1

    stage_a, cols_a = best_config(X[tr], is_moderate[tr], args.seed)
    stage_b, cols_b = best_config(X[tr & edges], (y[tr & edges] == 2).astype(np.int64),
                                  args.seed)
    predicted = np.where(
        stage_a.predict(X[va][:, cols_a]) == 1, 1,
        np.where(stage_b.predict(X[va][:, cols_b]) == 1, 2, 0),
    )
    report("cascade A: Moderate first, then Low vs High", y[va], predicted)

    predicted = np.where(
        stage_a.predict(X[va][:, cols_a]) == 1, 1,
        np.where(stage_b.predict(X[va][:, cols_b]) == 1, 2, 0),
    )
    # Order B asks the easy question first and only then whether it is really Moderate,
    # so a confident Low/High answer is allowed to override the Moderate detector.
    near_high = stage_b.predict(X[va][:, cols_b])
    moderate_probability = stage_a.predict_proba(X[va][:, cols_a])[:, 1]
    predicted = np.where(moderate_probability > 0.5, 1,
                         np.where(near_high == 1, 2, 0))
    report("cascade B: Low vs High first, then Moderate", y[va], predicted)

    # --- 2. band ratios, added to the pool so selection can choose them or not ---
    ratios, ratio_names = band_ratio_columns(X, names)
    if ratios.shape[1]:
        pooled = np.hstack([X, ratios])
        pooled_names = names + ratio_names
        model_r, cols_r = best_config(pooled[tr], y[tr], args.seed)
        report(f"pool + {ratios.shape[1]} band ratios, reselected",
               y[va], model_r.predict(pooled[va][:, cols_r]))
        picked = [pooled_names[i] for i in cols_r if pooled_names[i] in ratio_names]
        print(f"       ratios chosen into the top {SELECTED_FEATURES}: "
              f"{len(picked)}{' -> ' + ', '.join(picked[:4]) if picked else ''}")

    # --- 3. per-subject rating normalisation (different question) ---
    counts = np.bincount(y[tr], minlength=len(CLASS_NAMES)).astype(np.float64)
    z_labels = per_subject_z_labels(
        ratings, subject_keys, np.cumsum(counts / counts.sum())[:2]
    )
    model_z, cols_z = best_config(X[tr], z_labels[tr], args.seed)
    print("  " + "-" * 88)
    report("per-subject z-scored labels", z_labels[va],
           model_z.predict(X[va][:, cols_z]), note="DIFFERENT TASK")
    print(f"\n  Those labels agree with the original ones on "
          f"{100*float(np.mean(z_labels == y)):.1f}% of trials, so that row answers a")
    print( "  different question and is not an improvement on the rows above it.")
    print(f"\n  Current best reproduced here: {current:.4f} "
          f"(optimize_pipeline.py reported 0.4505)")


if __name__ == "__main__":
    main()
