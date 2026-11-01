"""Predict the 0-10 rating as a number, then map that number to Low/Moderate/High.

Classifying the three labels directly throws away the size of a mistake. Predicting
6.8 when the truth is 7.0 currently counts as a full class error, exactly as bad as
predicting 2.0, even though the two sensations are indistinguishable. The
rating-boundary report showed this is where the damage concentrates: in the latest
LSTM run, 51.3% of trials rated 6.0-6.9 were called High.

Training on the continuous rating makes near misses cheap. The output stays
Low/Moderate/High, so the task, the labels, and the RTL's three states are unchanged.
What changes is only how a predicted number becomes a class, and this script measures
five ways of doing that.

The shrinkage problem
---------------------
A regressor pulls predictions toward the mean, so predicted ratings are far less
spread out than real ones. Applying the official cut points 4 and 7 to compressed
predictions pushes nearly everything into Moderate, which looks like a failure of
the whole idea when it is really a failure of the mapping.

The mappings compared
---------------------
  fixed 4/7            the project's rule applied as-is, shrinkage and all
  distribution-matched cuts placed so predicted class rates match training rates
  optimized            the two cuts that maximize balanced accuracy
  isotonic -> 4/7      a monotone correction that puts predictions back on the real
                       rating scale, after which the official 4/7 rule applies to a
                       properly scaled number. This is the only option that keeps
                       the lab's rating-scale anchors and still fixes the shrinkage.
  logistic on score    a small classifier over the predicted score; soft boundaries,
                       and it yields a confidence per prediction rather than a bare
                       label

Why the mapping is fitted out-of-fold
-------------------------------------
The regressor is optimistically accurate on its own training rows, so cut points
fitted there would sit in the wrong place and every mapping would look better than
it is. Mappings are therefore fitted on out-of-fold predictions produced by
subject-grouped cross-validation inside the training split. No validation rating,
label, or prediction is ever used to choose a mapping.

Usage (from repo root):
    python scripts/preprocessing/regression_then_bin_check.py
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from feature_extraction import bin_rating
from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    select_dataset_view,
)

OFFICIAL_CUTS = (4.0, 7.0)
CV_FOLDS = 5


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def bin_with_cuts(values, cuts):
    low, high = cuts
    return np.where(values < low, 0, np.where(values < high, 1, 2)).astype(np.int64)


def official_bins(values):
    """Apply the project's own rule, clipped because a regressor can leave 0..10."""

    return np.asarray(
        [bin_rating(float(v)) for v in np.clip(values, 0.0, 10.0)], dtype=np.int64
    )


def distribution_matched_cuts(scores, labels):
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    proportions = counts / counts.sum()
    low = float(np.quantile(scores, proportions[0]))
    high = float(np.quantile(scores, proportions[0] + proportions[1]))
    return (low, high) if low < high else OFFICIAL_CUTS


def optimized_cuts(scores, labels):
    """Search the cut pair maximizing balanced accuracy on out-of-fold scores."""

    grid = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 49)))
    best, best_score = OFFICIAL_CUTS, -1.0
    for i, low in enumerate(grid):
        for high in grid[i + 1:]:
            score = balanced_accuracy_score(labels, bin_with_cuts(scores, (low, high)))
            if score > best_score:
                best, best_score = (float(low), float(high)), score
    return best


def report(tag, y_true, y_pred):
    bal = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"  {tag:<50}{bal:>12.4f}{acc:>10.4f}")
    return bal


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
    groups = np.asarray(
        [f"{d}||{s}" for d, s in zip(dataset_id[tr].tolist(), subject_id[tr].tolist())]
    )

    print(f"EEG features: {X.shape[1]} columns   train {int(tr.sum())}   val {int(va.sum())}")
    print(f"Chance balanced accuracy: {1/len(CLASS_NAMES):.4f}")
    print(f"Mappings fitted on {CV_FOLDS}-fold subject-grouped out-of-fold predictions.\n")

    print(f"  {'approach':<50}{'VAL balanced':>12}{'val acc':>10}")
    print("  " + "-" * 72)

    classifier = HistGradientBoostingClassifier(random_state=args.seed)
    classifier.fit(X[tr], labels[tr], sample_weight=balanced_sample_weights(labels[tr]))
    direct = report("classify Low/Moderate/High directly (current)",
                    labels[va], classifier.predict(X[va]))

    regressor = HistGradientBoostingRegressor(random_state=args.seed)
    # Out-of-fold scores: the regressor never saw the rows it scores here, so cut
    # points fitted on them are not inflated by its own memorization.
    oof = cross_val_predict(
        regressor, X[tr], ratings[tr],
        cv=GroupKFold(n_splits=CV_FOLDS), groups=groups,
    )
    regressor.fit(X[tr], ratings[tr])
    val_scores = regressor.predict(X[va])

    print("  " + "-" * 72)
    results = {}
    results["fixed 4/7"] = report(
        "regress -> bin at fixed 4/7", labels[va], official_bins(val_scores))

    cuts = distribution_matched_cuts(oof, labels[tr])
    results["distribution-matched"] = report(
        f"regress -> distribution-matched {cuts[0]:.2f}/{cuts[1]:.2f}",
        labels[va], bin_with_cuts(val_scores, cuts))

    best_cuts = optimized_cuts(oof, labels[tr])
    results["optimized"] = report(
        f"regress -> optimized cuts {best_cuts[0]:.2f}/{best_cuts[1]:.2f}",
        labels[va], bin_with_cuts(val_scores, best_cuts))

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=10.0)
    isotonic.fit(oof, ratings[tr])
    calibrated = isotonic.predict(val_scores)
    results["isotonic -> 4/7"] = report(
        "regress -> isotonic calibration -> official 4/7",
        labels[va], official_bins(calibrated))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        soft = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=args.seed
        ).fit(oof.reshape(-1, 1), labels[tr])
    probabilities = soft.predict_proba(val_scores.reshape(-1, 1))
    results["logistic on score"] = report(
        "regress -> logistic classifier on the score",
        labels[va], probabilities.argmax(axis=1))

    print("\n  Regression quality on validation subjects:")
    mae = float(np.mean(np.abs(val_scores - ratings[va])))
    corr = float(np.corrcoef(val_scores, ratings[va])[0, 1])
    print(f"    mean absolute error            {mae:.3f} rating points")
    print(f"    correlation with true rating   r = {corr:.3f}")
    print(f"    true ratings  mean {ratings[va].mean():.2f}  sd {ratings[va].std():.2f}")
    print(f"    raw scores    mean {val_scores.mean():.2f}  sd {val_scores.std():.2f}")
    print(f"    calibrated    mean {calibrated.mean():.2f}  sd {calibrated.std():.2f}")
    shrink = val_scores.std() / ratings[va].std()
    print(f"    spread retained before calibration: {100*shrink:.0f}%")

    confident = probabilities.max(axis=1) >= 0.5
    if confident.any():
        confident_bal = balanced_accuracy_score(
            labels[va][confident], probabilities.argmax(axis=1)[confident])
        print(f"\n  Confidence is usable as a gate: on the {100*confident.mean():.0f}% of "
              f"trials where the\n  soft classifier is at least 50% sure, balanced "
              f"accuracy is {confident_bal:.4f}.")

    best_name = max(results, key=results.get)
    print(f"\n  Best mapping: {best_name} at {results[best_name]:.4f} "
          f"({100*(results[best_name]-direct):+.2f} points vs direct classification)")
    print(
        "\n  Correlation r is the cleanest read on your idea: it says how much of the\n"
        "  0-10 rating the EEG explains before any threshold exists. No choice of cut\n"
        "  points can add information the score itself does not contain."
    )


if __name__ == "__main__":
    main()
