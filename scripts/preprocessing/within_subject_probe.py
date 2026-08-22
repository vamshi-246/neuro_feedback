"""Read-only probe: does the model tell one person's trials apart, or just rank people?

Two earlier probes narrowed the problem down.  feature_ceiling_check.py showed the
features memorize training trials perfectly (99.99%) but transfer poorly to new
people (~41%).  normalization_probe.py then showed that removing each person's
average makes validation WORSE, which means that average was carrying real
signal -- subjects genuinely differ in how much pain they reported.

Together those raise a specific worry: a model can score ~45% by learning "this
looks like a person who rates high" without ever learning "this trial hurt more
than that trial."  Those are different skills, and only the second one is useful
for neurofeedback, where the system tracks a single known user over time.

Part A measures within-subject discrimination directly.  For each validation
subject it compares the shared cross-subject model against a baseline that simply
predicts that subject's own most common class.  A model that cannot beat that
baseline is ranking people, not reading trials.

Part B simulates personalized calibration, which is the realistic neurofeedback
setup.  Each subject's trials are split by epoch order: their earlier trials act
as a labelled calibration session, their later trials are held out.  This keeps
the split honest -- calibrate first, predict afterwards -- instead of randomly
mixing a session, which would let near-duplicate neighbouring trials leak.

Nothing is written to disk and the locked test split is never touched.

Usage (from repo root):
    python scripts/preprocessing/within_subject_probe.py
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    model_feature_view,
    select_dataset_view,
)

MIN_TRIALS_FOR_PERSONALIZATION = 30

# Scoring one subject at a time means a prediction can name a class that subject
# never had.  balanced_accuracy_score already averages over the classes that are
# present, so the resulting notice is expected here and only adds noise.
warnings.filterwarnings(
    "ignore", message="y_pred contains classes not in y_true", category=UserWarning
)


def balanced_sample_weights(labels):
    """Give equal total mass to whichever classes are actually present.

    A single subject's calibration half often contains only two of the three
    classes, so the weights must balance the classes that exist rather than
    assume all three do.
    """

    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[label] for label in labels.tolist()], dtype=np.float64)


def majority_predictions(fit_labels, n_predict):
    """Predict the most common label seen during fitting."""

    counts = np.bincount(fit_labels, minlength=len(CLASS_NAMES))
    return np.full(n_predict, int(counts.argmax()), dtype=np.int64)


def summarize(name, accuracies, balanced):
    if not accuracies:
        print(f"  {name:<44} (no qualifying subjects)")
        return
    print(
        f"  {name:<44} accuracy={np.mean(accuracies):.4f}  "
        f"balanced={np.mean(balanced):.4f}"
    )


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
    epoch_index = archive["epoch_index"].astype(np.int64)
    y = archive["labels"].astype(np.int64)
    train_mask = keys_mask(dataset_id, subject_id, train_keys)
    val_mask = keys_mask(dataset_id, subject_id, val_keys)

    power, _order = model_feature_view(archive, args.feature_mode)
    X = np.log1p(power.reshape(power.shape[0], -1))

    print(f"Feature mode: {args.feature_mode} ({X.shape[1]} columns)")
    print(f"Train trials {int(train_mask.sum())}   Validation trials {int(val_mask.sum())}")
    print(f"Chance balanced accuracy: {1/len(CLASS_NAMES):.4f}\n")

    # The shared cross-subject model, trained exactly like the winning probe.
    global_model = HistGradientBoostingClassifier(random_state=args.seed)
    global_model.fit(X[train_mask], y[train_mask],
                     sample_weight=balanced_sample_weights(y[train_mask]))
    global_val_pred = global_model.predict(X[val_mask])

    val_subjects = np.array(
        [f"{d}||{s}" for d, s in zip(dataset_id[val_mask].tolist(),
                                     subject_id[val_mask].tolist())]
    )
    y_val = y[val_mask]
    epoch_val = epoch_index[val_mask]
    X_val = X[val_mask]

    # ---------------- Part A: within-subject discrimination ----------------
    print("=" * 72)
    print("PART A -- can the shared model tell one person's own trials apart?")
    print("=" * 72)
    model_acc, model_bal, base_acc, base_bal = [], [], [], []
    multiclass_subjects = 0
    for subject in sorted(set(val_subjects.tolist())):
        mask = val_subjects == subject
        truth = y_val[mask]
        if len(set(truth.tolist())) < 2:
            continue  # a single-class subject cannot show discrimination
        multiclass_subjects += 1
        pred = global_val_pred[mask]
        # Baseline gets an unfair advantage: it is told this subject's own
        # majority class up front.  Beating it is the bar that matters.
        base = majority_predictions(truth, truth.size)
        model_acc.append(accuracy_score(truth, pred))
        model_bal.append(balanced_accuracy_score(truth, pred))
        base_acc.append(accuracy_score(truth, base))
        base_bal.append(balanced_accuracy_score(truth, base))

    print(f"\nValidation subjects with more than one class: {multiclass_subjects}")
    print("Averaged over those subjects (each subject counted once):\n")
    summarize("shared cross-subject model", model_acc, model_bal)
    summarize("baseline: this subject's own majority class", base_acc, base_bal)
    gap = np.mean(model_bal) - np.mean(base_bal)
    print(
        f"\n  -> within-subject balanced-accuracy advantage over the baseline: "
        f"{100*gap:+.2f} points"
    )
    print(
        "     (the baseline is handed each subject's majority class, so a small\n"
        "      positive number here still means real trial-level discrimination)"
    )

    # ---------------- Part B: personalized calibration ----------------
    print("\n" + "=" * 72)
    print("PART B -- personalized calibration (earlier trials -> later trials)")
    print("=" * 72)
    per_subject_lr = ([], [])
    per_subject_gb = ([], [])
    global_on_same = ([], [])
    calib_majority = ([], [])
    qualifying = 0

    for subject in sorted(set(val_subjects.tolist())):
        mask = val_subjects == subject
        if int(mask.sum()) < MIN_TRIALS_FOR_PERSONALIZATION:
            continue
        order = np.argsort(epoch_val[mask], kind="stable")
        sub_X, sub_y = X_val[mask][order], y_val[mask][order]
        sub_global = global_val_pred[mask][order]
        cut = sub_y.size // 2
        cal_X, cal_y = sub_X[:cut], sub_y[:cut]
        test_X, test_y = sub_X[cut:], sub_y[cut:]
        # Both halves must be usable: something to learn from, something to score.
        if len(set(cal_y.tolist())) < 2 or len(set(test_y.tolist())) < 2:
            continue
        qualifying += 1

        scaler = StandardScaler().fit(cal_X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            lr = LogisticRegression(
                max_iter=2000, C=0.1, class_weight="balanced",
                random_state=args.seed,
            ).fit(scaler.transform(cal_X), cal_y)
        lr_pred = lr.predict(scaler.transform(test_X))

        gb = HistGradientBoostingClassifier(
            random_state=args.seed, max_depth=3, max_iter=100,
            learning_rate=0.1, l2_regularization=1.0,
        ).fit(cal_X, cal_y, sample_weight=balanced_sample_weights(cal_y))
        gb_pred = gb.predict(test_X)

        maj_pred = majority_predictions(cal_y, test_y.size)
        glob_pred = sub_global[cut:]

        for store, pred in (
            (per_subject_lr, lr_pred),
            (per_subject_gb, gb_pred),
            (global_on_same, glob_pred),
            (calib_majority, maj_pred),
        ):
            store[0].append(accuracy_score(test_y, pred))
            store[1].append(balanced_accuracy_score(test_y, pred))

    print(
        f"\nSubjects with >= {MIN_TRIALS_FOR_PERSONALIZATION} trials and both halves "
        f"usable: {qualifying}"
    )
    print("Averaged over those subjects, scored on their LATER trials only:\n")
    summarize("shared cross-subject model (no calibration)", *global_on_same)
    summarize("baseline: majority of their calibration half", *calib_majority)
    summarize("personalized logistic regression", *per_subject_lr)
    summarize("personalized gradient boosting", *per_subject_gb)

    if qualifying:
        best_personal = max(np.mean(per_subject_lr[1]), np.mean(per_subject_gb[1]))
        delta = best_personal - np.mean(global_on_same[1])
        print(
            f"\n  -> personalization changes balanced accuracy by "
            f"{100*delta:+.2f} points versus the shared model"
        )
        print(
            "     Calibration halves are small, so treat this as a direction\n"
            "     indicator rather than the ceiling personalization could reach."
        )


if __name__ == "__main__":
    main()
