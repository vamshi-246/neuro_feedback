"""How much does the model improve per calibration trial from the new user?

An earlier probe found personalization useless, but it gave each model about 15
examples of the person, because the median subject has only 30 trials and the split
was in half. Fifteen examples is not a calibration session.

Fifty-eight subjects here have over 100 trials, so the question can be asked
properly: give the model 5, then 10, then 20, then 40 real examples of a person and
watch what happens. The shape of that curve is the answer a neurofeedback device
needs, because it says how long the user must sit through calibration before the
device is useful, or whether calibration helps at all.

Design choices that keep the answer honest
------------------------------------------
The test trials never change. Each subject's LAST 20 trials are held out once, and
every calibration size is scored on exactly those trials. Without this, larger
calibration sets would leave different test sets behind and the comparison would
measure the test set rather than the calibration.

Calibration uses the EARLIEST trials, never trials surrounding the test block. That
matches how a device would work: the user calibrates at the start of a session and
the device is used afterwards. Sampling calibration trials from around the test block
would let neighbouring, near-duplicate trials leak in and inflate the result.

Only validation subjects are used. The global model trained on the training subjects
has already seen those people, so measuring personalization on them would flatter the
no-calibration baseline.

Three approaches are compared at every calibration size:

  global          the pooled model, no personalization at all. Flat by definition,
                  and the line the others must beat.
  personal        a model trained only on this person's calibration trials, knowing
                  nothing about anyone else.
  recalibrated    the global model's own class probabilities, corrected per person
                  using the calibration trials. Keeps everything the pooled model
                  learned from 406 people and only adjusts how it is read for this
                  one, which is why it needs far fewer examples than a personal model.

Usage (from repo root):
    python scripts/preprocessing/calibration_curve_check.py
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
from sklearn.metrics import balanced_accuracy_score

# Scoring one subject at a time means a prediction can name a class that subject
# never had. balanced_accuracy_score already averages over the classes present, so
# the resulting notice is expected here and only adds noise.
warnings.filterwarnings(
    "ignore", message="y_pred contains classes not in y_true", category=UserWarning
)

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    select_dataset_view,
)

CALIBRATION_SIZES = (0, 10, 20, 40, 60, 80)
TEST_BLOCK = 20
MIN_TRIALS = 100


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rich", default="outputs/rich_full/rich_all9_features.npz")
    ap.add_argument("--baseline", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--split-from-checkpoint",
                    default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt")
    ap.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    ap.add_argument("--test-block", type=int, default=TEST_BLOCK)
    ap.add_argument(
        "--max-calibration",
        type=int,
        default=max(CALIBRATION_SIZES),
        help=(
            "Largest calibration size to test. Lowering it admits subjects with "
            "fewer trials, trading the deepest point on the curve for a bigger "
            "cohort; the screening requires every subject to support every size."
        ),
    )
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    sizes = tuple(n for n in CALIBRATION_SIZES if n <= args.max_calibration)
    if len(sizes) < 2:
        ap.error("--max-calibration is too small to form a curve")

    rich = np.load(args.rich, allow_pickle=False)
    X = rich["features"]
    labels = rich["labels"].astype(np.int64)
    epoch_index = rich["epoch_index"].astype(np.int64)
    dataset_id = rich["dataset_id"].astype(str)
    subject_id = rich["subject_id"].astype(str)

    base_archive = load_feature_archive(args.baseline)
    _, training_dataset_ids = select_dataset_view(base_archive, None)
    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        args.split_from_checkpoint, base_archive, training_dataset_ids
    )
    tr = keys_mask(dataset_id, subject_id, train_keys)
    va = keys_mask(dataset_id, subject_id, val_keys)

    # One pooled model, trained once on the 406 training subjects.
    global_model = HistGradientBoostingClassifier(random_state=args.seed)
    global_model.fit(X[tr], labels[tr], sample_weight=balanced_sample_weights(labels[tr]))

    subject_keys = np.asarray(
        [f"{d}||{s}" for d, s in zip(dataset_id.tolist(), subject_id.tolist())]
    )
    qualifying = []
    for key in sorted(set(subject_keys[va].tolist())):
        mask = va & (subject_keys == key)
        if int(mask.sum()) >= args.min_trials:
            qualifying.append(key)

    print(f"Validation subjects with at least {args.min_trials} trials: {len(qualifying)}")
    if not qualifying:
        raise SystemExit("no subject has enough trials; lower --min-trials")
    by_dataset = {}
    for key in qualifying:
        by_dataset[key.split("||")[0]] = by_dataset.get(key.split("||")[0], 0) + 1
    print(f"  {by_dataset}")
    print(f"Test block: each subject's LAST {args.test_block} trials, identical at "
          f"every calibration size")
    print(f"Calibration: that subject's EARLIEST N trials")

    # Screen once, up front: keep only subjects usable at EVERY calibration size.
    # Otherwise a subject dropped at one size but kept at another makes the columns
    # score different cohorts, and the no-calibration row wobbles when it should be
    # perfectly flat.
    screened = []
    for key in qualifying:
        mask = va & (subject_keys == key)
        order = np.argsort(epoch_index[mask], kind="stable")
        sub_y = labels[mask][order]
        if len(np.unique(sub_y[-args.test_block:])) < 2:
            continue
        pool_y = sub_y[:-args.test_block]
        if any(
            n > pool_y.size or len(np.unique(pool_y[:n])) < 2
            for n in sizes if n > 0
        ):
            continue
        screened.append(key)
    dropped = len(qualifying) - len(screened)
    qualifying = screened
    print(f"Fixed cohort: {len(qualifying)} subjects usable at every size "
          f"({dropped} dropped)\n")
    if not qualifying:
        raise SystemExit("no subject is usable at every calibration size")

    results = {name: {n: [] for n in sizes}
               for name in ("global", "personal", "recalibrated")}
    usable_counts = {n: 0 for n in sizes}

    for key in qualifying:
        mask = va & (subject_keys == key)
        order = np.argsort(epoch_index[mask], kind="stable")
        sub_X = X[mask][order]
        sub_y = labels[mask][order]

        test_X, test_y = sub_X[-args.test_block:], sub_y[-args.test_block:]
        pool_X, pool_y = sub_X[:-args.test_block], sub_y[:-args.test_block]
        if len(np.unique(test_y)) < 2:
            continue  # a single-class test block cannot show discrimination

        global_probabilities = global_model.predict_proba(test_X)
        global_score = balanced_accuracy_score(test_y, global_probabilities.argmax(1))

        for n in sizes:
            if n == 0:
                results["global"][0].append(global_score)
                results["personal"][0].append(global_score)
                results["recalibrated"][0].append(global_score)
                usable_counts[0] += 1
                continue
            if n > pool_X.shape[0]:
                continue
            cal_X, cal_y = pool_X[:n], pool_y[:n]
            if len(np.unique(cal_y)) < 2:
                continue  # nothing to learn from a single-class calibration set
            usable_counts[n] += 1

            results["global"][n].append(global_score)

            personal = HistGradientBoostingClassifier(
                random_state=args.seed, max_depth=3, max_iter=100,
                learning_rate=0.1, l2_regularization=1.0,
            ).fit(cal_X, cal_y, sample_weight=balanced_sample_weights(cal_y))
            results["personal"][n].append(
                balanced_accuracy_score(test_y, personal.predict(test_X))
            )

            # Correct how the pooled model's probabilities are read for this person,
            # rather than relearning pain from scratch on a handful of trials.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                corrector = LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=args.seed
                ).fit(global_model.predict_proba(cal_X), cal_y)
            results["recalibrated"][n].append(
                balanced_accuracy_score(test_y, corrector.predict(global_probabilities))
            )

    print(f"  {'calibration trials':<22}" + "".join(f"{n:>9}" for n in sizes))
    print(f"  {'subjects usable':<22}"
          + "".join(f"{usable_counts[n]:>9}" for n in sizes))
    print("  " + "-" * 68)
    for name in ("global", "personal", "recalibrated"):
        row = f"  {name:<22}"
        for n in sizes:
            values = results[name][n]
            row += f"{np.mean(values):>9.4f}" if values else f"{'-':>9}"
        print(row)

    best_n = max(sizes)
    baseline = np.mean(results["global"][best_n]) if results["global"][best_n] else float("nan")
    print("\n  Change versus no calibration, at "
          f"{best_n} calibration trials:")
    for name in ("personal", "recalibrated"):
        values = results[name][best_n]
        if values:
            print(f"    {name:<16}{100*(np.mean(values)-baseline):+.2f} points")

    print(
        "\n  Reading the curve:\n"
        "    rises with more trials -> calibration works, and the slope tells you how\n"
        "                              long a session must be to reach a target.\n"
        "    flat                   -> the model cannot use examples of a person, and\n"
        "                              personalization is finished as an idea.\n"
        "  Every column scores the SAME held-out trials, so a rise cannot come from\n"
        "  an easier test set."
    )


if __name__ == "__main__":
    main()
