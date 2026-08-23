"""Settle whether calibration works, on whichever version of the question is asked.

calibration_curve_check.py suggested a personal model overtakes the pooled one at
60-80 calibration trials, but only 12 validation subjects have enough trials to reach
that point. Thirty-two more sit in the training split and cannot be used directly:
the pooled model has already seen them, so its no-calibration score on them would be
flattered and the comparison rigged in favour of doing nothing.

The fix is to retrain the pooled model once per subject, leaving that subject out.
The subject is then genuinely unseen, exactly as a validation subject is, and the
cohort grows from 11 to roughly 43. Held-out test subjects are never touched.

Tasks
-----
  3class   Low / Moderate / High, the project's current question (chance 1/3)
  severe   is this severe pain, rating >= 7 (chance 1/2)
  lowhigh  clearly low against clearly high, Moderate trials dropped (chance 1/2)

The three-class task tops out well below 70% however much calibration is supplied,
because the curve is already flattening by 80 trials. The binary framings start far
higher, so they are where a 70% target can actually be met, and this script measures
that rather than extrapolating it.

Methods
-------
  global        the pooled model, no personalization. Flat by construction.
  personal      a model built only from this person's calibration trials.
  recalibrated  the pooled model's probabilities, corrected per person.
  blended       a personal model that ALSO receives the pooled model's probabilities
                as inputs. With few trials it can lean on what 406 people taught it;
                with many it can overrule them. This is the one that should fix the
                10-40 trial region where a from-scratch personal model is worse than
                useless.

Validity check
--------------
Train-origin and validation-origin subjects are reported separately. They run through
identical code and differ only in which split they came from, so if the two disagree
the leave-one-out construction is at fault and neither number should be believed.

Usage (from repo root):
    python scripts/preprocessing/leave_one_out_calibration.py --task severe
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings(
    "ignore", message="y_pred contains classes not in y_true", category=UserWarning
)

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    select_dataset_view,
)

CALIBRATION_SIZES = (0, 10, 20, 40, 60, 80)
TEST_BLOCK = 20
MIN_TRIALS = 100
METHODS = ("global", "personal", "recalibrated", "blended")


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def task_labels(task, labels, ratings):
    """Return (labels, keep_mask, chance) for the requested question."""

    if task == "3class":
        return labels, np.ones(labels.size, dtype=bool), 1 / 3
    if task == "severe":
        return (ratings >= 7.0).astype(np.int64), np.ones(labels.size, dtype=bool), 0.5
    if task == "lowhigh":
        keep = labels != 1  # drop Moderate, the ambiguous middle
        return (labels == 2).astype(np.int64), keep, 0.5
    raise ValueError(f"unknown task {task!r}")


def fit_pooled(X, y, mask, seed):
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X[mask], y[mask], sample_weight=balanced_sample_weights(y[mask]))
    return model


def personal_model(seed):
    return HistGradientBoostingClassifier(
        random_state=seed, max_depth=3, max_iter=100,
        learning_rate=0.1, l2_regularization=1.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("3class", "severe", "lowhigh"), default="3class")
    ap.add_argument("--rich", default="outputs/rich_full/rich_all9_features.npz")
    ap.add_argument("--baseline", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--split-from-checkpoint",
                    default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt")
    ap.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    ap.add_argument("--test-block", type=int, default=TEST_BLOCK)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    rich = np.load(args.rich, allow_pickle=False)
    X_all = rich["features"]
    raw_labels = rich["labels"].astype(np.int64)
    ratings = rich["ratings"].astype(np.float64)
    epoch_all = rich["epoch_index"].astype(np.int64)
    dataset_id = rich["dataset_id"].astype(str)
    subject_id = rich["subject_id"].astype(str)

    base_archive = load_feature_archive(args.baseline)
    _, training_dataset_ids = select_dataset_view(base_archive, None)
    train_keys, val_keys, _test_keys, _prov = load_checkpoint_split(
        args.split_from_checkpoint, base_archive, training_dataset_ids
    )
    tr_all = keys_mask(dataset_id, subject_id, train_keys)
    va_all = keys_mask(dataset_id, subject_id, val_keys)

    y_all, keep, chance = task_labels(args.task, raw_labels, ratings)
    # lowhigh removes Moderate trials entirely, so every array must shrink together.
    X, y, epoch_index = X_all[keep], y_all[keep], epoch_all[keep]
    tr, va = tr_all[keep], va_all[keep]
    subject_keys = np.asarray(
        [f"{d}||{s}" for d, s in
         zip(dataset_id[keep].tolist(), subject_id[keep].tolist())]
    )
    eligible = tr | va

    print(f"Task: {args.task}   chance balanced accuracy {chance:.4f}")
    print(f"Trials after task filtering: {int(keep.sum())} of {keep.size}\n")

    candidates = []
    for key in sorted(set(subject_keys[eligible].tolist())):
        mask = eligible & (subject_keys == key)
        if int(mask.sum()) < args.min_trials:
            continue
        order = np.argsort(epoch_index[mask], kind="stable")
        sub_y = y[mask][order]
        if len(np.unique(sub_y[-args.test_block:])) < 2:
            continue
        pool_y = sub_y[:-args.test_block]
        if any(
            n > pool_y.size or len(np.unique(pool_y[:n])) < 2
            for n in CALIBRATION_SIZES if n > 0
        ):
            continue
        candidates.append(key)

    origins = {
        key: ("train" if (tr & (subject_keys == key)).any() else "validation")
        for key in candidates
    }
    counts = {"train": 0, "validation": 0}
    for origin in origins.values():
        counts[origin] += 1
    print(f"Deep subjects usable at every calibration size: {len(candidates)}")
    print(f"  training split:   {counts['train']}  (each needs a retrain)")
    print(f"  validation split: {counts['validation']}  (already unseen)")
    print("Test-split subjects: excluded entirely\n")
    if not candidates:
        raise SystemExit("no subject qualifies; lower --min-trials")

    results = {m: {n: [] for n in CALIBRATION_SIZES} for m in METHODS}
    by_origin = {
        o: {m: {n: [] for n in CALIBRATION_SIZES} for m in METHODS}
        for o in ("train", "validation")
    }

    shared_model = fit_pooled(X, y, tr, args.seed)
    started = time.time()

    for index, key in enumerate(candidates, 1):
        origin = origins[key]
        subject_mask = subject_keys == key
        model = (
            fit_pooled(X, y, tr & ~subject_mask, args.seed)
            if origin == "train" else shared_model
        )

        mask = eligible & subject_mask
        order = np.argsort(epoch_index[mask], kind="stable")
        sub_X, sub_y = X[mask][order], y[mask][order]
        test_X, test_y = sub_X[-args.test_block:], sub_y[-args.test_block:]
        pool_X, pool_y = sub_X[:-args.test_block], sub_y[:-args.test_block]

        test_probabilities = model.predict_proba(test_X)
        global_score = balanced_accuracy_score(test_y, test_probabilities.argmax(1))

        def record(method, n, value):
            results[method][n].append(value)
            by_origin[origin][method][n].append(value)

        for n in CALIBRATION_SIZES:
            if n == 0:
                for method in METHODS:
                    record(method, 0, global_score)
                continue
            cal_X, cal_y = pool_X[:n], pool_y[:n]
            cal_probabilities = model.predict_proba(cal_X)
            record("global", n, global_score)

            fitted = personal_model(args.seed).fit(
                cal_X, cal_y, sample_weight=balanced_sample_weights(cal_y))
            record("personal", n, balanced_accuracy_score(test_y, fitted.predict(test_X)))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                corrector = LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=args.seed
                ).fit(cal_probabilities, cal_y)
            record("recalibrated", n,
                   balanced_accuracy_score(test_y, corrector.predict(test_probabilities)))

            # The pooled model's opinion becomes extra input columns, so a personal
            # model can defer to it early and overrule it once it knows better.
            blended = personal_model(args.seed).fit(
                np.hstack([cal_X, cal_probabilities]), cal_y,
                sample_weight=balanced_sample_weights(cal_y))
            record("blended", n, balanced_accuracy_score(
                test_y, blended.predict(np.hstack([test_X, test_probabilities]))))

        if index % 10 == 0 or index == len(candidates):
            print(f"  {index}/{len(candidates)} subjects  ({time.time()-started:.0f}s)")

    def render(title, store, cohort):
        print(f"\n=== {title} (n={cohort}, chance {chance:.3f}) ===")
        print(f"  {'calibration trials':<22}"
              + "".join(f"{n:>9}" for n in CALIBRATION_SIZES))
        print("  " + "-" * 76)
        for method in METHODS:
            row = f"  {method:<22}"
            for n in CALIBRATION_SIZES:
                values = store[method][n]
                row += f"{np.mean(values):>9.4f}" if values else f"{'-':>9}"
            print(row)
        deepest = max(CALIBRATION_SIZES)
        base = np.asarray(store["global"][deepest])
        for method in METHODS[1:]:
            values = np.asarray(store[method][deepest])
            if not values.size:
                continue
            differences = values - base
            stderr = differences.std(ddof=1) / np.sqrt(differences.size)
            print(f"    {method:<18}{100*differences.mean():+.2f} points "
                  f"(standard error {100*stderr:.2f}, "
                  f"{int((differences > 0).sum())}/{differences.size} improved)")

    render("ALL deep subjects", results, len(candidates))
    for origin in ("train", "validation"):
        if counts[origin]:
            render(f"{origin}-origin only", by_origin[origin], counts[origin])

    best = max(
        (np.mean(results[m][n]), m, n)
        for m in METHODS for n in CALIBRATION_SIZES if results[m][n]
    )
    print(f"\n  Best result: {best[0]:.4f} using '{best[1]}' at {best[2]} "
          f"calibration trials (chance {chance:.3f}).")
    print(
        "  Both origin groups are measured identically and should broadly agree;\n"
        "  a gain beyond about twice its standard error is worth acting on."
    )


if __name__ == "__main__":
    main()
