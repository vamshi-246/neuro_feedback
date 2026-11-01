"""Export the deployable model as plain numbers, for the FPGA build.

What actually ships is small: sixteen scalar features per trial, one 3x16 matrix, one
3-vector, and an argmax. This script writes those numbers out, together with the
definition of every feature and the 8-bit bounds the RTL input path needs, so the
hardware team never has to read a Python object or re-run a fit.

Two arithmetic forms are exported, and they are numerically identical:

  standardized   z = (x - mean) / sd ;  logits = W  @ z     + b
  folded         logits = W_eff @ x + b_eff, where W_eff = W / sd and
                 b_eff = b - W_eff @ mean

The folded form is the one to build. It removes the subtract-and-divide stage entirely,
so the device does 3x16 multiply-accumulates, three adds and an argmax -- nothing else.
Softmax is not needed in hardware: it is monotonic, so argmax over the logits picks the
same class.

Two feature lists are exported per task
---------------------------------------
  fixed     the list the personalization result was measured on. Two of its sixteen
            columns are MVAR parametric power, which need a least-squares AR fit per
            trial -- by far the most expensive primitive on the list.
  no-MVAR   the same length, reselected on training rows only from a pool with MVAR
            excluded. It scores slightly HIGHER (0.4529 vs 0.4503 dataset-macro), so
            the AR solver buys nothing and should come out.

Every fit here uses training subjects only. Validation is scored, never fitted on.

Usage (from repo root):
    python scripts/preprocessing/export_deployment.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import winning_config as wc  # noqa: E402
from rich_feature_extraction import (  # noqa: E402
    apply_uint8_bounds,
    fit_uint8_bounds,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

OUT_DIR = "outputs/deploy"
MIN_TRIALS = 120        # matches personalized_head.py: 80 calibration + 40 scored
CALIBRATION_BLOCK = 80


def fit_head(X_train, y_train, n_classes, seed=wc.SEED):
    """The shared 16->K layer, fitted exactly as personalized_head.py fits it."""

    mean, sd = X_train.mean(0), X_train.std(0) + 1e-9
    model = LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=seed, C=1.0)
    model.fit((X_train - mean) / sd, y_train)
    W = np.zeros((n_classes, X_train.shape[1]), dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)
    if model.coef_.shape[0] == 1:          # sklearn gives one row for two classes
        W[1], b[1] = model.coef_[0], model.intercept_[0]
    else:
        W, b = model.coef_.astype(np.float64), model.intercept_.astype(np.float64)
    return mean, sd, W, b


def fold_standardization(mean, sd, W, b):
    """Push (x - mean)/sd into the weights, so the device sees one matrix and one bias."""

    W_eff = W / sd
    return W_eff, b - W_eff @ mean


def per_user_baseline(X, y, pred, groups, order, validation):
    """Per-subject balanced accuracy on the deep subjects' scored block.

    This is the 0.5674 column of the personalization table -- the head with no
    calibration -- reproduced here so the exported weights are checked against a
    published number rather than trusted.
    """

    scores = []
    for name in sorted(set(groups[validation].tolist())):
        rows = np.where(groups == name)[0]
        if rows.size < MIN_TRIALS:
            continue
        rows = rows[np.argsort(order[rows])][CALIBRATION_BLOCK:]
        if rows.size == 0 or np.unique(y[rows]).size < 2:
            continue
        scores.append(balanced_accuracy_score(y[rows], pred[rows]))
    return (float(np.mean(scores)) if scores else float("nan")), len(scores)


def describe_columns(names, X_train):
    """Feature definitions plus the 8-bit bounds, fitted on training rows only."""

    import hardcoded_16

    low, high = fit_uint8_bounds(X_train)
    entries = []
    for position, name in enumerate(names):
        record = hardcoded_16.describe(name)
        record.update({
            "index": position,
            "name": name,
            "family": wc.family_of(name),
            "quantizer_low": float(low[position]),
            "quantizer_high": float(high[position]),
        })
        entries.append(record)
    return entries


def build(task, X, y, names, tr, va, datasets, groups, order, n_classes):
    """Fit, verify and package one feature list."""

    mean, sd, W, b = fit_head(X[tr], y[tr], n_classes)
    W_eff, b_eff = fold_standardization(mean, sd, W, b)

    standardized = ((X - mean) / sd) @ W.T + b
    folded = X @ W_eff.T + b_eff
    drift = float(np.abs(standardized - folded).max())

    pred = np.argmax(folded, axis=1)
    macro, _ = wc.dataset_macro(y[va], pred[va], datasets[va])
    per_user, n_users = per_user_baseline(X, y, pred, groups, order, va)

    low, high = fit_uint8_bounds(X[tr])
    q = apply_uint8_bounds(X, low, high).astype(np.float64)
    q_pred = np.argmax(((q - q[tr].mean(0)) / (q[tr].std(0) + 1e-9)) @ W.T + b, axis=1)
    q_macro, _ = wc.dataset_macro(y[va], q_pred[va], datasets[va])

    print(f"    dataset-macro (validation)      {macro:.4f}")
    print(f"    per-user baseline, {n_users} deep users  {per_user:.4f}")
    print(f"    same after 8-bit quantization   {q_macro:.4f}")
    print(f"    folded vs standardized, max |diff|  {drift:.2e}")

    return {
        "task": task,
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "scores": {
            "dataset_macro_validation": macro,
            "per_user_baseline_deep_users": per_user,
            "deep_users": n_users,
            "dataset_macro_8bit": q_macro,
        },
        "standardized_form": {
            "mean": mean.tolist(), "sd": sd.tolist(),
            "W": W.tolist(), "b": b.tolist(),
        },
        "folded_form": {"W_eff": W_eff.tolist(), "b_eff": b_eff.tolist()},
        "max_abs_difference_between_forms": drift,
        "columns": describe_columns(names, X[tr]),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pool = wc.load_pool()
    lookup = {name: i for i, name in enumerate(pool["names"])}
    y_raw, datasets = pool["labels"], pool["dataset_id"]
    tr, va = wc.load_split(datasets, pool["subject_id"])
    groups = wc.subject_groups(datasets, pool["subject_id"])
    order = pool["epoch_index"]

    fixed = list(wc.SELECTED_16)
    allowed = [i for i, n in enumerate(pool["names"]) if wc.family_of(n) != "MVAR"]
    ranked = wc.rank_columns(pool["X"][tr][:, allowed], y_raw[tr])
    no_mvar = [pool["names"][allowed[i]] for i in ranked[:len(fixed)]]

    bundle = {}
    for task, y, n_classes in (("three_class", y_raw, 3),
                               ("severe", (y_raw >= 2).astype(np.int64), 2)):
        for tag, columns in (("fixed", fixed), ("no_mvar", no_mvar)):
            print(f"\n  {task} / {tag} ({len(columns)} columns)")
            X = pool["X"][:, [lookup[n] for n in columns]]
            bundle[f"{task}__{tag}"] = build(task, X, y, columns, tr, va,
                                             datasets, groups, order, n_classes)

    path = os.path.join(OUT_DIR, "deployment_model.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)
    print(f"\n  wrote {path}")

    overlap = len(set(fixed) & set(no_mvar))
    print(f"  the two 16-column lists share {overlap} columns")


if __name__ == "__main__":
    main()
