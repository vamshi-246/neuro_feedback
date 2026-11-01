"""Pivot 2: give each user a private head over the same frozen 16 features.

Experiment 12 in the report found the largest single effect in the project -- roughly
+10 balanced points from personalizing to one user -- and found that freezing the whole
network and retraining only the final 16->3 layer worked best. That is the version worth
building on, because 16->3 is 51 numbers: a device stores 51 numbers per user, not a
model per user.

This runs it on the standing 16-feature configuration rather than the older LSTM, and
stacks it on top of pivot 1, so the headline row is the shippable combination:
severe-pain detection with a personalized head.

The body is frozen, four heads compete
--------------------------------------
  global        the tuned gradient boosting model on the 16 features, untouched. This is
                the 49% model and the number every arm has to beat.
  bias-only     3 numbers per user: only the output offsets move. If this captures most
                of the gain, the device stores 3 numbers instead of 51.
  head-only     51 numbers per user: a fresh 16->3 linear layer, warm-started at the
                shared head and pulled back toward it.
  blended       the 16 features plus the global model's own three probabilities, so a
                user's head can defer to the shared model early and overrule it later.

Every personalized head is fitted with an L2 penalty on the distance from the shared
head, not on the distance from zero. With ten calibration trials the penalty dominates
and the head barely moves; with eighty the data wins. That is the whole mechanism by
which a small calibration set helps instead of overfitting.

Why the shared model is refitted 29 times
-----------------------------------------
Twenty-nine of the deep subjects sit in the training split, where the shared model has
already seen them. Scoring "no calibration" on those subjects would flatter the baseline
and manufacture a smaller apparent gain. So for each of them the shared model is refitted
with that subject removed. The twelve deep subjects of validation origin were never in
training and share one fit.

The shrinkage strength is chosen on the training-origin group and then applied unchanged
to the validation-origin group. Both are printed. The validation-origin column is the
clean one; the training-origin column is where the choice was made.

Calibration is the first N trials in recording order, and every arm is scored on the
same trials -- everything from trial 81 onward -- so the N columns are comparable and no
arm is ever scored on a trial it calibrated on.

Expect 10-25 minutes.

Usage (from repo root):
    python scripts/preprocessing/personalized_head.py
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# One user's scored block can miss a class the model still predicts. That is expected
# here and the score is correct; the warning would otherwise fire thousands of times.
warnings.filterwarnings(
    "ignore", message="y_pred contains classes not in y_true", category=UserWarning
)

from winning_config import (  # noqa: E402 -- path setup must precede this import
    SEED,
    SELECTED_16,
    balanced_sample_weights,
    fit_tuned,
    load_pool,
    load_split,
    subject_groups,
)

CALIBRATION_SIZES = (0, 10, 20, 40, 60, 80)
MIN_TRIALS = 120           # 80 calibration + at least 40 scored
TEST_FROM = 80             # the scored block starts here, identical for every arm
SHRINKAGE = (0.3, 1.0, 3.0, 10.0, 30.0)
ARMS = ("bias-only", "head-only", "blended")


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def head_loss(flat, X, y_onehot, weights, W0, b0, lam, bias_only):
    """Weighted cross-entropy plus an L2 pull toward the shared head."""

    classes, features = W0.shape
    if bias_only:
        W, b = W0, flat
    else:
        W = flat[: classes * features].reshape(classes, features)
        b = flat[classes * features:]
    probabilities = softmax(X @ W.T + b)
    log_probabilities = np.log(np.clip(probabilities, 1e-12, None))
    loss = -float((weights[:, None] * y_onehot * log_probabilities).sum())
    residual = probabilities - y_onehot
    grad_b = (weights[:, None] * residual).sum(axis=0)
    loss += 0.5 * lam * float(((b - b0) ** 2).sum())
    grad_b += lam * (b - b0)
    if bias_only:
        return loss, grad_b
    grad_W = (weights[:, None] * residual).T @ X
    loss += 0.5 * lam * float(((W - W0) ** 2).sum())
    grad_W += lam * (W - W0)
    return loss, np.concatenate([grad_W.ravel(), grad_b])


def fit_head(X, y, W0, b0, lam, bias_only, n_classes):
    """One user's head. Warm-started at the shared head, so N=0 returns it unchanged."""

    if X.shape[0] == 0:
        return W0.copy(), b0.copy()
    onehot = np.zeros((y.size, n_classes), dtype=np.float64)
    onehot[np.arange(y.size), y] = 1.0
    weights = balanced_sample_weights(y) if np.unique(y).size > 1 \
        else np.ones(y.size, dtype=np.float64)
    start = b0.copy() if bias_only else np.concatenate([W0.ravel(), b0])
    result = minimize(head_loss, start, jac=True, method="L-BFGS-B",
                      args=(X, onehot, weights, W0, b0, lam, bias_only),
                      options={"maxiter": 500})
    if bias_only:
        return W0.copy(), result.x
    classes, features = W0.shape
    return (result.x[: classes * features].reshape(classes, features),
            result.x[classes * features:])


def shared_head(X, y, n_classes, seed):
    """The 16->K layer everyone starts from, fitted on the training subjects."""

    model = LogisticRegression(max_iter=5000, class_weight="balanced",
                              random_state=seed, C=1.0)
    model.fit(X, y)
    W = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)
    if model.coef_.shape[0] == 1:  # sklearn returns one row for two classes
        W[1], b[1] = model.coef_[0], model.intercept_[0]
    else:
        W, b = model.coef_.astype(np.float64), model.intercept_.astype(np.float64)
    return W, b


def per_subject_score(y_true, y_pred):
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


class Body:
    """Everything that is shared and frozen: standardization, the 49% model, the head."""

    def __init__(self, X, y, n_classes, seed):
        self.mean, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mean) / self.sd
        self.gbdt = fit_tuned(X, y, seed=seed)
        self.W0, self.b0 = shared_head(Z, y, n_classes, seed)
        self.n_classes = n_classes

    def standardize(self, X):
        return (X - self.mean) / self.sd

    def probabilities(self, X):
        """The global model's class probabilities, widened to every class if a fold
        happened to miss one."""

        raw = self.gbdt.predict_proba(X)
        full = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)
        for column, label in enumerate(self.gbdt.classes_):
            full[:, int(label)] = raw[:, column]
        return full

    def inputs(self, X, arm):
        Z = self.standardize(X)
        if arm == "blended":
            probabilities = np.clip(self.probabilities(X), 1e-6, 1.0)
            return np.hstack([Z, np.log(probabilities)])
        return Z


def evaluate_subject(body, X_subject, y_subject, lam_by_arm, n_classes, arms=ARMS):
    """One subject, the given arms, every calibration size, scored on the same block."""

    test = slice(TEST_FROM, None)
    y_test = y_subject[test]
    rows = {("global", 0): per_subject_score(y_test,
                                            body.gbdt.predict(X_subject[test]))}

    for arm in arms:
        features = body.inputs(X_subject, arm)
        W0 = body.W0
        if arm == "blended":
            W0 = np.hstack([body.W0, np.zeros((n_classes, n_classes))])
            W0[np.arange(n_classes), body.W0.shape[1] + np.arange(n_classes)] = 1.0
        for size in CALIBRATION_SIZES:
            W, b = fit_head(features[:size], y_subject[:size], W0, body.b0,
                            lam_by_arm[arm], arm == "bias-only", n_classes)
            prediction = (features[test] @ W.T + b).argmax(1)
            rows[(arm, size)] = per_subject_score(y_test, prediction)
    return rows


def deep_subjects(groups, mask, min_trials=MIN_TRIALS):
    counts = {}
    for name in groups[mask].tolist():
        counts[name] = counts.get(name, 0) + 1
    return sorted(name for name, count in counts.items() if count >= min_trials)


def run_task(task_name, X, y, n_classes, tr, va, groups, order, seed, results):
    print(f"\n{'=' * 90}\n{task_name}")
    print(f"{'-' * 90}")

    train_deep = deep_subjects(groups, tr)
    val_deep = deep_subjects(groups, va)
    print(f"  deep subjects (>= {MIN_TRIALS} trials): "
          f"{len(train_deep)} of training origin, {len(val_deep)} of validation origin")
    print(f"  calibration = first N trials in recording order; "
          f"scored on trial {TEST_FROM + 1} onward")

    def subject_rows(name):
        idx = np.where(groups == name)[0]
        return idx[np.argsort(order[idx])]

    # ---- shrinkage chosen on the training-origin group ----
    started = time.time()
    print(f"\n  Choosing the shrinkage strength on the {len(train_deep)} "
          f"training-origin subjects")
    print(f"    (their shared model is refitted with that subject removed)")
    bodies = {}
    for name in train_deep:
        keep = tr & (groups != name)
        bodies[name] = Body(X[keep], y[keep], n_classes, seed)
    print(f"    {len(train_deep)} leave-one-subject-out fits in "
          f"{time.time() - started:.0f}s")

    chosen = {}
    for arm in ARMS:
        best_lam, best_score = None, -1.0
        for lam in SHRINKAGE:
            scores = []
            for name in train_deep:
                rows = subject_rows(name)
                sub = evaluate_subject(bodies[name], X[rows], y[rows],
                                       {arm: lam}, n_classes, arms=(arm,))
                scores.append(np.nanmean([sub[(arm, size)]
                                          for size in CALIBRATION_SIZES if size > 0]))
            mean = float(np.nanmean(scores))
            if mean > best_score:
                best_lam, best_score = lam, mean
        chosen[arm] = best_lam
        print(f"    {arm:<12} shrinkage {best_lam:>5.1f}   "
              f"training-origin mean {best_score:.4f}")

    # ---- report both groups with the choice frozen ----
    shared_body = Body(X[tr], y[tr], n_classes, seed)
    tables = {}
    for label, names, body_for in (
        ("training origin (where the shrinkage was chosen)", train_deep,
         lambda name: bodies[name]),
        ("validation origin (never trained on)", val_deep, lambda name: shared_body),
    ):
        collected = {}
        for name in names:
            rows = subject_rows(name)
            collected[name] = evaluate_subject(body_for(name), X[rows], y[rows],
                                               chosen, n_classes)
        tables[label] = collected

    parameters = {"global": 0, "bias-only": n_classes,
                  "head-only": n_classes * 17,
                  "blended": n_classes * (17 + n_classes)}
    task_result = {}
    for label, collected in tables.items():
        # A subject whose scored block holds only one class has no balanced accuracy;
        # on the severe task a few users never report 7 or above.
        names = sorted(name for name in collected
                       if not np.isnan(collected[name][("global", 0)]))
        dropped = len(collected) - len(names)
        print(f"\n  {label}   n = {len(names)} subjects"
              + (f"   ({dropped} dropped: only one class in the scored block)"
                 if dropped else ""))
        print(f"    {'arm':<12}{'per user':>10}" +
              "".join(f"{f'N={size}':>9}" for size in CALIBRATION_SIZES))
        print("    " + "-" * (22 + 9 * len(CALIBRATION_SIZES)))
        baseline = float(np.nanmean([collected[n][("global", 0)] for n in names]))
        line = f"    {'global':<12}{parameters['global']:>10}"
        line += "".join(f"{baseline:>9.4f}" for _ in CALIBRATION_SIZES)
        print(line + "   flat by construction")
        for arm in ARMS:
            means = [float(np.nanmean([collected[n][(arm, size)] for n in names]))
                     for size in CALIBRATION_SIZES]
            line = f"    {arm:<12}{parameters[arm]:>10}"
            line += "".join(f"{value:>9.4f}" for value in means)
            print(line)
            task_result[f"{label}|{arm}"] = means
        best_arm = max(ARMS, key=lambda a: task_result[f"{label}|{a}"][-1])
        best = task_result[f"{label}|{best_arm}"]
        wins = sum(
            1 for n in names
            if collected[n][(best_arm, CALIBRATION_SIZES[-1])]
            > collected[n][("global", 0)]
        )
        crossing = next((size for size, value in zip(CALIBRATION_SIZES, best)
                         if size > 0 and value > baseline), None)
        print(f"    global {baseline:.4f}   best arm {best_arm} at "
              f"N={CALIBRATION_SIZES[-1]}: {best[-1]:.4f}   "
              f"{100 * (best[-1] - baseline):+.2f} points")
        print(f"    beats the shared model for {wins} of {len(names)} users; "
              f"pays for itself from "
              + (f"{crossing} calibration trials" if crossing else "no size tested"))
        task_result[f"{label}|global"] = baseline
        task_result[f"{label}|subjects"] = len(names)

    task_result["shrinkage"] = chosen
    results[task_name] = task_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="outputs/pivots/personalized_head.json")
    args = ap.parse_args()

    pool = load_pool()
    index = {name: i for i, name in enumerate(pool["names"])}
    missing = [name for name in SELECTED_16 if name not in index]
    if missing:
        raise SystemExit(f"the pool is missing selected columns: {missing}")
    X = pool["X"][:, [index[name] for name in SELECTED_16]]

    datasets, subjects = pool["dataset_id"], pool["subject_id"]
    tr, va = load_split(datasets, subjects)
    groups = subject_groups(datasets, subjects)

    print(f"Frozen body: the {len(SELECTED_16)} selected columns. "
          f"Shape {X.shape}   train {int(tr.sum())}   validation {int(va.sum())}")
    print("A personalized head stores only its own numbers; the body never moves.")

    results = {}
    run_task("Three classes: Low / Moderate / High", X, pool["labels"], 3,
             tr, va, groups, pool["epoch_index"], args.seed, results)
    run_task("Severe or not (rating >= 7)  -- pivot 1 and pivot 2 together",
             X, (pool["ratings"] >= 7.0).astype(np.int64), 2,
             tr, va, groups, pool["epoch_index"], args.seed, results)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\n  Written to {args.out}")
    print(
        "\n  Read the validation-origin block as the result and the training-origin\n"
        "  block as the tuning log. Both use held-out trials from users the shared\n"
        "  model never saw, but the shrinkage strength was picked on the first group."
    )


if __name__ == "__main__":
    main()
