"""Personalize the LSTM itself, on the three-class task.

Every personalization result so far came from gradient boosting standing in for the
real model. It showed a personal model overtaking the pooled one at 60-80 calibration
trials (+5.71 points across 43 subjects, with train-origin and validation-origin
groups agreeing). This script asks whether the same holds for the LSTM that the
project actually ships.

Four ways of using a new user's calibration trials
--------------------------------------------------
  global        the pooled LSTM, unchanged. Flat by construction.
  head-only     the LSTM body is frozen and only its final 16->3 layer is retrained
                on that person. Interesting beyond accuracy: the recurrent weights
                never move, so an FPGA could hold one fixed network and store just
                51 numbers per user.
  recalibrated  a small correction fitted to the pooled LSTM's own output
                probabilities.
  blended       a compact model over the rich features PLUS the pooled LSTM's
                probabilities, so it can defer to the network early and overrule it
                once it knows the person. This was the best design in the gradient
                boosting version.

Why each subject gets their own pooled model
--------------------------------------------
Thirty-two of the deep subjects sit in the training split, where the pooled model has
already seen them; scoring no-calibration performance on them would flatter the
baseline and rig the comparison. So the pooled LSTM is retrained per subject with that
subject removed. Validation-origin subjects were never in the training mask and share
one fit. Held-out test subjects are excluded entirely.

Fixed epoch count
-----------------
Early stopping needs a validation split, and carving one out of every leave-one-out
fit would change the training set size per subject and make the runs incomparable.
The full run on these features chose epoch 25 as best, so a fixed 30 epochs is used
throughout. It is the same for every subject, which is what keeps the comparison fair.

Expect 30-60 minutes: one LSTM fit per training-origin subject.

Usage (from repo root):
    python scripts/preprocessing/lstm_personalization.py
"""

import argparse
import copy
import os
import sys
import time
import warnings

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings(
    "ignore", message="y_pred contains classes not in y_true", category=UserWarning
)

from rich_feature_extraction import apply_uint8_bounds, fit_uint8_bounds
from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    PainLSTM,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    model_feature_view,
    select_dataset_view,
    training_trial_weights,
)

CALIBRATION_SIZES = (0, 10, 20, 40, 60, 80)
TEST_BLOCK = 20
MIN_TRIALS = 100
POOLED_EPOCHS = 30
METHODS = ("global", "head-only", "recalibrated", "blended")


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def train_pooled_lstm(features, labels, dataset_id, subject_id, mask, args):
    """Fit the pooled LSTM on the given training mask, with train-only scaling."""

    torch.manual_seed(args.model_seed)
    low, high = fit_uint8_bounds(features[mask].reshape(-1, features.shape[2]))
    to_input = lambda v: apply_uint8_bounds(v, low, high).astype(np.float32) / 255.0

    X = torch.from_numpy(to_input(features[mask]))
    y = torch.from_numpy(labels[mask])
    weights_np, *_ = training_trial_weights(
        dataset_id[mask], subject_id[mask], labels[mask],
        class_balance_strength=args.class_balance_strength,
    )
    weights = torch.from_numpy(weights_np)

    model = PainLSTM(
        input_size=features.shape[2], hidden_size=args.hidden,
        n_classes=len(CLASS_NAMES), task_mode="categorical",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    generator = torch.Generator().manual_seed(args.loader_seed)

    model.train()
    for _ in range(POOLED_EPOCHS):
        order = torch.randperm(X.shape[0], generator=generator)
        for start in range(0, X.shape[0], args.batch_size):
            index = order[start:start + args.batch_size]
            opt.zero_grad()
            losses = loss_fn(model(X[index]), y[index])
            (torch.mean(losses * weights[index])).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model, to_input


def probabilities(model, X, batch_size=256):
    out = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            out.append(torch.softmax(model(X[start:start + batch_size]), dim=1))
    return torch.cat(out).numpy()


def finetune_head(model, cal_X, cal_y, cal_weights, seed, steps=200, lr=1e-2):
    """Retrain only the final layer. The recurrent weights never move."""

    torch.manual_seed(seed)
    personal = copy.deepcopy(model)
    for parameter in personal.lstm.parameters():
        parameter.requires_grad_(False)
    opt = torch.optim.AdamW(personal.classifier.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    personal.train()
    for _ in range(steps):
        opt.zero_grad()
        losses = loss_fn(personal(cal_X), cal_y)
        (torch.mean(losses * cal_weights)).backward()
        opt.step()
    personal.eval()
    return personal


def compact_model(seed):
    return HistGradientBoostingClassifier(
        random_state=seed, max_depth=3, max_iter=100,
        learning_rate=0.1, l2_regularization=1.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--rich-features", default="outputs/rich_full/rich_all9_features.npz")
    ap.add_argument("--split-from-checkpoint",
                    default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt")
    ap.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    ap.add_argument("--test-block", type=int, default=TEST_BLOCK)
    ap.add_argument(
        "--max-calibration",
        type=int,
        default=max(CALIBRATION_SIZES),
        help=(
            "Largest calibration size to test. Six of the nine datasets recorded only "
            "about 30 trials per person and can never support deep calibration, but "
            "lowering this admits ds005293's 76 subjects at 40 trials, which is where "
            "the gain is already most of its final size."
        ),
    )
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--class-balance-strength", type=float, default=0.5)
    ap.add_argument("--model-seed", type=int, default=20260726)
    ap.add_argument("--loader-seed", type=int, default=20260727)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    sizes = tuple(n for n in CALIBRATION_SIZES if n <= args.max_calibration)
    if len(sizes) < 2:
        ap.error('--max-calibration is too small to form a curve')

    archive = load_feature_archive(args.data)
    _, training_dataset_ids = select_dataset_view(archive, None)
    features, _order = model_feature_view(
        archive, "rich", rich_features_path=args.rich_features
    )
    labels = archive["labels"].astype(np.int64)
    epoch_index = archive["epoch_index"].astype(np.int64)
    dataset_id = archive["dataset_id"].astype(str)
    subject_id = archive["subject_id"].astype(str)
    flat_features = features.reshape(features.shape[0], -1)

    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        args.split_from_checkpoint, archive, training_dataset_ids
    )
    tr = keys_mask(dataset_id, subject_id, train_keys)
    va = keys_mask(dataset_id, subject_id, val_keys)
    subject_keys = np.asarray(
        [f"{d}||{s}" for d, s in zip(dataset_id.tolist(), subject_id.tolist())]
    )
    eligible = tr | va

    candidates = []
    for key in sorted(set(subject_keys[eligible].tolist())):
        mask = eligible & (subject_keys == key)
        if int(mask.sum()) < args.min_trials:
            continue
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
        candidates.append(key)

    origins = {
        key: ("train" if (tr & (subject_keys == key)).any() else "validation")
        for key in candidates
    }
    n_train_origin = sum(1 for o in origins.values() if o == "train")
    print(f"Task: 3-class Low/Moderate/High   chance 0.3333")
    print(f"Deep subjects usable at every calibration size: {len(candidates)}")
    print(f"  training split:   {n_train_origin}  (each needs an LSTM retrain)")
    print(f"  validation split: {len(candidates)-n_train_origin}  (already unseen)")
    print(f"Pooled LSTM: {POOLED_EPOCHS} epochs, hidden {args.hidden}, "
          f"input {features.shape[1]}x{features.shape[2]}\n")
    if not candidates:
        raise SystemExit("no subject qualifies; lower --min-trials")

    results = {m: {n: [] for n in sizes} for m in METHODS}
    by_origin = {
        o: {m: {n: [] for n in sizes} for m in METHODS}
        for o in ("train", "validation")
    }

    started = time.time()
    shared = train_pooled_lstm(features, labels, dataset_id, subject_id, tr, args)
    print(f"  shared pooled LSTM trained ({time.time()-started:.0f}s)")

    for index, key in enumerate(candidates, 1):
        origin = origins[key]
        subject_mask = subject_keys == key
        if origin == "train":
            model, to_input = train_pooled_lstm(
                features, labels, dataset_id, subject_id, tr & ~subject_mask, args
            )
        else:
            model, to_input = shared

        mask = eligible & subject_mask
        order = np.argsort(epoch_index[mask], kind="stable")
        sub_features = features[mask][order]
        sub_flat = flat_features[mask][order]
        sub_y = labels[mask][order]

        X_sub = torch.from_numpy(to_input(sub_features))
        test_X, test_y = X_sub[-args.test_block:], sub_y[-args.test_block:]
        test_flat = sub_flat[-args.test_block:]
        pool_X, pool_y = X_sub[:-args.test_block], sub_y[:-args.test_block]
        pool_flat = sub_flat[:-args.test_block]

        test_probabilities = probabilities(model, test_X)
        global_score = balanced_accuracy_score(test_y, test_probabilities.argmax(1))

        def record(method, n, value):
            results[method][n].append(value)
            by_origin[origin][method][n].append(value)

        for n in sizes:
            if n == 0:
                for method in METHODS:
                    record(method, 0, global_score)
                continue
            cal_X, cal_y = pool_X[:n], pool_y[:n]
            cal_flat = pool_flat[:n]
            cal_probabilities = probabilities(model, cal_X)
            cal_weights = torch.from_numpy(
                balanced_sample_weights(cal_y).astype(np.float32)
            )
            record("global", n, global_score)

            personal = finetune_head(
                model, cal_X, torch.from_numpy(cal_y), cal_weights, args.seed
            )
            record("head-only", n, balanced_accuracy_score(
                test_y, probabilities(personal, test_X).argmax(1)))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                corrector = LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=args.seed
                ).fit(cal_probabilities, cal_y)
            record("recalibrated", n, balanced_accuracy_score(
                test_y, corrector.predict(test_probabilities)))

            blended = compact_model(args.seed).fit(
                np.hstack([cal_flat, cal_probabilities]), cal_y,
                sample_weight=balanced_sample_weights(cal_y))
            record("blended", n, balanced_accuracy_score(
                test_y, blended.predict(np.hstack([test_flat, test_probabilities]))))

        if index % 5 == 0 or index == len(candidates):
            print(f"  {index}/{len(candidates)} subjects  ({time.time()-started:.0f}s)")

    def render(title, store, cohort):
        print(f"\n=== {title} (n={cohort}, chance 0.333) ===")
        print(f"  {'calibration trials':<22}"
              + "".join(f"{n:>9}" for n in sizes))
        print("  " + "-" * 76)
        for method in METHODS:
            row = f"  {method:<22}"
            for n in sizes:
                values = store[method][n]
                row += f"{np.mean(values):>9.4f}" if values else f"{'-':>9}"
            print(row)
        deepest = max(sizes)
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
        cohort = sum(1 for o in origins.values() if o == origin)
        if cohort:
            render(f"{origin}-origin only", by_origin[origin], cohort)

    print("\n  Gradient boosting reference on the same cohort and task:")
    print("    global 0.5086   personal +5.71   blended untested there")
    print(
        "\n  head-only is the result to watch for hardware: it leaves the recurrent\n"
        "  weights untouched, so personalizing a user costs 51 stored numbers rather\n"
        "  than a second network."
    )


if __name__ == "__main__":
    main()
