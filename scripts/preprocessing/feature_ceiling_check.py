"""Read-only diagnostic: is the accuracy ceiling caused by the model or the features?

The LSTM stops improving while its TRAINING loss is still near ln(3) = 1.0986,
the value random guessing produces.  That is the signature of underfitting, not
overfitting, so the usual responses (more data, augmentation, regularization)
would all target the wrong problem.

This script answers one question with no changes to the pipeline: how much
class information do the saved band-power features actually contain?

It fits gradient-boosted trees on exactly the same trials, the same locked
subject split, and the same feature views the trainer uses, in two settings:

  regularized  -- a fair competitor to the LSTM
  memorize     -- deliberately unregularized, allowed to overfit hard

Reading the result:

  memorize train accuracy stays near chance
      -> the features do not separate Low/Moderate/High even in-sample.
         Richer features are required; no model choice can rescue this.
  memorize train accuracy is high but validation stays near chance
      -> the features carry subject-specific detail that does not transfer to
         new people.  Personalization or subject-invariant features are needed.

Nothing is written to disk and the held-out test split is never touched.

Usage (from repo root):
    python scripts/preprocessing/feature_ceiling_check.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    model_feature_view,
    select_dataset_view,
)


def flatten_view(archive, feature_mode):
    """Flatten (trials, steps, features) into the (trials, steps*features) table trees need."""

    power, feature_order = model_feature_view(archive, feature_mode)
    flat = power.reshape(power.shape[0], -1)
    # log1p matches the trainer's transform.  Trees are invariant to monotone
    # rescaling, so this only keeps the two pipelines visually comparable.
    return np.log1p(flat), power.shape, feature_order


def balanced_sample_weights(labels):
    """Give every class equal total mass so balanced accuracy is the fair target."""

    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("training split is missing a class")
    per_class = labels.size / (len(CLASS_NAMES) * counts)
    return per_class[labels]


def score(name, y_true, y_pred):
    return {
        "split": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def run_model(tag, model, X_train, y_train, w_train, X_val, y_val):
    model.fit(X_train, y_train, sample_weight=w_train)
    rows = [
        score("train", y_train, model.predict(X_train)),
        score("validation", y_val, model.predict(X_val)),
    ]
    print(f"\n  {tag}")
    for row in rows:
        print(
            f"    {row['split']:<11} accuracy={row['accuracy']:.4f}  "
            f"balanced={row['balanced_accuracy']:.4f}"
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/all9_full/all9_features.npz")
    ap.add_argument(
        "--split-from-checkpoint",
        default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt",
        help="Reuse the exact locked subject split the LSTM runs used.",
    )
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    archive = load_feature_archive(args.data)
    _, training_dataset_ids = select_dataset_view(archive, None)
    train_keys, val_keys, _test_keys, provenance = load_checkpoint_split(
        args.split_from_checkpoint, archive, training_dataset_ids
    )

    dataset_id = archive["dataset_id"].astype(str)
    subject_id = archive["subject_id"].astype(str)
    y = archive["labels"].astype(np.int64)
    train_mask = keys_mask(dataset_id, subject_id, train_keys)
    val_mask = keys_mask(dataset_id, subject_id, val_keys)

    print(f"Split source: {provenance['mode']} ({os.path.basename(args.split_from_checkpoint)})")
    print(
        f"Train trials: {int(train_mask.sum())}   "
        f"Validation trials: {int(val_mask.sum())}   "
        f"(test split untouched)"
    )
    print(f"Chance balanced accuracy for {len(CLASS_NAMES)} classes: {1/len(CLASS_NAMES):.4f}")

    y_train, y_val = y[train_mask], y[val_mask]
    w_train = balanced_sample_weights(y_train)

    floor = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((y_train.size, 1)), y_train
    )
    floor_pred = floor.predict(np.zeros((y_val.size, 1)))
    print(
        f"\nDo-nothing floor (always predicts the biggest class): "
        f"accuracy={accuracy_score(y_val, floor_pred):.4f}  "
        f"balanced={balanced_accuracy_score(y_val, floor_pred):.4f}"
    )

    for feature_mode in ("channel-average", "per-channel"):
        X, shape, feature_order = flatten_view(archive, feature_mode)
        X_train, X_val = X[train_mask], X[val_mask]
        print(
            f"\n=== {feature_mode}: {shape[1]} steps x {len(feature_order)} features "
            f"= {X.shape[1]} columns per trial ==="
        )

        run_model(
            "gradient boosting (regularized -- fair competitor to the LSTM)",
            HistGradientBoostingClassifier(random_state=args.seed),
            X_train, y_train, w_train, X_val, y_val,
        )
        run_model(
            "gradient boosting (memorize -- deliberately allowed to overfit)",
            HistGradientBoostingClassifier(
                random_state=args.seed,
                max_iter=500,
                max_leaf_nodes=255,
                min_samples_leaf=1,
                l2_regularization=0.0,
                learning_rate=0.2,
                early_stopping=False,
            ),
            X_train, y_train, w_train, X_val, y_val,
        )

    print(
        "\nInterpretation:\n"
        "  If 'memorize' train balanced accuracy stays near chance, the features\n"
        "  themselves do not separate the classes -- no model can fix that.\n"
        "  If 'memorize' train is high but validation is near chance, the features\n"
        "  encode subject-specific detail that does not transfer to new people."
    )


if __name__ == "__main__":
    main()
