"""Turn the trained score model's numbers into classes, so it can be compared fairly.

train_lstm_score.py reports correlation and mean absolute error, never accuracy,
because it deliberately stores no class thresholds. That leaves an open question:
the classification LSTM reached 0.4750 balanced accuracy, and until the score model
produces Low/Moderate/High there is nothing to compare it against.

This script loads the already-trained score model, predicts on the same locked
split, and applies each candidate mapping so the two approaches can be placed side
by side on one metric.

On fitting the mapping
----------------------
Cut points are fitted on the model's TRAINING predictions. Normally that would be
too optimistic, because a model flatters itself on rows it has seen. Here it is
acceptable and the run prints the evidence: the score model's training r and
validation r came out at 0.404 and 0.403, so it is barely fitting the training set
more closely than new data. The script re-reports both numbers, and if that gap
grows in a future run the fitted cut points must be treated as optimistic.

No validation rating or label is ever used to choose a mapping.

Usage (from repo root):
    python scripts/preprocessing/score_to_class_check.py
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from feature_extraction import bin_rating, quantize_to_uint8
from rich_feature_extraction import apply_uint8_bounds
from train_lstm_score import PainScoreLSTM, predict_scores
from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    model_feature_view,
    select_dataset_view,
)

CLASSIFIER_BASELINE = 0.4750  # rich-feature classification LSTM, same locked split


def bin_with_cuts(values, cuts):
    low, high = cuts
    return np.where(values < low, 0, np.where(values < high, 1, 2)).astype(np.int64)


def official_bins(values):
    return np.asarray(
        [bin_rating(float(v)) for v in np.clip(values, 0.0, 10.0)], dtype=np.int64
    )


def distribution_matched_cuts(scores, labels):
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    proportions = counts / counts.sum()
    low = float(np.quantile(scores, proportions[0]))
    high = float(np.quantile(scores, proportions[0] + proportions[1]))
    return (low, high) if low < high else (4.0, 7.0)


def optimized_cuts(scores, labels):
    grid = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 49)))
    best, best_score = (4.0, 7.0), -1.0
    for i, low in enumerate(grid):
        for high in grid[i + 1:]:
            value = balanced_accuracy_score(labels, bin_with_cuts(scores, (low, high)))
            if value > best_score:
                best, best_score = (float(low), float(high)), value
    return best


def report(tag, y_true, y_pred):
    bal = balanced_accuracy_score(y_true, y_pred)
    print(f"  {tag:<46}{bal:>12.4f}{accuracy_score(y_true, y_pred):>10.4f}"
          f"{100*(bal-CLASSIFIER_BASELINE):>+11.2f}")
    return bal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="outputs/controlled_comparison/all9_rich_score_seed20260726.pt")
    ap.add_argument("--data", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--rich-features", default="outputs/rich_full/rich_all9_features.npz")
    ap.add_argument("--split-from-checkpoint",
                    default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "rating_regression":
        raise SystemExit(f"{args.model} is not a score model")

    archive = load_feature_archive(args.data)
    _, training_dataset_ids = select_dataset_view(archive, None)
    model_power, _order = model_feature_view(
        archive, checkpoint["feature_mode"], rich_features_path=args.rich_features
    )
    ratings = archive["ratings"].astype(np.float64)
    labels = archive["labels"].astype(np.int64)
    dataset_id = archive["dataset_id"].astype(str)
    subject_id = archive["subject_id"].astype(str)
    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        args.split_from_checkpoint, archive, training_dataset_ids
    )
    tr = keys_mask(dataset_id, subject_id, train_keys)
    va = keys_mask(dataset_id, subject_id, val_keys)

    low, high = checkpoint["quantization_low"], checkpoint["quantization_high"]
    if checkpoint["feature_mode"] == "rich":
        to_input = lambda v: apply_uint8_bounds(v, low, high).astype(np.float32) / 255.0
    else:
        to_input = lambda v: quantize_to_uint8(v, low, high).astype(np.float32) / 255.0

    model = PainScoreLSTM(checkpoint["input_size"], hidden_size=checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    train_scores = predict_scores(model, torch.from_numpy(to_input(model_power[tr])),
                                  args.batch_size)
    val_scores = predict_scores(model, torch.from_numpy(to_input(model_power[va])),
                                args.batch_size)

    train_r = checkpoint["training_report"]["r"]
    val_r = checkpoint["validation_report"]["r"]
    print(f"Score model: best epoch {checkpoint['best_epoch']}, "
          f"training r {train_r:.3f}, validation r {val_r:.3f}")
    print(f"Gap between them: {train_r - val_r:+.3f}. A small gap means cut points "
          f"fitted on\ntraining predictions are not meaningfully optimistic.\n")
    print(f"Predicted score spread: sd {val_scores.std():.2f} "
          f"(real ratings sd {ratings[va].std():.2f})\n")

    print(f"  {'approach':<46}{'VAL balanced':>12}{'val acc':>10}{'vs classifier':>11}")
    print("  " + "-" * 80)
    print(f"  {'classification LSTM (rich features)':<46}"
          f"{CLASSIFIER_BASELINE:>12.4f}{0.4570:>10.4f}{0.0:>+11.2f}")
    print("  " + "-" * 80)

    results = {}
    results["fixed 4/7"] = report("score -> fixed 4/7", labels[va],
                                  official_bins(val_scores))
    cuts = distribution_matched_cuts(train_scores, labels[tr])
    results["distribution-matched"] = report(
        f"score -> distribution-matched {cuts[0]:.2f}/{cuts[1]:.2f}",
        labels[va], bin_with_cuts(val_scores, cuts))
    best_cuts = optimized_cuts(train_scores, labels[tr])
    results["optimized"] = report(
        f"score -> optimized cuts {best_cuts[0]:.2f}/{best_cuts[1]:.2f}",
        labels[va], bin_with_cuts(val_scores, best_cuts))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        soft = LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=args.seed).fit(
            train_scores.reshape(-1, 1), labels[tr])
    probabilities = soft.predict_proba(val_scores.reshape(-1, 1))
    results["logistic on score"] = report(
        "score -> logistic classifier", labels[va], probabilities.argmax(axis=1))

    confident = probabilities.max(axis=1) >= 0.5
    if confident.any():
        print(f"\n  On the {100*confident.mean():.0f}% of trials where the model is at "
              f"least 50% sure,\n  balanced accuracy is "
              f"{balanced_accuracy_score(labels[va][confident], probabilities.argmax(axis=1)[confident]):.4f}.")

    best_name = max(results, key=results.get)
    print(f"\n  Best mapping: {best_name} at {results[best_name]:.4f} "
          f"({100*(results[best_name]-CLASSIFIER_BASELINE):+.2f} points vs the classifier)")
    print(
        "\n  Remember what the score model also gives that the classifier cannot:\n"
        "  a number per trial, and a per-person trend the classifier never modelled."
    )


if __name__ == "__main__":
    main()
