"""Train the LSTM to predict the 0-10 pain rating as a number, not a class.

The classifier treats Low, Moderate and High as three unrelated names, so guessing
6.8 when the truth is 7.0 is punished exactly as hard as guessing 2.0. Training on
the number itself makes a near miss cheap, which is the point of this script.

It is a separate file on purpose. train_lstm.py's loop, loss, evaluation, early
stopping and checkpoint fields all assume three classes, and every leakage
protection in it is tied to that assumption. Rather than thread a third task mode
through 2,200 validated lines, this script imports that file's loading, splitting
and QC functions unchanged and only replaces the head, the loss, and the metrics.
The classifier keeps working exactly as before.

The network is identical to the classifier apart from its last layer: the same
single LSTM over the same time steps, then Linear(hidden, 1) producing one number
instead of three scores.

Loss is Huber rather than plain squared error. Squared error is dominated by the
rare trials rated 0 or 10, and chasing those extremes costs accuracy on the bulk of
the data.

No class thresholds appear anywhere here. Turning a score into Low/Moderate/High is
a separate decision, deliberately left for later.

Metrics, and why the third one matters most
-------------------------------------------
  correlation r   do higher predictions go with higher real ratings, pooled
  MAE             average miss, in rating points on the 0-10 scale
  within-subject r  the same correlation computed inside each person, then averaged

The pooled number can look respectable purely because the model separates a
generally high-rating person from a generally low-rating one. The within-subject
number asks the question a neurofeedback device actually needs: as one person's
pain rises and falls, does the prediction follow?

Usage (from repo root):
    python scripts/preprocessing/train_lstm_score.py \
        --feature-mode rich --rich-features outputs/rich_full/rich_all9_features.npz
"""

import argparse
import copy
import json
import os
import platform
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich_feature_extraction import apply_uint8_bounds, fit_uint8_bounds
from feature_extraction import quantize_to_uint8
from train_lstm import (  # noqa: E402 -- path setup must precede this import
    _atomic_torch_save,
    base_trial_weights,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    make_training_loader,
    model_feature_view,
    normalize_selection_datasets,
    select_dataset_view,
    subject_split,
)

RATING_MIN, RATING_MAX = 0.0, 10.0


class PainScoreLSTM(nn.Module):
    """Same recurrent body as PainLSTM; one continuous output instead of class scores."""

    def __init__(self, input_size, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.score(h_n[-1]).squeeze(-1)


def predict_scores(model, X, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            out.append(model(X[start:start + batch_size]).cpu())
    # Ratings cannot leave 0..10, so neither should a prediction of one.
    return torch.cat(out).numpy().clip(RATING_MIN, RATING_MAX)


def safe_correlation(a, b):
    """Pearson r, returning NaN when a constant input makes it undefined."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_report(predictions, truth, dataset_id, subject_id):
    errors = predictions - truth
    per_dataset = {}
    for dataset in sorted(set(dataset_id.tolist())):
        mask = dataset_id == dataset
        per_dataset[dataset] = {
            "trials": int(mask.sum()),
            "subjects": int(len(set(subject_id[mask].tolist()))),
            "r": safe_correlation(predictions[mask], truth[mask]),
            "mae": float(np.mean(np.abs(errors[mask]))),
        }

    within = []
    for dataset, subject in sorted(set(zip(dataset_id.tolist(), subject_id.tolist()))):
        mask = (dataset_id == dataset) & (subject_id == subject)
        # A subject who reported the same number every time carries no within-person
        # signal to detect, so they cannot contribute a correlation.
        value = safe_correlation(predictions[mask], truth[mask])
        if np.isfinite(value):
            within.append(value)

    return {
        "r": safe_correlation(predictions, truth),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
        "prediction_sd": float(np.std(predictions)),
        "truth_sd": float(np.std(truth)),
        "within_subject_r_mean": float(np.mean(within)) if within else float("nan"),
        "within_subject_r_median": float(np.median(within)) if within else float("nan"),
        "within_subject_positive_fraction": (
            float(np.mean(np.asarray(within) > 0)) if within else float("nan")
        ),
        "within_subject_count": len(within),
        "per_dataset": per_dataset,
    }


def print_score_report(report, label):
    print(f"\n=== {label} ===")
    print("dataset     subjects  trials       r      MAE")
    for dataset, row in report["per_dataset"].items():
        print(f"{dataset:<12}{row['subjects']:>8}{row['trials']:>8}"
              f"{row['r']:>8.3f}{row['mae']:>9.3f}")
    print(f"\nPooled correlation r:        {report['r']:.3f}")
    print(f"Mean absolute error:         {report['mae']:.3f} rating points")
    print(f"Root mean squared error:     {report['rmse']:.3f}")
    print(f"Prediction spread (sd):      {report['prediction_sd']:.3f}"
          f"   vs real spread {report['truth_sd']:.3f}"
          f"   ({100*report['prediction_sd']/max(report['truth_sd'],1e-9):.0f}% retained)")
    print(f"Within-subject r (mean):     {report['within_subject_r_mean']:.3f}"
          f"   median {report['within_subject_r_median']:.3f}")
    print(f"Subjects tracked positively: "
          f"{100*report['within_subject_positive_fraction']:.0f}% "
          f"of {report['within_subject_count']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--feature-mode",
                    choices=("channel-average", "per-channel", "rich"), default="rich")
    ap.add_argument("--rich-features", default="outputs/rich_full/rich_all9_features.npz")
    ap.add_argument("--split-from-checkpoint",
                    default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt")
    ap.add_argument("--split-seed", type=int, default=20260725)
    ap.add_argument("--selection-datasets", nargs="+")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--huber-delta", type=float, default=2.0,
                    help="Rating-point error beyond which the loss stops growing quadratically.")
    ap.add_argument("--model-seed", type=int, default=20260726)
    ap.add_argument("--loader-seed", type=int, default=20260727)
    ap.add_argument("--save",
                    default="outputs/controlled_comparison/all9_rich_score_seed20260726.pt")
    args = ap.parse_args()

    torch.manual_seed(args.model_seed)
    torch.use_deterministic_algorithms(True)

    archive = load_feature_archive(args.data)
    _, training_dataset_ids = select_dataset_view(archive, None)
    selection_dataset_ids = normalize_selection_datasets(
        args.selection_datasets, training_dataset_ids
    )
    model_power, feature_order = model_feature_view(
        archive, args.feature_mode, rich_features_path=args.rich_features
    )
    ratings = archive["ratings"].astype(np.float64)
    dataset_id = archive["dataset_id"].astype(str)
    subject_id = archive["subject_id"].astype(str)

    if args.split_from_checkpoint:
        train_keys, val_keys, test_keys, split_source = load_checkpoint_split(
            args.split_from_checkpoint, archive, training_dataset_ids
        )
    else:
        train_keys, val_keys, test_keys = subject_split(
            dataset_id, subject_id, seed=args.split_seed
        )
        split_source = {"mode": "seed", "seed": int(args.split_seed)}
    train_mask = keys_mask(dataset_id, subject_id, train_keys)
    val_mask = keys_mask(dataset_id, subject_id, val_keys)

    print(f"Feature mode: {args.feature_mode}   {model_power.shape[1]} steps x "
          f"{model_power.shape[2]} features")
    print(f"Split source: {split_source['mode']}")
    print(f"Train {int(train_mask.sum())} trials / {len(train_keys)} subjects   "
          f"Validation {int(val_mask.sum())} trials / {len(val_keys)} subjects")
    print(f"Held-out test split reserved: {len(test_keys)} subjects, never touched here.")
    print(f"Training ratings: mean {ratings[train_mask].mean():.2f} "
          f"sd {ratings[train_mask].std():.2f}\n")

    # 8-bit input scaling, fitted on training rows only, exactly as the classifier does.
    if args.feature_mode == "rich":
        lo, hi = fit_uint8_bounds(model_power[train_mask].reshape(-1, model_power.shape[2]))
        to_input = lambda v: apply_uint8_bounds(v, lo, hi).astype(np.float32) / 255.0
    else:
        log_train = np.log1p(model_power[train_mask])
        lo = np.percentile(log_train, 1, axis=0)
        hi = np.percentile(log_train, 99, axis=0)
        to_input = lambda v: quantize_to_uint8(v, lo, hi).astype(np.float32) / 255.0

    X_train = torch.from_numpy(to_input(model_power[train_mask]))
    X_val = torch.from_numpy(to_input(model_power[val_mask]))
    y_train = torch.from_numpy(ratings[train_mask].astype(np.float32))
    y_val_np = ratings[val_mask]

    # Equal mass per subject and per dataset. No class correction exists here:
    # the target is continuous, so there are no classes to rebalance.
    weights = torch.from_numpy(
        base_trial_weights(dataset_id[train_mask], subject_id[train_mask]).astype(np.float32)
    )
    loader = make_training_loader(
        X_train, y_train, weights, batch_size=args.batch_size, seed=args.loader_seed
    )

    model = PainScoreLSTM(model_power.shape[2], hidden_size=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(reduction="none", delta=args.huber_delta)

    best_r, best_state, best_epoch, best_report = -np.inf, None, None, None
    since_improvement = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_weight = 0.0, 0.0
        for X_batch, y_batch, w_batch in loader:
            opt.zero_grad()
            losses = loss_fn(model(X_batch), y_batch)
            (torch.mean(losses * w_batch)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            total_loss += float(torch.sum(losses.detach() * w_batch))
            total_weight += float(torch.sum(w_batch))

        val_pred = predict_scores(model, X_val, args.batch_size)
        report = score_report(val_pred, y_val_np, dataset_id[val_mask], subject_id[val_mask])
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / total_weight,
            "val_r": report["r"],
            "val_mae": report["mae"],
            "val_within_subject_r": report["within_subject_r_mean"],
        })
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d} | train loss {total_loss/total_weight:.3f} "
                  f"| val r {report['r']:.3f} | val MAE {report['mae']:.3f} "
                  f"| within-subject r {report['within_subject_r_mean']:.3f}")

        if report["r"] > best_r + 1e-4:
            best_r, best_epoch, best_report = report["r"], epoch, report
            best_state = copy.deepcopy(model.state_dict())
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}: no improvement for "
                      f"{args.patience} epochs")
                break

    model.load_state_dict(best_state)
    print(f"\nRestored best epoch {best_epoch} (validation r {best_r:.3f})")
    print_score_report(best_report, "Best held-out VALIDATION (score prediction)")

    train_pred = predict_scores(model, X_train, args.batch_size)
    train_report = score_report(
        train_pred, ratings[train_mask], dataset_id[train_mask], subject_id[train_mask]
    )
    print(f"\nTraining-set r for reference: {train_report['r']:.3f} "
          f"(MAE {train_report['mae']:.3f})")
    print("A training r far above validation r means memorizing rather than learning.")

    _atomic_torch_save({
        "checkpoint_schema_version": 1,
        "task": "rating_regression",
        "model_state_dict": model.state_dict(),
        "input_size": model_power.shape[2],
        "hidden_size": args.hidden,
        "feature_mode": args.feature_mode,
        "feature_order": feature_order,
        "quantization_low": lo,
        "quantization_high": hi,
        "rating_range": [RATING_MIN, RATING_MAX],
        "huber_delta": args.huber_delta,
        "best_epoch": best_epoch,
        "validation_report": best_report,
        "training_report": train_report,
        "training_history": history,
        "split_source": split_source,
        "selection_dataset_ids": selection_dataset_ids,
        "train_subject_keys": sorted(train_keys),
        "val_subject_keys": sorted(val_keys),
        "test_subject_keys": sorted(test_keys),
        "model_seed": args.model_seed,
        "loader_seed": args.loader_seed,
        "class_thresholds": None,  # deliberately unset: mapping is a later decision
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }, args.save)
    print(f"\nSaved score model to {args.save}")
    print("No class thresholds were stored: score -> Low/Moderate/High is still open.")


if __name__ == "__main__":
    main()
