"""Force the model to forget which machine recorded a trial.

The nine datasets were collected on different EEG systems, and each leaves a
fingerprint: a classifier reads the source dataset from the features at 51.9% against
11.1% for guessing. Because the datasets also have very different class mixes -- 54%
High in ds005286 against 41% Low in ds005292 -- a model can score by recognising the
equipment and answering with that site's typical label. Dataset identity alone, with
no EEG whatsoever, reaches 0.4064 pooled.

Scoring by dataset-macro already stops that shortcut being rewarded. This script
attacks it one step earlier, by stopping the model from learning it at all.

How
---
Two heads share one LSTM. The first predicts pain as usual. The second tries to name
the source dataset, but a gradient reversal sits in front of it: during backpropagation
the sign of its gradient is flipped before reaching the LSTM. The dataset head still
learns to identify sites as well as it can, while the LSTM is pushed in the opposite
direction -- toward a representation where sites are indistinguishable. What survives
is whatever predicts pain the same way on every machine.

The reversal strength ramps up from zero rather than starting high, because an
untrained dataset head produces meaningless gradients, and letting those reshape the
LSTM from the first step tends to prevent it learning anything at all.

Reading the result
------------------
Two numbers matter together. Dataset-macro says whether pain prediction improved. The
dataset head's own accuracy says whether the method did what it claims: if it stays
near 51.9%, the fingerprint was never removed and any change in the pain score came
from somewhere else. Success looks like site accuracy falling toward 11.1% while
dataset-macro holds or rises.

Usage (from repo root):
    python scripts/preprocessing/domain_adversarial.py
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from rich_feature_extraction import apply_uint8_bounds, fit_uint8_bounds
from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    model_feature_view,
    select_dataset_view,
    training_trial_weights,
)

EPOCHS = 40


class GradientReversal(torch.autograd.Function):
    """Identity going forward; flips the gradient's sign coming back."""

    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


class AdversarialLSTM(nn.Module):
    def __init__(self, input_size, n_datasets, hidden=16, domain_hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=1, batch_first=True)
        self.pain = nn.Linear(hidden, len(CLASS_NAMES))
        self.domain = nn.Sequential(
            nn.Linear(hidden, domain_hidden), nn.ReLU(),
            nn.Linear(domain_hidden, n_datasets),
        )

    def forward(self, x, strength=0.0):
        _, (h_n, _) = self.lstm(x)
        hidden = h_n[-1]
        return self.pain(hidden), self.domain(GradientReversal.apply(hidden, strength))


def predict(model, X, batch_size=256):
    model.eval()
    pain, domain = [], []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            p, d = model(X[start:start + batch_size])
            pain.append(p.argmax(1).cpu())
            domain.append(d.argmax(1).cpu())
    return torch.cat(pain).numpy(), torch.cat(domain).numpy()


def dataset_macro(y_true, y_pred, datasets):
    return float(np.mean([
        balanced_accuracy_score(y_true[datasets == ds], y_pred[datasets == ds])
        for ds in sorted(set(datasets.tolist()))
    ]))


def train(X_train, y_train, d_train, weights, input_size, n_datasets,
          max_strength, args):
    torch.manual_seed(args.model_seed)
    model = AdversarialLSTM(input_size, n_datasets, hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pain_loss = nn.CrossEntropyLoss(reduction="none")
    domain_loss = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.loader_seed)

    model.train()
    for epoch in range(EPOCHS):
        # Ramp the reversal in: an untrained dataset head emits noise, and letting
        # that reshape the LSTM from step one stops it learning pain at all.
        progress = epoch / max(EPOCHS - 1, 1)
        strength = max_strength * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)
        order = torch.randperm(X_train.shape[0], generator=generator)
        for start in range(0, X_train.shape[0], args.batch_size):
            index = order[start:start + args.batch_size]
            opt.zero_grad()
            pain_logits, domain_logits = model(X_train[index], strength)
            loss = torch.mean(pain_loss(pain_logits, y_train[index]) * weights[index])
            if max_strength > 0:
                loss = loss + domain_loss(domain_logits, d_train[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--rich-features", default="outputs/rich_full/rich_all9_features.npz")
    ap.add_argument("--split-from-checkpoint",
                    default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt")
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.1, 0.3, 1.0])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--class-balance-strength", type=float, default=0.5)
    ap.add_argument("--model-seed", type=int, default=20260726)
    ap.add_argument("--loader-seed", type=int, default=20260727)
    args = ap.parse_args()

    archive = load_feature_archive(args.data)
    _, training_dataset_ids = select_dataset_view(archive, None)
    features, _order = model_feature_view(
        archive, "rich", rich_features_path=args.rich_features
    )
    y = archive["labels"].astype(np.int64)
    dataset_id = archive["dataset_id"].astype(str)
    subject_id = archive["subject_id"].astype(str)

    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        args.split_from_checkpoint, archive, training_dataset_ids
    )
    tr = keys_mask(dataset_id, subject_id, train_keys)
    va = keys_mask(dataset_id, subject_id, val_keys)

    codes = sorted(set(dataset_id.tolist()))
    # int64 explicitly: numpy defaults to int32 on Windows, which torch's
    # cross-entropy rejects as a target dtype.
    domain_index = np.asarray(
        [codes.index(v) for v in dataset_id.tolist()], dtype=np.int64
    )

    low, high = fit_uint8_bounds(features[tr].reshape(-1, features.shape[2]))
    to_input = lambda v: apply_uint8_bounds(v, low, high).astype(np.float32) / 255.0
    X_train = torch.from_numpy(to_input(features[tr]))
    X_val = torch.from_numpy(to_input(features[va]))
    y_train = torch.from_numpy(y[tr])
    d_train = torch.from_numpy(domain_index[tr])
    weights_np, *_ = training_trial_weights(
        dataset_id[tr], subject_id[tr], y[tr],
        class_balance_strength=args.class_balance_strength,
    )
    weights = torch.from_numpy(weights_np)

    print(f"Input {features.shape[1]} steps x {features.shape[2]} features   "
          f"{len(codes)} datasets")
    print(f"Train {int(tr.sum())}   Validation {int(va.sum())}")
    print(f"Chance for guessing the dataset: {1/len(codes):.4f}")
    print(f"A plain classifier reads the dataset from these features at about 0.52.\n")

    print(f"  {'reversal strength':<22}{'DATASET-MACRO':>15}{'pooled':>9}{'acc':>8}"
          f"{'site acc':>10}")
    print("  " + "-" * 66)
    started = time.time()
    results = {}
    for strength in args.strengths:
        model = train(X_train, y_train, d_train, weights,
                      features.shape[2], len(codes), strength, args)
        pain_pred, domain_pred = predict(model, X_val)
        macro = dataset_macro(y[va], pain_pred, dataset_id[va])
        results[strength] = macro
        label = "0.0 (off, baseline)" if strength == 0 else f"{strength}"
        print(f"  {label:<22}{macro:>15.4f}"
              f"{balanced_accuracy_score(y[va], pain_pred):>9.4f}"
              f"{accuracy_score(y[va], pain_pred):>8.4f}"
              f"{accuracy_score(domain_index[va], domain_pred):>10.4f}")

    baseline = results[args.strengths[0]]
    best = max(results, key=results.get)
    print(f"\n  Best: strength {best} at {results[best]:.4f} "
          f"({100*(results[best]-baseline):+.2f} points vs adversary off)")
    print(f"  Total time {time.time()-started:.0f}s")
    print(
        "\n  The site-accuracy column is the honesty check. If it stays near 0.52 the\n"
        "  fingerprint was never removed, so any change in the pain score came from\n"
        "  somewhere else and should not be credited to this method."
    )


if __name__ == "__main__":
    main()
