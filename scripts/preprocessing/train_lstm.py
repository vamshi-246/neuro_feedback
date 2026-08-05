"""
Trains the small LSTM that mirrors the RTL contract: 3 time steps, 3 features
per step (alpha/beta/theta, 0-255), 3 output classes (Low/Moderate/High,
matching pain_classifier_fsm.v's states).

Subject-level split (never mix one person's trials across train/val/test --
same discipline as DS005285_LSTM_ARCHITECTURE.md) is keyed on
(dataset_id, subject_id) so that "sub-001" from two different datasets is
correctly treated as two different people.

Usage:
    python scripts/preprocessing/train_lstm.py --data outputs/pipeline_dev/pooled_features.npz
"""

import argparse

import numpy as np
import torch
from torch import nn

from feature_extraction import quantize_to_uint8


class PainLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])


def subject_split(dataset_id, subject_id, seed=20260725, train_frac=0.6, val_frac=0.2):
    keys = sorted(set(zip(dataset_id.tolist(), subject_id.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac))) if n > 2 else 0
    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])
    return train_keys, val_keys, test_keys


def keys_mask(dataset_id, subject_id, keys):
    return np.array([(d, s) in keys for d, s in zip(dataset_id.tolist(), subject_id.tolist())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/pipeline_dev/pooled_features.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    raw_power = d["raw_power"].astype(np.float64)   # (N, 3 steps, 3 bands)
    y = d["labels"].astype(np.int64)
    dataset_id, subject_id = d["dataset_id"], d["subject_id"]

    train_keys, val_keys, test_keys = subject_split(dataset_id, subject_id)
    train_mask = keys_mask(dataset_id, subject_id, train_keys)
    val_mask = keys_mask(dataset_id, subject_id, val_keys)
    test_mask = keys_mask(dataset_id, subject_id, test_keys)

    print(f"Subjects -> train:{len(train_keys)} val:{len(val_keys)} test:{len(test_keys)}")
    print(f"Trials   -> train:{train_mask.sum()} val:{val_mask.sum()} test:{test_mask.sum()}")

    # Fit the 0-255 scaling on TRAIN trials only, then apply the same fixed
    # bounds to val/test -- this is the leakage-safe step that used to happen
    # (incorrectly, pool-wide) in build_dataset.py. See DS005285_LSTM_ARCHITECTURE.md
    # for why this discipline matters.
    log_power_train = np.log1p(raw_power[train_mask])
    band_lo = np.percentile(log_power_train, 1, axis=0)
    band_hi = np.percentile(log_power_train, 99, axis=0)
    print(f"Quantization bounds fit on train only -> lo:{band_lo.round(2).tolist()} "
          f"hi:{band_hi.round(2).tolist()}")

    X = quantize_to_uint8(raw_power, band_lo, band_hi).astype(np.float32) / 255.0  # (N,3,3) -> [0,1]

    X_train = torch.from_numpy(X[train_mask])
    y_train = torch.from_numpy(y[train_mask])
    X_val = torch.from_numpy(X[val_mask]) if val_mask.sum() else None
    y_val = torch.from_numpy(y[val_mask]) if val_mask.sum() else None
    X_test = torch.from_numpy(X[test_mask]) if test_mask.sum() else None
    y_test = torch.from_numpy(y[test_mask]) if test_mask.sum() else None

    model = PainLSTM(input_size=3, hidden_size=args.hidden, n_classes=3)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (model(X_train).argmax(-1) == y_train).float().mean().item()
                msg = f"epoch {epoch:3d} | train loss {loss.item():.3f} | train acc {train_acc:.3f}"
                if X_val is not None:
                    val_acc = (model(X_val).argmax(-1) == y_val).float().mean().item()
                    msg += f" | val acc {val_acc:.3f}"
                print(msg)

    if X_test is not None:
        model.eval()
        with torch.no_grad():
            test_acc = (model(X_test).argmax(-1) == y_test).float().mean().item()
        print(f"\nFinal held-out test accuracy: {test_acc:.3f} (on {test_mask.sum()} trials, "
              f"{len(test_keys)} subject(s))")
    else:
        print("\nNo held-out test subjects in this run -- too few unique subjects to split "
              "three ways meaningfully. This is expected for the current 5-subject smoke test; "
              "re-run after scaling up to the full subject lists.")


if __name__ == "__main__":
    main()
