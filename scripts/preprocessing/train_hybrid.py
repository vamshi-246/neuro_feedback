"""Hybrid model: a CNN reads the raw waveform, the LSTM reads our computed features.

The idea being tested is that two kinds of information suit two kinds of processing.
Quantities like N2-P2 amplitude are physically meaningful, cheap to compute exactly,
and hard for a network to rediscover from limited data, so they are handed over
directly. Waveform shape has local structure that convolution is built for, and
whatever we never thought to measure can only be found there.

    raw waveform (4 x 375)  ->  CNN  ->  embedding \
                                                    -> concatenate -> classifier
    rich features (3 x 114) ->  LSTM ->  hidden    /

Both branches train together, so the CNN learns whatever the engineered features are
missing rather than duplicating them.

Four modes make the comparison decidable
----------------------------------------
  lstm    engineered features only, reproducing the existing model
  cnn     raw waveform only, learning everything from scratch
  hybrid  both branches
  frozen-features  hybrid, but the LSTM branch is detached from the gradient, which
          answers whether the CNN adds anything ON TOP of the features rather than
          merely relearning them

The last mode matters most: hybrid beating lstm proves little if the CNN is simply
recomputing band power internally. What we want to know is whether the waveform
carries something the features do not.

The CNN is deliberately small, uses depthwise convolution per channel before mixing
channels, and is dropout-regularized. Earlier experiments showed this data punishes
extra capacity: 14 electrodes and 244 MVAR columns both made validation worse while
training improved.

Usage (from repo root):
    python scripts/preprocessing/train_hybrid.py --mode hybrid
"""

import argparse
import copy
import os
import platform
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich_feature_extraction import apply_uint8_bounds, fit_uint8_bounds
from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    _atomic_torch_save,
    evaluation_breakdown,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    make_training_loader,
    model_feature_view,
    print_evaluation_breakdown,
    select_dataset_view,
    training_trial_weights,
)


class WaveformCNN(nn.Module):
    """Small 1D convolutional encoder for a (channels, samples) epoch.

    The first layer convolves within each channel separately, so it can learn
    frequency-selective filters without immediately blending electrodes; the second
    mixes channels once those filters exist. That ordering is what EEG-specific
    architectures use, and it keeps the parameter count far below a dense stack.
    """

    def __init__(self, n_channels, n_samples, embedding=32, dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv1d(
            n_channels, n_channels * 8, kernel_size=25, padding=12,
            groups=n_channels, bias=False,
        )
        self.temporal_norm = nn.BatchNorm1d(n_channels * 8)
        self.spatial = nn.Conv1d(n_channels * 8, 32, kernel_size=1, bias=False)
        self.spatial_norm = nn.BatchNorm1d(32)
        self.pool = nn.AvgPool1d(8)
        self.separable = nn.Conv1d(32, 32, kernel_size=9, padding=4, groups=32, bias=False)
        self.separable_norm = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout)
        pooled = (n_samples // 8) // 4
        self.project = nn.Linear(32 * pooled, embedding)
        self.final_pool = nn.AvgPool1d(4)

    def forward(self, x):
        h = torch.nn.functional.elu(self.temporal_norm(self.temporal(x)))
        h = torch.nn.functional.elu(self.spatial_norm(self.spatial(h)))
        h = self.dropout(self.pool(h))
        h = torch.nn.functional.elu(self.separable_norm(self.separable(h)))
        h = self.dropout(self.final_pool(h))
        return self.project(h.flatten(1))


class HybridModel(nn.Module):
    def __init__(self, feature_size, n_channels, n_samples, hidden=16,
                 embedding=32, dropout=0.5, mode="hybrid"):
        super().__init__()
        if mode not in ("lstm", "cnn", "hybrid", "frozen-features"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        width = 0
        if mode != "cnn":
            self.lstm = nn.LSTM(feature_size, hidden, num_layers=1, batch_first=True)
            width += hidden
        if mode != "lstm":
            self.cnn = WaveformCNN(n_channels, n_samples, embedding, dropout)
            width += embedding
        self.classifier = nn.Linear(width, len(CLASS_NAMES))

    def forward(self, features, waveform):
        parts = []
        if self.mode != "cnn":
            _, (h_n, _) = self.lstm(features)
            hidden = h_n[-1]
            # Detaching stops gradients reaching the feature branch, so the CNN must
            # earn its contribution instead of reshaping the features to suit itself.
            parts.append(hidden.detach() if self.mode == "frozen-features" else hidden)
        if self.mode != "lstm":
            parts.append(self.cnn(waveform))
        return self.classifier(torch.cat(parts, dim=1))


def predict(model, features, waveform, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = start + batch_size
            out.append(model(features[start:stop], waveform[start:stop]).argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lstm", "cnn", "hybrid", "frozen-features"),
                    default="hybrid")
    ap.add_argument("--data", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--rich-features", default="outputs/rich_full/rich_all9_features.npz")
    ap.add_argument("--raw-epochs", default="outputs/rich_full/raw_epochs_all9.npz")
    ap.add_argument("--split-from-checkpoint",
                    default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--embedding", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--class-balance-strength", type=float, default=0.5)
    ap.add_argument("--model-seed", type=int, default=20260726)
    ap.add_argument("--loader-seed", type=int, default=20260727)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.model_seed)

    archive = load_feature_archive(args.data)
    _, training_dataset_ids = select_dataset_view(archive, None)
    features_all, _order = model_feature_view(
        archive, "rich", rich_features_path=args.rich_features
    )
    labels = archive["labels"].astype(np.int64)
    dataset_id = archive["dataset_id"].astype(str)
    subject_id = archive["subject_id"].astype(str)
    epoch_index = archive["epoch_index"].astype(np.int64)

    raw = np.load(args.raw_epochs, allow_pickle=False)
    raw_keys = {
        f"{d}||{s}||{int(e)}": i
        for i, (d, s, e) in enumerate(zip(
            raw["dataset_id"].astype(str).tolist(),
            raw["subject_id"].astype(str).tolist(),
            raw["epoch_index"].astype(np.int64).tolist(),
        ))
    }
    order = []
    for d, s, e in zip(dataset_id.tolist(), subject_id.tolist(), epoch_index.tolist()):
        key = f"{d}||{s}||{int(e)}"
        if key not in raw_keys:
            raise SystemExit(f"raw epochs are missing trial {key}; rebuild them")
        order.append(raw_keys[key])
    waveforms = raw["waveforms"][np.asarray(order)]

    train_keys, val_keys, _test, split_source = load_checkpoint_split(
        args.split_from_checkpoint, archive, training_dataset_ids
    )
    tr = keys_mask(dataset_id, subject_id, train_keys)
    va = keys_mask(dataset_id, subject_id, val_keys)

    lo, hi = fit_uint8_bounds(features_all[tr].reshape(-1, features_all.shape[2]))
    to_features = lambda v: apply_uint8_bounds(v, lo, hi).astype(np.float32) / 255.0
    # Waveform scaling uses training rows only, exactly as the feature path does.
    wave_scale = float(np.std(waveforms[tr]))
    to_waves = lambda v: (v / max(wave_scale, 1e-8)).astype(np.float32)

    X_train = torch.from_numpy(to_features(features_all[tr]))
    W_train = torch.from_numpy(to_waves(waveforms[tr]))
    y_train = torch.from_numpy(labels[tr])
    X_val = torch.from_numpy(to_features(features_all[va]))
    W_val = torch.from_numpy(to_waves(waveforms[va]))

    weights_np, *_ = training_trial_weights(
        dataset_id[tr], subject_id[tr], labels[tr],
        class_balance_strength=args.class_balance_strength,
    )
    weights = torch.from_numpy(weights_np)

    print(f"Mode: {args.mode}")
    print(f"Features {tuple(X_train.shape[1:])}   Waveforms {tuple(W_train.shape[1:])}")
    print(f"Train {int(tr.sum())} trials   Validation {int(va.sum())} trials\n")

    model = HybridModel(
        feature_size=features_all.shape[2],
        n_channels=waveforms.shape[1],
        n_samples=waveforms.shape[2],
        hidden=args.hidden, embedding=args.embedding,
        dropout=args.dropout, mode=args.mode,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loader = make_training_loader(
        torch.arange(X_train.shape[0]).unsqueeze(-1).unsqueeze(-1).float(),
        y_train, weights, batch_size=args.batch_size, seed=args.loader_seed,
    )

    best, best_state, best_epoch, best_report = -1.0, None, None, None
    since = 0
    for epoch in range(args.epochs):
        model.train()
        total, mass = 0.0, 0.0
        for index_batch, y_batch, w_batch in loader:
            index = index_batch.squeeze(-1).squeeze(-1).long()
            opt.zero_grad()
            logits = model(X_train[index], W_train[index])
            losses = loss_fn(logits, y_batch)
            (torch.mean(losses * w_batch)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(torch.sum(losses.detach() * w_batch))
            mass += float(torch.sum(w_batch))

        val_pred = predict(model, X_val, W_val, args.batch_size)
        report = evaluation_breakdown(labels[va], val_pred, dataset_id[va], subject_id[va])
        balanced = report["pooled"]["balanced_accuracy"]
        if epoch % 5 == 0:
            print(f"epoch {epoch:3d} | train loss {total/mass:.3f} "
                  f"| val balanced {balanced:.4f} | val acc {report['pooled']['accuracy']:.4f}")
        if balanced > best + 1e-4:
            best, best_epoch, best_report = balanced, epoch, report
            best_state = copy.deepcopy(model.state_dict())
            since = 0
        else:
            since += 1
            if since >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"\nRestored best epoch {best_epoch} (validation balanced {best:.4f})")
    print_evaluation_breakdown(best_report, f"Best held-out VALIDATION ({args.mode})")
    print("\nReference points on this identical split:")
    print("  LSTM, rich features        0.4750")
    print("  gradient boosting, rich    0.4803")

    if args.save:
        _atomic_torch_save({
            "mode": args.mode,
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "validation_report": best_report,
            "quantization_low": lo, "quantization_high": hi,
            "waveform_scale": wave_scale,
            "split_source": split_source,
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "torch": torch.__version__, "platform": platform.platform()},
        }, args.save)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
