"""
Trains the small LSTM using 3 time steps and 3 pain classes. The default input
is the current 3 channel-averaged bands per step. An optional per-channel mode
uses every requested channel-band pair and can therefore grow with the channel
list stored in the feature archive.

Subject-level split never mixes one subject record's trials across
train/validation/test.  The key is (dataset_id, subject_id), because subject
numbers restart in every dataset.  Cross-dataset participant identity is not
available in the released files, so different dataset keys are treated as
separate records and that limitation must be reported with the results.

Usage:
    python scripts/preprocessing/train_lstm.py \
        --data outputs/all9_full/all9_features.npz \
        --feature-mode channel-average
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
import platform
import sys
import tempfile
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature_extraction import (
    BASELINE_WINDOW_S,
    BANDS,
    MIN_BASELINE_POWER,
    WINDOWS_S,
    bin_rating,
    quantize_to_uint8,
)
from rich_feature_extraction import (
    apply_uint8_bounds,
    fit_uint8_bounds,
    rich_lstm_view,
)


ARCHIVE_SCHEMA_VERSION = 1
REQUIRED_ARCHIVE_KEYS = {
    "schema_version",
    "build_id",
    "archive_complete",
    "qc_report_relpath",
    "qc_summary_json",
    "selected_dataset_ids",
    "expected_subject_keys",
    "successful_subject_keys",
    "failed_subject_keys",
    "relative_power",
    "channel_relative_power",
    "channel_baseline_power",
    "ratings",
    "labels",
    "subject_id",
    "dataset_id",
    "epoch_index",
    "event_type",
    "laser_power",
    "event_onset_sample",
    "feature_onset_sample",
    "channel_order",
    "band_order",
    "baseline_window_s",
    "post_windows_s",
}


CLASS_NAMES = ["Low", "Moderate", "High"]
ORDINAL_RANK_SEMANTICS = ["label >= 1 (Moderate or High)", "label >= 2 (High)"]


class PainLSTM(nn.Module):
    def __init__(
        self,
        input_size=3,
        hidden_size=16,
        n_classes=3,
        task_mode="categorical",
    ):
        super().__init__()
        if task_mode not in ("categorical", "ordinal"):
            raise ValueError(f"unknown task mode {task_mode!r}")
        if task_mode == "ordinal" and n_classes != len(CLASS_NAMES):
            raise ValueError(
                f"ordinal mode requires exactly {len(CLASS_NAMES)} ordered classes"
            )
        self.task_mode = task_mode
        self.n_classes = int(n_classes)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        if task_mode == "categorical":
            # Keep the historical module name and parameter shapes so existing
            # categorical state dictionaries remain loadable without migration.
            self.classifier = nn.Linear(hidden_size, n_classes)
        else:
            # A single scalar score and two strictly ordered cut points implement
            # cumulative logits for y >= 1 and y >= 2. The score has no bias
            # because a shared score bias and a common threshold shift would be
            # unidentifiable.
            self.ordinal_score = nn.Linear(hidden_size, 1, bias=False)
            self.ordinal_first_threshold = nn.Parameter(torch.tensor(-0.5))
            self.ordinal_threshold_gap_unconstrained = nn.Parameter(
                torch.tensor(0.541324854612918)
            )

    def ordinal_thresholds(self):
        if self.task_mode != "ordinal":
            raise RuntimeError("ordinal thresholds are only defined in ordinal mode")
        first = self.ordinal_first_threshold
        second = first + nn.functional.softplus(
            self.ordinal_threshold_gap_unconstrained
        )
        return torch.stack((first, second))

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        final_hidden = h_n[-1]
        if self.task_mode == "categorical":
            return self.classifier(final_hidden)
        score = self.ordinal_score(final_hidden)
        return score - self.ordinal_thresholds().unsqueeze(0)


def _split_counts(n_subjects: int, train_frac: float, val_frac: float):
    """Allocate subjects while preserving a test subject whenever possible."""

    if n_subjects <= 0:
        raise ValueError("cannot split zero subjects")
    if not 0.0 < train_frac < 1.0 or not 0.0 <= val_frac < 1.0:
        raise ValueError("split fractions must satisfy 0 < train < 1 and 0 <= val < 1")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be below 1")
    if n_subjects == 1:
        return 1, 0, 0
    if n_subjects == 2:
        return 1, 1, 0

    n_train = max(1, int(round(n_subjects * train_frac)))
    n_val = max(1, int(round(n_subjects * val_frac)))
    n_test = n_subjects - n_train - n_val
    while n_test < 1:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError(f"cannot create three nonempty splits from {n_subjects} subjects")
        n_test += 1
    return n_train, n_val, n_test


def subject_split(dataset_id, subject_id, seed=20260725, train_frac=0.6, val_frac=0.2):
    """Split subjects independently inside every dataset.

    Repeating a subject's trial rows cannot change the split because allocation
    operates on unique ``(dataset_id, subject_id)`` keys. Every dataset with at
    least three subjects contributes subjects to train, validation, and test.
    """

    dataset_id = np.asarray(dataset_id, dtype=str)
    subject_id = np.asarray(subject_id, dtype=str)
    if dataset_id.ndim != 1 or subject_id.shape != dataset_id.shape or not dataset_id.size:
        raise ValueError("dataset_id and subject_id must be nonempty aligned vectors")

    rng = np.random.default_rng(seed)
    train_keys, val_keys, test_keys = set(), set(), set()
    for dataset in sorted(set(dataset_id.tolist())):
        subjects = sorted(set(subject_id[dataset_id == dataset].tolist()))
        rng.shuffle(subjects)
        n_train, n_val, _ = _split_counts(len(subjects), train_frac, val_frac)
        dataset_keys = [(dataset, subject) for subject in subjects]
        train_keys.update(dataset_keys[:n_train])
        val_keys.update(dataset_keys[n_train:n_train + n_val])
        test_keys.update(dataset_keys[n_train + n_val:])

    all_keys = set(zip(dataset_id.tolist(), subject_id.tolist()))
    if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
        raise AssertionError("subject split overlap detected")
    if train_keys | val_keys | test_keys != all_keys:
        raise AssertionError("subject split does not cover every subject")
    return train_keys, val_keys, test_keys


def keys_mask(dataset_id, subject_id, keys):
    return np.array([(d, s) in keys for d, s in zip(dataset_id.tolist(), subject_id.tolist())])


def select_dataset_view(archive: dict, requested_dataset_ids=None):
    """Select an in-memory trial view after the full source archive was validated."""

    source_ids = [
        str(dataset).strip()
        for dataset in np.asarray(archive["selected_dataset_ids"], dtype=str).tolist()
    ]
    requested = source_ids if requested_dataset_ids is None else [
        str(dataset).strip() for dataset in requested_dataset_ids
    ]
    if not requested or any(not dataset for dataset in requested):
        raise ValueError("--datasets must contain at least one nonempty dataset ID")
    if len(requested) != len(set(requested)):
        raise ValueError("--datasets contains duplicate dataset IDs")
    unknown = sorted(set(requested) - set(source_ids))
    if unknown:
        raise ValueError(
            f"requested datasets are absent from the validated archive: {unknown}"
        )

    dataset_id = np.asarray(archive["dataset_id"], dtype=str)
    mask = np.isin(dataset_id, requested)
    present = set(dataset_id[mask].tolist())
    if present != set(requested):
        raise ValueError(
            "filtered trials do not contain exactly the requested datasets: "
            f"requested={sorted(requested)}, present={sorted(present)}"
        )

    subject_id = np.asarray(archive["subject_id"], dtype=str)
    trial_subjects = set(zip(dataset_id[mask].tolist(), subject_id[mask].tolist()))
    manifest_subjects = {
        (str(dataset).strip(), str(subject).strip())
        for dataset, subject in np.asarray(
            archive["successful_subject_keys"], dtype=str
        )
        if str(dataset).strip() in set(requested)
    }
    if trial_subjects != manifest_subjects:
        raise ValueError(
            "filtered trial subjects do not match the validated source manifest"
        )
    return mask, requested


def normalize_selection_datasets(requested_dataset_ids, training_dataset_ids):
    """Validate which validation datasets are allowed to choose the best epoch."""

    training_ids = [str(dataset).strip() for dataset in training_dataset_ids]
    requested = training_ids if requested_dataset_ids is None else [
        str(dataset).strip() for dataset in requested_dataset_ids
    ]
    if not requested or any(not dataset for dataset in requested):
        raise ValueError("--selection-datasets must contain nonempty dataset IDs")
    if len(requested) != len(set(requested)):
        raise ValueError("--selection-datasets contains duplicate dataset IDs")
    outside = sorted(set(requested) - set(training_ids))
    if outside:
        raise ValueError(
            "selection datasets must be part of the training view; "
            f"outside={outside}"
        )
    return requested


def _require_exact_subject_keys(actual, expected, label: str):
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{label}: missing={len(missing)} {sorted(missing)[:10]}, "
            f"extra={len(extra)} {sorted(extra)[:10]}"
        )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_split(path: str, archive: dict, training_dataset_ids):
    """Load and strictly verify subject splits from a trusted local checkpoint."""

    if not os.path.isfile(path):
        raise ValueError(f"split checkpoint is not an ordinary file: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("split checkpoint must contain a dictionary")
    schema = int(
        _require_integer_values(
            checkpoint.get("checkpoint_schema_version"),
            "checkpoint_schema_version",
            scalar=True,
        ).item()
    )
    if schema not in (2, 3, 4):
        raise ValueError(f"unsupported split checkpoint schema {schema}")

    archive_schema = int(
        _require_integer_values(
            checkpoint.get("feature_archive_schema_version"),
            "checkpoint feature_archive_schema_version",
            scalar=True,
        ).item()
    )
    if archive_schema != ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            "split checkpoint feature-archive schema does not match the loaded archive"
        )
    archive_build_id = _scalar_text(archive["build_id"], "archive build_id")
    checkpoint_build_id = str(checkpoint.get("feature_archive_build_id", "")).strip()
    if checkpoint_build_id != archive_build_id:
        raise ValueError(
            "split checkpoint feature-archive build ID does not match the loaded archive"
        )

    source_dataset_ids = np.asarray(archive["selected_dataset_ids"], dtype=str).tolist()
    checkpoint_dataset_ids = [
        str(dataset).strip() for dataset in checkpoint.get("selected_dataset_ids", [])
    ]
    if checkpoint_dataset_ids != source_dataset_ids:
        raise ValueError(
            "split checkpoint dataset list does not match the validated source archive"
        )

    field_names = {
        "train": "train_subject_keys",
        "validation": "val_subject_keys",
        "test": "test_subject_keys",
    }
    full_splits = {}
    for split_name, field_name in field_names.items():
        values = np.asarray(checkpoint.get(field_name, []), dtype=str)
        full_splits[split_name] = set(_subject_manifest(values, field_name))
        if not full_splits[split_name]:
            raise ValueError(f"split checkpoint contains an empty {split_name} split")
    if (
        full_splits["train"] & full_splits["validation"]
        or full_splits["train"] & full_splits["test"]
        or full_splits["validation"] & full_splits["test"]
    ):
        raise ValueError("split checkpoint subject sets overlap")

    full_manifest = set(
        _subject_manifest(
            np.asarray(archive["successful_subject_keys"], dtype=str),
            "successful_subject_keys",
        )
    )
    _require_exact_subject_keys(
        set().union(*full_splits.values()),
        full_manifest,
        "split checkpoint full subject union",
    )

    allowed = set(training_dataset_ids)
    restricted = {
        name: {key for key in keys if key[0] in allowed}
        for name, keys in full_splits.items()
    }
    if any(not keys for keys in restricted.values()):
        empty = [name for name, keys in restricted.items() if not keys]
        raise ValueError(f"restricted checkpoint split is empty: {empty}")
    training_manifest = {key for key in full_manifest if key[0] in allowed}
    _require_exact_subject_keys(
        set().union(*restricted.values()),
        training_manifest,
        "restricted split subject union",
    )
    provenance = {
        "mode": "checkpoint",
        "checkpoint_path": os.path.abspath(path),
        "checkpoint_sha256": _sha256_file(path),
        "checkpoint_schema_version": schema,
        "feature_archive_build_id": archive_build_id,
    }
    return restricted["train"], restricted["validation"], restricted["test"], provenance


def build_split_audit(dataset_id, subject_id, labels, split_keys):
    """Return and validate a JSON-safe summary of every dataset in every split."""

    dataset_id = np.asarray(dataset_id, dtype=str)
    subject_id = np.asarray(subject_id, dtype=str)
    labels = _require_integer_values(labels, "split labels")
    if (
        dataset_id.ndim != 1
        or subject_id.shape != dataset_id.shape
        or labels.shape != dataset_id.shape
        or not dataset_id.size
    ):
        raise ValueError("split audit inputs must be nonempty aligned vectors")
    if np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
        raise ValueError(
            f"split labels must be class indices from 0 to {len(CLASS_NAMES) - 1}"
        )

    datasets = sorted(set(dataset_id.tolist()))
    audit = {}
    for split_name in ("train", "validation", "test"):
        keys = split_keys[split_name]
        mask = keys_mask(dataset_id, subject_id, keys)
        class_counts = np.bincount(labels[mask], minlength=len(CLASS_NAMES))
        if mask.sum() and np.any(class_counts == 0):
            absent = [
                CLASS_NAMES[index]
                for index, count in enumerate(class_counts)
                if count == 0
            ]
            raise ValueError(f"{split_name} split has no trials for class(es): {absent}")

        per_dataset = {}
        for dataset in datasets:
            dataset_mask = mask & (dataset_id == dataset)
            dataset_keys = {key for key in keys if key[0] == dataset}
            total_dataset_subjects = len(set(subject_id[dataset_id == dataset].tolist()))
            if total_dataset_subjects >= 3 and not dataset_keys:
                raise ValueError(f"{split_name} split has no subjects from {dataset}")
            counts = np.bincount(labels[dataset_mask], minlength=len(CLASS_NAMES))
            trial_count = int(dataset_mask.sum())
            proportions = (
                (counts / trial_count).astype(float).tolist()
                if trial_count
                else [0.0] * len(CLASS_NAMES)
            )
            per_dataset[dataset] = {
                "subjects": int(len(dataset_keys)),
                "trials": trial_count,
                "class_counts": counts.astype(int).tolist(),
                "class_proportions": proportions,
            }

        audit[split_name] = {
            "subjects": int(len(keys)),
            "trials": int(mask.sum()),
            "class_counts": class_counts.astype(int).tolist(),
            "class_proportions": (
                (class_counts / int(mask.sum())).astype(float).tolist()
                if mask.sum()
                else [0.0] * len(CLASS_NAMES)
            ),
            "per_dataset": per_dataset,
        }
    return audit


def redact_test_label_counts(audit: dict):
    """Hide held-out test labels while validation experiments are being chosen."""

    redacted = copy.deepcopy(audit)
    test = redacted["test"]
    test["class_counts"] = None
    test["class_proportions"] = None
    for row in test["per_dataset"].values():
        row["class_counts"] = None
        row["class_proportions"] = None
    return redacted


def print_split_audit(audit: dict):
    print("\nSubject-separated, per-dataset split audit:")
    for split_name in ("train", "validation", "test"):
        split = audit[split_name]
        classes = "reserved" if split["class_counts"] is None else split["class_counts"]
        print(
            f"  {split_name:<10} subjects={split['subjects']:>3} "
            f"trials={split['trials']:>5} classes={classes}"
        )
        for dataset, row in split["per_dataset"].items():
            row_classes = "reserved" if row["class_counts"] is None else row["class_counts"]
            print(
                f"    {dataset}: subjects={row['subjects']:>3} "
                f"trials={row['trials']:>4} classes={row_classes}"
            )


def base_trial_weights(dataset_id, subject_id, policy="dataset-subject-class"):
    """Assign equal starting mass to subjects, optionally also to datasets."""

    dataset_id = np.asarray(dataset_id, dtype=str)
    subject_id = np.asarray(subject_id, dtype=str)
    if dataset_id.ndim != 1 or subject_id.shape != dataset_id.shape or not dataset_id.size:
        raise ValueError("weight inputs must be nonempty aligned vectors")
    if policy not in ("subject-class", "dataset-subject-class"):
        raise ValueError(f"unknown training-weight policy {policy!r}")

    subject_counts = Counter(zip(dataset_id.tolist(), subject_id.tolist()))
    dataset_subject_counts = Counter(dataset for dataset, _ in subject_counts)
    weights = np.asarray(
        [
            1.0
            / subject_counts[(dataset, subject)]
            / (
                dataset_subject_counts[dataset]
                if policy == "dataset-subject-class"
                else 1.0
            )
            for dataset, subject in zip(dataset_id.tolist(), subject_id.tolist())
        ],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise AssertionError("base training weights must be positive and finite")
    return weights


def training_trial_weights(
    dataset_id,
    subject_id,
    labels,
    policy="dataset-subject-class",
    class_balance_strength=1.0,
):
    """Combine base balance with an adjustable train-only class correction."""

    labels = _require_integer_values(labels, "training labels")
    strength = float(class_balance_strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("class-balance strength must be finite and between 0 and 1")
    base = base_trial_weights(dataset_id, subject_id, policy=policy)
    if labels.shape != base.shape or np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
        raise ValueError("training labels must align and use every class index correctly")
    class_mass = np.bincount(labels, weights=base, minlength=len(CLASS_NAMES))
    if np.any(class_mass <= 0.0):
        absent = [
            CLASS_NAMES[index]
            for index, mass in enumerate(class_mass)
            if mass <= 0.0
        ]
        raise ValueError(f"training split has no weighted mass for class(es): {absent}")
    full_class_correction = base.sum() / (len(CLASS_NAMES) * class_mass)
    if strength == 0.0:
        applied_class_correction = np.ones_like(full_class_correction)
    elif strength == 1.0:
        applied_class_correction = full_class_correction.copy()
    else:
        applied_class_correction = (
            (1.0 - strength) + strength * full_class_correction
        )
    weights = base * applied_class_correction[labels]
    weights /= weights.mean()
    return (
        weights.astype(np.float32),
        applied_class_correction.astype(np.float64),
        full_class_correction.astype(np.float64),
        class_mass.astype(np.float64),
    )


def ordinal_targets_numpy(labels) -> np.ndarray:
    """Encode Low/Moderate/High as cumulative binary rank targets."""

    labels = _require_integer_values(labels, "ordinal labels")
    if labels.ndim != 1 or not labels.size:
        raise ValueError("ordinal labels must be a nonempty vector")
    if np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
        raise ValueError(
            f"ordinal labels must contain class indices in 0..{len(CLASS_NAMES) - 1}"
        )
    ranks = np.arange(1, len(CLASS_NAMES), dtype=np.int64)
    return (labels[:, np.newaxis] >= ranks[np.newaxis, :]).astype(np.int64)


def ordinal_targets_tensor(labels: torch.Tensor, *, dtype=None) -> torch.Tensor:
    """Torch counterpart of :func:`ordinal_targets_numpy` for minibatches."""

    if labels.ndim != 1 or not labels.numel():
        raise ValueError("ordinal labels must be a nonempty vector")
    if torch.any((labels < 0) | (labels >= len(CLASS_NAMES))).item():
        raise ValueError(
            f"ordinal labels must contain class indices in 0..{len(CLASS_NAMES) - 1}"
        )
    ranks = torch.arange(1, len(CLASS_NAMES), device=labels.device)
    target_dtype = torch.float32 if dtype is None else dtype
    return (labels.unsqueeze(1) >= ranks.unsqueeze(0)).to(dtype=target_dtype)


def ordinal_training_loss_weights(
    dataset_id,
    subject_id,
    labels,
    policy="dataset-subject-class",
    class_balance_strength=1.0,
):
    """Build train-only weights for each cumulative binary decision.

    Dataset/subject weights remain the base measure. Positive and negative mass
    is then balanced independently for ``label >= 1`` and ``label >= 2``; this
    avoids applying a nominal three-class correction to the two ordinal losses.
    """

    targets = ordinal_targets_numpy(labels)
    strength = float(class_balance_strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("class-balance strength must be finite and between 0 and 1")
    base = base_trial_weights(dataset_id, subject_id, policy=policy)
    if targets.shape[0] != base.shape[0]:
        raise ValueError("ordinal labels and base training weights must align")

    n_ranks = len(CLASS_NAMES) - 1
    target_mass = np.empty((n_ranks, 2), dtype=np.float64)
    raw_target_counts = np.empty((n_ranks, 2), dtype=np.int64)
    for rank_index in range(n_ranks):
        target_mass[rank_index] = np.bincount(
            targets[:, rank_index], weights=base, minlength=2
        )
        raw_target_counts[rank_index] = np.bincount(
            targets[:, rank_index], minlength=2
        )
    if np.any(target_mass <= 0.0):
        missing = [
            f"{ORDINAL_RANK_SEMANTICS[rank]} target={target}"
            for rank, target in zip(*np.where(target_mass <= 0.0))
        ]
        raise ValueError(
            "training split has no weighted mass for ordinal decision(s): "
            + ", ".join(missing)
        )

    full_target_correction = base.sum() / (2.0 * target_mass)
    if strength == 0.0:
        applied_target_correction = np.ones_like(full_target_correction)
    elif strength == 1.0:
        applied_target_correction = full_target_correction.copy()
    else:
        applied_target_correction = (
            (1.0 - strength) + strength * full_target_correction
        )

    rank_indices = np.arange(n_ranks)[np.newaxis, :]
    loss_weights = base[:, np.newaxis] * applied_target_correction[
        rank_indices, targets
    ]
    loss_weights /= loss_weights.mean()
    final_target_mass = np.stack(
        [
            np.bincount(
                targets[:, rank], weights=loss_weights[:, rank], minlength=2
            )
            for rank in range(n_ranks)
        ]
    )
    return (
        loss_weights.astype(np.float32),
        applied_target_correction.astype(np.float64),
        full_target_correction.astype(np.float64),
        target_mass.astype(np.float64),
        raw_target_counts,
        final_target_mass.astype(np.float64),
    )


def make_training_loader(X, y, weights, *, batch_size: int, seed: int):
    """Create a seeded, shuffled loader that visits each training row once per epoch."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if X.ndim != 3 or y.ndim != 1 or weights.ndim not in (1, 2):
        raise ValueError(
            "training tensors must have shapes X=(N,T,F), y=(N,), "
            "weights=(N,) or (N,R)"
        )
    if not (X.shape[0] == y.shape[0] == weights.shape[0]) or not X.shape[0]:
        raise ValueError("training tensors must be nonempty and row-aligned")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(X, y, weights),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _require_integer_values(value, name: str, *, scalar: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if scalar and raw.size != 1:
        raise ValueError(f"{name} must be one scalar integer, got shape {raw.shape}")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric integers") from exc
    if not np.isfinite(numeric).all() or not np.allclose(
        numeric, np.rint(numeric), atol=0.0, rtol=0.0
    ):
        raise ValueError(f"{name} must contain exact finite integers")
    return np.rint(numeric).astype(np.int64)


def _scalar_text(value, name: str) -> str:
    raw = np.asarray(value)
    if raw.size != 1:
        raise ValueError(f"{name} must contain one string, got shape {raw.shape}")
    text = str(raw.reshape(-1)[0]).strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    return text


def _subject_manifest(values: np.ndarray, name: str) -> list[tuple[str, str]]:
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must have shape (subjects, 2), got {values.shape}")
    rows = [(str(dataset).strip(), str(subject).strip()) for dataset, subject in values]
    if any(not dataset or not subject for dataset, subject in rows):
        raise ValueError(f"{name} contains an empty dataset or subject ID")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{name} contains duplicate subject rows")
    return rows


def _resolve_qc_path(archive_path: str, reference: str) -> str:
    if os.path.isabs(reference):
        return os.path.abspath(reference)
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(archive_path)), reference)
    )


def _atomic_torch_save(payload: dict, path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        prefix=".model_", suffix=".pt.tmp", dir=directory or "."
    )
    os.close(descriptor)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def validate_feature_archive(arrays: dict) -> dict:
    """Validate the complete builder/trainer contract before splitting data."""

    missing = sorted(REQUIRED_ARCHIVE_KEYS - set(arrays))
    if missing:
        raise ValueError(f"feature archive is missing required keys: {missing}")

    schema_version = int(
        _require_integer_values(arrays["schema_version"], "schema_version", scalar=True).item()
    )
    if schema_version != ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported feature archive schema {schema_version}; "
            f"expected {ARCHIVE_SCHEMA_VERSION}"
        )

    build_id = _scalar_text(arrays["build_id"], "build_id")
    archive_complete_raw = np.asarray(arrays["archive_complete"])
    if archive_complete_raw.size != 1:
        raise ValueError("archive_complete must be one boolean value")
    archive_complete = archive_complete_raw.reshape(-1)[0]
    if not isinstance(archive_complete, (bool, np.bool_)):
        raise ValueError("archive_complete must have boolean dtype")
    if not bool(archive_complete):
        raise ValueError("refusing to train from a partial feature archive")
    _scalar_text(arrays["qc_report_relpath"], "qc_report_relpath")
    qc_summary_text = _scalar_text(arrays["qc_summary_json"], "qc_summary_json")
    try:
        qc_summary = json.loads(qc_summary_text)
    except json.JSONDecodeError as exc:
        raise ValueError("qc_summary_json is not valid JSON") from exc
    if not isinstance(qc_summary, dict):
        raise ValueError("qc_summary_json must contain a JSON object")

    selected_datasets = np.asarray(arrays["selected_dataset_ids"], dtype=str)
    expected_subjects = np.asarray(arrays["expected_subject_keys"], dtype=str)
    successful_subjects = np.asarray(arrays["successful_subject_keys"], dtype=str)
    failed_subjects = np.asarray(arrays["failed_subject_keys"], dtype=str)
    if selected_datasets.ndim != 1 or selected_datasets.size == 0:
        raise ValueError("selected_dataset_ids must be a nonempty vector")
    selected_dataset_list = [dataset.strip() for dataset in selected_datasets.tolist()]
    if any(not dataset for dataset in selected_dataset_list) or len(
        selected_dataset_list
    ) != len(set(selected_dataset_list)):
        raise ValueError("selected_dataset_ids must contain unique, nonempty values")
    expected_rows = _subject_manifest(expected_subjects, "expected_subject_keys")
    successful_rows = _subject_manifest(successful_subjects, "successful_subject_keys")
    failed_rows = _subject_manifest(failed_subjects, "failed_subject_keys")
    if failed_rows:
        raise ValueError("complete archive cannot contain failed_subject_keys")
    if set(expected_rows) != set(successful_rows):
        raise ValueError("successful_subject_keys do not match the expected subject manifest")
    if {dataset for dataset, _ in expected_rows} != set(selected_dataset_list):
        raise ValueError("subject manifest does not match selected_dataset_ids")

    relative_power = np.asarray(arrays["relative_power"], dtype=np.float64)
    channel_relative_power = np.asarray(
        arrays["channel_relative_power"], dtype=np.float64
    )
    channel_baseline_power = np.asarray(
        arrays["channel_baseline_power"], dtype=np.float64
    )
    ratings = np.asarray(arrays["ratings"], dtype=np.float64)
    labels = _require_integer_values(arrays["labels"], "labels")
    dataset_id = np.asarray(arrays["dataset_id"], dtype=str)
    subject_id = np.asarray(arrays["subject_id"], dtype=str)
    epoch_index = _require_integer_values(arrays["epoch_index"], "epoch_index")

    if relative_power.ndim != 3 or relative_power.shape[1:] != (len(WINDOWS_S), len(BANDS)):
        raise ValueError(
            f"relative_power must have shape (N, {len(WINDOWS_S)}, {len(BANDS)}), "
            f"got {relative_power.shape}"
        )
    n_trials = relative_power.shape[0]
    if n_trials == 0:
        raise ValueError("feature archive contains zero trials")

    trial_keys = [
        "ratings",
        "labels",
        "subject_id",
        "dataset_id",
        "epoch_index",
        "event_type",
        "laser_power",
        "event_onset_sample",
        "feature_onset_sample",
    ]
    bad_lengths = {
        key: np.asarray(arrays[key]).shape
        for key in trial_keys
        if np.asarray(arrays[key]).shape != (n_trials,)
    }
    if bad_lengths:
        raise ValueError(f"trial-aligned arrays must each have shape ({n_trials},): {bad_lengths}")

    if not np.isfinite(relative_power).all() or np.any(relative_power < 0.0):
        raise ValueError("relative_power must contain only finite, nonnegative values")
    channel_order = np.asarray(arrays["channel_order"], dtype=str)
    if channel_order.ndim != 1 or channel_order.size == 0:
        raise ValueError("channel_order must be a nonempty vector")
    normalized_channels = [channel.strip().upper() for channel in channel_order.tolist()]
    if any(not channel for channel in normalized_channels) or len(normalized_channels) != len(
        set(normalized_channels)
    ):
        raise ValueError("channel_order must contain unique, nonempty labels")
    expected_channel_shape = (n_trials, len(WINDOWS_S), channel_order.size, len(BANDS))
    if channel_relative_power.shape != expected_channel_shape:
        raise ValueError(
            f"channel_relative_power must have shape {expected_channel_shape}, "
            f"got {channel_relative_power.shape}"
        )
    if not np.isfinite(channel_relative_power).all() or np.any(channel_relative_power < 0.0):
        raise ValueError("channel_relative_power must contain finite, nonnegative values")
    expected_baseline_shape = (n_trials, channel_order.size, len(BANDS))
    if channel_baseline_power.shape != expected_baseline_shape:
        raise ValueError(
            f"channel_baseline_power must have shape {expected_baseline_shape}, "
            f"got {channel_baseline_power.shape}"
        )
    if not np.isfinite(channel_baseline_power).all() or np.any(
        channel_baseline_power <= MIN_BASELINE_POWER
    ):
        raise ValueError(
            "channel_baseline_power must contain finite values above "
            f"{MIN_BASELINE_POWER:g}"
        )
    baseline_weights = channel_baseline_power[:, np.newaxis, :, :]
    expected_average = np.sum(
        channel_relative_power * baseline_weights, axis=2
    ) / np.sum(baseline_weights, axis=2)
    if not np.allclose(relative_power, expected_average):
        raise ValueError(
            "relative_power does not match the baseline-weighted channel calculation"
        )
    if not np.isfinite(ratings).all() or np.any((ratings < 0.0) | (ratings > 10.0)):
        raise ValueError("ratings must contain only finite values in [0, 10]")
    if np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
        raise ValueError(f"labels must be integers in 0..{len(CLASS_NAMES) - 1}")

    expected_labels = np.asarray([bin_rating(rating) for rating in ratings], dtype=np.int64)
    if not np.array_equal(labels, expected_labels):
        mismatch = np.flatnonzero(labels != expected_labels)[:10].tolist()
        raise ValueError(f"labels do not match rating-bin rules at trial indices {mismatch}")

    if np.any(np.char.str_len(np.char.strip(dataset_id)) == 0):
        raise ValueError("dataset_id contains an empty value")
    if np.any(np.char.str_len(np.char.strip(subject_id)) == 0):
        raise ValueError("subject_id contains an empty value")
    if np.any(epoch_index < 1):
        raise ValueError("epoch_index must use positive, one-based identifiers")
    identities = list(zip(dataset_id.tolist(), subject_id.tolist(), epoch_index.tolist()))
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate (dataset_id, subject_id, epoch_index) trial identity")
    trial_subject_rows = {
        (dataset.strip(), subject.strip())
        for dataset, subject in zip(dataset_id.tolist(), subject_id.tolist())
    }
    if trial_subject_rows != set(successful_rows):
        raise ValueError("trial subjects do not match successful_subject_keys")

    event_type = np.asarray(arrays["event_type"], dtype=str)
    laser_power = np.asarray(arrays["laser_power"], dtype=np.float64)
    event_onset = np.asarray(arrays["event_onset_sample"], dtype=np.float64)
    feature_onset = _require_integer_values(
        arrays["feature_onset_sample"], "feature_onset_sample"
    )
    if np.any(np.char.str_len(np.char.strip(event_type)) == 0):
        raise ValueError("event_type contains an empty value")
    if np.isinf(laser_power).any() or np.any(laser_power[np.isfinite(laser_power)] < 0.0):
        raise ValueError("laser_power must be NaN or a finite nonnegative value")
    if not np.isfinite(event_onset).all() or np.any(event_onset < 0.0):
        raise ValueError("event_onset_sample must contain finite, nonnegative values")
    if np.any(feature_onset < 0):
        raise ValueError("feature_onset_sample must contain nonnegative integers")
    if np.any(np.abs(event_onset - feature_onset) > 0.5):
        raise ValueError("feature_onset_sample must be the nearest sample to event_onset_sample")

    band_order = np.asarray(arrays["band_order"], dtype=str).tolist()
    expected_bands = [band[0] for band in BANDS]
    if band_order != expected_bands:
        raise ValueError(f"band_order must be {expected_bands}, got {band_order}")
    baseline_window = np.asarray(arrays["baseline_window_s"], dtype=np.float64)
    post_windows = np.asarray(arrays["post_windows_s"], dtype=np.float64)
    if baseline_window.shape != (2,) or not np.allclose(
        baseline_window, np.asarray(BASELINE_WINDOW_S, dtype=np.float64)
    ):
        raise ValueError(
            f"baseline_window_s must be {list(BASELINE_WINDOW_S)}, got {baseline_window.tolist()}"
        )
    if post_windows.shape != (len(WINDOWS_S), 2) or not np.allclose(
        post_windows, np.asarray(WINDOWS_S, dtype=np.float64)
    ):
        raise ValueError(f"post_windows_s must be {WINDOWS_S}, got {post_windows.tolist()}")

    if qc_summary.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise ValueError("embedded QC schema_version does not match the archive")
    if qc_summary.get("build_id") != build_id:
        raise ValueError("embedded QC build_id does not match the archive")
    if qc_summary.get("build_status") != "complete" or not qc_summary.get(
        "archive_written"
    ):
        raise ValueError("embedded QC report does not describe a complete archive")
    if qc_summary.get("subject_failures"):
        raise ValueError("embedded QC report contains subject failures")
    if qc_summary.get("selected_datasets") != selected_dataset_list:
        raise ValueError("embedded QC selected datasets do not match the archive")
    if qc_summary.get("channels") != channel_order.tolist():
        raise ValueError("embedded QC channel order does not match the archive")
    if qc_summary.get("expected_subject_keys") != [list(row) for row in expected_rows]:
        raise ValueError("embedded QC expected-subject manifest does not match the archive")
    if qc_summary.get("successful_subject_keys") != [list(row) for row in successful_rows]:
        raise ValueError("embedded QC successful-subject manifest does not match the archive")

    qc_subjects = qc_summary.get("subjects")
    if not isinstance(qc_subjects, list):
        raise ValueError("embedded QC subjects must be a list")
    actual_trial_counts = {}
    actual_accepted_epochs = {}
    for dataset, subject, epoch in zip(
        dataset_id.tolist(), subject_id.tolist(), epoch_index.tolist()
    ):
        normalized_row = (dataset.strip(), subject.strip())
        actual_trial_counts[normalized_row] = actual_trial_counts.get(normalized_row, 0) + 1
        actual_accepted_epochs.setdefault(normalized_row, set()).add(int(epoch))

    qc_subject_rows = []
    for index, subject_qc in enumerate(qc_subjects):
        if not isinstance(subject_qc, dict):
            raise ValueError(f"embedded QC subject {index} must be an object")
        row = (
            str(subject_qc.get("dataset_id", "")).strip(),
            str(subject_qc.get("subject_id", "")).strip(),
        )
        if not row[0] or not row[1]:
            raise ValueError(f"embedded QC subject {index} has an empty identity")
        qc_subject_rows.append(row)

        input_trials = int(
            _require_integer_values(
                subject_qc.get("input_trials"),
                f"embedded QC subject {index} input_trials",
                scalar=True,
            ).item()
        )
        accepted_trials = int(
            _require_integer_values(
                subject_qc.get("accepted_trials"),
                f"embedded QC subject {index} accepted_trials",
                scalar=True,
            ).item()
        )
        rejected_trials = int(
            _require_integer_values(
                subject_qc.get("rejected_trials"),
                f"embedded QC subject {index} rejected_trials",
                scalar=True,
            ).item()
        )
        if min(input_trials, accepted_trials, rejected_trials) < 0:
            raise ValueError(f"embedded QC subject {index} has a negative trial count")
        if input_trials != accepted_trials + rejected_trials:
            raise ValueError(f"embedded QC subject {index} trial counts do not add up")
        if accepted_trials != actual_trial_counts.get(row, 0):
            raise ValueError(
                f"embedded QC accepted count does not match archive trials for {row}"
            )

        rejections = subject_qc.get("rejections")
        if not isinstance(rejections, dict):
            raise ValueError(f"embedded QC subject {index} rejections must be an object")
        rejection_total = 0
        for reason, count in rejections.items():
            if not str(reason).strip():
                raise ValueError(f"embedded QC subject {index} has an empty rejection reason")
            count_value = int(
                _require_integer_values(
                    count,
                    f"embedded QC subject {index} rejection count",
                    scalar=True,
                ).item()
            )
            if count_value < 0:
                raise ValueError(f"embedded QC subject {index} has a negative rejection count")
            rejection_total += count_value
        if rejection_total != rejected_trials:
            raise ValueError(
                f"embedded QC rejection counts do not match rejected_trials for {row}"
            )

        rejected_details = subject_qc.get("rejected_trial_details")
        if not isinstance(rejected_details, list):
            raise ValueError(
                f"embedded QC subject {index} rejected_trial_details must be a list"
            )
        rejected_epochs = set()
        detail_rejections = {}
        for detail_index, detail in enumerate(rejected_details):
            if not isinstance(detail, dict):
                raise ValueError(
                    f"embedded QC subject {index} rejected detail {detail_index} "
                    "must be an object"
                )
            rejected_epoch = int(
                _require_integer_values(
                    detail.get("epoch_index"),
                    f"embedded QC subject {index} rejected epoch",
                    scalar=True,
                ).item()
            )
            reason = str(detail.get("reason", "")).strip()
            if not 1 <= rejected_epoch <= input_trials:
                raise ValueError(
                    f"embedded QC subject {index} rejected epoch is outside 1..{input_trials}"
                )
            if rejected_epoch in rejected_epochs:
                raise ValueError(
                    f"embedded QC subject {index} repeats rejected epoch {rejected_epoch}"
                )
            if not reason:
                raise ValueError(
                    f"embedded QC subject {index} rejected detail has an empty reason"
                )
            rejected_epochs.add(rejected_epoch)
            detail_rejections[reason] = detail_rejections.get(reason, 0) + 1
        if len(rejected_epochs) != rejected_trials:
            raise ValueError(
                f"embedded QC rejected details do not match rejected_trials for {row}"
            )
        if detail_rejections != {
            str(reason): int(count) for reason, count in rejections.items()
        }:
            raise ValueError(
                f"embedded QC rejected details do not match rejection counts for {row}"
            )
        accepted_epochs = actual_accepted_epochs.get(row, set())
        if accepted_epochs & rejected_epochs:
            raise ValueError(f"embedded QC accepts and rejects the same epoch for {row}")
        if accepted_epochs | rejected_epochs != set(range(1, input_trials + 1)):
            raise ValueError(
                f"embedded QC does not account for every source epoch for {row}"
            )

    if len(qc_subject_rows) != len(set(qc_subject_rows)):
        raise ValueError("embedded QC contains duplicate subject rows")
    if set(qc_subject_rows) != set(successful_rows):
        raise ValueError("embedded QC subjects do not match successful_subject_keys")

    return {
        key: np.array(value, copy=True)
        for key, value in arrays.items()
    }


def load_feature_archive(path: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validated = validate_feature_archive(arrays)

    qc_relpath = _scalar_text(validated["qc_report_relpath"], "qc_report_relpath")
    qc_path = _resolve_qc_path(path, qc_relpath)
    if os.path.isfile(qc_path):
        try:
            with open(qc_path, "r", encoding="utf-8") as handle:
                external_qc = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"linked QC report cannot be read ({qc_path}): {exc}; "
                "using the archive's verified embedded QC copy",
                RuntimeWarning,
            )
        else:
            archive_build_id = _scalar_text(validated["build_id"], "build_id")
            if external_qc.get("build_id") != archive_build_id:
                warnings.warn(
                    f"linked QC report does not match archive build {archive_build_id}; "
                    "using the archive's verified embedded QC copy",
                    RuntimeWarning,
                )
    else:
        warnings.warn(
            f"linked QC report is missing ({qc_path}); using the archive's "
            "verified embedded QC copy",
            RuntimeWarning,
        )
    return validated


def load_rich_features(path: str, archive: dict) -> tuple[np.ndarray, list[str]]:
    """Attach rebuilt rich features to the already-validated archive, row by row.

    The rich archive deliberately does not carry the schema-1 QC contract; that
    contract stays anchored to the pooled archive, which is still loaded and fully
    validated. Rich features are matched onto it by
    ``(dataset_id, subject_id, epoch_index)`` and every trial must match exactly
    once in both directions, so the split, the QC guarantees, and the protected
    test set all continue to apply unchanged.
    """

    with np.load(path, allow_pickle=False) as handle:
        rich = {key: handle[key] for key in handle.files}
    for required in ("features", "feature_names", "dataset_id", "subject_id", "epoch_index"):
        if required not in rich:
            raise ValueError(f"rich feature archive is missing '{required}'")

    def keys_of(source):
        return [
            f"{d}||{s}||{int(e)}"
            for d, s, e in zip(
                np.asarray(source["dataset_id"], dtype=str).tolist(),
                np.asarray(source["subject_id"], dtype=str).tolist(),
                _require_integer_values(source["epoch_index"], "epoch_index").tolist(),
            )
        ]

    rich_keys = keys_of(rich)
    archive_keys = keys_of(archive)
    if len(set(rich_keys)) != len(rich_keys):
        raise ValueError("rich feature archive contains duplicate trial identities")
    lookup = {key: index for index, key in enumerate(rich_keys)}
    missing = [key for key in archive_keys if key not in lookup]
    if missing:
        raise ValueError(
            f"rich features are missing {len(missing)} archive trials, "
            f"e.g. {missing[:5]}"
        )
    if len(rich_keys) != len(archive_keys):
        raise ValueError(
            f"rich archive has {len(rich_keys)} trials but the validated archive "
            f"has {len(archive_keys)}; they must describe the same trials"
        )

    order = np.asarray([lookup[key] for key in archive_keys], dtype=np.int64)
    features = np.asarray(rich["features"], dtype=np.float64)[order]
    if not np.isfinite(features).all():
        raise ValueError("rich features contain nonfinite values")
    names = np.asarray(rich["feature_names"], dtype=str).tolist()
    sequence, step_names = rich_lstm_view(features, names)
    return sequence, step_names


def model_feature_view(
    archive: dict, feature_mode: str, rich_features_path: str = None
) -> tuple[np.ndarray, list[str]]:
    """Select model inputs while keeping flatten order explicit and testable."""

    if feature_mode == "rich":
        if not rich_features_path:
            raise ValueError("feature mode 'rich' requires --rich-features")
        return load_rich_features(rich_features_path, archive)
    if feature_mode == "channel-average":
        return (
            archive["relative_power"].astype(np.float64),
            archive["band_order"].astype(str).tolist(),
        )
    if feature_mode != "per-channel":
        raise ValueError(f"unknown feature mode {feature_mode!r}")
    channel_power = archive["channel_relative_power"].astype(np.float64)
    model_power = channel_power.reshape(channel_power.shape[0], channel_power.shape[1], -1)
    feature_order = [
        f"{channel}:{band}"
        for channel in archive["channel_order"].astype(str).tolist()
        for band in archive["band_order"].astype(str).tolist()
    ]
    return model_power, feature_order


def decode_task_logits(logits: torch.Tensor, task_mode: str) -> torch.Tensor:
    """Decode categorical argmax or cumulative ordinal threshold crossings."""

    if logits.ndim != 2:
        raise ValueError("model logits must be a two-dimensional tensor")
    if task_mode == "categorical":
        if logits.shape[1] != len(CLASS_NAMES):
            raise ValueError(
                f"categorical logits must have {len(CLASS_NAMES)} columns"
            )
        return logits.argmax(dim=1)
    if task_mode == "ordinal":
        if logits.shape[1] != len(CLASS_NAMES) - 1:
            raise ValueError(
                f"ordinal logits must have {len(CLASS_NAMES) - 1} columns"
            )
        return (logits > 0.0).sum(dim=1).to(torch.int64)
    raise ValueError(f"unknown task mode {task_mode!r}")


def model_head_provenance(model: PainLSTM) -> dict:
    """Return a JSON-safe, explicit description of the trained output head."""

    if model.task_mode == "categorical":
        return {
            "type": "categorical_linear",
            "output_logit_count": len(CLASS_NAMES),
            "logit_semantics": [
                f"unnormalized score for class {index} ({name})"
                for index, name in enumerate(CLASS_NAMES)
            ],
            "decoder": "argmax(logits)",
            "rank_consistent": False,
        }
    thresholds = model.ordinal_thresholds().detach().cpu().numpy()
    return {
        "type": "monotonic_cumulative_ordinal",
        "output_logit_count": len(ORDINAL_RANK_SEMANTICS),
        "logit_semantics": ORDINAL_RANK_SEMANTICS.copy(),
        "target_encoding": {
            "Low": [0, 0],
            "Moderate": [1, 0],
            "High": [1, 1],
        },
        "decoder": "count(logit > 0)",
        "rank_consistent": True,
        "shared_score": "linear(hidden_size, 1, bias=False)",
        "threshold_parameterization": (
            "theta[0]=first; theta[1]=first+softplus(gap_unconstrained)"
        ),
        "learned_thresholds": thresholds.astype(float).tolist(),
        "learned_threshold_gap": float(thresholds[1] - thresholds[0]),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, class_names=CLASS_NAMES):
    """Confusion matrix + per-class precision/recall/F1 + balanced accuracy.
    Plain accuracy alone hides class imbalance (see chat notes: 41% of all
    trials are "Moderate", so a model that always says "Moderate" scores 41%
    without learning anything) -- these numbers don't hide that.
    """
    y_true = _require_integer_values(y_true, "y_true")
    y_pred = _require_integer_values(y_pred, "y_pred")
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or y_true.size == 0:
        raise ValueError(
            f"y_true/y_pred must be nonempty aligned vectors, got {y_true.shape}/{y_pred.shape}"
        )
    n = len(class_names)
    if np.any((y_true < 0) | (y_true >= n)) or np.any((y_pred < 0) | (y_pred >= n)):
        raise ValueError(f"evaluation labels and predictions must be in 0..{n - 1}")
    cm = np.zeros((n, n), dtype=int)  # rows = true label, cols = predicted label
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append({"class": class_names[c], "precision": precision,
                           "recall": recall, "f1": f1, "support": int(support)})

    supported_recalls = [row["recall"] for row in per_class if row["support"] > 0]
    balanced_acc = float(np.mean(supported_recalls))
    overall_acc = float(np.trace(cm) / cm.sum())
    return cm, per_class, overall_acc, balanced_acc


def _ordinal_metrics_from_confusion(cm: np.ndarray) -> dict:
    """Compute distance-aware metrics for a square ordered-class matrix."""

    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] < 2:
        raise ValueError("ordinal confusion matrix must be square with at least two classes")
    total = float(cm.sum())
    if total <= 0.0:
        raise ValueError("ordinal confusion matrix must contain observations")
    indices = np.arange(cm.shape[0], dtype=np.float64)
    distance = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    class_index_mae = float(np.sum(cm * distance) / total)
    severe_error_rate = float(np.sum(cm[distance == cm.shape[0] - 1]) / total)

    quadratic_weights = (distance / (cm.shape[0] - 1)) ** 2
    observed_disagreement = float(np.sum(quadratic_weights * cm) / total)
    expected_probabilities = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / (total ** 2)
    expected_disagreement = float(np.sum(quadratic_weights * expected_probabilities))
    if expected_disagreement == 0.0:
        quadratic_weighted_kappa = 1.0 if observed_disagreement == 0.0 else 0.0
    else:
        quadratic_weighted_kappa = 1.0 - (
            observed_disagreement / expected_disagreement
        )
    return {
        "class_index_mae": class_index_mae,
        "severe_low_high_error_rate": severe_error_rate,
        "quadratic_weighted_kappa": float(quadratic_weighted_kappa),
    }


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Distance-aware metrics for the ordered Low/Moderate/High labels."""

    cm, _, _, _ = evaluate(y_true, y_pred)
    return _ordinal_metrics_from_confusion(cm)


def _evaluation_payload(y_true: np.ndarray, y_pred: np.ndarray):
    cm, per_class, overall_acc, balanced_acc = evaluate(y_true, y_pred)
    supported_f1 = [row["f1"] for row in per_class if row["support"] > 0]
    payload = {
        "accuracy": float(overall_acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(np.mean(supported_f1)),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {
                "class": row["class"],
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "support": int(row["support"]),
            }
            for row in per_class
        ],
    }
    payload.update(_ordinal_metrics_from_confusion(cm))
    return payload


def evaluation_breakdown(y_true, y_pred, dataset_id, subject_id):
    """Report pooled, per-dataset, and subject-macro performance."""

    y_true = _require_integer_values(y_true, "y_true")
    y_pred = _require_integer_values(y_pred, "y_pred")
    dataset_id = np.asarray(dataset_id, dtype=str)
    subject_id = np.asarray(subject_id, dtype=str)
    if (
        y_true.ndim != 1
        or y_pred.shape != y_true.shape
        or dataset_id.shape != y_true.shape
        or subject_id.shape != y_true.shape
        or not y_true.size
    ):
        raise ValueError("evaluation breakdown inputs must be nonempty aligned vectors")

    per_dataset = {}
    for dataset in sorted(set(dataset_id.tolist())):
        mask = dataset_id == dataset
        payload = _evaluation_payload(y_true[mask], y_pred[mask])
        payload["subjects"] = int(len(set(subject_id[mask].tolist())))
        payload["trials"] = int(mask.sum())
        per_dataset[dataset] = payload

    subject_accuracies = []
    for dataset, subject in sorted(set(zip(dataset_id.tolist(), subject_id.tolist()))):
        mask = (dataset_id == dataset) & (subject_id == subject)
        subject_accuracies.append(float(np.mean(y_true[mask] == y_pred[mask])))

    pooled = _evaluation_payload(y_true, y_pred)
    return {
        "pooled": pooled,
        "per_dataset": per_dataset,
        "dataset_macro_accuracy": float(
            np.mean([row["accuracy"] for row in per_dataset.values()])
        ),
        "dataset_macro_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in per_dataset.values()])
        ),
        "dataset_macro_f1": float(
            np.mean([row["macro_f1"] for row in per_dataset.values()])
        ),
        "dataset_macro_class_index_mae": float(
            np.mean([row["class_index_mae"] for row in per_dataset.values()])
        ),
        "dataset_macro_severe_low_high_error_rate": float(
            np.mean(
                [row["severe_low_high_error_rate"] for row in per_dataset.values()]
            )
        ),
        "dataset_macro_quadratic_weighted_kappa": float(
            np.mean(
                [row["quadratic_weighted_kappa"] for row in per_dataset.values()]
            )
        ),
        "subject_macro_accuracy": float(np.mean(subject_accuracies)),
    }


def print_evaluation_breakdown(report: dict, label: str):
    pooled = report["pooled"]
    print(f"\n=== {label}: dataset summary ===")
    print(
        "dataset     subjects  trials  accuracy  balanced  macro-F1"
    )
    for dataset, row in report["per_dataset"].items():
        print(
            f"{dataset:<12}{row['subjects']:>8}{row['trials']:>8}"
            f"{row['accuracy']:>10.3f}{row['balanced_accuracy']:>10.3f}"
            f"{row['macro_f1']:>10.3f}"
        )
    print(
        f"Pooled trial accuracy:           {pooled['accuracy']:.3f}\n"
        f"Pooled balanced accuracy:        {pooled['balanced_accuracy']:.3f}\n"
        f"Dataset-macro balanced accuracy: "
        f"{report['dataset_macro_balanced_accuracy']:.3f}\n"
        f"Dataset-macro F1:                {report['dataset_macro_f1']:.3f}\n"
        f"Subject-macro accuracy:          {report['subject_macro_accuracy']:.3f}\n"
        f"Class-index MAE:                 {pooled['class_index_mae']:.3f}\n"
        f"Severe Low<->High error rate:    "
        f"{pooled['severe_low_high_error_rate']:.3f}\n"
        f"Quadratic weighted kappa:        "
        f"{pooled['quadratic_weighted_kappa']:.3f}"
    )


def rating_boundary_breakdown(ratings, y_true, y_pred) -> dict:
    """Summarize Moderate/High behavior in one-point rating intervals."""

    ratings = np.asarray(ratings, dtype=np.float64)
    y_true = _require_integer_values(y_true, "rating-boundary true labels")
    y_pred = _require_integer_values(y_pred, "rating-boundary predictions")
    if (
        ratings.ndim != 1
        or y_true.shape != ratings.shape
        or y_pred.shape != ratings.shape
        or not ratings.size
        or not np.isfinite(ratings).all()
    ):
        raise ValueError("rating-boundary inputs must be finite nonempty aligned vectors")
    if np.any((ratings < 0.0) | (ratings > 10.0)):
        raise ValueError("rating-boundary ratings must be in 0..10")

    rows = []
    for lower in range(4, 10):
        upper = lower + 1
        upper_inclusive = upper == 10
        mask = (ratings >= lower) & (
            (ratings <= upper) if upper_inclusive else (ratings < upper)
        )
        interval = f"[{lower},{upper}{']' if upper_inclusive else ')'}"
        if np.any(mask):
            payload = _evaluation_payload(y_true[mask], y_pred[mask])
            cm = np.asarray(payload["confusion_matrix"], dtype=np.int64)
            moderate_support = int(cm[1].sum())
            high_support = int(cm[2].sum())
            row = {
                "interval": interval,
                "lower_inclusive": float(lower),
                "upper": float(upper),
                "upper_inclusive": upper_inclusive,
                "trials": int(mask.sum()),
                "accuracy": payload["accuracy"],
                "true_class_counts": cm.sum(axis=1).astype(int).tolist(),
                "predicted_class_counts": cm.sum(axis=0).astype(int).tolist(),
                "confusion_matrix": cm.tolist(),
                "moderate_predicted_high_count": int(cm[1, 2]),
                "moderate_predicted_high_rate": (
                    None
                    if moderate_support == 0
                    else float(cm[1, 2] / moderate_support)
                ),
                "high_predicted_moderate_count": int(cm[2, 1]),
                "high_predicted_moderate_rate": (
                    None if high_support == 0 else float(cm[2, 1] / high_support)
                ),
            }
        else:
            row = {
                "interval": interval,
                "lower_inclusive": float(lower),
                "upper": float(upper),
                "upper_inclusive": upper_inclusive,
                "trials": 0,
                "accuracy": None,
                "true_class_counts": [0] * len(CLASS_NAMES),
                "predicted_class_counts": [0] * len(CLASS_NAMES),
                "confusion_matrix": [
                    [0] * len(CLASS_NAMES) for _ in range(len(CLASS_NAMES))
                ],
                "moderate_predicted_high_count": 0,
                "moderate_predicted_high_rate": None,
                "high_predicted_moderate_count": 0,
                "high_predicted_moderate_rate": None,
            }
        rows.append(row)
    return {
        "rating_scale": [0.0, 10.0],
        "focus": "Moderate/High boundary at rating 7",
        "interval_semantics": "left-closed/right-open, except [9,10] is closed",
        "rows": rows,
    }


def print_rating_boundary_breakdown(report: dict, label: str):
    print(f"\n=== {label}: rating-boundary summary ===")
    print("rating      trials  accuracy  Moderate->High  High->Moderate")
    for row in report["rows"]:
        accuracy = "n/a" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
        moderate_high = row["moderate_predicted_high_rate"]
        high_moderate = row["high_predicted_moderate_rate"]
        print(
            f"{row['interval']:<11}{row['trials']:>7}  {accuracy:>8}  "
            f"{('n/a' if moderate_high is None else f'{moderate_high:.3f}'):>14}  "
            f"{('n/a' if high_moderate is None else f'{high_moderate:.3f}'):>14}"
        )


def prediction_payload(dataset_id, subject_id, epoch_index, ratings, y_true, y_pred):
    """Build an exact, row-aligned prediction audit payload."""

    dataset_id = np.asarray(dataset_id, dtype=str)
    subject_id = np.asarray(subject_id, dtype=str)
    epoch_index = _require_integer_values(epoch_index, "prediction epoch_index")
    ratings = np.asarray(ratings, dtype=np.float64)
    y_true = _require_integer_values(y_true, "prediction true labels")
    y_pred = _require_integer_values(y_pred, "prediction predicted labels")
    shape = dataset_id.shape
    if (
        dataset_id.ndim != 1
        or not dataset_id.size
        or subject_id.shape != shape
        or epoch_index.shape != shape
        or ratings.shape != shape
        or y_true.shape != shape
        or y_pred.shape != shape
        or not np.isfinite(ratings).all()
    ):
        raise ValueError("prediction fields must be finite nonempty aligned vectors")
    return {
        "dataset_id": dataset_id.tolist(),
        "subject_id": subject_id.tolist(),
        "epoch_index": epoch_index.astype(int).tolist(),
        "rating": ratings.astype(float).tolist(),
        "true_label": y_true.astype(int).tolist(),
        "predicted_label": y_pred.astype(int).tolist(),
    }


def predict_in_batches(model, X, batch_size: int, task_mode=None) -> np.ndarray:
    if X is None or X.ndim != 3 or not X.shape[0]:
        raise ValueError("prediction tensor must be a nonempty (N,T,F) tensor")
    if batch_size <= 0:
        raise ValueError("prediction batch size must be positive")
    resolved_task_mode = (
        getattr(model, "task_mode", "categorical") if task_mode is None else task_mode
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            logits = model(X[start:start + batch_size])
            predictions.append(decode_task_logits(logits, resolved_task_mode).cpu())
    return torch.cat(predictions).numpy()


def print_evaluation(y_true: np.ndarray, y_pred: np.ndarray, label: str):
    cm, per_class, overall_acc, balanced_acc = evaluate(y_true, y_pred)

    print(f"\n=== {label} evaluation ===")
    print(f"Overall accuracy:  {overall_acc:.3f}")
    print(f"Balanced accuracy: {balanced_acc:.3f}  "
          f"(average of each class's own recall -- fair even if classes are uneven sizes)")

    header = "Confusion matrix (rows = real label, columns = model's guess):"
    print(header)
    col_header = "                " + "".join(f"{c:>10}" for c in CLASS_NAMES)
    print(col_header)
    for i, name in enumerate(CLASS_NAMES):
        row = "".join(f"{cm[i, j]:>10d}" for j in range(len(CLASS_NAMES)))
        print(f"true {name:<11}{row}")

    print(f"\n{'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for row in per_class:
        print(f"{row['class']:<10}{row['precision']:>10.3f}{row['recall']:>10.3f}"
              f"{row['f1']:>10.3f}{row['support']:>10d}")
    return cm, per_class, overall_acc, balanced_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/all9_full/all9_features.npz")
    ap.add_argument(
        "--datasets",
        nargs="+",
        help="Train on only these datasets from the fully validated source archive.",
    )
    ap.add_argument(
        "--selection-datasets",
        nargs="+",
        help=(
            "Validation datasets that choose the best epoch; defaults to all "
            "datasets in the training view."
        ),
    )
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    split_group = ap.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split-seed",
        "--seed",
        dest="split_seed",
        type=int,
        help=(
            "Seed for a new per-dataset subject split (--seed is a compatibility alias); "
            "defaults to 20260725 when no checkpoint split is supplied."
        ),
    )
    split_group.add_argument(
        "--split-from-checkpoint",
        help="Reuse the exact subject split stored in this trusted local checkpoint.",
    )
    ap.add_argument("--model-seed", type=int, default=20260726)
    ap.add_argument("--loader-seed", type=int, default=20260727)
    ap.add_argument(
        "--training-weight-policy",
        choices=("subject-class", "dataset-subject-class"),
        default="dataset-subject-class",
        help=(
            "subject-class gives every subject equal starting mass; "
            "dataset-subject-class also gives every dataset equal starting mass"
        ),
    )
    ap.add_argument(
        "--class-balance-strength",
        type=float,
        default=1.0,
        help="0 disables class correction, 1 uses full correction, fractions interpolate.",
    )
    ap.add_argument(
        "--task-mode",
        choices=("categorical", "ordinal"),
        default="categorical",
        help=(
            "categorical uses the historical three-logit cross-entropy head; "
            "ordinal uses two monotonic cumulative logits for label >= 1 and label >= 2"
        ),
    )
    ap.add_argument(
        "--evaluation-mode",
        choices=("validation-only", "final-test"),
        default="validation-only",
        help=(
            "validation-only reserves test labels/predictions while tuning; "
            "final-test evaluates the locked test split."
        ),
    )
    ap.add_argument(
        "--feature-mode",
        choices=("channel-average", "per-channel", "rich"),
        default="channel-average",
        help=(
            "channel-average keeps the current 3 features per step; per-channel "
            "uses every channel-band pair separately; rich uses the rebuilt "
            "evoked-potential/five-band/coupling features and needs --rich-features"
        ),
    )
    ap.add_argument(
        "--rich-features",
        help=(
            "Path to a rich feature archive from build_rich_pilot.py. Required by "
            "--feature-mode rich; matched onto the validated archive trial by trial."
        ),
    )
    ap.add_argument(
        "--save",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "outputs",
            "all9_full",
            "all9_pain_lstm.pt",
        ),
        help="Where to save the trained weights + everything needed to reuse them.",
    )
    args = ap.parse_args()

    if args.epochs <= 0:
        ap.error("--epochs must be positive")
    if args.hidden <= 0:
        ap.error("--hidden must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0.0:
        ap.error("--lr must be positive and finite")
    if args.patience <= 0:
        ap.error("--patience must be positive")
    if args.batch_size <= 0:
        ap.error("--batch-size must be positive")
    if not np.isfinite(args.gradient_clip) or args.gradient_clip <= 0.0:
        ap.error("--gradient-clip must be positive and finite")
    if (
        not np.isfinite(args.class_balance_strength)
        or not 0.0 <= args.class_balance_strength <= 1.0
    ):
        ap.error("--class-balance-strength must be finite and between 0 and 1")

    if os.path.normcase(os.path.abspath(args.data)) == os.path.normcase(
        os.path.abspath(args.save)
    ):
        ap.error("--data and --save must be different files")
    if args.split_from_checkpoint and os.path.normcase(
        os.path.abspath(args.split_from_checkpoint)
    ) == os.path.normcase(os.path.abspath(args.save)):
        ap.error("--save must not overwrite --split-from-checkpoint")

    torch.manual_seed(args.model_seed)
    torch.use_deterministic_algorithms(True)

    archive = load_feature_archive(args.data)
    qc_path = _resolve_qc_path(
        args.data, _scalar_text(archive["qc_report_relpath"], "qc_report_relpath")
    )
    if os.path.normcase(os.path.abspath(args.save)) == os.path.normcase(qc_path):
        ap.error("--save must not overwrite the feature archive's QC report")
    dataset_mask, training_dataset_ids = select_dataset_view(archive, args.datasets)
    selection_dataset_ids = normalize_selection_datasets(
        args.selection_datasets, training_dataset_ids
    )
    full_model_power, feature_order = model_feature_view(
        archive, args.feature_mode, rich_features_path=args.rich_features
    )
    model_power = full_model_power[dataset_mask]
    y = archive["labels"].astype(np.int64)[dataset_mask]
    ratings = archive["ratings"].astype(np.float64)[dataset_mask]
    dataset_id = archive["dataset_id"].astype(str)[dataset_mask]
    subject_id = archive["subject_id"].astype(str)[dataset_mask]
    epoch_index = archive["epoch_index"].astype(np.int64)[dataset_mask]

    if args.split_from_checkpoint:
        train_keys, val_keys, test_keys, split_source = load_checkpoint_split(
            args.split_from_checkpoint, archive, training_dataset_ids
        )
        effective_split_seed = None
    else:
        effective_split_seed = 20260725 if args.split_seed is None else args.split_seed
        train_keys, val_keys, test_keys = subject_split(
            dataset_id, subject_id, seed=effective_split_seed
        )
        split_source = {"mode": "seed", "seed": int(effective_split_seed)}
    train_mask = keys_mask(dataset_id, subject_id, train_keys)
    val_mask = keys_mask(dataset_id, subject_id, val_keys)
    test_mask = keys_mask(dataset_id, subject_id, test_keys)
    full_split_audit = build_split_audit(
        dataset_id,
        subject_id,
        y,
        {"train": train_keys, "validation": val_keys, "test": test_keys},
    )
    split_audit = (
        redact_test_label_counts(full_split_audit)
        if args.evaluation_mode == "validation-only"
        else full_split_audit
    )
    print_split_audit(split_audit)
    print(f"Training datasets: {training_dataset_ids}")
    print(f"Best-epoch selection datasets: {selection_dataset_ids}")
    print(f"Split source: {split_source['mode']}")

    # Fit the 0-255 scaling on TRAIN trials only, then apply the same fixed
    # bounds to val/test -- this is the leakage-safe step that used to happen
    # (incorrectly, pool-wide) in build_dataset.py. See DS005285_LSTM_ARCHITECTURE.md
    # for why this discipline matters.
    # Rich features cannot use the power quantizer: it requires nonnegative input
    # and log-transforms internally, but an N2 amplitude is negative by definition
    # and several rich columns are already logarithmic. They get a signed linear
    # 8-bit scaling instead, still fitted on training rows only.
    rich_mode = args.feature_mode == "rich"
    if rich_mode:
        flat_train = model_power[train_mask].reshape(-1, model_power.shape[2])
        band_lo, band_hi = fit_uint8_bounds(flat_train)

        def to_model_input(values):
            return apply_uint8_bounds(values, band_lo, band_hi).astype(np.float32) / 255.0

        print(
            f"Signed 8-bit bounds fit on train only over {band_lo.size} features "
            f"-> lo range {band_lo.min():.3g}..{band_lo.max():.3g}, "
            f"hi range {band_hi.min():.3g}..{band_hi.max():.3g}"
        )
    else:
        log_power_train = np.log1p(model_power[train_mask])
        band_lo = np.percentile(log_power_train, 1, axis=0)
        band_hi = np.percentile(log_power_train, 99, axis=0)

        def to_model_input(values):
            return quantize_to_uint8(values, band_lo, band_hi).astype(np.float32) / 255.0

        print(f"Quantization bounds fit on train only -> lo:{band_lo.round(2).tolist()} "
              f"hi:{band_hi.round(2).tolist()}")

    X_train = torch.from_numpy(to_model_input(model_power[train_mask]))
    y_train = torch.from_numpy(y[train_mask])
    ordinal_balance_report = None
    if args.task_mode == "categorical":
        (
            train_weights_np,
            applied_class_correction,
            full_class_correction,
            class_mass,
        ) = training_trial_weights(
            dataset_id[train_mask],
            subject_id[train_mask],
            y[train_mask],
            policy=args.training_weight_policy,
            class_balance_strength=args.class_balance_strength,
        )
    else:
        (
            train_weights_np,
            applied_target_correction,
            full_target_correction,
            base_target_mass,
            raw_target_counts,
            final_target_mass,
        ) = ordinal_training_loss_weights(
            dataset_id[train_mask],
            subject_id[train_mask],
            y[train_mask],
            policy=args.training_weight_policy,
            class_balance_strength=args.class_balance_strength,
        )
        base = base_trial_weights(
            dataset_id[train_mask],
            subject_id[train_mask],
            policy=args.training_weight_policy,
        )
        class_mass = np.bincount(
            y[train_mask], weights=base, minlength=len(CLASS_NAMES)
        ).astype(np.float64)
        applied_class_correction = None
        full_class_correction = None
        ordinal_balance_report = {
            "rank_semantics": ORDINAL_RANK_SEMANTICS.copy(),
            "binary_target_order": [0, 1],
            "raw_target_counts": raw_target_counts.tolist(),
            "base_target_mass": base_target_mass.tolist(),
            "full_target_correction": full_target_correction.tolist(),
            "applied_target_correction": applied_target_correction.tolist(),
            "final_weighted_target_mass": final_target_mass.tolist(),
        }
    train_weights = torch.from_numpy(train_weights_np)
    X_val = (
        torch.from_numpy(to_model_input(model_power[val_mask]))
        if val_mask.sum()
        else None
    )
    y_val = torch.from_numpy(y[val_mask]) if val_mask.sum() else None
    X_test = (
        torch.from_numpy(to_model_input(model_power[test_mask]))
        if args.evaluation_mode == "final-test" and test_mask.sum()
        else None
    )
    y_test = (
        torch.from_numpy(y[test_mask])
        if args.evaluation_mode == "final-test" and test_mask.sum()
        else None
    )

    class_counts = np.bincount(y[train_mask], minlength=len(CLASS_NAMES))
    if args.task_mode == "categorical":
        weighted_class_mass = np.bincount(
            y[train_mask], weights=train_weights_np, minlength=len(CLASS_NAMES)
        )
        print(
            f"Training weighting: {args.training_weight_policy}; "
            f"class-balance strength={args.class_balance_strength:.3f}\n"
            f"  raw class counts: {class_counts.tolist()}\n"
            f"  base class mass: {class_mass.round(4).tolist()}\n"
            f"  full class correction: {full_class_correction.round(4).tolist()}\n"
            f"  applied class correction: {applied_class_correction.round(4).tolist()}\n"
            f"  final weighted class mass: {weighted_class_mass.round(2).tolist()}"
        )
    else:
        weighted_class_mass = None
        print(
            f"Training weighting: {args.training_weight_policy}; ordinal binary-rank "
            f"balance strength={args.class_balance_strength:.3f}\n"
            f"  raw class counts: {class_counts.tolist()}\n"
            f"  base class mass (audit only): {class_mass.round(4).tolist()}\n"
            f"  rank semantics: {ORDINAL_RANK_SEMANTICS}\n"
            f"  raw binary target counts [negative, positive]: "
            f"{raw_target_counts.tolist()}\n"
            f"  base binary target mass: {base_target_mass.round(4).tolist()}\n"
            f"  full binary corrections: {full_target_correction.round(4).tolist()}\n"
            f"  applied binary corrections: "
            f"{applied_target_correction.round(4).tolist()}\n"
            f"  final weighted binary target mass: "
            f"{final_target_mass.round(2).tolist()}"
        )

    train_loader = make_training_loader(
        X_train,
        y_train,
        train_weights,
        batch_size=args.batch_size,
        seed=args.loader_seed,
    )

    input_size = model_power.shape[2]
    model = PainLSTM(
        input_size=input_size,
        hidden_size=args.hidden,
        n_classes=len(CLASS_NAMES),
        task_mode=args.task_mode,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = (
        nn.CrossEntropyLoss(reduction="none")
        if args.task_mode == "categorical"
        else nn.BCEWithLogitsLoss(reduction="none")
    )

    best_selection_val_dataset_macro_balanced = -1.0
    best_state = None
    best_epoch = None
    best_validation_report = None
    best_selection_validation_report = None
    epochs_since_improvement = 0
    training_history = []
    epochs_completed = 0
    optimizer_steps = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_weighted_loss = 0.0
        epoch_weight = 0.0
        for X_batch, y_batch, weight_batch in train_loader:
            opt.zero_grad()
            logits = model(X_batch)
            if args.task_mode == "categorical":
                losses = loss_fn(logits, y_batch)
            else:
                rank_targets = ordinal_targets_tensor(y_batch, dtype=logits.dtype)
                losses = loss_fn(logits, rank_targets)
            loss = torch.mean(losses * weight_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            optimizer_steps += 1
            epoch_weighted_loss += float(torch.sum(losses.detach() * weight_batch).item())
            epoch_weight += float(torch.sum(weight_batch).item())
        train_loss = epoch_weighted_loss / epoch_weight

        train_pred = predict_in_batches(model, X_train, args.batch_size)
        train_acc = float(np.mean(train_pred == y_train.numpy()))
        val_report = None
        val_acc = None
        val_balanced = None
        val_dataset_macro_balanced = None
        selection_val_dataset_macro_balanced = None
        selection_val_report = None
        if X_val is not None:
            val_pred = predict_in_batches(model, X_val, args.batch_size)
            val_report = evaluation_breakdown(
                y_val.numpy(),
                val_pred,
                dataset_id[val_mask],
                subject_id[val_mask],
            )
            val_acc = val_report["pooled"]["accuracy"]
            val_balanced = val_report["pooled"]["balanced_accuracy"]
            val_dataset_macro_balanced = val_report[
                "dataset_macro_balanced_accuracy"
            ]
            selection_in_validation = np.isin(
                dataset_id[val_mask], selection_dataset_ids
            )
            if not np.any(selection_in_validation):
                raise AssertionError("best-epoch selection has zero validation trials")
            selection_val_report = evaluation_breakdown(
                y_val.numpy()[selection_in_validation],
                val_pred[selection_in_validation],
                dataset_id[val_mask][selection_in_validation],
                subject_id[val_mask][selection_in_validation],
            )
            selection_val_dataset_macro_balanced = selection_val_report[
                "dataset_macro_balanced_accuracy"
            ]

        if selection_val_dataset_macro_balanced is not None:
            if (
                selection_val_dataset_macro_balanced
                > best_selection_val_dataset_macro_balanced
            ):
                best_selection_val_dataset_macro_balanced = (
                    selection_val_dataset_macro_balanced
                )
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_epoch = epoch
                best_validation_report = val_report
                best_selection_validation_report = selection_val_report
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

        training_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "validation_accuracy": (
                    None if val_acc is None else float(val_acc)
                ),
                "validation_balanced_accuracy": (
                    None if val_balanced is None else float(val_balanced)
                ),
                "validation_dataset_macro_balanced_accuracy": (
                    None
                    if val_dataset_macro_balanced is None
                    else float(val_dataset_macro_balanced)
                ),
                "selection_validation_dataset_macro_balanced_accuracy": (
                    None
                    if selection_val_dataset_macro_balanced is None
                    else float(selection_val_dataset_macro_balanced)
                ),
                "optimizer_steps_this_epoch": int(len(train_loader)),
            }
        )
        epochs_completed = epoch + 1

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            msg = f"epoch {epoch:3d} | train loss {train_loss:.3f} | train acc {train_acc:.3f}"
            if val_balanced is not None:
                msg += (
                    f" | val acc {val_acc:.3f} | val balanced {val_balanced:.3f}"
                    f" | val dataset-macro balanced {val_dataset_macro_balanced:.3f}"
                    f" | selection balanced "
                    f"{selection_val_dataset_macro_balanced:.3f}"
                )
            print(msg)

        if best_state is not None and epochs_since_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}: no val improvement for {args.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            "Restored best epoch's weights "
            f"(selection-validation dataset-macro balanced accuracy "
            f"{best_selection_val_dataset_macro_balanced:.3f})"
        )

    validation_report = None
    selection_validation_report = None
    validation_predictions = None
    selection_validation_predictions = None
    validation_rating_boundary_report = None
    selection_validation_rating_boundary_report = None
    if X_val is not None:
        best_val_pred = predict_in_batches(model, X_val, args.batch_size)
        validation_report = evaluation_breakdown(
            y_val.numpy(),
            best_val_pred,
            dataset_id[val_mask],
            subject_id[val_mask],
        )
        validation_rating_boundary_report = rating_boundary_breakdown(
            ratings[val_mask], y_val.numpy(), best_val_pred
        )
        validation_report["rating_boundary_report"] = (
            validation_rating_boundary_report
        )
        validation_predictions = prediction_payload(
            dataset_id[val_mask],
            subject_id[val_mask],
            epoch_index[val_mask],
            ratings[val_mask],
            y_val.numpy(),
            best_val_pred,
        )
        print_evaluation_breakdown(validation_report, "Best held-out VALIDATION")
        print_rating_boundary_breakdown(
            validation_rating_boundary_report, "Best held-out VALIDATION"
        )
        selection_in_validation = np.isin(dataset_id[val_mask], selection_dataset_ids)
        selection_validation_report = evaluation_breakdown(
            y_val.numpy()[selection_in_validation],
            best_val_pred[selection_in_validation],
            dataset_id[val_mask][selection_in_validation],
            subject_id[val_mask][selection_in_validation],
        )
        selection_validation_rating_boundary_report = rating_boundary_breakdown(
            ratings[val_mask][selection_in_validation],
            y_val.numpy()[selection_in_validation],
            best_val_pred[selection_in_validation],
        )
        selection_validation_report["rating_boundary_report"] = (
            selection_validation_rating_boundary_report
        )
        selection_validation_predictions = prediction_payload(
            dataset_id[val_mask][selection_in_validation],
            subject_id[val_mask][selection_in_validation],
            epoch_index[val_mask][selection_in_validation],
            ratings[val_mask][selection_in_validation],
            y_val.numpy()[selection_in_validation],
            best_val_pred[selection_in_validation],
        )
        if set(selection_dataset_ids) != set(training_dataset_ids):
            print_evaluation_breakdown(
                selection_validation_report,
                "BEST-EPOCH SELECTION VALIDATION",
            )
            print_rating_boundary_breakdown(
                selection_validation_rating_boundary_report,
                "BEST-EPOCH SELECTION VALIDATION",
            )

    test_report = None
    test_predictions = None
    if X_test is not None:
        y_pred_test = predict_in_batches(model, X_test, args.batch_size)
        y_true_test = y_test.numpy()
        print_evaluation(
            y_true_test, y_pred_test, f"Held-out TEST ({len(test_keys)} never-trained-on subjects)"
        )
        test_report = evaluation_breakdown(
            y_true_test,
            y_pred_test,
            dataset_id[test_mask],
            subject_id[test_mask],
        )
        print_evaluation_breakdown(test_report, "Held-out TEST")
        test_predictions = {
            "dataset_id": dataset_id[test_mask].tolist(),
            "subject_id": subject_id[test_mask].tolist(),
            "epoch_index": epoch_index[test_mask].astype(int).tolist(),
            "true_label": y_true_test.astype(int).tolist(),
            "predicted_label": y_pred_test.astype(int).tolist(),
        }
    elif args.evaluation_mode == "validation-only":
        print(
            "\nHeld-out test split is reserved: validation-only mode performed "
            "no test inference and saved no test predictions."
        )
    else:
        print("\nNo held-out test subjects in this run -- too few unique subjects to split "
              "three ways meaningfully. This is expected for the current 5-subject smoke test; "
              "re-run after scaling up to the full subject lists.")

    # Everything needed to actually USE this model later lives in one file:
    # the learned weights themselves, plus the exact train-only quantization
    # bounds (band_lo/band_hi) -- without those, a future input can't be
    # turned into the same 0-255 scale the model was trained on.
    save_directory = os.path.dirname(args.save)
    if save_directory:
        os.makedirs(save_directory, exist_ok=True)
    head_provenance = model_head_provenance(model)
    _atomic_torch_save(
        {
            "checkpoint_schema_version": 4,
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_size": args.hidden,
            "n_classes": len(CLASS_NAMES),
            "n_outputs": head_provenance["output_logit_count"],
            "class_names": CLASS_NAMES,
            "task_mode": args.task_mode,
            "model_head": head_provenance,
            "training_loss": (
                "cross_entropy"
                if args.task_mode == "categorical"
                else "binary_cross_entropy_over_cumulative_ranks"
            ),
            "band_lo": band_lo,
            "band_hi": band_hi,
            "feature_archive": os.path.abspath(args.data),
            "feature_archive_schema_version": ARCHIVE_SCHEMA_VERSION,
            "feature_archive_build_id": _scalar_text(archive["build_id"], "build_id"),
            "split_seed": effective_split_seed,
            "split_source": split_source,
            "model_seed": args.model_seed,
            "loader_seed": args.loader_seed,
            "deterministic_algorithms": True,
            "requested_epochs": args.epochs,
            "epochs_completed": epochs_completed,
            "batch_size": args.batch_size,
            "batches_per_epoch": len(train_loader),
            "optimizer_steps": optimizer_steps,
            "gradient_clip_norm": args.gradient_clip,
            "learning_rate": args.lr,
            "weight_decay": 1e-4,
            "early_stopping_patience": args.patience,
            "early_stopping_metric": (
                "selection_validation_dataset_macro_balanced_accuracy"
            ),
            "feature_mode": args.feature_mode,
            "feature_order": feature_order,
            "best_epoch": best_epoch,
            "best_selection_validation_dataset_macro_balanced_accuracy": (
                None
                if best_epoch is None
                else float(best_selection_val_dataset_macro_balanced)
            ),
            "best_validation_report_during_training": best_validation_report,
            "best_selection_validation_report_during_training": (
                best_selection_validation_report
            ),
            "validation_report": validation_report,
            "selection_validation_report": selection_validation_report,
            "validation_predictions": validation_predictions,
            "selection_validation_predictions": selection_validation_predictions,
            "validation_rating_boundary_report": (
                validation_rating_boundary_report
            ),
            "selection_validation_rating_boundary_report": (
                selection_validation_rating_boundary_report
            ),
            "training_history": training_history,
            "training_weight_policy": args.training_weight_policy,
            "class_balance_strength": args.class_balance_strength,
            "balance_semantics": (
                "three_class_correction"
                if args.task_mode == "categorical"
                else "independent_negative_positive_correction_per_cumulative_rank"
            ),
            "train_class_counts": class_counts.tolist(),
            "base_class_mass": class_mass.tolist(),
            "class_correction": (
                None
                if applied_class_correction is None
                else applied_class_correction.tolist()
            ),
            "applied_class_correction": (
                None
                if applied_class_correction is None
                else applied_class_correction.tolist()
            ),
            "full_class_correction": (
                None
                if full_class_correction is None
                else full_class_correction.tolist()
            ),
            "final_weighted_class_mass": (
                None if weighted_class_mass is None else weighted_class_mass.tolist()
            ),
            "ordinal_balance_report": ordinal_balance_report,
            "split_audit": split_audit,
            "evaluation_mode": args.evaluation_mode,
            "test_evaluated": args.evaluation_mode == "final-test" and X_test is not None,
            "test_report": test_report,
            "test_predictions": test_predictions,
            "n_train_subjects": len(train_keys),
            "n_val_subjects": len(val_keys),
            "n_test_subjects": len(test_keys),
            "n_train_trials": int(train_mask.sum()),
            "n_val_trials": int(val_mask.sum()),
            "n_test_trials": int(test_mask.sum()),
            "train_subject_keys": [list(key) for key in sorted(train_keys)],
            "val_subject_keys": [list(key) for key in sorted(val_keys)],
            "test_subject_keys": [list(key) for key in sorted(test_keys)],
            "channel_order": archive["channel_order"].tolist(),
            "band_order": archive["band_order"].tolist(),
            "baseline_window_s": archive["baseline_window_s"].tolist(),
            "post_windows_s": archive["post_windows_s"].tolist(),
            "training_dataset_ids": training_dataset_ids,
            "selection_dataset_ids": selection_dataset_ids,
            "feature_archive_selected_dataset_ids": (
                archive["selected_dataset_ids"].tolist()
            ),
            "selected_dataset_ids": archive["selected_dataset_ids"].tolist(),
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "platform": platform.platform(),
            },
        },
        args.save,
    )
    print(f"Saved trained model + quantization bounds to {args.save}")


if __name__ == "__main__":
    main()
