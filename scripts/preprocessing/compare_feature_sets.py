"""Score old and new features on identical trials, then attribute any gain.

The pilot archive admits trials under exactly the same rules as the pooled
archive, so the two feature sets can be matched row by row on
(dataset_id, subject_id, epoch_index).  Both are then given the same subjects,
the same split, and the same classifier, which leaves the feature columns as the
only difference between them.

Gradient boosting stands in for the LSTM here because it trains in seconds and
already matched or beat it on the old features, so it measures what the features
support rather than what one architecture happens to extract.

The ablation rows matter as much as the headline.  "everything except ERP"
isolates how much the evoked potential alone is worth: if the full set wins but
that row does not, the N2-P2 complex is carrying the improvement and the RTL
feature vector must grow to include it.

Nothing is written to disk and the pooled archive's test split is never touched.

Usage (from repo root):
    python scripts/preprocessing/compare_feature_sets.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    model_feature_view,
    select_dataset_view,
    subject_split,
)


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def trial_keys(dataset_id, subject_id, epoch_index):
    return [
        f"{d}||{s}||{int(e)}"
        for d, s, e in zip(
            np.asarray(dataset_id, dtype=str).tolist(),
            np.asarray(subject_id, dtype=str).tolist(),
            np.asarray(epoch_index).tolist(),
        )
    ]


# Set once by main(); the dataset label of every validation row, needed for the
# dataset-macro column.
VALIDATION_DATASETS = None


def evaluate_columns(tag, X, y, train_mask, val_mask, seed):
    """Fit on the training rows and report pooled and dataset-macro scores.

    Pooled throws every validation trial into one pile. That rewards a model for
    recognising WHICH dataset a trial came from, because the nine datasets have very
    different class mixes and are partly identifiable from the signal -- dataset
    identity alone scores 0.4064 pooled with no EEG at all. Dataset-macro scores each
    dataset separately and averages, where knowing the dataset is worth nothing
    because it is constant within each group. Both are printed: pooled for continuity
    with earlier numbers, dataset-macro as the one that cannot be gamed.
    """

    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X[train_mask], y[train_mask],
              sample_weight=balanced_sample_weights(y[train_mask]))
    pred = model.predict(X[val_mask])
    bal = balanced_accuracy_score(y[val_mask], pred)
    acc = accuracy_score(y[val_mask], pred)
    truth = y[val_mask]
    macro = float(np.mean([
        balanced_accuracy_score(truth[VALIDATION_DATASETS == ds],
                                pred[VALIDATION_DATASETS == ds])
        for ds in sorted(set(VALIDATION_DATASETS.tolist()))
    ]))
    print(f"  {tag:<44}{X.shape[1]:>6}{macro:>14.4f}{bal:>10.4f}{acc:>9.4f}")
    return macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rich", default="outputs/rich_pilot/rich_pilot_features.npz")
    ap.add_argument("--baseline", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--split-seed", type=int, default=20260725)
    ap.add_argument(
        "--split-from-checkpoint",
        help=(
            "Reuse the locked subject split from a trained checkpoint. Required to "
            "compare against earlier runs; a fresh seeded split is only meaningful "
            "for a subset that the checkpoint's split does not fully cover."
        ),
    )
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    rich = np.load(args.rich, allow_pickle=False)
    rich_names = list(rich["feature_names"].astype(str))
    rich_keys = trial_keys(rich["dataset_id"], rich["subject_id"], rich["epoch_index"])

    base_archive = load_feature_archive(args.baseline)
    base_power, _order = model_feature_view(base_archive, "per-channel")
    base_flat = np.log1p(base_power.reshape(base_power.shape[0], -1))
    base_keys = trial_keys(
        base_archive["dataset_id"], base_archive["subject_id"], base_archive["epoch_index"]
    )

    # Keep only trials both archives admitted, so the comparison is row-identical.
    base_lookup = {key: i for i, key in enumerate(base_keys)}
    pairs = [(i, base_lookup[key]) for i, key in enumerate(rich_keys) if key in base_lookup]
    if not pairs:
        raise SystemExit("no overlapping trials between the two archives")
    rich_idx = np.asarray([p[0] for p in pairs])
    base_idx = np.asarray([p[1] for p in pairs])

    X_new = rich["features"][rich_idx]
    X_old = base_flat[base_idx]
    y = rich["labels"].astype(np.int64)[rich_idx]
    dataset_id = rich["dataset_id"].astype(str)[rich_idx]
    subject_id = rich["subject_id"].astype(str)[rich_idx]

    if not np.array_equal(y, base_archive["labels"].astype(np.int64)[base_idx]):
        raise SystemExit("matched rows disagree on labels; archives are not aligned")

    if args.split_from_checkpoint:
        _, training_dataset_ids = select_dataset_view(base_archive, None)
        train_keys, val_keys, _test, provenance = load_checkpoint_split(
            args.split_from_checkpoint, base_archive, training_dataset_ids
        )
        split_label = f"locked checkpoint ({os.path.basename(args.split_from_checkpoint)})"
    else:
        train_keys, val_keys, _test = subject_split(
            dataset_id, subject_id, seed=args.split_seed
        )
        split_label = f"fresh seeded split (seed {args.split_seed})"
    train_mask = keys_mask(dataset_id, subject_id, train_keys)
    val_mask = keys_mask(dataset_id, subject_id, val_keys)
    global VALIDATION_DATASETS
    VALIDATION_DATASETS = dataset_id[val_mask]
    print(f"Split: {split_label}")

    print(f"Matched trials: {len(pairs)}   "
          f"subjects: {len(set(zip(dataset_id.tolist(), subject_id.tolist())))}")
    print(f"Datasets: {sorted(set(dataset_id.tolist()))}")
    print(f"Train trials {int(train_mask.sum())}   Validation trials {int(val_mask.sum())}")
    counts = np.bincount(y[val_mask], minlength=len(CLASS_NAMES))
    print(f"Validation class counts: {counts.tolist()}")
    print(f"Chance balanced accuracy: {1/len(CLASS_NAMES):.4f}\n")

    def pick(predicate):
        cols = [i for i, n in enumerate(rich_names) if predicate(n)]
        return X_new[:, cols]

    is_erp = lambda n: ":erp_" in n
    is_band = lambda n: any(f":{b}:" in n for b in
                            ("delta", "theta", "alpha", "beta", "gamma")) and "coupling" not in n
    is_new_band = lambda n: (":delta:" in n or ":gamma:" in n) and "coupling" not in n
    is_shape = lambda n: "hjorth" in n or "spectral_entropy" in n
    is_coupling = lambda n: "coupling" in n
    # PAC names read "Cz:pac:delta_phase_gamma_amp:response", so they carry band words
    # without the ":band:" pattern is_band matches. No overlap between the groups.
    is_pac = lambda n: ":pac:" in n or "peak_alpha_frequency" in n \
        or "alpha_peak_power" in n or "alpha_asymmetry" in n

    # Stimulus intensity alone already beat all 36 old EEG columns on the pooled
    # archive (0.4727 vs 0.4584).  It is therefore the real bar: an EEG feature
    # set that cannot add anything on top of it is not measuring pain, it is
    # rediscovering how hard the laser fired.
    laser = rich["laser_power"].astype(np.float64)[rich_idx].reshape(-1, 1)

    print(f"  {'feature set':<44}{'cols':>6}{'DATASET-MACRO':>14}"
          f"{'pooled':>10}{'acc':>9}")
    print("  " + "-" * 83)
    baseline_laser = evaluate_columns("BAR: laser power alone (no EEG)",
                                      laser, y, train_mask, val_mask, args.seed)
    old = evaluate_columns("OLD: alpha/beta/theta relative power",
                           X_old, y, train_mask, val_mask, args.seed)
    print("  " + "-" * 83)
    evaluate_columns("NEW: ERP (N2-P2) only", pick(is_erp), y, train_mask, val_mask, args.seed)
    evaluate_columns("NEW: 5 bands only", pick(is_band), y, train_mask, val_mask, args.seed)
    evaluate_columns("NEW: delta + gamma only", pick(is_new_band), y, train_mask, val_mask, args.seed)
    evaluate_columns("NEW: shape (Hjorth + entropy) only", pick(is_shape), y, train_mask, val_mask, args.seed)
    evaluate_columns("NEW: coupling only", pick(is_coupling), y, train_mask, val_mask, args.seed)
    if any(is_pac(n) for n in rich_names):
        evaluate_columns("NEW: PAC + alpha frequency only",
                         pick(is_pac), y, train_mask, val_mask, args.seed)
    print("  " + "-" * 83)
    if any(is_pac(n) for n in rich_names):
        without_pac = evaluate_columns("NEW: everything EXCEPT PAC",
                                       pick(lambda n: not is_pac(n)),
                                       y, train_mask, val_mask, args.seed)
    else:
        without_pac = None
    without_erp = evaluate_columns("NEW: everything EXCEPT ERP",
                                   pick(lambda n: not is_erp(n)), y, train_mask, val_mask, args.seed)
    full = evaluate_columns("NEW: everything", X_new, y, train_mask, val_mask, args.seed)

    print("  " + "-" * 83)
    erp_plus_laser = evaluate_columns(
        "NEW: ERP + laser power",
        np.hstack([pick(is_erp), laser]), y, train_mask, val_mask, args.seed)
    full_plus_laser = evaluate_columns(
        "NEW: everything + laser power",
        np.hstack([X_new, laser]), y, train_mask, val_mask, args.seed)

    print(f"\n  All figures below are dataset-macro, the metric site fingerprinting"
          f" cannot inflate.")
    print(f"  Full new set vs old set:       {100*(full-old):+6.2f} points")
    print(f"  Value of ERP within new set:   {100*(full-without_erp):+6.2f} points")
    if without_pac is not None:
        print(f"  Value of PAC within new set:   {100*(full-without_pac):+6.2f} points")
    print(f"  EEG gain ON TOP of laser power:{100*(full_plus_laser-baseline_laser):+6.2f} points")
    print(
        "\n  How to read the last line, which is the one that decides this:\n"
        "    clearly positive -> the EEG is measuring something about pain that\n"
        "                        the stimulus setting does not already say, and\n"
        "                        the full rebuild plus an RTL change are justified.\n"
        "    near zero        -> the features are re-deriving stimulus intensity.\n"
        "                        A device could skip the EEG entirely, so the task\n"
        "                        itself has to change rather than the feature list.\n"
        "\n  A large ERP contribution also means the RTL feature vector must carry\n"
        "  time-domain evoked-potential values, not just band powers."
    )


if __name__ == "__main__":
    main()
