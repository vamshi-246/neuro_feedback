"""Let every feature family compete for the same slots.

Each family was judged the same unfair way: computed, appended to whatever the baseline
was at the time, and scored as a block. MVAR added 244 columns at once and lost. PAC
added 41 and lost. But optimize_pipeline.py then found that column count itself is the
problem -- 16 features beat all 235, with the score falling monotonically as more were
added. A family that loses when it contributes 244 columns has not been shown to be
worthless; it has been shown that 244 extra columns hurt.

This runs the fair version. Every family goes into one pool, the ranker sees them all
on equal terms, and the selection keeps whichever columns earn a place regardless of
where they came from:

  rich    194  evoked potential, five bands, shape, channel coupling
  PAC      41  phase-amplitude coupling, peak alpha frequency, asymmetry
  MVAR    244  autoregressive coefficients, partial directed coherence, parametric power
  ratios  120  within-channel band ratios, computed here as differences of log power

PAC has effectively been through this once already: optimize_pipeline searched the
235-column rich+PAC pool and no PAC column reached the top 16. MVAR never got that
chance, which is the gap this closes.

Honest protocol
---------------
The pool size is chosen by cross-validation inside the training subjects, grouped so no
subject spans a fold. Validation is scored once, at the end. Choosing the size by
checking validation would let one of several candidates win on luck.

Reported on dataset-macro, which site fingerprinting cannot inflate, with pooled
alongside for continuity.

Usage (from repo root):
    python scripts/preprocessing/all_families_selection.py
"""

import argparse
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    CLASS_NAMES,
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    select_dataset_view,
)

TUNED = {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300,
         "l2_regularization": 1.0}
SUBSET_SIZES = (8, 16, 24, 32, 48, 64)
CV_FOLDS = 3
LOG_POWER = re.compile(r"^(?P<ch>[^:]+):(?P<band>delta|theta|alpha|beta|gamma):"
                       r"w(?P<w>\d+):log_absolute$")


def balanced_sample_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = labels.size / (classes.size * counts.astype(np.float64))
    lookup = dict(zip(classes.tolist(), per_class.tolist()))
    return np.asarray([lookup[v] for v in labels.tolist()], dtype=np.float64)


def family_of(name):
    if name.startswith("mvar:"):
        return "MVAR"
    if "_over_" in name:
        return "ratio"
    if ":pac:" in name or "peak_alpha" in name or "alpha_peak" in name \
            or "alpha_asymmetry" in name:
        return "PAC"
    return "rich"


def trial_keys(archive):
    return [
        f"{d}||{s}||{int(e)}"
        for d, s, e in zip(
            archive["dataset_id"].astype(str).tolist(),
            archive["subject_id"].astype(str).tolist(),
            archive["epoch_index"].astype(np.int64).tolist(),
        )
    ]


def band_ratio_columns(X, names):
    grouped = {}
    for index, name in enumerate(names):
        match = LOG_POWER.match(name)
        if match:
            grouped.setdefault((match.group("ch"), match.group("w")), []).append(
                (match.group("band"), index))
    columns, labels = [], []
    for (channel, window), entries in sorted(grouped.items()):
        entries.sort()
        for i, (band_a, col_a) in enumerate(entries):
            for band_b, col_b in entries[i + 1:]:
                columns.append(X[:, col_a] - X[:, col_b])
                labels.append(f"{channel}:w{window}:{band_a}_over_{band_b}")
    return np.stack(columns, axis=1), labels


def dataset_macro(y_true, y_pred, datasets):
    return float(np.mean([
        balanced_accuracy_score(y_true[datasets == ds], y_pred[datasets == ds])
        for ds in sorted(set(datasets.tolist()))
    ]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pac", default="outputs/rich_full/rich_all9_pac.npz")
    ap.add_argument("--mvar", default="outputs/rich_full/rich_all9_mvar.npz")
    ap.add_argument("--baseline", default="outputs/all9_full/all9_features.npz")
    ap.add_argument("--split-from-checkpoint",
                    default="outputs/all9_full/all9_pain_lstm_final_seed20260726.pt")
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    pac = np.load(args.pac, allow_pickle=False)
    mvar = np.load(args.mvar, allow_pickle=False)

    # Align the two archives by trial identity rather than trusting row order.
    pac_keys, mvar_keys = trial_keys(pac), trial_keys(mvar)
    lookup = {key: i for i, key in enumerate(mvar_keys)}
    missing = [k for k in pac_keys if k not in lookup]
    if missing:
        raise SystemExit(f"{len(missing)} trials are absent from the MVAR archive")
    order = np.asarray([lookup[k] for k in pac_keys])

    pac_names = list(pac["feature_names"].astype(str))
    mvar_names = list(mvar["feature_names"].astype(str))
    mvar_only = [i for i, n in enumerate(mvar_names) if n.startswith("mvar:")]

    X_pac = pac["features"]
    ratios, ratio_names = band_ratio_columns(X_pac, pac_names)
    X = np.hstack([X_pac, mvar["features"][order][:, mvar_only], ratios])
    names = pac_names + [mvar_names[i] for i in mvar_only] + ratio_names
    if X.shape[1] != len(names):
        raise SystemExit("feature matrix and name list disagree")

    y = pac["labels"].astype(np.int64)
    dataset_id = pac["dataset_id"].astype(str)
    subject_id = pac["subject_id"].astype(str)

    archive = load_feature_archive(args.baseline)
    _, training_dataset_ids = select_dataset_view(archive, None)
    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        args.split_from_checkpoint, archive, training_dataset_ids
    )
    tr = keys_mask(dataset_id, subject_id, train_keys)
    va = keys_mask(dataset_id, subject_id, val_keys)
    ds_val = dataset_id[va]
    groups = np.asarray(
        [f"{d}||{s}" for d, s in zip(dataset_id[tr].tolist(), subject_id[tr].tolist())]
    )

    counts = {}
    for n in names:
        counts[family_of(n)] = counts.get(family_of(n), 0) + 1
    print(f"Combined pool: {X.shape[1]} features   " +
          "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"Train {int(tr.sum())}   validation {int(va.sum())}")
    print(f"Pool size chosen by {CV_FOLDS}-fold subject-grouped CV inside training.\n")
    started = time.time()

    ranker = ExtraTreesClassifier(n_estimators=400, random_state=args.seed, n_jobs=-1)
    ranker.fit(X[tr], y[tr], sample_weight=balanced_sample_weights(y[tr]))
    ranked = np.argsort(ranker.feature_importances_)[::-1]

    print("Choosing how many features to keep:")
    print(f"  {'columns':>9}{'CV dataset-macro':>20}")
    best_k, best_score = None, -1.0
    for size in SUBSET_SIZES:
        cols = ranked[:size]
        scores = []
        for fit_idx, score_idx in GroupKFold(n_splits=CV_FOLDS).split(
                X[tr], y[tr], groups):
            model = HistGradientBoostingClassifier(random_state=args.seed, **TUNED)
            model.fit(X[tr][fit_idx][:, cols], y[tr][fit_idx],
                      sample_weight=balanced_sample_weights(y[tr][fit_idx]))
            scores.append(dataset_macro(
                y[tr][score_idx], model.predict(X[tr][score_idx][:, cols]),
                dataset_id[tr][score_idx]))
        mean = float(np.mean(scores))
        print(f"  {size:>9}{mean:>20.4f}")
        if mean > best_score:
            best_k, best_score = size, mean
    print(f"  -> chose {best_k} ({time.time()-started:.0f}s)\n")

    chosen = ranked[:best_k]
    model = HistGradientBoostingClassifier(random_state=args.seed, **TUNED)
    model.fit(X[tr][:, chosen], y[tr], sample_weight=balanced_sample_weights(y[tr]))
    prediction = model.predict(X[va][:, chosen])
    macro = dataset_macro(y[va], prediction, ds_val)

    print("Validation, scored once:")
    print(f"  {'model':<46}{'DATASET-MACRO':>15}{'pooled':>9}{'acc':>8}")
    print("  " + "-" * 78)
    print(f"  {'previous best: 16 of 235 (rich+PAC)':<46}{0.4505:>15.4f}"
          f"{0.4878:>9.4f}{0.4427:>8.4f}")
    print(f"  {f'all families, {best_k} of {X.shape[1]}':<46}{macro:>15.4f}"
          f"{balanced_accuracy_score(y[va], prediction):>9.4f}"
          f"{accuracy_score(y[va], prediction):>8.4f}")
    print(f"\n  change vs previous best: {100*(macro-0.4505):+.2f} points")

    picked = {}
    for i in chosen:
        picked[family_of(names[i])] = picked.get(family_of(names[i]), 0) + 1
    print(f"\n  Where the {best_k} selected columns came from:")
    for family in sorted(counts):
        available = counts[family]
        taken = picked.get(family, 0)
        share = 100 * taken / best_k
        print(f"    {family:<8}{taken:>3} of {available:>4} available   "
              f"{share:>5.1f}% of the selection")
    print(
        "\n  A family taking no slots here has been given a fair hearing: it competed\n"
        "  column by column against every other family and none of its columns earned\n"
        "  a place. That is a stronger verdict than losing as a 244-column block.\n"
    )
    print("  Selected columns, in order of importance:")
    for rank, i in enumerate(chosen, 1):
        print(f"    {rank:>2}  [{family_of(names[i]):<5}] {names[i]}")


if __name__ == "__main__":
    main()
