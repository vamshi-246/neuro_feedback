"""The configuration everything now builds on, in one place.

Six weeks of experiments ended on one standing model, and every later idea has to be
measured against it rather than against a fresh default. Re-deriving it inside each new
script is how a comparison quietly slips back to a weaker baseline, so it lives here:

  pool        rich 194 + PAC 41 + MVAR 244 + band ratios 120 = 599 candidate columns
  selection   ExtraTrees importance ranking, pool size chosen by subject-grouped CV
              inside the training split
  model       HistGradientBoosting, max_depth 4, lr 0.05, 300 iterations, l2 1.0
  weighting   inverse-frequency sample weights, so the majority class earns nothing
  split       the locked subject split carried in the all9 checkpoint

Scored once on validation, that configuration reaches 0.4503 dataset-macro / 0.4952
pooled on the three-class task -- the "49%" this project is trying to beat.

Two rules the helpers here enforce, because both were violated at least once earlier:

  Anything chosen by looking at validation is not a result. Pool size, hyperparameters
  and decision thresholds are all chosen by GroupKFold inside the training subjects,
  where a subject never spans a fold.

  Dataset-macro is the honest headline. Pooled scoring rewards a model for recognising
  which of the nine datasets a trial came from -- the class mixes differ and the
  equipment is partly identifiable -- so pooled is printed for continuity only.

Not a script; imported by severe_detection.py, personalized_head.py and hardcoded_16.py.
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from train_lstm import (  # noqa: E402 -- path setup must precede this import
    keys_mask,
    load_checkpoint_split,
    load_feature_archive,
    select_dataset_view,
)

TUNED = {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300,
         "l2_regularization": 1.0}
SEED = 20260726
CV_FOLDS = 3

PAC_ARCHIVE = "outputs/rich_full/rich_all9_pac.npz"
MVAR_ARCHIVE = "outputs/rich_full/rich_all9_mvar.npz"
BASELINE_ARCHIVE = "outputs/all9_full/all9_features.npz"
SPLIT_CHECKPOINT = "outputs/all9_full/all9_pain_lstm_final_seed20260726.pt"

# The 16 columns the all-families search kept out of 599, in importance order. Recorded
# so the hardware target is a fixed list rather than whatever a rerun happens to pick.
SELECTED_16 = (
    "Cz:delta:w0:log_absolute",
    "Cz:erp_erp_rms",
    "Cz:theta:w0:log_absolute",
    "Cz:erp_n2p2_amplitude",
    "C3-C4:delta:coupling",
    "Cz:erp_p2_amplitude",
    "Cz:erp_n2_amplitude",
    "Cz:w0:delta_over_gamma",
    "mvar:parametric_power:theta:Cz",
    "C4:delta:w0:log_absolute",
    "mvar:parametric_power:delta:Cz",
    "Cz:erp_p2_window_mean",
    "Cz:delta:w1:log_absolute",
    "C3:delta:w0:log_absolute",
    "Cz-C3:delta:coupling",
    "Cz:erp_n2_window_mean",
)

_LOG_POWER = re.compile(r"^(?P<ch>[^:]+):(?P<band>delta|theta|alpha|beta|gamma):"
                        r"w(?P<w>\d+):log_absolute$")


def balanced_sample_weights(labels):
    """Inverse-frequency weights, so a majority-class guess scores nothing."""

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


def dataset_macro(y_true, y_pred, datasets):
    """Balanced accuracy per dataset, then averaged.

    Datasets carrying only one class of the target are skipped rather than scored:
    balanced accuracy is undefined there, and on the severe-pain task several of the
    nine datasets contain no rating at or above 7 at all.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    datasets = np.asarray(datasets)
    scores = []
    for dataset in sorted(set(datasets.tolist())):
        mask = datasets == dataset
        if np.unique(y_true[mask]).size < 2:
            continue
        scores.append(balanced_accuracy_score(y_true[mask], y_pred[mask]))
    return float(np.mean(scores)), len(scores)


def _trial_keys(archive):
    return [
        f"{d}||{s}||{int(e)}"
        for d, s, e in zip(
            archive["dataset_id"].astype(str).tolist(),
            archive["subject_id"].astype(str).tolist(),
            archive["epoch_index"].astype(np.int64).tolist(),
        )
    ]


def _band_ratio_columns(X, names):
    """Within-channel band ratios, as differences of log power."""

    grouped = {}
    for index, name in enumerate(names):
        match = _LOG_POWER.match(name)
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


def load_pool(pac_path=PAC_ARCHIVE, mvar_path=MVAR_ARCHIVE, with_mvar=True):
    """The 599-column candidate pool, plus the labels and identity columns.

    The two archives are aligned by trial identity rather than by row order, because
    nothing guarantees two separately built archives enumerate trials the same way.
    """

    pac = np.load(pac_path, allow_pickle=False)
    pac_names = list(pac["feature_names"].astype(str))
    X_pac = pac["features"]

    ratios, ratio_names = _band_ratio_columns(X_pac, pac_names)
    blocks, names = [X_pac], list(pac_names)

    if with_mvar:
        mvar = np.load(mvar_path, allow_pickle=False)
        lookup = {key: i for i, key in enumerate(_trial_keys(mvar))}
        keys = _trial_keys(pac)
        absent = [k for k in keys if k not in lookup]
        if absent:
            raise SystemExit(f"{len(absent)} trials are absent from the MVAR archive")
        order = np.asarray([lookup[k] for k in keys])
        mvar_names = list(mvar["feature_names"].astype(str))
        mvar_only = [i for i, n in enumerate(mvar_names) if n.startswith("mvar:")]
        blocks.append(mvar["features"][order][:, mvar_only])
        names += [mvar_names[i] for i in mvar_only]

    blocks.append(ratios)
    names += ratio_names

    X = np.hstack(blocks)
    if X.shape[1] != len(names):
        raise SystemExit("feature matrix and name list disagree")

    return {
        "X": X,
        "names": names,
        "labels": pac["labels"].astype(np.int64),
        "ratings": pac["ratings"].astype(np.float64),
        "laser_power": pac["laser_power"].astype(np.float64),
        "dataset_id": pac["dataset_id"].astype(str),
        "subject_id": pac["subject_id"].astype(str),
        "epoch_index": pac["epoch_index"].astype(np.int64),
    }


def load_split(dataset_id, subject_id, baseline=BASELINE_ARCHIVE,
               checkpoint=SPLIT_CHECKPOINT):
    """The locked subject split, so every number stays comparable to earlier runs."""

    archive = load_feature_archive(baseline)
    _, training_dataset_ids = select_dataset_view(archive, None)
    train_keys, val_keys, _test, _prov = load_checkpoint_split(
        checkpoint, archive, training_dataset_ids
    )
    return (keys_mask(dataset_id, subject_id, train_keys),
            keys_mask(dataset_id, subject_id, val_keys))


def subject_groups(dataset_id, subject_id):
    return np.asarray([f"{d}||{s}" for d, s
                       in zip(np.asarray(dataset_id).tolist(),
                              np.asarray(subject_id).tolist())])


def rank_columns(X, y, seed=SEED, n_estimators=400):
    ranker = ExtraTreesClassifier(n_estimators=n_estimators, random_state=seed,
                                  n_jobs=-1)
    ranker.fit(X, y, sample_weight=balanced_sample_weights(y))
    return np.argsort(ranker.feature_importances_)[::-1]


def cv_dataset_macro(X, y, groups, datasets, params, seed=SEED, folds=CV_FOLDS):
    """Mean dataset-macro across training folds. Never touches validation."""

    scores = []
    for fit_idx, score_idx in GroupKFold(n_splits=folds).split(X, y, groups):
        model = HistGradientBoostingClassifier(random_state=seed, **params)
        model.fit(X[fit_idx], y[fit_idx],
                  sample_weight=balanced_sample_weights(y[fit_idx]))
        macro, _ = dataset_macro(y[score_idx], model.predict(X[score_idx]),
                                 datasets[score_idx])
        scores.append(macro)
    return float(np.mean(scores))


def fit_tuned(X, y, params=None, seed=SEED):
    model = HistGradientBoostingClassifier(random_state=seed,
                                           **(TUNED if params is None else params))
    model.fit(X, y, sample_weight=balanced_sample_weights(y))
    return model
