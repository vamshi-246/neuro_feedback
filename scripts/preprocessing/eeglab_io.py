"""
Minimal reader for the lab's EEGLAB `.set`/`.fdt` derivative files.

Two quirks discovered while inspecting the actual files (not assumed from docs):

1. Some `.set` files store the EEGLAB structure nested under a top-level `EEG`
   key; others store the same fields flat at the root of the .mat dict. Both
   forms show up across our 5 datasets, so every accessor here goes through
   `_get()` rather than assuming one layout.
2. The `.set` file only holds metadata -- the actual voltage samples live in a
   sibling `.fdt` file, stored as raw float32 in MATLAB's column-major order
   for a (channels, time_points, trials) array. Confirmed by checking that
   nbchan * pnts * trials * 4 bytes exactly equals the .fdt file size for every
   dataset checked.
"""

from dataclasses import dataclass
import numpy as np
import scipy.io as sio


def _get(root, field, is_nested):
    return getattr(root, field) if is_nested else root[field]


@dataclass
class Recording:
    dataset_id: str
    subject_id: str
    channel_labels: list       # length nbchan
    srate: float
    data: np.ndarray           # shape (nbchan, pnts, trials), float32, microvolts
    ratings: np.ndarray        # shape (trials,), float, 0-10 pain score (may contain NaN)
    trial_ok: np.ndarray       # shape (trials,), bool -- False where rating is NaN/missing


def load_recording(dataset_id: str, subject_id: str, set_path: str, fdt_path: str) -> Recording:
    mat = sio.loadmat(set_path, struct_as_record=False, squeeze_me=True)
    is_nested = "EEG" in mat
    root = mat["EEG"] if is_nested else mat

    nbchan = int(_get(root, "nbchan", is_nested))
    pnts = int(_get(root, "pnts", is_nested))
    trials = int(_get(root, "trials", is_nested))
    srate = float(_get(root, "srate", is_nested))

    chanlocs = np.atleast_1d(_get(root, "chanlocs", is_nested))
    channel_labels = [str(c.labels) for c in chanlocs]

    raw = np.fromfile(fdt_path, dtype="<f4")
    expected = nbchan * pnts * trials
    if raw.size != expected:
        raise ValueError(
            f"{dataset_id}/{subject_id}: .fdt has {raw.size} floats, expected "
            f"{nbchan}*{pnts}*{trials}={expected}. File layout assumption is wrong."
        )
    data = raw.reshape((nbchan, pnts, trials), order="F")

    events = np.atleast_1d(_get(root, "event", is_nested))
    ratings = np.full(trials, np.nan, dtype=float)
    for e in events:
        epoch_idx = int(e.epoch) - 1  # EEGLAB epoch numbers are 1-based
        r = e.rating
        if isinstance(r, (int, float)) and not (isinstance(r, float) and np.isnan(r)):
            ratings[epoch_idx] = float(r)

    trial_ok = ~np.isnan(ratings)

    return Recording(
        dataset_id=dataset_id,
        subject_id=subject_id,
        channel_labels=channel_labels,
        srate=srate,
        data=data,
        ratings=ratings,
        trial_ok=trial_ok,
    )


def pick_channels(labels: list, wanted: list) -> list:
    """Case-insensitive lookup; raises if any wanted channel is missing."""
    lut = {lbl.strip().upper(): i for i, lbl in enumerate(labels)}
    indices = []
    for w in wanted:
        key = w.strip().upper()
        if key not in lut:
            raise KeyError(f"channel '{w}' not found among {labels}")
        indices.append(lut[key])
    return indices
