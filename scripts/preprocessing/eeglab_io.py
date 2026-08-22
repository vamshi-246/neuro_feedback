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

from quality_control import (
    AMBIGUOUS_RATING_EVENT,
    MISSING_EVENT,
    MISSING_RATING,
    MISSING_RATING_EVENT,
    RATING_OK,
    rating_value_and_status,
)


def _get(root, field, is_nested):
    return getattr(root, field) if is_nested else root[field]


@dataclass
class Recording:
    dataset_id: str
    subject_id: str
    channel_labels: list       # length nbchan
    srate: float
    data: np.ndarray           # shape (nbchan, pnts, trials), float32, microvolts
    ratings: np.ndarray        # shape (trials,), original numeric rating where available
    trial_ok: np.ndarray       # shape (trials,), bool -- admitted for supervised training
    trial_status: np.ndarray   # shape (trials,), stable rejection reason or "ok"
    event_types: np.ndarray    # shape (trials,), source event/trigger text
    laser_power: np.ndarray    # shape (trials,), stimulus power when supplied
    onset_samples: np.ndarray  # shape (trials,), zero-based, possibly fractional onset


def _event_scalar(event, field: str, *, required: bool = True):
    if not hasattr(event, field):
        if required:
            raise ValueError(f"EEGLAB event is missing required field '{field}'")
        return None

    value = getattr(event, field)
    values = np.asarray(value)
    if values.size == 0:
        if required:
            raise ValueError(f"EEGLAB event field '{field}' is empty")
        return None
    if values.size != 1:
        raise ValueError(
            f"EEGLAB event field '{field}' must be scalar, got shape {values.shape}"
        )
    return values.reshape(-1)[0]


def _event_epoch_index(event, trials: int) -> int:
    raw_epoch = _event_scalar(event, "epoch")
    try:
        epoch = float(raw_epoch)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid EEGLAB event epoch {raw_epoch!r}") from exc

    if not np.isfinite(epoch) or not epoch.is_integer():
        raise ValueError(f"EEGLAB event epoch must be a finite integer, got {raw_epoch!r}")
    epoch_idx = int(epoch) - 1
    if not 0 <= epoch_idx < trials:
        raise ValueError(
            f"EEGLAB event epoch {int(epoch)} is outside valid range 1..{trials}"
        )
    return epoch_idx


def _event_onset_sample(event, epoch_idx: int, pnts: int) -> float:
    raw_latency = _event_scalar(event, "latency")
    try:
        latency = float(raw_latency)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid EEGLAB event latency {raw_latency!r}") from exc

    if not np.isfinite(latency):
        raise ValueError(f"EEGLAB event latency must be finite, got {raw_latency!r}")

    # EEGLAB latency is one-based and continues across concatenated epochs.
    zero_based = latency - epoch_idx * pnts - 1.0
    if not 0.0 <= zero_based <= pnts - 1:
        raise ValueError(
            f"event onset sample {zero_based} is outside epoch sample range 0..{pnts - 1}"
        )
    return zero_based


def _optional_event_float(event, field: str) -> float:
    if not hasattr(event, field):
        return np.nan
    values = np.asarray(getattr(event, field))
    if values.size != 1:
        return np.nan
    try:
        result = float(values.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _event_field_has_value(event, field: str) -> bool:
    return hasattr(event, field) and np.asarray(getattr(event, field)).size > 0


def _select_rating_event(epoch_events: list):
    """Select the target event without rejecting unrelated events in an epoch.

    EEGLAB epochs may contain stimulus, response, and boundary events.  The
    released laser-pain derivatives identify the supervised target through a
    custom rating field; a finite/missing laser-power field is the fallback
    for a target whose rating field is empty.  Ambiguity rejects only this
    trial, never the entire subject.
    """

    rating_candidates = [e for e in epoch_events if _event_field_has_value(e, "rating")]
    if len(rating_candidates) == 1:
        return rating_candidates[0], RATING_OK
    if len(rating_candidates) > 1:
        return None, AMBIGUOUS_RATING_EVENT

    power_candidates = [e for e in epoch_events if _event_field_has_value(e, "laser_power")]
    if len(power_candidates) == 1:
        return power_candidates[0], RATING_OK
    if len(power_candidates) > 1:
        return None, AMBIGUOUS_RATING_EVENT

    if len(epoch_events) == 1:
        return epoch_events[0], RATING_OK
    return None, MISSING_RATING_EVENT


def load_recording(dataset_id: str, subject_id: str, set_path: str, fdt_path: str) -> Recording:
    mat = sio.loadmat(set_path, struct_as_record=False, squeeze_me=True)
    is_nested = "EEG" in mat
    root = mat["EEG"] if is_nested else mat

    nbchan = int(_get(root, "nbchan", is_nested))
    pnts = int(_get(root, "pnts", is_nested))
    trials = int(_get(root, "trials", is_nested))
    srate = float(_get(root, "srate", is_nested))

    if nbchan <= 0 or pnts <= 0 or trials <= 0:
        raise ValueError(
            f"{dataset_id}/{subject_id}: invalid dimensions "
            f"nbchan={nbchan}, pnts={pnts}, trials={trials}"
        )
    if not np.isfinite(srate) or srate <= 0:
        raise ValueError(f"{dataset_id}/{subject_id}: invalid sampling rate {srate}")

    chanlocs = np.atleast_1d(_get(root, "chanlocs", is_nested))
    channel_labels = [str(c.labels) for c in chanlocs]
    if len(channel_labels) != nbchan:
        raise ValueError(
            f"{dataset_id}/{subject_id}: {len(channel_labels)} channel labels for "
            f"nbchan={nbchan}"
        )

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
    trial_status = np.full(trials, MISSING_RATING, dtype=object)
    event_types = np.full(trials, "", dtype=object)
    laser_power = np.full(trials, np.nan, dtype=float)
    onset_samples = np.full(trials, np.nan, dtype=float)
    events_by_epoch = [[] for _ in range(trials)]

    for e in events:
        epoch_idx = _event_epoch_index(e, trials)
        events_by_epoch[epoch_idx].append(e)

    for epoch_idx, epoch_events in enumerate(events_by_epoch):
        if not epoch_events:
            trial_status[epoch_idx] = MISSING_EVENT
            continue

        event, selection_status = _select_rating_event(epoch_events)
        if event is None:
            trial_status[epoch_idx] = selection_status
            continue

        onset_samples[epoch_idx] = _event_onset_sample(event, epoch_idx, pnts)
        raw_type = _event_scalar(event, "type", required=False)
        event_types[epoch_idx] = "" if raw_type is None else str(raw_type)
        laser_power[epoch_idx] = _optional_event_float(event, "laser_power")

        raw_rating = getattr(event, "rating", None)
        rating, status = rating_value_and_status(raw_rating)
        ratings[epoch_idx] = rating
        trial_status[epoch_idx] = status

    trial_ok = trial_status == RATING_OK

    return Recording(
        dataset_id=dataset_id,
        subject_id=subject_id,
        channel_labels=channel_labels,
        srate=srate,
        data=data,
        ratings=ratings,
        trial_ok=trial_ok,
        trial_status=trial_status,
        event_types=event_types,
        laser_power=laser_power,
        onset_samples=onset_samples,
    )


def pick_channels(labels: list, wanted: list) -> list:
    """Case-insensitive lookup; raises if any wanted channel is missing."""
    indices = []
    for w in wanted:
        key = w.strip().upper()
        matches = [i for i, label in enumerate(labels) if label.strip().upper() == key]
        if not matches:
            raise KeyError(f"channel '{w}' not found among {labels}")
        if len(matches) > 1:
            raise ValueError(
                f"channel '{w}' is ambiguous: found at indices {matches} among {labels}"
            )
        indices.append(matches[0])
    return indices
