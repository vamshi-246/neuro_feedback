#!/usr/bin/env python3
"""Read ds005285 EEG recordings together with their pain ratings.

The raw BIDS ``events.tsv`` files contain stimulus timing and low/high trigger
codes, but not the participant's 0--10 rating.  The ratings and laser powers
are stored as custom EEGLAB event fields in the processed ``.set`` files under
``derivatives/mark_ica`` and ``derivatives/rerefer``.

The main function for analysis is :func:`load_processed_subject`.  It returns
an array shaped ``(trials, channels, time_samples)`` plus a pandas table whose
rows align exactly with the trials.

Examples
--------
Inspect ratings without loading the large signal array::

    python scripts/read_ds005285.py --subject sub-001 --metadata-only

Load only the no-intervention session and four central channels::

    python scripts/read_ds005285.py \
        --subject sub-001 --session 1 --channels Fz Cz C3 C4

Export those trials to a portable NumPy archive::

    python scripts/read_ds005285.py \
        --subject sub-001 --session 1 --channels Fz Cz C3 C4 \
        --save-npz outputs/sub-001_ses-1_eeg_ratings.npz

Inspect a raw, continuous session (ratings are unavailable in raw BIDS files)::

    python scripts/read_ds005285.py \
        --source raw --subject sub-001 --session 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat


EXPECTED_DATASET_DOI = "doi:10.18112/openneuro.ds005285.v1.0.0"
TASK_LABEL = "29ByANT"
N_SESSIONS = 4
TRIALS_PER_SESSION = 40
EXPECTED_ORIGINAL_TRIALS = N_SESSIONS * TRIALS_PER_SESSION

# Verified from the original-file provenance stored in each raw EEGLAB .set.
SESSION_CONDITIONS = {
    1: "SIT",
    2: "VR",
    3: "VR_cTENS",
    4: "VR_sTENS",
}

EVENT_LABELS = {
    32: ("low", 0),
    64: ("high", 1),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "ds005285-download"


@dataclass
class ProcessedEEG:
    """Processed EEG trials and their aligned metadata."""

    eeg: np.ndarray | None
    times_s: np.ndarray
    channel_names: list[str]
    metadata: pd.DataFrame
    sfreq_hz: float
    unit: str
    source_set: Path


def verify_dataset_root(dataset_root: str | Path) -> Path:
    """Resolve the root and prevent accidental loading of a nearby dataset."""

    root = Path(dataset_root).expanduser().resolve()
    description_path = root / "dataset_description.json"
    participants_path = root / "participants.tsv"

    if not description_path.is_file():
        raise FileNotFoundError(f"Missing {description_path}")
    if not participants_path.is_file():
        raise FileNotFoundError(f"Missing {participants_path}")

    description = json.loads(description_path.read_text(encoding="utf-8"))
    actual_doi = description.get("DatasetDOI")
    if actual_doi != EXPECTED_DATASET_DOI:
        raise ValueError(
            "Dataset lock failed: expected "
            f"{EXPECTED_DATASET_DOI!r}, found {actual_doi!r} in "
            f"{description_path}"
        )
    return root


def normalize_subject(subject: str | int) -> str:
    """Convert ``1``, ``001`` or ``sub-1`` to ``sub-001``."""

    text = str(subject).strip()
    if text.lower().startswith("sub-"):
        text = text[4:]
    if not text.isdigit():
        raise ValueError(f"Invalid subject {subject!r}; expected e.g. sub-001")
    return f"sub-{int(text):03d}"


def _records(value: Any) -> list[Any]:
    """Normalize scipy's one-record/list/array MATLAB representations."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return list(value.ravel())
    return [value]


def _load_eeglab_header(set_path: Path) -> dict[str, Any]:
    """Load EEGLAB metadata without reading the external FDT signal file."""

    if not set_path.is_file():
        raise FileNotFoundError(f"Missing EEGLAB file: {set_path}")
    contents = loadmat(set_path, simplify_cells=True)
    eeg = contents.get("EEG", contents)
    if not isinstance(eeg, dict) or "trials" not in eeg:
        raise ValueError(f"Could not find an EEGLAB structure in {set_path}")
    return eeg


def _event_code(value: Any) -> int | None:
    """Normalize EEGLAB/BIDS forms such as 32, '32', 's32' and '32.0'."""

    text = str(value).strip().lower()
    match = re.fullmatch(r"s?(32|64)(?:\.0+)?", text)
    return int(match.group(1)) if match else None


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_matlab_indices(expression: str) -> list[int]:
    """Parse the simple MATLAB integer/range syntax used by pop_rejepoch."""

    text = expression.strip().strip("[]").strip()
    if not text:
        return []

    result: list[int] = []
    for token in re.split(r"[\s,;]+", text):
        if not token:
            continue
        parts = token.split(":")
        if len(parts) == 1:
            result.append(int(parts[0]))
        elif len(parts) == 2:
            start, stop = map(int, parts)
            step = 1 if stop >= start else -1
            result.extend(range(start, stop + step, step))
        elif len(parts) == 3:
            start, step, stop = map(int, parts)
            if step == 0:
                raise ValueError(f"Invalid zero step in MATLAB range {token!r}")
            end = stop + (1 if step > 0 else -1)
            result.extend(range(start, end, step))
        else:
            raise ValueError(f"Unsupported MATLAB index expression {token!r}")
    return result


def _retained_original_indices(
    history: str,
    retained_count: int,
    original_count: int,
) -> tuple[list[int], list[int]]:
    """Recover original trial numbers from EEGLAB's rejection history.

    ``pop_rejepoch`` indices refer to the dataset state at the time of each
    call.  Applying calls sequentially makes this work even if a file contains
    more than one rejection operation.
    """

    pattern = re.compile(
        r"pop_rejepoch\s*\(\s*EEG\s*,\s*"
        r"(\[[^\]]*\]|[-+]?\d+(?:\s*:\s*[-+]?\d+){0,2})\s*,"
    )
    expressions = pattern.findall(history)
    remaining = list(range(1, original_count + 1))
    rejected_original: list[int] = []

    for expression in expressions:
        current_positions = _parse_matlab_indices(expression)
        if len(set(current_positions)) != len(current_positions):
            raise ValueError(
                f"Duplicate trial indices in pop_rejepoch expression {expression!r}"
            )
        for position in sorted(current_positions, reverse=True):
            if position < 1 or position > len(remaining):
                raise ValueError(
                    f"Rejected trial {position} is outside 1..{len(remaining)}"
                )
            rejected_original.append(remaining.pop(position - 1))

    if len(remaining) != retained_count:
        raise ValueError(
            "Could not reconstruct rejected trials: EEGLAB contains "
            f"{retained_count} epochs, but its history implies {len(remaining)} "
            f"retained from {original_count}."
        )
    return remaining, sorted(rejected_original)


def _raw_session_paths(root: Path, subject: str, session: int) -> dict[str, Path]:
    prefix = f"{subject}_ses-{session}_task-{TASK_LABEL}"
    eeg_dir = root / subject / f"ses-{session}" / "eeg"
    return {
        "set": eeg_dir / f"{prefix}_eeg.set",
        "events": eeg_dir / f"{prefix}_events.tsv",
    }


def _validate_session_provenance(set_path: Path, session: int) -> None:
    eeg = _load_eeglab_header(set_path)
    comments = str(eeg.get("comments", "")).replace("/", "\\").lower()
    expected = f"\\pre\\{SESSION_CONDITIONS[session]}\\".lower()
    if expected not in comments:
        raise ValueError(
            f"Session provenance mismatch for {set_path}: expected {expected!r} "
            f"inside EEGLAB comments, found {comments!r}"
        )


def _raw_laser_table(root: Path, subject: str) -> pd.DataFrame:
    """Build the authoritative 160-row raw event table for one subject."""

    tables: list[pd.DataFrame] = []
    original_index = 1

    for session in range(1, N_SESSIONS + 1):
        paths = _raw_session_paths(root, subject, session)
        if not paths["events"].is_file():
            raise FileNotFoundError(f"Missing BIDS events file: {paths['events']}")
        _validate_session_provenance(paths["set"], session)

        events = pd.read_csv(paths["events"], sep="\t")
        codes = events["value"].map(_event_code)
        laser = events.loc[codes.notna()].copy()
        laser["event_code"] = codes.loc[codes.notna()].astype(int).to_numpy()

        if len(laser) != TRIALS_PER_SESSION:
            raise ValueError(
                f"Expected {TRIALS_PER_SESSION} laser events in "
                f"{paths['events']}, found {len(laser)}"
            )

        laser.insert(0, "original_epoch_index", np.arange(
            original_index,
            original_index + len(laser),
            dtype=int,
        ))
        laser.insert(1, "session", session)
        laser.insert(2, "session_condition", SESSION_CONDITIONS[session])
        laser.insert(
            3,
            "trial_in_session",
            np.arange(1, len(laser) + 1, dtype=int),
        )
        laser = laser.rename(columns={"onset": "onset_s"})
        tables.append(
            laser[
                [
                    "original_epoch_index",
                    "session",
                    "session_condition",
                    "trial_in_session",
                    "event_code",
                    "onset_s",
                    "sample",
                ]
            ]
        )
        original_index += len(laser)

    combined = pd.concat(tables, ignore_index=True)
    if len(combined) != EXPECTED_ORIGINAL_TRIALS:
        raise AssertionError("Internal error while assembling raw laser events")
    return combined


def _participant_row(root: Path, subject: str) -> pd.Series:
    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    matches = participants.loc[participants["participant_id"] == subject]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one participants.tsv row for {subject}, found {len(matches)}"
        )
    return matches.iloc[0]


def _channel_names(eeg: dict[str, Any]) -> list[str]:
    chanlocs = _records(eeg.get("chanlocs"))
    labels = [
        str(record.get("labels", "")).strip()
        for record in chanlocs
        if isinstance(record, dict)
    ]
    expected = int(eeg["nbchan"])
    if len(labels) != expected or any(not label for label in labels):
        raise ValueError(
            f"Expected {expected} valid channel labels, found {len(labels)}"
        )
    return labels


def _processed_metadata(
    root: Path,
    subject: str,
    eeg: dict[str, Any],
) -> pd.DataFrame:
    events = _records(eeg.get("event"))
    retained_count = int(eeg["trials"])
    if len(events) != retained_count:
        raise ValueError(
            f"Expected one EEGLAB event per epoch, found {len(events)} events "
            f"for {retained_count} epochs"
        )

    raw_table = _raw_laser_table(root, subject)
    retained_original, rejected_original = _retained_original_indices(
        history=str(eeg.get("history", "")),
        retained_count=retained_count,
        original_count=len(raw_table),
    )
    retained = (
        raw_table.set_index("original_epoch_index")
        .loc[retained_original]
        .reset_index()
    )

    derivative_codes = [_event_code(record.get("type")) for record in events]
    if any(code is None for code in derivative_codes):
        raise ValueError("A processed epoch has an unknown event type")
    if derivative_codes != retained["event_code"].tolist():
        raise ValueError(
            "Processed event sequence does not match retained raw BIDS events; "
            "refusing to attach possibly misaligned ratings."
        )

    participant = _participant_row(root, subject)
    ratings = np.asarray(
        [_float_or_nan(record.get("rating")) for record in events],
        dtype=float,
    )
    laser_power = np.asarray(
        [_float_or_nan(record.get("laser_power")) for record in events],
        dtype=float,
    )
    if np.any((ratings < 0) | (ratings > 10)):
        raise ValueError("Found a pain rating outside the documented 0..10 scale")

    expected_power = np.where(
        retained["event_code"].to_numpy() == 32,
        float(participant["laser_low"]),
        float(participant["laser_high"]),
    )
    valid_power = ~np.isnan(laser_power)
    if not np.allclose(
        laser_power[valid_power],
        expected_power[valid_power],
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError(
            "Processed laser_power values disagree with participants.tsv"
        )

    retained.insert(0, "subject", subject)
    retained.insert(
        1,
        "retained_epoch_index",
        np.arange(1, retained_count + 1, dtype=int),
    )
    retained["intensity_label"] = retained["event_code"].map(
        lambda code: EVENT_LABELS[int(code)][0]
    )
    retained["intensity_class"] = retained["event_code"].map(
        lambda code: EVENT_LABELS[int(code)][1]
    )
    retained["rating"] = ratings
    retained["laser_power_j"] = laser_power
    retained["laser_low_j"] = float(participant["laser_low"])
    retained["laser_high_j"] = float(participant["laser_high"])
    retained["age"] = int(participant["Age"])
    retained["gender"] = str(participant["Gender"])
    retained["dominant_hand"] = str(participant["Dominant_hand"])
    retained["tens_mv"] = float(participant["TENS"])
    retained.attrs["rejected_original_epochs"] = rejected_original
    return retained


def load_processed_subject(
    subject: str | int,
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    derivative: str = "rerefer",
    session: int | None = None,
    intensity: str | None = None,
    channels: Sequence[str] | None = None,
    load_eeg: bool = True,
    unit: str = "uV",
) -> ProcessedEEG:
    """Load processed epochs and aligned 0--10 ratings for one subject.

    Parameters
    ----------
    subject
        Subject number or label, for example ``1`` or ``"sub-001"``.
    dataset_root
        Path to the ds005285 dataset root.
    derivative
        ``"rerefer"`` (final ICA-cleaned, 30 Hz low-pass, average-reference)
        or ``"mark_ica"`` (components marked but not yet removed).
    session
        Optional original session number 1--4.  Session 1 is the SIT
        no-intervention condition.
    intensity
        Optional ``"low"`` or ``"high"`` filter.
    channels
        Optional channel names such as ``["Fz", "Cz", "C3", "C4"]``.
    load_eeg
        If false, read only metadata/ratings and do not load the FDT array.
    unit
        Output signal unit: ``"uV"`` or ``"V"``.
    """

    root = verify_dataset_root(dataset_root)
    subject_label = normalize_subject(subject)
    if derivative not in {"rerefer", "mark_ica"}:
        raise ValueError("derivative must be 'rerefer' or 'mark_ica'")
    if session is not None and session not in SESSION_CONDITIONS:
        raise ValueError("session must be one of 1, 2, 3, 4 or None")
    if intensity is not None:
        intensity = intensity.lower()
        if intensity not in {"low", "high"}:
            raise ValueError("intensity must be 'low', 'high' or None")
    if unit not in {"uV", "V"}:
        raise ValueError("unit must be 'uV' or 'V'")

    set_path = (
        root
        / "derivatives"
        / derivative
        / f"{subject_label}_{TASK_LABEL}.set"
    )
    eeglab = _load_eeglab_header(set_path)
    all_channel_names = _channel_names(eeglab)
    metadata = _processed_metadata(root, subject_label, eeglab)

    mask = np.ones(len(metadata), dtype=bool)
    if session is not None:
        mask &= metadata["session"].to_numpy() == session
    if intensity is not None:
        mask &= metadata["intensity_label"].to_numpy() == intensity
    metadata = metadata.loc[mask].reset_index(drop=True)

    if channels:
        requested_channels = list(channels)
        missing = sorted(set(requested_channels) - set(all_channel_names))
        if missing:
            raise ValueError(
                f"Unknown channels {missing}; available channels are "
                f"{all_channel_names}"
            )
        pick_indices = [all_channel_names.index(name) for name in requested_channels]
        selected_channel_names = requested_channels
    else:
        pick_indices = list(range(len(all_channel_names)))
        selected_channel_names = all_channel_names

    sfreq_hz = float(eeglab["srate"])
    n_samples = int(eeglab["pnts"])
    times_s = float(eeglab["xmin"]) + np.arange(n_samples) / sfreq_hz
    data: np.ndarray | None = None

    if load_eeg:
        epochs = mne.read_epochs_eeglab(set_path, verbose="ERROR")
        if len(epochs) != int(eeglab["trials"]):
            raise ValueError("MNE and EEGLAB disagree on the number of epochs")
        if epochs.ch_names != all_channel_names:
            raise ValueError("MNE and EEGLAB disagree on channel order")

        data = epochs.get_data(copy=True)[mask][:, pick_indices, :]
        if unit == "uV":
            data *= 1e6
        data = data.astype(np.float32, copy=False)
        times_s = epochs.times.copy()

    return ProcessedEEG(
        eeg=data,
        times_s=times_s,
        channel_names=selected_channel_names,
        metadata=metadata,
        sfreq_hz=sfreq_hz,
        unit=unit,
        source_set=set_path,
    )


def load_raw_session(
    subject: str | int,
    session: int,
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    preload: bool = False,
) -> tuple[mne.io.BaseRaw, pd.DataFrame]:
    """Load one continuous raw session and its 40 stimulus events.

    The returned event table intentionally has ``rating = NaN`` because the
    raw BIDS files do not contain the subjective trial ratings.  Use
    :func:`load_processed_subject` when ratings are required.
    """

    root = verify_dataset_root(dataset_root)
    subject_label = normalize_subject(subject)
    if session not in SESSION_CONDITIONS:
        raise ValueError("session must be one of 1, 2, 3 or 4")

    paths = _raw_session_paths(root, subject_label, session)
    _validate_session_provenance(paths["set"], session)
    raw = mne.io.read_raw_eeglab(
        paths["set"],
        preload=preload,
        verbose="ERROR",
    )

    events = pd.read_csv(paths["events"], sep="\t")
    codes = events["value"].map(_event_code)
    events = events.loc[codes.notna()].copy()
    events["event_code"] = codes.loc[codes.notna()].astype(int).to_numpy()
    events["intensity_label"] = events["event_code"].map(
        lambda code: EVENT_LABELS[int(code)][0]
    )
    events["intensity_class"] = events["event_code"].map(
        lambda code: EVENT_LABELS[int(code)][1]
    )
    participant = _participant_row(root, subject_label)
    events["laser_power_j"] = np.where(
        events["event_code"].to_numpy() == 32,
        float(participant["laser_low"]),
        float(participant["laser_high"]),
    )
    events["rating"] = np.nan
    events.insert(0, "subject", subject_label)
    events.insert(1, "session", session)
    events.insert(2, "session_condition", SESSION_CONDITIONS[session])
    return raw, events.reset_index(drop=True)


def _parse_session(value: str) -> int | None:
    if value.lower() == "all":
        return None
    session = int(value)
    if session not in SESSION_CONDITIONS:
        raise argparse.ArgumentTypeError("session must be all, 1, 2, 3 or 4")
    return session


def _save_processed_npz(path: Path, result: ProcessedEEG) -> None:
    if result.eeg is None:
        raise ValueError("--save-npz requires EEG loading; remove --metadata-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "eeg": result.eeg,
        "times_s": result.times_s,
        "channel_names": np.asarray(result.channel_names, dtype=str),
        "sfreq_hz": np.asarray(result.sfreq_hz),
        "unit": np.asarray(result.unit),
    }
    for column in result.metadata.columns:
        values = result.metadata[column].to_numpy()
        if values.dtype == object:
            values = values.astype(str)
        payload[f"meta_{column}"] = values
    np.savez_compressed(path, **payload)


def _print_processed_summary(result: ProcessedEEG, head: int) -> None:
    metadata = result.metadata
    print(f"Dataset:       {EXPECTED_DATASET_DOI}")
    print(f"Source:        {result.source_set}")
    print(f"Sampling:      {result.sfreq_hz:g} Hz")
    print(f"Time window:   {result.times_s[0]:.3f} to {result.times_s[-1]:.3f} s")
    print(f"Trials:        {len(metadata)}")
    print(f"Channels:      {len(result.channel_names)}")
    print(f"Channel names: {', '.join(result.channel_names)}")
    if result.eeg is None:
        print("EEG array:     not loaded (--metadata-only)")
    else:
        print(f"EEG shape:     {result.eeg.shape} [trials, channels, samples]")
        print(f"EEG unit:      {result.unit}")
    print(
        "Sessions:      "
        + ", ".join(
            f"{number}={SESSION_CONDITIONS[number]}"
            for number in sorted(metadata["session"].unique())
        )
    )
    print(
        "Intensity:     "
        + str(metadata["intensity_label"].value_counts().sort_index().to_dict())
    )
    print(
        f"Ratings:       min={metadata['rating'].min():g}, "
        f"mean={metadata['rating'].mean():.3f}, "
        f"max={metadata['rating'].max():g}"
    )
    rejected = metadata.attrs.get("rejected_original_epochs", [])
    print(f"Rejected:      {rejected or 'none'} (original 1-based epoch numbers)")
    print("\nAligned trial metadata:")
    columns = [
        "subject",
        "retained_epoch_index",
        "original_epoch_index",
        "session",
        "session_condition",
        "trial_in_session",
        "event_code",
        "intensity_label",
        "rating",
        "laser_power_j",
        "onset_s",
        "sample",
    ]
    print(metadata[columns].head(head).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read ds005285 EEG and its aligned pain ratings. Processed "
            "rereferenced epochs are used by default."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"ds005285 root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument(
        "--source",
        choices=("processed", "raw"),
        default="processed",
        help="Processed epochs with ratings, or a raw continuous session",
    )
    parser.add_argument(
        "--derivative",
        choices=("rerefer", "mark_ica"),
        default="rerefer",
        help="Processed EEGLAB stage (default: rerefer)",
    )
    parser.add_argument(
        "--session",
        type=_parse_session,
        default=None,
        metavar="{all,1,2,3,4}",
        help="Original session; 1 is SIT/no-intervention (default: all)",
    )
    parser.add_argument(
        "--intensity",
        choices=("all", "low", "high"),
        default="all",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Optional channel subset, for example --channels Fz Cz C3 C4",
    )
    parser.add_argument(
        "--unit",
        choices=("uV", "V"),
        default="uV",
        help="Processed EEG output unit (default: uV)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Read ratings/metadata without loading the processed FDT array",
    )
    parser.add_argument("--save-npz", type=Path)
    parser.add_argument("--save-metadata-csv", type=Path)
    parser.add_argument("--head", type=int, default=10)
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List available subjects and exit",
    )
    return parser


def _list_subjects(root: Path) -> None:
    verified = verify_dataset_root(root)
    participants = pd.read_csv(verified / "participants.tsv", sep="\t")
    print("\n".join(participants["participant_id"].astype(str)))


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.list_subjects:
            _list_subjects(args.dataset_root)
            return 0

        if args.source == "raw":
            if args.session is None:
                raise ValueError("--source raw requires --session 1, 2, 3 or 4")
            if args.save_npz:
                raise ValueError(
                    "--save-npz currently exports processed epochs only"
                )
            raw, events = load_raw_session(
                args.subject,
                args.session,
                dataset_root=args.dataset_root,
                preload=False,
            )
            if args.channels:
                missing = sorted(set(args.channels) - set(raw.ch_names))
                if missing:
                    raise ValueError(f"Unknown raw channels: {missing}")
                raw.pick(args.channels)
            print(raw)
            print(f"Session: {args.session}={SESSION_CONDITIONS[args.session]}")
            print("Ratings: unavailable in raw BIDS; values are NaN")
            print(events.head(args.head).to_string(index=False))
            if args.save_metadata_csv:
                args.save_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
                events.to_csv(args.save_metadata_csv, index=False)
            return 0

        result = load_processed_subject(
            args.subject,
            dataset_root=args.dataset_root,
            derivative=args.derivative,
            session=args.session,
            intensity=None if args.intensity == "all" else args.intensity,
            channels=args.channels,
            load_eeg=not args.metadata_only,
            unit=args.unit,
        )
        _print_processed_summary(result, args.head)

        if args.save_metadata_csv:
            args.save_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
            result.metadata.to_csv(args.save_metadata_csv, index=False)
            print(f"\nSaved metadata: {args.save_metadata_csv}")
        if args.save_npz:
            _save_processed_npz(args.save_npz, result)
            print(f"Saved EEG + metadata: {args.save_npz}")
        return 0

    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
