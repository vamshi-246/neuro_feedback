"""Build the validated, unquantized feature archive used by the LSTM trainer.

Every admitted trial has the same requested-channel, three-window, three-band
contract. This stage saves both the current channel-averaged view and a
per-channel view. It does not quantize features: quantization bounds must be
fit on training subjects only, after the subject split is known.

Usage (from repo root):
    python scripts/preprocessing/build_dataset.py --subjects-per-dataset 1
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import tempfile
import urllib.request
import uuid

import numpy as np

from dataset_registry import REGISTRY, s3_set_url, s3_fdt_url, subject_ids
from eeglab_io import load_recording, pick_channels
from feature_extraction import (
    BASELINE_WINDOW_S,
    BANDS,
    WINDOWS_S,
    TrialFeatureError,
    bin_rating,
    extract_trial_feature_views,
)
from quality_control import RATING_OK

DEFAULT_CHANNELS = ["Fz", "Cz", "C3", "C4"]
ONSET_TOLERANCE_SAMPLES = 0.51
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_S = 30


@dataclass
class SubjectResult:
    relative_power: np.ndarray
    channel_relative_power: np.ndarray
    channel_baseline_power: np.ndarray
    ratings: np.ndarray
    epoch_index: np.ndarray
    event_type: np.ndarray
    laser_power: np.ndarray
    event_onset_sample: np.ndarray
    feature_onset_sample: np.ndarray
    qc: dict


def _remote_file_size(url: str, timeout: float = DOWNLOAD_TIMEOUT_S) -> int:
    """Read the exact object size without downloading the object body."""

    request = urllib.request.Request(
        url, headers={"Accept-Encoding": "identity"}, method="HEAD"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = response.headers.get("Content-Length")
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise OSError(f"server did not provide a valid Content-Length for {url}") from exc
    if size <= 0:
        raise OSError(f"server reported an invalid file size ({size}) for {url}")
    return size


def _download(
    url: str,
    dest: str,
    timeout: float = DOWNLOAD_TIMEOUT_S,
    *,
    stop_event=None,
    force: bool = False,
) -> str:
    """Ensure one cache file is complete, without exposing a partial replacement.

    The existing destination is reused only when its byte count exactly matches
    OpenNeuro's Content-Length.  New bytes go to a temporary file beside the
    destination.  Only a complete, size-checked temporary file is atomically
    moved into place, so a network failure or Ctrl+C cannot overwrite the last
    cache file.

    Returns ``"cached"``, ``"downloaded"``, or ``"repaired"``.
    """

    def raise_if_cancelled():
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError(f"download cancelled before completing {dest}")

    raise_if_cancelled()
    expected_size = _remote_file_size(url, timeout=timeout)
    raise_if_cancelled()
    destination_existed = os.path.lexists(dest)
    if destination_existed:
        if os.path.islink(dest):
            raise OSError(f"refusing to use a symbolic-link cache destination: {dest}")
        if not os.path.isfile(dest):
            raise OSError(f"download destination exists but is not a file: {dest}")
        if not force and os.path.getsize(dest) == expected_size:
            return "cached"

    directory = os.path.dirname(dest) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(dest)}.", suffix=".part", dir=directory
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            request = urllib.request.Request(
                url, headers={"Accept-Encoding": "identity"}, method="GET"
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                get_length_value = response.headers.get("Content-Length")
                try:
                    get_length = int(get_length_value)
                except (TypeError, ValueError) as exc:
                    raise OSError(
                        f"download response did not provide a valid Content-Length for {url}"
                    ) from exc
                if get_length != expected_size:
                    raise OSError(
                        f"remote size changed for {url}: HEAD reported {expected_size} bytes, "
                        f"GET reported {get_length}"
                    )
                while True:
                    raise_if_cancelled()
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        downloaded_size = os.path.getsize(temp_path)
        if downloaded_size != expected_size:
            raise OSError(
                f"incomplete download for {url}: got {downloaded_size} bytes, "
                f"expected {expected_size}"
            )
        raise_if_cancelled()
        os.replace(temp_path, dest)
        return "repaired" if destination_existed else "downloaded"
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _require_cache_file(path: str):
    """Fail clearly when cache-only mode cannot use a local ordinary file."""

    if not os.path.lexists(path):
        raise FileNotFoundError(f"required cache file is missing: {path}")
    if os.path.islink(path):
        raise OSError(f"refusing to use a symbolic-link cache file: {path}")
    if not os.path.isfile(path):
        raise OSError(f"required cache path is not a file: {path}")
    if os.path.getsize(path) <= 0:
        raise OSError(f"required cache file is empty: {path}")


def process_subject(
    spec,
    subject_id: str,
    cache_dir: str,
    channels=None,
    *,
    cache_only: bool = False,
):
    channels = list(DEFAULT_CHANNELS if channels is None else channels)
    normalized_channels = [str(channel).strip().upper() for channel in channels]
    if not normalized_channels or any(not channel for channel in normalized_channels):
        raise ValueError("channels must contain at least one nonempty label")
    if len(normalized_channels) != len(set(normalized_channels)):
        raise ValueError("channels must not contain duplicate labels")
    cache_stem = f"{subject_id}_{spec.derivative_stage}"
    set_path = os.path.join(cache_dir, spec.dataset_id, f"{cache_stem}.set")
    fdt_path = os.path.join(cache_dir, spec.dataset_id, f"{cache_stem}.fdt")
    if cache_only:
        _require_cache_file(set_path)
        _require_cache_file(fdt_path)
    else:
        _download(s3_set_url(spec, subject_id), set_path)
        _download(s3_fdt_url(spec, subject_id), fdt_path)

    rec = load_recording(spec.dataset_id, subject_id, set_path, fdt_path)
    ch_idx = pick_channels(rec.channel_labels, channels)
    expected_onset = spec.pre_stim_s * rec.srate

    n_trials = rec.data.shape[2]
    relative_power = []
    channel_relative_power = []
    channel_baseline_power = []
    ratings = []
    epoch_index = []
    event_type = []
    laser_power = []
    event_onset_sample = []
    feature_onset_sample = []
    rejection_counts = Counter()
    rejected_trial_details = []

    def record_rejection(trial_index: int, reason: str):
        reason = str(reason).strip()
        if not reason or reason == RATING_OK:
            raise AssertionError(
                f"{spec.dataset_id}/{subject_id}: invalid rejection reason {reason!r} "
                f"for epoch {trial_index + 1}"
            )
        rejection_counts[reason] += 1
        rejected_trial_details.append(
            {"epoch_index": int(trial_index + 1), "reason": reason}
        )

    for t in range(n_trials):
        if not rec.trial_ok[t]:
            record_rejection(t, rec.trial_status[t])
            continue

        stored_onset = float(rec.onset_samples[t])
        if not np.isfinite(stored_onset):
            record_rejection(t, "missing_event_onset")
            continue
        if abs(stored_onset - expected_onset) > ONSET_TOLERANCE_SAMPLES:
            record_rejection(t, "event_onset_mismatch")
            continue
        if not str(rec.event_types[t]).strip():
            record_rejection(t, "missing_event_type")
            continue

        onset_sample = int(round(stored_onset))
        trial_channels = rec.data[ch_idx, :, t]  # (requested channels, pnts)
        try:
            averaged_features, channel_features, baseline_features = extract_trial_feature_views(
                trial_channels, rec.srate, onset_sample
            )
        except TrialFeatureError as exc:
            record_rejection(t, f"feature_{exc.reason}")
            continue

        channel_relative_power.append(channel_features)
        channel_baseline_power.append(baseline_features)
        relative_power.append(averaged_features)
        ratings.append(rec.ratings[t])
        epoch_index.append(t + 1)  # preserve EEGLAB's one-based epoch identity
        event_type.append(str(rec.event_types[t]))
        laser_power.append(rec.laser_power[t])
        event_onset_sample.append(stored_onset)
        feature_onset_sample.append(onset_sample)

    accepted = len(ratings)
    rejected = int(sum(rejection_counts.values()))
    if accepted + rejected != n_trials:
        raise AssertionError(
            f"{spec.dataset_id}/{subject_id}: QC accounting mismatch: "
            f"accepted={accepted}, rejected={rejected}, input={n_trials}"
        )

    qc = {
        "dataset_id": spec.dataset_id,
        "subject_id": subject_id,
        "derivative_stage": spec.derivative_stage,
        "sampling_rate_hz": float(rec.srate),
        "input_trials": int(n_trials),
        "accepted_trials": int(accepted),
        "rejected_trials": rejected,
        "rejections": dict(sorted(rejection_counts.items())),
        "rejected_trial_details": rejected_trial_details,
    }

    if relative_power:
        features_array = np.asarray(relative_power, dtype=np.float64)
        channel_features_array = np.asarray(channel_relative_power, dtype=np.float64)
        baseline_features_array = np.asarray(channel_baseline_power, dtype=np.float64)
    else:
        features_array = np.empty((0, len(WINDOWS_S), len(BANDS)), dtype=np.float64)
        channel_features_array = np.empty(
            (0, len(WINDOWS_S), len(channels), len(BANDS)), dtype=np.float64
        )
        baseline_features_array = np.empty(
            (0, len(channels), len(BANDS)), dtype=np.float64
        )

    return SubjectResult(
        relative_power=features_array,
        channel_relative_power=channel_features_array,
        channel_baseline_power=baseline_features_array,
        ratings=np.asarray(ratings, dtype=np.float64),
        epoch_index=np.asarray(epoch_index, dtype=np.int64),
        event_type=np.asarray(event_type, dtype=str),
        laser_power=np.asarray(laser_power, dtype=np.float64),
        event_onset_sample=np.asarray(event_onset_sample, dtype=np.float64),
        feature_onset_sample=np.asarray(feature_onset_sample, dtype=np.int64),
        qc=qc,
    )


def _default_qc_path(output_path: str) -> str:
    stem, _ = os.path.splitext(output_path)
    return f"{stem}_qc.json"


def _normalize_npz_path(path: str) -> str:
    return path if path.lower().endswith(".npz") else f"{path}.npz"


def _same_path(first: str, second: str) -> bool:
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def _qc_reference(output_path: str, qc_path: str) -> str:
    """Return a portable relative link, or an absolute link across Windows drives."""

    output_directory = os.path.dirname(os.path.abspath(output_path))
    absolute_qc = os.path.abspath(qc_path)
    try:
        return os.path.relpath(absolute_qc, output_directory)
    except ValueError:
        return absolute_qc


def _failed_qc_path(qc_path: str, build_id: str) -> str:
    """Keep a failed attempt separate so an older valid report is not replaced."""

    stem, extension = os.path.splitext(qc_path)
    extension = extension or ".json"
    return f"{stem}_failed_{build_id[:12]}{extension}"


def _atomic_write_qc_report(path: str, report: dict):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp_directory = directory or "."
    descriptor, temp_path = tempfile.mkstemp(
        prefix=".qc_", suffix=".json.tmp", dir=temp_directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def _atomic_save_npz(path: str, **arrays):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp_directory = directory or "."
    descriptor, temp_path = tempfile.mkstemp(
        prefix=".features_", suffix=".npz", dir=temp_directory
    )
    os.close(descriptor)
    try:
        np.savez(temp_path, **arrays)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects-per-dataset", type=int, default=1)
    ap.add_argument(
        "--all-subjects",
        action="store_true",
        help="Process every registered subject in each selected dataset.",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(REGISTRY),
        help="Dataset IDs to process; default is every currently registered dataset.",
    )
    ap.add_argument(
        "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="Case-insensitive EEG channel labels; order is preserved in the archive.",
    )
    ap.add_argument(
        "--allow-subject-skip",
        action="store_true",
        help="Permit a partial archive when a subject fails. Default is fail-closed.",
    )
    ap.add_argument(
        "--cache-dir",
        default=os.path.join(
            os.environ.get("TEMP", "."), "neuro_feedback_pipeline_cache"
        ),
    )
    ap.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "Never access the network; require local cache files. The EEGLAB loader "
            "still validates metadata and the exact FDT byte count."
        ),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "pipeline_dev", "pooled_features.npz"
        ),
    )
    ap.add_argument(
        "--qc-report",
        help="JSON QC report path; defaults beside --out with suffix _qc.json.",
    )
    args = ap.parse_args()

    args.out = _normalize_npz_path(args.out)
    qc_path = args.qc_report or _default_qc_path(args.out)
    if _same_path(args.out, qc_path):
        ap.error("--out and --qc-report must be different files")
    if not args.all_subjects and args.subjects_per_dataset <= 0:
        ap.error("--subjects-per-dataset must be positive; use --all-subjects for a full build")

    selected_dataset_ids = args.datasets or list(REGISTRY)
    if len(selected_dataset_ids) != len(set(selected_dataset_ids)):
        ap.error("--datasets must not contain the same dataset more than once")
    channels = [channel.strip() for channel in args.channels]
    if any(not channel for channel in channels):
        ap.error("--channels cannot contain an empty label")
    normalized_channels = [channel.upper() for channel in channels]
    if len(normalized_channels) != len(set(normalized_channels)):
        ap.error("--channels must not contain duplicate labels")

    build_id = uuid.uuid4().hex
    qc_reference = _qc_reference(args.out, qc_path)
    all_power, all_channel_power, all_channel_baseline = [], [], []
    all_ratings, all_subject, all_dataset = [], [], []
    all_epoch, all_event_type, all_laser_power = [], [], []
    all_event_onset, all_feature_onset = [], []
    subject_qc = []
    subject_failures = []
    expected_subject_keys = []
    successful_subject_keys = []

    for ds_id in selected_dataset_ids:
        spec = REGISTRY[ds_id]
        subs = subject_ids(spec)
        if not args.all_subjects:
            subs = subs[: args.subjects_per_dataset]
        expected_subject_keys.extend([[ds_id, subject_id] for subject_id in subs])
        for subject_id in subs:
            try:
                result = process_subject(
                    spec,
                    subject_id,
                    args.cache_dir,
                    channels=channels,
                    cache_only=args.cache_only,
                )
            except Exception as exc:  # noqa: BLE001 -- report all failures before aborting
                failure = {
                    "dataset_id": ds_id,
                    "subject_id": subject_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                subject_failures.append(failure)
                print(f"[FAIL] {ds_id}/{subject_id}: {type(exc).__name__}: {exc}")
                continue

            subject_qc.append(result.qc)
            if result.ratings.size == 0:
                subject_failures.append(
                    {
                        "dataset_id": ds_id,
                        "subject_id": subject_id,
                        "error_type": "NoAcceptedTrials",
                        "error": "all trials were rejected by QC",
                    }
                )
                print(f"[FAIL] {ds_id}/{subject_id}: all trials were rejected by QC")
                continue

            rejection_text = result.qc["rejections"] or "none"
            print(
                f"[OK] {ds_id}/{subject_id}: {result.ratings.size}/"
                f"{result.qc['input_trials']} trials accepted; rejected={rejection_text}; "
                f"rating range {result.ratings.min():.2f}-{result.ratings.max():.2f}"
            )
            all_power.append(result.relative_power)
            all_channel_power.append(result.channel_relative_power)
            all_channel_baseline.append(result.channel_baseline_power)
            all_ratings.append(result.ratings)
            all_epoch.append(result.epoch_index)
            all_event_type.append(result.event_type)
            all_laser_power.append(result.laser_power)
            all_event_onset.append(result.event_onset_sample)
            all_feature_onset.append(result.feature_onset_sample)
            all_subject.extend([subject_id] * result.ratings.size)
            all_dataset.extend([ds_id] * result.ratings.size)
            successful_subject_keys.append([ds_id, subject_id])

    qc_report = {
        "schema_version": 1,
        "build_id": build_id,
        "build_status": "failed" if subject_failures else "admission_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": os.path.abspath(args.out),
        "selected_datasets": selected_dataset_ids,
        "allow_subject_skip": bool(args.allow_subject_skip),
        "cache_only": bool(args.cache_only),
        "channels": channels,
        "bands_hz": [
            {"name": name, "low": low, "high": high} for name, low, high in BANDS
        ],
        "baseline_window_s": list(BASELINE_WINDOW_S),
        "post_windows_s": [list(window) for window in WINDOWS_S],
        "subjects": subject_qc,
        "subject_failures": subject_failures,
        "expected_subject_keys": expected_subject_keys,
        "successful_subject_keys": successful_subject_keys,
    }
    if subject_failures and not args.allow_subject_skip:
        failed_report_path = _failed_qc_path(qc_path, build_id)
        qc_report["failed_report_path"] = os.path.abspath(failed_report_path)
        _atomic_write_qc_report(failed_report_path, qc_report)
        raise RuntimeError(
            f"refusing to save a partial archive: {len(subject_failures)} subject(s) failed; "
            f"inspect {failed_report_path}; any older valid archive and QC report were left unchanged"
        )
    if not all_power:
        failed_report_path = _failed_qc_path(qc_path, build_id)
        qc_report["failed_report_path"] = os.path.abspath(failed_report_path)
        _atomic_write_qc_report(failed_report_path, qc_report)
        raise RuntimeError(f"no trials passed QC; inspect {failed_report_path}")

    relative_power = np.concatenate(all_power, axis=0)
    channel_relative_power = np.concatenate(all_channel_power, axis=0)
    channel_baseline_power = np.concatenate(all_channel_baseline, axis=0)
    ratings = np.concatenate(all_ratings, axis=0)
    labels = np.asarray([bin_rating(rating) for rating in ratings], dtype=np.int64)
    epoch_index = np.concatenate(all_epoch, axis=0)
    event_type = np.concatenate(all_event_type, axis=0)
    laser_power = np.concatenate(all_laser_power, axis=0)
    event_onset_sample = np.concatenate(all_event_onset, axis=0)
    feature_onset_sample = np.concatenate(all_feature_onset, axis=0)
    subject_array = np.asarray(all_subject, dtype=str)
    dataset_array = np.asarray(all_dataset, dtype=str)

    n_trials = ratings.size
    aligned_lengths = {
        relative_power.shape[0],
        channel_relative_power.shape[0],
        channel_baseline_power.shape[0],
        labels.size,
        epoch_index.size,
        event_type.size,
        laser_power.size,
        event_onset_sample.size,
        feature_onset_sample.size,
        subject_array.size,
        dataset_array.size,
        n_trials,
    }
    if aligned_lengths != {n_trials}:
        raise AssertionError(f"output arrays are not trial-aligned: lengths={aligned_lengths}")

    failed_subject_keys = [
        [failure["dataset_id"], failure["subject_id"]] for failure in subject_failures
    ]
    archive_complete = not subject_failures
    qc_report["build_status"] = "complete" if archive_complete else "partial"
    qc_report["archive_written"] = True
    archive_arrays = dict(
        schema_version=np.asarray(1, dtype=np.int64),
        build_id=np.asarray(build_id),
        archive_complete=np.asarray(archive_complete, dtype=np.bool_),
        qc_report_relpath=np.asarray(qc_reference),
        qc_summary_json=np.asarray(
            json.dumps(qc_report, sort_keys=True, separators=(",", ":"))
        ),
        selected_dataset_ids=np.asarray(selected_dataset_ids, dtype=str),
        expected_subject_keys=np.asarray(expected_subject_keys, dtype=str).reshape(-1, 2),
        successful_subject_keys=np.asarray(successful_subject_keys, dtype=str).reshape(-1, 2),
        failed_subject_keys=np.asarray(failed_subject_keys, dtype=str).reshape(-1, 2),
        relative_power=relative_power,
        channel_relative_power=channel_relative_power,
        channel_baseline_power=channel_baseline_power,
        ratings=ratings,
        labels=labels,
        subject_id=subject_array,
        dataset_id=dataset_array,
        epoch_index=epoch_index,
        event_type=event_type,
        laser_power=laser_power,
        event_onset_sample=event_onset_sample,
        feature_onset_sample=feature_onset_sample,
        channel_order=np.asarray(channels, dtype=str),
        band_order=np.asarray([band[0] for band in BANDS], dtype=str),
        baseline_window_s=np.asarray(BASELINE_WINDOW_S, dtype=np.float64),
        post_windows_s=np.asarray(WINDOWS_S, dtype=np.float64),
    )
    _atomic_save_npz(args.out, **archive_arrays)
    _atomic_write_qc_report(qc_path, qc_report)
    print(f"\nSaved {n_trials} validated trials to {args.out}")
    print(f"Saved QC report to {qc_path}")
    print("Label distribution:", {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})


if __name__ == "__main__":
    main()
