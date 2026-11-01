"""Safely predownload OpenNeuro derivative files with bounded parallelism.

This command only fills and verifies the local cache. It does not load EEG,
extract features, write a feature archive, or train the LSTM. Each subject is
one worker job; that job downloads its small .set file and then its .fdt file.

Example (from the repository root):
    python -B scripts/preprocessing/download_dataset.py \
        --datasets ds005280 --all-subjects --cache-dir data_cache --workers 4
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import sys
import threading

from build_dataset import DOWNLOAD_TIMEOUT_S, _download
from dataset_registry import DOWNLOAD_REGISTRY, s3_fdt_url, s3_set_url, subject_ids

MAX_WORKERS = 8
DEFAULT_WORKERS = 4
DEFAULT_ATTEMPTS = 3


@dataclass(frozen=True)
class SubjectDownloadJob:
    dataset_id: str
    subject_id: str
    set_url: str
    fdt_url: str
    set_path: str
    fdt_path: str


@dataclass(frozen=True)
class SubjectDownloadResult:
    job: SubjectDownloadJob
    set_action: str
    fdt_action: str


def worker_count(value) -> int:
    """Argparse validator that limits simultaneous network transfers."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("worker count must be an integer") from exc
    if not 1 <= parsed <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(
            f"worker count must be between 1 and {MAX_WORKERS}"
        )
    return parsed


def positive_int(value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_subject_jobs(
    dataset_ids,
    cache_dir: str,
    *,
    all_subjects: bool,
    subjects_per_dataset: int,
):
    """Create a deterministic, duplicate-free list of subject download jobs."""

    selected = list(dataset_ids)
    if not selected:
        raise ValueError("at least one dataset must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError("dataset IDs must not be repeated")
    if not all_subjects and subjects_per_dataset <= 0:
        raise ValueError("subjects_per_dataset must be positive")

    jobs = []
    destinations = set()
    for dataset_id in selected:
        if dataset_id not in DOWNLOAD_REGISTRY:
            raise ValueError(f"unknown dataset: {dataset_id}")
        spec = DOWNLOAD_REGISTRY[dataset_id]
        selected_subjects = subject_ids(spec)
        if not all_subjects:
            selected_subjects = selected_subjects[:subjects_per_dataset]
        for subject_id in selected_subjects:
            cache_stem = f"{subject_id}_{spec.derivative_stage}"
            dataset_cache = os.path.join(cache_dir, dataset_id)
            set_path = os.path.join(dataset_cache, f"{cache_stem}.set")
            fdt_path = os.path.join(dataset_cache, f"{cache_stem}.fdt")
            for destination in (set_path, fdt_path):
                normalized = os.path.normcase(os.path.abspath(destination))
                if normalized in destinations:
                    raise AssertionError(f"duplicate cache destination: {destination}")
                destinations.add(normalized)
            jobs.append(
                SubjectDownloadJob(
                    dataset_id=dataset_id,
                    subject_id=subject_id,
                    set_url=s3_set_url(spec, subject_id),
                    fdt_url=s3_fdt_url(spec, subject_id),
                    set_path=set_path,
                    fdt_path=fdt_path,
                )
            )
    return jobs


def _download_with_retries(
    url: str,
    path: str,
    *,
    attempts: int,
    timeout: float,
    stop_event: threading.Event,
    force: bool,
) -> str:
    last_error = None
    for attempt in range(1, attempts + 1):
        if stop_event.is_set():
            raise InterruptedError(f"download cancelled before completing {path}")
        try:
            return _download(
                url,
                path,
                timeout=timeout,
                stop_event=stop_event,
                force=force,
            )
        except InterruptedError:
            raise
        except Exception as exc:  # a later attempt may recover a transient network error
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"[RETRY {attempt + 1}/{attempts}] {os.path.basename(path)}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if stop_event.wait(min(float(attempt), 3.0)):
                raise InterruptedError(f"download cancelled before completing {path}")
    raise last_error


def download_subject(
    job: SubjectDownloadJob,
    *,
    attempts: int,
    timeout: float,
    stop_event: threading.Event,
    force: bool = False,
) -> SubjectDownloadResult:
    print(f"[START] {job.dataset_id}/{job.subject_id}", flush=True)
    set_action = _download_with_retries(
        job.set_url,
        job.set_path,
        attempts=attempts,
        timeout=timeout,
        stop_event=stop_event,
        force=force,
    )
    fdt_action = _download_with_retries(
        job.fdt_url,
        job.fdt_path,
        attempts=attempts,
        timeout=timeout,
        stop_event=stop_event,
        force=force,
    )
    return SubjectDownloadResult(job, set_action, fdt_action)


def run_downloads(
    jobs,
    *,
    workers: int,
    attempts: int,
    timeout: float,
    force: bool = False,
) -> int:
    """Run subject downloads and return a process-style status code."""

    if not jobs:
        raise ValueError("there are no subject jobs to download")
    workers = worker_count(workers)
    attempts = positive_int(attempts)
    timeout = positive_float(timeout)
    stop_event = threading.Event()
    failures = []
    completed = 0
    executor = None
    futures = {}
    interrupted = False
    try:
        executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="openneuro"
        )
        for job in jobs:
            future = executor.submit(
                download_subject,
                job,
                attempts=attempts,
                timeout=timeout,
                stop_event=stop_event,
                force=force,
            )
            futures[future] = job

        for future in as_completed(futures):
            job = futures[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:  # report every subject, then allow a resumable rerun
                failures.append((job, exc))
                print(
                    f"[FAIL {completed}/{len(jobs)}] {job.dataset_id}/{job.subject_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            else:
                print(
                    f"[OK {completed}/{len(jobs)}] {job.dataset_id}/{job.subject_id}: "
                    f"set={result.set_action}, fdt={result.fdt_action}",
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        for future in futures:
            future.cancel()
        print(
            "\nStop requested. Active workers are closing temporary files; "
            f"verified cache files will remain safe. This can take up to {timeout:g} seconds.",
            flush=True,
        )
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=stop_event.is_set())

    if interrupted:
        return 130

    if failures:
        print(
            f"\nDownload finished with {len(failures)} failed subject(s). "
            "Run the same command again; verified files will be reused.",
            flush=True,
        )
        return 1
    print(
        f"\nDownload cache verified for all {len(jobs)} subject(s).",
        flush=True,
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely predownload and size-check OpenNeuro derivative files."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=sorted(DOWNLOAD_REGISTRY),
        help="Explicit dataset IDs to predownload.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all-subjects",
        action="store_true",
        help="Download every registered subject in each selected dataset.",
    )
    selection.add_argument(
        "--subjects-per-dataset",
        type=positive_int,
        default=1,
        help="Download the first N subjects per dataset (default: 1).",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(
            os.environ.get("TEMP", "."), "neuro_feedback_pipeline_cache"
        ),
    )
    parser.add_argument(
        "--workers",
        type=worker_count,
        default=DEFAULT_WORKERS,
        help=f"Parallel subject downloads, 1-{MAX_WORKERS} (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--attempts",
        type=positive_int,
        default=DEFAULT_ATTEMPTS,
        help=f"Attempts per file after network errors (default: {DEFAULT_ATTEMPTS}).",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DOWNLOAD_TIMEOUT_S,
        help=f"Network timeout in seconds (default: {DOWNLOAD_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even files whose local and remote sizes already match.",
    )
    args = parser.parse_args(argv)
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("--datasets must not contain the same dataset more than once")

    jobs = build_subject_jobs(
        args.datasets,
        args.cache_dir,
        all_subjects=args.all_subjects,
        subjects_per_dataset=args.subjects_per_dataset,
    )
    print(
        f"Planned {len(jobs)} subject(s), {2 * len(jobs)} files, "
        f"using {args.workers} worker(s).",
        flush=True,
    )
    return run_downloads(
        jobs,
        workers=args.workers,
        attempts=args.attempts,
        timeout=args.timeout,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
