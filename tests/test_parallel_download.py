"""Offline tests for resumable, atomic OpenNeuro cache downloads."""

import argparse
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))

from build_dataset import (  # noqa: E402
    _download,
    _remote_file_size,
    _require_cache_file,
    process_subject,
)
from download_dataset import (  # noqa: E402
    SubjectDownloadResult,
    build_subject_jobs,
    run_downloads,
    worker_count,
)


class FakeResponse:
    def __init__(self, payload=b"", *, content_length=None, read_error=None):
        self._stream = io.BytesIO(payload)
        self._read_error = read_error
        if content_length is None:
            content_length = len(payload)
        self.headers = {"Content-Length": str(content_length)}

    def read(self, size=-1):
        if self._read_error is not None:
            error = self._read_error
            self._read_error = None
            raise error
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stream.close()
        return False


def part_files(directory: Path):
    return list(directory.glob(".*.part"))


class AtomicDownloadTests(unittest.TestCase):
    def test_cache_only_requires_nonempty_ordinary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            missing = directory / "missing.fdt"
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                _require_cache_file(str(missing))

            empty = directory / "empty.fdt"
            empty.touch()
            with self.assertRaisesRegex(OSError, "empty"):
                _require_cache_file(str(empty))

            nested_directory = directory / "not-a-file.fdt"
            nested_directory.mkdir()
            with self.assertRaisesRegex(OSError, "not a file"):
                _require_cache_file(str(nested_directory))

            valid = directory / "valid.fdt"
            valid.write_bytes(b"data")
            self.assertIsNone(_require_cache_file(str(valid)))

    def test_process_subject_cache_only_bypasses_network(self):
        from types import SimpleNamespace

        spec = SimpleNamespace(
            dataset_id="synthetic",
            task_label="synthetic",
            derivative_stage="rerefer",
            pre_stim_s=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            dataset_cache = Path(tmp) / spec.dataset_id
            dataset_cache.mkdir()
            (dataset_cache / "sub-001_rerefer.set").write_bytes(b"set")
            (dataset_cache / "sub-001_rerefer.fdt").write_bytes(b"fdt")
            with patch("build_dataset._download") as download, patch(
                "build_dataset.load_recording",
                side_effect=RuntimeError("local loader reached"),
            ):
                with self.assertRaisesRegex(RuntimeError, "local loader reached"):
                    process_subject(
                        spec,
                        "sub-001",
                        tmp,
                        channels=["Fz"],
                        cache_only=True,
                    )
            download.assert_not_called()

    def test_new_file_is_downloaded_atomically(self):
        complete = b"the-complete-remote-payload"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            destination = directory / "recording.fdt"
            with patch(
                "build_dataset._remote_file_size", return_value=len(complete)
            ), patch(
                "build_dataset.urllib.request.urlopen",
                return_value=FakeResponse(complete),
            ):
                action = _download("https://example.test/recording.fdt", str(destination))
            self.assertEqual(action, "downloaded")
            self.assertEqual(destination.read_bytes(), complete)
            self.assertEqual(part_files(directory), [])

    def test_exact_size_cached_file_is_reused_without_get(self):
        payload = b"complete cache file"
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "recording.fdt"
            destination.write_bytes(payload)
            with patch(
                "build_dataset._remote_file_size", return_value=len(payload)
            ), patch("build_dataset.urllib.request.urlopen") as urlopen:
                action = _download("https://example.test/recording.fdt", str(destination))
            self.assertEqual(action, "cached")
            self.assertEqual(destination.read_bytes(), payload)
            urlopen.assert_not_called()

    def test_partial_file_is_repaired_only_after_complete_temp_exists(self):
        old_partial = b"old-partial"
        complete = b"the-complete-remote-payload"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            destination = directory / "recording.fdt"
            destination.write_bytes(old_partial)
            real_replace = os.replace

            def checked_replace(source, target):
                self.assertEqual(destination.read_bytes(), old_partial)
                self.assertEqual(Path(source).read_bytes(), complete)
                real_replace(source, target)

            with patch(
                "build_dataset._remote_file_size", return_value=len(complete)
            ), patch(
                "build_dataset.urllib.request.urlopen",
                return_value=FakeResponse(complete),
            ), patch("build_dataset.os.replace", side_effect=checked_replace):
                action = _download("https://example.test/recording.fdt", str(destination))

            self.assertEqual(action, "repaired")
            self.assertEqual(destination.read_bytes(), complete)
            self.assertEqual(part_files(directory), [])

    def test_network_failure_preserves_partial_and_removes_temp(self):
        old_partial = b"old-partial"
        complete_size = 100
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            destination = directory / "recording.fdt"
            destination.write_bytes(old_partial)
            response = FakeResponse(
                content_length=complete_size,
                read_error=OSError("planned connection loss"),
            )
            with patch(
                "build_dataset._remote_file_size", return_value=complete_size
            ), patch("build_dataset.urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(OSError, "planned connection loss"):
                    _download("https://example.test/recording.fdt", str(destination))
            self.assertEqual(destination.read_bytes(), old_partial)
            self.assertEqual(part_files(directory), [])

    def test_keyboard_interrupt_preserves_partial_and_removes_temp(self):
        old_partial = b"old-partial"
        complete_size = 100
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            destination = directory / "recording.fdt"
            destination.write_bytes(old_partial)
            response = FakeResponse(
                content_length=complete_size,
                read_error=KeyboardInterrupt(),
            )
            with patch(
                "build_dataset._remote_file_size", return_value=complete_size
            ), patch("build_dataset.urllib.request.urlopen", return_value=response):
                with self.assertRaises(KeyboardInterrupt):
                    _download("https://example.test/recording.fdt", str(destination))
            self.assertEqual(destination.read_bytes(), old_partial)
            self.assertEqual(part_files(directory), [])

    def test_short_download_is_rejected_and_old_file_is_preserved(self):
        old_partial = b"old-partial"
        complete_size = 20
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            destination = directory / "recording.fdt"
            destination.write_bytes(old_partial)
            response = FakeResponse(b"only-ten!!", content_length=complete_size)
            with patch(
                "build_dataset._remote_file_size", return_value=complete_size
            ), patch("build_dataset.urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(OSError, "incomplete download"):
                    _download("https://example.test/recording.fdt", str(destination))
            self.assertEqual(destination.read_bytes(), old_partial)
            self.assertEqual(part_files(directory), [])

    def test_head_and_get_lengths_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            destination = directory / "recording.fdt"
            response = FakeResponse(b"12345", content_length=5)
            with patch(
                "build_dataset._remote_file_size", return_value=10
            ), patch("build_dataset.urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(OSError, "remote size changed"):
                    _download("https://example.test/recording.fdt", str(destination))
            self.assertFalse(destination.exists())
            self.assertEqual(part_files(directory), [])

    def test_get_content_length_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            destination = directory / "recording.fdt"
            response = FakeResponse(b"12345")
            response.headers = {}
            with patch(
                "build_dataset._remote_file_size", return_value=5
            ), patch("build_dataset.urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(OSError, "download response"):
                    _download("https://example.test/recording.fdt", str(destination))
            self.assertFalse(destination.exists())
            self.assertEqual(part_files(directory), [])

    def test_cancelled_download_does_not_make_a_network_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "recording.fdt"
            stop_event = threading.Event()
            stop_event.set()
            with patch("build_dataset._remote_file_size") as remote_size:
                with self.assertRaisesRegex(InterruptedError, "cancelled"):
                    _download(
                        "https://example.test/recording.fdt",
                        str(destination),
                        stop_event=stop_event,
                    )
            remote_size.assert_not_called()
            self.assertFalse(destination.exists())

    def test_remote_content_length_must_be_a_positive_integer(self):
        invalid_values = [None, "", "not-a-number", "0", "-1"]
        for value in invalid_values:
            with self.subTest(value=value):
                response = FakeResponse()
                response.headers = {}
                if value is not None:
                    response.headers["Content-Length"] = value
                with patch(
                    "build_dataset.urllib.request.urlopen", return_value=response
                ):
                    with self.assertRaises(OSError):
                        _remote_file_size("https://example.test/recording.fdt")


class ParallelDownloadPlanningTests(unittest.TestCase):
    def test_ds005280_job_list_is_complete_unique_and_rereferenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = build_subject_jobs(
                ["ds005280"],
                tmp,
                all_subjects=True,
                subjects_per_dataset=1,
            )
        self.assertEqual(len(jobs), 223)
        self.assertEqual(jobs[0].subject_id, "sub-001")
        self.assertEqual(jobs[-1].subject_id, "sub-223")
        self.assertIn("/derivatives/rerefer/sub-001_223ByBP.set", jobs[0].set_url)
        self.assertIn("/derivatives/rerefer/sub-001_223ByBP.fdt", jobs[0].fdt_url)
        self.assertTrue(jobs[0].set_path.endswith("sub-001_rerefer.set"))
        self.assertTrue(jobs[0].fdt_path.endswith("sub-001_rerefer.fdt"))
        destinations = [path for job in jobs for path in (job.set_path, job.fdt_path)]
        self.assertEqual(len(destinations), len(set(map(os.path.normcase, destinations))))

    def test_limited_subject_job_list_is_deterministic(self):
        jobs = build_subject_jobs(
            ["ds005280"],
            "cache",
            all_subjects=False,
            subjects_per_dataset=2,
        )
        self.assertEqual([job.subject_id for job in jobs], ["sub-001", "sub-002"])

    def test_three_new_datasets_share_one_download_plan(self):
        jobs = build_subject_jobs(
            ["ds005292", "ds005289", "ds005293"],
            "cache",
            all_subjects=True,
            subjects_per_dataset=1,
        )
        self.assertEqual(len(jobs), 142 + 39 + 95)
        self.assertEqual((jobs[0].dataset_id, jobs[0].subject_id), ("ds005292", "sub-001"))
        self.assertEqual((jobs[141].dataset_id, jobs[141].subject_id), ("ds005292", "sub-142"))
        self.assertEqual((jobs[142].dataset_id, jobs[142].subject_id), ("ds005289", "sub-001"))
        self.assertEqual((jobs[180].dataset_id, jobs[180].subject_id), ("ds005289", "sub-039"))
        self.assertEqual((jobs[181].dataset_id, jobs[181].subject_id), ("ds005293", "sub-001"))
        self.assertEqual((jobs[-1].dataset_id, jobs[-1].subject_id), ("ds005293", "sub-095"))

    def test_worker_count_is_limited_to_one_through_eight(self):
        for value in (1, "4", 8):
            with self.subTest(value=value):
                self.assertEqual(worker_count(value), int(value))
        for value in (0, -1, 9, "1.5", "not-an-integer"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    worker_count(value)

    def test_parallel_runner_returns_success_without_network(self):
        jobs = build_subject_jobs(
            ["ds005280"],
            "cache",
            all_subjects=False,
            subjects_per_dataset=2,
        )

        def successful(job, **kwargs):
            return SubjectDownloadResult(job, "cached", "downloaded")

        with patch("download_dataset.download_subject", side_effect=successful), redirect_stdout(
            io.StringIO()
        ):
            status = run_downloads(jobs, workers=2, attempts=1, timeout=1)
        self.assertEqual(status, 0)

    def test_parallel_runner_reports_failure_without_losing_successes(self):
        jobs = build_subject_jobs(
            ["ds005280"],
            "cache",
            all_subjects=False,
            subjects_per_dataset=2,
        )

        def one_failure(job, **kwargs):
            if job.subject_id == "sub-002":
                raise OSError("planned subject failure")
            return SubjectDownloadResult(job, "cached", "cached")

        with patch("download_dataset.download_subject", side_effect=one_failure), redirect_stdout(
            io.StringIO()
        ):
            status = run_downloads(jobs, workers=2, attempts=1, timeout=1)
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
