"""Offline unit tests for shared preprocessing admission rules.

These tests use synthetic arrays only.  They do not download data, modify the
feature archive, train the model, or write anywhere outside unittest's own
process.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.io as sio


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))

from eeglab_io import (  # noqa: E402
    Recording,
    _event_epoch_index,
    _event_onset_sample,
    _select_rating_event,
    load_recording,
    pick_channels,
)
from build_dataset import process_subject  # noqa: E402
from feature_extraction import (  # noqa: E402
    TrialFeatureError,
    bin_rating,
    extract_trial_channel_features,
    extract_trial_feature_views,
    extract_trial_features,
    quantize_to_uint8,
)
from quality_control import (  # noqa: E402
    AMBIGUOUS_RATING_EVENT,
    INVALID_RATING,
    MISSING_RATING,
    NONFINITE_RATING,
    OUT_OF_RANGE_RATING,
    RATING_OK,
    rating_value_and_status,
)


class RatingQualityControlTests(unittest.TestCase):
    def test_valid_rating_boundaries_and_bins(self):
        for value, expected_class in [(0, 0), (3.999, 0), (4, 1), (6.999, 1), (7, 2), (10, 2)]:
            parsed, status = rating_value_and_status(value)
            self.assertEqual(status, RATING_OK)
            self.assertEqual(parsed, float(value))
            self.assertEqual(bin_rating(value), expected_class)

    def test_missing_nonfinite_invalid_and_out_of_range_are_distinct(self):
        cases = [
            (None, MISSING_RATING),
            (np.array([]), MISSING_RATING),
            (np.nan, MISSING_RATING),
            (np.inf, NONFINITE_RATING),
            ("not-a-rating", INVALID_RATING),
            (np.array([3.0, 4.0]), INVALID_RATING),
            (-0.01, OUT_OF_RANGE_RATING),
            (10.01, OUT_OF_RANGE_RATING),
            (34, OUT_OF_RANGE_RATING),
        ]
        for value, expected_status in cases:
            with self.subTest(value=value):
                _, status = rating_value_and_status(value)
                self.assertEqual(status, expected_status)
                with self.assertRaises(ValueError):
                    bin_rating(value)


class EeglabMetadataValidationTests(unittest.TestCase):
    @staticmethod
    def _write_recording(directory: Path, *, nested: bool) -> tuple[Path, Path]:
        chanlocs = np.array(
            [{"labels": label} for label in ("Fz", "Cz", "C3", "C4")],
            dtype=object,
        )
        events = np.array(
            [
                {
                    "epoch": 1,
                    "latency": 6.0,
                    "type": "laser",
                    "rating": 5.0,
                    "laser_power": 3.5,
                },
                {"epoch": 1, "latency": 8.0, "type": "response"},
                {
                    "epoch": 2,
                    "latency": 16.5,
                    "type": "laser",
                    "rating": 34.0,
                    "laser_power": 4.0,
                },
            ],
            dtype=object,
        )
        metadata = {
            "nbchan": 4,
            "pnts": 10,
            "trials": 2,
            "srate": 1000.0,
            "chanlocs": chanlocs,
            "event": events,
        }

        set_path = directory / ("nested.set" if nested else "flat.set")
        fdt_path = directory / ("nested.fdt" if nested else "flat.fdt")
        sio.savemat(set_path, {"EEG": metadata} if nested else metadata)

        data = np.arange(4 * 10 * 2, dtype="<f4").reshape((4, 10, 2), order="F")
        data.ravel(order="F").tofile(fdt_path)
        return set_path, fdt_path

    def test_loader_accepts_flat_and_nested_mat_and_rejects_only_bad_trial(self):
        for nested in (False, True):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as tmp:
                set_path, fdt_path = self._write_recording(Path(tmp), nested=nested)
                recording = load_recording("synthetic", "sub-001", set_path, fdt_path)

                self.assertEqual(recording.data.shape, (4, 10, 2))
                self.assertEqual(recording.trial_ok.tolist(), [True, False])
                self.assertEqual(recording.trial_status[1], OUT_OF_RANGE_RATING)
                self.assertEqual(recording.ratings.tolist(), [5.0, 34.0])
                self.assertEqual(recording.event_types.tolist(), ["laser", "laser"])
                self.assertEqual(recording.laser_power.tolist(), [3.5, 4.0])
                np.testing.assert_allclose(recording.onset_samples, [5.0, 5.5])

    def test_epoch_and_latency_map_to_zero_based_onset(self):
        first = SimpleNamespace(epoch=1, latency=6.0)
        second = SimpleNamespace(epoch=2, latency=16.0)
        self.assertEqual(_event_epoch_index(first, trials=2), 0)
        self.assertEqual(_event_epoch_index(second, trials=2), 1)
        self.assertEqual(_event_onset_sample(first, epoch_idx=0, pnts=10), 5)
        self.assertEqual(_event_onset_sample(second, epoch_idx=1, pnts=10), 5)

    def test_invalid_epoch_is_rejected_and_fractional_latency_is_preserved(self):
        with self.assertRaises(ValueError):
            _event_epoch_index(SimpleNamespace(epoch=0, latency=1), trials=2)
        self.assertEqual(
            _event_onset_sample(SimpleNamespace(latency=6.5), epoch_idx=0, pnts=10),
            5.5,
        )
        with self.assertRaises(ValueError):
            _event_onset_sample(SimpleNamespace(latency=0), epoch_idx=0, pnts=10)

    def test_unrelated_events_do_not_make_the_rating_event_ambiguous(self):
        response = SimpleNamespace(type="response")
        target = SimpleNamespace(type="laser", rating=5.0, laser_power=3.5)
        selected, status = _select_rating_event([response, target])
        self.assertIs(selected, target)
        self.assertEqual(status, RATING_OK)

        other_target = SimpleNamespace(type="laser", rating=6.0, laser_power=4.0)
        selected, status = _select_rating_event([target, other_target])
        self.assertIsNone(selected)
        self.assertEqual(status, AMBIGUOUS_RATING_EVENT)

    def test_channel_lookup_is_case_insensitive_but_not_ambiguous(self):
        labels = ["FZ", "cz", "C3", "C4"]
        self.assertEqual(pick_channels(labels, ["Fz", "Cz", "C3", "C4"]), [0, 1, 2, 3])
        with self.assertRaises(KeyError):
            pick_channels(labels, ["Pz"])
        with self.assertRaises(ValueError):
            pick_channels(["Fz", "FZ"], ["Fz"])


class FeatureQualityControlTests(unittest.TestCase):
    @staticmethod
    def synthetic_trial():
        srate = 250.0
        samples = 400
        time = np.arange(samples) / srate
        channels = np.stack(
            [
                np.sin(2 * np.pi * 6 * time + phase)
                + 0.7 * np.sin(2 * np.pi * 10 * time + phase)
                + 0.4 * np.sin(2 * np.pi * 20 * time + phase)
                for phase in (0.0, 0.2, 0.4, 0.6)
            ]
        )
        return channels, srate, 125

    def test_feature_contract_is_finite_three_by_three(self):
        trial, srate, onset = self.synthetic_trial()
        channel_features = extract_trial_channel_features(trial, srate, onset)
        features = extract_trial_features(trial, srate, onset)
        feature_view, channel_view, baseline = extract_trial_feature_views(
            trial, srate, onset
        )
        self.assertEqual(channel_features.shape, (3, 4, 3))
        self.assertEqual(features.shape, (3, 3))
        self.assertEqual(baseline.shape, (4, 3))
        np.testing.assert_allclose(channel_features, channel_view)
        np.testing.assert_allclose(features, feature_view)
        expected = np.sum(
            channel_view * baseline[np.newaxis, :, :], axis=1
        ) / np.sum(baseline, axis=0)
        np.testing.assert_allclose(features, expected)
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue((features >= 0).all())

    def test_nonfinite_eeg_out_of_bounds_window_and_flat_baseline_are_rejected(self):
        trial, srate, onset = self.synthetic_trial()

        nonfinite = trial.copy()
        nonfinite[0, 10] = np.nan
        with self.assertRaisesRegex(TrialFeatureError, "NaN or infinite") as caught:
            extract_trial_features(nonfinite, srate, onset)
        self.assertEqual(caught.exception.reason, "nonfinite_eeg")

        with self.assertRaisesRegex(TrialFeatureError, "outside trial length") as caught:
            extract_trial_features(trial, srate, 100)
        self.assertEqual(caught.exception.reason, "window_out_of_bounds")

        with self.assertRaisesRegex(TrialFeatureError, "baseline.*power") as caught:
            extract_trial_features(np.zeros_like(trial), srate, onset)
        self.assertEqual(caught.exception.reason, "baseline_power_too_small")

    def test_quantization_rejects_zero_denominator_or_invalid_input(self):
        raw = np.ones((2, 3, 3), dtype=float)
        low = np.zeros((3, 3), dtype=float)
        high = np.ones((3, 3), dtype=float)
        quantized = quantize_to_uint8(raw, low, high)
        self.assertEqual(quantized.shape, raw.shape)
        self.assertEqual(quantized.dtype, np.uint8)

        with self.assertRaisesRegex(ValueError, "high bound"):
            quantize_to_uint8(raw, low, low)

        invalid = raw.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite, nonnegative"):
            quantize_to_uint8(invalid, low, high)

    def test_subject_processing_supports_five_channels_and_counts_rejections(self):
        four_channels, srate, onset = self.synthetic_trial()
        five_channels = np.vstack(
            [four_channels, 0.8 * four_channels[0] + 0.2 * four_channels[1]]
        )
        recording = Recording(
            dataset_id="synthetic",
            subject_id="sub-001",
            channel_labels=["Fz", "Cz", "C3", "C4", "Pz"],
            srate=srate,
            data=np.stack([five_channels, five_channels, five_channels], axis=2),
            ratings=np.asarray([5.0, 34.0, 6.0]),
            trial_ok=np.asarray([True, False, True]),
            trial_status=np.asarray([RATING_OK, OUT_OF_RANGE_RATING, RATING_OK]),
            event_types=np.asarray(["laser", "laser", ""]),
            laser_power=np.asarray([3.0, 4.0, 3.5]),
            onset_samples=np.asarray([float(onset), float(onset), float(onset)]),
        )
        spec = SimpleNamespace(
            dataset_id="synthetic",
            task_label="synthetic",
            derivative_stage="mark_ica",
            pre_stim_s=0.5,
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "build_dataset._download"
        ), patch("build_dataset.load_recording", return_value=recording):
            result = process_subject(
                spec,
                "sub-001",
                tmp,
                channels=["Fz", "Cz", "C3", "C4", "Pz"],
            )

        self.assertEqual(result.relative_power.shape, (1, 3, 3))
        self.assertEqual(result.channel_relative_power.shape, (1, 3, 5, 3))
        self.assertEqual(result.channel_baseline_power.shape, (1, 5, 3))
        self.assertEqual(result.qc["accepted_trials"], 1)
        self.assertEqual(result.qc["rejected_trials"], 2)
        self.assertEqual(
            result.qc["rejections"],
            {"missing_event_type": 1, "out_of_range_rating": 1},
        )
        self.assertEqual(
            result.qc["rejected_trial_details"],
            [
                {"epoch_index": 2, "reason": "out_of_range_rating"},
                {"epoch_index": 3, "reason": "missing_event_type"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
