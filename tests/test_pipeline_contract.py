"""Tests for the feature-archive and split contracts; no data build or training."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import torch


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))

from build_dataset import (  # noqa: E402
    _atomic_save_npz,
    _atomic_write_qc_report,
    _default_qc_path,
    _failed_qc_path,
    _normalize_npz_path,
    _qc_reference,
    _same_path,
)
from train_lstm import (  # noqa: E402
    PainLSTM,
    _atomic_torch_save,
    base_trial_weights,
    build_split_audit,
    decode_task_logits,
    evaluate,
    evaluation_breakdown,
    load_feature_archive,
    load_checkpoint_split,
    make_training_loader,
    model_feature_view,
    normalize_selection_datasets,
    ordinal_metrics,
    ordinal_targets_numpy,
    ordinal_training_loss_weights,
    prediction_payload,
    rating_boundary_breakdown,
    redact_test_label_counts,
    select_dataset_view,
    subject_split,
    training_trial_weights,
    validate_feature_archive,
)


def valid_archive() -> dict:
    ratings = np.asarray([2.0, 5.0, 8.0], dtype=np.float64)
    channel_power = np.arange(1, 3 * 3 * 5 * 3 + 1, dtype=np.float64).reshape(3, 3, 5, 3)
    channel_baseline = np.arange(1, 3 * 5 * 3 + 1, dtype=np.float64).reshape(3, 5, 3)
    weights = channel_baseline[:, np.newaxis, :, :]
    averaged_power = np.sum(channel_power * weights, axis=2) / np.sum(weights, axis=2)
    expected_subjects = [
        ["dataset", "sub-001"], ["dataset", "sub-002"], ["dataset", "sub-003"]
    ]
    qc_summary = {
        "schema_version": 1,
        "build_id": "test-build",
        "build_status": "complete",
        "archive_written": True,
        "selected_datasets": ["dataset"],
        "channels": ["Fz", "Cz", "C3", "C4", "Pz"],
        "subject_failures": [],
        "expected_subject_keys": expected_subjects,
        "successful_subject_keys": expected_subjects,
        "subjects": [
            {
                "dataset_id": "dataset",
                "subject_id": subject,
                "input_trials": 2 if subject == "sub-001" else 1,
                "accepted_trials": 1,
                "rejected_trials": 1 if subject == "sub-001" else 0,
                "rejections": {"missing_rating": 1} if subject == "sub-001" else {},
                "rejected_trial_details": (
                    [{"epoch_index": 2, "reason": "missing_rating"}]
                    if subject == "sub-001"
                    else []
                ),
            }
            for _, subject in expected_subjects
        ],
    }
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "build_id": np.asarray("test-build"),
        "archive_complete": np.asarray(True, dtype=np.bool_),
        "qc_report_relpath": np.asarray("features_qc.json"),
        "qc_summary_json": np.asarray(json.dumps(qc_summary, sort_keys=True)),
        "selected_dataset_ids": np.asarray(["dataset"]),
        "expected_subject_keys": np.asarray(expected_subjects),
        "successful_subject_keys": np.asarray(expected_subjects),
        "failed_subject_keys": np.empty((0, 2), dtype=str),
        "relative_power": averaged_power,
        "channel_relative_power": channel_power,
        "channel_baseline_power": channel_baseline,
        "ratings": ratings,
        "labels": np.asarray([0, 1, 2], dtype=np.int64),
        "subject_id": np.asarray(["sub-001", "sub-002", "sub-003"]),
        "dataset_id": np.asarray(["dataset", "dataset", "dataset"]),
        "epoch_index": np.asarray([1, 1, 1], dtype=np.int64),
        "event_type": np.asarray(["laser", "laser", "laser"]),
        "laser_power": np.asarray([3.0, 3.5, 4.0], dtype=np.float64),
        "event_onset_sample": np.asarray([1000.0, 1000.25, 999.75]),
        "feature_onset_sample": np.asarray([1000, 1000, 1000], dtype=np.int64),
        "channel_order": np.asarray(["Fz", "Cz", "C3", "C4", "Pz"]),
        "band_order": np.asarray(["alpha", "beta", "theta"]),
        "baseline_window_s": np.asarray([-0.5, 0.0]),
        "post_windows_s": np.asarray([[0.0, 0.3], [0.3, 0.6], [0.6, 1.0]]),
    }


class FeatureArchiveContractTests(unittest.TestCase):
    def test_valid_archive_passes_and_is_copied(self):
        source = valid_archive()
        validated = validate_feature_archive(source)
        self.assertEqual(validated["relative_power"].shape, (3, 3, 3))
        self.assertIsNot(validated["relative_power"], source["relative_power"])

    def test_invalid_rating_label_and_duplicate_identity_are_rejected(self):
        archive = valid_archive()
        archive["ratings"][0] = 18.0
        with self.assertRaisesRegex(ValueError, "ratings"):
            validate_feature_archive(archive)

        archive = valid_archive()
        archive["labels"] = archive["labels"].astype(np.float64)
        archive["labels"][0] = 1.9
        with self.assertRaisesRegex(ValueError, "exact finite integers"):
            validate_feature_archive(archive)

        archive = valid_archive()
        archive["labels"][0] = 2
        with self.assertRaisesRegex(ValueError, "rating-bin"):
            validate_feature_archive(archive)

        archive = valid_archive()
        archive["subject_id"][1] = archive["subject_id"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_feature_archive(archive)

    def test_missing_key_or_wrong_feature_shape_is_rejected(self):
        archive = valid_archive()
        del archive["event_type"]
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            validate_feature_archive(archive)

        archive = valid_archive()
        archive["relative_power"] = np.ones((3, 9))
        with self.assertRaisesRegex(ValueError, "relative_power"):
            validate_feature_archive(archive)

    def test_manifest_must_match_the_trial_subjects(self):
        archive = valid_archive()
        archive["subject_id"][2] = "sub-999"
        with self.assertRaisesRegex(ValueError, "trial subjects"):
            validate_feature_archive(archive)

        archive = valid_archive()
        archive["expected_subject_keys"][1] = archive["expected_subject_keys"][0]
        archive["successful_subject_keys"][1] = archive["successful_subject_keys"][0]
        with self.assertRaisesRegex(ValueError, "duplicate subject"):
            validate_feature_archive(archive)

        archive = valid_archive()
        qc = json.loads(archive["qc_summary_json"].item())
        qc["subjects"][0]["accepted_trials"] = 2
        qc["subjects"][0]["input_trials"] = 3
        archive["qc_summary_json"] = np.asarray(json.dumps(qc))
        with self.assertRaisesRegex(ValueError, "accepted count"):
            validate_feature_archive(archive)

    def test_partial_archive_and_fractional_schema_are_rejected(self):
        archive = valid_archive()
        archive["archive_complete"] = np.asarray(False, dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "partial"):
            validate_feature_archive(archive)

        archive = valid_archive()
        archive["schema_version"] = np.asarray(1.9)
        with self.assertRaisesRegex(ValueError, "exact finite integers"):
            validate_feature_archive(archive)

    def test_archive_and_qc_report_build_ids_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            archive_path = directory / "features.npz"
            qc_path = directory / "features_qc.json"
            arrays = valid_archive()
            np.savez(archive_path, **arrays)
            qc_path.write_text(json.dumps({"build_id": "test-build"}), encoding="utf-8")
            loaded = load_feature_archive(str(archive_path))
            self.assertEqual(loaded["relative_power"].shape, (3, 3, 3))

            report = json.loads(qc_path.read_text(encoding="utf-8"))
            report["build_id"] = "different-build"
            qc_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertWarnsRegex(RuntimeWarning, "does not match"):
                loaded = load_feature_archive(str(archive_path))
            self.assertEqual(loaded["relative_power"].shape, (3, 3, 3))

    def test_per_channel_flatten_order_is_channel_then_band(self):
        archive = validate_feature_archive(valid_archive())
        power, order = model_feature_view(archive, "per-channel")
        self.assertEqual(power.shape, (3, 3, 15))
        np.testing.assert_array_equal(
            power[0, 0], archive["channel_relative_power"][0, 0].reshape(-1)
        )
        self.assertEqual(
            order[:6],
            ["Fz:alpha", "Fz:beta", "Fz:theta", "Cz:alpha", "Cz:beta", "Cz:theta"],
        )


class SplitAndEvaluationContractTests(unittest.TestCase):
    def test_dataset_view_is_exact_and_does_not_mutate_source(self):
        archive = {
            "selected_dataset_ids": np.asarray(["a", "b"]),
            "dataset_id": np.asarray(["a", "a", "b"]),
            "subject_id": np.asarray(["s1", "s2", "s3"]),
            "successful_subject_keys": np.asarray(
                [["a", "s1"], ["a", "s2"], ["b", "s3"]]
            ),
        }
        original_ids = archive["dataset_id"].copy()
        mask, selected = select_dataset_view(archive, ["b"])
        self.assertEqual(selected, ["b"])
        np.testing.assert_array_equal(mask, [False, False, True])
        np.testing.assert_array_equal(archive["dataset_id"], original_ids)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_dataset_view(archive, ["a", "a"])
        with self.assertRaisesRegex(ValueError, "absent"):
            select_dataset_view(archive, ["missing"])
        with self.assertRaisesRegex(ValueError, "selection datasets"):
            normalize_selection_datasets(["b"], ["a"])

    def test_checkpoint_split_is_verified_and_reused_exactly(self):
        archive = validate_feature_archive(valid_archive())
        checkpoint = {
            "checkpoint_schema_version": 2,
            "feature_archive_schema_version": 1,
            "feature_archive_build_id": "test-build",
            "selected_dataset_ids": ["dataset"],
            "train_subject_keys": [["dataset", "sub-001"]],
            "val_subject_keys": [["dataset", "sub-002"]],
            "test_subject_keys": [["dataset", "sub-003"]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.pt"
            torch.save(checkpoint, path)
            train, validation, test, provenance = load_checkpoint_split(
                str(path), archive, ["dataset"]
            )
            self.assertEqual(train, {("dataset", "sub-001")})
            self.assertEqual(validation, {("dataset", "sub-002")})
            self.assertEqual(test, {("dataset", "sub-003")})
            self.assertEqual(provenance["mode"], "checkpoint")
            self.assertEqual(len(provenance["checkpoint_sha256"]), 64)

            checkpoint["feature_archive_build_id"] = "wrong-build"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "build ID"):
                load_checkpoint_split(str(path), archive, ["dataset"])

    def test_checkpoint_schema_four_is_accepted_as_a_split_source(self):
        archive = validate_feature_archive(valid_archive())
        checkpoint = {
            "checkpoint_schema_version": 4,
            "feature_archive_schema_version": 1,
            "feature_archive_build_id": "test-build",
            "selected_dataset_ids": ["dataset"],
            "train_subject_keys": [["dataset", "sub-001"]],
            "val_subject_keys": [["dataset", "sub-002"]],
            "test_subject_keys": [["dataset", "sub-003"]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split-v4.pt"
            torch.save(checkpoint, path)
            _, _, _, provenance = load_checkpoint_split(
                str(path), archive, ["dataset"]
            )
            self.assertEqual(provenance["checkpoint_schema_version"], 4)

    def test_validation_only_split_audit_hides_test_labels(self):
        datasets = np.asarray(["d"] * 18)
        subjects = np.repeat([f"s{index}" for index in range(6)], 3)
        labels = np.tile(np.arange(3), 6)
        train, validation, test = subject_split(datasets, subjects, seed=4)
        audit = build_split_audit(
            datasets,
            subjects,
            labels,
            {"train": train, "validation": validation, "test": test},
        )
        redacted = redact_test_label_counts(audit)
        self.assertIsNone(redacted["test"]["class_counts"])
        self.assertIsNone(redacted["test"]["per_dataset"]["d"]["class_counts"])
        self.assertIsNotNone(audit["test"]["class_counts"])

    def test_subject_split_is_seeded_and_disjoint(self):
        dataset = np.asarray(["d"] * 10)
        subjects = np.asarray([f"sub-{i:03d}" for i in range(10)])
        split_a = subject_split(dataset, subjects, seed=123)
        split_b = subject_split(dataset, subjects, seed=123)
        self.assertEqual(split_a, split_b)
        train, val, test = split_a
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(train | val | test, set(zip(dataset.tolist(), subjects.tolist())))

    def test_split_is_stratified_by_dataset_and_ignores_repeated_trials(self):
        datasets = []
        subjects = []
        for dataset, n_subjects in (("large", 10), ("small", 5)):
            for subject_index in range(n_subjects):
                for _ in range(subject_index % 3 + 1):
                    datasets.append(dataset)
                    subjects.append(f"sub-{subject_index:03d}")
        datasets = np.asarray(datasets)
        subjects = np.asarray(subjects)
        train, val, test = subject_split(datasets, subjects, seed=11)

        expected = {
            "large": (6, 2, 2),
            "small": (3, 1, 1),
        }
        for dataset, counts in expected.items():
            self.assertEqual(
                tuple(sum(key[0] == dataset for key in split) for split in (train, val, test)),
                counts,
            )

        unique_datasets = []
        unique_subjects = []
        for dataset, subject in sorted(set(zip(datasets.tolist(), subjects.tolist()))):
            unique_datasets.append(dataset)
            unique_subjects.append(subject)
        self.assertEqual(
            (train, val, test),
            subject_split(
                np.asarray(unique_datasets), np.asarray(unique_subjects), seed=11
            ),
        )

    def test_all_nine_like_split_has_expected_dataset_coverage(self):
        sizes = [29, 26, 30, 65, 29, 223, 142, 39, 95]
        datasets = []
        subjects = []
        labels = []
        for dataset_index, n_subjects in enumerate(sizes):
            dataset = f"d{dataset_index}"
            for subject_index in range(n_subjects):
                for label in range(3):
                    datasets.append(dataset)
                    subjects.append(f"sub-{subject_index:03d}")
                    labels.append(label)
        datasets = np.asarray(datasets)
        subjects = np.asarray(subjects)
        labels = np.asarray(labels)
        train, val, test = subject_split(datasets, subjects, seed=123)
        audit = build_split_audit(
            datasets,
            subjects,
            labels,
            {"train": train, "validation": val, "test": test},
        )
        self.assertEqual(
            tuple(audit[name]["subjects"] for name in ("train", "validation", "test")),
            (406, 136, 136),
        )
        for name in ("train", "validation", "test"):
            self.assertTrue(
                all(row["subjects"] > 0 for row in audit[name]["per_dataset"].values())
            )
            self.assertTrue(all(count > 0 for count in audit[name]["class_counts"]))

    def test_split_audit_rejects_an_absent_class(self):
        datasets = np.asarray(["d"] * 6)
        subjects = np.asarray([f"sub-{index:03d}" for index in range(6)])
        labels = np.zeros(6, dtype=np.int64)
        train, val, test = subject_split(datasets, subjects, seed=3)
        with self.assertRaisesRegex(ValueError, "has no trials for class"):
            build_split_audit(
                datasets,
                subjects,
                labels,
                {"train": train, "validation": val, "test": test},
            )

    def test_subject_and_dataset_base_weight_totals_are_equal(self):
        dataset = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"])
        subject = np.asarray(["s1", "s2", "s2", "s2", "s3", "s3", "s4", "s4"])

        subject_weights = base_trial_weights(dataset, subject, policy="subject-class")
        subject_totals = {
            key: float(subject_weights[(dataset == key[0]) & (subject == key[1])].sum())
            for key in set(zip(dataset.tolist(), subject.tolist()))
        }
        np.testing.assert_allclose(list(subject_totals.values()), 1.0)

        dataset_weights = base_trial_weights(
            dataset, subject, policy="dataset-subject-class"
        )
        np.testing.assert_allclose(
            [dataset_weights[dataset == value].sum() for value in ("a", "b")],
            [1.0, 1.0],
        )

    def test_train_only_class_correction_equalizes_weighted_mass(self):
        dataset = np.asarray(["a"] * 6 + ["b"] * 6)
        subject = np.asarray(
            ["s1", "s1", "s1", "s2", "s2", "s2"]
            + ["s3", "s3", "s3", "s4", "s4", "s4"]
        )
        labels = np.asarray([0, 0, 0, 1, 1, 2, 0, 1, 1, 2, 2, 2])
        weights, correction, full_correction, base_mass = training_trial_weights(
            dataset, subject, labels, policy="dataset-subject-class"
        )
        final_mass = np.bincount(labels, weights=weights, minlength=3)
        np.testing.assert_allclose(final_mass, np.repeat(final_mass.mean(), 3))
        self.assertTrue(np.all(correction > 0))
        np.testing.assert_allclose(correction, full_correction)
        self.assertTrue(np.all(base_mass > 0))

    def test_class_balance_strength_has_exact_endpoints_and_midpoint(self):
        dataset = np.asarray(["a"] * 6 + ["b"] * 6)
        subject = np.asarray(
            ["s1", "s1", "s1", "s2", "s2", "s2"]
            + ["s3", "s3", "s3", "s4", "s4", "s4"]
        )
        labels = np.asarray([0, 0, 0, 1, 1, 2, 0, 1, 1, 2, 2, 2])
        base = base_trial_weights(dataset, subject, policy="dataset-subject-class")

        zero, applied_zero, _, base_mass = training_trial_weights(
            dataset, subject, labels, class_balance_strength=0.0
        )
        np.testing.assert_allclose(applied_zero, np.ones(3))
        np.testing.assert_allclose(zero, base / base.mean())

        one, applied_one, full_one, _ = training_trial_weights(
            dataset, subject, labels, class_balance_strength=1.0
        )
        np.testing.assert_allclose(applied_one, full_one)
        one_mass = np.bincount(labels, weights=one, minlength=3)
        np.testing.assert_allclose(one_mass, np.repeat(one_mass.mean(), 3))

        half, applied_half, full_half, _ = training_trial_weights(
            dataset, subject, labels, class_balance_strength=0.5
        )
        np.testing.assert_allclose(applied_half, 0.5 + 0.5 * full_half)
        half_mass = np.bincount(labels, weights=half, minlength=3)
        expected_mass = 0.5 * base_mass + 0.5 * base.sum() / 3
        np.testing.assert_allclose(
            half_mass / half_mass.sum(), expected_mass / expected_mass.sum()
        )

        for invalid in (-0.1, 1.1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                training_trial_weights(
                    dataset, subject, labels, class_balance_strength=invalid
                )

    def test_ordinal_targets_head_order_and_decoder_are_rank_consistent(self):
        np.testing.assert_array_equal(
            ordinal_targets_numpy(np.asarray([0, 1, 2])),
            [[0, 0], [1, 0], [1, 1]],
        )
        model = PainLSTM(input_size=3, hidden_size=16, task_mode="ordinal")
        thresholds = model.ordinal_thresholds().detach()
        self.assertGreater(float(thresholds[1]), float(thresholds[0]))
        logits = model(torch.zeros(4, 3, 3))
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertTrue(torch.all(logits[:, 0] > logits[:, 1]).item())
        decoded = decode_task_logits(
            torch.tensor([[-1.0, -2.0], [1.0, -1.0], [2.0, 1.0]]),
            "ordinal",
        )
        self.assertEqual(decoded.tolist(), [0, 1, 2])

    def test_default_categorical_model_keeps_historical_head_contract(self):
        model = PainLSTM(input_size=3, hidden_size=16)
        self.assertEqual(model.task_mode, "categorical")
        self.assertEqual(tuple(model(torch.zeros(2, 3, 3)).shape), (2, 3))
        self.assertIn("classifier.weight", model.state_dict())
        self.assertNotIn("ordinal_score.weight", model.state_dict())

    def test_ordinal_balance_is_independent_for_each_binary_threshold(self):
        dataset = np.asarray(["a"] * 6 + ["b"] * 6)
        subject = np.asarray(
            ["s1", "s1", "s1", "s2", "s2", "s2"]
            + ["s3", "s3", "s3", "s4", "s4", "s4"]
        )
        labels = np.asarray([0, 0, 0, 1, 1, 2, 0, 1, 1, 2, 2, 2])
        weights, applied, full, base_mass, counts, final_mass = (
            ordinal_training_loss_weights(
                dataset,
                subject,
                labels,
                policy="dataset-subject-class",
                class_balance_strength=1.0,
            )
        )
        self.assertEqual(weights.shape, (12, 2))
        self.assertEqual(applied.shape, (2, 2))
        self.assertEqual(counts.shape, (2, 2))
        np.testing.assert_allclose(applied, full)
        np.testing.assert_allclose(final_mass[:, 0], final_mass[:, 1])
        self.assertTrue(np.all(base_mass > 0))

    def test_ordinal_metrics_penalize_distance_and_report_severe_errors(self):
        adjacent = ordinal_metrics(
            np.asarray([0, 1, 2]), np.asarray([1, 2, 2])
        )
        severe = ordinal_metrics(
            np.asarray([0, 1, 2]), np.asarray([2, 2, 2])
        )
        self.assertGreater(severe["class_index_mae"], adjacent["class_index_mae"])
        self.assertEqual(adjacent["severe_low_high_error_rate"], 0.0)
        self.assertAlmostEqual(severe["severe_low_high_error_rate"], 1 / 3)
        self.assertLess(
            severe["quadratic_weighted_kappa"],
            adjacent["quadratic_weighted_kappa"],
        )

    def test_rating_boundary_report_and_prediction_payload_keep_exact_ratings(self):
        ratings = np.asarray([4.0, 4.9, 6.9, 7.0, 8.2, 10.0])
        y_true = np.asarray([1, 1, 1, 2, 2, 2])
        y_pred = np.asarray([1, 2, 1, 1, 2, 2])
        report = rating_boundary_breakdown(ratings, y_true, y_pred)
        self.assertEqual(
            [row["interval"] for row in report["rows"]],
            ["[4,5)", "[5,6)", "[6,7)", "[7,8)", "[8,9)", "[9,10]"],
        )
        self.assertEqual([row["trials"] for row in report["rows"]], [2, 0, 1, 1, 1, 1])
        payload = prediction_payload(
            np.asarray(["d"] * 6),
            np.asarray([f"s{i}" for i in range(6)]),
            np.arange(6),
            ratings,
            y_true,
            y_pred,
        )
        self.assertEqual(payload["rating"], ratings.tolist())
        self.assertEqual(payload["predicted_label"], y_pred.tolist())

    def test_seeded_minibatches_cover_every_training_row_once(self):
        X = torch.arange(30, dtype=torch.float32).reshape(10, 3, 1)
        y = torch.arange(10, dtype=torch.int64) % 3
        weights = torch.ones(10, dtype=torch.float32)

        def order_for_new_loader():
            loader = make_training_loader(X, y, weights, batch_size=4, seed=19)
            order = []
            batch_sizes = []
            for X_batch, _, _ in loader:
                order.extend((X_batch[:, 0, 0] / 3).to(torch.int64).tolist())
                batch_sizes.append(len(X_batch))
            return order, batch_sizes

        first_order, first_sizes = order_for_new_loader()
        second_order, second_sizes = order_for_new_loader()
        self.assertEqual(first_order, second_order)
        self.assertEqual(first_sizes, [4, 4, 2])
        self.assertEqual(second_sizes, [4, 4, 2])
        self.assertEqual(sorted(first_order), list(range(10)))

    def test_dataset_macro_metrics_expose_large_dataset_dominance(self):
        y_true = np.zeros(110, dtype=np.int64)
        y_pred = np.concatenate(
            [np.zeros(100, dtype=np.int64), np.ones(10, dtype=np.int64)]
        )
        dataset = np.asarray(["large"] * 100 + ["small"] * 10)
        subject = np.asarray(
            [f"large-{index // 10}" for index in range(100)]
            + [f"small-{index // 5}" for index in range(10)]
        )
        report = evaluation_breakdown(y_true, y_pred, dataset, subject)
        self.assertAlmostEqual(report["pooled"]["accuracy"], 100 / 110)
        self.assertAlmostEqual(report["dataset_macro_accuracy"], 0.5)
        self.assertAlmostEqual(report["dataset_macro_balanced_accuracy"], 0.5)

    def test_evaluation_rejects_empty_misaligned_or_out_of_range_inputs(self):
        with self.assertRaises(ValueError):
            evaluate(np.asarray([]), np.asarray([]))
        with self.assertRaises(ValueError):
            evaluate(np.asarray([0, 1]), np.asarray([0]))
        with self.assertRaises(ValueError):
            evaluate(np.asarray([0, 3]), np.asarray([0, 1]))
        with self.assertRaisesRegex(ValueError, "exact finite integers"):
            evaluate(np.asarray([0.0, 1.9]), np.asarray([0, 1]))

    def test_qc_report_default_is_beside_archive(self):
        self.assertEqual(
            _default_qc_path(str(Path("outputs") / "checkpoint.npz")),
            str(Path("outputs") / "checkpoint_qc.json"),
        )
        self.assertEqual(_normalize_npz_path("checkpoint"), "checkpoint.npz")
        self.assertEqual(_normalize_npz_path("checkpoint.NPZ"), "checkpoint.NPZ")
        self.assertTrue(_same_path("checkpoint.npz", str(Path(".") / "checkpoint.npz")))
        self.assertEqual(
            _qc_reference(
                str(Path("outputs") / "checkpoint.npz"),
                str(Path("outputs") / "checkpoint_qc.json"),
            ),
            "checkpoint_qc.json",
        )
        self.assertEqual(
            _failed_qc_path("checkpoint_qc.json", "1234567890abcdef"),
            "checkpoint_qc_failed_1234567890ab.json",
        )

    def test_failed_atomic_archive_write_preserves_existing_file(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            path.write_bytes(b"old archive")
            with patch("build_dataset.np.savez", side_effect=RuntimeError("planned failure")):
                with self.assertRaisesRegex(RuntimeError, "planned failure"):
                    _atomic_save_npz(str(path), values=np.asarray([1]))
            self.assertEqual(path.read_bytes(), b"old archive")

    def test_failed_reports_and_checkpoint_writes_preserve_valid_files(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            current_qc = directory / "features_qc.json"
            current_qc.write_text("old valid report", encoding="utf-8")
            failed_qc = Path(_failed_qc_path(str(current_qc), "abcdef1234567890"))
            _atomic_write_qc_report(str(failed_qc), {"build_status": "failed"})
            self.assertEqual(current_qc.read_text(encoding="utf-8"), "old valid report")
            self.assertTrue(failed_qc.is_file())

            checkpoint = directory / "model.pt"
            checkpoint.write_bytes(b"old checkpoint")
            with patch("train_lstm.torch.save", side_effect=RuntimeError("planned failure")):
                with self.assertRaisesRegex(RuntimeError, "planned failure"):
                    _atomic_torch_save({"value": 1}, str(checkpoint))
            self.assertEqual(checkpoint.read_bytes(), b"old checkpoint")


if __name__ == "__main__":
    unittest.main()
