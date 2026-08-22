"""Local tests for staged dataset-registry admission; no downloads."""

from pathlib import Path
import sys
import unittest


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))

from dataset_registry import (  # noqa: E402
    CANDIDATE_REGISTRY,
    DOWNLOAD_REGISTRY,
    REGISTRY,
    s3_fdt_url,
    s3_set_url,
    subject_ids,
)


class DatasetRegistryAdmissionTests(unittest.TestCase):
    def test_ds005280_contract(self):
        spec = REGISTRY["ds005280"]
        self.assertEqual(spec.dataset_id, "ds005280")
        self.assertEqual(spec.task_label, "223ByBP")
        self.assertEqual(spec.n_subjects, 223)
        self.assertEqual(spec.derivative_stage, "rerefer")
        self.assertEqual(spec.pre_stim_s, 1.0)
        self.assertEqual(subject_ids(spec)[0], "sub-001")
        self.assertEqual(subject_ids(spec)[-1], "sub-223")
        self.assertEqual(len(subject_ids(spec)), 223)
        self.assertTrue(
            s3_set_url(spec, "sub-001").endswith(
                "/ds005280/derivatives/rerefer/sub-001_223ByBP.set"
            )
        )
        self.assertTrue(
            s3_fdt_url(spec, "sub-001").endswith(
                "/ds005280/derivatives/rerefer/sub-001_223ByBP.fdt"
            )
        )

    def test_ds005292_contract(self):
        spec = REGISTRY["ds005292"]
        self.assertEqual(spec.dataset_id, "ds005292")
        self.assertEqual(spec.task_label, "142ByBiosemi")
        self.assertEqual(spec.n_subjects, 142)
        self.assertEqual(spec.derivative_stage, "rerefer")
        self.assertEqual(spec.pre_stim_s, 1.0)
        self.assertEqual(subject_ids(spec)[0], "sub-001")
        self.assertEqual(subject_ids(spec)[-1], "sub-142")
        self.assertTrue(
            s3_set_url(spec, "sub-001").endswith(
                "/ds005292/derivatives/rerefer/sub-001_142ByBiosemi.set"
            )
        )
        self.assertTrue(
            s3_fdt_url(spec, "sub-001").endswith(
                "/ds005292/derivatives/rerefer/sub-001_142ByBiosemi.fdt"
            )
        )

    def test_ds005289_contract(self):
        spec = REGISTRY["ds005289"]
        self.assertEqual(spec.dataset_id, "ds005289")
        self.assertEqual(spec.task_label, "39ByBP")
        self.assertEqual(spec.n_subjects, 39)
        self.assertEqual(spec.derivative_stage, "rerefer")
        self.assertEqual(spec.pre_stim_s, 1.0)
        self.assertEqual(subject_ids(spec)[0], "sub-001")
        self.assertEqual(subject_ids(spec)[-1], "sub-039")
        self.assertTrue(
            s3_set_url(spec, "sub-001").endswith(
                "/ds005289/derivatives/rerefer/sub-001_39ByBP.set"
            )
        )
        self.assertTrue(
            s3_fdt_url(spec, "sub-001").endswith(
                "/ds005289/derivatives/rerefer/sub-001_39ByBP.fdt"
            )
        )

    def test_ds005293_contract(self):
        spec = REGISTRY["ds005293"]
        self.assertEqual(spec.dataset_id, "ds005293")
        self.assertEqual(spec.task_label, "95ByBP")
        self.assertEqual(spec.n_subjects, 95)
        self.assertEqual(spec.derivative_stage, "rerefer")
        self.assertEqual(spec.pre_stim_s, 1.0)
        self.assertEqual(subject_ids(spec)[0], "sub-001")
        self.assertEqual(subject_ids(spec)[-1], "sub-095")
        self.assertTrue(
            s3_set_url(spec, "sub-001").endswith(
                "/ds005293/derivatives/rerefer/sub-001_95ByBP.set"
            )
        )
        self.assertTrue(
            s3_fdt_url(spec, "sub-001").endswith(
                "/ds005293/derivatives/rerefer/sub-001_95ByBP.fdt"
            )
        )

    def test_every_admitted_dataset_uses_one_final_stage(self):
        self.assertEqual(
            {spec.derivative_stage for spec in REGISTRY.values()},
            {"rerefer"},
        )

    def test_every_downloaded_dataset_is_now_admitted(self):
        self.assertEqual(CANDIDATE_REGISTRY, {})
        self.assertTrue(set(CANDIDATE_REGISTRY).isdisjoint(REGISTRY))
        self.assertEqual(DOWNLOAD_REGISTRY, REGISTRY)


if __name__ == "__main__":
    unittest.main()
