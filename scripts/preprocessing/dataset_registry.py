"""
Registry of the OpenNeuro laser-pain datasets this pipeline knows how to read.

Every dataset below comes from the same lab (Institute of Psychology, CAS) and
follows the same processing convention: the lab's own MATLAB pipeline hides the
real 0-10 pain rating inside `derivatives/mark_ica/sub-XXX_<task>.set` (paired
with a `.fdt` file holding the actual voltage samples) as an `EEG.event().rating`
field. The raw `sub-XXX/.../events.tsv` files never carry the rating, only the
stimulus trigger code -- see DS005285_LSTM_ARCHITECTURE.md and the chat history
for how this was discovered and verified.

Only the 5 datasets with a steady, predictable gap between laser pulses and no
known structural bugs are listed here (the "clean five" chosen after reviewing
all 9 candidates: ds005285, ds005284, ds005286, ds005291, ds005473). The other
4 (ds005289, ds005292, ds005280, ds005293) are deliberately left out for now --
see the chat notes on why (wildly uneven trial timing, or a trigger-code bug
that silently drops most trials).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str        # OpenNeuro accession, e.g. "ds005285"
    task_label: str        # BIDS task name suffix used in filenames, e.g. "29ByANT"
    n_subjects: int        # subjects known to exist in the release (sub-001..sub-NNN)
    derivative_stage: str  # which derivatives/ subfolder carries the rating field
    pre_stim_s: float      # seconds of pre-stimulus baseline in each epoch (varies!
                            # confirmed per-dataset from each lab script's own
                            # pop_epoch(...) call -- do not assume it's the same
                            # across datasets, two of these use -1.1s not -1.0s.


REGISTRY = {
    "ds005285": DatasetSpec("ds005285", "29ByANT", 29, "mark_ica", pre_stim_s=1.0),
    "ds005284": DatasetSpec("ds005284", "26ByBiosemi", 26, "mark_ica", pre_stim_s=1.0),
    "ds005286": DatasetSpec("ds005286", "30ByANT", 30, "mark_ica", pre_stim_s=1.1),
    "ds005291": DatasetSpec("ds005291", "65ByANT", 65, "mark_ica", pre_stim_s=1.1),
    "ds005473": DatasetSpec("ds005473", "29ByBP", 29, "mark_ica", pre_stim_s=1.0),
}


def s3_set_url(spec: DatasetSpec, subject_id: str) -> str:
    fname = f"{subject_id}_{spec.task_label}.set"
    return f"https://s3.amazonaws.com/openneuro.org/{spec.dataset_id}/derivatives/{spec.derivative_stage}/{fname}"


def s3_fdt_url(spec: DatasetSpec, subject_id: str) -> str:
    fname = f"{subject_id}_{spec.task_label}.fdt"
    return f"https://s3.amazonaws.com/openneuro.org/{spec.dataset_id}/derivatives/{spec.derivative_stage}/{fname}"


def subject_ids(spec: DatasetSpec):
    return [f"sub-{i:03d}" for i in range(1, spec.n_subjects + 1)]
