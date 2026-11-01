"""Shared validation rules for the pooled laser-pain EEG pipeline.

The rating is the supervised target, so an invalid rating must never be
silently converted into a pain class.  Keep the status strings stable: they
are also used in subject-level QC reports produced by ``build_dataset.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


RATING_MIN = 0.0
RATING_MAX = 10.0

RATING_OK = "ok"
MISSING_RATING = "missing_rating"
NONFINITE_RATING = "nonfinite_rating"
INVALID_RATING = "invalid_rating"
OUT_OF_RANGE_RATING = "out_of_range_rating"
MISSING_EVENT = "missing_event"
MISSING_RATING_EVENT = "missing_rating_event"
AMBIGUOUS_RATING_EVENT = "ambiguous_rating_event"


def rating_value_and_status(value: Any) -> tuple[float, str]:
    """Return a scalar rating and a stable admission status.

    EEGLAB fields loaded through SciPy may be Python scalars, NumPy scalars,
    empty arrays, or strings.  A literal/empty NaN represents a missing
    behavioral response; infinity is kept separate as a malformed numeric
    value.  Out-of-range numeric values are returned unchanged for auditing,
    but are not admitted for training.
    """

    if value is None:
        return np.nan, MISSING_RATING

    values = np.asarray(value)
    if values.size == 0:
        return np.nan, MISSING_RATING
    if values.size != 1:
        return np.nan, INVALID_RATING

    try:
        rating = float(values.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        return np.nan, INVALID_RATING

    if np.isnan(rating):
        return np.nan, MISSING_RATING
    if not np.isfinite(rating):
        return rating, NONFINITE_RATING
    if not RATING_MIN <= rating <= RATING_MAX:
        return rating, OUT_OF_RANGE_RATING
    return rating, RATING_OK


def require_valid_rating(value: Any) -> float:
    """Return a valid 0--10 rating or raise instead of silently binning it."""

    rating, status = rating_value_and_status(value)
    if status != RATING_OK:
        raise ValueError(
            f"rating must be one finite scalar in [{RATING_MIN:g}, {RATING_MAX:g}]; "
            f"status={status}, value={value!r}"
        )
    return rating
