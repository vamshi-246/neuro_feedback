"""
Turns a raw (channels, time, trials) epoch array into the RTL-shaped feature
sequence: per trial, 3 consecutive post-stimulus time windows x 3 EEG bands
(alpha, beta, theta -- GSR intentionally dropped, see project notes on why).

Band and window choices, and why:

- Bands are theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz -- exactly the three
  bands the existing pain_classification_engine.v RTL already expects
  (feature_vector_generator.v concatenates {alpha, beta, theta, gsr}; this
  drops gsr and keeps that same alpha/beta/theta order).
- The RTL's own `power.v` module works on a already-band-filtered signal, not
  a raw voltage trace -- its 16-sample sliding window is far too short to
  resolve an 8-13 Hz oscillation on its own. So the "band split" has to happen
  here in software; the RTL is only ever meant to smooth+threshold the output
  of that split. This module is filling that gap.
- 3 time windows (0.0-0.3s, 0.3-0.6s, 0.6-1.0s after the laser fires) give the
  LSTM a real short sequence to learn from, matching the "sequential EEG
  features" language in the Phase III/IV instructions, while staying inside
  the well-established laser-evoked-response window.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from quality_control import require_valid_rating

BANDS = [
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("theta", 4.0, 8.0),
]

WINDOWS_S = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
BASELINE_WINDOW_S = (-0.5, 0.0)
MIN_BASELINE_POWER = 1e-12


class TrialFeatureError(ValueError):
    """A stable, countable reason for rejecting one trial's EEG features."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _reject(reason: str, message: str):
    raise TrialFeatureError(reason, message)


def _validate_trial_input(
    trial_channels: np.ndarray, srate: float, onset_sample: int
) -> np.ndarray:
    values = np.asarray(trial_channels)
    if values.ndim != 2:
        _reject(
            "invalid_trial_shape",
            f"trial_channels must have shape (channels, samples), got {values.shape}",
        )
    if values.shape[0] == 0 or values.shape[1] == 0:
        _reject("empty_trial", f"trial_channels cannot be empty, got {values.shape}")
    if not np.issubdtype(values.dtype, np.number):
        _reject("nonnumeric_eeg", f"EEG dtype must be numeric, got {values.dtype}")
    if not np.isfinite(values).all():
        _reject("nonfinite_eeg", "trial contains NaN or infinite EEG samples")
    if not np.isfinite(srate) or srate <= 0:
        _reject("invalid_srate", f"sampling rate must be positive and finite, got {srate}")
    if not isinstance(onset_sample, (int, np.integer)):
        _reject("invalid_onset", f"onset_sample must be an integer, got {onset_sample!r}")
    if not 0 <= int(onset_sample) < values.shape[1]:
        _reject(
            "invalid_onset",
            f"onset sample {onset_sample} is outside 0..{values.shape[1] - 1}",
        )
    return values


def _window_indices(
    n_samples: int, srate: float, onset_sample: int, w_start: float, w_end: float
) -> tuple[int, int]:
    s0 = onset_sample + int(round(w_start * srate))
    s1 = onset_sample + int(round(w_end * srate))
    if not 0 <= s0 < s1 <= n_samples:
        _reject(
            "window_out_of_bounds",
            f"requested {w_start:g}..{w_end:g} s window maps to samples {s0}:{s1}, "
            f"outside trial length {n_samples}",
        )
    return s0, s1


def _filter_band(
    trial_channels: np.ndarray, srate: float, low: float, high: float
) -> np.ndarray:
    nyq = srate / 2.0
    if not 0.0 < low < high < nyq:
        _reject(
            "invalid_band",
            f"band {low:g}-{high:g} Hz must satisfy 0 < low < high < Nyquist {nyq:g} Hz",
        )
    # SOS form is numerically safer than a single high-order numerator/
    # denominator polynomial.  Filter the complete epoch once, not each short
    # analysis slice independently, so slice-edge transients do not dominate
    # the 0.3-second windows.
    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    try:
        filtered = sosfiltfilt(sos, trial_channels, axis=1)
    except ValueError as exc:
        _reject(
            "window_too_short",
            f"full epoch is too short for zero-phase SOS filtering: {exc}",
        )
    if not np.isfinite(filtered).all():
        _reject("nonfinite_filtered_eeg", "band-pass filter produced a nonfinite value")
    return filtered


def _channel_mean_square_power(
    filtered_channels: np.ndarray, s0: int, s1: int
) -> np.ndarray:
    power = np.mean(
        np.square(filtered_channels[:, s0:s1], dtype=np.float64), axis=1
    )
    if not np.isfinite(power).all() or np.any(power < 0.0):
        _reject("invalid_bandpower", f"computed invalid channel powers {power.tolist()}")
    return power


def extract_trial_channel_features(
    trial_channels: np.ndarray, srate: float, onset_sample: int
) -> np.ndarray:
    """Return relative power with shape (time windows, channels, bands)."""

    _, channel_relative_power, _ = extract_trial_feature_views(
        trial_channels, srate, onset_sample
    )
    return channel_relative_power


def extract_trial_feature_views(
    trial_channels: np.ndarray, srate: float, onset_sample: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return channel-average, per-channel, and per-channel baseline power.

    The first result preserves the current model's calculation: channel powers
    are averaged first, then post-stimulus power is divided by baseline power.
    The second result preserves channel identity for a future wider-channel
    model.  The baseline tensor makes the exact relationship between those two
    views checkable when the archive is loaded.
    """

    trial_channels = _validate_trial_input(trial_channels, srate, onset_sample)

    channel_relative_power = np.zeros(
        (len(WINDOWS_S), trial_channels.shape[0], len(BANDS)), dtype=float
    )
    channel_average_power = np.zeros((len(WINDOWS_S), len(BANDS)), dtype=float)
    channel_baseline_power = np.zeros(
        (trial_channels.shape[0], len(BANDS)), dtype=float
    )
    baseline_s0, baseline_s1 = _window_indices(
        trial_channels.shape[1],
        srate,
        onset_sample,
        BASELINE_WINDOW_S[0],
        BASELINE_WINDOW_S[1],
    )
    post_indices = [
        _window_indices(trial_channels.shape[1], srate, onset_sample, w_start, w_end)
        for w_start, w_end in WINDOWS_S
    ]

    for bi, (_, low, high) in enumerate(BANDS):
        filtered = _filter_band(trial_channels, srate, low, high)
        baseline = _channel_mean_square_power(filtered, baseline_s0, baseline_s1)
        if np.any(baseline <= MIN_BASELINE_POWER):
            bad_channels = np.flatnonzero(baseline <= MIN_BASELINE_POWER).tolist()
            _reject(
                "baseline_power_too_small",
                f"baseline channel powers for {low:g}-{high:g} Hz must exceed "
                f"{MIN_BASELINE_POWER:g}; bad channel indices={bad_channels}",
            )
        channel_baseline_power[:, bi] = baseline
        for wi, (s0, s1) in enumerate(post_indices):
            post = _channel_mean_square_power(filtered, s0, s1)
            channel_relative_power[wi, :, bi] = post / baseline
            channel_average_power[wi, bi] = np.mean(post) / np.mean(baseline)
    if not (
        np.isfinite(channel_relative_power).all()
        and np.isfinite(channel_average_power).all()
        and np.isfinite(channel_baseline_power).all()
    ):
        _reject("nonfinite_features", "relative-power calculation produced a nonfinite value")
    return channel_average_power, channel_relative_power, channel_baseline_power


def extract_trial_features(
    trial_channels: np.ndarray, srate: float, onset_sample: int
) -> np.ndarray:
    """Return the current model input: channel-mean relative power, shape (3, 3).

    The separate channel tensor is available through
    ``extract_trial_channel_features`` so a later model can use more channels
    without changing or losing the feature-extraction stage.
    """

    channel_average_power, _, _ = extract_trial_feature_views(
        trial_channels, srate, onset_sample
    )
    return channel_average_power


def quantize_to_uint8(raw_power: np.ndarray, band_lo: np.ndarray, band_hi: np.ndarray) -> np.ndarray:
    """
    Log-transform (power distributions are heavy-tailed) then linearly map
    each band's [band_lo, band_hi] range to 0-255, matching the RTL's 8-bit
    feature width. band_lo/band_hi must come from TRAIN-only statistics in
    the real pipeline -- this function just applies whatever bounds it's given.
    """
    raw_power = np.asarray(raw_power, dtype=np.float64)
    band_lo = np.asarray(band_lo, dtype=np.float64)
    band_hi = np.asarray(band_hi, dtype=np.float64)
    if raw_power.ndim != 3:
        raise ValueError(f"raw_power must have shape (trials, steps, bands), got {raw_power.shape}")
    if band_lo.shape != raw_power.shape[1:] or band_hi.shape != raw_power.shape[1:]:
        raise ValueError(
            f"quantization bounds must each have shape {raw_power.shape[1:]}, "
            f"got lo={band_lo.shape}, hi={band_hi.shape}"
        )
    if not np.isfinite(raw_power).all() or np.any(raw_power < 0.0):
        raise ValueError("raw_power must contain only finite, nonnegative values")
    if not np.isfinite(band_lo).all() or not np.isfinite(band_hi).all():
        raise ValueError("quantization bounds must be finite")
    if np.any(band_hi <= band_lo):
        bad = np.argwhere(band_hi <= band_lo).tolist()
        raise ValueError(f"every high bound must exceed its low bound; invalid cells={bad}")

    log_power = np.log1p(raw_power)
    scaled = (log_power - band_lo) / (band_hi - band_lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 255).astype(np.uint8)


def bin_rating(rating: float) -> int:
    """0=Low, 1=Moderate, 2=High -- matches pain_classifier_fsm.v's 3 states.
    Bin edges follow the lab's own rating-scale anchors (0=no sensation,
    4=pain onset, 6=moderate, 8=severe, 10=intolerable).  The executable
    project rule is Low below 4, Moderate from 4 to below 7, and High from 7.
    """
    rating = require_valid_rating(rating)
    if rating < 4:
        return 0
    if rating < 7:
        return 1
    return 2
