"""Richer per-trial EEG features aimed at the markers the current pipeline discards.

Why this module exists
----------------------
Seven separate experiments (ordinal labels, per-channel inputs, gradient boosting,
a memorization test, per-subject normalization, a within-subject probe, and a
personalized-calibration probe) all landed between 42% and 47% balanced accuracy.
The memorization test settled the cause: the saved features separate TRAINING
trials perfectly while transferring almost nothing to new people.  The learner was
never the bottleneck -- the feature set is.

feature_extraction.py computes relative power for alpha/beta/theta only, because
`RTL/feature_vector_generator.v` concatenates exactly `{alpha, beta, theta}`.  That
choice came from the hardware, not from what predicts laser pain, and it omits the
three best-established single-trial markers:

1. The N2-P2 vertex complex.  For laser-evoked pain this is the largest and most
   reliable cortical response, a time-domain deflection roughly 150-450 ms after
   the laser, biggest at Cz, whose peak-to-peak amplitude tracks rated intensity
   trial by trial.  Squaring a signal to get power destroys the polarity that
   distinguishes the negative N2 from the positive P2, and averaging across a
   300 ms window smears the peaks, so the existing pipeline cannot represent it
   at all.
2. Gamma (30-45 Hz).  Gamma-band responses follow *perceived* intensity closely.
   The current bands stop at 30 Hz.  45 Hz is the ceiling here so that 50/60 Hz
   line noise is never admitted.
3. Delta (1-4 Hz).  Much of the evoked deflection's energy is below 4 Hz.

Hjorth parameters, spectral entropy, and band-limited channel coupling are also
computed: they are cheap once the band filtering is done, and they describe signal
shape rather than raw magnitude, which is the property that failed to transfer.

Design notes
------------
Filtering runs once per band over every trial of a subject at once, rather than
per trial, because a full rebuild covers 858 recordings and per-trial filtering
would dominate the runtime.

Absolute power is kept alongside the baseline-relative ratio.  A ratio discards
overall magnitude, and the within-subject probe showed that a subject's overall
level genuinely carries label information -- per-subject centering, which removes
exactly that, made validation accuracy worse.
"""

import re

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from feature_extraction import BASELINE_WINDOW_S, MIN_BASELINE_POWER, WINDOWS_S

# Five bands.  The upper edge stays at 45 Hz so mains interference at 50/60 Hz
# can never enter the gamma estimate.
RICH_BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]

# Broadband range used for Hjorth parameters and spectral entropy.
BROADBAND_HZ = (1.0, 45.0)

# Evoked-potential settings.  A 1-30 Hz passband is the conventional choice for
# laser-evoked potentials: it removes drift and muscle noise while preserving the
# N2-P2 deflection.
ERP_BAND_HZ = (1.0, 30.0)
N2_SEARCH_S = (0.15, 0.30)
P2_SEARCH_END_S = 0.55
N2_MEAN_WINDOW_S = (0.15, 0.25)
P2_MEAN_WINDOW_S = (0.25, 0.45)
ERP_RMS_WINDOW_S = (0.0, 0.60)

EPS = 1e-12


class RichFeatureError(ValueError):
    """A stable, countable reason for rejecting one subject's rich features."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _sos(srate: float, low: float, high: float):
    nyq = srate / 2.0
    if not 0.0 < low < high < nyq:
        raise RichFeatureError(
            "invalid_band",
            f"band {low:g}-{high:g} Hz needs 0 < low < high < Nyquist {nyq:g} Hz",
        )
    return butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")


def _window_slice(n_samples: int, srate: float, onset: int, start_s: float, end_s: float):
    s0 = onset + int(round(start_s * srate))
    s1 = onset + int(round(end_s * srate))
    if not 0 <= s0 < s1 <= n_samples:
        raise RichFeatureError(
            "window_out_of_bounds",
            f"window {start_s:g}..{end_s:g}s maps to samples {s0}:{s1}, "
            f"outside epoch length {n_samples}",
        )
    return s0, s1


def _hjorth(signal: np.ndarray, axis: int = 1):
    """Return (mobility, complexity), which describe shape rather than amplitude."""

    d1 = np.diff(signal, axis=axis)
    d2 = np.diff(d1, axis=axis)
    var0 = np.var(signal, axis=axis)
    var1 = np.var(d1, axis=axis)
    var2 = np.var(d2, axis=axis)
    mobility = np.sqrt(np.maximum(var1, 0.0) / np.maximum(var0, EPS))
    mobility_d1 = np.sqrt(np.maximum(var2, 0.0) / np.maximum(var1, EPS))
    complexity = mobility_d1 / np.maximum(mobility, EPS)
    return mobility, complexity


def _spectral_entropy(segment: np.ndarray, srate: float):
    """Shannon entropy of the normalized 1-45 Hz spectrum, scaled to 0..1.

    A flat spectrum scores near 1 and a strongly peaked one near 0, so this
    measures how ordered the rhythm is independently of how large it is.
    """

    nperseg = min(segment.shape[1], int(round(srate)))
    freqs, psd = welch(segment, fs=srate, nperseg=nperseg, axis=1)
    keep = (freqs >= BROADBAND_HZ[0]) & (freqs <= BROADBAND_HZ[1])
    if not np.any(keep):
        raise RichFeatureError("empty_spectrum", "no spectral bins inside 1-45 Hz")
    psd = psd[:, keep, ...]
    total = np.sum(psd, axis=1, keepdims=True)
    probability = psd / np.maximum(total, EPS)
    entropy = -np.sum(probability * np.log(np.maximum(probability, EPS)), axis=1)
    return entropy / np.log(probability.shape[1])


def rich_feature_names(channels, include_coupling: bool = True) -> list:
    """Feature column names, in exactly the order extract_rich_features produces.

    Coupling counts one value per channel pair per band, so it grows with the
    square of the channel count: 4 channels give 30 columns but 30 channels give
    2,175. It was also the weakest group when measured on its own, so wide-channel
    builds can switch it off and keep the per-channel features that carry the signal.
    """

    names = []
    for channel in channels:
        for stat in (
            "n2_amplitude", "n2_latency", "p2_amplitude", "p2_latency",
            "n2p2_amplitude", "n2_window_mean", "p2_window_mean", "erp_rms",
        ):
            names.append(f"{channel}:erp_{stat}")
        for band, _, _ in RICH_BANDS:
            for window_index in range(len(WINDOWS_S)):
                names.append(f"{channel}:{band}:w{window_index}:relative")
                names.append(f"{channel}:{band}:w{window_index}:log_absolute")
        names.append(f"{channel}:hjorth_mobility")
        names.append(f"{channel}:hjorth_complexity")
        names.append(f"{channel}:spectral_entropy")
    if include_coupling:
        for i, first in enumerate(channels):
            for second in channels[i + 1:]:
                for band, _, _ in RICH_BANDS:
                    names.append(f"{first}-{second}:{band}:coupling")
    return names


def extract_rich_features(
    data: np.ndarray, srate: float, onset_sample: int, include_coupling: bool = True
):
    """Compute rich features for every trial of one subject at once.

    ``data`` has shape (channels, samples, trials) in microvolts, matching
    ``Recording.data`` after channel selection.  Returns (trials, features).
    """

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 3:
        raise RichFeatureError(
            "invalid_shape", f"expected (channels, samples, trials), got {values.shape}"
        )
    n_channels, n_samples, n_trials = values.shape
    if min(n_channels, n_samples, n_trials) == 0:
        raise RichFeatureError("empty_input", f"empty input array {values.shape}")
    if not np.isfinite(values).all():
        raise RichFeatureError("nonfinite_eeg", "input contains NaN or infinite samples")
    if not np.isfinite(srate) or srate <= 0:
        raise RichFeatureError("invalid_srate", f"sampling rate must be positive, got {srate}")
    if not 0 <= int(onset_sample) < n_samples:
        raise RichFeatureError(
            "invalid_onset", f"onset {onset_sample} outside 0..{n_samples - 1}"
        )
    onset = int(onset_sample)

    base_s0, base_s1 = _window_slice(
        n_samples, srate, onset, BASELINE_WINDOW_S[0], BASELINE_WINDOW_S[1]
    )
    post_slices = [
        _window_slice(n_samples, srate, onset, start, end) for start, end in WINDOWS_S
    ]

    # ---- Evoked potential: keep polarity, so N2 and P2 stay distinguishable ----
    erp = sosfiltfilt(_sos(srate, *ERP_BAND_HZ), values, axis=1)
    erp = erp - np.mean(erp[:, base_s0:base_s1, :], axis=1, keepdims=True)

    n2_s0, n2_s1 = _window_slice(n_samples, srate, onset, *N2_SEARCH_S)
    p2_end = onset + int(round(P2_SEARCH_END_S * srate))
    if p2_end > n_samples:
        raise RichFeatureError(
            "window_out_of_bounds",
            f"P2 search needs {P2_SEARCH_END_S:g}s after onset, epoch has "
            f"{(n_samples - onset) / srate:.3f}s",
        )
    n2m_s0, n2m_s1 = _window_slice(n_samples, srate, onset, *N2_MEAN_WINDOW_S)
    p2m_s0, p2m_s1 = _window_slice(n_samples, srate, onset, *P2_MEAN_WINDOW_S)
    rms_s0, rms_s1 = _window_slice(n_samples, srate, onset, *ERP_RMS_WINDOW_S)

    n2_segment = erp[:, n2_s0:n2_s1, :]
    n2_offset = np.argmin(n2_segment, axis=1)
    n2_amplitude = np.min(n2_segment, axis=1)
    n2_index = n2_offset + n2_s0
    n2_latency = (n2_index - onset) / srate

    # P2 is searched strictly after this trial's own N2, which enforces the
    # physiological ordering instead of letting a fixed window invert it.
    p2_amplitude = np.empty((n_channels, n_trials), dtype=np.float64)
    p2_latency = np.empty((n_channels, n_trials), dtype=np.float64)
    for c in range(n_channels):
        for t in range(n_trials):
            start = int(n2_index[c, t]) + 1
            if start >= p2_end:
                start = p2_end - 1
            window = erp[c, start:p2_end, t]
            local = int(np.argmax(window))
            p2_amplitude[c, t] = window[local]
            p2_latency[c, t] = (start + local - onset) / srate

    erp_features = np.stack(
        [
            n2_amplitude,
            n2_latency,
            p2_amplitude,
            p2_latency,
            p2_amplitude - n2_amplitude,          # the classic N2-P2 measure
            np.mean(erp[:, n2m_s0:n2m_s1, :], axis=1),
            np.mean(erp[:, p2m_s0:p2m_s1, :], axis=1),
            np.sqrt(np.mean(np.square(erp[:, rms_s0:rms_s1, :]), axis=1)),
        ],
        axis=-1,
    )  # (channels, trials, 8)

    # ---- Band power: relative to baseline and absolute, for five bands ----
    band_features = []
    coupling_features = []
    for _, low, high in RICH_BANDS:
        filtered = sosfiltfilt(_sos(srate, low, high), values, axis=1)
        baseline = np.mean(np.square(filtered[:, base_s0:base_s1, :]), axis=1)
        baseline = np.maximum(baseline, MIN_BASELINE_POWER)
        per_window = []
        for s0, s1 in post_slices:
            post = np.mean(np.square(filtered[:, s0:s1, :]), axis=1)
            per_window.append(post / baseline)
            per_window.append(np.log(np.maximum(post, EPS)))
        band_features.append(np.stack(per_window, axis=-1))  # (channels, trials, 6)

        # Band-limited coupling: how similarly two electrodes move during the
        # response window.  This is a relationship between channels, so it does
        # not inherit either electrode's individual amplitude offset.
        if include_coupling:
            response = filtered[:, post_slices[0][0]:post_slices[-1][1], :]
            centred = response - np.mean(response, axis=1, keepdims=True)
            norm = np.sqrt(np.sum(np.square(centred), axis=1))
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    numerator = np.sum(centred[i] * centred[j], axis=0)
                    coupling_features.append(
                        numerator / np.maximum(norm[i] * norm[j], EPS)
                    )

    band_features = np.concatenate(band_features, axis=-1)  # (channels, trials, 30)

    # ---- Shape descriptors on the broadband response ----
    broadband = sosfiltfilt(_sos(srate, *BROADBAND_HZ), values, axis=1)
    response = broadband[:, post_slices[0][0]:post_slices[-1][1], :]
    mobility, complexity = _hjorth(response, axis=1)
    entropy = np.stack(
        [_spectral_entropy(response[:, :, t], srate) for t in range(n_trials)], axis=-1
    )
    shape_features = np.stack([mobility, complexity, entropy], axis=-1)

    per_channel = np.concatenate(
        [erp_features, band_features, shape_features], axis=-1
    )  # (channels, trials, 41)
    # Channel-major flattening matches the order rich_feature_names() emits.
    flat = per_channel.transpose(1, 0, 2).reshape(n_trials, -1)

    if include_coupling:
        # Coupling was appended band-inner/pair-outer; regroup to pair-outer/band-inner
        # so the columns line up with rich_feature_names().
        n_pairs = n_channels * (n_channels - 1) // 2
        coupling = np.stack(coupling_features, axis=0).reshape(
            len(RICH_BANDS), n_pairs, n_trials
        )
        coupling = coupling.transpose(2, 1, 0).reshape(n_trials, -1)
        features = np.concatenate([flat, coupling], axis=1)
    else:
        features = flat
    if not np.isfinite(features).all():
        raise RichFeatureError(
            "nonfinite_features", "rich feature calculation produced a nonfinite value"
        )
    return features


_WINDOW_TAG = re.compile(r":w(\d+):")


def rich_lstm_view(features: np.ndarray, names, n_windows: int = len(WINDOWS_S)):
    """Reshape flat rich features into the (trials, time steps, features) LSTM input.

    Band power is measured once per time window, so those columns form the part of
    the sequence that actually changes from step to step.  Evoked-potential, shape
    and coupling values describe the trial as a whole and have no per-window
    version, so they are repeated at every step.  Repetition costs a little width
    but keeps one honest sequence: the recurrent layer sees the band response
    evolving against a constant description of the trial.
    """

    features = np.asarray(features, dtype=np.float64)
    names = [str(name) for name in names]
    if features.ndim != 2 or features.shape[1] != len(names):
        raise ValueError(
            f"features {features.shape} do not match {len(names)} feature names"
        )

    per_window = [[] for _ in range(n_windows)]
    per_trial = []
    for index, name in enumerate(names):
        match = _WINDOW_TAG.search(name)
        if match is None:
            per_trial.append(index)
            continue
        window = int(match.group(1))
        if not 0 <= window < n_windows:
            raise ValueError(f"feature {name!r} names a window outside 0..{n_windows-1}")
        per_window[window].append(index)

    # Each window must expose the same measurements in the same order, otherwise
    # a feature would silently change meaning between time steps.
    canonical = [_WINDOW_TAG.sub(":", names[i]) for i in per_window[0]]
    for window in range(1, n_windows):
        if [_WINDOW_TAG.sub(":", names[i]) for i in per_window[window]] != canonical:
            raise ValueError(f"window {window} features do not match window 0")
    if not canonical:
        raise ValueError("no per-window features found; cannot build a sequence")

    step_names = canonical + [names[i] for i in per_trial]
    sequence = np.stack(
        [
            np.concatenate(
                [features[:, per_window[window]], features[:, per_trial]], axis=1
            )
            for window in range(n_windows)
        ],
        axis=1,
    )
    return sequence, step_names


def fit_uint8_bounds(train_values: np.ndarray, low_percentile=1.0, high_percentile=99.0):
    """Fit per-feature 8-bit scaling bounds on TRAINING rows only.

    feature_extraction.quantize_to_uint8 assumes nonnegative power and log-transforms
    internally.  Rich features break both assumptions: an N2 amplitude is negative by
    definition, and several columns are already logarithmic.  These bounds therefore
    scale each column on its own linear range, which is what the FPGA's 8-bit input
    path needs anyway.
    """

    train_values = np.asarray(train_values, dtype=np.float64)
    if train_values.ndim != 2 or not train_values.shape[0]:
        raise ValueError("quantization bounds need a nonempty (rows, features) matrix")
    low = np.percentile(train_values, low_percentile, axis=0)
    high = np.percentile(train_values, high_percentile, axis=0)
    # A constant column would divide by zero; give it a unit span so it maps to 0.
    high = np.where(high <= low, low + 1.0, high)
    return low, high


def apply_uint8_bounds(values: np.ndarray, low: np.ndarray, high: np.ndarray):
    """Clip to the fitted range and quantize to 8 bits, matching the RTL input width."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != low.shape[-1] or low.shape != high.shape:
        raise ValueError("quantization bounds must match the feature width")
    scaled = (values - low) / (high - low)
    return np.round(np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)
