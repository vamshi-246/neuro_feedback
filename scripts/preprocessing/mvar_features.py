"""Multivariate autoregressive (MVAR) features: signal dynamics and directed flow.

Every feature the pipeline currently uses is a snapshot -- how much energy sat in a
band during a window. None of them describe how the signal moves, and the one
relationship feature we have, band-limited correlation, is symmetric: it can say two
electrodes rise and fall together but never which one leads.

An MVAR model describes the motion instead. It fits rules of the form

    x(t) = A1 x(t-1) + A2 x(t-2) + ... + Ap x(t-p) + noise

where x(t) holds every channel at one instant, so each coefficient in Ak says how
strongly one channel's past drives another channel's present. That asymmetry is the
point: the coefficient from Fz to Cz and the one from Cz to Fz are separate numbers,
so direction of influence becomes measurable.

Three design decisions worth stating
------------------------------------
Sampling rate. The recordings are 1000 Hz, where a model order of 10 looks back only
10 ms -- far too brief to describe a 1-45 Hz rhythm. The signal is therefore
decimated to 250 Hz first, where order 10 reaches back 40 ms and spans the periods
that matter. Decimation applies an anti-aliasing filter, so this discards nothing
the bands care about.

Window length. An MVAR fit estimates order * channels^2 coefficients, so it needs far
more samples than a 0.3 s slice provides. With four channels and order 10 that is 160
numbers; a 1.0 s post-stimulus window at 250 Hz supplies 1000 equations, about six per
coefficient. Fitting each 0.3 s window separately would leave roughly one and a half,
which estimates noise.

Stationarity. MVAR assumes the signal's statistics hold steady across the fitted
window, and an evoked response is by definition a transient that violates this. The
honest reading is that these coefficients summarise the average dynamics over the
response, not the dynamics at any instant within it. That limitation is real and
belongs in any write-up alongside the numbers.

Features produced per trial
---------------------------
  AR coefficients             order x channels x channels, the raw dynamics
  Partial directed coherence  directed channel-to-channel influence per band
  Residual variance           per channel, how much the model failed to explain
  Parametric band power       spectra derived from the model rather than by FFT,
                              the estimator better suited to short segments
"""

import numpy as np
from scipy.signal import decimate

from rich_feature_extraction import RICH_BANDS

MVAR_TARGET_HZ = 250.0
MVAR_DEFAULT_ORDER = 10
MVAR_WINDOW_S = (0.0, 1.0)
# Ridge term keeps the normal equations invertible when channels are near-collinear,
# which average referencing makes likely.
MVAR_RIDGE = 1e-6
PDC_FREQ_STEP_HZ = 1.0
EPS = 1e-12


class MVARError(ValueError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def fit_mvar(window, order, ridge=MVAR_RIDGE):
    """Least-squares MVAR fit. window is (channels, samples); returns (A, Sigma).

    A has shape (order, channels, channels), where A[k, i, j] is how much channel j
    k+1 steps ago contributes to channel i now.
    """

    window = np.asarray(window, dtype=np.float64)
    n_channels, n_samples = window.shape
    if n_samples <= order * n_channels + order + 1:
        raise MVARError(
            "window_too_short",
            f"{n_samples} samples cannot identify order {order} on {n_channels} channels",
        )

    # Present values to predict, and the stacked lagged history that predicts them.
    target = window[:, order:]
    history = np.concatenate(
        [window[:, order - k: n_samples - k] for k in range(1, order + 1)], axis=0
    )
    gram = history @ history.T
    gram += ridge * np.trace(gram) / gram.shape[0] * np.eye(gram.shape[0])
    coefficients = np.linalg.solve(gram, history @ target.T).T  # (C, C*order)

    residual = target - coefficients @ history
    sigma = (residual @ residual.T) / residual.shape[1]
    A = np.stack(
        [coefficients[:, k * n_channels:(k + 1) * n_channels] for k in range(order)]
    )
    return A, sigma


def _transfer_matrices(A, srate, freqs):
    """Return A(f) = I - sum_k A_k exp(-2i pi f k / srate) at every frequency."""

    order, n_channels, _ = A.shape
    identity = np.eye(n_channels)[None, :, :]
    lags = np.arange(1, order + 1)[None, :]
    phase = np.exp(-2j * np.pi * freqs[:, None] * lags / srate)  # (F, order)
    return identity - np.einsum("fk,kij->fij", phase, A)


def partial_directed_coherence(A, srate, freqs):
    """PDC[f, i, j]: how much channel j drives channel i, normalized per source."""

    Af = _transfer_matrices(A, srate, freqs)
    magnitude = np.abs(Af)
    # Normalising down each column makes the measure per-source, which is what
    # separates outgoing influence from incoming.
    column_norm = np.sqrt(np.sum(magnitude ** 2, axis=1, keepdims=True))
    return magnitude / np.maximum(column_norm, EPS)


def parametric_band_power(A, sigma, srate, freqs):
    """Band power from the fitted model: S(f) = H(f) Sigma H(f)^H, with H = A(f)^-1."""

    Af = _transfer_matrices(A, srate, freqs)
    H = np.linalg.inv(Af)
    spectra = np.einsum("fij,jk,flk->fil", H, sigma, np.conj(H))
    return np.real(np.diagonal(spectra, axis1=1, axis2=2))  # (F, channels)


def mvar_feature_names(channels, order=MVAR_DEFAULT_ORDER):
    """Feature names in exactly the order extract_mvar_features emits them."""

    names = []
    for k in range(1, order + 1):
        for target in channels:
            for source in channels:
                names.append(f"mvar:a{k}:{source}->{target}")
    for target in channels:
        for source in channels:
            if source == target:
                continue
            for band, _, _ in RICH_BANDS:
                names.append(f"mvar:pdc:{band}:{source}->{target}")
    for channel in channels:
        names.append(f"mvar:residual_variance:{channel}")
    for channel in channels:
        for band, _, _ in RICH_BANDS:
            names.append(f"mvar:parametric_power:{band}:{channel}")
    return names


def extract_mvar_features(data, srate, onset_sample, order=MVAR_DEFAULT_ORDER):
    """Compute MVAR features for every trial. data is (channels, samples, trials)."""

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 3:
        raise MVARError("invalid_shape", f"expected 3 dimensions, got {values.shape}")
    n_channels, n_samples, n_trials = values.shape
    if not np.isfinite(values).all():
        raise MVARError("nonfinite_eeg", "input contains NaN or infinite samples")

    start = onset_sample + int(round(MVAR_WINDOW_S[0] * srate))
    stop = onset_sample + int(round(MVAR_WINDOW_S[1] * srate))
    if not 0 <= start < stop <= n_samples:
        raise MVARError(
            "window_out_of_bounds",
            f"MVAR window maps to {start}:{stop}, outside epoch length {n_samples}",
        )

    segment = values[:, start:stop, :]
    factor = int(round(srate / MVAR_TARGET_HZ))
    if factor > 1:
        # decimate low-pass filters before downsampling, so no aliasing enters.
        segment = decimate(segment, factor, axis=1, ftype="fir", zero_phase=True)
        effective_rate = srate / factor
    else:
        effective_rate = srate

    freqs = np.arange(
        RICH_BANDS[0][1], RICH_BANDS[-1][2] + PDC_FREQ_STEP_HZ / 2, PDC_FREQ_STEP_HZ
    )
    band_masks = []
    for index, (band, low, high) in enumerate(RICH_BANDS):
        last = index == len(RICH_BANDS) - 1
        band_masks.append(
            (freqs >= low) & (freqs <= high) if last else (freqs >= low) & (freqs < high)
        )
    off_diagonal = [
        (i, j) for i in range(n_channels) for j in range(n_channels) if i != j
    ]

    rows = []
    for t in range(n_trials):
        window = segment[:, :, t]
        window = window - window.mean(axis=1, keepdims=True)
        A, sigma = fit_mvar(window, order)

        pdc = partial_directed_coherence(A, effective_rate, freqs)
        power = parametric_band_power(A, sigma, effective_rate, freqs)

        parts = [A.reshape(-1)]
        parts.append(
            np.asarray([
                pdc[mask, i, j].mean()
                for i, j in off_diagonal
                for mask in band_masks
            ])
        )
        parts.append(np.diag(sigma))
        parts.append(
            np.asarray([
                np.log(np.maximum(power[mask, c].mean(), EPS))
                for c in range(n_channels)
                for mask in band_masks
            ])
        )
        rows.append(np.concatenate(parts))

    features = np.stack(rows)
    if not np.isfinite(features).all():
        raise MVARError("nonfinite_features", "MVAR calculation produced a nonfinite value")
    return features
