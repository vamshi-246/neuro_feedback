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
from scipy.signal import butter, filtfilt

BANDS = [
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("theta", 4.0, 8.0),
]

WINDOWS_S = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]


def _bandpower(signal_1d: np.ndarray, srate: float, low: float, high: float) -> float:
    nyq = srate / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="bandpass") # type: ignore
    filtered = filtfilt(b, a, signal_1d)
    return float(np.mean(filtered ** 2))


def extract_trial_features(trial_channels: np.ndarray, srate: float, onset_sample: int) -> np.ndarray:
    """
    trial_channels: (n_selected_channels, n_pnts) for one trial, already
    restricted to the common 4-channel set.
    Returns: (3 windows, 3 bands) raw (unquantized) power values, each band
    averaged across the selected channels.
    """
    out = np.zeros((len(WINDOWS_S), len(BANDS)), dtype=float)
    for wi, (w_start, w_end) in enumerate(WINDOWS_S):
        s0 = onset_sample + int(round(w_start * srate))
        s1 = onset_sample + int(round(w_end * srate))
        for bi, (_, low, high) in enumerate(BANDS):
            powers = [
                _bandpower(trial_channels[ch, s0:s1], srate, low, high)
                for ch in range(trial_channels.shape[0])
            ]
            out[wi, bi] = float(np.mean(powers))
    return out


def quantize_to_uint8(raw_power: np.ndarray, band_lo: np.ndarray, band_hi: np.ndarray) -> np.ndarray:
    """
    Log-transform (power distributions are heavy-tailed) then linearly map
    each band's [band_lo, band_hi] range to 0-255, matching the RTL's 8-bit
    feature width. band_lo/band_hi must come from TRAIN-only statistics in
    the real pipeline -- this function just applies whatever bounds it's given.
    """
    log_power = np.log1p(raw_power)
    scaled = (log_power - band_lo) / (band_hi - band_lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 255).astype(np.uint8)


def bin_rating(rating: float) -> int:
    """0=Low, 1=Moderate, 2=High -- matches pain_classifier_fsm.v's 3 states.
    Bin edges follow the lab's own rating-scale anchors (0=no sensation,
    4=pain onset, 6=moderate, 8=severe, 10=intolerable): below the pain-onset
    anchor is Low, up to the severe anchor is Moderate, at/above it is High.
    """
    if rating < 4:
        return 0
    if rating < 7:
        return 1
    return 2
