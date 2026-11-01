"""Phase-amplitude coupling, peak alpha frequency, and alpha asymmetry.

Every feature the pipeline computes so far asks how MUCH energy sat in a band. None
ask whether the bands are talking to each other. Phase-amplitude coupling does: it
measures whether the phase of a slow rhythm controls the strength of a fast one, so a
trial where gamma bursts arrive at a particular point in every beta cycle scores high
even if the total gamma power is unremarkable.

That distinction matters here because the project's NFB guidebook lists coupling as a
pain biomarker in its own right -- "beta phase x gamma amplitude correlates with pain
severity" -- and cites a 2024 frontal-EEG study finding beta-gamma asymmetric coupling
significantly correlated with clinical pain intensity. Nothing in the existing 194
features can represent it, because squaring a signal for band power discards the phase
that coupling is defined by.

Two smaller biomarkers from the same source are included:

  peak alpha frequency  where inside 8-13 Hz the alpha peak actually sits, which the
                        guidebook reports slows in chronic pain. Band power cannot see
                        this: a peak at 9 Hz and one at 12 Hz give identical alpha power.
  alpha asymmetry       the left-right alpha difference across C3 and C4, linked to the
                        affective side of pain.

How coupling is measured
------------------------
Mean vector length: take the fast band's amplitude envelope, weight each sample by
where it fell in the slow band's cycle, and average as vectors. If the fast rhythm is
indifferent to the slow one, those vectors cancel and the score is near zero; if it
consistently peaks at one phase, they add up. The result is divided by mean amplitude,
so a trial cannot score highly merely by being loud -- which keeps this genuinely
independent of the band-power features rather than a restatement of them.

Coupling is computed over the post-stimulus response window, and over the pre-stimulus
baseline as well, since a change in coupling is more likely to reflect the response
than its absolute level is.
"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from feature_extraction import BASELINE_WINDOW_S

# Slow bands supply the phase, fast bands the amplitude; a pair is only meaningful
# when the phase band sits entirely below the amplitude band.
PHASE_BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0),
               ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0)]
AMPLITUDE_BANDS = [("beta", 13.0, 30.0), ("gamma", 30.0, 45.0)]
ALPHA_BAND = (8.0, 13.0)
RESPONSE_WINDOW_S = (0.0, 1.0)
EPS = 1e-12


class PACError(ValueError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def _pairs():
    """Phase/amplitude pairs this recording bandwidth can actually resolve.

    When a slow rhythm modulates a fast one, the energy carrying that modulation sits
    at carrier +/- modulator. If those sidebands fall outside the amplitude band, the
    band-pass filter removes the modulation before it can be measured, and the feature
    returns noise rather than a small true value.

    Verified on synthetic signals with a known coupling: delta (3 Hz) and theta (6 Hz)
    modulation of a 38 Hz carrier were detected at 50x and 53x above an uncoupled
    control, alpha (10 Hz) fell to 4.9x, and beta (20 Hz) to 1.2x -- indistinguishable
    from no coupling.

    The amplitude band must therefore span at least twice the phase band's upper edge.
    Gamma here is 30-45 Hz, only 15 Hz wide, because these recordings carry no
    measurable content above about 45 Hz (checked directly: power at 52-70, 70-90 and
    90-100 Hz is zero relative to alpha). That rules out beta-phase/gamma-amplitude
    coupling, which is the pair the project's NFB guidebook highlights. It is a limit
    of the released data, not of the method, and belongs in any write-up that cites
    that biomarker.
    """

    pairs = []
    for phase_name, phase_low, phase_high in PHASE_BANDS:
        for amp_name, amp_low, amp_high in AMPLITUDE_BANDS:
            if phase_high > amp_low:
                continue  # bands must not overlap
            # Judge by the phase band's centre, not its upper edge: most of a band's
            # energy sits near the middle, and the edge test rejected theta/gamma even
            # though the synthetic check detected it at 53x. Centre frequencies
            # reproduce the measured outcome exactly -- delta and theta pass, alpha and
            # beta fail.
            if (amp_high - amp_low) < 2 * (phase_low + phase_high) / 2:
                continue  # sidebands would fall outside the amplitude band
            pairs.append((phase_name, phase_low, phase_high, amp_name, amp_low, amp_high))
    return pairs


def _sos(srate, low, high):
    nyq = srate / 2.0
    if not 0.0 < low < high < nyq:
        raise PACError("invalid_band", f"band {low}-{high} Hz invalid at {srate} Hz")
    return butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")


def _window(n_samples, srate, onset, start_s, end_s):
    s0 = onset + int(round(start_s * srate))
    s1 = onset + int(round(end_s * srate))
    if not 0 <= s0 < s1 <= n_samples:
        raise PACError("window_out_of_bounds",
                       f"window {start_s}..{end_s}s maps to {s0}:{s1} of {n_samples}")
    return s0, s1


def mean_vector_length(phase, amplitude):
    """Coupling strength in 0..1, normalized so loudness alone cannot raise it."""

    vectors = amplitude * np.exp(1j * phase)
    return np.abs(vectors.mean(axis=1)) / np.maximum(amplitude.mean(axis=1), EPS)


def pac_feature_names(channels):
    names = []
    for channel in channels:
        for phase_name, _, _, amp_name, _, _ in _pairs():
            names.append(f"{channel}:pac:{phase_name}_phase_{amp_name}_amp:response")
            names.append(f"{channel}:pac:{phase_name}_phase_{amp_name}_amp:baseline")
        names.append(f"{channel}:peak_alpha_frequency")
        names.append(f"{channel}:alpha_peak_power")
    upper = [str(c).strip().upper() for c in channels]
    if "C3" in upper and "C4" in upper:
        names.append("C3-C4:alpha_asymmetry")
    return names


def extract_pac_features(data, srate, onset_sample, channels):
    """Return (trials, features) matching pac_feature_names(channels)."""

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 3:
        raise PACError("invalid_shape", f"expected 3 dimensions, got {values.shape}")
    n_channels, n_samples, n_trials = values.shape
    if n_channels != len(channels):
        raise PACError("channel_mismatch",
                       f"{n_channels} channels of data against {len(channels)} labels")
    if not np.isfinite(values).all():
        raise PACError("nonfinite_eeg", "input contains NaN or infinite samples")

    response = _window(n_samples, srate, onset_sample, *RESPONSE_WINDOW_S)
    baseline = _window(n_samples, srate, onset_sample, *BASELINE_WINDOW_S)

    # Filter once per band, then reuse for every pair that needs it.
    phase_cache, amplitude_cache = {}, {}
    for name, low, high in PHASE_BANDS:
        filtered = sosfiltfilt(_sos(srate, low, high), values, axis=1)
        phase_cache[name] = np.angle(hilbert(filtered, axis=1))
    for name, low, high in AMPLITUDE_BANDS:
        filtered = sosfiltfilt(_sos(srate, low, high), values, axis=1)
        amplitude_cache[name] = np.abs(hilbert(filtered, axis=1))

    per_channel = []
    for c in range(n_channels):
        columns = []
        for phase_name, _, _, amp_name, _, _ in _pairs():
            for s0, s1 in (response, baseline):
                columns.append(
                    mean_vector_length(
                        phase_cache[phase_name][c, s0:s1, :].T,
                        amplitude_cache[amp_name][c, s0:s1, :].T,
                    )
                )
        segment = values[c, response[0]:response[1], :].T
        nperseg = min(segment.shape[1], int(round(srate)))
        freqs, psd = welch(segment, fs=srate, nperseg=nperseg, axis=1)
        band = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
        if not np.any(band):
            raise PACError("empty_alpha_band", "no spectral bins inside 8-13 Hz")
        alpha_psd = psd[:, band]
        alpha_freqs = freqs[band]
        # Where the alpha peak sits, which band power alone cannot distinguish.
        columns.append(alpha_freqs[np.argmax(alpha_psd, axis=1)])
        columns.append(np.log(np.maximum(alpha_psd.max(axis=1), EPS)))
        per_channel.append(np.stack(columns, axis=-1))

    features = np.concatenate(per_channel, axis=1)

    upper = [str(c).strip().upper() for c in channels]
    if "C3" in upper and "C4" in upper:
        left = per_channel[upper.index("C3")][:, -1]
        right = per_channel[upper.index("C4")][:, -1]
        # Log powers already, so their difference is the log ratio.
        features = np.concatenate([features, (right - left)[:, None]], axis=1)

    if not np.isfinite(features).all():
        raise PACError("nonfinite_features", "PAC calculation produced a nonfinite value")
    return features
