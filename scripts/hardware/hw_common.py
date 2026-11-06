"""Shared pieces for the hardware golden model: data loading and the float reference.

The float reference recomputes the 16 no-MVAR features straight from their
definitions in rich_feature_extraction.py, for three channels only (Cz, C3, C4).
Every filter is per channel, so dropping Fz cannot change any of these values; the
check against the training archive in check_reference() proves it.
"""

import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "preprocessing"))

MODEL_KEY = "three_class__no_mvar"
CHANNELS = ["Cz", "C3", "C4"]            # channel index 0, 1, 2 in hardware
FS = 1000.0

# Bands, in the order the hardware's coefficient ROM stores them.
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("beta", 13.0, 30.0),
         ("gamma", 30.0, 45.0), ("erp", 1.0, 30.0), ("broad", 1.0, 45.0)]
BAND_ID = {name: i for i, (name, _, _) in enumerate(BANDS)}

# All windows, as sample offsets from onset (start inclusive, end exclusive).
BASE = (-500, 0)
W0 = (0, 300)
W1 = (300, 600)
N2_SEARCH = (150, 300)
N2_MEAN = (150, 250)
P2_MEAN = (250, 450)
P2_END = 550
RMS = (0, 600)
RESP = (0, 1000)                          # coupling and Hjorth window


def load_model():
    path = os.path.join(REPO, "outputs", "deploy", "deployment_model.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)[MODEL_KEY]


def butter_sos(low, high):
    return butter(4, [low / (FS / 2), high / (FS / 2)], btype="bandpass", output="sos")


def load_subject(dataset_id, subject_id, cache_dir=None):
    """Admitted trials of one subject: (data[3, samples, trials], onset, epoch_index)."""

    from build_dataset import ONSET_TOLERANCE_SAMPLES
    from dataset_registry import REGISTRY
    from eeglab_io import load_recording, pick_channels

    cache_dir = cache_dir or os.path.join(REPO, "data_cache")
    spec = REGISTRY[dataset_id]
    stem = os.path.join(cache_dir, dataset_id, f"{subject_id}_{spec.derivative_stage}")
    rec = load_recording(dataset_id, subject_id, stem + ".set", stem + ".fdt")
    if abs(rec.srate - FS) > 1e-6:
        raise ValueError(f"{dataset_id} {subject_id}: sample rate {rec.srate}, need 1000")
    ch = pick_channels(rec.channel_labels, CHANNELS)
    expected = spec.pre_stim_s * rec.srate
    keep, epochs = [], []
    for t in range(rec.data.shape[2]):
        onset = float(rec.onset_samples[t])
        if not rec.trial_ok[t] or not np.isfinite(onset):
            continue
        if abs(onset - expected) > ONSET_TOLERANCE_SAMPLES:
            continue
        if not str(rec.event_types[t]).strip():
            continue
        keep.append(t)
        epochs.append(t + 1)
    data = rec.data[np.ix_(ch, np.arange(rec.data.shape[1]), keep)].astype(np.float64)
    return data, int(round(expected)), np.asarray(epochs, dtype=np.int64)


def _sl(onset, window):
    return slice(onset + window[0], onset + window[1])


def float_features(data, onset):
    """The 16 features, float64, in deployment-JSON column order. Returns (trials, 16)."""

    def filt(band):
        _, lo, hi = BANDS[BAND_ID[band]]
        return sosfiltfilt(butter_sos(lo, hi), data, axis=1)

    def logpow(x, window):
        return np.log(np.maximum(np.mean(np.square(x[:, _sl(onset, window), :]), axis=1),
                                 1e-12))

    cz, c3, c4 = 0, 1, 2
    delta, theta, beta, gamma = filt("delta"), filt("theta"), filt("beta"), filt("gamma")
    erp = filt("erp")[cz]
    erp = erp - erp[_sl(onset, BASE)].mean(axis=0)
    broad = filt("broad")[cz][_sl(onset, RESP)]

    n2_seg = erp[_sl(onset, N2_SEARCH)]
    n2 = n2_seg.min(axis=0)
    n2_idx = n2_seg.argmin(axis=0) + onset + N2_SEARCH[0]
    p2 = np.array([erp[n2_idx[t] + 1:onset + P2_END, t].max() for t in range(erp.shape[1])])

    d_w0, d_w1 = logpow(delta, W0), logpow(delta, W1)
    t_w0, b_w0, g_w0 = logpow(theta, W0), logpow(beta, W0), logpow(gamma, W0)

    resp = delta[:, _sl(onset, RESP), :]
    c = resp - resp.mean(axis=1, keepdims=True)
    coupling = (c[c3] * c[c4]).sum(0) / np.maximum(
        np.sqrt((c[c3] ** 2).sum(0)) * np.sqrt((c[c4] ** 2).sum(0)), 1e-12)

    mobility = np.sqrt(np.var(np.diff(broad, axis=0), axis=0) / np.var(broad, axis=0))

    cols = {
        "Cz:delta:w0:log_absolute": d_w0[cz],
        "Cz:erp_erp_rms": np.sqrt(np.mean(np.square(erp[_sl(onset, RMS)]), axis=0)),
        "Cz:theta:w0:log_absolute": t_w0[cz],
        "C3-C4:delta:coupling": coupling,
        "Cz:erp_p2_amplitude": p2,
        "Cz:erp_n2p2_amplitude": p2 - n2,
        "Cz:erp_n2_amplitude": n2,
        "C3:delta:w0:log_absolute": d_w0[c3],
        "Cz:w0:delta_over_gamma": d_w0[cz] - g_w0[cz],
        "Cz:delta:w1:log_absolute": d_w1[cz],
        "C4:delta:w0:log_absolute": d_w0[c4],
        "C4:w0:beta_over_delta": b_w0[c4] - d_w0[c4],
        "Cz:erp_p2_window_mean": erp[_sl(onset, P2_MEAN)].mean(axis=0),
        "Cz:erp_n2_window_mean": erp[_sl(onset, N2_MEAN)].mean(axis=0),
        "Cz:w0:beta_over_delta": b_w0[cz] - d_w0[cz],
        "Cz:hjorth_mobility": mobility,
    }
    names = [c["name"] for c in load_model()["columns"]]
    return np.stack([cols[n] for n in names], axis=1)
