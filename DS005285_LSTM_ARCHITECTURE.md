# DS005285 LSTM Architecture

> **Superseded in part — see [RESULTS.md](RESULTS.md).**
>
> This document describes the original single-dataset (ds005285) design: three bands
> (alpha/beta/theta), 20 features, 29 participants. The pipeline has since expanded to
> nine datasets and 678 participants, and the feature set was rebuilt with
> evoked-potential, delta and gamma features that measurably outperform the three
> bands recorded here. The subject-splitting, normalization and leakage discipline
> below still applies unchanged.

Generated only after the executable sanity gate passed on 2026-07-22 18:07:10.

## Dataset scope and control-session selection

- Dataset: OpenNeuro `ds005285`, DOI `doi:10.18112/openneuro.ds005285.v1.0.0`, local BIDS release only.
- Modality: raw EEG `.set`/`.fdt` recordings and their BIDS sidecars only; derivatives and every
  other dataset directory are excluded.
- Participants: 29 (`sub-001` through `sub-029`), participant IDs retained.
- Metadata audit: 116 raw recordings = 29 participants × 4 sessions, 40 stimulus trials each.
- Embedded EEGLAB provenance maps `ses-1→SIT`, `ses-2→VR`, `ses-3→VR_cTENS`, and
  `ses-4→VR_sTENS` for every participant. `ses-1/SIT` is the unique session without VR or
  TENS and is therefore the control/no-intervention condition. This mapping is validated at
  runtime and the pipeline stops if it changes.

## Final preprocessing and input contract

1. Load each `ses-1/SIT` EEGLAB recording with MNE and validate sidecars.
2. Select `Fz`, `Cz`, `C3`, `C4`; all are present in the audited release. If a future file lacks
   one, choose the closest unique standard-1020 coordinate and record the mapping.
3. Apply a zero-phase 50 Hz FIR notch, zero-phase 1–45 Hz FIR band-pass, then resample from
   1000 Hz to 250 Hz.
4. Epoch −0.5 to +1.0 s around each documented pain-stimulus onset and apply −0.5 to 0 s
   mean baseline correction.
5. Reject only explicit corruption: non-finite samples, a channel below 0.05 µV peak-to-peak,
   or a channel above 500 µV peak-to-peak. Save all reasons and abort for review if >25% of
   any participant's requested trials would be removed.
6. Estimate DPSS multitaper spectra in 1.0 s windows with a nominal 0.25 s stride; at 250 Hz,
   the explicit 63-sample stride is 0.252 s (74.8% overlap), bandwidth 4 Hz. One second
   provides 1 Hz Fourier spacing, the shortest defensible window
   for a band beginning at 1 Hz; shorter windows are prohibited. Overlap supplies temporal
   resolution but does not make windows statistically independent.
7. For delta 1–4, theta 4–8, alpha 8–13, beta 13–30, and gamma 30–45 Hz, compute band
   power / total 1–45 Hz power, clip only at floating-point epsilon, then take natural log.

Final tensor: **`[trial, 3 time_steps, 20 features]`** (`float32`).
The 3 window starts are [-0.5, -0.248, 0.0040000000000000036] s relative
to the pain stimulus; feature order is four channel-major groups × five bands. The verified sanity
tensor was `[16, 3, 20]` with no NaN/Inf.

Channel verification:
- `sub-001`: Fz→Fz, Cz→Cz, C3→C3, C4→C4
- `sub-002`: Fz→Fz, Cz→Cz, C3→C3, C4→C4

## Labels and retained ratings

The dataset's `task-29ByANT_events.json` defines `s32` as low-intensity laser and `s64` as
high-intensity laser. BIDS files store numeric `32`/`64`; labels are exactly `32→0` (low) and
`64→1` (high). Unknown event codes are never coerced.

No trial-level 0–10 rating column was found. All 116 BIDS events files contain only `onset`, `duration`, `sample`, and `value`; raw EEGLAB events contain stimulus types `32`/`64` plus impedance markers. The separate rating field is therefore retained as NaN. Participant-level `laser_low`/`laser_high` fields are calibration energies, not trial ratings, and are deliberately not substituted.

## Subject-wise splitting and normalization

A seeded permutation of unique participant IDs creates approximately 60/20/20% train,
validation, and held-out test subject sets. Assertions require pairwise-disjoint sets whose
union is all participants. Every trial from one participant stays in exactly one split; no
overlapping epoch can cross a split. The 20 feature means and standard deviations are fit
over training-subject trials/time steps only, saved, and reused unchanged for validation and
test. Test data do not control preprocessing, early stopping, normalization, or class weights.

## LSTM equations and architecture

For time step `t`, input `x_t ∈ R^20`, previous hidden/cell states `h_(t-1), c_(t-1) ∈ R^16`,
and PyTorch gate order `(i, f, g, o)`:

```text
i_t = sigmoid(W_ii x_t + b_ii + W_hi h_(t-1) + b_hi)
f_t = sigmoid(W_if x_t + b_if + W_hf h_(t-1) + b_hf)
g_t = tanh   (W_ig x_t + b_ig + W_hg h_(t-1) + b_hg)
o_t = sigmoid(W_io x_t + b_io + W_ho h_(t-1) + b_ho)
c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
logits = W_y h_T + b_y ∈ R^2
prediction = argmax(logits)
```

It is one unidirectional, one-layer LSTM with 16 hidden units followed by a 2-logit dense
layer. There is no attention, bidirectionality, dropout, convolution, or recurrence beyond
the single layer.

## Parameter counts

| Tensor | Shape | Parameters |
|---|---:|---:|
| `lstm.weight_ih_l0` | `(64, 20)` | 1,280 |
| `lstm.weight_hh_l0` | `(64, 16)` | 1,024 |
| `lstm.bias_ih_l0` | `(64,)` | 64 |
| `lstm.bias_hh_l0` | `(64,)` | 64 |
| `classifier.weight` | `(2, 16)` | 32 |
| `classifier.bias` | `(2,)` | 2 |
| **Total** | | **2,466** |

## Training procedure

Cross-entropy loss, AdamW (`lr=1e-3`, `weight_decay=1e-4`), batch size 64, maximum 100
epochs, gradient norm clipping at 1.0, deterministic seed 20250722, and early stopping on
validation loss (patience 12, minimum improvement 1e-4). Inverse-frequency class weights are
enabled only if the training-set majority/minority ratio exceeds 1.20, and are computed from
training labels only. The best validation-loss state is restored before final evaluation.

## Evaluation

Validation and held-out test reports save accuracy, balanced accuracy, macro F1, sensitivity
(`TP/(TP+FN)` for high pain), specificity (`TN/(TN+FP)` for low pain), and the fixed-label
`[[TN, FP], [FN, TP]]` confusion matrix.

## Data-leakage protections

- Dataset root DOI/name lock; raw `ds005285` EEG only.
- Control mapping verified from every raw file before selection.
- Unique-subject split before normalization; split intersections asserted empty.
- Train-only normalization and optional train-only class weights.
- Validation only for early stopping; held-out test evaluated once after restoring best state.
- Cache keys include the complete preprocessing configuration hash.
- Trial IDs and subject IDs are exported for audit; no trial duplication across splits.

## Assumptions and discovered limitations

- `SIT` is interpreted as seated/no-intervention control because all alternative sessions are
  explicitly intervention-labelled in embedded provenance. No top-level `sessions.tsv` exists.
- No trial-level 0–10 rating column was found. All 116 BIDS events files contain only `onset`, `duration`, `sample`, and `value`; raw EEGLAB events contain stimulus types `32`/`64` plus impedance markers. The separate rating field is therefore retained as NaN. Participant-level `laser_low`/`laser_high` fields are calibration energies, not trial ratings, and are deliberately not substituted.
- The 1.5 s epoch bounds the lowest-frequency estimation. A 1.0 s window is the minimum valid
  choice for 1 Hz resolution; delta estimates remain noisier than higher bands.
- Filtering is non-causal zero-phase offline preprocessing. A future streaming FPGA front end
  must replace it with validated causal filters and quantify the domain shift.

## Future fixed-point and FPGA export plan

1. Freeze the saved train-only normalization constants and feature ordering.
2. Calibrate per-tensor or per-row symmetric scales on training data only for 8–16 bit inputs,
   recurrent weights, dense weights, cell state, and hidden state.
3. Implement sigmoid/tanh with bounded LUT or piecewise-linear approximations; preserve PyTorch
   gate order `(i,f,g,o)` from the exported arrays.
4. Run bit-accurate Python inference against float32 test logits and select word lengths using
   saturation/error/metric trade-offs without tuning on test labels.
5. Export golden feature sequences, intermediate gates/states, logits, and decisions for RTL
   co-simulation; verify every tensor dimension and bias addition.
6. Replace offline spectral/filter blocks with streaming equivalents only after separate
   numerical validation, latency/resource analysis, and end-to-end regression tests.
