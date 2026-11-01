# FPGA deployment guide

Everything the hardware build needs, and where each piece comes from.

Read this instead of [LSTM_HARDWARE_REFERENCE.md](LSTM_HARDWARE_REFERENCE.md). That
document is accurate, but it specifies the LSTM prototype, which scored **0.4479** and
was superseded. Nothing in the current result chain uses it.

## What actually ships

```
4 EEG channels (Fz, Cz, C3, C4)
        |
        v
16 scalar features per trial          <- the extractor, fixed at synthesis
        |
        v
logits = W_eff @ x + b_eff            <- 48 MACs, 3 adds
        |
        v
argmax                                <- the predicted class
```

That is the whole model. **51 numbers**: `W_eff` is 3x16, `b_eff` is 3.

Three things you do *not* need to build:

- **No softmax.** It is monotonic, so `argmax` over the raw logits picks the same class.
  Softmax is only needed if you want a confidence number out.
- **No standardization stage.** The `(x - mean) / sd` step is folded into the weights
  (see below), so there is no subtract-and-divide in the datapath.
- **No LSTM, no gates, no cell state, no sequence buffer.** The model is not recurrent.
  One trial in, one class out.

### Why the weights are pre-folded

The model was fitted on standardized inputs, `logits = W @ ((x - mean) / sd) + b`. That
is algebraically identical to a single matrix multiply:

```
W_eff = W / sd
b_eff = b - W_eff @ mean
```

Both forms are in the export, and the exporter checks them against each other on every
validation trial — they agree to **1.3e-15**, which is float64 rounding. Build the
folded form; the standardized form is there only if you want to verify against Python.

## The five things you need, and where to get them

| # | what | where |
|---|---|---|
| 1 | the 51 weights | `outputs/deploy/deployment_model.json` -> `three_class__fixed.folded_form` |
| 2 | the 16 feature definitions | same file -> `three_class__fixed.columns` |
| 3 | the 8-bit input bounds | same file -> each column's `quantizer_low` / `quantizer_high` |
| 4 | how each feature is computed | [rich_feature_extraction.py](scripts/preprocessing/rich_feature_extraction.py) — the reference implementation |
| 5 | per-user calibration values | generated per user at calibration time; see below |

Regenerate the whole export at any time with:

```bash
python scripts/preprocessing/export_deployment.py
```

It refits from the locked subject split, verifies the weights reproduce the published
per-user baseline, and rewrites the JSON.

## The 16 features

All windows are seconds relative to stimulus onset. `sgn` marks columns that take
negative values — 15 of 16 do, so **the input path must be signed**. An unsigned
quantizer will destroy the evoked-potential columns.

| # | column | ch | band | window | sgn | 8-bit low | 8-bit high |
|---|---|---|---|---|---|---|---|
| 1 | `Cz:delta:w0:log_absolute` | Cz | delta 1-4 | 0.00-0.30 | y | -1.142 | 5.095 |
| 2 | `Cz:erp_erp_rms` | Cz | 1-30 passband | 0.00-0.60 | n | 2.059 | 15.242 |
| 3 | `Cz:theta:w0:log_absolute` | Cz | theta 4-8 | 0.00-0.30 | y | -1.498 | 4.525 |
| 4 | `Cz:erp_n2p2_amplitude` | Cz | 1-30 passband | 0.15-0.55 | y | 3.790 | 68.589 |
| 5 | `C3-C4:delta:coupling` | C3-C4 | delta 1-4 | full | y | -0.722 | 0.958 |
| 6 | `Cz:erp_p2_amplitude` | Cz | 1-30 passband | 0.30-0.55 | y | 1.877 | 31.202 |
| 7 | `Cz:erp_n2_amplitude` | Cz | 1-30 passband | 0.15-0.30 | y | -43.259 | 1.772 |
| 8 | `Cz:w0:delta_over_gamma` | Cz | delta/gamma | 0.00-0.30 | y | 0.782 | 8.165 |
| 9 | `mvar:parametric_power:theta:Cz` | Cz | theta 4-8 | full | y | 2.523 | 6.788 |
| 10 | `C4:delta:w0:log_absolute` | C4 | delta 1-4 | 0.00-0.30 | y | -1.770 | 4.174 |
| 11 | `mvar:parametric_power:delta:Cz` | Cz | delta 1-4 | full | y | 3.103 | 7.742 |
| 12 | `Cz:erp_p2_window_mean` | Cz | 1-30 passband | 0.25-0.45 | y | -5.917 | 12.830 |
| 13 | `Cz:delta:w1:log_absolute` | Cz | delta 1-4 | 0.30-0.60 | y | -0.881 | 5.115 |
| 14 | `C3:delta:w0:log_absolute` | C3 | delta 1-4 | 0.00-0.30 | y | -1.851 | 3.976 |
| 15 | `Cz-C3:delta:coupling` | Cz-C3 | delta 1-4 | full | y | -0.341 | 0.980 |
| 16 | `Cz:erp_n2_window_mean` | Cz | 1-30 passband | 0.15-0.25 | y | -23.293 | 11.720 |

Cz carries 13 of the 16. C3 appears in 3, C4 in 2. **Fz is never used** — it can be
dropped from the montage, or kept only for artifact rejection.

### The arithmetic primitives, which is what sets area

| count | primitive |
|---|---|
| 5 | one band power, then log |
| 3 | mean or RMS over a window |
| 3 | peak search in a window |
| 2 | band-pass two channels, one Pearson correlation |
| 2 | **least-squares AR fit per trial, then spectrum evaluation** |
| 1 | subtract two log powers already computed |

Six distinct primitives. The AR fit (rows 9 and 11) is by far the most expensive — it
needs a per-trial least-squares solve.

### On dropping the AR solver

**Correcting something I said earlier in this project's notes:** the "no-MVAR list
scores better" result (0.4529 vs 0.4503) was measured with gradient boosting, not with
the linear head that actually ships. Re-measured on the linear head, dropping the AR
columns *costs* accuracy on the three-class task:

| task | with AR columns | AR removed |
|---|---|---|
| three-class, dataset-macro | **0.4459** | 0.4386 |
| three-class, per-user baseline | **0.5674** | 0.5541 |
| severe, dataset-macro | **0.6201** | 0.6171 |
| severe, per-user baseline | 0.7486 | **0.7540** |

So: **keep the AR solver for three-class, drop it for severe.** The no-MVAR variants are
exported as `three_class__no_mvar` and `severe__no_mvar` if the area cost is decisive —
the three-class penalty is about 1.3 points, which is at the edge of this project's
±0.7-point noise floor. The two lists share 13 of 16 columns, so one extractor with a
3-column option can serve both.

## Fixed point

The RTL input path is 8 bits. Each feature is clipped to its `[low, high]` bound and
mapped to 0-255:

```
q = round(clip((x - low) / (high - low), 0, 1) * 255)
```

Bounds were fitted on training rows only. Quantizing costs **nothing measurable**:

| | float64 | 8-bit |
|---|---|---|
| three-class, dataset-macro | 0.4459 | 0.4464 |
| severe, dataset-macro | 0.6201 | 0.6209 |

Both differences are well inside noise. Note the 8-bit number is marginally *higher* —
that is coincidence, not a benefit of quantization.

The weights themselves have not been quantized yet. If you need a fixed-point `W_eff`,
say so and it can be fitted and rescored the same way.

## Per-user calibration (optional)

The device can be shipped without this. If a setup wizard is acceptable, calibration
gives the largest single gain in the project.

| what moves | numbers stored per user | three-class | severe |
|---|---|---|---|
| nothing (shared model) | 0 | 0.5674 | 0.7486 |
| **biases only** | **3** (2 for severe) | 0.5899 | — |
| all weights | 51 | **0.6104** | **0.8022** |

Calibration needs **60 labeled trials**, roughly 10-25 minutes. The bias-only arm
captures +7.2 of the +7.9 total, so if storage or wizard length is tight, store three
numbers per user and add them to `b_eff`. That requires no change to the datapath at
all — only the bias register file becomes writable.

The fitting procedure is in [personalized_head.py](scripts/preprocessing/personalized_head.py):
a warm-started L2 pull toward the shared head, so a short calibration barely moves the
model and a long one is trusted. It runs off-device.

## What accuracy to put on the datasheet

Be careful here — the numbers above are not all on the same population.

| figure | what it means | use it for |
|---|---|---|
| **0.4459** | dataset-macro, 5,722 held-out trials, 132 unseen subjects | **the datasheet** |
| 0.5674 | per-user, the 12 subjects with enough trials to calibrate | calibration comparisons only |
| 0.6104 | the same 12 subjects, after 60 calibration trials | calibration comparisons only |

The 12 deep subjects all come from ds005285 and ds005473, the two datasets the model
already handles best (0.5193 and 0.5935, against 0.4527 across the other seven). They
are a favourable sample, not a representative one.

**Claim 0.4459 three-class and 0.6201 severe on unseen users.** Report the calibration
gain as a *delta* (+7.9 points three-class, +4.5 severe), which is valid because it is
measured within one population with calibration as the only thing that changes.

Chance is 0.3333 for three classes and 0.5000 for severe, so read the margin above
chance, not the raw number. Severe looks much better than three-class mostly because it
is an easier question, not because the model understands more.

## Files that are not part of this

| file | why it is here |
|---|---|
| [train_lstm.py](scripts/preprocessing/train_lstm.py) | the 0.4479 LSTM baseline; the current pipeline imports only its data loaders |
| `outputs/all9_full/all9_pain_lstm_final_seed20260726.pt` | **a subject-split record, not a model.** The name is legacy |
| `outputs/ds005285_lstm/` | a binary stimulus-intensity model on one dataset — a different task |
| [LSTM_HARDWARE_REFERENCE.md](LSTM_HARDWARE_REFERENCE.md) | correct, but specifies the superseded LSTM |
