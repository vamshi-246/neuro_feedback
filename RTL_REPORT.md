# RTL implementation report: no-MVAR pain classifier

This report covers the synthesizable Verilog for the `three_class__no_mvar` model in
[outputs/deploy/deployment_model.json](outputs/deploy/deployment_model.json): what it computes, how it is
built, how it was verified, and what is still open. The old threshold-based RTL has been
moved to [archive/legacy_rtl/](archive/legacy_rtl/).

## 1. Result in one paragraph

The design takes three EEG channels (Cz, C3, C4) at 1 kHz and a stimulus trigger, and
outputs one pain class (Low / Moderate / High) per stimulus. It reproduces the software
pipeline exactly: the same zero-phase filters, the same 16 features, the same 8-bit
quantizer and the same linear layer. It uses no floating point, divider or log unit. A
bit-exact Python model of the integer arithmetic matches the float software on **all
5,722 validation trials**: the predicted class agrees on 100% of them, and
dataset-macro accuracy is **0.4384** for both. The Verilog matches that integer model bit for bit
(section 7): 10 out of 10 test epochs, bit for bit.

## 2. What the hardware computes

| # | Feature | Computation (window = ms after stimulus) |
|---|---|---|
| 0 | Cz delta power, w0 | ln mean(δ²), 0–300 |
| 1 | Cz ERP RMS | √mean(e²), 0–600, e = 1–30 Hz, baseline-corrected |
| 2 | Cz theta power, w0 | ln mean(θ²), 0–300 |
| 3 | C3–C4 delta coupling | Pearson r of δ(C3), δ(C4), 0–1000 |
| 4 | Cz P2 peak | max e, after N2 … 550 |
| 5 | Cz N2–P2 amplitude | P2 − N2 |
| 6 | Cz N2 peak | min e, 150–300 |
| 7 | C3 delta power, w0 | ln mean(δ²), 0–300 |
| 8 | Cz delta/gamma | ln Pδ − ln Pγ, 0–300 |
| 9 | Cz delta power, w1 | ln mean(δ²), 300–600 |
| 10 | C4 delta power, w0 | ln mean(δ²), 0–300 |
| 11 | C4 beta/delta | ln Pβ − ln Pδ, 0–300 |
| 12 | Cz P2 window mean | mean e, 250–450 |
| 13 | Cz N2 window mean | mean e, 150–250 |
| 14 | Cz beta/delta | ln Pβ − ln Pδ, 0–300 |
| 15 | Cz Hjorth mobility | √(var(Δx)/var(x)), 1–45 Hz, 0–1000 |

Then `q_i = round(clip((f_i − lo_i)/(hi_i − lo_i), 0, 1)·255)`, `logit_k = bq_k + Σ Wq_ki·q_i`,
class = argmax. Bands are 4th-order Butterworth band-passes: δ 1–4, θ 4–8, β 13–30, γ 30–45,
ERP 1–30 and broadband 1–45 Hz.

## 3. Architecture

```
 s_cz/s_c3/s_c4 (20 b, 1 kHz)        s_trigger
        │                                │
        ▼                                ▼
 ┌──────────────────┐   epoch ready ┌───────────┐
 │ raw circular RAM │──────────────▶│ sequencer │  9 runs: (band, channel)
 │ 4096 × 60 b      │               └─────┬─────┘
 └────────┬─────────┘                     │
          ▼                               ▼
 ┌─────────────────────────┐   ┌───────────────────────┐
 │ filtfilt_engine         │──▶│ feature buffers A, B  │ 1500 × 24 b each
 │ 4 DF2T biquads, 1 mult  │   └──────────┬────────────┘
 │ fwd → work RAM → bwd    │              ▼
 └─────────────────────────┘   ┌───────────────────────┐
                               │ feature_engine        │ window sums, min/max,
                               │ → 16 (A, B) pairs     │ exact wide products, isqrt
                               └──────────┬────────────┘
                                          ▼
                               ┌───────────────────────┐
                               │ threshold_quantizer   │ 8-step binary search
                               │ → 16 × 8-bit codes    │ A·2^40 ≥ C_k·B
                               └──────────┬────────────┘
                                          ▼
                               ┌───────────────────────┐
                               │ linear_classifier     │ 48 MACs + argmax
                               └──────────┬────────────┘
                                          ▼
                         result_class, 3 logits, 16 codes
```

The filter engine runs 9 times per epoch, once per (band, channel):

| Run | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Band | ERP | δ | θ | β | γ | 1–45 | β | δ | δ |
| Channel | Cz | Cz | Cz | Cz | Cz | Cz | C4 | C3 | C4 → buffer B |

## 4. Key design decisions

1. **The filters are zero-phase, as in training.** The model was trained on `sosfiltfilt`
   (forward + reverse) output, and a causal streaming filter would give different
   features. The core therefore buffers the 3 s epoch (1 s before the stimulus, 2 s after) and runs
   both passes after it ends, including scipy's 27-sample odd padding and its `zi·x0`
   initial states. Samples keep being written to the 4096-deep ring while it works, so
   no input is lost.
2. **The biquads are gain-balanced.** scipy puts the whole filter gain (7.7e-9 for δ) in
   the first section, which wrecks fixed-point precision. The gain is redistributed so
   that each partial cascade peaks at 1. It is the same filter (float difference ≤ 3e-12).
3. **The quantizer uses threshold comparisons, not log/div/sqrt.** Each 8-bit code is the count
   of thresholds `T_k = lo + (k−½)(hi−lo)/255` that the feature reaches. Every feature
   is monotone in a ratio A/B, so `f ≥ T_k` becomes the exact integer test
   `A·2^40 ≥ C_k·B`, with `C_k = c(T_k)·2^40` precomputed in ROM (c = T, e^T or T²).
   For example, `ln mean(x²) ≥ T` becomes `Σx² ≥ e^T·N·2^20`, and Pearson r ≥ T becomes
   `u ≥ T·isqrt(v·w)`. The only root left is one exact integer square root.
4. **The arithmetic is wide but slow.** Wide operations (up to 72×72 bit) run on shift-add
   multipliers that cost adders, not DSPs. Only the filter multiply (48×36) and the
   per-sample square (27×27) use DSP blocks. Throughput is irrelevant: the epoch lasts 2 s
   and processing takes milliseconds.

## 5. Number formats

| Signal | Format |
|---|---|
| Input sample | signed 20 b, LSB 2⁻¹⁰ µV (±512 µV) |
| Filter state / samples | signed 48 b, LSB 2⁻²⁴ µV |
| Filter coefficients, zi | signed 36 b, 32 fraction bits |
| Feature buffers | signed 24 b, LSB 2⁻¹⁰ µV |
| Window accumulators | signed 64 b |
| A, B pairs | signed 72 b |
| Thresholds C_k | signed 56 b, 40 fraction bits |
| Weights Wq, bias bq | signed 28 b, 24 fraction bits |
| Logits | signed 40 b |

Rounding is round-half-up (`(P + 2^(s−1)) >>> s`) and every filter result saturates. The
input resolution was chosen by measurement. At 2⁻⁶ µV, input rounding alone flipped
gamma-power codes. At 2⁻¹⁰ µV, filter error (≈3·10⁻⁴ in ln P) sits about 100× below one
8-bit code step.

## 6. Interface and timing

| Port | Dir | Meaning |
|---|---|---|
| `clk`, `rst` | in | clock, synchronous reset |
| `s_valid`, `s_cz`, `s_c3`, `s_c4` | in | one 1 kHz sample per channel, 20 b signed |
| `s_trigger` | in | high with `s_valid` on the stimulus-onset sample |
| `result_valid` | out | one-cycle pulse when a result is ready |
| `result_class` | out | 0 Low (rating < 4), 1 Moderate (4 to < 7), 2 High (≥ 7) |
| `logit_low/mod/high` | out | raw scores (scaled 2²⁴), for confidence or debugging |
| `feature_codes` | out | the 16 8-bit codes, `q[i] = [8i +: 8]` |
| `busy`, `triggers_dropped` | out | status; triggers that arrive mid-epoch or with < 1 s of history are counted and ignored |

Latency is 2 s of data after the trigger, plus 2.31 M clock cycles of processing
(46 ms at 50 MHz, measured in simulation). The filter runs take 9 × 250k cycles. The default constraint is 50 MHz
([pain_classifier.xdc](RTL/pain_classifier.xdc)), and anything above about 5 MHz meets real time.

## 7. Verification

| Check | Scope | Result |
|---|---|---|
| Float reference vs training archive | 2 subjects | identical features |
| Integer golden model vs float software | **all 5,722 validation trials, 136 subjects** | class agreement **100%**; dataset-macro **0.4384 vs 0.4384**; 99.80% of codes identical, the rest off by 1 |
| `filtfilt_engine` vs golden | 6 bands on Cz, 1500 output samples each | **bit-exact** (C3/C4 runs are covered by the system test) |
| `pain_classifier_top` vs golden | 10 epochs: 8 real trials from 4 datasets (all 3 classes), 1 saturation stress, 1 all-zero | **10/10 bit-exact**: all 32 A/B values, 16 codes, 3 logits and class; stray trigger dropped every time |

The system test checks every (A, B) pair, all 16 codes, all 3 logits and the class. It
also keeps streaming samples while the core is busy and sends one stray trigger, which
must be counted as dropped. The golden-vs-float numbers are in
[outputs/deploy/rtl_golden_validation.json](outputs/deploy/rtl_golden_validation.json).

**Not yet done:** Vivado synthesis and timing. Vivado is not installed on the machine this was
built on. The code uses only plain synthesizable Verilog-2005: inferred RAM/ROM with
`$readmemh`, synchronous reset, no latches and no initial blocks outside the ROMs.
Rough estimate for the Zynq-7010: about 25 BRAM36 (raw ring 7, work RAM 6, threshold ROM 7,
buffers 4), about 12 DSP48 and a few thousand LUTs. That fits comfortably, but it is an
estimate until [synth_vivado.tcl](RTL/synth_vivado.tcl) is run.

## 8. Files

| Path | What |
|---|---|
| [RTL/pain_classifier_top.v](RTL/pain_classifier_top.v) | top: ring buffer, trigger capture, run sequencer |
| [RTL/filtfilt_engine.v](RTL/filtfilt_engine.v) | zero-phase biquad engine |
| [RTL/feature_engine.v](RTL/feature_engine.v) | window scans → 16 (A, B) pairs |
| [RTL/threshold_quantizer.v](RTL/threshold_quantizer.v) | 8-bit codes by threshold search |
| [RTL/linear_classifier.v](RTL/linear_classifier.v) | 3×16 MAC + argmax |
| `RTL/mul_seq.v`, `isqrt_seq.v`, `sdp_ram.v`, `rom_sync.v` | building blocks |
| [RTL/mem/](RTL/mem/) | ROMs: filter coefficients, thresholds, weights, bias |
| [RTL/synth_vivado.tcl](RTL/synth_vivado.tcl), [RTL/pain_classifier.xdc](RTL/pain_classifier.xdc) | synthesis script, clock constraint |
| [Testbench/](Testbench/) | unit test (filter) and system test |
| [sim/](sim/), [sim/run_sim.sh](sim/run_sim.sh) | committed test vectors; `bash sim/run_sim.sh` |
| [scripts/hardware/hw_golden.py](scripts/hardware/hw_golden.py) | bit-exact golden model and ROM generator |
| [scripts/hardware/evaluate_golden.py](scripts/hardware/evaluate_golden.py) | golden vs float on the validation set |

If the model is retrained, regenerate everything with `py scripts/hardware/hw_golden.py --write-mem`
followed by the two vector scripts. The RTL needs no change unless the feature list does.

## 9. Open issues, most important first

1. **The model is stimulus-locked, but the product is meant to be continuous.** These features
   (especially N2/P2) assume a known stimulus onset. The stated goal is monitoring any
   pain on any person with no trigger. That needs a model retrained on untriggered
   windows. The filter engine, feature engine, quantizer and classifier carry over. The
   trigger logic would become a periodic window scheduler.
2. **Electrode reference and cleaning: the device sees a different signal than the model
   was trained on, even on the same 3 electrodes.** This is not about which channels were
   picked — it is about what each channel's voltage was measured *against*, and that was
   decided before channel selection, using the full montage:
   - **Re-referencing.** The dataset authors took the original 32–64-channel recording and,
     at every sample, subtracted the average of all those channels from every channel
     (the `_rerefer.set` files). Only *after* that finished does
     [build_rich_pilot.py](scripts/preprocessing/build_rich_pilot.py) pick out the Cz/C3/C4
     columns. So "Cz" in training is `Cz_raw − average(32–64 channels)`, not raw Cz.
   - **ICA cleaning.** Likewise, blink/muscle/heartbeat removal (`_mark_ica.set`) runs on
     the full montage before any channel is selected, because ICA needs many channels to
     separate an artifact pattern from a brain pattern.
   - A 3-electrode device cannot reproduce either step: the other 29–61 electrodes needed
     for the average, or for ICA, simply are not there. Whatever physical reference the
     analog front end uses (an ear clip, a mastoid electrode, two electrodes differenced)
     gives `Cz_raw − (something else)`, a different number from what the weights, the
     8-bit bounds and the filter coefficients were fitted on. The size of the resulting
     accuracy loss is currently unmeasured.
   - **Fix, in order of effort:**
     1. **Re-reference the training data to match the device before retraining.** Most of
        these datasets also recorded a mastoid or earlobe channel; re-run the feature
        pipeline on `Cz_raw − mastoid` (or whatever reference the hardware will use)
        instead of the average, and refit `export_deployment.py`. This changes only the
        ROM files, not the RTL.
     2. **Retrain without ICA**, or with a cheap substitute, so the training signal matches
        what an uncleaned 3-electrode stream looks like.
     3. **Add artifact rejection in hardware** as a stopgap while 1–2 are done: discard an
        epoch if any channel's raw sample exceeds roughly ±100 µV (a blink), which is one
        comparator per channel feeding `result_valid`/a new `artifact` flag — cheap, and it
        turns a wrong answer into "no decision" rather than silently presenting one.
     4. **Optionally wire up one extra electrode above the eye** purely to flag blinks for
        step 3, if the false-rejection rate from amplitude thresholding alone is too high.
     Step 1 should happen before any accuracy number is trusted for the real device; steps
     3–4 reduce risk while that retraining is done.
3. **Synthesis and timing are unverified.** Run `synth_vivado.tcl`. The longest paths are
   expected in the 48×36 filter multiply and the 88-bit rounding/saturation behind it.
   Both can take an extra register if needed: the schedule is a few states in
   `filtfilt_engine.v`.
4. **The Vivado project still holds the old files.** `neuro_feedback.xpr` imported copies of
   the legacy RTL. Remove those sources and add the files above, or use the TCL script.
5. **Simulation speed.** Icarus simulates about 2.5k cycles/s here (≈15–20 min per epoch),
   so the scripts run epochs in parallel. Verilator would be about 100× faster.
