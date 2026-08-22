# Pain classification: what was tried, what worked, what did not

All numbers below are **balanced accuracy on held-out subjects** — people no model
ever trained on — using one locked subject split
(`outputs/all9_full/all9_pain_lstm_final_seed20260726.pt`), 678 participants and
28,452 trials across nine OpenNeuro datasets. The test split was never touched;
every figure is validation.

Balanced accuracy is used throughout rather than plain accuracy because the classes
are uneven. A model that always answers "Moderate" scores **43.7% plain accuracy**
while learning nothing; the same model scores **33.3% balanced**, which is chance for
three classes. Plain accuracy hides failure, balanced accuracy does not.

## Headline

| | balanced accuracy |
|---|---|
| chance (3 classes) | 0.3333 |
| starting point (channel-averaged alpha/beta/theta LSTM) | 0.4479 |
| **current best** (rich features, score model, tuned cut points) | **0.4856** |

Seven changes were tested. **One worked.**

| change | result | verdict |
|---|---|---|
| Ordinal labels (Low < Moderate < High) | 0.4353 vs 0.4479 | worse |
| Per-channel input (4 electrodes kept separate) | 0.4567 vs 0.4479 | within noise |
| **Richer features (evoked potential, delta, gamma)** | **0.4750 vs 0.4479** | **+2.7 points** |
| Predicting the 0–10 score, then binning | 0.4856 vs 0.4750 | within noise |
| 14-electrode montage instead of 4 | r 0.418 vs 0.403, tracking 89% vs 93% | no gain |
| MVAR (directed connectivity) | 0.4753 vs 0.4803 | slightly worse |
| Training on one dataset instead of pooling | −2.69 points average, lost on 7 of 9 | worse |

Noise on this validation set is roughly ±0.7 points, so anything smaller than about
1.5 points is not a real difference.

## The diagnosis that redirected the work

Early tuning kept landing near 45% regardless of what changed. A memorization test
settled why:

```
gradient boosting, deliberately unregularized:
    training   99.99%
    validation 40.4%
```

The features could separate training trials **perfectly** while transferring almost
nothing to new people. That ruled out the model, the labels, the split and the
training recipe in one step, and pointed at the feature set. Two unrelated model
families — a recurrent network and decision trees — then converged to the same
45–48% band, confirming it.

Supporting evidence:

- **Per-subject normalization made things worse** (0.4584 → 0.4283). Centring each
  person removes their average, and that average genuinely carries signal: subjects
  differ in how much pain they reported.
- **Within a single person the model barely discriminates.** Predicting that
  subject's own most common class scores 61.1% against the model's 40.6%. The model's
  real within-person advantage is about +3.9 balanced points.
- **Personalized calibration did not help** at roughly 15 calibration trials per
  person (0.4420 vs 0.4562 shared).

## The one change that worked

`feature_extraction.py` computed relative power for alpha, beta and theta only,
because `RTL/feature_vector_generator.v` line 15 concatenates exactly
`{alpha, beta, theta}`. That choice came from the hardware, not from what predicts
laser pain, and it omitted the best-established markers.

`rich_feature_extraction.py` rebuilds from raw signal: **194 features from 4
channels**, 41 per channel plus 30 cross-channel coupling values.

| group | per channel | what it adds |
|---|---|---|
| Evoked potential (N2/P2) | 8 | amplitude, latency and peak-to-peak of the vertex complex |
| Band power | 30 | five bands including **delta and gamma**, relative *and* absolute |
| Shape | 3 | Hjorth mobility, Hjorth complexity, spectral entropy |
| Coupling | (30 total) | band-limited correlation between channel pairs |

Measured contribution of each group alone:

| feature set | columns | balanced |
|---|---|---|
| OLD alpha/beta/theta | 36 | 0.4596 |
| shape only | 12 | 0.4310 |
| coupling only | 30 | 0.4369 |
| **evoked potential only** | **32** | **0.4711** |
| **delta + gamma only** | 48 | **0.4715** |
| everything | 194 | **0.4803** |

**32 evoked-potential columns beat all 36 alpha/beta/theta columns.** So do delta and
gamma alone. The three bands currently wired into the RTL are the weakest option
available.

### The evoked potential is real, and the windows were verified

Measured from the grand-average waveform with **no search window imposed**:

| channel | N2 | P2 | N2–P2 |
|---|---|---|---|
| Fz | −9.92 µV @ 243 ms | +7.74 µV @ 422 ms | 17.66 µV |
| **Cz** | −9.40 µV @ 221 ms | +9.06 µV @ 403 ms | **18.46 µV** |
| C3 | −6.04 µV @ 238 ms | +6.54 µV @ 439 ms | 12.57 µV |
| C4 | −8.06 µV @ 219 ms | +7.73 µV @ 405 ms | 15.80 µV |

Largest at Cz, which is the vertex maximum a laser-evoked potential must show. That
ordering was not imposed — all four channels ran through identical code. Amplitude
also rises with reported pain (Low 16.07 µV → Moderate 17.41 → High 22.78 at Cz).

## The uncomfortable result

**A single number describing the laser setting predicts pain better than all 36
original EEG features.**

| predictor | balanced |
|---|---|
| original EEG features (36 columns) | 0.4596 |
| **laser power alone (1 column, no EEG)** | **0.4758** |
| rich EEG features (194 columns) | 0.4803 |
| rich features + laser power | 0.4946 |

After the rebuild the EEG contributes **+1.88 points on top of** knowing the stimulus
intensity. Before it, it contributed nothing. That gain is the honest measure of what
the brain signal adds.

A related confound worth recording: in ds005280, trials marked `S 32` and `S 33` use
identical laser energies `[3.0, 3.5, 4.0]` but average **4.34 vs 5.89** rating. An
unmodelled experimental manipulation moves the rating by 1.5 points with no change in
stimulus, and no EEG feature can predict it.

## What the task itself can support

Same subjects, same split, EEG features only, laser power never used as a feature:

| task | classes | chance | balanced | above chance |
|---|---|---|---|---|
| **Low vs High (Moderate dropped)** | 2 | 50% | **72.4%** | **+22.4** |
| **Severe or not (rating ≥ 7)** | 2 | 50% | **67.5%** | **+17.5** |
| Laser energy, lowest vs highest | 2 | 50% | 65.6% | +15.6 |
| Low / Moderate / High (current task) | 3 | 33% | 48.0% | +14.7 |
| Any pain (rating ≥ 4) | 2 | 50% | 63.2% | +13.1 |
| Laser energy, 3 levels | 3 | 33% | 43.0% | +9.7 |

Two findings here matter:

**Predicting the laser is harder than predicting the person.** Stimulus intensity
scores below subjective rating on both the 3-class and 2-class comparisons. The EEG
reflects the experience more than the physical stimulus — which is what a pain device
needs to be true.

**90% is not reachable on this task.** The best framing, on the easiest version of
the question, with 44% of trials discarded, reaches 72%. Predicting a stranger's
subjective rating from scalp EEG is an open research problem; the dataset's own
authors published it as a resource for that problem and report no accuracy figures.

## Score prediction

Trained on the 0–10 rating directly (`train_lstm_score.py`) rather than three classes:

| metric | value |
|---|---|
| correlation with true rating | r = 0.403 |
| mean absolute error | 1.80 rating points |
| within-subject correlation | 0.362 |
| **subjects tracked in the right direction** | **93% of 136** |
| training r vs validation r | 0.404 vs 0.403 (no memorization) |

Converted back to classes it scores 0.4856 — a tie with direct classification. Its
value is elsewhere: a number per trial, a per-person trend, and a confidence level.

**The cut points matter more than the model.** The identical score model scores
anywhere from **0.3348 to 0.4856** depending only on where the class boundaries are
drawn. The official 4/7 rule performs worst, because a regressor's predictions bunch
near the middle (spread 1.15 against a real spread of 2.39) and almost everything
lands in Moderate. Isotonic calibration made this worse, not better: shrinkage toward
the mean is the correct response to uncertainty, not a scaling error to undo.

## Implications for the RTL

`RTL/feature_vector_generator.v` line 15 hardcodes `{alpha, beta, theta}`. The
measurements above show that vector is the weakest of the options tested. A revision
should carry, in priority order:

1. **Time-domain evoked-potential values** — N2 amplitude, P2 amplitude, N2–P2
   peak-to-peak, per channel. Highest value per column of anything measured.
2. **Delta (1–4 Hz) and gamma (30–45 Hz)** band power. Gamma stops at 45 Hz so
   50/60 Hz mains interference can never enter.
3. **Absolute power alongside the baseline ratio.** The ratio discards overall
   magnitude, and subject-level magnitude carries real signal.

Note that evoked-potential features are **signed** — an N2 amplitude is negative by
definition — so the existing unsigned power quantizer cannot represent them. See
`fit_uint8_bounds` / `apply_uint8_bounds` in `rich_feature_extraction.py`.

Four electrodes are sufficient. A 14-electrode montage produced no gain and began
overfitting (training/validation gap widened from 0.001 to 0.038).

## Reproducing

Feature archives and checkpoints are gitignored — several exceed GitHub's 100 MB
limit and all regenerate from source. Raw recordings are expected in `data_cache/`.

```bash
# Rich features: 28,452 trials x 194 features, roughly 6 minutes
python scripts/preprocessing/build_rich_pilot.py \
    --datasets ds005284 ds005285 ds005286 ds005289 ds005291 \
               ds005292 ds005293 ds005280 ds005473 \
    --out outputs/rich_full/rich_all9_features.npz

# Train the classifier on them
python scripts/preprocessing/train_lstm.py \
    --feature-mode rich \
    --rich-features outputs/rich_full/rich_all9_features.npz \
    --class-balance-strength 0.5 \
    --split-from-checkpoint outputs/all9_full/all9_pain_lstm_final_seed20260726.pt

# Train the score model instead
python scripts/preprocessing/train_lstm_score.py \
    --rich-features outputs/rich_full/rich_all9_features.npz
```

## Scripts

**Feature pipeline**
- `rich_feature_extraction.py` — the feature computation
- `build_rich_pilot.py` — builds archives from raw EEG
- `build_raw_epochs.py` — saves raw waveforms for CNN experiments

**Models**
- `train_lstm.py` — classifier, `--feature-mode rich`
- `train_lstm_score.py` — 0–10 score prediction
- `train_hybrid.py` — CNN + LSTM, **untested at time of writing**

**Diagnostics that drove the decisions**
- `feature_ceiling_check.py` — showed features, not the model, were the limit
- `erp_validation.py` — verified the evoked potential and its time windows
- `compare_feature_sets.py` — measured old versus new feature sets
- `task_ceiling_check.py` — what each version of the task can support

**Negative results, kept so they are not repeated**
- `normalization_probe.py` — per-subject normalization hurts
- `within_subject_probe.py` — the model largely rides subject base rates
- `single_vs_pooled_check.py` — pooling beats single-dataset training
- `mvar_features.py` — directed connectivity adds nothing here
- `regression_then_bin_check.py`, `score_to_class_check.py` — score-to-class mappings
