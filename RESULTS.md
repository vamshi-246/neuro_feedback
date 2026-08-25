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

Feature work then closed at **0.4503 dataset-macro / 0.4952 pooled** (16 columns chosen
from a 599-column pool, tuned gradient boosting), and three pivots were run on top of
that configuration rather than on top of a fresh default:

| pivot | what it changes | result |
|---|---|---|
| **severe-pain detection** (rating >= 7) | the question | **0.6738 pooled / 0.6039 dataset-macro**, full coverage |
| **per-user calibration** | what the model knows about the user | **+7.9 points** three-class, **+4.5** severe, from 51 numbers per user |
| **hardcoded 16-column extractor** | the silicon, not the accuracy | 16 columns beat all 599; no AR solver; 8-bit safe |

Details in [After the report: three pivots](#after-the-report-three-pivots). Read the
margin above chance, not the raw number: a two-class task starts at 50%.

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

Same subjects, same split, EEG features only, laser power never used as a feature.
These figures are pooled balanced accuracy from 194 columns at library defaults, before
feature selection and tuning existed; the tuned versions are in
[Pivot 1](#pivot-1--ask-is-this-severe-instead-of-which-of-three-grades) and supersede
them.

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

## After the report: three pivots

The progress report closed the feature avenue. Four model families, 599 candidate
columns and four feature families had all landed in the same 44–49% band, and the
standing configuration — **16 columns selected from 599, tuned gradient boosting,
0.4503 dataset-macro / 0.4952 pooled** — was the ceiling of that approach.

Three pivots were then run *on top of that configuration*, not against a fresh
default. Each one is a different kind of change: pivot 1 changes the question, pivot 2
changes what the model is allowed to know about the user, pivot 3 changes nothing about
accuracy and everything about the silicon.

The three-class configuration reproduces to the digit inside the new scripts
(**0.4503 / 0.4952 / 0.4425**), so every row below is measured from the same anchor.

### Pivot 1 — ask "is this severe?" instead of "which of three grades?"

`severe_detection.py`. Same subjects, same locked split, EEG only, laser power never a
feature. Pool size *and* decision threshold chosen by subject-grouped CV inside the
training subjects; validation scored once per task.

| task | columns | chance | dataset-macro | pooled | plain | AUC | sens | spec | coverage |
|---|---|---|---|---|---|---|---|---|---|
| 3-class Low/Moderate/High (anchor) | 16 | 0.333 | 0.4503 | 0.4952 | 0.4425 | — | — | — | 100% |
| **severe, rating ≥ 7** | **8** | 0.500 | **0.6039** | **0.6738** | **0.6732** | 0.716 | 0.675 | 0.672 | **100%** |
| low vs high, Moderate dropped | 128 | 0.500 | 0.6994 | 0.7415 | 0.7419 | 0.796 | 0.702 | 0.781 | 55% |
| any pain, rating ≥ 4 | 16 | 0.500 | 0.6153 | 0.6483 | 0.6405 | 0.701 | 0.630 | 0.667 | 100% |

Against the untuned 194-column figures in the report, the winning recipe adds **+1.8
points to low-vs-high** (72.4 → 74.2 pooled) and **+1.6 to any-pain** (63.2 → 64.8),
and leaves **severe unchanged** (67.5 → 67.4). So most of the jump from 44% to 67% is
the reframing, not the recipe.

**Two things have to be said plainly about that jump.**

A two-class task starts at 50%, so the honest comparison is the margin above chance:

| task | above chance |
|---|---|
| low vs high (45% of trials discarded) | **+19.9** |
| 3-class Low/Moderate/High | +11.7 |
| any pain ≥ 4 | +11.5 |
| severe ≥ 7 | +10.4 |

Severe detection is **not** a better-informed model than the three-class one — on
dataset-macro it is marginally *less* above chance. What it is, is a far more useful
one: 67% correct with a real 0.675/0.672 sensitivity/specificity pair is a state a
device can act on, where 44% across three grades is not. Low-vs-high scores highest by
a wide margin and is the only row that genuinely reads more information out of the EEG,
but it buys that by **refusing to answer on 45% of trials**, which a device cannot do.

Severe detection is the row worth shipping: full coverage, no abstention, and the
threshold is a single tunable knob. The validation operating curve, for the record:

| cut | sensitivity | specificity | pooled balanced |
|---|---|---|---|
| 0.40 | 0.781 | 0.508 | 0.6445 |
| **0.50** (chosen inside training) | **0.675** | **0.672** | **0.6738** |
| 0.60 | 0.571 | 0.761 | 0.6659 |

One gap this exposed and closed: the pool-size search originally stopped at 64 columns,
and the whole 599-column pool then scored *higher* on validation for the severe task
(0.6310 vs 0.6039). The grid was widened to include 128, 256 and all 599 — and CV,
seeing all of them, still chose **8**. The validation preference was noise. Keeping 8
is the protocol working, not the protocol losing.

### Pivot 2 — a private head per user, over the same frozen 16 features

`personalized_head.py`. This is the largest effect in the project and the only change
that improves the standing three-class task itself.

The body never moves: the same 16 columns, the same standardization, the same tuned
gradient boosting. Only a small linear layer on top is personal, fitted on that user's
first *N* trials with an L2 penalty pulling it back toward the shared layer — which is
the whole mechanism by which ten trials help instead of overfitting. Scored on trial 81
onward for every user, so the *N* columns are comparable and no arm is scored on a trial
it calibrated on.

Shrinkage strength chosen on the 29 training-origin deep users (each with the shared
model refitted without them, so "no calibration" is not flattered), then applied
unchanged to the **12 validation-origin users the shared model never saw**. Those 12
are the result:

| arm | numbers per user | N=0 | N=20 | N=40 | N=60 | N=80 |
|---|---|---|---|---|---|---|
| *3-class* shared model | 0 | 0.5317 | 0.5317 | 0.5317 | 0.5317 | 0.5317 |
| bias only | **3** | 0.5674 | 0.5897 | 0.5887 | 0.5899 | **0.6037** |
| **head only (16→3)** | **51** | 0.5674 | 0.5708 | 0.5969 | **0.6104** | 0.6064 |
| blended (+ shared probabilities) | 60 | 0.5476 | 0.5751 | 0.5963 | 0.6184 | 0.6149 |
| *severe ≥ 7* shared model | 0 | 0.7573 | 0.7573 | 0.7573 | 0.7573 | 0.7573 |
| bias only | **2** | 0.7486 | 0.7725 | 0.7703 | 0.7819 | 0.7818 |
| **head only (16→2)** | **34** | 0.7486 | 0.7759 | 0.7903 | **0.8022** | 0.7958 |
| blended | 38 | 0.7526 | 0.7795 | 0.7896 | 0.7911 | 0.7907 |

**Three-class: +7.9 points** (0.5317 → 0.6104) from 51 numbers and 60 calibration
trials. **Severe: +4.5 points** (0.7573 → 0.8022), so *pivots 1 and 2 together put a
user-calibrated severe-pain detector at 80%.* Severe pays for itself from 20
calibration trials and beats the shared model for 8 of the 12 users; three-class pays
from 10 trials and beats it for 9 of 12. The 29 training-origin users agree (+7.3 and
+3.3), which is what makes the effect believable on a group of twelve.

**The finding that matters for the hardware is the first row of each block.** Retraining
only the output offsets — **3 numbers per user, 2 for the binary task** — captures
+7.2 of the three-class +7.9 and +2.5 of the severe +4.5. Not a weight in the model
moves; the device stores two or three numbers per person and adds them. If the
calibration wizard has to be cheap, that is the version to build.

Note the N=0 column: the linear head alone already scores 0.5674 against the gradient
boosting model's 0.5317 on these deep users. On 16 features, with no personalization at
all, a 51-number linear layer is not worse than a 13,000-parameter tree ensemble.

**The caveat that belongs with these numbers:** they are per-user balanced accuracy on
12 users with ≥120 trials, which is not the same population as the 0.4503 dataset-macro
figure and must not be compared to it directly. The comparison inside the table — same
users, same scored trials, calibration on or off — is the valid one.

Where the 0.5317 baseline comes from, since it sits above the 0.4952 headline while
being the *same model on the same locked split*:

| scored on | balanced |
|---|---|
| dataset-macro, all 5,722 validation trials — the headline | 0.4503 |
| pooled, same trials | 0.4952 |
| per-subject, averaged over all 132 validation subjects | 0.4349 |
| per-subject, the 12 deep subjects, all their trials | 0.5552 |
| per-subject, the 12 deep subjects, trials 81+ — the calibration baseline | 0.5317 |

The third row is the one that settles it. Switching to per-subject averaging on the
*full* validation set gives 0.4349 — **lower** than 0.4503 — so the averaging is not
what inflates the number. The whole gap is which twelve subjects: only subjects with
≥120 trials can be calibrated at all, and every one of them lives in ds005285 (0.5193)
and ds005473 (0.5935), the two datasets the model already handles best, against 0.4527
across the other seven. Chance level is not the explanation either — 11 of the 12 have
all three classes in their scored block, so mean chance is 0.347, not 0.333.

So `0.6104` means "61% on twelve favourable users", not "the model is now 61%". The
`+7.9` and `+4.5` deltas survive this, because they are measured within that same
subsample with calibration as the only thing that changes. Whether the gain holds on
ordinary subjects is untested and needs a lower depth requirement to answer.

### Pivot 3 — hardcode the extractor, and cost it for silicon

`hardcoded_16.py`. No accuracy is expected here; the point is to prove the short list is
free to adopt and to say exactly what the device has to compute.

| feature set | columns | dataset-macro | pooled | plain |
|---|---|---|---|---|
| *3-class* everything in the pool | 599 | 0.4325 | 0.4866 | 0.4479 |
| **the fixed list** | **16** | **0.4503** | 0.4952 | 0.4425 |
| no-MVAR list, reselected on training | 16 | 0.4529 | 0.4836 | 0.4371 |
| the fixed list, 8-bit quantized | 16 | 0.4481 | 0.4973 | 0.4436 |
| *severe* everything in the pool | 599 | 0.6310 | 0.6812 | 0.6830 |
| **the fixed list** | **8** | **0.6039** | 0.6738 | 0.6732 |
| no-MVAR list, reselected on training | 8 | 0.6099 | 0.6718 | 0.6728 |
| the fixed list, 8-bit quantized | 8 | 0.6076 | 0.6778 | 0.6784 |

Three results, all of them permissions rather than gains:

**16 columns beat 599 by 1.8 points**, confirming the earlier finding from the other
direction. Computing more is not merely wasteful here, it is worse.

**The autoregressive solver can come out.** Two of the sixteen three-class columns are
MVAR parametric power, which needs a least-squares AR fit per trial — the most expensive
item on the list by a wide margin. A no-MVAR list of the same length, selected on
training rows only, scores **+0.26 points** — it replaces them with two band ratios and
Hjorth mobility, all of which are subtractions and differences of numbers already
computed. The severe list contains no MVAR column at all.

**8-bit fixed point is safe.** −0.22 points on the three-class task, +0.37 on severe;
both inside noise. Bounds are fitted per column on training rows only, and are written
into `outputs/pivots/hardcoded_features.json` for the RTL to use directly. Fifteen of
the sixteen columns are **signed**, so the unsigned power quantizer cannot be reused.

The manifest for the shippable configuration — severe detection, eight columns:

| # | column | channel | primitive |
|---|---|---|---|
| 1 | `Cz:delta:w0:log_absolute` | Cz | delta power, 0.00–0.30 s, log |
| 2 | `Cz:erp_erp_rms` | Cz | RMS of the 1–30 Hz waveform, 0.00–0.60 s |
| 3 | `Cz:theta:w0:log_absolute` | Cz | theta power, 0.00–0.30 s, log |
| 4 | `Cz:erp_n2p2_amplitude` | Cz | N2–P2 peak-to-peak, 0.15–0.55 s |
| 5 | `C4:delta:w0:log_absolute` | C4 | delta power, 0.00–0.30 s, log |
| 6 | `Cz:erp_p2_amplitude` | Cz | P2 peak, 0.30–0.55 s |
| 7 | `C3-C4:delta:coupling` | C3–C4 | delta-band correlation |
| 8 | `C3:delta:w0:log_absolute` | C3 | delta power, 0.00–0.30 s, log |

What that costs, in full:

- **3 electrodes, not 4.** Neither list uses **Fz** for anything. Cz carries five of the
  eight columns.
- **2 bands: delta (1–4 Hz) and theta (4–8 Hz).** Alpha, beta and gamma appear nowhere
  in the severe list — and the three bands currently in
  `RTL/feature_vector_generator.v` are alpha, beta and theta, of which only theta
  survives.
- **4 time windows**: 0.00–0.30, 0.00–0.60, 0.15–0.55, 0.30–0.55 s.
- **3 evoked-potential values**, one of them a peak search.
- **1 correlation**, one channel pair, one band.
- **No AR solver, no FFT beyond two bands, no coupling matrix.**
- **7 of 8 columns signed.**

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

### The specification, now that the feature list is frozen

`hardcoded_16.py` writes `outputs/pivots/hardcoded_features.json`, which carries every
column's channel, band, time window, signedness and fitted 8-bit bounds. The short
version, for the severe-pain configuration that is the one worth building:

| | requirement |
|---|---|
| electrodes | **Cz, C3, C4. Fz is not used by any selected column.** |
| bands | **delta 1-4 Hz and theta 4-8 Hz only.** Of the three bands currently wired in, only theta survives; alpha and beta appear nowhere. |
| evoked potential | N2-P2 peak-to-peak, P2 peak, and RMS over 0.00-0.60 s, all at Cz |
| windows | 0.00-0.30, 0.00-0.60, 0.15-0.55, 0.30-0.55 s |
| coupling | one pair, one band: C3-C4 delta correlation |
| autoregressive solver | **not needed.** Dropping the two MVAR columns from the three-class list and reselecting costs +0.26 points, and the severe list never selected one. |
| input path | **signed**, 7 of 8 columns. 8-bit quantization costs +0.37 points on severe and -0.22 on three-class -- both inside noise. |
| per-user storage | 51 numbers for a personalized 16->3 head, or **3 numbers** for offsets alone, which captures most of the gain |

Sixteen columns also beat the whole 599-column pool by 1.8 points, so the compact
extractor is not a compromise made for area -- it is the more accurate option.

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

The three pivots read the PAC and MVAR archives and the locked split, and take their
defaults from `winning_config.py`, so they need no arguments:

```bash
python scripts/preprocessing/severe_detection.py    # 15-30 min
python scripts/preprocessing/personalized_head.py   # 10-25 min
python scripts/preprocessing/hardcoded_16.py        # 5-10 min, reads the first one's JSON
```

Each writes its numbers to `outputs/pivots/`.

## Scripts

**The standing configuration and the pivots built on it**
- `winning_config.py` — the 599-column pool, the tuned settings, the locked split and
  the selected 16 columns, in one place so no experiment silently re-baselines
- `severe_detection.py` — pivot 1, the binary reframings of the task
- `personalized_head.py` — pivot 2, a per-user head over the frozen features
- `hardcoded_16.py` — pivot 3, the fixed extractor and its hardware manifest
- `all_families_selection.py` — every feature family competing for the same slots
- `cascade_and_labels_check.py` — two-stage classification and per-subject labels

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
