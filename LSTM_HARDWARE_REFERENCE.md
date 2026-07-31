# Hardware Reference: `scripts/preprocessing/train_lstm.py`

## Scope: do not mix the two LSTM models in this repository

This document specifies the model constructed by
`scripts/preprocessing/train_lstm.py`.  It is the **pooled-dataset prototype**
model, not the final saved `ds005285` model under `outputs/ds005285_lstm/`.

| Property | `train_lstm.py` prototype | Saved `ds005285` model |
|---|---:|---:|
| Input at one time step | 3: alpha, beta, theta | 20: 4 channels x 5 bands |
| Sequence length | 3 | 3 |
| Hidden units | 16 | 16 |
| Classes/logits | 3: Low, Moderate, High | 2: low, high stimulus intensity |
| Parameters | 1,395 | 2,466 |
| Saved trained weights | No | Yes |

Therefore, a hardware implementation of this file must use the **3-input,
3-output** specification below.  It must not load the exported weights in
`outputs/ds005285_lstm/lstm_weights/`: those arrays have 20 input columns and
two classifier rows, so they are incompatible with this prototype.

## Exact architecture

The Python class is equivalent to:

```python
nn.LSTM(input_size=3, hidden_size=16, num_layers=1, batch_first=True)
nn.Linear(in_features=16, out_features=3)
```

It is a standard, vanilla LSTM with these settings:

- one layer, unidirectional;
- three time steps and no variable-length packing;
- 16 cell-state values and 16 hidden-state values per trial;
- `bias=True` (the PyTorch default): there are **two** 64-element bias vectors;
- no peephole connections, projection, attention, bidirectionality, dropout,
  layer normalization, recurrent dropout, or residual connection;
- no supplied initial state, so PyTorch starts every trial with
  `h[-1] = 0` and `c[-1] = 0` (16 float32 zeros each);
- only the final hidden state is classified.  Intermediate output states are
  calculated but discarded by `forward()`.

The image of a standard LSTM cell maps directly to this model.  Its left
top sigmoid is the forget gate `f`; its middle sigmoid is the input gate `i`;
its middle tanh is the candidate/cell gate `g`; and its right sigmoid is the
output gate `o`.

## Input contract

`features_uint8` is loaded with shape `[B, 3, 3]`, where `B` is the number of
trials.  Its last dimension is exactly `[alpha, beta, theta]`; time steps are
the three post-stimulus windows in their stored order.

Before the model sees an input byte `q`, the script performs this operation:

```text
x = float32(q) / 255.```

Thus the LSTM receives `float32` values nominally in `[0, 1]`, not raw 8-bit
integers in `[0, 255]`.  A first floating-point hardware draft should implement
that division before the LSTM.  An equivalent later optimization is to retain
the byte input and replace every input-weight row by `W_ih / 255`; do not do
both.

## PyTorch parameter layout

The tensors are float32 and have the following shapes.

| PyTorch state-dict name | Shape | Meaning |
|---|---:|---|
| `lstm.weight_ih_l0` | `[64, 3]` | input-to-gate weights |
| `lstm.weight_hh_l0` | `[64, 16]` | previous-hidden-to-gate weights |
| `lstm.bias_ih_l0` | `[64]` | input-side gate biases |
| `lstm.bias_hh_l0` | `[64]` | recurrent-side gate biases |
| `classifier.weight` | `[3, 16]` | final hidden state to logits |
| `classifier.bias` | `[3]` | classifier bias |

PyTorch stacks gates in **`(i, f, g, o)` order**, where `g` is the cell
candidate sometimes labelled `c_tilde` in diagrams:

| Row range, zero based | Gate |
|---|---|
| `0:16` | input `i` |
| `16:32` | forget `f` |
| `32:48` | candidate `g` |
| `48:64` | output `o` |

For example, `weight_ih_l0[32:48, :]` is `W_ig`, the 16 by 3 input-weight
matrix for `g`.  The exact same row partition applies to `weight_hh_l0` and
both bias vectors.  Do not use the common alternate ordering `(f, i, o, g)`.

Parameter count:

```text
4 * 16 * 3   =   192  input weights
4 * 16 * 16  = 1,024  recurrent weights
4 * 16       =    64  input biases
4 * 16       =    64  recurrent biases
3 * 16       =    48  classifier weights
3            =     3  classifier biases
                         ----
                        1,395 total float32 parameters
```

## Exact inference equations

For one trial at time step `t = 0, 1, 2`, let `x_t` have length 3, and let
`h_(t-1)` and `c_(t-1)` have length 16.  For each gate
`r in {i, f, g, o}`, compute a 16-element preactivation:

```text
a_r,t = W_ih,r x_t + b_ih,r + W_hh,r h_(t-1) + b_hh,r

i_t = sigmoid(a_i,t)
f_t = sigmoid(a_f,t)
g_t = tanh(   a_g,t)
o_t = sigmoid(a_o,t)

c_t = f_t * c_(t-1) + i_t * g_t
h_t = o_t * tanh(c_t)
```

All `*` in the last two equations are elementwise length-16 products.  Matrix
products use rows times column vectors.  Both biases must be included exactly
once.  At inference they may safely be pre-added as
`b_r = b_ih,r + b_hh,r`, but omitting either one will not reproduce PyTorch.

After `t = 2`, the three raw classifier logits are:

```text
z = classifier.weight h_2 + classifier.bias
predicted_class = argmax(z)
```

There is **no softmax layer in the model**.  `CrossEntropyLoss` applies the
needed log-softmax internally only while training.  For inference, `argmax(z)`
and `argmax(softmax(z))` are identical, so a hardware classifier should compare
the three logits directly.

Prototype label meanings are `0 = rating < 4`, `1 = 4 <= rating < 7`, and
`2 = rating >= 7`.

## Arithmetic needed for a first equivalent hardware draft

The source script converts NumPy data to `torch.float32`; on the inspected CPU
environment it runs PyTorch `2.13.0+cpu` with float32 as the default dtype.
The reference-level arithmetic is therefore float32 affine transforms followed
by float32 sigmoid/tanh and float32 elementwise state updates.

At each time step the four gates need:

```text
4 * 16 * (3 + 16) = 1,216 dot-product multiplications
```

Across the three-step sequence this is 3,648 such multiplications, followed by
the elementwise cell/hidden operations and a final `3 * 16 = 48`-multiply
classifier.  A parallel-gate hardware design may form all 64 preactivations
together; a resource-shared design may use one MAC engine and iterate through
gate rows.  Both represent the same model if they preserve weights, bias sums,
gate ordering, float arithmetic, and state-update order.

Bit-identical results across an RTL implementation and PyTorch are not
automatic even when both nominally use IEEE float32: fused multiply-add,
reduction order, and sigmoid/tanh library implementations can create small
differences.  For the first comparison, keep float32 values and avoid
quantizing or rounding gate/state values.  Compare each gate preactivation,
each `c_t`, each `h_t`, and the three final logits with a stated numerical
tolerance before comparing only the final class.

## Training behavior in this specific script

Training is not part of inference hardware, but it determines the weights:

- It uses one full training-split batch per epoch, not a DataLoader or
  mini-batches.
- It uses `AdamW(lr=1e-3, weight_decay=1e-4)` and unweighted cross entropy.
- It runs 30 epochs by default; there is no early stopping, checkpoint, or
  gradient clipping.
- The script does **not** call `torch.manual_seed()` and does **not** save the
  trained `state_dict`.  Re-running it therefore creates different initial
  weights and usually different final weights.

Consequently, `train_lstm.py` alone is not yet a stable golden reference for
hardware equivalence.  Before implementing RTL against this prototype, freeze
one run's float32 input sequence(s), all six parameter arrays, and expected
gate/state/logit traces.  Otherwise there is no particular software model to
which the hardware can be compared.

## Recommended first-draft comparison contract

1. Train once with an explicitly recorded seed, then export the six arrays
   above as raw float32 values.
2. Pick one or more fixed `[3, 3]` uint8 feature sequences and save both the
   bytes and their float32 `/ 255.0` versions.
3. Run a scalar Python reference that uses the equations above and records
   `a_i`, `a_f`, `a_g`, `a_o`, `c_t`, and `h_t` for every time step.
4. Feed exactly the same float inputs and parameters to RTL.  First validate
   all intermediate traces, then the three logits, then `argmax`.
5. Only after that baseline passes should you introduce fixed-point formats,
   LUT/piecewise sigmoid/tanh, saturation, reordered reductions, or folded
   input scaling.  Each is an intentional arithmetic change and needs a new
   error budget against the frozen float32 reference.

If the intended hardware target is the repository's already-trained and
exported ds005285 model rather than this prototype, use its separate contract:
20 input features, 16 hidden units, 2 logits, and 2,466 parameters.  Its
weights and normalization constants already exist under
`outputs/ds005285_lstm/`.
