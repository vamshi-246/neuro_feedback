# Sigmoid Cell Implementation Report

## Purpose and scope

This work implements the sigmoid activation cell needed by the LSTM portion of
the neurofeedback hardware project. The present implementation covers the
software approximation model, fixed-point coefficient generation, and a
combinational Verilog realization. It does not implement the surrounding LSTM
cell or the final system-level datapath.

The central design objective was to avoid an expensive exponential operation in
hardware. Instead, the sigmoid is approximated by a set of linear functions,
one selected according to the input interval.

## Mathematical basis

The sigmoid function is

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

Its symmetry makes it unnecessary to store two independent sets of linear
segments:

```text
sigmoid(-x) = 1 - sigmoid(x)
```

Therefore, the coefficient table covers only the right half of the curve
(`x >= 0`). A negative input is converted to a positive magnitude, evaluated
by the right-half table, and mirrored around 0.5.

## Fixed-point representation and saturation

The selected formats are:

| Quantity | Format | Notes |
|---|---|---|
| Sigmoid input | signed Q4.12, 16 bits | Pre-activation input to the LSTM gate |
| Sigmoid output | Q4.12, 16 bits | Non-negative output in the range 0 to 1 |
| Slope and intercept coefficients | signed Q2.14, 16 bits | Extra fractional precision without increasing coefficient storage width |

For signed Q4.12, the representable input range is `[-8.0,
7.999755859375]`. The current saturation boundary is parameterized as a raw
magnitude of 32768, corresponding to 8.0. The selected policy is strict:

```text
x >  8  -> 1
x < -8  -> 0
```

Because `+8.0` is not representable as a signed 16-bit Q4.12 input, it is the
end-of-table boundary rather than a positive input code. `-8.0` is representable
and is evaluated by the last piecewise-linear segment under the strict policy.

## Software-model development

Two Python models were developed and compared.

### Initial derivative-weighted model

`scripts/sigmoid_pwl_model.py` distributes right-half segment boundaries using
equal mass of the sigmoid derivative. This initially followed the idea of using
more segments near the steep central part of the curve. It fits each segment,
quantizes its coefficients, mirrors the negative half, and evaluates error over
all 65,536 signed input codes.

This approach correctly models fixed-point arithmetic, but it leaves a very
wide final segment near saturation. Piecewise-linear approximation error is
driven more directly by curve curvature than by slope alone, so the large final
segment limits maximum-error performance.

### Selected uniform, continuous model

`scripts/sigmoid_pwl_uniform_continuous.py` is the selected software model.
It makes the following improvements:

- Splits the positive input range `[0, 8]` uniformly in `x`.
- Fits a line for each interval and searches nearby quantized slope values.
- Uses Q2.14 coefficients while retaining Q4.12 input and output values.
- Computes each following intercept from the previous segment's quantized end
  value. This guarantees continuity at every quantized segment boundary.
- Uses ties-away-from-zero rounding, output clipping, symmetry reconstruction,
  and the same saturation policy intended for RTL.
- Exhaustively compares the fixed-point approximation against the mathematical
  sigmoid for every signed 16-bit input code.

At an exact internal boundary, the evaluator selects the segment on the right.
The consecutive segments produce the same quantized output at that point, so
the selection rule does not create a discontinuity.

## Approximation results

The uniform, continuous approach was materially more accurate than the initial
derivative-weighted implementation under matched small-model settings.

| Configuration | Maximum absolute error | RMS error |
|---|---:|---:|
| Derivative-weighted, 8 segments, Q4.12 coefficients | `0.02567983` | `0.00717039` |
| Uniform continuous, 8 segments, Q4.12 coefficients | `0.00770758` | `0.00189260` |
| Uniform continuous, 32 segments, Q2.14 coefficients | `0.00057717` | `0.00015334` |
| **Selected: uniform continuous, 40 segments, Q2.14 coefficients** | **`0.00039040`** | **`0.00011615`** |

The selected 40-segment model has a maximum error of approximately `1.599`
Q4.12 output LSB and a mean absolute error of approximately `0.00009259`.
The generated source-of-truth coefficient table is
`outputs/hardware/sigmoid_uniform_40seg.json`.

## RTL implementation

The hardware implementation is in `RTL/sigmoid_cell.v`.

### Interface and operation

`sigmoid_cell` is a purely combinational module:

```verilog
input  wire signed [15:0] x_in;   // Q4.12 pre-activation
output reg        [15:0] y_out;  // Q4.12 sigmoid result, 0 to 1
```

It has no clock, reset, or registered latency. The module contains parameters
for data/coefficient formats and saturation behavior. The embedded 40-segment
table itself is generated for the default Q4.12/Q2.14 configuration; changing
a table-related parameter requires regenerating the coefficients and decision
thresholds from the Python model.

### Hardware datapath

The RTL implements the following steps:

1. Form a widened, non-negative magnitude of the signed input. The extra bit
   preserves `abs(-32768) = 32768` safely.
2. Select one of 40 uniform right-half intervals. The 39 threshold values and
   all coefficient pairs are embedded as constants from the selected JSON file.
3. Multiply the Q2.14 slope by the Q4.12 magnitude, creating a Q6.26 product.
4. Round that product back to Q4.12 with the same ties-away-from-zero rule used
   in the software model.
5. Convert and add the Q2.14 intercept, then clip the positive result to the
   valid sigmoid range `[0, 1]` (`0` to raw `4096`).
6. For a negative original input, return `4096 - positive_result`; otherwise
   return the positive result.
7. Apply the parameterized saturation rule before the final mirrored output.

One high-range quantized segment has a small negative fitted slope. This is
retained intentionally because it is part of the quantized continuous fit and
the final output clamp preserves the valid sigmoid range.

## Verification performed

- The software model evaluates all 65,536 input codes, not just sampled points.
- The 40 RTL slope/intercept pairs were programmatically compared against
  `sigmoid_uniform_40seg.json`; all 40 pairs matched exactly.
- The 39 RTL segment-decision thresholds were also compared against the JSON
  boundaries; all thresholds matched exactly.
- RTL syntax/simulation was not run in this workspace because no Verilog
  simulator was installed. A next verification step is a self-checking testbench
  that applies all 65,536 input patterns and compares the RTL output with the
  Python fixed-point model.

## Tanh cell implementation direction

We are implementing the tanh cell through the derivation of sigmoid cell
implementation using the relationship between them and adjusting all the
intermediate hardware inclusions.

The relationship is:

```text
tanh(x) = 2 * sigmoid(2x) - 1
```

The existing sigmoid datapath can therefore be reused as the core of the tanh
cell. The tanh implementation will require these intermediate hardware changes:

1. Pre-scale the tanh input by two before it enters the sigmoid approximation.
   This must use a widened signed intermediate so the shift cannot overflow.
2. Apply sigmoid saturation in the scaled domain. With the current sigmoid
   threshold of `|u| = 8`, where `u = 2x`, tanh saturation occurs at
   `|x| = 4`.
3. Transform the sigmoid result using a widened signed intermediate:

   ```text
   tanh_raw = (2 * sigmoid_raw) - 4096
   ```

   Here `sigmoid_raw` is Q4.12. The intermediate needs an extra signed bit
   before it is clipped and returned as a signed Q4.12 tanh value in `[-1, 1]`.
4. Preserve correct signed rounding, clipping, saturation, and boundary handling
   throughout the pre-scale and post-scale operations.

This approach avoids a separate tanh coefficient table and keeps the sigmoid
cell as the reusable nonlinear-function primitive for the LSTM hardware.

## Recommended next steps

1. Create a self-checking Verilog/SystemVerilog testbench for `sigmoid_cell`.
2. Run exhaustive RTL-versus-Python comparison once an HDL simulator is
   available.
3. Measure synthesis area and timing, especially the 40-way selector and the
   multiplier, before choosing a final segment count.
4. Implement and verify the tanh wrapper around the sigmoid datapath using the
   derived fixed-point scaling described above.
