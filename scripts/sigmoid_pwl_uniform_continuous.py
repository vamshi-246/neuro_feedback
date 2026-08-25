from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FixedPointFormat:
    total_bits: int = 16
    fractional_bits: int = 12

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    @property
    def minimum_raw(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def maximum_raw(self) -> int:
        return (1 << (self.total_bits - 1)) - 1

    @property
    def minimum(self) -> float:
        return self.minimum_raw / self.scale

    @property
    def maximum(self) -> float:
        return self.maximum_raw / self.scale


@dataclass(frozen=True)
class Segment:
    index: int
    x_start_raw: int
    x_end_raw: int
    slope_raw: int       # coefficient format, default Q2.14
    intercept_raw: int   # coefficient format, default Q2.14
    slope: float
    intercept: float


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    a = np.asarray(x, dtype=np.float64)
    y = np.empty_like(a)
    pos = a >= 0
    y[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    ea = np.exp(a[~pos])
    y[~pos] = ea / (1.0 + ea)
    return float(y) if np.ndim(x) == 0 else y


def round_away(values: np.ndarray | float) -> np.ndarray | int:
    a = np.asarray(values, dtype=np.float64)
    r = np.where(a >= 0, np.floor(a + 0.5), np.ceil(a - 0.5)).astype(np.int64)
    return int(r) if np.ndim(values) == 0 else r


def rounded_shift(values: np.ndarray | int, shift: int) -> np.ndarray:
    """Round signed integer values right by shift bits, ties away from zero."""
    v = np.asarray(values, dtype=np.int64)
    if shift == 0:
        return v
    if shift < 0:
        return v << (-shift)
    mag = np.abs(v)
    r = (mag + (1 << (shift - 1))) >> shift
    return np.where(v < 0, -r, r)


def uniform_boundaries(segment_count: int, saturation_raw: int) -> np.ndarray:
    if segment_count < 1:
        raise ValueError("segment_count must be >= 1")
    b = np.rint(np.linspace(0, saturation_raw, segment_count + 1)).astype(np.int64)
    b[0], b[-1] = 0, saturation_raw
    if np.any(np.diff(b) <= 0):
        raise ValueError("Too many segments for the available Q format; boundaries collide.")
    return b


def clip_coeff(raw: int, fmt: FixedPointFormat) -> int:
    return int(np.clip(raw, fmt.minimum_raw, fmt.maximum_raw))


def slope_x_to_output_raw(
    slope_raw: np.ndarray | int,
    x_raw: np.ndarray | int,
    in_fmt: FixedPointFormat,
    coeff_fmt: FixedPointFormat,
) -> np.ndarray:
    # slope Qc.Fc * x Qx.Fx -> product has Fc+Fx fractional bits.
    product = np.asarray(slope_raw, dtype=np.int64) * np.asarray(x_raw, dtype=np.int64)
    return rounded_shift(product, coeff_fmt.fractional_bits)


def intercept_to_output_raw(
    intercept_raw: int,
    in_fmt: FixedPointFormat,
    coeff_fmt: FixedPointFormat,
) -> int:
    return int(rounded_shift(intercept_raw, coeff_fmt.fractional_bits - in_fmt.fractional_bits))


def line_raw(
    slope_raw: np.ndarray | int,
    intercept_raw: np.ndarray | int,
    x_raw: np.ndarray | int,
    in_fmt: FixedPointFormat,
    coeff_fmt: FixedPointFormat,
) -> np.ndarray:
    prod = slope_x_to_output_raw(slope_raw, x_raw, in_fmt, coeff_fmt)
    b = np.asarray(intercept_raw, dtype=np.int64)
    b_out = rounded_shift(b, coeff_fmt.fractional_bits - in_fmt.fractional_bits)
    return prod + b_out


def continuous_intercept_raw(
    y_start_raw: int,
    slope_raw: int,
    x_start_raw: int,
    in_fmt: FixedPointFormat,
    coeff_fmt: FixedPointFormat,
) -> int:
    """Choose b so the *quantized datapath* evaluates to y_start at x_start."""
    px = int(slope_x_to_output_raw(slope_raw, x_start_raw, in_fmt, coeff_fmt))
    b_out = y_start_raw - px
    shift = coeff_fmt.fractional_bits - in_fmt.fractional_bits
    b_raw = b_out << shift
    return clip_coeff(b_raw, coeff_fmt)


def fit_segment(
    index: int,
    x_start_raw: int,
    x_end_raw: int,
    y_start_raw: int,
    in_fmt: FixedPointFormat,
    coeff_fmt: FixedPointFormat,
    refine_radius: int,
    pin_zero: bool,
) -> tuple[Segment, int]:
    x_raw = np.arange(x_start_raw, x_end_raw + 1, dtype=np.int64)
    x = x_raw / in_fmt.scale
    y = sigmoid(x)
    x0 = x_start_raw / in_fmt.scale
    dx = x - x0
    y0 = y_start_raw / in_fmt.scale

    # Continuous line anchored at the left endpoint.
    if np.any(dx != 0):
        slope = float(np.dot(dx, y - y0) / np.dot(dx, dx))
    else:
        slope = 0.0

    center = clip_coeff(int(round_away(slope * coeff_fmt.scale)), coeff_fmt)
    if index == 0 and pin_zero:
        y_start_raw = in_fmt.scale // 2

    best = None
    for sr in range(center - refine_radius, center + refine_radius + 1):
        if not (coeff_fmt.minimum_raw <= sr <= coeff_fmt.maximum_raw):
            continue
        br = continuous_intercept_raw(y_start_raw, sr, x_start_raw, in_fmt, coeff_fmt)
        out = np.clip(line_raw(sr, br, x_raw, in_fmt, coeff_fmt), 0, in_fmt.scale)
        err = out / in_fmt.scale - y
        score = (float(np.max(np.abs(err))), float(np.mean(err * err)), abs(sr - center), sr, br)
        if best is None or score < best:
            best = score

    assert best is not None
    _, _, _, sr, br = best
    seg = Segment(
        index=index,
        x_start_raw=x_start_raw,
        x_end_raw=x_end_raw,
        slope_raw=int(sr),
        intercept_raw=int(br),
        slope=int(sr) / coeff_fmt.scale,
        intercept=int(br) / coeff_fmt.scale,
    )
    y_end_raw = int(np.clip(line_raw(sr, br, x_end_raw, in_fmt, coeff_fmt), 0, in_fmt.scale))
    return seg, y_end_raw


def build_model(
    segment_count: int,
    saturation: float,
    in_fmt: FixedPointFormat,
    coeff_fmt: FixedPointFormat,
    refine_radius: int,
    pin_zero: bool,
):
    saturation_raw = int(round_away(saturation * in_fmt.scale))
    boundaries = uniform_boundaries(segment_count, saturation_raw)

    segments: list[Segment] = []
    y_start_raw = in_fmt.scale // 2

    for i in range(segment_count):
        seg, y_start_raw = fit_segment(
            i,
            int(boundaries[i]),
            int(boundaries[i + 1]),
            y_start_raw,
            in_fmt,
            coeff_fmt,
            refine_radius,
            pin_zero,
        )
        segments.append(seg)

    return boundaries, segments, saturation_raw


def evaluate_fixed_raw(
    input_raw: np.ndarray,
    boundaries: np.ndarray,
    segments: list[Segment],
    saturation_raw: int,
    in_fmt: FixedPointFormat,
    coeff_fmt: FixedPointFormat,
    saturation_inclusive: bool,
) -> np.ndarray:
    input_raw = np.asarray(input_raw, dtype=np.int64)
    # Keep magnitude wide enough for abs(-32768).
    mag = np.abs(input_raw)
    saturated = mag >= saturation_raw if saturation_inclusive else mag > saturation_raw

    idx = np.searchsorted(boundaries[1:], mag, side="right")
    idx = np.clip(idx, 0, len(segments) - 1)
    slopes = np.array([s.slope_raw for s in segments], dtype=np.int64)[idx]
    ints = np.array([s.intercept_raw for s in segments], dtype=np.int64)[idx]

    pos = np.clip(line_raw(slopes, ints, mag, in_fmt, coeff_fmt), 0, in_fmt.scale)
    out = np.where(input_raw < 0, in_fmt.scale - pos, pos)

    if np.any(saturated):
        out = np.where(saturated & (input_raw < 0), 0, out)
        out = np.where(saturated & (input_raw >= 0), in_fmt.scale, out)
    return out.astype(np.int64)


def evaluate_error(boundaries, segments, saturation_raw, in_fmt, coeff_fmt, saturation_inclusive):
    raw = np.arange(in_fmt.minimum_raw, in_fmt.maximum_raw + 1, dtype=np.int64)
    actual = evaluate_fixed_raw(raw, boundaries, segments, saturation_raw, in_fmt, coeff_fmt, saturation_inclusive) / in_fmt.scale
    ref = sigmoid(raw / in_fmt.scale)
    err = actual - ref
    return {
        "max_abs_error": float(np.max(np.abs(err))),
        "rms_error": float(np.sqrt(np.mean(err * err))),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "max_abs_error_lsb": float(np.max(np.abs(err)) * in_fmt.scale),
    }


def select_model(counts: Iterable[int], args, in_fmt, coeff_fmt):
    candidates = []
    for n in counts:
        b, s, sr = build_model(n, args.saturation, in_fmt, coeff_fmt, args.refine_radius, not args.allow_zero_error)
        m = evaluate_error(b, s, sr, in_fmt, coeff_fmt, args.saturation_inclusive)
        candidates.append((n, b, s, sr, m))

    target_met = True
    if args.target_max_error is not None:
        ok = [c for c in candidates if c[4]["max_abs_error"] <= args.target_max_error]
        if ok:
            chosen = ok[0]  # smallest count meeting target
        else:
            chosen = min(candidates, key=lambda c: (c[4]["max_abs_error"], c[4]["rms_error"], c[0]))
            target_met = False
    else:
        chosen = min(candidates, key=lambda c: (c[4]["max_abs_error"], c[4]["rms_error"], c[0]))

    summary = [{"segment_count": n, **m} for n, _, _, _, m in candidates]
    n, b, s, sr, m = chosen
    if args.target_max_error is not None:
        m = {**m, "target_max_error": args.target_max_error, "target_met": target_met}
    return n, b, s, sr, m, summary


def export_model(path, in_fmt, coeff_fmt, saturation, saturation_raw, saturation_inclusive, boundaries, segments, metrics, search_summary):
    payload = {
        "input_output_format": {**asdict(in_fmt), "notation": f"Q{in_fmt.total_bits - in_fmt.fractional_bits}.{in_fmt.fractional_bits}"},
        "coefficient_format": {**asdict(coeff_fmt), "notation": f"Q{coeff_fmt.total_bits - coeff_fmt.fractional_bits}.{coeff_fmt.fractional_bits}"},
        "segmentation": {"type": "uniform_x", "continuity": "exact_at_quantized_boundaries"},
        "input_range": {"minimum": in_fmt.minimum, "maximum": in_fmt.maximum},
        "saturation": {
            "magnitude": saturation,
            "magnitude_raw": saturation_raw,
            "inclusive": saturation_inclusive,
            "rule": "abs(x) >= saturation" if saturation_inclusive else "abs(x) > saturation",
        },
        "symmetry": "sigmoid(-x) = 1 - sigmoid(x)",
        "right_half_boundaries_raw": boundaries.tolist(),
        "right_half_boundaries": (boundaries / in_fmt.scale).tolist(),
        "segments": [asdict(s) for s in segments],
        "error_over_all_input_codes": metrics,
        "segment_search": search_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_model(path, boundaries, segments, saturation_raw, in_fmt, coeff_fmt, saturation_inclusive):
    try:
        cache = Path(tempfile.gettempdir()) / "sigmoid_pwl_matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plot requires matplotlib") from exc

    raw = np.arange(in_fmt.minimum_raw, in_fmt.maximum_raw + 1, dtype=np.int64)
    x = raw / in_fmt.scale
    model = evaluate_fixed_raw(raw, boundaries, segments, saturation_raw, in_fmt, coeff_fmt, saturation_inclusive) / in_fmt.scale
    ref = sigmoid(x)

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    top.plot(x, ref, label="mathematical sigmoid", linewidth=2)
    top.plot(x, model, label="uniform continuous PWL", linewidth=1)
    for b in boundaries:
        top.axvline(b / in_fmt.scale, color="tab:gray", alpha=0.2, linewidth=0.7)
        top.axvline(-b / in_fmt.scale, color="tab:gray", alpha=0.2, linewidth=0.7)
    top.set_ylabel("sigmoid output")
    top.grid(True, alpha=0.25)
    top.legend()

    bottom.plot(x, model - ref, color="tab:red")
    bottom.set_xlabel("pre-activation x (Q4.12)")
    bottom.set_ylabel("model error")
    bottom.grid(True, alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Uniform continuous fixed-point PWL sigmoid")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--segments", type=int, default=32, help="right-half segment count (default: 32)")
    g.add_argument("--search-segments", type=int, nargs=2, metavar=("MIN", "MAX"), help="search an inclusive segment-count range")
    p.add_argument("--saturation", type=float, default=8.0)
    p.add_argument("--saturation-inclusive", action="store_true")
    p.add_argument("--target-max-error", type=float)
    p.add_argument("--refine-radius", type=int, default=4, help="coefficient LSB search radius")
    p.add_argument("--coeff-frac-bits", type=int, default=14, help="coefficient fractional bits; 14 => signed Q2.14")
    p.add_argument("--allow-zero-error", action="store_true", help="do not pin sigmoid(0)=0.5 exactly")
    p.add_argument("--export", type=Path)
    p.add_argument("--plot", type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    if args.saturation <= 0:
        raise SystemExit("--saturation must be positive")
    if args.refine_radius < 0:
        raise SystemExit("--refine-radius must be non-negative")
    if not 12 <= args.coeff_frac_bits <= 15:
        raise SystemExit("--coeff-frac-bits must be between 12 and 15")

    in_fmt = FixedPointFormat(16, 12)             # Q4.12 input/output
    coeff_fmt = FixedPointFormat(16, args.coeff_frac_bits)  # default Q2.14
    saturation_raw = int(round_away(args.saturation * in_fmt.scale))
    if saturation_raw > -in_fmt.minimum_raw:
        raise SystemExit(f"Saturation {args.saturation} exceeds Q4.12 magnitude of {-in_fmt.minimum}.")

    if args.search_segments:
        lo, hi = args.search_segments
        if lo < 1 or hi < lo:
            raise SystemExit("--search-segments needs MIN >= 1 and MAX >= MIN")
        counts = range(lo, hi + 1)
    else:
        counts = (args.segments,)

    n, boundaries, segments, saturation_raw, metrics, summary = select_model(counts, args, in_fmt, coeff_fmt)

    print(f"Selected right-half segments: {n}")
    print(f"Input/output: signed Q4.12; range [{in_fmt.minimum}, {in_fmt.maximum}]")
    print(f"Coefficients: signed Q{16 - args.coeff_frac_bits}.{args.coeff_frac_bits}")
    print("Segmentation: uniform in x")
    print("Continuity: exact at quantized segment boundaries")
    print(f"Saturation: abs(x) {'>=' if args.saturation_inclusive else '>'} {args.saturation} (raw {saturation_raw})")
    print(f"Error over every signed 16-bit input code: max={metrics['max_abs_error']:.8f} ({metrics['max_abs_error_lsb']:.3f} LSB), RMS={metrics['rms_error']:.8f}, mean_abs={metrics['mean_abs_error']:.8f}")

    if args.target_max_error is not None:
        print(f"Target {args.target_max_error:.8f}: {'MET' if metrics.get('target_met') else 'NOT MET; best candidate selected'}")

    print("\nRight-half coefficient table:")
    print(" idx | x_start..x_end | slope_raw | intercept_raw | slope      | intercept")
    for s in segments:
        print(f" {s.index:3d} | {s.x_start_raw:5d}..{s.x_end_raw:5d} | {s.slope_raw:9d} | {s.intercept_raw:13d} | {s.slope:10.7f} | {s.intercept:10.7f}")

    if saturation_raw > in_fmt.maximum_raw:
        print("\nNote: +saturation is not a representable signed Q4.12 input code; it is only the end-of-table boundary.")

    if args.export:
        export_model(args.export, in_fmt, coeff_fmt, args.saturation, saturation_raw, args.saturation_inclusive, boundaries, segments, metrics, summary)
        print(f"\nWrote coefficient table: {args.export}")

    if args.plot:
        plot_model(args.plot, boundaries, segments, saturation_raw, in_fmt, coeff_fmt, args.saturation_inclusive)
        print(f"Wrote plot: {args.plot}")


if __name__ == "__main__":
    main()
