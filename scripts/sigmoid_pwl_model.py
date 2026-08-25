"""Build and evaluate a fixed-point piecewise-linear sigmoid model.

The model stores only the x >= 0 half of the sigmoid.  For a negative input it
uses sigmoid(-x) = 1 - sigmoid(x), which is the same symmetry that an RTL
implementation can use.  Segment boundaries are distributed by equal areas of
the sigmoid derivative: this makes segments short near zero, where the curve
changes fastest, and long close to saturation.

All stored inputs, slopes, intercepts, and outputs use signed Q4.12 by
default.  A Q4.12 * Q4.12 product is rounded back to Q4.12 before the
intercept is added; this mirrors the usual fixed-point datapath.  The sigmoid
output itself is non-negative, but it is stored in the same signed format.

Examples
--------
Build an eight-segment right-half model and save a coefficient table::

    python scripts/sigmoid_pwl_model.py --segments 8 \
        --export outputs/sigmoid_pwl_8seg.json

Find the smallest model from 2 through 32 segments that reaches a maximum
error of 0.002 (in sigmoid-value units)::

    python scripts/sigmoid_pwl_model.py --search-segments 2 32 \
        --target-max-error 0.002 --plot outputs/sigmoid_pwl_search.png

The default saturation rule is strict: x > +8 maps to 1 and x < -8 maps to
0.  This matches the stated requirement.  A signed 16-bit Q4.12 input has a
representable range of [-8, 7.999755859375], so +8 itself is a boundary used
by the model but cannot occur as a positive input code.  This is expected and
is reported by the script.
"""

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
    """Signed fixed-point format used for the input and PWL datapath."""

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
    slope_raw: int
    intercept_raw: int
    slope: float
    intercept: float


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid for scalar or NumPy inputs."""
    x_array = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x_array)
    positive = x_array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x_array[positive]))
    exp_x = np.exp(x_array[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return float(result) if np.ndim(x) == 0 else result


def round_nearest_away_from_zero(values: np.ndarray | float) -> np.ndarray | int:
    """Round in a hardware-friendly way: ties are rounded away from zero."""
    array = np.asarray(values, dtype=np.float64)
    rounded = np.where(array >= 0, np.floor(array + 0.5), np.ceil(array - 0.5))
    rounded = rounded.astype(np.int64)
    return int(rounded) if np.ndim(values) == 0 else rounded


def rounded_q_product(product_raw: np.ndarray | int, fractional_bits: int) -> np.ndarray:
    """Round an integer Q(.,2F) product back to Q(.,F)."""
    product = np.asarray(product_raw, dtype=np.int64)
    magnitude = np.abs(product)
    rounded = (magnitude + (1 << (fractional_bits - 1))) >> fractional_bits
    return np.where(product < 0, -rounded, rounded)


def derivative_weighted_boundaries(
    segment_count: int, saturation_raw: int, fmt: FixedPointFormat
) -> np.ndarray:
    """Return Q4.12 boundary codes with equal sigmoid-derivative mass."""
    if segment_count < 1:
        raise ValueError("segment_count must be at least one")
    if saturation_raw <= 0:
        raise ValueError("saturation must be positive")

    saturation = saturation_raw / fmt.scale
    top_y = float(sigmoid(saturation))
    # Integral(sigmoid'(x), 0, x) = sigmoid(x) - 0.5.
    target_y = 0.5 + (top_y - 0.5) * np.arange(segment_count + 1) / segment_count
    floating_boundaries = np.log(target_y / (1.0 - target_y))
    floating_boundaries[0] = 0.0
    floating_boundaries[-1] = saturation
    boundaries = round_nearest_away_from_zero(floating_boundaries * fmt.scale)
    boundaries[0] = 0
    boundaries[-1] = saturation_raw

    if np.any(np.diff(boundaries) <= 0):
        raise ValueError(
            "Requested too many segments for the saturation range and Q format; "
            "some quantized boundaries coincide."
        )
    return boundaries.astype(np.int64)


def _fit_one_segment(
    index: int,
    x_start_raw: int,
    x_end_raw: int,
    fmt: FixedPointFormat,
    refine_radius: int,
    pin_zero_to_half: bool,
) -> Segment:
    """Fit then locally refine a quantized line for one positive-x segment."""
    x_raw = np.arange(x_start_raw, x_end_raw + 1, dtype=np.int64)
    x = x_raw.astype(np.float64) / fmt.scale
    desired = sigmoid(x)

    pin_intercept = index == 0 and pin_zero_to_half
    if pin_intercept:
        intercept = 0.5
        slope = float(np.dot(x, desired - intercept) / np.dot(x, x)) if x_end_raw else 0.0
    else:
        slope, intercept = np.linalg.lstsq(
            np.column_stack((x, np.ones_like(x))), desired, rcond=None
        )[0]

    center_slope_raw = int(round_nearest_away_from_zero(slope * fmt.scale))
    center_intercept_raw = int(round_nearest_away_from_zero(intercept * fmt.scale))
    center_slope_raw = int(np.clip(center_slope_raw, fmt.minimum_raw, fmt.maximum_raw))
    center_intercept_raw = int(np.clip(center_intercept_raw, fmt.minimum_raw, fmt.maximum_raw))

    slope_candidates = range(center_slope_raw - refine_radius, center_slope_raw + refine_radius + 1)
    if pin_intercept:
        intercept_candidates: Iterable[int] = (fmt.scale // 2,)
    else:
        intercept_candidates = range(
            center_intercept_raw - refine_radius, center_intercept_raw + refine_radius + 1
        )

    best: tuple[float, float, int, int] | None = None
    for slope_raw in slope_candidates:
        if not fmt.minimum_raw <= slope_raw <= fmt.maximum_raw:
            continue
        product = rounded_q_product(slope_raw * x_raw, fmt.fractional_bits)
        for intercept_raw in intercept_candidates:
            if not fmt.minimum_raw <= intercept_raw <= fmt.maximum_raw:
                continue
            output_raw = np.clip(product + intercept_raw, 0, fmt.scale)
            error = output_raw.astype(np.float64) / fmt.scale - desired
            # Minimize worst error first, then squared error as a tie breaker.
            score = (float(np.max(np.abs(error))), float(np.mean(error * error)), slope_raw, intercept_raw)
            if best is None or score < best:
                best = score

    assert best is not None
    _, _, slope_raw, intercept_raw = best
    return Segment(
        index=index,
        x_start_raw=x_start_raw,
        x_end_raw=x_end_raw,
        slope_raw=slope_raw,
        intercept_raw=intercept_raw,
        slope=slope_raw / fmt.scale,
        intercept=intercept_raw / fmt.scale,
    )


def build_model(
    segment_count: int,
    saturation: float,
    fmt: FixedPointFormat,
    refine_radius: int,
    pin_zero_to_half: bool,
) -> tuple[np.ndarray, list[Segment], int]:
    """Create a derivative-weighted, quantized right-half PWL model."""
    saturation_raw = int(round_nearest_away_from_zero(saturation * fmt.scale))
    boundaries = derivative_weighted_boundaries(segment_count, saturation_raw, fmt)
    segments = [
        _fit_one_segment(
            index=i,
            x_start_raw=int(boundaries[i]),
            x_end_raw=int(boundaries[i + 1]),
            fmt=fmt,
            refine_radius=refine_radius,
            pin_zero_to_half=pin_zero_to_half,
        )
        for i in range(segment_count)
    ]
    return boundaries, segments, saturation_raw


def evaluate_fixed_raw(
    input_raw: np.ndarray,
    boundaries: np.ndarray,
    segments: list[Segment],
    saturation_raw: int,
    fmt: FixedPointFormat,
    saturation_inclusive: bool,
) -> np.ndarray:
    """Evaluate the full mirrored sigmoid using only the stored right-half lines."""
    input_raw = np.asarray(input_raw, dtype=np.int64)
    magnitude = np.abs(input_raw)
    saturated = magnitude >= saturation_raw if saturation_inclusive else magnitude > saturation_raw

    # searchsorted selects the segment on the right at an internal boundary.
    segment_index = np.searchsorted(boundaries[1:], magnitude, side="right")
    segment_index = np.clip(segment_index, 0, len(segments) - 1)
    slopes = np.asarray([segment.slope_raw for segment in segments], dtype=np.int64)
    intercepts = np.asarray([segment.intercept_raw for segment in segments], dtype=np.int64)
    positive_raw = rounded_q_product(slopes[segment_index] * magnitude, fmt.fractional_bits)
    positive_raw = np.clip(positive_raw + intercepts[segment_index], 0, fmt.scale)

    output_raw = np.where(input_raw < 0, fmt.scale - positive_raw, positive_raw)
    if np.any(saturated):
        output_raw = np.where(saturated & (input_raw < 0), 0, output_raw)
        output_raw = np.where(saturated & (input_raw >= 0), fmt.scale, output_raw)
    return output_raw.astype(np.int64)


def evaluate_error(
    boundaries: np.ndarray,
    segments: list[Segment],
    saturation_raw: int,
    fmt: FixedPointFormat,
    saturation_inclusive: bool,
) -> dict[str, float]:
    """Exhaustively compare all signed input codes against mathematical sigmoid."""
    input_raw = np.arange(fmt.minimum_raw, fmt.maximum_raw + 1, dtype=np.int64)
    actual = evaluate_fixed_raw(
        input_raw, boundaries, segments, saturation_raw, fmt, saturation_inclusive
    ).astype(np.float64) / fmt.scale
    reference = sigmoid(input_raw.astype(np.float64) / fmt.scale)
    error = actual - reference
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "rms_error": float(np.sqrt(np.mean(error * error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "max_abs_error_lsb": float(np.max(np.abs(error)) * fmt.scale),
    }


def select_model(
    counts: Iterable[int], args: argparse.Namespace, fmt: FixedPointFormat
) -> tuple[int, np.ndarray, list[Segment], int, dict[str, float], list[dict[str, float]]]:
    """Evaluate candidate segment counts and choose a practical optimum."""
    candidates = []
    for count in counts:
        boundaries, segments, saturation_raw = build_model(
            count, args.saturation, fmt, args.refine_radius, not args.allow_zero_error
        )
        metrics = evaluate_error(
            boundaries, segments, saturation_raw, fmt, args.saturation_inclusive
        )
        candidates.append(
            (count, boundaries, segments, saturation_raw, metrics))

    if args.target_max_error is not None:
        acceptable = [item for item in candidates if item[4]["max_abs_error"] <= args.target_max_error]
        chosen = acceptable[0] if acceptable else min(
            candidates, key=lambda item: item[4]["max_abs_error"]
        )
    else:
        chosen = min(
            candidates,
            key=lambda item: (item[4]["max_abs_error"], item[4]["rms_error"], item[0]),
        )

    summary = [
        {"segment_count": count, **metrics}
        for count, _, _, _, metrics in candidates
    ]
    count, boundaries, segments, saturation_raw, metrics = chosen
    return count, boundaries, segments, saturation_raw, metrics, summary


def export_model(
    path: Path,
    fmt: FixedPointFormat,
    saturation: float,
    saturation_raw: int,
    saturation_inclusive: bool,
    boundaries: np.ndarray,
    segments: list[Segment],
    metrics: dict[str, float],
    search_summary: list[dict[str, float]],
) -> None:
    """Write a hardware-friendly coefficient table and metadata as JSON."""
    payload = {
        "format": {**asdict(fmt), "notation": f"Q{fmt.total_bits - fmt.fractional_bits}.{fmt.fractional_bits}"},
        "input_range": {"minimum": fmt.minimum, "maximum": fmt.maximum},
        "saturation": {
            "magnitude": saturation,
            "magnitude_raw": saturation_raw,
            "inclusive": saturation_inclusive,
            "rule": "abs(x) >= saturation" if saturation_inclusive else "abs(x) > saturation",
        },
        "symmetry": "sigmoid(-x) = 1 - sigmoid(x)",
        "right_half_boundaries_raw": boundaries.tolist(),
        "right_half_boundaries": (boundaries / fmt.scale).tolist(),
        "segments": [asdict(segment) for segment in segments],
        "error_over_all_input_codes": metrics,
        "segment_search": search_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_model(
    path: Path,
    boundaries: np.ndarray,
    segments: list[Segment],
    saturation_raw: int,
    fmt: FixedPointFormat,
    saturation_inclusive: bool,
) -> None:
    """Create a comparison plot; matplotlib is only needed for this option."""
    try:
        # Keep Matplotlib's cache out of a possibly read-only user profile.
        cache_dir = Path(tempfile.gettempdir()) / "sigmoid_pwl_matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib

        # The script is normally run from a terminal or CI, so avoid requiring
        # a local Tk/Qt GUI backend merely to save a PNG.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plot requires matplotlib") from exc

    raw = np.arange(fmt.minimum_raw, fmt.maximum_raw + 1, dtype=np.int64)
    x = raw / fmt.scale
    modeled = evaluate_fixed_raw(raw, boundaries, segments, saturation_raw, fmt, saturation_inclusive)
    modeled = modeled / fmt.scale
    reference = sigmoid(x)
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    top.plot(x, reference, label="mathematical sigmoid", linewidth=2)
    top.plot(x, modeled, label="Q4.12 PWL model", linewidth=1)
    for boundary in boundaries:
        top.axvline(boundary / fmt.scale, color="tab:gray", alpha=0.2, linewidth=0.7)
        top.axvline(-boundary / fmt.scale, color="tab:gray", alpha=0.2, linewidth=0.7)
    top.set_ylabel("sigmoid output")
    top.grid(True, alpha=0.25)
    top.legend()
    bottom.plot(x, modeled - reference, color="tab:red")
    bottom.set_xlabel("pre-activation x (Q4.12)")
    bottom.set_ylabel("model error")
    bottom.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--segments", type=int, default=8, help="right-half segment count (default: 8)")
    group.add_argument(
        "--search-segments",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="evaluate every right-half segment count in this inclusive range",
    )
    parser.add_argument("--saturation", type=float, default=8.0, help="positive saturation magnitude")
    parser.add_argument("--saturation-inclusive", action="store_true", help="use abs(x) >= saturation")
    parser.add_argument(
        "--target-max-error",
        type=float,
        help="when searching, choose the smallest count meeting this error target",
    )
    parser.add_argument(
        "--refine-radius", type=int, default=2, help="coefficient LSB search radius after least-squares fit"
    )
    parser.add_argument(
        "--allow-zero-error",
        action="store_true",
        help="do not force the first segment to produce sigmoid(0) = exactly 0.5",
    )
    parser.add_argument("--export", type=Path, help="write selected model coefficients to JSON")
    parser.add_argument("--plot", type=Path, help="save a comparison plot (requires matplotlib)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.saturation <= 0:
        raise SystemExit("--saturation must be positive")
    if args.refine_radius < 0:
        raise SystemExit("--refine-radius must be non-negative")
    fmt = FixedPointFormat()
    saturation_raw = int(round_nearest_away_from_zero(args.saturation * fmt.scale))
    if saturation_raw > -fmt.minimum_raw:
        raise SystemExit(
            f"Saturation {args.saturation} exceeds the available Q4.12 magnitude of {-fmt.minimum}."
        )

    if args.search_segments:
        start, end = args.search_segments
        if start < 1 or end < start:
            raise SystemExit("--search-segments needs MIN >= 1 and MAX >= MIN")
        counts = range(start, end + 1)
    else:
        counts = (args.segments,)

    count, boundaries, segments, saturation_raw, metrics, search_summary = select_model(counts, args, fmt)
    print(f"Selected right-half segments: {count}")
    print(f"Format: signed Q4.12; input range [{fmt.minimum}, {fmt.maximum}]")
    print(
        f"Saturation: abs(x) {'>=' if args.saturation_inclusive else '>'} {args.saturation} "
        f"(raw {saturation_raw})"
    )
    print(
        "Error over every signed 16-bit input code: "
        f"max={metrics['max_abs_error']:.8f} ({metrics['max_abs_error_lsb']:.3f} LSB), "
        f"RMS={metrics['rms_error']:.8f}"
    )
    print("\nRight-half coefficient table (evaluate m*x + b in Q4.12):")
    print(" idx | x_start..x_end | slope_raw | intercept_raw | slope      | intercept")
    for segment in segments:
        print(
            f" {segment.index:3d} | {segment.x_start_raw:5d}..{segment.x_end_raw:5d} "
            f"| {segment.slope_raw:9d} | {segment.intercept_raw:13d} "
            f"| {segment.slope:10.7f} | {segment.intercept:10.7f}"
        )
    if saturation_raw > fmt.maximum_raw:
        print(
            "\nNote: +saturation is not a representable signed Q4.12 input code; "
            "it remains the mathematical/end-of-table boundary."
        )
    if args.export:
        export_model(
            args.export, fmt, args.saturation, saturation_raw, args.saturation_inclusive,
            boundaries, segments, metrics, search_summary,
        )
        print(f"\nWrote coefficient table: {args.export}")
    if args.plot:
        plot_model(args.plot, boundaries, segments, saturation_raw, fmt, args.saturation_inclusive)
        print(f"Wrote plot: {args.plot}")


if __name__ == "__main__":
    main()
