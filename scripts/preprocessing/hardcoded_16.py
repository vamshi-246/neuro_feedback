"""Pivot 3: freeze the feature extractor to a fixed short list, and cost it for silicon.

The selection search has been run three times now, over 235 and then 599 candidate
columns, and it keeps returning the same handful: Cz delta power, Cz theta power, the
N2-P2 complex at Cz, and delta coupling across C3-C4. Nothing is left to discover there.
So the extractor should stop computing 599 columns and compute exactly the ones that
earn their place.

Four questions this answers, in order of how much they matter to the hardware
----------------------------------------------------------------------------
  Does the short list cost accuracy?   Scored against the full 599-column pool.

  Does the device need an autoregressive solver?  Two of the sixteen three-class
  columns are MVAR parametric power, which means fitting an AR model per trial --
  by far the most expensive thing on the list. So a no-MVAR list is selected on
  training only and scored beside it. If the gap is noise, the solver comes out.

  Does 8-bit fixed point cost accuracy?  The RTL input path is 8 bits wide. Evoked
  potential columns are signed, which the older unsigned power quantizer could not
  represent, so bounds are fitted per column on training rows and the whole thing
  is rescored through them.

  What arithmetic is actually required?  The manifest lists, per column, which
  channel, which band, which time window and which primitive -- and totals the
  distinct primitives, which is the number that sets area.

Every list here is chosen on training rows only. Validation is scored once per row.

Usage (from repo root):
    python scripts/preprocessing/hardcoded_16.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from feature_extraction import WINDOWS_S
from rich_feature_extraction import (
    BROADBAND_HZ,
    ERP_RMS_WINDOW_S,
    N2_MEAN_WINDOW_S,
    N2_SEARCH_S,
    P2_MEAN_WINDOW_S,
    P2_SEARCH_END_S,
    RICH_BANDS,
    apply_uint8_bounds,
    fit_uint8_bounds,
)
from winning_config import (  # noqa: E402 -- path setup must precede this import
    SEED,
    SELECTED_16,
    dataset_macro,
    family_of,
    fit_tuned,
    load_pool,
    load_split,
    rank_columns,
)

# The eight columns the severe-pain task selected in severe_detection.py. Kept as a
# fallback; the live list is read from that run's JSON when it is present.
SEVERE_8_FALLBACK = (
    "Cz:delta:w0:log_absolute",
    "Cz:erp_erp_rms",
    "Cz:theta:w0:log_absolute",
    "Cz:erp_n2p2_amplitude",
    "C4:delta:w0:log_absolute",
    "Cz:erp_p2_amplitude",
    "C3-C4:delta:coupling",
    "C3:delta:w0:log_absolute",
)

BAND_HZ = {name: (low, high) for name, low, high in RICH_BANDS}


def score(tag, X, y, tr, va, datasets, seed, rows):
    model = fit_tuned(X[tr], y[tr], seed=seed)
    prediction = model.predict(X[va])
    macro, _ = dataset_macro(y[va], prediction, datasets[va])
    pooled = float(balanced_accuracy_score(y[va], prediction))
    plain = float(accuracy_score(y[va], prediction))
    print(f"  {tag:<44}{X.shape[1]:>6}{macro:>11.4f}{pooled:>10.4f}{plain:>9.4f}")
    rows[tag] = {"columns": int(X.shape[1]), "dataset_macro": macro,
                 "pooled_balanced": pooled, "plain_accuracy": plain}
    return macro


def columns_for(names, wanted):
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in wanted if name not in index]
    if missing:
        raise SystemExit(f"columns absent from the pool: {missing}")
    return [index[name] for name in wanted]


def describe(name):
    """Turn a column name into the arithmetic a device has to perform for it."""

    if name.startswith("mvar:parametric_power:"):
        _, _, band, channel = name.split(":")
        return {"channel": channel, "primitive": "AR-model band power", "band": band,
                "band_hz": BAND_HZ.get(band), "window_s": None, "signed": True,
                "hardware": "least-squares AR fit per trial, then spectrum evaluation"}
    if ":coupling" in name:
        pair, band, _ = name.split(":")
        return {"channel": pair, "primitive": "cross-channel correlation", "band": band,
                "band_hz": BAND_HZ.get(band), "window_s": None, "signed": True,
                "hardware": "band-pass both channels, one Pearson correlation"}
    if "_over_" in name:
        channel, window, ratio = name.split(":")
        first, second = ratio.split("_over_")
        return {"channel": channel, "primitive": "log-power difference",
                "band": f"{first}/{second}",
                "band_hz": [BAND_HZ.get(first), BAND_HZ.get(second)],
                "window_s": WINDOWS_S[int(window[1:])], "signed": True,
                "hardware": "subtract two log powers already computed"}
    if ":erp_" in name:
        channel, stat = name.split(":erp_", 1)
        window = {
            "n2_amplitude": N2_SEARCH_S,
            "p2_amplitude": (N2_SEARCH_S[1], P2_SEARCH_END_S),
            "n2p2_amplitude": (N2_SEARCH_S[0], P2_SEARCH_END_S),
            "n2_window_mean": N2_MEAN_WINDOW_S,
            "p2_window_mean": P2_MEAN_WINDOW_S,
            "erp_rms": ERP_RMS_WINDOW_S,
        }.get(stat)
        peak = stat.endswith("_amplitude")
        return {"channel": channel, "primitive": f"evoked potential {stat}",
                "band": "1-30 Hz passband", "band_hz": (1.0, 30.0),
                "window_s": window, "signed": stat != "erp_rms",
                "hardware": "peak search in the window" if peak
                            else "mean or RMS over the window"}
    if ":hjorth_" in name:
        channel, stat = name.split(":hjorth_", 1)
        # Both are variance ratios on the broadband response window: mobility is
        # sd(dx)/sd(x), complexity is mobility(dx)/mobility(x). No spectrum needed.
        return {"channel": channel, "primitive": f"Hjorth {stat}",
                "band": "broadband", "band_hz": list(BROADBAND_HZ),
                "window_s": [WINDOWS_S[0][0], WINDOWS_S[-1][1]], "signed": False,
                "hardware": "two first differences, three variances, two square roots"
                            if stat == "complexity"
                            else "one first difference, two variances, one square root"}
    channel, band, window, kind = name.split(":")
    return {"channel": channel, "primitive": f"band power ({kind})", "band": band,
            "band_hz": BAND_HZ.get(band), "window_s": WINDOWS_S[int(window[1:])],
            "signed": kind == "log_absolute",
            "hardware": "one band power, then log" if kind == "log_absolute"
                        else "one band power divided by the baseline"}


def print_manifest(title, wanted, X_train, columns):
    print(f"\n{title}")
    print(f"  {'#':>3}  {'column':<34}{'channel':<9}{'band':<19}{'window (s)':<13}"
          f"{'sgn':>4}")
    print("  " + "-" * 86)
    entries = []
    low, high = fit_uint8_bounds(X_train[:, columns])
    for position, (name, column) in enumerate(zip(wanted, columns), 1):
        record = describe(name)
        window = "-" if record["window_s"] is None else \
            f"{record['window_s'][0]:.2f}-{record['window_s'][1]:.2f}"
        print(f"  {position:>3}  {name:<34}{record['channel']:<9}"
              f"{str(record['band']):<19}{window:<13}"
              f"{'yes' if record['signed'] else 'no':>4}")
        record.update({
            "name": name, "family": family_of(name),
            "quantizer_low": float(low[position - 1]),
            "quantizer_high": float(high[position - 1]),
        })
        entries.append(record)

    channels = sorted({e["channel"] for e in entries if "-" not in e["channel"]})
    pairs = sorted({e["channel"] for e in entries if "-" in e["channel"]})
    bands = sorted({e["band"] for e in entries if e["primitive"].startswith("band")})
    windows = sorted({tuple(e["window_s"]) for e in entries
                      if e["window_s"] is not None})
    erp = sorted({e["primitive"] for e in entries if "evoked" in e["primitive"]})
    ar = [e["name"] for e in entries if "AR-model" in e["primitive"]]
    signed = [e["name"] for e in entries if e["signed"]]

    print(f"\n  What the extractor has to contain for these {len(entries)} columns:")
    print(f"    channels                {', '.join(channels)}"
          f"   ({len(channels)} of the 4 recorded)")
    print(f"    coupling pairs          {', '.join(pairs) if pairs else 'none'}")
    print(f"    power bands             {', '.join(bands) if bands else 'none'}")
    print(f"    analysis windows        "
          f"{', '.join(f'{a:.2f}-{b:.2f}s' for a, b in windows) if windows else 'none'}")
    print(f"    evoked-potential values {len(erp)} distinct: "
          f"{', '.join(s.replace('evoked potential ', '') for s in erp)}")
    print(f"    AR solver required      {'YES -- ' + ', '.join(ar) if ar else 'no'}")
    print(f"    signed columns          {len(signed)} of {len(entries)}, so the input"
          f" path cannot be unsigned")
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--from-severe-run", default="outputs/pivots/severe_detection.json")
    ap.add_argument("--out", default="outputs/pivots/hardcoded_features.json")
    args = ap.parse_args()

    pool = load_pool()
    X, names = pool["X"], pool["names"]
    datasets, subjects = pool["dataset_id"], pool["subject_id"]
    tr, va = load_split(datasets, subjects)
    labels3 = pool["labels"]
    severe = (pool["ratings"] >= 7.0).astype(np.int64)

    severe_columns = list(SEVERE_8_FALLBACK)
    if os.path.isfile(args.from_severe_run):
        with open(args.from_severe_run, encoding="utf-8") as handle:
            recorded = json.load(handle)
        for key, value in recorded.items():
            if key.startswith("severe") and value.get("selected"):
                severe_columns = list(value["selected"])
    print(f"Pool {X.shape[1]} columns   train {int(tr.sum())}   "
          f"validation {int(va.sum())}")
    print(f"Severe-task list: {len(severe_columns)} columns "
          f"({'from ' + args.from_severe_run if os.path.isfile(args.from_severe_run) else 'fallback'})")

    results = {}

    for task_name, y, wanted in (
        ("Three classes: Low / Moderate / High", labels3, list(SELECTED_16)),
        ("Severe or not (rating >= 7)", severe, severe_columns),
    ):
        print(f"\n{'=' * 88}\n{task_name}\n{'-' * 88}")
        print(f"  {'feature set':<44}{'cols':>6}{'macro':>11}{'pooled':>10}"
              f"{'acc':>9}")
        rows = {}
        score("everything in the pool", X, y, tr, va, datasets, args.seed, rows)

        fixed = columns_for(names, wanted)
        full_list = score(f"the fixed list ({len(wanted)} columns)", X[:, fixed], y, tr,
                          va, datasets, args.seed, rows)

        # The same length of list, chosen from a pool with no MVAR in it. Selected on
        # training rows only, so this is a real alternative rather than a subset.
        allowed = [i for i, name in enumerate(names) if family_of(name) != "MVAR"]
        ranked = rank_columns(X[tr][:, allowed], y[tr], seed=args.seed)
        no_mvar = [allowed[i] for i in ranked[:len(wanted)]]
        without = score(f"no-MVAR list, reselected ({len(wanted)} columns)",
                        X[:, no_mvar], y, tr, va, datasets, args.seed, rows)

        low, high = fit_uint8_bounds(X[tr][:, fixed])
        quantized = apply_uint8_bounds(X[:, fixed], low, high).astype(np.float64)
        score("the fixed list, 8-bit quantized", quantized, y, tr, va, datasets,
              args.seed, rows)

        has_mvar = [n for n in wanted if family_of(n) == "MVAR"]
        if has_mvar:
            print(f"\n  Dropping the AR solver costs "
                  f"{100 * (without - full_list):+.2f} points "
                  f"({len(has_mvar)} MVAR columns replaced). Columns it swapped in:")
            for i in no_mvar:
                if names[i] not in wanted:
                    print(f"    + [{family_of(names[i]):<5}] {names[i]}")
        else:
            print("\n  This list contains no MVAR column, so no AR solver is needed"
                  " for it at all.")

        entries = print_manifest(
            f"  Manifest -- {task_name}", wanted, X[tr], fixed)
        results[task_name] = {"scores": rows, "columns": entries}

    print(f"\n{'=' * 88}")
    print("  Read the 'macro' column. A row within about 1.5 points of another is the\n"
          "  same result: the validation noise floor on this split is around 0.7\n"
          "  points. The point of the exercise is that the short list is not worse,\n"
          "  which is what makes it free to adopt.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\n  Written to {args.out}")


if __name__ == "__main__":
    main()
