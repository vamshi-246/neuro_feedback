"""Score the bit-exact hardware model on the validation subjects.

For every validation trial this computes two things from the same raw EEG:
  float path   the training-time features (float64), 8-bit codes, folded weights
  hardware     hw_golden.golden(): integer filters, threshold quantizer, integer weights
and reports how often the 8-bit codes and the predicted class agree, plus the
dataset-macro balanced accuracy of each.

Usage (repo root):  py scripts/hardware/evaluate_golden.py [--limit N]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hw_common as hc          # noqa: E402
import hw_golden as hg          # noqa: E402
import winning_config as wc     # noqa: E402
from rich_feature_extraction import apply_uint8_bounds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N subjects")
    ap.add_argument("--out", default="outputs/deploy/rtl_golden_validation.json")
    args = ap.parse_args()

    model = hc.load_model()
    names = [c["name"] for c in model["columns"]]
    lo = np.array([c["quantizer_low"] for c in model["columns"]])
    hi = np.array([c["quantizer_high"] for c in model["columns"]])
    W = np.array(model["folded_form"]["W_eff"])
    b = np.array(model["folded_form"]["b_eff"])
    Wq, bq = W * (hi - lo) / 255.0, b + W @ lo
    _, tables = hg.coef_rom()

    pool = wc.load_pool(with_mvar=False)
    _, va = wc.load_split(pool["dataset_id"], pool["subject_id"])
    subjects = sorted({(d, s) for d, s in zip(pool["dataset_id"][va], pool["subject_id"][va])})
    if args.limit:
        subjects = subjects[:args.limit]
    cols = [pool["names"].index(n) for n in names]

    rows = []
    t0 = time.time()
    for n, (ds, sub) in enumerate(subjects):
        data, onset, epochs = hc.load_subject(ds, sub)
        mask = va & (pool["dataset_id"] == ds) & (pool["subject_id"] == sub)
        lookup = {int(e): i for i, e in zip(np.where(mask)[0], pool["epoch_index"][mask])}
        keep = [k for k, e in enumerate(epochs) if int(e) in lookup]
        if not keep:
            continue
        data = data[:, :, keep]
        idx = [lookup[int(epochs[k])] for k in keep]

        f_float = hc.float_features(data, onset)
        q_float = apply_uint8_bounds(f_float, lo, hi).astype(np.int64)
        cls_float = np.argmax(q_float @ Wq.T + bq, axis=1)
        archive_q = apply_uint8_bounds(pool["X"][idx][:, cols], lo, hi).astype(np.int64)

        res = hg.golden(hg.quantize_input(data), onset, model, tables)
        for k, r in enumerate(res):
            rows.append({
                "dataset": ds, "label": int(pool["labels"][idx[k]]),
                "q_float": q_float[k].tolist(), "q_hw": r["q"],
                "q_archive_match": bool((archive_q[k] == q_float[k]).all()),
                "cls_float": int(cls_float[k]), "cls_hw": r["cls"],
            })
        print(f"  [{n + 1}/{len(subjects)}] {ds} {sub}: {len(keep)} trials "
              f"({time.time() - t0:.0f}s)", flush=True)

    y = np.array([r["label"] for r in rows])
    d = np.array([r["dataset"] for r in rows])
    cf = np.array([r["cls_float"] for r in rows])
    ch = np.array([r["cls_hw"] for r in rows])
    qf = np.array([r["q_float"] for r in rows])
    qh = np.array([r["q_hw"] for r in rows])
    diff = np.abs(qf - qh)
    summary = {
        "trials": len(rows),
        "subjects": len(subjects),
        "dataset_macro_float_path": wc.dataset_macro(y, cf, d)[0],
        "dataset_macro_hardware": wc.dataset_macro(y, ch, d)[0],
        "class_agreement": float((cf == ch).mean()),
        "codes_identical_fraction": float((diff == 0).mean()),
        "codes_max_abs_diff": int(diff.max()),
        "codes_off_by_one_fraction": float((diff == 1).mean()),
        "per_feature_mismatch_fraction": (diff != 0).mean(axis=0).round(5).tolist(),
        "float_codes_match_archive": float(np.mean([r["q_archive_match"] for r in rows])),
    }
    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
