"""Test vectors for tb_pain_classifier_top.v, from the bit-exact golden model.

Writes into the simulation directory (default sim/):
  ep_raw.mem     NEP*3000 words {C4, C3, Cz}, 20 bit each, onset at sample 1000
  exp_q.mem      NEP*16 feature codes
  exp_logit.mem  NEP*3 logits (40 bit)
  exp_cls.mem    NEP classes
  exp_ab.mem     NEP*32 values: A0 B0 A1 B1 ... (72 bit)
  vectors.txt    what each epoch is, with the float-model class for reference

Epochs: real validation trials from onset-1000 datasets (chosen to cover all three
predicted classes), one +-500 uV stress epoch that drives the input and filters into
saturation, and one all-zero epoch (every power is zero).

Usage (repo root):  py scripts/hardware/gen_test_vectors.py [--out sim] [--real 8]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hw_common as hc   # noqa: E402
import hw_golden as hg   # noqa: E402

PICKS = [("ds005285", "sub-001"), ("ds005473", "sub-011"), ("ds005293", "sub-043"),
         ("ds005280", "sub-004")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim")
    ap.add_argument("--real", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = hc.load_model()
    _, tables = hg.coef_rom()

    epochs, notes = [], []
    per_subject = max(1, args.real // len(PICKS))
    for ds, sub in PICKS:
        data, onset, ep_idx = hc.load_subject(ds, sub)
        assert onset == 1000
        x = hg.quantize_input(data)
        res = hg.golden(x[:, :, :40], onset, model, tables)
        # take trials of distinct predicted classes first
        chosen, seen = [], set()
        for t, r in enumerate(res):
            if r["cls"] not in seen:
                chosen.append(t)
                seen.add(r["cls"])
        for t in range(len(res)):
            if len(chosen) >= per_subject:
                break
            if t not in chosen:
                chosen.append(t)
        for t in chosen[:per_subject]:
            epochs.append((x[:, :, t], res[t]))
            notes.append(f"{ds} {sub} epoch {int(ep_idx[t])}: class {res[t]['cls']}")

    rng = np.random.default_rng(7)
    stress = rng.standard_normal((3, 3000)) * 200.0
    stress[:, 1100:1300] += 490.0 * np.sign(np.sin(np.arange(200) / 7.0))
    stress = hg.quantize_input(stress[:, :, None])
    epochs.append((stress[:, :, 0], hg.golden(stress, 1000, model, tables)[0]))
    notes.append("stress: 200 uV noise + 490 uV square burst (input/filters saturate)")
    zero = np.zeros((3, 3000, 1), dtype=np.int64)
    epochs.append((zero[:, :, 0], hg.golden(zero, 1000, model, tables)[0]))
    notes.append("all-zero epoch")

    def w(name, lines):
        with open(os.path.join(args.out, name), "w") as f:
            f.write("\n".join(lines) + "\n")

    raw, q, lg, cl, ab = [], [], [], [], []
    for x, r in epochs:
        for n in range(3000):
            word = 0
            for c in range(3):
                word |= (int(x[c, n]) & 0xFFFFF) << (20 * c)
            raw.append(f"{word:015x}")
        q += [f"{v:02x}" for v in r["q"]]
        lg += [hg.tc(v, 40) for v in r["logits"]]
        cl.append(f"{r['cls']:x}")
        for A, B in r["AB"]:
            ab += [hg.tc(A, 72), hg.tc(B, 72)]
    w("ep_raw.mem", raw)
    w("exp_q.mem", q)
    w("exp_logit.mem", lg)
    w("exp_cls.mem", cl)
    w("exp_ab.mem", ab)
    w("vectors.txt", [f"{i}: {n}" for i, n in enumerate(notes)])
    print(f"{len(epochs)} epochs written to {args.out}")
    for i, n in enumerate(notes):
        print(f"  {i}: {n}")


if __name__ == "__main__":
    main()
