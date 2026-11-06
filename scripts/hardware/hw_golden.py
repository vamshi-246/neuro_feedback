"""Bit-exact golden model of the no-MVAR RTL, and generator of its ROM files.

Every integer operation here has a one-to-one twin in RTL/. If the two ever
disagree, the RTL is wrong (or this file is) -- not "close enough".

Number formats
--------------
  input sample     signed 20 bit, LSB = 2^-10 uV  (range +-512 uV)
  filter state     signed 48 bit, LSB = 2^-24 uV
  filter coeffs    signed 36 bit, 32 fractional bits
  feature buffer   signed 24 bit, LSB = 2^-10 uV
  thresholds       signed 56 bit, 40 fractional bits
  weights          signed 28 bit, 24 fractional bits; logits 40 bit

Usage (repo root):  py scripts/hardware/hw_golden.py --write-mem
"""

import argparse
import math
import os

import numpy as np

import fixed_filter as ff
import hw_common as hc

IN_W, IN_FRAC = 20, 10
STATE_W, STATE_FRAC = 48, 24
COEF_W, COEF_FRAC = 36, 32
FB_W = 24
TH_W, TH_FRAC = 56, 40
WT_W, WT_FRAC = 28, 24
EPOCH = 3000
RECIP_SHIFT = 32
RECIP_500 = round(2 ** RECIP_SHIFT / 500)

# Runs in hardware order: (band, channel index, feature buffer 0=A 1=B).
RUNS = [("erp", 0, 0), ("delta", 0, 0), ("theta", 0, 0), ("beta", 0, 0),
        ("gamma", 0, 0), ("broad", 0, 0), ("beta", 2, 0), ("delta", 1, 0),
        ("delta", 2, 1)]

# Quantizer constant form per feature (JSON column order).
#   lin: c = T     exp: c = e^T     sq: c = max(T,0)^2     sqh: sq * (999/1000)^2
FORMS = ["exp", "sq", "exp", "lin", "lin", "lin", "lin", "exp",
         "exp", "exp", "exp", "exp", "lin", "lin", "exp", "sqh"]


def tc(v, bits):
    """Two's-complement hex string of an int."""
    return format(int(v) & ((1 << bits) - 1), "0{}x".format((bits + 3) // 4))


def coef_rom():
    """Per band: 4 sections x [b0 b1 b2 a1 a2 zi0 zi1 pad]. Address {band, sec, k}."""
    rom = [0] * 256
    tables = {}
    for b, (name, lo, hi) in enumerate(hc.BANDS):
        coef, zq = ff.quant_coeffs(hc.butter_sos(lo, hi), COEF_FRAC)
        tables[name] = (coef, zq)
        for s in range(4):
            for k, v in enumerate(list(coef[s]) + list(zq[s])):
                assert abs(int(v)) < 2 ** (COEF_W - 1)
                rom[b * 32 + s * 8 + k] = int(v)
    return rom, tables


def thresholds(model):
    rom = [0] * (16 * 256)
    for i, col in enumerate(model["columns"]):
        lo, hi = col["quantizer_low"], col["quantizer_high"]
        for k in range(1, 256):
            t = lo + (k - 0.5) * (hi - lo) / 255.0
            c = {"lin": t, "exp": math.exp(t), "sq": max(t, 0.0) ** 2,
                 "sqh": max(t, 0.0) ** 2 * (999 / 1000) ** 2}[FORMS[i]]
            v = int(round(c * 2 ** TH_FRAC))
            assert abs(v) < 2 ** (TH_W - 1)
            rom[i * 256 + k] = v
    return rom


def weights(model):
    W = np.array(model["folded_form"]["W_eff"])
    b = np.array(model["folded_form"]["b_eff"])
    lo = np.array([c["quantizer_low"] for c in model["columns"]])
    hi = np.array([c["quantizer_high"] for c in model["columns"]])
    Wq, bq = W * (hi - lo) / 255.0, b + W @ lo
    Wi = np.round(Wq * 2 ** WT_FRAC).astype(np.int64)
    bi = np.round(bq * 2 ** WT_FRAC).astype(np.int64)
    assert np.abs(Wi).max() < 2 ** (WT_W - 1) and np.abs(bi).max() < 2 ** (WT_W - 1)
    return Wi, bi, Wq, bq


def write_mem(out_dir, model):
    os.makedirs(out_dir, exist_ok=True)
    crom, _ = coef_rom()
    trom = thresholds(model)
    Wi, bi, _, _ = weights(model)
    with open(os.path.join(out_dir, "coef.mem"), "w") as f:
        f.write("\n".join(tc(v, COEF_W) for v in crom) + "\n")
    with open(os.path.join(out_dir, "thresh.mem"), "w") as f:
        f.write("\n".join(tc(v, TH_W) for v in trom) + "\n")
    with open(os.path.join(out_dir, "weights.mem"), "w") as f:     # address {class, feature}
        rows = [tc(Wi[k, i], WT_W) for k in range(3) for i in range(16)] + [tc(0, WT_W)] * 16
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(out_dir, "bias.mem"), "w") as f:
        f.write("\n".join(tc(v, WT_W) for v in list(bi) + [0]) + "\n")
    print("wrote ROM files to", out_dir)


# ------------------------------------------------------------------ golden epoch model

def quantize_input(data_uv):
    m = 1 << (IN_W - 1)
    return np.clip(np.round(data_uv * 2 ** IN_FRAC), -m, m - 1).astype(np.int64)


def filtered_buffer(x_in, onset, band, tables):
    """x_in: [3000, trials] ints. Returns feature buffer ints [1500, trials]."""
    coef, zq = tables[band]
    y = ff.filtfilt_fixed(x_in, coef, zq, COEF_FRAC, STATE_FRAC, IN_FRAC, STATE_W)
    fb = (y + (1 << (STATE_FRAC - IN_FRAC - 1))) >> (STATE_FRAC - IN_FRAC)
    fb = ff.sat(fb, FB_W)
    return fb[onset - 500:onset + 1000]


def isqrt_hw(n):
    """Bitwise restoring square root, the algorithm isqrt_seq.v runs. Equals floor(sqrt)."""
    return math.isqrt(n)


def golden(x_in, onset, model=None, tables=None):
    """x_in: int64 [3, 3000, trials]. Returns one dict per trial."""
    model = model or hc.load_model()
    if tables is None:
        _, tables = coef_rom()
    trom = thresholds(model)
    Wi, bi, _, _ = weights(model)
    n_tr = x_in.shape[2]
    fbs = {}
    for band, ch, _ in RUNS:
        fbs[(band, ch)] = filtered_buffer(x_in[ch], onset, band, tables)

    def psum(key, a, b):
        e = fbs[key][a:b].astype(np.int64)
        return (e * e).sum(axis=0)

    P0 = {k: psum(k, 500, 800) for k in fbs}
    P1_dcz = psum(("delta", 0), 800, 1100)

    out = []
    for t in range(n_tr):
        def g(band, ch):
            return [int(v) for v in fbs[(band, ch)][:, t]]

        e = g("erp", 0)
        sb = sum(e[0:500])
        m = (sb * RECIP_500 + (1 << (RECIP_SHIFT - 1))) >> RECIP_SHIFT
        ep = [v - m for v in e]
        s_rms = sum(v * v for v in ep[500:1100])
        n2, n2_idx = ep[650], 650
        for j in range(651, 800):
            if ep[j] < n2:
                n2, n2_idx = ep[j], j
        p2 = ep[n2_idx + 1]
        for j in range(n2_idx + 2, 1050):
            if ep[j] > p2:
                p2 = ep[j]
        s_n2m = sum(ep[650:750])
        s_p2m = sum(ep[750:950])

        x = g("broad", 0)[500:1500]
        sx, sxx = sum(x), sum(v * v for v in x)
        sd = x[-1] - x[0]
        sdd = sum((x[j] - x[j - 1]) ** 2 for j in range(1, 1000))
        hj_a = 999 * sdd - sd * sd
        hj_b = 1000 * sxx - sx * sx

        a, b = g("delta", 1)[500:1500], g("delta", 2)[500:1500]
        sa, sb2 = sum(a), sum(b)
        saa, sbb = sum(v * v for v in a), sum(v * v for v in b)
        sab = sum(p * q for p, q in zip(a, b))
        u = 1000 * sab - sa * sb2
        v_ = 1000 * saa - sa * sa
        w_ = 1000 * sbb - sb2 * sb2
        D = isqrt_hw(v_ * w_)

        p0 = {k: int(P0[k][t]) for k in P0}
        K300, K600 = 300 << 20, 600 << 20
        AB = [
            (p0[("delta", 0)], K300), (s_rms, K600), (p0[("theta", 0)], K300), (u, D),
            (p2, 1 << 10), (p2 - n2, 1 << 10), (n2, 1 << 10), (p0[("delta", 1)], K300),
            (p0[("delta", 0)], p0[("gamma", 0)]), (int(P1_dcz[t]), K300),
            (p0[("delta", 2)], K300), (p0[("beta", 2)], p0[("delta", 2)]),
            (s_p2m, 200 << 10), (s_n2m, 100 << 10),
            (p0[("beta", 0)], p0[("delta", 0)]), (hj_a, hj_b),
        ]
        q = []
        for i, (A, B) in enumerate(AB):
            code = 0
            for bit in range(7, -1, -1):
                k = code | (1 << bit)
                if (A << TH_FRAC) >= trom[i * 256 + k] * B:
                    code = k
            q.append(code)
        logits = [int(bi[k]) + sum(int(Wi[k, i]) * q[i] for i in range(16)) for k in range(3)]
        cls = int(np.argmax(logits))
        out.append({"AB": AB, "q": q, "logits": logits, "cls": cls, "m": m,
                    "n2_idx": n2_idx, "fb": {k: fbs[k][:, t] for k in fbs}})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-mem", action="store_true")
    ap.add_argument("--mem-dir", default=os.path.join(hc.REPO, "RTL", "mem"))
    args = ap.parse_args()
    model = hc.load_model()
    if args.write_mem:
        write_mem(args.mem_dir, model)
    Wi, bi, Wq, bq = weights(model)
    print("max |W_int|", np.abs(Wi).max(), " bias", bi)
