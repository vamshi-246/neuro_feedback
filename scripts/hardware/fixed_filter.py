"""Bit-exact integer model of the hardware filtfilt engine (numpy int64, vectorised)."""
import numpy as np
from scipy.signal import sosfreqz, sosfilt_zi

PADLEN = 27


def balance(sos):
    """Spread the cascade gain so every partial cascade peaks at 1. Same filter."""
    sos = sos.copy()
    w = np.linspace(0, np.pi, 8192)
    cum = np.ones_like(w, dtype=complex)
    prev = 1.0
    for k in range(len(sos)):
        _, h = sosfreqz(sos[k:k + 1], worN=w)
        cum = cum * h
        s = 1.0 / np.abs(cum).max() if k < len(sos) - 1 else 1.0
        sos[k, :3] *= s / prev
        prev = s
    return sos


def quant_coeffs(sos, cf):
    """Integer coefficients: per section [b0,b1,b2,a1,a2] and zi [2], scaled 2^cf."""
    s = balance(sos)
    zi = sosfilt_zi(s)
    coef = np.round(np.concatenate([s[:, :3], s[:, 4:6]], axis=1) * 2.0 ** cf).astype(np.int64)
    zq = np.round(zi * 2.0 ** cf).astype(np.int64)
    return coef, zq


def mul_round(a, c, cf):
    """floor((a*c + 2^(cf-1)) / 2^cf) exactly, for |a| < 2^55 and |c| < 2^36 (cf >= 24)."""
    ah = a >> 24
    al = a & ((1 << 24) - 1)
    hi = ah * c                       # |a| < 2^48 here, so |hi| < 2^59
    lo = al * c + (1 << (cf - 1))     # < 2^60
    return (hi + (lo >> 24)) >> (cf - 24)


def mul_sum_round(pairs, cf):
    """floor((sum a_i*c_i + 2^(cf-1)) / 2^cf) exactly."""
    hi = 0
    lo = 1 << (cf - 1)
    for a, c in pairs:
        hi = hi + (a >> 24) * c
        lo = lo + (a & ((1 << 24) - 1)) * c
    return (hi + (lo >> 24)) >> (cf - 24)


def sat(v, bits):
    m = (1 << (bits - 1))
    return np.clip(v, -m, m - 1)


def sos_pass(x, coef, zq, cf, ws, x0):
    """One DF2T cascade pass over x[n, trials] (state units). x0: value scaling zi."""
    n_sec = coef.shape[0]
    z = [[sat(mul_round(x0, zq[k, 0], cf), ws), sat(mul_round(x0, zq[k, 1], cf), ws)]
         for k in range(n_sec)]
    out = np.empty_like(x)
    for n in range(x.shape[0]):
        v = x[n]
        for k in range(n_sec):
            b0, b1, b2, a1, a2 = (int(c) for c in coef[k])
            y = sat(z[k][0] + mul_round(v, b0, cf), ws)
            z0 = sat(z[k][1] + mul_sum_round([(v, b1), (y, -a1)], cf), ws)
            z1 = sat(mul_sum_round([(v, b2), (y, -a2)], cf), ws)
            z[k] = [z0, z1]
            v = y
        out[n] = v
    return out


def filtfilt_fixed(x_in, coef, zq, cf, fs_state, in_frac, ws):
    """x_in: int64 [samples, trials] in input LSBs. Returns state-unit output [samples]."""
    x = x_in.astype(np.int64)
    n = x.shape[0]
    pre = 2 * x[0] - x[PADLEN:0:-1]
    post = 2 * x[-1] - x[-2:-(PADLEN + 2):-1]
    ext = np.concatenate([pre, x, post], axis=0) << (fs_state - in_frac)
    fwd = sos_pass(ext, coef, zq, cf, ws, ext[0])
    bwd = sos_pass(fwd[::-1], coef, zq, cf, ws, fwd[-1])[::-1]
    return bwd[PADLEN:PADLEN + n]
