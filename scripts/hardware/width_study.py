"""Word-width study: fixed-point filtfilt error vs float, per band (one subject).

Usage (repo root):  py scripts/hardware/width_study.py <input fraction bits, e.g. 10>
"""
import sys, time
import numpy as np
from scipy.signal import sosfiltfilt
import hw_common as hc
import fixed_filter as ff

data, onset, _ = hc.load_subject("ds005285", "sub-001")
data = data[:, :, :40]
IN_FRAC = int(sys.argv[1])
xq = np.clip(np.round(data * 2 ** IN_FRAC), -(1<<19), (1<<19)-1).astype(np.int64)
print("max |x| uV", np.abs(data).max())
for cf, fsx, ws in [(32, 24, 48), (32, 28, 52)]:
    t = time.time()
    worst = []
    for name, lo, hi in hc.BANDS:
        sos = hc.butter_sos(lo, hi)
        coef, zq = ff.quant_coeffs(sos, cf)
        ref = sosfiltfilt(sos, data[0], axis=0)
        got = ff.filtfilt_fixed(xq[0], coef, zq, cf, fsx, IN_FRAC, ws) / 2.0 ** fsx
        sl = slice(onset, onset + 300)
        p_ref = np.mean(ref[sl] ** 2, axis=0); p_got = np.mean(got[sl] ** 2, axis=0)
        worst.append((name, np.abs(got - ref)[onset - 500:onset + 1000].max(),
                      np.abs(np.log(p_got / p_ref)).max(), np.abs(ref[onset-500:onset+1000]).max()))
    print(f"cf={cf} fs={fsx} ws={ws}  ({time.time()-t:.0f}s)")
    for w in worst:
        print(f"   {w[0]:6s} max|err| {w[1]:.2e} uV   max|dlnP w0| {w[2]:.2e}   max|y| {w[3]:.1f}")
