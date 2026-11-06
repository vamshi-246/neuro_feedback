"""Vectors for the filtfilt_engine unit test: one epoch, golden feature buffers."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw_common as hc, hw_golden as hg
out = sys.argv[1]
data, onset, _ = hc.load_subject("ds005285", "sub-001")
x = hg.quantize_input(data[:, :, :1])
_, tables = hg.coef_rom()
with open(os.path.join(out, "unit_raw.mem"), "w") as f:
    for n in range(3000):
        w = 0
        for c in range(3):
            w |= (int(x[c, n, 0]) & 0xFFFFF) << (20 * c)
        f.write(f"{w:015x}\n")
with open(os.path.join(out, "unit_fb.mem"), "w") as f:
    for b, (name, _, _) in enumerate(hc.BANDS):
        for ch in range(3):
            fb = hg.filtered_buffer(x[ch], onset, name, tables)[:, 0]
            for v in fb:
                f.write(hg.tc(v, 24) + "\n")
print("ok")
