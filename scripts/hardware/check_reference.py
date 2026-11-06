"""Prove the float reference reproduces the training archive for a few subjects."""
import sys
import numpy as np
import hw_common as hc
import winning_config as wc

pool = wc.load_pool(with_mvar=False)
names = [c["name"] for c in hc.load_model()["columns"]]
cols = [pool["names"].index(n) for n in names]
subjects = sys.argv[1:] or ["ds005285:sub-001", "ds005286:sub-002"]
for item in subjects:
    ds, sub = item.split(":")
    data, onset, epochs = hc.load_subject(ds, sub)
    ref = hc.float_features(data, onset)
    mask = (pool["dataset_id"] == ds) & (pool["subject_id"] == sub)
    lookup = {int(e): i for i, e in zip(np.where(mask)[0], pool["epoch_index"][mask])}
    rows = [lookup[int(e)] for e in epochs if int(e) in lookup]
    got = ref[[k for k, e in enumerate(epochs) if int(e) in lookup]]
    want = pool["X"][rows][:, cols]
    err = np.abs(got - want).max(axis=0)
    print(f"{item}: {len(rows)} trials, onset {onset}, epoch len {data.shape[1]}, "
          f"max |diff| per feature {np.array2string(err, precision=1)}")
