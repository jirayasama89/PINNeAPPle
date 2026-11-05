"""Slim cache for remote workers: everything the feature models and spectral models (max_order 8) read, nothing more.
Spectra truncated to 0-8X (801 points, float32, identical values); no waveforms / angle signals (local only).
python make_slim.py <dataset>  -> data_slim/<dataset>_slim.npz"""
import sys, numpy as np, q1lib as L
KEEP = ("path", "state", "level", "level_value", "level_unit", "nominal_hz", "speed_hz", "speed_cv", "tach_ok",
        "rms", "kurtosis", "skewness", "peak", "crest", "labels", "combined", "levels", "inner", "fs", "error", "sha1")
ds = sys.argv[1]
rows = L.load_rows(ds)
slim = []
for r in rows:
    s = {k: r[k] for k in KEEP if k in r}
    if "order_spectrum" in r:
        s["order_spectrum"] = np.asarray(r["order_spectrum"][:801], np.float32)
        s["envelope_spectrum"] = np.asarray(r["envelope_spectrum"][:801], np.float32)
    slim.append(s)
np.savez_compressed(f"data_slim/{ds}_slim.npz", rows=np.array(slim, dtype=object))
print(ds, len(slim))
