"""Compact copy of the MAFAULDA band-envelope order spectra (0-8X, float16) keyed by run path, for remote Phase 6
shards; phase6.py attaches them to the slim rows when the full cache is absent. Run once on the full cache."""
import numpy as np
import q1lib as L

rows = L.load_rows("mafaulda")
assert "env_bands" in rows[0]
paths = np.array([r["path"] for r in rows])
E = np.stack([np.asarray(r["env_bands"])[:, :801, :].astype(np.float16) for r in rows])
np.savez_compressed(L.RESULTS.parent / "data_slim" / "mafaulda_envbands.npz", paths=paths, env_bands=E)
print(E.shape, E.dtype)
