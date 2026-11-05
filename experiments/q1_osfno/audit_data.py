"""Data part of the Phase 0 audit, on the re-extracted caches: population counts, tachometer exclusions, duplicates,
repeated nominal speeds inside a folder, speed ranges per tercile, spectral grid sizes. Writes results/audit_data.json.
"""
import json
from collections import Counter, defaultdict

import numpy as np

import q1lib as L

out = {}
for ds in ("mafaulda", "comfaulda"):
    rows = L.load_rows(ds)
    if not rows:
        continue
    ok = [r for r in rows if "error" not in r]
    d = {"n_rows": len(rows), "n_errors": len(rows) - len(ok),
         "errors": [(r["path"], r["error"]) for r in rows if "error" in r][:20]}
    if ds == "mafaulda":
        bad = [r for r in ok if not r["tach_ok"]]
        d["tach_excluded"] = len(bad)
        d["tach_excluded_by_folder"] = Counter("/".join(r["path"].split("/")[:2]) for r in bad)
        d["tach_excluded_reason"] = Counter(("cv>5%" if r["speed_cv"] > 0.05 else "") +
                                            ("speed_ratio" if not 0.9 < r["speed_hz"] / r["nominal_hz"] < 1.05 else "")
                                            for r in bad)
        d["n_used"] = len(ok) - len(bad)
    by_folder = defaultdict(list)
    for r in ok:
        by_folder[r["path"].rsplit("/", 1)[0]].append(r)
    rep = {}
    for f, rs in by_folder.items():
        noms = Counter(round(r["nominal_hz"], 1) for r in rs)
        dup = {k: v for k, v in noms.items() if v > 1}
        if dup:
            rep[f] = dup
    d["folders_with_repeated_nominal_speed"] = rep
    sha = Counter(r["sha1"] for r in ok)
    d["exact_duplicate_files"] = [[r["path"] for r in ok if r["sha1"] == h] for h, c in sha.items() if c > 1]
    S = np.stack([np.concatenate([r["order_spectrum"][:801].ravel(), r["envelope_spectrum"][:801].ravel()]) for r in ok]).astype(np.float64)
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
    near = []
    for i in range(len(S)):
        c = S[i + 1:] @ S[i]
        for j in np.flatnonzero(c > 0.9999):
            near.append((ok[i]["path"], ok[i + 1 + j]["path"], float(c[j])))
    d["near_duplicate_spectra_cos>0.9999"] = near[:50]
    d["n_near_duplicates"] = len(near)
    d["grid_points_cache"] = int(ok[0]["order_spectrum"].shape[0])
    d["speed_range_hz"] = [float(min(r["speed_hz"] for r in ok)), float(max(r["speed_hz"] for r in ok))]
    d["states"] = Counter(r["state"] for r in ok)
    for task in (["misalignment", "unbalance", "bearings", "all"] if ds == "mafaulda" else ["comfaulda7"]):
        sel, y, paths, sp = L.task_data(rows, task)
        edges = np.quantile(sp, [0, 1 / 3, 2 / 3, 1])
        d[f"task_{task}"] = {"n": int(len(y)), "counts": np.bincount(y).tolist(), "tercile_edges_hz": edges.round(2).tolist()}
    out[ds] = d
json.dump(out, open(L.RESULTS / "audit_data.json", "w"), indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
print(json.dumps(out, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))[:6000])
