"""Late fusion of two models from their stored per-run class probabilities (no new training).

p = 0.5 p_A + 0.5 p_B, alpha fixed at 0.5 BEFORE computing (no selection on any data). Status: exploratory — the
pair was chosen after seeing the Phase 1 per-band results (RF strong at low speed, spectral models at high speed),
so it is reported as a hypothesis-generating result; a learned gate must be selected on inner validation (Phase 4/5).
Writes results/phase1/<task>/fuse-<A>+<B>/speed_seed<k>.json in the standard format (same partitions, same seeds).
"""
import json
import sys

import numpy as np

from q1lib import RESULTS, metrics

PAIRS = [("rf", "cnn-spec"), ("mlp", "cnn-spec"), ("rf", "osfno-A-nospeed")]
TASKS = ["misalignment", "unbalance", "bearings", "all", "comfaulda7"]


def fuse(task, a, b, seed):
    fa = RESULTS / "phase1" / task / a / f"speed_seed{seed}.json"
    fb = RESULTS / "phase1" / task / b / f"speed_seed{seed}.json"
    if not (fa.exists() and fb.exists()):
        return None
    A, B = json.load(open(fa)), json.load(open(fb))
    assert A["partition_hash"] == B["partition_hash"]
    pb = {r["path"]: r["proba"] for r in B["runs"]}
    n_cls = len(A["classes"])
    runs, y, pred = [], [], []
    for r in A["runs"]:
        p = 0.5 * np.array(r["proba"]) + 0.5 * np.array(pb[r["path"]])
        runs.append({**r, "pred": int(p.argmax()), "proba": [round(float(v), 5) for v in p]})
        y.append(r["y"]); pred.append(int(p.argmax()))
    y, pred = np.array(y), np.array(pred)
    idx = {r["path"]: i for i, r in enumerate(runs)}
    per = {}
    for name, part in A["partitions"].items():
        te = np.array([idx[p] for p in part["test"]])
        per[name] = metrics(y[te], pred[te], n_cls)
    out = {**{k: A[k] for k in ("task", "seed", "protocol", "dataset", "classes", "n", "partition_hash", "partitions")},
           "model": f"fuse-{a}+{b}", "config": {"fusion": "mean of probabilities", "alpha": 0.5, "status": "exploratory, post-hoc pair"},
           "pooled": metrics(y, pred, n_cls), "per_split": per, "split_info": {}, "seconds_total": A["seconds_total"] + B["seconds_total"],
           "runs": runs}
    f = RESULTS / "phase1" / task / f"fuse-{a}+{b}" / f"speed_seed{seed}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f, "w"))
    return out["pooled"]["macro_f1"]


if __name__ == "__main__":
    for t in TASKS:
        for a, b in PAIRS:
            v = [fuse(t, a, b, s) for s in range(5)]
            v = [x for x in v if x is not None]
            if v:
                print(f"{t:12s} fuse-{a}+{b:16s} {np.mean(v):.3f} ({len(v)} seeds)")
