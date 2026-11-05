"""Phase 8 (part 2): robustness of trained models to test-time measurement errors (no change to training).

For every outer speed-tercile fold (same partitions as Phase 1) and seed 0-4, the model is trained ONCE on the clean
training runs (identical to Phase 1) and evaluated on the held-out runs under each perturbation below, applied to the
test runs only. Perturbed test copies are appended to the row list and only their indices are passed as `te`; every
normalisation in the models uses the training indices only, so the copies cannot influence training.

Perturbations (documented physical meaning):
  speed error e in {-5, -2, -1, +1, +2, +5} %  the speed estimate is f(1+e): a component at true order k appears at
        k/(1+e), i.e. S'(o) = S(o(1+e)) (linear interpolation on the 0.01-order grid) for the amplitude and envelope
        order spectra, and the speed given to the model is f(1+e). Time-domain statistics are unchanged.
  dead sensor c       channel c replaced by a noise floor: spectra set to 1e-3 x the median of that channel's own
        spectrum, rms x 0.01, kurtosis 3 (Gaussian, Pearson), skewness 0, crest 4.5, peak = 4.5 rms.
  gain x2 on sensor c sensitivity drift: spectra, rms and peak of channel c multiplied by 2 (shape statistics unchanged).
Writes results/phase8_robust/<task>/<model>/seed<k>.json: macro-F1 / accuracy per condition (pooled over the 3 folds)
and the predicted class of every test run under every condition.
usage: python phase8_robust.py <dataset> <task> <model> [seeds]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import q1lib as L
from models import REGISTRY, _register_extra

STEP = 0.01
SPEED_ERR = (-0.05, -0.02, -0.01, 0.01, 0.02, 0.05)
OUT = L.RESULTS / "phase8_robust"


def _shift(S, e):
    """S'(o) = S(o (1+e)) on the stored grid, rows = orders; returns the first 801 orders (0-8X)."""
    o = np.arange(S.shape[0]) * STEP
    src = np.minimum(o[:801] * (1 + e), o[-1])
    return np.stack([np.interp(src, o, S[:, c]) for c in range(S.shape[1])], 1)


def perturb(r, kind, arg):
    if kind == "clean":
        return r                                    # the original row: clean predictions reproduce Phase 1
    q = {k: r[k] for k in ("path", "speed_hz", "rms", "kurtosis", "skewness", "peak", "crest")}
    S, E = np.asarray(r["order_spectrum"], float), np.asarray(r["envelope_spectrum"], float)
    if kind == "speed":
        q["order_spectrum"], q["envelope_spectrum"] = _shift(S, arg), _shift(E, arg)
        q["speed_hz"] = r["speed_hz"] * (1 + arg)
    else:
        S, E = S[:801].copy(), E[:801].copy()
        for k in ("rms", "kurtosis", "skewness", "peak", "crest"):
            q[k] = np.array(r[k], float).copy()
        c = arg
        if kind == "dead":
            S[:, c] = 1e-3 * np.median(S[:, c]); E[:, c] = 1e-3 * np.median(E[:, c])
            q["rms"][c] *= 0.01; q["kurtosis"][c] = 3.0; q["skewness"][c] = 0.0; q["crest"][c] = 4.5
            q["peak"][c] = 4.5 * q["rms"][c]
        elif kind == "gain2":
            S[:, c] *= 2; E[:, c] *= 2; q["rms"][c] *= 2; q["peak"][c] *= 2
        q["order_spectrum"], q["envelope_spectrum"] = S, E
    return q


def conditions(n_ch):
    return ([("clean", None)] + [("speed", e) for e in SPEED_ERR] + [("dead", c) for c in range(n_ch)]
            + [("gain2", c) for c in range(n_ch)])


def main(dataset, task, model, seeds):
    from sklearn.metrics import f1_score
    _register_extra()
    fp = REGISTRY[model]
    rows = L.load_rows(dataset)
    sel, y, paths, sp = L.task_data(rows, task)
    n_cls = int(y.max()) + 1
    n_ch = np.asarray(sel[0]["order_spectrum"]).shape[1]
    conds = conditions(n_ch)
    base = list(sel)                                          # training rows: the original clean rows
    folds = list(L.outer_splits(y, sp))
    for seed in seeds:
        f = OUT / task / model / f"seed{seed}.json"
        if f.exists():
            continue
        t0 = time.time()
        preds = {f"{k}:{a}": np.full(len(y), -1) for k, a in conds}
        for name, tr, te in folds:
            te = np.asarray(te)
            aug, y_aug, sp_aug, te_aug = list(base), list(y), list(sp), []
            for k, a in conds:
                idx = []
                for i in te:
                    q = perturb(sel[i], k, a)
                    idx.append(len(aug)); aug.append(q); y_aug.append(y[i]); sp_aug.append(q["speed_hz"])
                te_aug.append(np.array(idx))
            p, _, _ = fp(aug, np.array(y_aug), np.array(sp_aug, float), np.asarray(tr), np.concatenate(te_aug), seed)
            for j, (k, a) in enumerate(conds):
                preds[f"{k}:{a}"][te] = p[j * len(te):(j + 1) * len(te)]
            print(task, model, seed, name, f"clean F1 {f1_score(y[te], p[:len(te)], average='macro'):.3f}", flush=True)
        res = {c: {"macro_f1": float(f1_score(y, v, average="macro", labels=list(range(n_cls)), zero_division=0)),
                   "accuracy": float(np.mean(v == y))} for c, v in preds.items()}
        out = {"task": task, "model": model, "seed": seed, "dataset": dataset, "protocol": "speed",
               "partition_hash": L.partition_hash(paths, folds),
               "conditions": res, "n_channels": n_ch, "seconds": time.time() - t0,
               "runs": [{"path": paths[i], "y": int(y[i]), "pred": {c: int(v[i]) for c, v in preds.items()}} for i in range(len(y))]}
        f.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(f, "w"))
        print(task, model, seed, {c: round(v["macro_f1"], 3) for c, v in res.items()}, flush=True)


if __name__ == "__main__":
    ds, task, model = sys.argv[1:4]
    seeds = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else list(range(5))
    main(ds, task, model, seeds)
