"""Phase 3: CNN vs FNO at comparable budget, hyper-parameters chosen ONLY by inner validation (nested).

Selection (seed 0): for each outer speed-tercile fold, every grid config is trained on 3 of the 4 contiguous speed
blocks of the outer-training runs and scored (macro-F1) on the 4th, leave-one-block-out (q1lib.inner_splits); the
config with the best mean inner macro-F1 is written to locks/phase3_<task>_<family>.json with all inner scores,
BEFORE any outer evaluation of that family. Confirmation: `<family>-tuned` trains the locked config of each outer
fold with seeds 0-4 and is evaluated on the held-out tercile (standard result files, same partitions).

usage: python phase3.py select <dataset> <task> <family>      (family: cnn | fno)
       python phase3.py confirm <dataset> <task> <family>
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np

import q1lib as L
from models import OSFNO, n_params, spectra
from models2 import cnn_spec

GRIDS = {
    "cnn": [{"width": 16, "blocks": 4}, {"width": 32, "blocks": 3}, {"width": 32, "blocks": 4}, {"width": 48, "blocks": 5}],
    "fno": [{"width": 32, "modes": 64, "layers": 3}, {"width": 64, "modes": 64, "layers": 3},
            {"width": 32, "modes": 128, "layers": 3}, {"width": 32, "modes": 64, "layers": 4}],
}
LOCKS = L.RESULTS.parent / "locks"


import os
EPOCHS = 1 if os.environ.get("Q1_SMOKE") == "1" else 100       # smoke test only; real runs use 100


def make(family, cfg):
    if family == "cnn":
        return cnn_spec(width=cfg["width"], blocks=cfg["blocks"], epochs=EPOCHS)

    def fp(sel, y, sp, tr, te, seed):
        X = spectra(sel, 8.0)
        m = OSFNO(seed=seed, speed="none", width=cfg["width"], modes=cfg["modes"], layers=cfg["layers"], epochs=EPOCHS).fit(X[tr], sp[tr], y[tr])
        pr = m.predict_proba(X[te], sp[te])
        return pr.argmax(1), pr, dict(m.info)
    return fp


def select(dataset, task, family):
    from sklearn.metrics import f1_score
    rows = L.load_rows(dataset)
    sel, y, paths, sp = L.task_data(rows, task)
    n_cls = int(y.max()) + 1
    lock = {"task": task, "family": family, "dataset": dataset, "grid": GRIDS[family], "selection_seed": 0,
            "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "folds": {}}
    for name, tr, te in L.outer_splits(y, sp):
        tr = np.asarray(tr)
        scores = []
        for cfg in GRIDS[family]:
            fp = make(family, cfg)
            s = []
            for iname, itr, iva in L.inner_splits(sp[tr]):
                a_tr, a_va = tr[itr], tr[iva]
                if len(np.unique(y[a_tr])) < n_cls:
                    continue
                p, _, _ = fp(sel, y, sp, a_tr, a_va, 0)
                s.append(float(f1_score(y[a_va], p, average="macro", labels=list(range(n_cls)), zero_division=0)))
            scores.append({"config": cfg, "inner_macro_f1": s, "mean": float(np.mean(s))})
            print(task, family, name, cfg, f"inner {np.mean(s):.3f}", flush=True)
        best = max(scores, key=lambda d: d["mean"])
        lock["folds"][name] = {"chosen": best["config"], "candidates": scores}
    lock["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    LOCKS.mkdir(exist_ok=True)
    f = LOCKS / (f"phase3_{task}_{family}.json" if EPOCHS == 100 else f"SMOKE_phase3_{task}_{family}.json")
    json.dump(lock, open(f, "w"), indent=1)
    print("locked", f, hashlib.sha256(f.read_bytes()).hexdigest()[:12])


def confirm(dataset, task, family):
    lock = json.load(open(LOCKS / (f"phase3_{task}_{family}.json" if EPOCHS == 100 else f"SMOKE_phase3_{task}_{family}.json")))
    rows = L.load_rows(dataset)
    sel, y, paths, sp = L.task_data(rows, task)
    names = [n for n, _, _ in L.outer_splits(y, sp)]
    fold_of = {}
    for n, tr, te in L.outer_splits(y, sp):
        fold_of[tuple(np.asarray(te)[:5])] = n

    def tuned(sel_, y_, sp_, tr, te, seed):
        n = fold_of[tuple(np.asarray(te)[:5])]
        cfg = lock["folds"][n]["chosen"]
        p, pr, info = make(family, cfg)(sel_, y_, sp_, tr, te, seed)
        return p, pr, {**(info or {}), "config": cfg}
    for s in range(5):
        L.run(task, f"{family}-tuned", tuned, rows, s, exp="phase3" if EPOCHS == 100 else "SMOKE", dataset=dataset,
              config={"family": family, "lock": f"locks/phase3_{task}_{family}.json", "folds": names})


if __name__ == "__main__":
    cmd, dataset, task, family = sys.argv[1:5]
    (select if cmd == "select" else confirm)(dataset, task, family)
