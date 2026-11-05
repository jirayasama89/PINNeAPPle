"""Phase 4/5: multi-expert fusion with a gate selected ONLY on inner validation (nested; no test speed is used).

Experts (configurations of Phase 1, unchanged): `rf` (rotor + statistical features), `mlp` (same features) and
`cnn-spec` (residual CNN on the amplitude + envelope order spectra).
For each outer speed-tercile fold:
  1. inner leave-one-speed-block-out on the outer-training runs (q1lib.inner_splits, 4 contiguous speed blocks):
     each expert is trained on 3 blocks and predicts the 4th -> out-of-fold (OOF) probabilities for every
     outer-training run;
  2. candidate gates are scored on the OOF probabilities (macro-F1): every convex weighting of the three experts on a
     0.25 grid (includes each single expert and the fixed 0.5/0.5 pairs), a multinomial logistic stacker on the
     log-probabilities, and a speed-aware stacker (log-probabilities and their products with the standardised speed);
     the best gate is chosen (ties -> the first in this order);
  3. the experts are retrained on all outer-training runs and the chosen gate (refit on all OOF probabilities for the
     stackers) combines their test probabilities.
Step 3 reuses the stored Phase 1 test probabilities of each expert (identical training), else retrains.
The chosen gate and every candidate's OOF score are stored per fold (split_info). Model name `stack-rf-mlp-cnn`.
usage: python phase5.py <dataset> <task> [seeds]
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

import q1lib as L
from models import REGISTRY, _register_extra

EXPERTS = ("rf", "mlp", "cnn-spec")


def _grid(step=0.25, k=3):
    n = int(round(1 / step))
    return [tuple(a * step for a in w) for w in itertools.product(range(n + 1), repeat=k) if sum(w) == n]


def _stack_feats(P, z, speed_aware):
    F = np.concatenate([np.log(np.clip(p, 1e-6, 1)) for p in P], 1)
    return np.concatenate([F, F * z[:, None], z[:, None]], 1) if speed_aware else F


TASK = {"name": None}


def _phase1_proba(expert, sel, te, seed):
    import json
    f = L.RESULTS / "phase1" / TASK["name"] / expert / f"speed_seed{seed}.json"
    if TASK["name"] is None or not f.exists():
        return None
    pr = {r["path"]: r["proba"] for r in json.load(open(f))["runs"]}
    out = [pr.get(sel[i]["path"]) for i in te]
    return None if any(p is None for p in out) else np.array(out, float)


def stacker():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    _register_extra()
    fps = [REGISTRY[e] for e in EXPERTS]

    def fp(sel, y, sp, tr, te, seed):
        tr, te = np.asarray(tr), np.asarray(te)
        n_cls = int(y.max()) + 1
        labels = list(range(n_cls))
        oof = [np.full((len(tr), n_cls), np.nan) for _ in EXPERTS]
        for _, itr, iva in L.inner_splits(sp[tr]):
            for k, f in enumerate(fps):
                _, pr, _ = f(sel, y, sp, tr[itr], tr[iva], seed)
                oof[k][iva] = pr
        ok = ~np.any([np.isnan(o).any(1) for o in oof], 0)
        yo = y[tr][ok]
        Po = [o[ok] for o in oof]
        mu, sd = sp[tr].mean(), sp[tr].std() + 1e-9
        zo = ((sp[tr] - mu) / sd)[ok]
        cands = []
        for w in _grid():
            p = sum(wi * pi for wi, pi in zip(w, Po))
            cands.append((f"convex{w}", float(f1_score(yo, p.argmax(1), average="macro", labels=labels, zero_division=0)), w))
        for name, sa in (("stacker", False), ("stacker-speed", True)):
            # OOF score of a stacker: fit on 3 inner blocks' OOF, score on the 4th (second-level cross-validation)
            pred = np.full(len(yo), -1)
            blocks = L.inner_splits(sp[tr][ok])
            for _, a, b in blocks:
                lr = LogisticRegression(C=1.0, max_iter=2000).fit(_stack_feats([p[a] for p in Po], zo[a], sa), yo[a])
                pr = np.zeros((len(b), n_cls))
                pr[:, lr.classes_] = lr.predict_proba(_stack_feats([p[b] for p in Po], zo[b], sa))
                pred[b] = pr.argmax(1)
            cands.append((name, float(f1_score(yo, pred, average="macro", labels=labels, zero_division=0)), sa))
        best = max(cands, key=lambda c: c[1])                      # max() keeps the first of equal scores
        # experts trained on all outer-training runs = the Phase 1 confirmations (same code, data, partition, seed):
        # their stored test probabilities are reused when present, otherwise the expert is retrained
        Pt = [_phase1_proba(e, sel, te, seed) if _phase1_proba(e, sel, te, seed) is not None else f(sel, y, sp, tr, te, seed)[1]
              for e, f in zip(EXPERTS, fps)]
        if best[0].startswith("convex"):
            pr = sum(wi * pi for wi, pi in zip(best[2], Pt))
        else:
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(_stack_feats(Po, zo, best[2]), yo)
            pr = np.zeros((len(te), n_cls))
            pr[:, lr.classes_] = lr.predict_proba(_stack_feats(Pt, (sp[te] - mu) / sd, best[2]))
        return pr.argmax(1), pr, {"chosen_gate": best[0], "inner_oof_macro_f1": {c[0]: round(c[1], 4) for c in cands},
                                  "n_oof": int(ok.sum())}
    return fp


def main(dataset, task, seeds):
    rows = L.load_rows(dataset)
    TASK["name"] = task
    for s in seeds:
        L.run(task, "stack-rf-mlp-cnn", stacker(), rows, s, exp="phase5", dataset=dataset,
              config={"experts": list(EXPERTS), "gates": "convex 0.25 grid | logistic stacker | speed-aware stacker",
                      "selection": "inner leave-one-speed-block-out OOF macro-F1 (q1lib.inner_splits)"})


if __name__ == "__main__":
    ds, task = sys.argv[1:3]
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(5))
    main(ds, task, seeds)
