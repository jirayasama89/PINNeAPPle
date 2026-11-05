"""Paired statistics between two models on the same runs, seeds and partitions (PROTOCOL.md, 'Statistics').

compare(exp_a, model_a, exp_b, model_b, task) ->
  per-seed pooled macro-F1 of each, paired seed difference (mean, 95 % t-interval, n positive),
  per-tercile (low/mid/high) seed-mean macro-F1 and difference,
  record-level paired bootstrap of the macro-F1 difference on seed-averaged probabilities (or majority votes),
  McNemar test on seed-majority correctness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from q1lib import BANDS, RESULTS, band_of


def load(exp, model, task, protocol="speed"):
    files = sorted((RESULTS / exp / task / model).glob(f"{protocol}_seed*.json"))
    return [json.load(open(f)) for f in files]


def macro_f1(y, p, n):
    from sklearn.metrics import f1_score
    return f1_score(y, p, average="macro", labels=list(range(n)), zero_division=0)


def seed_vote(res):
    """Majority vote over seeds per run (ties -> lowest class), aligned by path."""
    paths = [r["path"] for r in res[0]["runs"]]
    y = np.array([r["y"] for r in res[0]["runs"]])
    n = len(res[0]["classes"])
    votes = np.zeros((len(paths), n))
    for s in res:
        idx = {r["path"]: r["pred"] for r in s["runs"]}
        for i, p in enumerate(paths):
            votes[i, idx[p]] += 1
    return paths, y, votes.argmax(1), n


def ci(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()), float("nan")
    return float(d.mean()), float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


def compare(exp_a, model_a, exp_b, model_b, task, n_boot=10000, seed=0):
    A, B = load(exp_a, model_a, task), load(exp_b, model_b, task)
    assert A and B, (model_a, model_b, task)
    assert A[0]["partition_hash"] == B[0]["partition_hash"], "different partitions: not comparable"
    sa = {s["seed"]: s for s in A}
    sb = {s["seed"]: s for s in B}
    common = sorted(set(sa) & set(sb))
    fa = np.array([sa[k]["pooled"]["macro_f1"] for k in common])
    fb = np.array([sb[k]["pooled"]["macro_f1"] for k in common])
    d_mean, d_ci = ci(fa - fb)
    bands = {}
    for b in BANDS:
        va = [v["macro_f1"] for k in common for n, v in sa[k]["per_split"].items() if band_of(n) == b]
        vb = [v["macro_f1"] for k in common for n, v in sb[k]["per_split"].items() if band_of(n) == b]
        bands[b] = {"a": float(np.mean(va)), "b": float(np.mean(vb)), "diff": ci(np.array(va) - np.array(vb))}
    pa, y, va_, n = seed_vote([sa[k] for k in common])
    pb, yb, vb_, _ = seed_vote([sb[k] for k in common])
    assert pa == pb
    rng = np.random.default_rng(seed)
    diffs = []
    idx_by_cls = [np.flatnonzero(y == c) for c in range(n)]
    for _ in range(n_boot):
        ii = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_by_cls if len(ix)])
        diffs.append(macro_f1(y[ii], va_[ii], n) - macro_f1(y[ii], vb_[ii], n))
    diffs = np.array(diffs)
    ca, cb = va_ == y, vb_ == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p_mcn = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return {"task": task, "a": f"{exp_a}/{model_a}", "b": f"{exp_b}/{model_b}", "seeds": common,
            "macro_f1_a": ci(fa), "macro_f1_b": ci(fb), "diff_seed": (d_mean, d_ci),
            "n_seeds_positive": int(np.sum(fa > fb)), "bands": bands,
            "worst_band_a": float(min(bands[b]["a"] for b in BANDS)), "worst_band_b": float(min(bands[b]["b"] for b in BANDS)),
            "vote_diff": float(macro_f1(y, va_, n) - macro_f1(y, vb_, n)),
            "bootstrap_ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
            "mcnemar": {"a_only_right": n01, "b_only_right": n10, "p": p_mcn}}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()
