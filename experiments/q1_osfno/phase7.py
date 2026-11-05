"""Phase 7: compound faults of COMFAULDA recognised from single-fault training (multi-label: unbalance, horizontal,
vertical). Train = normal + single-fault runs (all speeds); test = the 1143 combined runs never seen.

Methods (5 seeds each):
  br-rf / br-mlp      binary relevance on the chosen features (one detector per fault; the manuscript's baseline)
  ml-cnn              residual CNN on order spectra with a shared encoder and 3 sigmoid outputs
  ml-cnn-mix          ml-cnn + controlled synthetic mixtures: for a pair of single faults (A, B) and a normal run at
                      speeds within 2 % of each other, Y = max(Y_A + Y_B - Y_0, 0) in linear amplitude, label A∪B;
                      mixtures are built from TRAINING runs only and added to each epoch (30 % of the batch)
Metrics: exact match, per-label F1, Hamming loss, exact match per combination; plus a superposition analysis:
for every real combined run, the relative error of the superposition prediction Y_A + Y_B - Y_0 (nearest-speed single
and normal runs) at orders 1X-4X and over 0-8X, against the same error between two single-fault runs of the same
class (noise floor).
usage: python phase7.py run <method> | python phase7.py superposition
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import q1lib as L

FAULTS = ("unbalance", "horizontal", "vertical")
OUT = L.RESULTS / "phase7"


def data():
    rows = [r for r in L.load_rows("comfaulda") if "error" not in r]
    Y = np.array([[bool(r["labels"][f]) for f in FAULTS] for r in rows], int)
    comb = np.array([bool(r["combined"]) for r in rows])
    sp = np.array([r["speed_hz"] for r in rows])
    X = L.C.spectrum_tensor(rows, max_order=8.0, raw=True)[0]
    return rows, Y, comb, sp, X


def metrics_ml(Y, P, rows):
    from sklearn.metrics import f1_score
    ex = float(np.mean(np.all(P == Y, 1)))
    per = {f: float(f1_score(Y[:, j], P[:, j], zero_division=0)) for j, f in enumerate(FAULTS)}
    ham = float(np.mean(P != Y))
    by = {}
    for st in sorted({r["state"] for r in rows}):
        m = np.array([r["state"] == st for r in rows])
        by[st] = {"exact": float(np.mean(np.all(P[m] == Y[m], 1))), "n": int(m.sum()),
                  "predicted_sets": {"".join(map(str, k)): int(v) for k, v in zip(*np.unique(P[m], axis=0, return_counts=True))}}
    return {"exact_match": ex, "per_label_f1": per, "hamming_loss": ham, "by_combination": by}


def mixtures(X, Y, sp, idx, rng, n):
    """Synthetic mixtures from training singles + normals at speeds within 2 %."""
    singles = {j: idx[(Y[idx].sum(1) == 1) & (Y[idx][:, j] == 1)] for j in range(3)}
    normals = idx[Y[idx].sum(1) == 0]
    pairs = [(0, 1), (0, 2), (1, 2)]
    Xs, Ys = [], []
    tries = 0
    while len(Xs) < n and tries < 50 * n:
        tries += 1
        a, b = pairs[rng.integers(3)]
        ia = rng.choice(singles[a])
        cand_b = singles[b][np.abs(sp[singles[b]] / sp[ia] - 1) < 0.02]
        cand_0 = normals[np.abs(sp[normals] / sp[ia] - 1) < 0.02]
        if len(cand_b) == 0 or len(cand_0) == 0:
            continue
        ib, i0 = rng.choice(cand_b), rng.choice(cand_0)
        Xs.append(np.maximum(X[ia] + X[ib] - X[i0], 0.0))
        y = np.zeros(3, int); y[a] = y[b] = 1
        Ys.append(y)
    return np.array(Xs, np.float32), np.array(Ys)


def ml_cnn(X, Y, tr, te, seed, mix=False, epochs=60, lr=1e-3):
    import torch
    from models2 import _resnet1d
    torch.manual_seed(seed); torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    med = np.maximum(np.median(X[tr], axis=(0, 2)), 1e-12)
    norm = lambda A: torch.as_tensor(np.log1p(A / med[None, :, None]).astype(np.float32))
    net = _resnet1d(X.shape[1], 3, 32, 4, n_sp=0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    nb = int(np.ceil(len(tr) / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nb)
    pos = Y[tr].mean(0)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor((1 - pos) / np.maximum(pos, 1e-3), dtype=torch.float32))
    empty = torch.zeros((1, 0))
    sp_dummy = None
    for _ in range(epochs):
        net.train()
        for idx in np.array_split(rng.permutation(tr), nb):
            xb, yb = X[idx], Y[idx]
            if mix:
                xm, ym = mixtures(X, Y, SP, np.asarray(tr), rng, max(1, int(0.3 * len(idx))))
                if len(xm):
                    xb, yb = np.concatenate([xb, xm]), np.concatenate([yb, ym])
            opt.zero_grad()
            out = net(norm(xb), empty.expand(len(xb), 0))
            loss = lossf(out, torch.as_tensor(yb, dtype=torch.float32))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        pr = torch.sigmoid(net(norm(X[te]), empty.expand(len(te), 0))).numpy()
    return (pr > 0.5).astype(int), pr


SP = None


def run(method):
    global SP
    rows, Y, comb, sp, X = data()
    SP = sp
    tr, te = np.flatnonzero(~comb), np.flatnonzero(comb)
    Xf = L.C.feature_matrix(rows)
    for seed in range(5):
        f = OUT / method / f"seed{seed}.json"
        if f.exists():
            continue
        t0 = time.time()
        if method.startswith("br-"):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            P = np.zeros((len(te), 3), int); PR = np.zeros((len(te), 3))
            for j in range(3):
                clf = (RandomForestClassifier(500, class_weight="balanced", random_state=seed, n_jobs=4) if method == "br-rf"
                       else make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=3000, random_state=seed)))
                clf.fit(Xf[tr], Y[tr, j])
                PR[:, j] = clf.predict_proba(Xf[te])[:, 1]
                P[:, j] = (PR[:, j] > 0.5).astype(int)
        else:
            P, PR = ml_cnn(X, Y, tr, te, seed, mix=(method == "ml-cnn-mix"))
        res = {"method": method, "seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)), "seconds": time.time() - t0,
               **metrics_ml(Y[te], P, [rows[i] for i in te]),
               "runs": [{"path": rows[i]["path"], "y": Y[i].tolist(), "pred": P[k].tolist(), "proba": PR[k].round(4).tolist()}
                        for k, i in enumerate(te)]}
        f.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(f, "w"))
        print(method, seed, f"exact {res['exact_match']:.3f}", {k: round(v, 2) for k, v in res["per_label_f1"].items()},
              f"hamming {res['hamming_loss']:.3f}", flush=True)


def superposition():
    rows, Y, comb, sp, X = data()
    singles = {j: np.flatnonzero((Y.sum(1) == 1) & (Y[:, j] == 1)) for j in range(3)}
    normals = np.flatnonzero(Y.sum(1) == 0)
    orders = {"1X": 100, "2X": 200, "3X": 300, "4X": 400}

    def near(pool, s):
        return pool[np.argmin(np.abs(sp[pool] - s))]

    def amp(Xr, k):
        return Xr[:, k - 3:k + 4].max(1)
    out = {"by_combination": {}, "noise_floor": {}}
    for i in np.flatnonzero(comb):
        a, b = np.flatnonzero(Y[i])
        ia, ib, i0 = near(singles[a], sp[i]), near(singles[b], sp[i]), near(normals, sp[i])
        pred = np.maximum(X[ia] + X[ib] - X[i0], 0)
        st = rows[i]["state"]
        d = out["by_combination"].setdefault(st, {k: [] for k in list(orders) + ["full"]})
        for k, c in orders.items():
            d[k].append(float(np.linalg.norm(amp(pred, c) - amp(X[i], c)) / (np.linalg.norm(amp(X[i], c)) + 1e-12)))
        d["full"].append(float(np.linalg.norm(pred - X[i]) / (np.linalg.norm(X[i]) + 1e-12)))
    for j, f in enumerate(FAULTS):                       # same class, nearest-speed pair: measurement floor
        e = []
        for i in singles[j]:
            others = singles[j][singles[j] != i]
            k = near(others, sp[i])
            e.append(float(np.linalg.norm(X[k] - X[i]) / (np.linalg.norm(X[i]) + 1e-12)))
        out["noise_floor"][f] = float(np.median(e))
    summ = {st: {k: float(np.median(v)) for k, v in d.items()} for st, d in out["by_combination"].items()}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"median_relative_error": summ, "noise_floor_median_same_class": out["noise_floor"]},
              open(OUT / "superposition.json", "w"), indent=1)
    print(json.dumps({"median_relative_error": summ, "noise_floor": out["noise_floor"]}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        superposition()
