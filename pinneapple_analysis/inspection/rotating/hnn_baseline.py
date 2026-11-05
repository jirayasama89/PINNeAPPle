"""An HNN-energy baseline for MAFAULDA, in the spirit of Shen, Chowdhury, Banerjee & Terejanu (ICPRAM 2023,
arXiv 2301.02243) -- not their code: a Hamiltonian neural network is fitted to healthy trajectories and the energy
it assigns to a run becomes that run's features.

Trajectories: for each bearing, ``q`` = radial and ``p`` = tangential acceleration (the orbit of the housing), from
the decimated waveform snippets (:func:`.mafaulda.snippet`), each standardised by its own run, with time measured in
shaft revolutions (``t' = t f_r``) so that one HNN serves every speed. The HNN
(:class:`pinneapple_neural.architectures.continuous.hamiltonian.HamiltonianNeuralNetwork`, one degree of freedom) is
trained on the healthy runs of the training split only, on the finite-difference time derivatives.

Features of a run, per bearing: mean and coefficient of variation of the learned energy ``H`` along the trajectory,
mean absolute energy rate ``|dH/dt'| = |grad H . z'|`` (a healthy rotor keeps its orbit), and the RMS residual of
Hamilton's equations ``||z' - J grad H|| / ||z'||``; plus the speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FS_SNIPPET = 5000.0
BEARINGS = ((1, 2), (4, 5))           # (radial, tangential) columns of the snippet: underhang, overhang


def trajectories(row: dict[str, Any], n: int = 2000) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(z, dz/dt')`` for each bearing, ``z = [q, p]`` standardised, time in revolutions."""
    x = row["snippet"][:n].astype(np.float64)
    fr = row["speed_hz"]
    out = []
    for a, b in BEARINGS:
        z = np.stack([x[:, a], x[:, b]], 1)
        z = (z - z.mean(0)) / (z.std(0) + 1e-12)
        dz = np.gradient(z, axis=0) * FS_SNIPPET / fr
        out.append((z[1:-1], dz[1:-1]))
    return out


@dataclass
class HNNEnergy:
    hidden: int = 64
    epochs: int = 80
    lr: float = 1e-3
    points_per_run: int = 300
    seed: int = 0

    def fit(self, rows: list[dict[str, Any]]) -> HNNEnergy:
        import torch

        from pinneapple_neural.architectures.continuous.hamiltonian import HamiltonianNeuralNetwork
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Z, D = [], []
        for r in rows:
            for z, dz in trajectories(r):
                i = rng.choice(len(z), min(self.points_per_run, len(z)), replace=False)
                Z.append(z[i])
                D.append(dz[i])
        Z, D = torch.as_tensor(np.concatenate(Z), dtype=torch.float32), torch.as_tensor(np.concatenate(D),
                                                                                         dtype=torch.float32)
        self.scale = float(D.std())
        self.net = HamiltonianNeuralNetwork(dim_q=1, hidden=self.hidden, num_layers=3)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(Z)).split(1024):
                opt.zero_grad()
                out = self.net(Z[idx], y_true=D[idx] / self.scale, return_loss=True)
                out.losses["total"].backward()
                opt.step()
        return self

    def features(self, rows: list[dict[str, Any]]) -> np.ndarray:
        import torch
        feats = []
        for r in rows:
            f = []
            for z, dz in trajectories(r):
                zt = torch.as_tensor(z, dtype=torch.float32)
                out = self.net(zt)
                H = out.extras["H"].detach().numpy()[:, 0]
                field = out.y.detach().numpy() * self.scale
                zr = zt.detach().requires_grad_(True)
                g = torch.autograd.grad(self.net.H(zr).sum(), zr)[0].numpy()
                rate = np.abs((g * dz).sum(1))
                res = np.linalg.norm(dz - field, axis=1).mean() / (np.linalg.norm(dz, axis=1).mean() + 1e-12)
                f += [H.mean(), H.std() / (abs(H.mean()) + 1e-9), rate.mean(), res]
            feats.append(f + [r["speed_hz"]])
        return np.array(feats)


def run_task(rows, task, protocols=("folds", "speed"), seed=0, log=print) -> dict[str, Any]:
    """HNN-energy features (HNN fitted on the healthy runs of each training split) with a logistic regression and a
    random forest, same tasks and splits as :mod:`.classify`."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .classify import TASKS, select, splits
    sel, y = select(rows, task)
    sp = np.array([r["speed_hz"] for r in sel])
    normal = TASKS[task].index("normal")
    out = {"task": task, "classes": TASKS[task], "n": int(len(y)), "results": {}}
    for proto in protocols:
        preds = {m: np.full(len(y), -1) for m in ("hnn_logreg", "hnn_rf")}
        for _name, tr, te in splits(y, sp, proto, seed=seed):
            hnn = HNNEnergy(seed=seed).fit([sel[i] for i in tr if y[i] == normal])
            F = hnn.features(sel)
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
            preds["hnn_logreg"][te] = lr.fit(F[tr], y[tr]).predict(F[te])
            rf = RandomForestClassifier(500, class_weight="balanced", random_state=seed, n_jobs=4)
            preds["hnn_rf"][te] = rf.fit(F[tr], y[tr]).predict(F[te])
        for m, p in preds.items():
            out["results"][f"{proto}/{m}"] = {"macro_f1": float(f1_score(y, p, average="macro")),
                                              "accuracy": float(np.mean(p == y)),
                                              "confusion": confusion_matrix(y, p).tolist()}
            log(f"{task:12s} {proto:5s} {m:10s} macro-F1 {out['results'][f'{proto}/{m}']['macro_f1']:.3f}")
    return out
