"""Model registry: each entry is fit_predict(sel, y, sp, tr, te, seed) -> (pred, proba, info).

Feature models use the repository's chosen features (classify.feature_matrix). Spectral models use the raw order
spectra 0-``max_order`` (classify.spectrum_tensor(raw=True)) normalised on the TRAINING runs only.

OS-FNO (order-spectral FNO) generalises the repository's OCFNOClassifier with a ``speed`` mode:
  "none"  A: no speed anywhere
  "head"  B: standardised speed appended to the pooled features only (= repository's ocfno-nofilm)
  "film"  C: FiLM on every spectral block + speed in the head (= repository's OC-FNO)
  "phys"  E: compact physical speed descriptor in the head: [log(w/w_ref), (w/w_ref)^2] (mass-controlled 1X law),
          w_ref = median training speed; no FiLM
(D, FiLM restricted to a rotor 1X branch, needs the multi-branch model of Phase 4.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from q1lib import C


def n_params(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))


# ------------------------------------------------------------------ feature models
def _features(sel):
    return C.feature_matrix(sel)


def feature_model(kind: str, **hp):
    def fp(sel, y, sp, tr, te, seed):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        X = _features(sel)
        if kind == "rf":
            clf = RandomForestClassifier(hp.get("n_estimators", 500), class_weight="balanced", random_state=seed,
                                         n_jobs=4, max_features=hp.get("max_features", "sqrt"),
                                         min_samples_leaf=hp.get("min_samples_leaf", 1))
        elif kind == "mlp":
            clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=hp.get("hidden", (128, 64)),
                                                                alpha=hp.get("alpha", 1e-3), max_iter=3000,
                                                                random_state=seed))
        elif kind == "svm":
            clf = make_pipeline(StandardScaler(), SVC(C=hp.get("C", 10.0), gamma=hp.get("gamma", "scale"),
                                                      class_weight="balanced", probability=True, random_state=seed))
        elif kind == "xgb":
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=hp.get("n_estimators", 400), max_depth=hp.get("max_depth", 4),
                                learning_rate=hp.get("learning_rate", 0.05), subsample=0.8, colsample_bytree=0.8,
                                random_state=seed, n_jobs=4, tree_method="hist")
        elif kind == "kan":
            k = C.KANClassifier(seed=seed).fit(X[tr], y[tr])
            return k.predict(X[te]), None, {}
        else:
            raise ValueError(kind)
        if kind == "xgb":                         # xgboost needs classes 0..k-1 present in train
            classes = np.unique(y[tr])
            remap = {c: i for i, c in enumerate(classes)}
            clf.fit(X[tr], np.array([remap[v] for v in y[tr]]))
            pr_small = clf.predict_proba(X[te])
            pr = np.zeros((len(te), int(y.max()) + 1))
            pr[:, classes] = pr_small
        else:
            clf.fit(X[tr], y[tr])
            pr = np.zeros((len(te), int(y.max()) + 1))
            pr[:, clf.classes_] = clf.predict_proba(X[te])
        return pr.argmax(1), pr, {}
    return fp


# ------------------------------------------------------------------ OS-FNO
@dataclass
class OSFNO:
    width: int = 32
    modes: int = 64
    layers: int = 3
    epochs: int = 100
    lr: float = 1e-3
    batch: int = 64
    dropout: float = 0.1
    smoothing: float = 0.05
    gain_sigma: float = 0.3
    gain_mode: str = "sensor"          # "sensor" (per sensor, repository) or "common" (one gain per run)
    speed: str = "film"                # none | head | film | phys
    order_prior: bool = True
    max_order: float = 8.0
    seed: int = 0
    info: dict = field(default_factory=dict)

    def _speed_feats(self, sp):
        if self.speed == "phys":
            r = sp / self.sp_ref
            return np.stack([np.log(r), r ** 2], 1).astype(np.float32)
        return ((sp - self.sp_mu) / self.sp_sd)[:, None].astype(np.float32)

    def _norm(self, X):
        Z = np.log1p(X / self.med[None, :, None])
        if self.order_prior:
            Z = np.concatenate([Z, np.broadcast_to(self.prior, (len(Z),) + self.prior.shape)], 1)
        return Z.astype(np.float32)

    def _net(self, in_c, n_cls, n_sp):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from pinneapple_neural.architectures.neural_operators.fno import _SpectralConv
        W, mode, drop = self.width, self.speed, self.dropout
        head_sp = 0 if mode == "none" else n_sp

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.lift = nn.Conv1d(in_c + 1, W, 1)
                s.spec = nn.ModuleList([_SpectralConv(W, W, self.modes) for _ in range(self.layers)])
                s.loc = nn.ModuleList([nn.Conv1d(W, W, 1) for _ in range(self.layers)])
                s.film = nn.ModuleList([nn.Linear(n_sp, 2 * W) for _ in range(self.layers)]) if mode == "film" else None
                s.head = nn.Sequential(nn.Linear(2 * W + head_sp, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, n_cls))

            def embed(s, x, spf):
                B, _, L = x.shape
                grid = torch.linspace(0, 1, L, device=x.device).view(1, 1, L).expand(B, 1, L)
                h = s.lift(torch.cat([x, grid], 1))
                for k in range(len(s.spec)):
                    h = F.gelu(s.spec[k](h) + s.loc[k](h))
                    if s.film is not None:
                        g, b = s.film[k](spf).chunk(2, -1)
                        h = h * (1 + g[..., None]) + b[..., None]
                return torch.cat([h.mean(-1), h.amax(-1)], 1)

            def forward(s, x, spf):
                z = s.embed(x, spf)
                if head_sp:
                    z = torch.cat([z, spf], 1)
                return s.head(z)
        return Net()

    def fit(self, X, sp, y, sample_weight_fn=None):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        rng = np.random.default_rng(self.seed)
        self.med = np.maximum(np.median(X, axis=(0, 2)), 1e-12)
        self.prior = C.order_channels(X.shape[2])
        self.n_cls = int(y.max()) + 1
        self.sp_mu, self.sp_sd, self.sp_ref = float(sp.mean()), float(sp.std() + 1e-9), float(np.median(sp))
        n_sens = X.shape[1]
        spf = self._speed_feats(sp)
        self.net = self._net(n_sens + (2 if self.order_prior else 0), self.n_cls, spf.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        nb = int(np.ceil(len(y) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * nb)
        w = torch.as_tensor(len(y) / (self.n_cls * np.bincount(y, minlength=self.n_cls).clip(1)), dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=self.smoothing)
        S = torch.as_tensor(spf)
        yt = torch.as_tensor(y)
        half = n_sens // 2
        t0 = time.perf_counter()
        for _ in range(self.epochs):
            self.net.train()
            for idx in np.array_split(rng.permutation(len(y)), nb):
                xb = X[idx]
                if self.gain_sigma > 0:
                    if self.gain_mode == "sensor":
                        g = np.exp(rng.normal(0, self.gain_sigma, (len(idx), half)))
                        xb = xb * np.concatenate([g, g], 1)[:, :, None]
                    else:
                        xb = xb * np.exp(rng.normal(0, self.gain_sigma, (len(idx), 1, 1)))
                opt.zero_grad()
                loss = loss_fn(self.net(torch.as_tensor(self._norm(xb)), S[idx]), yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
        self.info = {"fit_seconds": time.perf_counter() - t0, "n_params": n_params(self.net)}
        return self

    def predict_proba(self, X, sp):
        import torch
        self.net.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            p = torch.softmax(self.net(torch.as_tensor(self._norm(X)), torch.as_tensor(self._speed_feats(sp))), 1).numpy()
            self.info["infer_seconds"] = time.perf_counter() - t0
        return p


_SPEC_CACHE: dict = {}


def spectra(sel, max_order):
    key = (id(sel), max_order)
    if key not in _SPEC_CACHE:
        _SPEC_CACHE.clear()
        _SPEC_CACHE[key] = C.spectrum_tensor(sel, max_order=max_order, raw=True)[0]
    return _SPEC_CACHE[key]


def osfno(**hp):
    def fp(sel, y, sp, tr, te, seed):
        X = spectra(sel, hp.get("max_order", 8.0))
        m = OSFNO(seed=seed, **hp).fit(X[tr], sp[tr], y[tr])
        pr = m.predict_proba(X[te], sp[te])
        return pr.argmax(1), pr, dict(m.info)
    return fp


def fno_trainonly(**hp):
    """The manuscript's plain FNO with log scaling estimated on the training runs only (audit item C)."""
    def fp(sel, y, sp, tr, te, seed):
        X = spectra(sel, 8.0)
        med = np.median(X[tr][X[tr] > 0])
        Z = np.log1p(X / med).astype(np.float32)
        m = C.FNOClassifier(seed=seed, **hp).fit(Z[tr], sp[tr].astype(np.float32), y[tr])
        return m.predict(Z[te], sp[te].astype(np.float32)), None, {"fit_seconds": m.seconds}
    return fp


REGISTRY = {
    "rf": feature_model("rf"), "mlp": feature_model("mlp"), "kan": feature_model("kan"),
    "osfno-A-nospeed": osfno(speed="none"), "osfno-B-head": osfno(speed="head"),
    "osfno-C-film": osfno(speed="film"), "osfno-E-phys": osfno(speed="phys"),
    "osfno-B-noaug": osfno(speed="head", gain_sigma=0.0), "osfno-B-noprior": osfno(speed="head", order_prior=False),
    "osfno-B-commonaug": osfno(speed="head", gain_mode="common"),
    "fno-trainonly": fno_trainonly(),
}


def _register_extra():
    from models2 import EXTRA
    REGISTRY.update(EXTRA)


try:
    _register_extra()
except Exception as _e:  # noqa: BLE001  (models2 imports models; resolved lazily)
    pass
