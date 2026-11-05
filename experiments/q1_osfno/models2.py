"""Phase 2 baselines and Phase 5 generalisation variants.

* cnn-spec   : residual 1-D CNN on the same normalised order spectra as OS-FNO (speed in the head, mode B)
* cnn-angle  : residual 1-D CNN on order-tracked waveforms (64 samples / rev, six accelerometers), random crops of
               32 revolutions in training, mean of crop probabilities at test
* osfno-B-dro    : OS-FNO-B trained with GroupDRO (Sagawa et al. 2020) over training speed blocks
* osfno-B-supcon : OS-FNO-B + supervised contrastive loss (Khosla et al. 2020) on the pooled embedding
* fusion-rf-osfno: convex combination of RF and OS-FNO-B probabilities; weight chosen on the inner speed-block
                   validation of the training runs only (never on the held-out tercile)
All use per-channel normalisation estimated on training runs only.
"""
from __future__ import annotations

import time

import numpy as np

import q1lib as L
from models import OSFNO, feature_model, n_params, spectra


def _torch():
    import torch
    torch.set_num_threads(4)
    return torch


# ------------------------------------------------------------------ residual 1-D CNN
def _resnet1d(in_c, n_cls, width=32, blocks=4, n_sp=1, drop=0.1):
    torch = _torch()
    nn = torch.nn

    class Block(nn.Module):
        def __init__(s, c_in, c_out, stride):
            super().__init__()
            s.c1 = nn.Conv1d(c_in, c_out, 7, stride, 3)
            s.b1 = nn.BatchNorm1d(c_out)
            s.c2 = nn.Conv1d(c_out, c_out, 7, 1, 3)
            s.b2 = nn.BatchNorm1d(c_out)
            s.sk = nn.Conv1d(c_in, c_out, 1, stride) if (c_in != c_out or stride != 1) else nn.Identity()

        def forward(s, x):
            h = torch.relu(s.b1(s.c1(x)))
            return torch.relu(s.b2(s.c2(h)) + s.sk(x))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            layers, c = [nn.Conv1d(in_c, width, 7, 1, 3), nn.BatchNorm1d(width), nn.ReLU()], width
            for i in range(blocks):
                c2 = width * (2 ** min(i, 2))
                layers.append(Block(c, c2, 2))
                c = c2
            s.body = nn.Sequential(*layers)
            s.head = nn.Sequential(nn.Linear(2 * c + n_sp, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, n_cls))

        def embed(s, x):
            h = s.body(x)
            return torch.cat([h.mean(-1), h.amax(-1)], 1)

        def forward(s, x, spf):
            z = s.embed(x)
            return s.head(torch.cat([z, spf], 1) if spf.shape[1] else z)
    return Net()


def _train_loop(net, batches_fn, n, y, epochs, lr, seed, n_cls, smoothing=0.05, extra_loss=None, dro_groups=None,
                dro_eta=0.01):
    torch = _torch()
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    nb = int(np.ceil(n / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nb)
    w = torch.as_tensor(n / (n_cls * np.bincount(y, minlength=n_cls).clip(1)), dtype=torch.float32)
    ce = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=smoothing, reduction="none")
    q = None if dro_groups is None else np.ones(dro_groups.max() + 1) / (dro_groups.max() + 1)
    yt = torch.as_tensor(y)
    t0 = time.perf_counter()
    for _ in range(epochs):
        net.train()
        for idx in np.array_split(rng.permutation(n), nb):
            xb, sb = batches_fn(idx, rng)
            opt.zero_grad()
            logits = net(xb, sb)
            li = ce(logits, yt[idx])
            if q is None:
                loss = li.mean()
            else:                                     # GroupDRO: exponentiated-gradient weights over groups
                g = torch.as_tensor(dro_groups[idx])
                gl = torch.stack([li[g == k].mean() if (g == k).any() else li.new_zeros(()) for k in range(len(q))])
                q = q * np.exp(dro_eta * gl.detach().numpy())
                q = q / q.sum()
                loss = (torch.as_tensor(q, dtype=li.dtype) * gl).sum()
            if extra_loss is not None:
                loss = loss + extra_loss(net, xb, sb, idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    return time.perf_counter() - t0


def cnn_spec(epochs=100, lr=1e-3, width=32, blocks=4):
    def fp(sel, y, sp, tr, te, seed):
        torch = _torch()
        torch.manual_seed(seed)
        X = spectra(sel, 8.0)
        med = np.maximum(np.median(X[tr], axis=(0, 2)), 1e-12)
        Z = np.log1p(X / med[None, :, None]).astype(np.float32)
        mu, sd = sp[tr].mean(), sp[tr].std() + 1e-9
        S = torch.as_tensor(((sp - mu) / sd)[:, None], dtype=torch.float32)
        n_cls = int(y.max()) + 1
        net = _resnet1d(Z.shape[1], n_cls, width, blocks)
        Zt = torch.as_tensor(Z)
        trv = np.asarray(tr)
        secs = _train_loop(net, lambda idx, rng: (Zt[trv[idx]], S[trv[idx]]), len(tr), y[tr], epochs, lr, seed, n_cls)
        net.eval()
        with torch.no_grad():
            pr = torch.softmax(net(Zt[te], S[te]), 1).numpy()
        return pr.argmax(1), pr, {"fit_seconds": secs, "n_params": n_params(net)}
    return fp


def cnn_angle(epochs=60, lr=1e-3, crop_revs=32, width=32, blocks=5, n_crops_test=8):
    def fp(sel, y, sp, tr, te, seed):
        torch = _torch()
        torch.manual_seed(seed)
        sigs = [np.asarray(r["angle_sig"], np.float32) for r in sel]
        L_ = crop_revs * 64
        scale = np.median(np.stack([s.std(0) for s in (sigs[i] for i in tr)]), 0) + 1e-9
        n_cls = int(y.max()) + 1
        net = _resnet1d(sigs[0].shape[1], n_cls, width, blocks, n_sp=0)
        empty = torch.zeros((1, 0))

        def crop(s, rng):
            if len(s) <= L_:
                s = np.pad(s, ((0, L_ - len(s) + 1), (0, 0)))
            a = rng.integers(0, len(s) - L_)
            return (s[a:a + L_] / scale).T

        trv = np.asarray(tr)

        def batches(idx, rng):
            xb = torch.as_tensor(np.stack([crop(sigs[trv[i]], rng) for i in idx]))
            return xb, empty.expand(len(idx), 0)
        secs = _train_loop(net, batches, len(tr), y[tr], epochs, lr, seed, n_cls)
        net.eval()
        rng = np.random.default_rng(1234)
        out = []
        with torch.no_grad():
            for i in te:
                xb = torch.as_tensor(np.stack([crop(sigs[i], rng) for _ in range(n_crops_test)]))
                out.append(torch.softmax(net(xb, empty.expand(len(xb), 0)), 1).mean(0).numpy())
        pr = np.stack(out)
        return pr.argmax(1), pr, {"fit_seconds": secs, "n_params": n_params(net)}
    return fp


# ------------------------------------------------------------------ OS-FNO with GroupDRO / SupCon
def osfno_variant(kind, lam=0.1, tau=0.1, n_groups=4, **hp):
    def fp(sel, y, sp, tr, te, seed):
        torch = _torch()
        torch.manual_seed(seed)
        X = spectra(sel, 8.0)
        m = OSFNO(seed=seed, speed="head", **hp)
        m.med = np.maximum(np.median(X[tr], axis=(0, 2)), 1e-12)
        from q1lib import C
        m.prior = C.order_channels(X.shape[2])
        m.n_cls = int(y.max()) + 1
        m.sp_mu, m.sp_sd, m.sp_ref = float(sp[tr].mean()), float(sp[tr].std() + 1e-9), float(np.median(sp[tr]))
        spf = m._speed_feats(sp)
        net = m._net(X.shape[1] + 2, m.n_cls, spf.shape[1])
        m.net = net
        S = torch.as_tensor(spf)
        trv = np.asarray(tr)
        half = X.shape[1] // 2

        def batches(idx, rng):
            xb = X[trv[idx]]
            g = np.exp(rng.normal(0, m.gain_sigma, (len(idx), half)))
            xb = xb * np.concatenate([g, g], 1)[:, :, None]
            return torch.as_tensor(m._norm(xb)), S[trv[idx]]
        edges = np.quantile(sp[tr], np.linspace(0, 1, n_groups + 1))
        groups = np.clip(np.searchsorted(edges, sp[tr], side="right") - 1, 0, n_groups - 1)
        extra = None
        if kind == "supcon":
            ytr = torch.as_tensor(y[tr])

            def extra(net_, xb, sb, idx):
                z = torch.nn.functional.normalize(net_.embed(xb, sb), dim=1)
                sim = z @ z.T / tau
                yy = ytr[idx]
                pos = (yy[:, None] == yy[None, :]).float() - torch.eye(len(idx))
                logits = sim - 1e9 * torch.eye(len(idx))
                logp = logits - torch.logsumexp(logits, 1, keepdim=True)
                denom = pos.sum(1).clamp(min=1)
                return lam * (-(pos * logp).sum(1) / denom).mean()
        secs = _train_loop(net, batches, len(tr), y[tr], m.epochs, m.lr, seed, m.n_cls, extra_loss=extra,
                           dro_groups=groups if kind == "dro" else None)
        net.eval()
        with torch.no_grad():
            pr = torch.softmax(net(torch.as_tensor(m._norm(X[te])), S[te]), 1).numpy()
        return pr.argmax(1), pr, {"fit_seconds": secs, "n_params": n_params(net)}
    return fp


# ------------------------------------------------------------------ fusion RF + OS-FNO-B
def fusion_rf_osfno(grid=(0.0, 0.25, 0.5, 0.75, 1.0), epochs=100):
    rf_fp = feature_model("rf")

    def fno_fp(sel, y, sp, tr, te, seed):
        X = spectra(sel, 8.0)
        m = OSFNO(seed=seed, speed="head", epochs=epochs).fit(X[tr], sp[tr], y[tr])
        return m.predict_proba(X[te], sp[te])

    def fp(sel, y, sp, tr, te, seed):
        n_cls = int(y.max()) + 1
        # alpha chosen on inner speed blocks of the training runs only
        scores = {a: [] for a in grid}
        for _, itr, iva in L.inner_splits(sp[tr]):
            a_tr, a_va = np.asarray(tr)[itr], np.asarray(tr)[iva]
            if len(np.unique(y[a_tr])) < n_cls:
                continue
            p_rf = rf_fp(sel, y, sp, a_tr, a_va, seed)[1]
            p_fn = fno_fp(sel, y, sp, a_tr, a_va, seed)
            from sklearn.metrics import f1_score
            for a in grid:
                scores[a].append(f1_score(y[a_va], (a * p_rf + (1 - a) * p_fn).argmax(1), average="macro"))
        alpha = max(grid, key=lambda a: np.mean(scores[a]) if scores[a] else -1)
        p_rf = rf_fp(sel, y, sp, tr, te, seed)[1]
        p_fn = fno_fp(sel, y, sp, tr, te, seed)
        pr = alpha * p_rf + (1 - alpha) * p_fn
        return pr.argmax(1), pr, {"alpha": alpha, "inner_scores": {str(k): float(np.mean(v)) for k, v in scores.items() if v}}
    return fp


EXTRA = {
    "svm": feature_model("svm"), "xgb": feature_model("xgb"),
    "cnn-spec": cnn_spec(), "cnn-angle": cnn_angle(),
    "osfno-B-dro": osfno_variant("dro"), "osfno-B-supcon": osfno_variant("supcon"),
    "fusion-rf-osfno": fusion_rf_osfno(),
}
