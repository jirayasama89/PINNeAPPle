"""Phase 6: bearings. Pre-registered (PROTOCOL.md, "Phase 6 pre-registration") BEFORE any run; no tuning.

Diagnosis (Phase 1/8): on T3/T4 the weakest classes are `normal` and the overhang cage / outer-race faults; the
feature models see the bearing defect orders only in ONE broadband envelope spectrum per sensor.
Hypothesis H6: envelope order spectra demodulated in four carrier bands (0.5-2, 2-5, 5-10, 10-20 kHz; extracted once
by extract_mafaulda.py, `env_bands`) carry bearing information that the broadband envelope dilutes.

Models (configuration fixed in advance, identical to their Phase 1 parents except the added input):
  rf-mb   RF of Phase 1 + amplitudes at FTF, BSF, BPFO, BPFI (±0.03 order) of each band envelope and sensor
  cnn-mb  cnn-spec of Phase 1 with the 4 x sensors band-envelope spectra (0-8X) appended as input channels,
          log-normalised by the per-channel median of the training split
Tasks: bearings (T3) and all (T4), 5 seeds, Phase 1 partitions; compared with rf and cnn-spec by stats_q1.compare.
Also `diagnose`: Fisher ratio (between/within class variance, log amplitude) of the defect-order amplitudes, broadband
vs best band, computed on all T3 runs (descriptive physics, not used for any selection).
Uses the full cache, or the slim cache + data_slim/mafaulda_envbands.npz (make_envbands_slim.py).
usage: python phase6.py run <task> <model> | python phase6.py diagnose
"""
from __future__ import annotations

import json
import sys

import numpy as np

import q1lib as L
from models import n_params
from models2 import _resnet1d, _torch, _train_loop

BO = {"FTF": 0.3750, "BSF": 1.8710, "BPFO": 2.9980, "BPFI": 5.0020}
STEP = 0.01


def _amp(S, o, hw=0.03):
    i0, i1 = int(round((o - hw) / STEP)), int(round((o + hw) / STEP)) + 1
    return S[max(i0, 0):i1].max(0)


def band_features(sel):
    return np.array([np.concatenate([_amp(np.asarray(r["env_bands"][b], float), o) for b in range(4) for o in BO.values()])
                     for r in sel])


def rf_mb():
    def fp(sel, y, sp, tr, te, seed):
        from sklearn.ensemble import RandomForestClassifier
        X = np.concatenate([L.C.feature_matrix(sel), band_features(sel)], 1)
        clf = RandomForestClassifier(500, class_weight="balanced", random_state=seed, n_jobs=4)
        clf.fit(X[tr], y[tr])
        pr = np.zeros((len(te), int(y.max()) + 1))
        pr[:, clf.classes_] = clf.predict_proba(X[te])
        return pr.argmax(1), pr, {"n_features": int(X.shape[1])}
    return fp


def cnn_mb(epochs=100, lr=1e-3, width=32, blocks=4):
    def fp(sel, y, sp, tr, te, seed):
        torch = _torch()
        torch.manual_seed(seed)
        X = L.C.spectrum_tensor(sel, max_order=8.0, raw=True)[0]
        B = np.stack([np.concatenate([np.asarray(r["env_bands"][b][:801], np.float32).T for b in range(4)], 0) for r in sel])
        X = np.concatenate([X, B], 1)
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
        return pr.argmax(1), pr, {"fit_seconds": secs, "n_params": n_params(net), "in_channels": int(Z.shape[1])}
    return fp


MODELS = {"rf-mb": rf_mb, "cnn-mb": cnn_mb}


def load_rows():
    """Full cache rows, or slim rows + data_slim/mafaulda_envbands.npz (identical float16 values, 0-8X)."""
    rows = L.load_rows("mafaulda")
    if "env_bands" not in rows[0]:
        z = np.load(L.RESULTS.parent / "data_slim" / "mafaulda_envbands.npz", allow_pickle=True)
        eb = dict(zip(z["paths"], z["env_bands"]))
        rows = [{**r, "env_bands": eb[r["path"]]} for r in rows]
    return rows


def run(task, model):
    rows = load_rows()
    for s in range(5):
        L.run(task, model, MODELS[model](), rows, s, exp="phase6", dataset="mafaulda",
              config={"model": model, "preregistered": "PROTOCOL.md Phase 6", "bands_khz": [[0.5, 2], [2, 5], [5, 10], [10, 20]],
                      "orders": BO})


def diagnose():
    rows = load_rows()
    sel, y, paths, sp = L.task_data(rows, "bearings")
    classes = sorted(set(y))

    def fisher(A):                              # A: (n, k) log amplitudes -> Fisher ratio per column
        A = np.log(A + 1e-12)
        mu = A.mean(0)
        sb = sum((y == c).sum() * (A[y == c].mean(0) - mu) ** 2 for c in classes)
        sw = sum(((A[y == c] - A[y == c].mean(0)) ** 2).sum(0) for c in classes)
        return sb / np.maximum(sw, 1e-12)
    out = {}
    n_ch = np.asarray(sel[0]["envelope_spectrum"]).shape[1]
    for name, o in BO.items():
        broad = np.array([_amp(np.asarray(r["envelope_spectrum"], float), o) for r in sel])
        bands = [np.array([_amp(np.asarray(r["env_bands"][b], float), o) for r in sel]) for b in range(4)]
        fb = fisher(broad)
        fbands = np.array([fisher(Bb) for Bb in bands])            # (4, n_ch)
        out[name] = {"broadband_per_sensor": fb.round(4).tolist(), "bands_per_sensor": fbands.round(4).tolist(),
                     "best_band_over_broadband_median": float(np.median(fbands.max(0) / np.maximum(fb, 1e-12)))}
    json.dump({"task": "bearings", "n": len(y), "n_channels": n_ch, "fisher_ratio": out},
              open(L.RESULTS / "phase6_diagnose.json", "w"), indent=1)
    print(json.dumps({k: v["best_band_over_broadband_median"] for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        diagnose()
