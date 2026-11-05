"""MAFAULDA classification plumbing on synthetic rows: features and spectra shapes, splits that never share a run,
and a separable synthetic problem learned by every model."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.inspection.rotating import classify as C


def _rows(n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k, st in enumerate(C.TASKS["misalignment"]):
        for _ in range(n_per):
            S = rng.random((1201, 7)).astype(np.float32) * 0.01
            E = rng.random((1201, 7)).astype(np.float32) * 0.01
            S[100] += 1.0                                       # 1X
            S[200] += 0.3 * k                                   # 2X grows with the class
            rows.append({"state": st, "tach_ok": True, "speed_hz": float(rng.uniform(12, 60)),
                         "rms": rng.random(7), "kurtosis": 3 + rng.random(7), "skewness": rng.random(7),
                         "crest": 3 + rng.random(7), "peak": rng.random(7), "order_spectrum": S,
                         "envelope_spectrum": E})
    return rows


def test_shapes_and_splits():
    rows, y = C.select(_rows(), "misalignment")
    assert C.feature_matrix(rows).shape == (90, len(C.feature_names()))
    X, sp = C.spectrum_tensor(rows)
    assert X.shape == (90, 14, 801) and sp.shape == (90,)
    for proto in ("folds", "speed"):
        seen = np.zeros(len(y), int)
        for _, tr, te in C.splits(y, sp, proto):
            assert not set(tr) & set(te)
            seen[te] += 1
        assert np.all(seen == 1)                                 # every run tested exactly once


@pytest.mark.parametrize("model", ["rf", "mlp", "kan", "fno"])
def test_models_learn_a_separable_problem(model, monkeypatch):
    monkeypatch.setattr(C.FNOClassifier, "epochs", 25)
    out = C.run_task(_rows(n_per=80), "misalignment", models=(model,), protocols=("folds",), log=lambda *_: None)
    assert out["results"][f"folds/{model}"]["macro_f1"] > 0.8          # chance is 1/3


def test_order_conditioned_fno_learns_and_its_ablations_run(monkeypatch):
    monkeypatch.setattr(C.OCFNOClassifier, "epochs", 30)
    monkeypatch.setattr(C.OCFNOClassifier, "modes", 16)
    monkeypatch.setattr(C.OCFNOClassifier, "width", 8)
    out = C.run_task(_rows(n_per=30), "misalignment", models=("ocfno",), protocols=("speed",), log=lambda *_: None)
    assert out["results"]["speed/ocfno"]["macro_f1"] > 0.45                     # chance is 1/3 (plumbing only)
    monkeypatch.setattr(C.OCFNOClassifier, "epochs", 2)
    for m in ("ocfno-nofilm", "ocfno-noprior", "ocfno-noaug"):                  # ablations run
        C.run_task(_rows(n_per=10), "misalignment", models=(m,), protocols=("speed",), log=lambda *_: None)


def test_order_channels_peak_at_harmonics_and_bearing_orders():
    ch = C.order_channels(801)
    assert ch.shape == (2, 801)
    assert ch[0, 100] > 0.99 and ch[0, 150] < 1e-6                      # 1X yes, 1.5X no
    assert ch[1, int(round(C.BEARING_ORDERS["BSF"] / C.STEP))] > 0.9
