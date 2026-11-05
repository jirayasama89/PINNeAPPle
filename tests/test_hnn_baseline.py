"""HNN-energy baseline on synthetic orbits: the HNN learned on circular orbits sees an elliptic, damped orbit as less
conservative (larger energy rate and Hamilton residual)."""
from __future__ import annotations

import numpy as np

from pinneapple_analysis.inspection.rotating.hnn_baseline import FS_SNIPPET, HNNEnergy


def _row(ecc=1.0, damp=0.0, f=30.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(5000) / FS_SNIPPET
    env = np.exp(-damp * t)
    x = np.zeros((5000, 6), np.float32)
    for a, b in ((1, 2), (4, 5)):
        x[:, a] = env * np.cos(2 * np.pi * f * t) + 0.01 * rng.normal(size=5000)
        x[:, b] = ecc * env * np.sin(2 * np.pi * f * t) + 0.01 * rng.normal(size=5000)
    return {"snippet": x, "speed_hz": f}


def test_energy_features_separate_conservative_from_damped_orbits():
    hnn = HNNEnergy(epochs=60).fit([_row(seed=s, f=20 + 5 * s) for s in range(4)])
    F = hnn.features([_row(seed=10), _row(ecc=0.4, damp=3.0, seed=11)])
    assert F.shape == (2, 9)
    assert F[1, 3] > F[0, 3]                       # Hamilton residual, underhang
    assert F[1, 7] > F[0, 7]                       # Hamilton residual, overhang
