"""GPVS residuals on a synthetic three-phase run: Kirchhoff's current law breaks for a stuck current sensor and the
admittance rule names its phase; a process change (lower current amplitude) keeps both laws."""
from __future__ import annotations

import numpy as np

from pinneapple_systems.digital_twin.gpvs import windows


def _run(n_cycles=100, stuck_phase=None, amp_after=0.5, seed=0):
    fs, f = 10_000.0, 50.0
    t = np.arange(int(n_cycles * fs / f)) / fs
    rng = np.random.default_rng(seed)
    ph = [0, -2 * np.pi / 3, 2 * np.pi / 3]
    half = t >= t[-1] / 2
    amp = np.where(half, amp_after, 0.5)
    run = {"Time": t, "Ipv": np.full_like(t, 1.5), "Vpv": np.full_like(t, 100.0), "Vdc": np.full_like(t, 143.0)}
    for k, p in zip("abc", ph, strict=True):
        run[f"v{k}"] = 155 * np.sin(2 * np.pi * f * t + p) + rng.normal(0, 0.5, len(t))
        run[f"i{k}"] = -amp * np.sin(2 * np.pi * f * t + p) + rng.normal(0, 0.005, len(t))
    if stuck_phase is not None:
        run[f"i{stuck_phase}"] = np.where(half, -0.22, run[f"i{stuck_phase}"])
    return run


def test_stuck_current_sensor_breaks_kcl_and_is_isolated():
    w = windows(_run(stuck_phase="b"))
    after = w["t"] > w["t"][-1] * 0.6
    before = w["t"] < w["t"][-1] * 0.4
    assert np.median(w["R1"][after]) > 10 * np.median(w["R1"][before])
    assert np.all(np.argmax(w["admittance_residual"][after], 1) == 1)       # phase b
    assert np.median(w["R2"][after]) < 0.05


def test_process_change_keeps_the_laws():
    w = windows(_run(amp_after=0.3))
    after = w["t"] > w["t"][-1] * 0.6
    assert np.max(w["R1"][after]) < 0.1 and np.max(w["R2"][after]) < 0.05
