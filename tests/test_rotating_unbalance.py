"""Rotating-machinery order tracking and unbalance inversion: exact phasors of a synthetic run, the MAFAULDA index
from folder names, and the calibrated inversion recovering known masses on a synthetic structure."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.inspection.rotating import (
    AmplitudeInfluence,
    ModalInfluence,
    PhaseNuisanceInfluence,
    SplineInfluence,
    estimate_mass,
    level_unbalance,
    mafaulda,
    order_phasors,
    order_resolution,
    shaft_angle,
)
from pinneapple_analysis.inspection.rotating.influence import ScreenedEstimator

FS = 50_000.0


def _run(f_hz, comps, seconds=2.0, drift=0.01):
    """A tachometer (one pulse per revolution, speed drifting by ``drift``) and a channel sum of order components."""
    t = np.arange(int(seconds * FS)) / FS
    f = f_hz * (1 + drift * np.sin(2 * np.pi * 0.7 * t))
    theta = 2 * np.pi * np.cumsum(f) / FS
    tach = np.where(np.mod(theta, 2 * np.pi) < 0.3, 5.0, 0.0)
    x = sum(abs(c) * np.cos(k * theta + np.angle(c)) for k, c in comps.items())
    return tach, x, f


def test_order_phasors_recover_amplitude_and_phase():
    tach, x, f = _run(30.0, {1: 0.8 * np.exp(0.6j), 2: 0.3 * np.exp(-1.1j), 3: 0.05j})
    ang = shaft_angle(tach, FS)
    assert ang.speed_hz == pytest.approx(f[ang.first:ang.last].mean(), rel=1e-3)   # mean over whole revolutions
    P = order_phasors(x, ang, (1, 2, 3))[:, 0]
    # the pulse starts at the rising edge, theta = 0 there
    assert np.allclose(P, [0.8 * np.exp(0.6j), 0.3 * np.exp(-1.1j), 0.05j], atol=5e-3)
    assert order_resolution(ang) == pytest.approx(1 / ang.revolutions)


def test_mafaulda_index_reads_state_level_and_speed(tmp_path):
    for rel in ("normal/12.288.csv", "imbalance/6g/30.72.csv", "underhang/outer_race/20g/45.4656.csv",
                "horizontal-misalignment/0.5mm/20.2752.csv"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("0,0,0,0,0,0,0,0\n")
    got = {(r.state, r.level_value, r.level_unit, r.nominal_hz) for r in mafaulda.index(tmp_path)}
    assert got == {("normal", 0.0, "", 12.288), ("imbalance", 6.0, "g", 30.72),
                   ("underhang/outer_race", 20.0, "g", 45.4656), ("horizontal-misalignment", 0.5, "mm", 20.2752)}


def _synthetic(n_per=40, seed=0, phase_drift=0.0):
    """A two-mode structure (the same for every call), six channels, alpha = -w^4 H(w), residual unbalance 3 g,
    1 % noise; ``seed`` draws the runs."""
    R = np.random.default_rng(123).normal(size=(2, 6)) + 1j * np.random.default_rng(124).normal(size=(2, 6))
    rng = np.random.default_rng(seed)
    levels = [0, 6, 10, 15, 20, 25, 30, 35]
    w = np.concatenate([rng.uniform(12, 60, n_per) for _ in levels])
    m = np.repeat(levels, n_per).astype(float)
    W = 2 * np.pi * np.array([25.0, 48.0])
    om = 2 * np.pi * w[:, None]
    H = sum(R[j][None, :] / (W[j] ** 2 - om ** 2 + 2j * 0.05 * W[j] * om) for j in range(2))
    alpha = -(om ** 4) * H * 1e-6
    A = alpha * (m[:, None] + 3.0 * np.exp(0.4j))
    A = A * np.exp(1j * phase_drift * (m[:, None] / 35.0))
    A = A + 0.01 * np.abs(A).mean() * (rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape))
    return w, m, A


@pytest.mark.parametrize("model", [SplineInfluence(n_knots=40), ModalInfluence(n_modes=2, n_starts=8)])
def test_calibrated_inversion_recovers_the_mass(model):
    w, m, A = _synthetic()
    wt, mt, At = _synthetic(n_per=5, seed=1)
    model.fit(w, m, A)
    assert np.mean(np.abs(estimate_mass(model, wt, At) - mt)) < 1.0


def test_modal_model_finds_the_natural_frequencies():
    w, m, A = _synthetic()
    mod = ModalInfluence(n_modes=2, n_starts=12).fit(w, m, A)
    assert np.allclose(mod.poles[:, 0], [25.0, 48.0], rtol=0.03)
    assert np.all(mod.poles[:, 1] > 0)


def test_phase_drift_between_levels_breaks_the_phase_reference_but_not_the_magnitude():
    w, m, A = _synthetic(phase_drift=0.8)
    lu = level_unbalance(w, m, A)
    assert np.all(np.diff(lu["magnitude"]) > 0)
    assert np.ptp(lu["angle_deg"][1:]) > 20.0                        # the drift shows in the angle
    wt, mt, At = _synthetic(n_per=5, seed=1, phase_drift=0.8)
    amp = AmplitudeInfluence(n_knots=20).fit(w, m, A)
    nuis = PhaseNuisanceInfluence("tach_reference", n_knots=20).fit(w, m, A)    # the whole response rotates
    assert np.mean(np.abs(nuis.estimate(wt, At) - mt)) < 2.0
    assert np.mean(np.abs(estimate_mass(amp, wt, At) - mt)) < 4.0


def test_spurious_tachometer_spikes_are_rejected_and_a_missing_pulse_is_flagged():
    tach, x, f = _run(30.0, {1: 0.5})
    noisy = tach.copy()
    rng = np.random.default_rng(3)
    for i in rng.choice(len(tach), 40, replace=False):           # short spikes between pulses
        noisy[i:i + 3] = 5.0
    ang = shaft_angle(noisy, FS)
    assert ang.reliable
    assert ang.speed_hz == pytest.approx(f[ang.first:ang.last].mean(), rel=2e-3)
    edges = np.flatnonzero((tach[:-1] < 2.5) & (tach[1:] >= 2.5))
    gap = tach.copy()
    gap[edges[10]:edges[10] + 400] = 0.0                          # one pulse lost
    assert not shaft_angle(gap, FS).reliable


def test_screened_estimator_rejects_a_faulty_channel_that_the_joint_estimate_believes():
    w, m, A = _synthetic()
    wt, mt, At = _synthetic(n_per=5, seed=1)
    bad = At.copy()
    bad[:, 4] *= 40.0                                              # one accelerometer with a wrong gain
    amp = AmplitudeInfluence(n_knots=40).fit(w, m, A)
    assert np.mean(np.abs(estimate_mass(amp, wt, bad) - mt)) > 10.0
    scr = ScreenedEstimator(make=lambda: AmplitudeInfluence(n_knots=40), c=12.0).fit(w, m, A)
    est, flags = scr.estimate(wt, bad)
    assert np.mean(np.abs(est - mt)) < 2.0
    assert flags[:, 4].mean() > 0.9 and flags[:, [0, 1, 2, 3, 5]].mean() < 0.1
    clean, _ = scr.estimate(wt, At)
    assert np.mean(np.abs(clean - mt)) < np.mean(np.abs(estimate_mass(amp, wt, At) - mt)) + 0.5


def test_order_spectrum_grid_keeps_peaks():
    from pinneapple_analysis.inspection.rotating import (
        mafaulda as _m,  # noqa: F401  (module imports cleanly)
    )
    from pinneapple_analysis.inspection.rotating.orders import order_spectrum
    tach, x, _ = _run(30.0, {1: 0.8, 3: 0.2}, seconds=3.0)
    grid, S = order_spectrum(x, shaft_angle(tach, FS), FS)
    assert S[100, 0] == pytest.approx(0.8, rel=0.02) and S[300, 0] == pytest.approx(0.2, rel=0.05)
    assert np.delete(S[:, 0], [100, 300]).max() < 0.01
