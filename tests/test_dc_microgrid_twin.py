"""DC microgrid twin: the reference simulator conserves current and energy, the residuals vanish on clean readings
and isolate each fault structurally, the whitened sensor directions are orthonormal, the diagnosis names injected
faults, and the PINN residual in torch matches the numpy residual."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from pinneapple_systems.digital_twin.dc_microgrid import (
    COMPONENTS,
    INPUTS,
    NORMAL_BOX,
    RESIDUALS,
    SENSORS,
    Diagnoser,
    Microgrid,
    Residuals,
    Scenario,
    Surrogate,
    component_faults,
    isolability,
    raw_residuals,
    readings,
    run_plant,
    signature_matrix,
    steady_state,
)

GRID = Microgrid()
CENTRE = {k: 0.5 * sum(NORMAL_BOX[k]) for k in INPUTS}


def test_simulator_conserves_current_and_energy_at_every_corner():
    for corner in itertools.product(*[NORMAL_BOX[k] for k in INPUTS]):
        s = steady_state(GRID, dict(zip(INPUTS, corner, strict=True)))
        assert s["kcl_residual_A"] < 1e-9
        assert abs(s["energy_balance_W"]) < 1e-6 * s["P_loads_W"]
        assert 330.0 < s["v_min"] <= s["v_max"] < 400.0


def test_line_resistance_from_iec_60228():
    assert GRID.r_line("12") == pytest.approx(2 * 0.387 * 0.150)


def test_residuals_vanish_on_clean_readings_with_and_without_faults():
    for c in COMPONENTS:
        f = component_faults(c, 0.05)
        y = readings(steady_state(GRID, CENTRE, f))
        assert np.max(np.abs(raw_residuals(GRID, y, CENTRE, f))) < 1e-8
        r = raw_residuals(GRID, y, CENTRE)                 # the healthy model sees only the faulty component's law
        moved = {RESIDUALS[i] for i in np.flatnonzero(np.abs(r) > 1e-8)}
        assert moved == set(signature_matrix(GRID)["components"][c])
        assert len(moved) == 1


def test_signatures_distinct_and_sensor_directions_orthonormal():
    assert signature_matrix(GRID)["all_distinct"]
    e = Residuals(GRID).evaluate(readings(steady_state(GRID, CENTRE)), CENTRE)
    D = e["D_sensor"]
    assert np.allclose(D.T @ D, np.eye(len(SENSORS)), atol=1e-6)
    iso = isolability(GRID, n_points=64)
    assert iso["sensor_directions_orthonormal"]
    assert iso["smallest_median_deg"] > 5.0
    # adjacent joints around bus 2 look alike when their currents meet at (or leave) the bus: the only such pair
    assert [(p["a"], p["b"]) for p in iso["confusable_pairs"]] == [("process:line_12", "process:line_23")]


@pytest.fixture(scope="module")
def diagnoser():
    d = Diagnoser(GRID, Surrogate(n_train=40).fit(GRID))
    return d.calibrate([run_plant(GRID, Scenario(f"cal_{i}", "normal", 900_000 + i, duration_s=240.0))
                        for i in range(4)])


@pytest.mark.parametrize("sc, kind, target", [
    (Scenario("joint", "process", 5, duration_s=240.0, onset_s=100.0, target="line_23", magnitude=0.15),
     "process", "line_23"),
    (Scenario("drift", "sensor", 6, duration_s=240.0, onset_s=100.0, target="I_12", magnitude=1.0, mode="drift"),
     "sensor", "I_12"),
    (Scenario("noise", "sensor", 7, duration_s=240.0, onset_s=100.0, target="V3", magnitude=6.0, mode="noise"),
     "sensor", "V3"),
    (Scenario("domain", "domain", 8, duration_s=240.0, onset_s=100.0, target="P_L4_kW", magnitude=34.0),
     "domain", "P_L4_kW"),
    (Scenario("healthy", "normal", 9, duration_s=240.0), "normal", ""),
])
def test_diagnosis_names_the_injected_fault(diagnoser, sc, kind, target):
    run = run_plant(GRID, sc)
    d = diagnoser.diagnose(run)
    assert (d["kind"], d.get("target", "")) == (kind, target)
    if kind == "process":
        assert abs(d["size"] - sc.magnitude) <= max(3 * d["size_std"], 0.2 * sc.magnitude)


def test_torch_residuals_match_numpy():
    torch = pytest.importorskip("torch")
    from pinneapple_systems.digital_twin.dc_microgrid.pinn import residual_scale, torch_residuals
    scale = residual_scale(GRID)
    rng = np.random.default_rng(0)
    U = np.array([[rng.uniform(*NORMAL_BOX[k]) for k in INPUTS] for _ in range(5)])
    Y = np.array([readings(steady_state(GRID, dict(zip(INPUTS, x, strict=True)))) for x in U])
    Y = Y + rng.normal(0, 0.5, Y.shape)
    ref = np.array([raw_residuals(GRID, y, dict(zip(INPUTS, x, strict=True))) / scale for x, y in zip(U, Y,
                                                                                                         strict=True)])
    got = torch_residuals(GRID, torch.as_tensor(U), torch.as_tensor(Y), torch.as_tensor(scale)).numpy()
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)
