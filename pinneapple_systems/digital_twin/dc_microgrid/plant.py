"""The plant: a four-bus DC microgrid in a ring, its reference steady-state simulator and controlled faults.

    bus 1: battery converter, voltage droop      V1 = Vref1 - Rd1 I_S1
    bus 2: PV boost converter in MPPT             V2 I_PV = eta P_pv   (constant power) + resistive load R_L2
    bus 3: second converter, voltage droop       V3 = Vref3 - Rd3 I_S3
    bus 4: constant-power load (EV charger)       V4 I_L4 = P_L4
    lines 1-2, 2-3, 3-4, 4-1: two-conductor copper cables, loop resistance R = 2 r' L

Quasi-steady: the converters' voltage and current loops settle in milliseconds (an assumption stated by
:func:`quasi_steady_valid`), the twin samples once a second and the operator moves are slower still, so every sample
is a steady state of the network: the four bus voltages solve Kirchhoff's current law with the droop, MPPT and load
laws (Newton, analytic Jacobian).

Thirteen sensors (bus voltages in V, currents in A) with Gaussian noise. Faults go on the *truth* (an extra
resistance in a line -- a loose or corroded joint; a lower efficiency of the PV converter; a higher output resistance
of a droop converter) or on the *readings* (a drifting bias, a noisier sensor), so every test case has a known answer.

Cable data: IEC 60228, class 2 copper conductor of 50 mm2, maximum DC resistance 0.387 ohm/km at 20 °C. Converter
droop gains, PV converter efficiency, sensor noise and the operating envelope are assumptions of this demo, not the
data of a measured installation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

#: IEC 60228, class 2 copper, 50 mm2: maximum DC resistance at 20 °C (ohm/km)
R_PER_KM_50MM2_CU = 0.387

LINES = ("12", "23", "34", "41")
_ENDS = {"12": (0, 1), "23": (1, 2), "34": (2, 3), "41": (3, 0)}
SENSORS = ("V1", "V2", "V3", "V4", "I_S1", "I_PV", "I_S3", "I_R2", "I_L4", "I_12", "I_23", "I_34", "I_41")
UNITS = {s: ("V" if s.startswith("V") else "A") for s in SENSORS}
INPUTS = ("P_pv_kW", "R_L2", "P_L4_kW", "Vref1", "Vref3")
INPUT_UNITS = {"P_pv_kW": "kW", "R_L2": "ohm", "P_L4_kW": "kW", "Vref1": "V", "Vref3": "V"}
#: operating envelope the twin's fast model is trained on (the grid can be driven outside it)
NORMAL_BOX = {"P_pv_kW": (2.0, 20.0), "R_L2": (10.0, 40.0), "P_L4_kW": (5.0, 25.0), "Vref1": (376.0, 384.0),
              "Vref3": (376.0, 384.0)}
#: one-sigma sensor noise: 0.4 V (0.1 % of a 400 V span) and 0.1 A (0.1 % of a 100 A span) -- an assumption of the
#: demo for Hall-effect transducers after anti-alias filtering, not a datasheet value
SIGMA = np.array([0.4] * 4 + [0.1] * 9)


@dataclass
class Microgrid:
    """Network and converter parameters (SI)."""
    lengths_m: dict[str, float] = field(default_factory=lambda: {"12": 150.0, "23": 200.0, "34": 120.0, "41": 180.0})
    r_per_km: float = R_PER_KM_50MM2_CU
    rd1: float = 0.25            # ohm, droop of the battery converter (about 4 % at 60 A)
    rd3: float = 0.35            # ohm, droop of the second converter
    eta_pv: float = 0.97         # PV boost converter efficiency at its operating point
    v_nominal: float = 380.0
    settling_s: float = 0.02     # closed-loop settling time of the converters (assumption, see quasi_steady_valid)

    def r_line(self, line: str) -> float:
        """Loop (go and return) resistance of a line, ohm."""
        return 2.0 * self.r_per_km * self.lengths_m[line] / 1000.0


@dataclass
class Faults:
    """What is wrong in the network at one instant (all zero: healthy)."""
    d_r: dict[str, float] = field(default_factory=dict)    # line -> extra resistance (ohm)
    d_eta: float = 0.0                                      # efficiency lost by the PV converter
    d_rd: dict[str, float] = field(default_factory=dict)    # "1" / "3" -> extra output resistance (ohm)


#: process faults the twin can name, the fault parameter each sets, and the size used for its direction
COMPONENTS = ("line_12", "line_23", "line_34", "line_41", "pv_converter", "droop_1", "droop_3")
UNIT_FAULT = {"line_12": 0.1, "line_23": 0.1, "line_34": 0.1, "line_41": 0.1, "pv_converter": 0.01,
              "droop_1": 0.1, "droop_3": 0.1}


def component_faults(name: str, size: float) -> Faults:
    """The :class:`Faults` of one named process fault of a given size."""
    if name.startswith("line_"):
        return Faults(d_r={name[5:]: size})
    if name == "pv_converter":
        return Faults(d_eta=size)
    return Faults(d_rd={name[-1]: size})


def effective(grid: Microgrid, faults: Faults | None) -> dict[str, Any]:
    """Line resistances, droop gains and PV efficiency with the faults applied."""
    f = faults or Faults()
    return {"R": {k: grid.r_line(k) + f.d_r.get(k, 0.0) for k in LINES},
            "rd1": grid.rd1 + f.d_rd.get("1", 0.0), "rd3": grid.rd3 + f.d_rd.get("3", 0.0),
            "eta": grid.eta_pv - f.d_eta}


def _laplacian(R: dict[str, float]) -> np.ndarray:
    L = np.zeros((4, 4))
    for k, (i, j) in _ENDS.items():
        g = 1.0 / R[k]
        L[i, i] += g
        L[j, j] += g
        L[i, j] -= g
        L[j, i] -= g
    return L


def steady_state(grid: Microgrid, u: dict[str, float], faults: Faults | None = None) -> dict[str, float]:
    """True state of the network for inputs ``u`` (keys of INPUTS): bus voltages, every current and power, and the
    two solver checks (largest KCL mismatch in A, energy balance in W)."""
    p = effective(grid, faults)
    Lap = _laplacian(p["R"])
    ppv, pl4, rl2 = u["P_pv_kW"] * 1e3, u["P_L4_kW"] * 1e3, u["R_L2"]
    vr1, vr3 = u["Vref1"], u["Vref3"]

    def injections(V):
        return np.array([(vr1 - V[0]) / p["rd1"], p["eta"] * ppv / V[1] - V[1] / rl2, (vr3 - V[2]) / p["rd3"],
                         -pl4 / V[3]])

    V = np.full(4, grid.v_nominal)
    for _ in range(50):
        F = injections(V) - Lap @ V
        dinj = np.diag([-1.0 / p["rd1"], -p["eta"] * ppv / V[1] ** 2 - 1.0 / rl2, -1.0 / p["rd3"], pl4 / V[3] ** 2])
        step = np.linalg.solve(dinj - Lap, -F)
        V = V + step
        if np.max(np.abs(step)) < 1e-11:
            break
    kcl = float(np.max(np.abs(injections(V) - Lap @ V)))
    i_line = {k: (V[i] - V[j]) / p["R"][k] for k, (i, j) in _ENDS.items()}
    s = {"V1": V[0], "V2": V[1], "V3": V[2], "V4": V[3], "I_S1": (vr1 - V[0]) / p["rd1"],
         "I_PV": p["eta"] * ppv / V[1], "I_S3": (vr3 - V[2]) / p["rd3"], "I_R2": V[1] / rl2, "I_L4": pl4 / V[3],
         **{f"I_{k}": v for k, v in i_line.items()}}
    p_src = s["V1"] * s["I_S1"] + s["V2"] * s["I_PV"] + s["V3"] * s["I_S3"]
    p_load = s["V2"] * s["I_R2"] + s["V4"] * s["I_L4"]
    p_loss = sum(i_line[k] ** 2 * p["R"][k] for k in LINES)
    s.update(P_sources_W=p_src, P_loads_W=p_load, P_line_loss_W=p_loss, kcl_residual_A=kcl,
             energy_balance_W=p_src - p_load - p_loss, v_min=float(V.min()), v_max=float(V.max()))
    return {k: float(v) for k, v in s.items()}


def readings(state: dict[str, float]) -> np.ndarray:
    """True sensor values in sensor units (V, A), order SENSORS."""
    return np.array([state[s] for s in SENSORS])


def quasi_steady_valid(grid: Microgrid, dt_s: float, factor: float = 10.0) -> bool:
    """Each sample is a steady state when the sampling period spans many closed-loop settling times."""
    return dt_s >= factor * grid.settling_s


# ----------------------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    """A controlled test case: an operating history and (optionally) one fault with its onset and ramp."""
    name: str
    kind: str                      # "normal" | "process" | "sensor" | "domain"
    seed: int
    duration_s: float = 360.0
    dt_s: float = 1.0
    onset_s: float = 180.0
    ramp_s: float = 30.0
    target: str = ""               # component (process), sensor (sensor), input (domain)
    magnitude: float = 0.0         # fault size, bias in sensor units, noise gain, or input value reached
    mode: str = ""                 # sensor: "drift" | "noise"

    def truth(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "magnitude": self.magnitude, "mode": self.mode,
                "onset_s": self.onset_s}


def operating_history(sc: Scenario) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Smooth load, PV and set-point moves inside NORMAL_BOX (a few low-frequency sines per input, seeded); a domain
    scenario then drives its ``target`` input to ``magnitude`` (outside the box) from the onset."""
    rng = np.random.default_rng(sc.seed)
    t = np.arange(0.0, sc.duration_s + 1e-9, sc.dt_s)
    u: dict[str, np.ndarray] = {}
    for k in INPUTS:
        lo, hi = NORMAL_BOX[k]
        mid, amp = 0.5 * (lo + hi), 0.5 * (hi - lo)
        s = np.zeros_like(t)
        for _ in range(3):
            period = rng.uniform(90.0, 400.0)
            s += rng.uniform(0.2, 0.45) * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
        u[k] = mid + amp * np.clip(s, -0.95, 0.95)
    if sc.kind == "domain":
        r = np.clip((t - sc.onset_s) / sc.ramp_s, 0.0, 1.0)
        u[sc.target] = u[sc.target] + r * (sc.magnitude - u[sc.target])
    return t, u


def run_plant(grid: Microgrid, sc: Scenario) -> dict[str, Any]:
    """Simulate one scenario: inputs, true sensor values, noisy readings, and the solver checks over the run."""
    t, u = operating_history(sc)
    rng = np.random.default_rng(sc.seed + 7919)
    n = len(SENSORS)
    Y_true, Y = np.empty((len(t), n)), np.empty((len(t), n))
    kcl = energy = 0.0
    vmin, vmax = math.inf, -math.inf
    for i, ti in enumerate(t):
        r = float(np.clip((ti - sc.onset_s) / sc.ramp_s, 0.0, 1.0)) if sc.kind in ("process", "sensor") else 0.0
        faults = component_faults(sc.target, r * sc.magnitude) if sc.kind == "process" else None
        st = steady_state(grid, {k: float(v[i]) for k, v in u.items()}, faults)
        kcl, energy = max(kcl, st["kcl_residual_A"]), max(energy, abs(st["energy_balance_W"]))
        vmin, vmax = min(vmin, st["v_min"]), max(vmax, st["v_max"])
        y = readings(st)
        Y_true[i] = y
        sig = SIGMA.copy()
        bias = np.zeros(n)
        if sc.kind == "sensor":
            j = SENSORS.index(sc.target)
            if sc.mode == "drift":
                bias[j] = r * sc.magnitude
            else:
                sig[j] *= 1.0 + r * (sc.magnitude - 1.0)
        Y[i] = y + bias + rng.normal(0.0, 1.0, n) * sig
    return {"t": t, "u": u, "y_true": Y_true, "y": Y, "scenario": sc, "kcl_max_A": kcl, "energy_balance_max_W": energy,
            "v_range": (vmin, vmax)}


def with_grid(grid: Microgrid, **kw) -> Microgrid:
    return replace(grid, **kw)
