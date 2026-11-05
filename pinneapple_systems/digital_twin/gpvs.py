"""Physics residuals on the GPVS-Faults laboratory data: a changed process or a wrong sensor?

GPVS-Faults (Bakdi, Guichi, Mekhilef, Bounoua; Mendeley Data, DOI 10.17632/n76t439f65.1, CC BY 4.0) records a
grid-connected PV system in the laboratory (PV array, boost converter, DC link, three-phase inverter, grid): 16 runs
``Fxy`` (``x`` = 0 healthy, 1-7 the faults; ``y`` = L limited-power / IPPT, M maximum-power / MPPT), the fault
introduced halfway. Columns: Time, Ipv, Vpv, Vdc, ia, ib, ic, va, vb, vc, Iabc, If, Vabc, Vf.

Residuals per grid cycle (``window`` samples):

* ``R1 = rms(ia + ib + ic) / rms(i)`` -- Kirchhoff's current law of the three-wire output: involves the current
  sensors only;
* ``R2 = rms(va + vb + vc) / rms(v)`` -- phase voltages of a three-wire system sum to zero: voltage sensors only;
  the phase that best explains ``va + vb + vc`` (least squares on each phase voltage) points at the sensor;
* ``R3`` -- DC-link power balance ``Vpv Ipv - sum(v i) - C d(Vdc^2 / 2)/dt - (a + b Vpv Ipv)``, with ``C, a, b``
  fitted on the first half of a healthy run.

A residual that involves only sensors and breaks says "a sensor is wrong"; a process fault (an IGBT, the array, the
controller, the grid) changes the measured variables but keeps these laws. :func:`process_shift` measures the change
of the process variables against the first part of the same run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FAULTS = {0: "healthy", 1: "inverter (IGBT)", 2: "feedback sensor (one phase voltage)", 3: "grid voltage fluctuations",
          4: "partial shading", 5: "15 % open circuit of the array", 6: "boost PI gain", 7: "boost PI time constant"}
KIND = {0: "normal", 1: "process", 2: "sensor", 3: "process", 4: "process", 5: "process", 6: "process", 7: "process"}
COLUMNS = ("Time", "Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc", "Iabc", "If", "Vabc", "Vf")
PROCESS_VARS = ("Ipv", "Vpv", "Vdc", "i_rms", "v_rms", "p_ac")


def read(path: str | Path) -> dict[str, np.ndarray]:
    import pandas as pd
    d = pd.read_csv(path)
    return {c: d[c].to_numpy(float) for c in COLUMNS}


def windows(run: dict[str, np.ndarray], window: int = 200) -> dict[str, np.ndarray]:
    """Per-window residuals and process variables (one window ~ one 50 Hz cycle at the CSV's ~10 kHz)."""
    n = len(run["Time"]) // window
    cut = n * window

    def W(x):
        return x[:cut].reshape(n, window)

    ia, ib, ic, va, vb, vc = (W(run[k]) for k in ("ia", "ib", "ic", "va", "vb", "vc"))
    i_rms = np.sqrt((ia ** 2 + ib ** 2 + ic ** 2).mean(1) / 3)
    v_rms = np.sqrt((va ** 2 + vb ** 2 + vc ** 2).mean(1) / 3)
    si, sv = ia + ib + ic, va + vb + vc
    R1 = np.sqrt((si ** 2).mean(1)) / np.maximum(i_rms, 1e-9)
    R2 = np.sqrt((sv ** 2).mean(1)) / np.maximum(v_rms, 1e-9)
    # which phase voltage explains the voltage sum best: least squares sv ~ g_k v_k, residual fraction per phase
    fit = []
    for v in (va, vb, vc):
        g = (sv * v).sum(1) / np.maximum((v * v).sum(1), 1e-12)
        fit.append(((sv - g[:, None] * v) ** 2).sum(1) / np.maximum((sv ** 2).sum(1), 1e-12))
    fit = np.stack(fit, 1)
    # balanced current control at about unity power factor: i_k ~ y v_k with one admittance y for the three phases;
    # the phase that departs most from the common y is the suspect current sensor
    ys = np.stack([(i * v).sum(1) / np.maximum((v * v).sum(1), 1e-12) for i, v in ((ia, va), (ib, vb), (ic, vc))], 1)
    y = np.median(ys, 1)[:, None]
    adm = np.stack([np.sqrt(((i - y * v) ** 2).mean(1)) for i, v in zip((ia, ib, ic), (va, vb, vc), strict=True)], 1)
    adm = adm / np.maximum(i_rms, 1e-9)[:, None]
    t = W(run["Time"]).mean(1)
    Vdc = W(run["Vdc"]).mean(1)
    p_pv = (W(run["Vpv"]) * W(run["Ipv"])).mean(1)
    p_ac = (va * ia + vb * ib + vc * ic).mean(1)
    dt = np.gradient(t)
    dE = np.gradient(0.5 * Vdc ** 2) / np.maximum(dt, 1e-9)
    return {"t": t, "R1": R1, "R2": R2, "phase_fit": fit, "admittance_residual": adm, "p_pv": p_pv, "p_ac": p_ac, "dE": dE,
            "Ipv": W(run["Ipv"]).mean(1), "Vpv": W(run["Vpv"]).mean(1), "Vdc": Vdc, "i_rms": i_rms, "v_rms": v_rms}


@dataclass
class PowerBalance:
    """``p_pv - p_ac = C dE + a + b p_pv`` fitted by least squares on healthy windows."""
    coef: np.ndarray = field(default=None)

    def fit(self, w: dict[str, np.ndarray], sel) -> PowerBalance:
        X = np.stack([w["dE"][sel], np.ones(sel.sum()), w["p_pv"][sel]], 1)
        self.coef, *_ = np.linalg.lstsq(X, (w["p_pv"] - w["p_ac"])[sel], rcond=None)
        return self

    def residual(self, w: dict[str, np.ndarray]) -> np.ndarray:
        C, a, b = self.coef
        return w["p_pv"] - w["p_ac"] - C * w["dE"] - a - b * w["p_pv"]


def process_shift(w: dict[str, np.ndarray], ref, test) -> dict[str, float]:
    """Robust z-score of the median of each process variable after the onset against its spread before it."""
    out = {}
    for k in PROCESS_VARS:
        x = w[k]
        mad = 1.4826 * np.median(np.abs(x[ref] - np.median(x[ref])))
        out[k] = float((np.median(x[test]) - np.median(x[ref])) / max(mad, 1e-9))
    return out


def run_pilot(folder: str | Path, window: int = 200, margin: float = 1.25, pre: float = 0.40, post: float = 0.55
              ) -> dict[str, Any]:
    """The pre-registered GPVS pilot (docs/research/q1_research_program.md): thresholds from the first ``pre`` of
    F0L and F0M, then every file's residuals before ``pre`` (false alarms) and after ``post`` (the faulty part)."""
    folder = Path(folder)
    W = {p.stem: windows(read(p), window) for p in sorted(folder.glob("F[0-7][LM].csv"))}

    def part(w, lo, hi):
        u = (w["t"] - w["t"][0]) / (w["t"][-1] - w["t"][0])
        return (u >= lo) & (u < hi)

    th = {k: margin * max(float(np.max(W[f][k][part(W[f], 0, pre)])) for f in ("F0L", "F0M")) for k in ("R1", "R2")}
    pb = {}
    for f in ("F0L", "F0M"):
        h1 = part(W[f], 0, 0.5)
        m = PowerBalance().fit(W[f], h1)
        r = m.residual(W[f])
        h2 = ~h1
        pb[f[-1]] = {"model": m, "check_median_abs_over_ppv": float(np.median(np.abs(r[h2]) / np.abs(W[f]["p_pv"][h2])))}
    files = {}
    for f, w in W.items():
        x, y = int(f[1]), f[2]
        before, after = part(w, 0, pre), part(w, post, 1.0)
        r3 = pb[y]["model"].residual(w)
        phase = np.argmin(w["phase_fit"], 1)
        frac = {k: float(np.mean(w[k][after] > th[k])) for k in ("R1", "R2")}
        files[f] = {"fault": FAULTS[x], "kind": KIND[x], "mode": "IPPT" if y == "L" else "MPPT",
                    "false_alarm_windows": {k: int(np.sum(w[k][before] > th[k])) for k in ("R1", "R2")},
                    "after_exceed_fraction": frac,
                    "R2_after_median": float(np.median(w["R2"][after])), "R1_after_median": float(np.median(w["R1"][after])),
                    "R2_phase_mode_after": "abc"[int(np.bincount(phase[after][w["R2"][after] > th["R2"]], minlength=3).argmax())]
                    if np.any(w["R2"][after] > th["R2"]) else None,
                    "R3_after_median_over_ppv": float(np.median(np.abs(r3[after]) / np.maximum(np.abs(w["p_pv"][after]), 1e-9))),
                    "R3_before_median_over_ppv": float(np.median(np.abs(r3[before]) / np.maximum(np.abs(w["p_pv"][before]), 1e-9))),
                    "process_shift_z": process_shift(w, before, after), "n_windows": int(len(w["t"]))}
    # exploratory (added after the first run, see the research log): residuals only while the inverter delivers
    # current (a shutdown is a process state, not a sensor fault), and the suspect current phase
    i_ref = float(np.median(np.concatenate([W[f]["i_rms"][part(W[f], 0, pre)] for f in ("F0L", "F0M")])))
    for f, w in W.items():
        on = w["i_rms"] > 0.3 * i_ref
        before, after = part(w, 0, pre) & on, part(w, post, 1.0) & on
        hit = after & (w["R1"] > th["R1"])
        files[f]["exploratory"] = {
            "operating_share_after": float(np.mean(on[part(w, post, 1.0)])),
            "R1_false_alarm_windows_operating": int(np.sum(w["R1"][before] > th["R1"])),
            "R1_after_exceed_fraction_operating": float(np.mean(w["R1"][after] > th["R1"])) if after.any() else None,
            "suspect_current_phase": "abc"[int(np.bincount(np.argmax(w["admittance_residual"][hit], 1),
                                                           minlength=3).argmax())] if hit.any() else None}
    sensor_detected = {f: files[f]["after_exceed_fraction"]["R2"] > 0.5 for f in ("F2L", "F2M") if f in files}
    proc = [f for f in files if KIND[int(f[1])] == "process"]
    g1 = all(sensor_detected.values()) and len({files[f]["R2_phase_mode_after"] for f in sensor_detected}) == 1
    g2_quiet = all(max(files[f]["after_exceed_fraction"].values()) < 0.05 for f in proc)
    g2_visible = {f: max(abs(v) for v in files[f]["process_shift_z"].values()) > 5 for f in proc}
    g3 = all(sum(v["false_alarm_windows"].values()) == 0 for v in files.values())
    g4 = all(v["check_median_abs_over_ppv"] < 0.05 for v in pb.values())
    return {"thresholds": th, "files": files,
            "power_balance": {k: {"C_a_b": v["model"].coef.tolist(), "check": v["check_median_abs_over_ppv"]}
                              for k, v in pb.items()},
            "criteria": {"G1_sensor_fault_detected_same_phase": g1, "G2_process_faults_quiet_residuals": g2_quiet,
                         "G2_process_faults_visible": g2_visible, "G3_no_false_alarm": g3, "G4_power_balance_ok": g4}}
