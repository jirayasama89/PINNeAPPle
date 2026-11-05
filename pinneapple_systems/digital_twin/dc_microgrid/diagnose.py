"""How the microgrid twin knows what is wrong: whitened physics residuals, a generalised likelihood ratio per fault
hypothesis, a Kalman filter on the fault parameters, a fast model and its domain of validity.

1. **Physics residuals** (:class:`Residuals`), from the readings and the known set-points alone: Kirchhoff's current
   law at the four buses, Ohm's law on the four lines, the two droop laws, the MPPT power of the PV converter, the
   resistive and the constant-power load. Each residual involves a few sensors and at most one component, so every
   fault moves the residual vector along its own *direction*: a biased sensor along the column of the Jacobian for
   that sensor; an extra resistance in line 2-3 along the Ohm residual of that line only.
2. **Whitening.** The residual covariance ``S = J diag(sigma^2) J^T`` is propagated from the sensor noise (``J`` the
   Jacobian with respect to the readings); with ``S = L L^T`` the whitened residual ``z = L^-1 r`` is N(0, I) when
   healthy. With one residual per sensor and an invertible ``J``, the whitened directions of the thirteen sensor biases
   are orthonormal; a process fault is isolable from every single-sensor fault when its direction is not collinear
   with any of them (:func:`isolability` measures the angles over the operating envelope).
3. **Generalised likelihood ratio** over a window of ``W`` samples, with the directions of each sample (they move
   with the operating point): for hypothesis ``h`` of constant size ``a``, ``a* = sum d.z / sum d.d`` and the
   log-likelihood gain ``(sum d.z)^2 / sum d.d``, divided by the variance of ``z`` along ``d`` over the window when it
   exceeds one (the likelihood ratio of a mean shift with unknown variance, so a noisier sensor does not pass for a
   bias). The largest gain is the detection statistic of a bias-type fault (process faults are one-sided: a joint
   only adds resistance, a converter only loses efficiency). A noise increase on sensor ``j`` shows as a variance
   excess of the projection of ``z`` on its direction.
4. **Kalman filter** (:class:`FaultTracker`, :class:`pinneapple_analysis.state_estimation.kalman.ExtendedKalmanFilter`)
   tracks the seven fault parameters as random walks from the measured line drops, PV current and droop drops, so a
   process change comes with its size and uncertainty.
5. **Fast model and domain of validity** (:class:`Surrogate`): a POD + Gaussian-process model of the thirteen readings
   over the operating envelope; its innovations say "the network does not do what I predict", and the inputs against
   the training box say where the fast model may not be used (the physics residuals hold everywhere).

:class:`Diagnoser` turns the evidence into one of four answers -- normal, process changed (which line or
converter, and how much), sensor wrong (which sensor, bias or noise), fast model outside its domain (which input) --
with thresholds calibrated on healthy runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .plant import (
    COMPONENTS,
    INPUTS,
    LINES,
    NORMAL_BOX,
    SENSORS,
    SIGMA,
    UNIT_FAULT,
    Faults,
    Microgrid,
    component_faults,
    effective,
    readings,
    steady_state,
)

RESIDUALS = ("kcl_1", "kcl_2", "kcl_3", "kcl_4", "ohm_12", "ohm_23", "ohm_34", "ohm_41", "droop_1", "droop_3",
             "pv_mppt", "load_R2", "load_cpl4")
RESIDUAL_UNITS = {r: ("V" if r.startswith(("ohm", "droop")) else "A") for r in RESIDUALS}
_IX = {s: i for i, s in enumerate(SENSORS)}


def raw_residuals(grid: Microgrid, y: np.ndarray, u: dict[str, float], faults: Faults | None = None) -> np.ndarray:
    """The thirteen residuals of one reading vector (order RESIDUALS), zero for noise-free readings of a network that
    has exactly ``faults``."""
    p = effective(grid, faults)
    V1, V2, V3, V4, IS1, IPV, IS3, IR2, IL4, I12, I23, I34, I41 = (y[_IX[s]] for s in SENSORS)
    R = p["R"]
    return np.array([
        IS1 - I12 + I41,
        IPV - IR2 + I12 - I23,
        IS3 + I23 - I34,
        -IL4 + I34 - I41,
        V1 - V2 - R["12"] * I12,
        V2 - V3 - R["23"] * I23,
        V3 - V4 - R["34"] * I34,
        V4 - V1 - R["41"] * I41,
        V1 - u["Vref1"] + p["rd1"] * IS1,
        V3 - u["Vref3"] + p["rd3"] * IS3,
        IPV - p["eta"] * u["P_pv_kW"] * 1e3 / V2,
        IR2 - V2 / u["R_L2"],
        IL4 - u["P_L4_kW"] * 1e3 / V4])


def signature_matrix(grid: Microgrid) -> dict[str, Any]:
    """Structural signatures: which residuals each sensor and each process fault enters (boolean, from the
    Jacobians at the centre of the envelope)."""
    u = {k: 0.5 * sum(NORMAL_BOX[k]) for k in INPUTS}
    y = readings(steady_state(grid, u))
    res = Residuals(grid)
    J = res.jacobian(y, u)
    sens = {s: [RESIDUALS[i] for i in np.flatnonzero(np.abs(J[:, j]) > 1e-9)] for j, s in enumerate(SENSORS)}
    r0 = raw_residuals(grid, y, u)
    comp = {c: [RESIDUALS[i] for i in np.flatnonzero(np.abs(r0 - raw_residuals(grid, y, u, component_faults(
        c, UNIT_FAULT[c]))) > 1e-9)] for c in COMPONENTS}
    keys = [tuple(v) for v in list(sens.values()) + list(comp.values())]
    return {"sensors": sens, "components": comp, "all_distinct": len(set(keys)) == len(keys)}


class Residuals:
    """Physics residuals of one reading vector, their propagated covariance, the whitened residual and the whitened
    directions of every fault hypothesis."""

    def __init__(self, grid: Microgrid):
        self.grid = grid

    def raw(self, y: np.ndarray, u: dict[str, float], faults: Faults | None = None) -> np.ndarray:
        return raw_residuals(self.grid, y, u, faults)

    def jacobian(self, y: np.ndarray, u: dict[str, float]) -> np.ndarray:
        """d residual / d reading (13 x 13), central differences of a tenth of each sensor's sigma."""
        J = np.empty((len(RESIDUALS), len(SENSORS)))
        for j in range(len(SENSORS)):
            dy = np.zeros(len(SENSORS))
            dy[j] = 0.1 * SIGMA[j]
            J[:, j] = (self.raw(y + dy, u) - self.raw(y - dy, u)) / (2 * dy[j])
        return J

    def evaluate(self, y: np.ndarray, u: dict[str, float]) -> dict[str, np.ndarray]:
        """Residuals, their covariance, the whitened residual ``z`` and the whitened directions: columns of
        ``D_sensor`` a 1-sigma bias of each sensor, columns of ``D_comp`` a fault of size UNIT_FAULT in each
        component, as seen by the healthy model."""
        r = self.raw(y, u)
        J = self.jacobian(y, u)
        S = (J * SIGMA ** 2) @ J.T
        L = np.linalg.cholesky(S + 1e-15 * np.eye(len(r)))
        z = np.linalg.solve(L, r)
        D_sensor = np.linalg.solve(L, J * SIGMA[None, :])
        dc = np.stack([r - self.raw(y, u, component_faults(c, UNIT_FAULT[c])) for c in COMPONENTS], 1)
        return {"r": r, "sigma": np.sqrt(np.diag(S)), "z": z, "D_sensor": D_sensor, "D_comp": np.linalg.solve(L, dc)}


HYPOTHESES = [f"sensor:{s}" for s in SENSORS] + [f"process:{c}" for c in COMPONENTS]


def isolability(grid: Microgrid, n_points: int = 64, seed: int = 0, confusable_deg: float = 5.0) -> dict[str, Any]:
    """Angles (degrees) between the whitened directions of every pair of fault hypotheses over ``n_points`` operating
    points of the envelope: for each pair the median and the smallest angle; 0 means two faults look the same.

    The angle follows the decision rule. A sensor bias has either sign, so its direction and its opposite are the
    same hypothesis: the angle is folded to [0, 90]. Two process faults are one-sided (a joint only adds resistance,
    a converter only loses efficiency), so a fault that looks like the *opposite* of another one is told apart by the
    sign of its fitted size: the angle stays in [0, 180]. ``unsigned_*`` gives the folded angle of every pair (what a
    two-sided test would see), and ``confusable_fraction`` the share of operating points where a pair is closer than
    ``confusable_deg``."""
    from scipy.stats import qmc
    lo = np.array([NORMAL_BOX[k][0] for k in INPUTS])
    hi = np.array([NORMAL_BOX[k][1] for k in INPUTS])
    U = lo + qmc.LatinHypercube(d=len(INPUTS), seed=seed).random(n_points) * (hi - lo)
    res = Residuals(grid)
    one_sided = np.array([h.startswith("process") for h in HYPOTHESES])
    both = one_sided[:, None] & one_sided[None, :]
    angles, unsigned, norms, gram = [], [], [], []
    for x in U:
        u = dict(zip(INPUTS, x, strict=True))
        e = res.evaluate(readings(steady_state(grid, u)), u)
        D = np.concatenate([e["D_sensor"], e["D_comp"]], 1)
        n = np.linalg.norm(D, axis=0)
        norms.append(n)
        Dn = D / np.maximum(n, 1e-300)
        C = np.clip(Dn.T @ Dn, -1.0, 1.0)
        gram.append(e["D_sensor"].T @ e["D_sensor"])
        unsigned.append(np.degrees(np.arccos(np.abs(C))))
        angles.append(np.where(both, np.degrees(np.arccos(C)), unsigned[-1]))
    A, Au = np.array(angles), np.array(unsigned)
    med, mn, conf = np.median(A, 0), A.min(0), (A < confusable_deg).mean(0)
    medu = np.median(Au, 0)
    pairs = []
    for i in range(len(HYPOTHESES)):
        for j in range(i + 1, len(HYPOTHESES)):
            pairs.append({"a": HYPOTHESES[i], "b": HYPOTHESES[j], "median_deg": float(med[i, j]),
                          "min_deg": float(mn[i, j]), "unsigned_median_deg": float(medu[i, j]),
                          "confusable_fraction": float(conf[i, j])})
    pairs.sort(key=lambda p: p["median_deg"])
    pairs_u = sorted(pairs, key=lambda p: p["unsigned_median_deg"])
    N = np.array(norms)
    eye = np.eye(len(SENSORS))
    return {"hypotheses": HYPOTHESES, "median_deg": med.tolist(), "min_deg": mn.tolist(),
            "unsigned_median_deg": medu.tolist(), "closest_pairs": pairs[:8], "closest_pairs_unsigned": pairs_u[:8],
            "smallest_median_deg": pairs[0]["median_deg"], "smallest_unsigned_median_deg":
                pairs_u[0]["unsigned_median_deg"],
            "largest_confusable_fraction": float(max(p["confusable_fraction"] for p in pairs)),
            "confusable_pairs": sorted([p for p in pairs if p["confusable_fraction"] > 0],
                                       key=lambda p: -p["confusable_fraction"]),
            "confusable_deg": confusable_deg,
            "sensor_directions_orthonormal": bool(all(np.allclose(G, eye, atol=1e-6) for G in gram)),
            "detectability_median": dict(zip(HYPOTHESES, np.median(N, 0).tolist(), strict=True)),
            "detectability_min": dict(zip(HYPOTHESES, N.min(0).tolist(), strict=True)), "n_points": n_points}


# ----------------------------------------------------------------------------- fast model
@dataclass
class Surrogate:
    """POD + Gaussian-process model of the thirteen readings over NORMAL_BOX (the twin's fast model)."""
    n_train: int = 80
    seed: int = 0
    model: Any = field(default=None, repr=False)
    lo: np.ndarray = field(default=None, repr=False)
    hi: np.ndarray = field(default=None, repr=False)

    def fit(self, grid: Microgrid, U: np.ndarray | None = None, Y: np.ndarray | None = None) -> Surrogate:
        from scipy.stats import qmc

        from pinneapple_neural.architectures.rom.parametric import ParametricPOD
        self.lo = np.array([NORMAL_BOX[k][0] for k in INPUTS])
        self.hi = np.array([NORMAL_BOX[k][1] for k in INPUTS])
        if U is None:
            U = self.lo + qmc.LatinHypercube(d=len(INPUTS), seed=self.seed).random(self.n_train) * (self.hi - self.lo)
        if Y is None:
            Y = np.array([readings(steady_state(grid, dict(zip(INPUTS, x, strict=True)))) for x in U])
        self._scale = Y.std(0) + 1e-12
        self._mean = Y.mean(0)
        self.model = ParametricPOD(r=len(SENSORS), energy=None, regressor="gpr").fit(U, (Y - self._mean) / self._scale)
        self.U_, self.Y_ = U, Y
        return self

    def predict(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m, s = self.model.predict(np.atleast_2d(U), return_std=True)
        return m * self._scale + self._mean, s * self._scale

    def excess(self, u: np.ndarray) -> float:
        """How far the inputs are outside the training box, in box widths (0 inside)."""
        w = self.hi - self.lo
        return float(np.max(np.maximum(np.maximum(self.lo - u, u - self.hi), 0.0) / w))


# ----------------------------------------------------------------------------- Kalman filter
class FaultTracker:
    """EKF on the seven fault parameters (four extra line resistances, the PV efficiency loss, two extra droop
    resistances), random walks, observed through the measured line drops, PV current and droop drops."""

    def __init__(self, grid: Microgrid, q_walk: float = 0.004, x0_sigma: float = 0.2):
        from pinneapple_analysis.state_estimation.kalman import ExtendedKalmanFilter
        self.grid = grid
        self._ctx: dict[str, float] = {}
        sv, si = SIGMA[0], SIGMA[4]
        Rn = np.diag([2 * sv ** 2 + (grid.r_line(k) * si) ** 2 for k in LINES]
                     + [si ** 2, sv ** 2 + (grid.rd1 * si) ** 2, sv ** 2 + (grid.rd3 * si) ** 2])
        scale = np.array([1, 1, 1, 1, 0.1, 1, 1.0])          # efficiency is ten times smaller than an ohm here
        self.ekf = ExtendedKalmanFilter(7, 7, f=lambda x: x, h=self._h, Q=np.diag((q_walk * scale) ** 2), R=Rn,
                                        F_jac=lambda x: np.eye(7), H_jac=self._H)
        self.ekf.initialize(np.zeros(7), np.diag((x0_sigma * scale) ** 2))

    def _h(self, x):
        c, g = self._ctx, self.grid
        return np.array([(g.r_line(k) + x[i]) * c[f"I_{k}"] for i, k in enumerate(LINES)]
                        + [(g.eta_pv - x[4]) * c["P_pv"] / c["V2"], (g.rd1 + x[5]) * c["I_S1"],
                           (g.rd3 + x[6]) * c["I_S3"]])

    def _H(self, x):
        c = self._ctx
        return np.diag([c["I_12"], c["I_23"], c["I_34"], c["I_41"], -c["P_pv"] / c["V2"], c["I_S1"], c["I_S3"]])

    def step(self, y: np.ndarray, u: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        m = dict(zip(SENSORS, y, strict=True))
        self._ctx = {**m, "P_pv": u["P_pv_kW"] * 1e3}
        obs = np.array([m["V1"] - m["V2"], m["V2"] - m["V3"], m["V3"] - m["V4"], m["V4"] - m["V1"], m["I_PV"],
                        u["Vref1"] - m["V1"], u["Vref3"] - m["V3"]])
        out = self.ekf.step(obs)
        return out["x"].copy(), np.sqrt(np.diag(out["P"]))


# ----------------------------------------------------------------------------- decision
@dataclass
class Diagnoser:
    """Windowed evidence -> hypothesis. ``window`` samples; thresholds from :meth:`calibrate` (healthy runs)."""
    grid: Microgrid
    surrogate: Surrogate
    window: int = 30
    thresholds: dict[str, float] = field(default_factory=dict)
    ood_threshold: float = 0.02             # 2 % of a box width outside the training envelope

    def evidence(self, run: dict[str, Any]) -> dict[str, np.ndarray]:
        """Per-sample evidence over a run (output of :func:`.plant.run_plant`)."""
        t, Y = run["t"], run["y"]
        U = np.stack([run["u"][k] for k in INPUTS], 1)
        res = Residuals(self.grid)
        Z, Ds, Dc, R = [], [], [], []
        for i in range(len(t)):
            e = res.evaluate(Y[i], dict(zip(INPUTS, U[i], strict=True)))
            Z.append(e["z"])
            Ds.append(e["D_sensor"])
            Dc.append(e["D_comp"])
            R.append(e["r"] / e["sigma"])
        Yp, Sp = self.surrogate.predict(U)
        innov = (Y - Yp) / np.sqrt(SIGMA ** 2 + Sp ** 2)
        kf = FaultTracker(self.grid)
        K, Ks = zip(*(kf.step(Y[i], dict(zip(INPUTS, U[i], strict=True))) for i in range(len(t))), strict=True)
        return {"t": t, "z": np.array(Z), "r_over_sigma": np.array(R), "D_sensor": np.array(Ds), "D_comp": np.array(Dc),
                "innovation": innov, "surrogate_pred": Yp, "surrogate_std": Sp,
                "ood": np.array([self.surrogate.excess(u) for u in U]), "faults": np.array(K),
                "faults_std": np.array(Ks)}

    def _glr(self, z: np.ndarray, D: np.ndarray, one_sided: bool) -> tuple[np.ndarray, np.ndarray]:
        """Gain and size per hypothesis over a window: z (W, m), D (W, m, H)."""
        num = np.einsum("wm,wmh->h", z, D)
        den = np.maximum(np.einsum("wmh,wmh->h", D, D), 1e-300)
        a = num / den
        gain = num ** 2 / den
        if one_sided:
            gain = np.where(a > 0, gain, 0.0)
        return gain, a

    def window_stats(self, ev: dict[str, np.ndarray], i: int) -> dict[str, Any]:
        """Statistics of the window ending at sample ``i``."""
        W = self.window
        sl = slice(max(0, i - W + 1), i + 1)
        z = ev["z"][sl]
        gs, as_ = self._glr(z, ev["D_sensor"][sl], one_sided=False)
        gc, ac = self._glr(z, ev["D_comp"][sl], one_sided=True)
        size = np.concatenate([as_, ac])

        def noise_var(D):
            """Variance of z along each normalised direction over the window, left after the fitted fault: the
            projection is ``a |d_t| + noise``, and ``|d_t|`` moves with the operating point (a joint's signature
            scales with its line current), so the fault itself must not count as noise."""
            n = np.linalg.norm(D, axis=1)                                  # (W, H)
            p = np.einsum("wm,wmh->wh", z, D / np.maximum(n, 1e-300)[:, None, :])
            a = (p * n).sum(0) / np.maximum((n * n).sum(0), 1e-300)
            return ((p - a * n) ** 2).sum(0) / (len(z) - 1) if len(z) > 2 else np.ones(D.shape[2])

        vs, vc = noise_var(ev["D_sensor"][sl]), noise_var(ev["D_comp"][sl])
        # mean shift against the variance seen in the window (unknown-variance likelihood ratio): a noisier sensor
        # widens the spread of its projection without moving its mean, so it does not pass for a bias
        gain = np.concatenate([gs / np.maximum(vs, 1.0), gc / np.maximum(vc, 1.0)])
        spread = vs - 1.0
        inn = ev["innovation"][sl]
        return {"shift": float(gain.max()), "gain": gain, "size": size,
                "spread": float(spread.max()), "spread_per_sensor": spread,
                "model": float(len(inn) * np.sum(inn.mean(0) ** 2)), "ood": float(ev["ood"][i])}

    def statistics(self, ev: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rows = [self.window_stats(ev, i) for i in range(len(ev["t"]))]
        return {k: np.array([r[k] for r in rows]) for k in ("shift", "spread", "model", "ood")}

    def calibrate(self, runs: list[dict[str, Any]], quantile: float = 1.0, margin: float = 1.25) -> Diagnoser:
        """Thresholds = ``margin`` x the ``quantile`` of each statistic over healthy runs (after the first window)."""
        acc: dict[str, list] = {"shift": [], "spread": [], "model": []}
        for r in runs:
            s = self.statistics(self.evidence(r))
            for k in acc:
                acc[k].append(s[k][self.window:])
        self.thresholds = {k: margin * float(np.quantile(np.concatenate(v), quantile)) for k, v in acc.items()}
        return self

    def classify(self, ev: dict[str, np.ndarray], i: int, ws: dict[str, Any] | None = None) -> str:
        ws = ws or self.window_stats(ev, i)
        th = self.thresholds
        if ws["ood"] > self.ood_threshold:
            return "domain"
        if i < self.window:
            return "normal"
        if ws["shift"] > th["shift"]:
            return "shift"
        if ws["spread"] > th["spread"]:
            return "spread"
        if ws["model"] > th["model"]:
            return "model"
        return "normal"

    def diagnose(self, run: dict[str, Any], ev: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        """The diagnosis at every sample and the final verdict: kind, target, size, first alarm, next action."""
        ev = ev or self.evidence(run)
        n = len(ev["t"])
        stats = [self.window_stats(ev, i) for i in range(n)]
        kinds = [self.classify(ev, i, stats[i]) for i in range(n)]
        alarm = [i for i, k in enumerate(kinds) if k != "normal"]
        first = alarm[0] if alarm else None
        i = n - 1
        ws, k = stats[i], kinds[i]
        out: dict[str, Any] = {"first_alarm_s": None if first is None else float(ev["t"][first]),
                               "statistics_final": {s: ws[s] for s in ("shift", "spread", "model", "ood")},
                               "thresholds": dict(self.thresholds), "kinds": kinds, "evidence": ev,
                               "stats": {s: np.array([w[s] for w in stats]) for s in ("shift", "spread", "model",
                                                                                     "ood")}}
        if k == "normal":
            out.update(kind="normal", target="", size=None, action="none: physics, fast model and domain agree")
        elif k == "domain":
            x = np.array([float(run["u"][kk][i]) for kk in INPUTS])
            w = self.surrogate.hi - self.surrogate.lo
            j = int(np.argmax(np.maximum(np.maximum(self.surrogate.lo - x, x - self.surrogate.hi), 0) / w))
            out.update(kind="domain", target=INPUTS[j], size=float(x[j]),
                       action=f"run the reference simulation at {INPUTS[j]} = {x[j]:.4g} and add it to the fast "
                              "model; until then trust the physics residuals, not the fast model")
        elif k == "model":
            out.update(kind="domain", target="inside the envelope", size=None,
                       action="the fast model disagrees with physics-consistent readings: run the reference here and "
                              "refit")
        elif k == "shift":
            h = HYPOTHESES[int(np.argmax(ws["gain"]))]
            a = float(ws["size"][HYPOTHESES.index(h)])
            kind, target = h.split(":")
            out["gains"] = dict(zip(HYPOTHESES, ws["gain"].tolist(), strict=True))
            if kind == "process":
                j = COMPONENTS.index(target)
                fx, fs = float(ev["faults"][i, j]), float(ev["faults_std"][i, j])
                unit = "ohm" if target != "pv_converter" else "(efficiency)"
                out.update(kind="process", target=target, size=fx, size_std=fs, size_glr=a * UNIT_FAULT[target],
                           action=f"{target}: {fx:.3g} +- {2 * fs:.2g} {unit} (Kalman); "
                                  + ("inspect the joints and terminations of the line" if target.startswith("line")
                                     else "inspect the converter (power stage, output filter, control set-up)"))
            else:
                j = SENSORS.index(target)
                out.update(kind="sensor", target=target, mode="drift", size=float(a * SIGMA[j]),
                           action=f"{target}: bias {a * SIGMA[j]:+.3g} {('V' if target[0] == 'V' else 'A')} -- check / "
                                  "recalibrate it; meanwhile estimate it from the other sensors")
        else:
            j = int(np.argmax(ws["spread_per_sensor"]))
            target = SENSORS[j]
            g = float(np.sqrt(1.0 + max(ws["spread_per_sensor"][j], 0.0)))
            out.update(kind="sensor", target=target, mode="noise", size=g,
                       action=f"{target}: noise x{g:.2g} -- check the wiring, shielding and grounding of the transducer")
        return out
