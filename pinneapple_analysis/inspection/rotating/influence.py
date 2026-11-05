"""Calibrated unbalance inversion: the 1X response as a linear function of the unbalance mass, across speeds.

An unbalance mass ``m`` at radius ``e`` exerts ``m e w^2`` at the shaft speed ``w``; the structure answers through its
transfer function, so the 1X acceleration phasor of channel ``c`` (phase against the tachometer) is

    A_c(w, m) = alpha_c(w) m + beta_c(w)

with ``alpha_c(w) = -w^4 e H_c(w)`` the influence coefficient of a unit mass at the calibration angle and
``beta_c(w)`` the response of the healthy rotor (residual unbalance and any other 1X source). This is the
influence-coefficient model of rotor balancing. The radius ``e`` and the modal masses cannot be separated from the data
(only their product enters), so the scale is fixed by the known trial masses: the inversion returns an equivalent mass
in grams at the calibration radius and angle.

Three models of ``alpha`` and ``beta`` across speeds, fitted by least squares on runs of known mass:

* :class:`SplineInfluence` -- complex cubic B-splines in ``w`` (no physics beyond linearity in ``m``; interpolates,
  does not extrapolate);
* :class:`ModalInfluence` -- a linear dissipative structure: ``n`` modes shared by every channel and by ``alpha`` and
  ``beta``, ``H_c(w) = sum_j R_cj / (W_j^2 - w^2 + 2 i z_j W_j w)`` with ``W_j > 0`` and damping ``z_j > 0``, plus the
  usual residual terms of the modes outside the band (``w^2`` and ``w^4`` in acceleration). Poles by bounded nonlinear
  least squares with several starts, residues by linear least squares (variable projection);
* :class:`AmplitudeInfluence` -- the classical phase-free calibration ``|A_c| = a_c(w) + b_c(w) m`` (splines), the
  baseline of order analysis without a phase reference.

``estimate_mass`` inverts any of them for one run: the mass that brings the predicted phasors closest to the measured
ones, every channel weighted by the inverse of its residual variance on the calibration runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _bspline_design(w: np.ndarray, lo: float, hi: float, n_knots: int, degree: int = 3) -> np.ndarray:
    from scipy.interpolate import BSpline
    inner = np.linspace(lo, hi, n_knots)
    t = np.concatenate([[lo] * degree, inner, [hi] * degree])
    x = np.clip(np.asarray(w, float), lo, hi)
    return BSpline.design_matrix(x, t, degree).toarray()


def _weights(resid: np.ndarray) -> np.ndarray:
    v = np.mean(np.abs(resid) ** 2, 0)
    return 1.0 / np.maximum(v, 1e-12 * max(v.max(), 1e-300))


@dataclass
class SplineInfluence:
    """alpha(w), beta(w) as complex cubic B-splines on [lo, hi] (per channel); ``quadratic`` adds ``gamma(w) m^2`` to
    test the linearity in mass."""
    n_knots: int = 8
    ridge: float = 1e-8
    quadratic: bool = False
    lo: float = 0.0
    hi: float = 0.0
    coef: np.ndarray = field(default=None, repr=False)
    w_ch: np.ndarray = field(default=None, repr=False)

    def _design(self, w, m):
        B = _bspline_design(w, self.lo, self.hi, self.n_knots)
        cols = [B * m[:, None], B] + ([B * (m ** 2)[:, None]] if self.quadratic else [])
        return np.concatenate(cols, 1)

    def fit(self, w, m, A):
        w, m = np.asarray(w, float), np.asarray(m, float)
        self.lo, self.hi = float(w.min()), float(w.max())
        X = self._design(w, m)
        G = X.T @ X + self.ridge * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1])
        self.coef = np.linalg.solve(G, X.T @ A)
        self.w_ch = _weights(A - X @ self.coef)
        return self

    def alpha_beta(self, w):
        B = _bspline_design(np.atleast_1d(w), self.lo, self.hi, self.n_knots)
        k = B.shape[1]
        return B @ self.coef[:k], B @ self.coef[k:2 * k]

    def predict(self, w, m):
        return self._design(np.atleast_1d(np.asarray(w, float)), np.atleast_1d(np.asarray(m, float))) @ self.coef


@dataclass
class ModalInfluence:
    """Linear dissipative modal model shared by alpha and beta (see the module docstring)."""
    n_modes: int = 3
    band: tuple[float, float] = (5.0, 120.0)        # Hz, where the natural frequencies may lie
    zeta_bounds: tuple[float, float] = (0.005, 0.5)
    residual_terms: bool = True
    n_starts: int = 24
    seed: int = 0
    poles: np.ndarray = field(default=None, repr=False)   # (n_modes, 2): natural frequency (Hz), damping ratio
    coef: np.ndarray = field(default=None, repr=False)
    w_ch: np.ndarray = field(default=None, repr=False)
    cost: float = float("nan")

    def _basis(self, w_hz, poles):
        w = 2 * np.pi * np.asarray(w_hz, float)
        cols = []
        for fn, z in poles:
            W = 2 * np.pi * fn
            cols.append(-(w ** 4) / (W ** 2 - w ** 2 + 2j * z * W * w))
        if self.residual_terms:
            cols += [w ** 2, w ** 4]
        S = np.stack(cols, 1)
        return S / np.maximum(np.abs(S).max(0), 1e-300)          # scaled columns, residues absorb the scale

    def _design(self, w_hz, m, poles):
        S = self._basis(w_hz, poles)
        return np.concatenate([S * np.asarray(m, float)[:, None], S], 1)

    def _solve(self, w_hz, m, A, poles):
        X = self._design(w_hz, m, poles)
        coef, *_ = np.linalg.lstsq(X, A, rcond=None)
        return coef, A - X @ coef

    def fit(self, w_hz, m, A):
        from scipy.optimize import least_squares
        rng = np.random.default_rng(self.seed)
        n = self.n_modes
        lo = np.array([self.band[0]] * n + [self.zeta_bounds[0]] * n)
        hi = np.array([self.band[1]] * n + [self.zeta_bounds[1]] * n)
        scale = np.sqrt(np.mean(np.abs(A) ** 2, 0))

        def res(p):
            poles = np.stack([p[:n], p[n:]], 1)
            _, r = self._solve(w_hz, m, A, poles)
            r = r / scale
            return np.concatenate([r.real.ravel(), r.imag.ravel()])

        best = None
        for _ in range(self.n_starts):
            p0 = np.concatenate([np.sort(rng.uniform(*self.band, n)), np.exp(rng.uniform(np.log(0.01), np.log(0.2), n))])
            sol = least_squares(res, p0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
            if best is None or sol.cost < best.cost:
                best = sol
        self.poles = np.stack([best.x[:n], best.x[n:]], 1)
        order = np.argsort(self.poles[:, 0])
        self.poles = self.poles[order]
        self.coef, r = self._solve(w_hz, m, A, self.poles)
        self.w_ch = _weights(r)
        self.cost = float(best.cost)
        return self

    def predict(self, w_hz, m):
        return self._design(np.atleast_1d(w_hz), np.atleast_1d(m), self.poles) @ self.coef

    def alpha_beta(self, w_hz):
        S = self._basis(np.atleast_1d(w_hz), self.poles)
        k = S.shape[1]
        return S @ self.coef[:k], S @ self.coef[k:]


@dataclass
class AmplitudeInfluence:
    """Phase-free classical calibration: ``|A_c| = a_c(w) + b_c(w) m`` with cubic B-splines in ``w``.

    Outside the calibrated speed range, ``extrapolate`` says how ``a`` and ``b`` continue: ``"hold"`` (their value at
    the edge), ``"physics"`` (the power laws of an unbalance response in acceleration: ``w^4`` below the resonances,
    where the structure is stiffness-controlled, ``w^2`` above them, where it is mass-controlled) or ``"logslope"``
    (the log-log slope of the fitted curve over the outer ``edge_frac`` of the range, clipped to [-2, 6])."""
    n_knots: int = 8
    extrapolate: str = "hold"
    edge_frac: float = 0.15
    lo: float = 0.0
    hi: float = 0.0
    coef: np.ndarray = field(default=None, repr=False)
    w_ch: np.ndarray = field(default=None, repr=False)

    def _edge_exponents(self):
        if self.extrapolate == "physics":
            return 4.0, 2.0
        span = self.hi - self.lo
        w_lo = np.array([self.lo, self.lo + self.edge_frac * span])
        w_hi = np.array([self.hi - self.edge_frac * span, self.hi])

        def slope(ww):
            a, b = self._raw(ww)
            out = []
            for v in (a, b):
                r = np.abs(v[1]) / np.maximum(np.abs(v[0]), 1e-300)
                out.append(np.clip(np.log(np.maximum(r, 1e-300)) / np.log(ww[1] / ww[0]), -2.0, 6.0))
            return out

        return slope(w_lo), slope(w_hi)

    def _raw(self, w):
        B = _bspline_design(np.atleast_1d(w), self.lo, self.hi, self.n_knots)
        k = B.shape[1]
        return B @ self.coef[:k], B @ self.coef[k:]

    def fit(self, w, m, A):
        w, m = np.asarray(w, float), np.asarray(m, float)
        self.lo, self.hi = float(w.min()), float(w.max())
        B = _bspline_design(w, self.lo, self.hi, self.n_knots)
        X = np.concatenate([B * m[:, None], B], 1)
        Y = np.abs(A)
        self.coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        self.w_ch = _weights(Y - X @ self.coef)
        return self

    def alpha_beta(self, w):
        w = np.atleast_1d(np.asarray(w, float))
        a, b = self._raw(w)
        if self.extrapolate == "hold":
            return a, b
        below, above = (w < self.lo)[:, None], (w > self.hi)[:, None]
        if self.extrapolate == "physics":
            p_lo, p_hi = 4.0, 2.0
            f_lo = (w / self.lo)[:, None] ** p_lo
            f_hi = (w / self.hi)[:, None] ** p_hi
            return (np.where(below, a * f_lo, np.where(above, a * f_hi, a)),
                    np.where(below, b * f_lo, np.where(above, b * f_hi, b)))
        (pa_lo, pb_lo), (pa_hi, pb_hi) = self._edge_exponents()
        r_lo, r_hi = (w / self.lo)[:, None], (w / self.hi)[:, None]
        a = np.where(below, a * r_lo ** pa_lo[None, :], np.where(above, a * r_hi ** pa_hi[None, :], a))
        b = np.where(below, b * r_lo ** pb_lo[None, :], np.where(above, b * r_hi ** pb_hi[None, :], b))
        return a, b


def estimate_mass(model, w, A, *, phase_free: bool | None = None) -> np.ndarray:
    """Equivalent unbalance mass (g) of each run: weighted least squares over the channels of
    ``A_c = alpha_c(w) m + beta_c(w)`` (complex), or of ``|A_c| = a_c + b_c m`` for :class:`AmplitudeInfluence`."""
    w = np.atleast_1d(np.asarray(w, float))
    A = np.atleast_2d(A)
    a, b = model.alpha_beta(w)
    wc = model.w_ch[None, :]
    if phase_free if phase_free is not None else isinstance(model, AmplitudeInfluence):
        y = np.abs(A) - b
        return np.sum(wc * a * y, 1) / np.maximum(np.sum(wc * a * a, 1), 1e-300)
    return np.real(np.sum(wc * np.conj(a) * (A - b), 1)) / np.maximum(np.sum(wc * np.abs(a) ** 2, 1), 1e-300)


def nearest_level(m_hat: np.ndarray, levels) -> np.ndarray:
    levels = np.asarray(sorted(levels), float)
    return levels[np.argmin(np.abs(np.asarray(m_hat)[:, None] - levels[None, :]), 1)]


def level_unbalance(w, m, A, n_knots: int = 8, iters: int = 60) -> dict:
    """Descriptive fit with one free complex unbalance per known level: ``A = alpha(w) c_level + beta(w)``, ``c_0 = 0``
    and the largest level fixed real at its nominal mass (the scale). ``|c_level|`` against the nominal mass tests the
    magnitude physics; ``arg c_level`` shows whether the 1X phase reference is the same for every level."""
    w, m = np.asarray(w, float), np.asarray(m, float)
    B = _bspline_design(w, w.min(), w.max(), n_knots)
    k = B.shape[1]
    lv = sorted(set(m))
    top = lv[-1]
    c = {x: complex(x) for x in lv}
    for _ in range(iters):
        cm = np.array([c[x] for x in m])
        coef, *_ = np.linalg.lstsq(np.concatenate([B * cm[:, None], B], 1), A, rcond=None)
        al, be = B @ coef[:k], B @ coef[k:]
        for x in lv[1:]:
            s = m == x
            c[x] = np.sum(np.conj(al[s]) * (A[s] - be[s])) / np.sum(np.abs(al[s]) ** 2)
        ref = c[top]
        c = {x: (v / ref * top if x else 0j) for x, v in c.items()}
    cm = np.array([c[x] for x in m])
    X = np.concatenate([B * cm[:, None], B], 1)
    coef, *_ = np.linalg.lstsq(X, A, rcond=None)
    r = A - X @ coef
    return {"levels": lv, "magnitude": [abs(c[x]) for x in lv], "angle_deg": [float(np.degrees(np.angle(c[x])))
                                                                             for x in lv],
            "R2": float(1 - np.sum(np.abs(r) ** 2) / np.sum(np.abs(A - A.mean(0)) ** 2))}


_PHI = np.linspace(-np.pi, np.pi, 721)[:-1]


@dataclass
class PhaseNuisanceInfluence:
    """The influence model with a global 1X phase that may change between recording sessions, estimated per level on
    the calibration runs and searched per run at inversion. ``kind``: ``"mass_angle"`` (the trial mass sits at a
    different angle: ``A = alpha m e^{i phi} + beta``) or ``"tach_reference"`` (the tachometer reference moved:
    ``A = e^{i phi} (alpha m + beta)``). ``backend``: ``"spline"`` or a fitted-on-rotated-data :class:`ModalInfluence`
    (``"modal3"``, ``"modal4"``; tach_reference only)."""
    kind: str = "tach_reference"
    backend: str = "spline"
    n_knots: int = 8
    iters: int = 40
    phi: dict = field(default_factory=dict)
    model: object = field(default=None, repr=False)
    w_ch: np.ndarray = field(default=None, repr=False)

    def _spline(self):
        return SplineInfluence(n_knots=self.n_knots, ridge=0.0)

    def fit(self, w, m, A):
        w, m = np.asarray(w, float), np.asarray(m, float)
        lv = sorted(set(m))
        self.phi = {x: 0.0 for x in lv}
        sp = self._spline()
        for _ in range(self.iters):
            e = np.exp(1j * np.array([self.phi[x] for x in m]))[:, None]
            if self.kind == "mass_angle":
                sp.lo, sp.hi = float(w.min()), float(w.max())
                B = _bspline_design(w, sp.lo, sp.hi, sp.n_knots)
                kk = B.shape[1]
                coef, *_ = np.linalg.lstsq(np.concatenate([B * (m[:, None] * e), B], 1), A, rcond=None)
                sp.coef = coef
                al, be = B @ coef[:kk], B @ coef[kk:]
            else:
                sp.fit(w, m, A * np.conj(e))
                al, be = sp.alpha_beta(w)
            for x in lv:
                if (self.kind == "mass_angle" and x in (0.0, lv[-1])) or (self.kind != "mass_angle" and x == lv[0]):
                    continue
                s = m == x
                z = (np.sum(np.conj(al[s] * x) * (A[s] - be[s])) if self.kind == "mass_angle"
                     else np.sum(np.conj(al[s] * x + be[s]) * A[s]))
                self.phi[x] = float(np.angle(z))
        e = np.exp(1j * np.array([self.phi[x] for x in m]))[:, None]
        if self.backend == "spline":
            self.model = sp
            pred = al * m[:, None] * e + be if self.kind == "mass_angle" else e * (al * m[:, None] + be)
        else:
            if self.kind == "mass_angle":
                raise ValueError("a modal backend is implemented for kind='tach_reference' only")
            self.model = ModalInfluence(n_modes=int(self.backend[-1])).fit(w, m, A * np.conj(e))
            al, be = self.model.alpha_beta(w)
            pred = e * (al * m[:, None] + be)
        self.w_ch = _weights(A - pred)
        return self

    def alpha_beta(self, w):
        return self.model.alpha_beta(w)

    def estimate(self, w, A) -> np.ndarray:
        w = np.atleast_1d(np.asarray(w, float))
        A = np.atleast_2d(A)
        al, be = self.alpha_beta(w)
        wc = self.w_ch[None, :]
        if self.kind == "mass_angle":
            z = np.sum(wc * np.conj(al) * (A - be), 1) / np.maximum(np.sum(wc * np.abs(al) ** 2, 1), 1e-300)
            return np.abs(z)
        out = np.empty(len(w))
        for i in range(len(w)):
            y = A[i][None, :] * np.exp(-1j * _PHI)[:, None] - be[i][None, :]
            den = np.maximum(np.sum(wc * np.abs(al[i]) ** 2), 1e-300)
            mm = np.clip(np.real(np.sum(wc * np.conj(al[i])[None, :] * y, 1)) / den, 0.0, None)
            out[i] = mm[np.argmin(np.sum(wc * np.abs(y - mm[:, None] * al[i][None, :]) ** 2, 1))]
        return out


def channel_estimates(model, w, A) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mass estimates of each run and their weights (inverse variance of each estimate): phase-free
    ``(|A_c| - a_c) / b_c`` for :class:`AmplitudeInfluence`, the complex projection otherwise."""
    w = np.atleast_1d(np.asarray(w, float))
    A = np.atleast_2d(A)
    a, b = model.alpha_beta(w)
    wc = model.w_ch[None, :]
    if isinstance(model, AmplitudeInfluence):
        est = (np.abs(A) - b) / np.where(np.abs(a) > 1e-300, a, 1e-300)
        weight = wc * a * a
    else:
        est = np.real(np.conj(a) * (A - b)) / np.maximum(np.abs(a) ** 2, 1e-300)
        weight = wc * np.abs(a) ** 2
    return est, weight


def estimate_mass_robust(model, w, A, k: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Channel-robust mass estimate: the weighted median of the per-channel estimates (one mass, many channels: a
    channel that disagrees with the others is not believed), and a flag per channel that lies more than ``k`` robust
    standard deviations of its own estimate from that median (a sensor, cable or mounting problem, not unbalance)."""
    est, weight = channel_estimates(model, w, A)
    out = np.empty(len(est))
    for i in range(len(est)):
        o = np.argsort(est[i])
        cw = np.cumsum(weight[i][o])
        out[i] = est[i][o][np.searchsorted(cw, 0.5 * cw[-1])]
    sd = 1.0 / np.sqrt(np.maximum(weight, 1e-300))
    flags = np.abs(est - out[:, None]) > k * np.maximum(sd, 1e-12)
    return out, flags


@dataclass
class ScreenedEstimator:
    """A channel-robust mass estimate. Every channel gives its own estimate with a weight ``W_c`` (its sensitivity at
    that speed squared over its calibration residual variance: the terms of the joint least squares of
    :func:`estimate_mass`); cross-validation on the calibration runs measures the spread ``s_c`` of each channel's
    standardised error ``(est_c - m) sqrt(W_c)``, and the biweight constant ``c`` is set to the ``quantile`` of the
    absolute standardised errors ``|est_c - m| sqrt(W_c) / s_c`` over all channels (0.5 % false rejection on the
    calibration sessions: their errors are heavier-tailed than a normal). At inversion the common mass is the
    Tukey-biweight M-estimate of the channel estimates (distances standardised by ``sqrt(W_c) / s_c``, weights
    ``W_c``), iterated from their unweighted median, so one very sensitive but faulty channel cannot become the
    centre; a channel at a standardised distance of ``c`` or more gets zero weight and is flagged (a sensor, cable or
    mounting problem: the mass is the same for every channel). With every channel well inside ``c`` the estimate is
    close to the joint one."""
    make: object = None                         # callable returning an unfitted influence model
    quantile: float = 0.995
    c: float | None = None                      # None: calibrated at ``quantile``; a number: used as given
    n_folds: int = 10
    iters: int = 30
    seed: int = 0
    model: object = field(default=None, repr=False)
    scale: np.ndarray = field(default=None, repr=False)
    c_: float = field(default=float("nan"), repr=False)

    def fit(self, w, m, A):
        make = self.make or AmplitudeInfluence
        rng = np.random.default_rng(self.seed)
        z = np.zeros(np.atleast_2d(A).shape)
        for te in np.array_split(rng.permutation(len(m)), self.n_folds):
            tr = np.setdiff1d(np.arange(len(m)), te)
            est, W = channel_estimates(make().fit(w[tr], m[tr], A[tr]), w[te], A[te])
            z[te] = (est - m[te][:, None]) * np.sqrt(W)
        self.scale = np.maximum(1.4826 * np.median(np.abs(z - np.median(z, 0)), 0), 1e-12)
        self.c_ = float(np.quantile(np.abs(z) / self.scale[None, :], self.quantile)) if self.c is None else float(self.c)
        self.model = make().fit(w, m, A)
        return self

    def estimate(self, w, A) -> tuple[np.ndarray, np.ndarray]:
        est, W = channel_estimates(self.model, w, A)
        P = W / self.scale[None, :] ** 2
        out, flags = np.empty(len(est)), np.zeros(est.shape, bool)
        for i in range(len(est)):
            mu = np.median(est[i])
            for _ in range(self.iters):
                u = (est[i] - mu) * np.sqrt(P[i]) / self.c_
                h = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
                if h.sum() == 0:
                    break
                new = np.sum(h * W[i] * est[i]) / np.sum(h * W[i])
                if abs(new - mu) < 1e-9:
                    mu = new
                    break
                mu = new
            flags[i] = np.abs(est[i] - mu) * np.sqrt(P[i]) >= self.c_
            out[i] = mu
        return out, flags
