"""A physics-informed fast model of the microgrid, and an equal-budget study against a data-only network and POD + GP.

The fast model maps the five operating inputs (PV power, resistive load, constant-power load, two droop set-points) to
the thirteen readings. The PINN is an MLP trained on ``N`` labelled reference solves *and* on the network's own laws
at unlabelled operating points: the thirteen residuals of :func:`.diagnose.raw_residuals` (Kirchhoff, Ohm, droop,
MPPT and load laws) evaluated on its predictions, in units of their noise standard deviation at the centre of the
envelope. With as many independent laws as readings, the laws alone determine the readings at every operating point,
so the labels only have to anchor the solution; the collocation box can be wider than the labelled one.

Training uses :func:`pinneapple_neural.trainer.graybox.train_graybox` with no unknown term (phase 1 on the labels,
phase 2 on labels + physics with the loss weights balanced by their running magnitudes); the data-only network is
the same call with the physics weight at zero, so both have the same architecture, epochs and labels.

:func:`surrogate_study` scores each fast model on held-out operating points inside the envelope and in the shell
around it (error in sensor sigmas), with the physics residual of its predictions, and measures whether the PINN's own
residual -- computable without any label -- ranks its true error (Spearman correlation).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .diagnose import RESIDUALS, Residuals, Surrogate, raw_residuals
from .plant import INPUTS, LINES, NORMAL_BOX, SENSORS, SIGMA, Microgrid, readings, steady_state

#: collocation box of the extended PINN and the region of the shell test points: each input widened by about half
#: its width on both sides, kept physical (positive powers and resistances)
EXTENDED_BOX = {"P_pv_kW": (0.5, 29.0), "R_L2": (5.0, 55.0), "P_L4_kW": (1.0, 35.0), "Vref1": (372.0, 388.0),
                "Vref3": (372.0, 388.0)}
MODELS = ("mlp_data", "pinn_box", "pinn_extended", "pod_gp")


def _box(b: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([b[k][0] for k in INPUTS]), np.array([b[k][1] for k in INPUTS])


def residual_scale(grid: Microgrid) -> np.ndarray:
    """Noise standard deviation of each residual at the centre of the envelope (its units, RESIDUAL_UNITS)."""
    u = {k: 0.5 * sum(NORMAL_BOX[k]) for k in INPUTS}
    return Residuals(grid).evaluate(readings(steady_state(grid, u)), u)["sigma"]


def torch_residuals(grid: Microgrid, U, Y, scale):
    """The healthy-network residuals of :func:`.diagnose.raw_residuals` on torch tensors, divided by ``scale``:
    ``U`` (n, 5) inputs in INPUTS order and units, ``Y`` (n, 13) readings in SENSORS order."""
    import torch
    V1, V2, V3, V4, IS1, IPV, IS3, IR2, IL4, I12, I23, I34, I41 = Y.unbind(1)
    ppv, rl2, pl4, vr1, vr3 = U.unbind(1)
    R = {k: grid.r_line(k) for k in LINES}
    r = torch.stack([
        IS1 - I12 + I41, IPV - IR2 + I12 - I23, IS3 + I23 - I34, -IL4 + I34 - I41,
        V1 - V2 - R["12"] * I12, V2 - V3 - R["23"] * I23, V3 - V4 - R["34"] * I34, V4 - V1 - R["41"] * I41,
        V1 - vr1 + grid.rd1 * IS1, V3 - vr3 + grid.rd3 * IS3,
        IPV - grid.eta_pv * ppv * 1e3 / V2, IR2 - V2 / rl2, IL4 - pl4 * 1e3 / V4], 1)
    return r / scale


@dataclass
class MLPSurrogate:
    """A trained MLP with the normalisation the trainer used; ``predict`` takes and returns physical units."""
    net: Any
    cmin: Any
    crange: Any
    vmean: Any
    vstd: Any
    kind: str
    history: list
    seconds: float

    def predict(self, U: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            x = (torch.as_tensor(np.atleast_2d(U), dtype=torch.float32) - self.cmin) / self.crange
            return (self.net(x) * self.vstd + self.vmean).double().numpy()


def fit_mlp(grid: Microgrid, U: np.ndarray, Y: np.ndarray, *, physics: bool, box: dict | None = None,
            epochs: tuple[int, int] = (500, 2500), width: int = 64, depth: int = 3, n_colloc: int = 512,
            seed: int = 0) -> MLPSurrogate:
    """Train the fast model on ``(U, Y)`` labelled reference solves; ``physics``: add the residual loss at
    ``n_colloc`` operating points drawn each epoch from ``box`` (default NORMAL_BOX)."""
    import torch

    from pinneapple_neural.trainer.graybox import GrayBoxConfig, GrayBoxNet, train_graybox
    torch.manual_seed(seed)
    net = GrayBoxNet(n_in=len(INPUTS), n_out=len(SENSORS), width=width, depth=depth)
    obs_c = torch.as_tensor(U, dtype=torch.float32)
    obs_v = torch.as_tensor(Y, dtype=torch.float32)
    cmin = obs_c.min(0).values                                      # the trainer's own normalisation
    crange = (obs_c.max(0).values - cmin).clamp(min=1e-6)
    vmean = obs_v.mean(0, keepdim=True)
    vstd = obs_v.std(0, keepdim=True).clamp(min=1e-6)
    lo, hi = (torch.as_tensor(a, dtype=torch.float32) for a in _box(box or NORMAL_BOX))
    scale = torch.as_tensor(residual_scale(grid), dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed + 1)

    def sampler():
        return lo + torch.rand((n_colloc, len(INPUTS)), generator=gen) * (hi - lo)

    def residual(model, coll, learnable, known, nets):
        return torch_residuals(grid, coll, model((coll - cmin) / crange) * vstd + vmean, scale)

    cfg = GrayBoxConfig(epochs_supervised=epochs[0], epochs_joint=epochs[1], w_pde=1.0 if physics else 0.0,
                        device="cpu", log_every=max(1, epochs[1] // 50))
    t0 = time.perf_counter()
    net, _, _, hist = train_graybox(net, {}, residual, obs_c, obs_v, sampler, config=cfg)
    net.eval()
    kind = ("pinn" if physics else "mlp_data")
    return MLPSurrogate(net, cmin, crange, vmean, vstd, kind, hist, time.perf_counter() - t0)


def test_sets(grid: Microgrid, n: int = 200, seed: int = 123) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Held-out operating points inside NORMAL_BOX and in the shell of EXTENDED_BOX outside it, with their exact
    readings."""
    from scipy.stats import qmc
    lo, hi = _box(NORMAL_BOX)
    U_in = lo + qmc.LatinHypercube(d=len(INPUTS), seed=seed).random(n) * (hi - lo)
    elo, ehi = _box(EXTENDED_BOX)
    rng = np.random.default_rng(seed)
    U_out = []
    while len(U_out) < n:
        x = elo + rng.random(len(INPUTS)) * (ehi - elo)
        if np.any((x < lo) | (x > hi)):
            U_out.append(x)
    U_out = np.array(U_out)

    def exact(U):
        return np.array([readings(steady_state(grid, dict(zip(INPUTS, x, strict=True)))) for x in U])

    return {"inside": (U_in, exact(U_in)), "shell": (U_out, exact(U_out))}


def score(grid: Microgrid, predict, U: np.ndarray, Y: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    """Error in sensor sigmas (RMS over sensors per point: median, 95th percentile, max) and the RMS physics residual
    of the predictions (residual sigmas) per point."""
    P = predict(U)
    err = np.sqrt(np.mean(((P - Y) / SIGMA) ** 2, 1))
    res = np.array([np.sqrt(np.mean((raw_residuals(grid, p, dict(zip(INPUTS, x, strict=True))) / scale) ** 2))
                    for p, x in zip(P, U, strict=True)])
    return {"err_median": float(np.median(err)), "err_p95": float(np.quantile(err, 0.95)), "err_max": float(err.max()),
            "residual_median": float(np.median(res)), "residual_p95": float(np.quantile(res, 0.95)),
            "_err": err, "_res": res}


def _train_job(args):
    import torch
    torch.set_num_threads(1)
    grid, U, Y, model, seed, epochs = args
    box = EXTENDED_BOX if model == "pinn_extended" else NORMAL_BOX
    return fit_mlp(grid, U, Y, physics=model != "mlp_data", box=box, seed=seed, epochs=epochs)


def surrogate_study(grid: Microgrid | None = None, *, budgets: tuple[int, ...] = (8, 16, 32, 64),
                    seeds: tuple[int, ...] = (0, 1, 2), epochs: tuple[int, int] = (500, 5000), n_test: int = 200,
                    workers: int = 4, log=None) -> dict[str, Any]:
    """Equal-budget comparison of the fast models: for each number of labelled reference solves and seed, the same
    labels for every model; scores inside the envelope and in the shell around it."""
    from scipy.stats import qmc, spearmanr
    grid = grid or Microgrid()
    scale = residual_scale(grid)
    tests = test_sets(grid, n_test)
    lo, hi = _box(NORMAL_BOX)
    jobs, meta, labelled = [], [], {}
    for n in budgets:
        for s in seeds:
            U = lo + qmc.LatinHypercube(d=len(INPUTS), seed=1000 + 17 * s + n).random(n) * (hi - lo)
            Y = np.array([readings(steady_state(grid, dict(zip(INPUTS, x, strict=True)))) for x in U])
            labelled[(n, s)] = (U, Y)
            for m in MODELS[:3]:
                jobs.append((grid, U, Y, m, s, epochs))
                meta.append((n, s, m))
    t0 = time.perf_counter()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            fitted = pool.map(_train_job, jobs)
    else:
        fitted = [_train_job(j) for j in jobs]
    t_train = time.perf_counter() - t0
    rows = []
    rank = {}
    for (n, s, m), f in zip(meta, fitted, strict=True):
        row = {"n_labelled": n, "seed": s, "model": m, "train_s": f.seconds}
        for k, (U, Y) in tests.items():
            sc = score(grid, f.predict, U, Y, scale)
            if m == "pinn_extended" and k == "shell" or m == "pinn_box" and k == "inside":
                rank[(m, k, n, s)] = float(spearmanr(sc["_res"], sc["_err"]).statistic)
            row[k] = {kk: v for kk, v in sc.items() if not kk.startswith("_")}
        rows.append(row)
        if log:
            log(f"{m:14s} N={n:3d} seed={s}: inside {row['inside']['err_median']:.2f} sigma, shell "
                f"{row['shell']['err_median']:.2f} sigma ({f.seconds:.1f} s)")
    for n in budgets:
        for s in seeds:
            U, Y = labelled[(n, s)]
            gp = Surrogate().fit(grid, U, Y)
            row = {"n_labelled": n, "seed": s, "model": "pod_gp", "train_s": None}
            for k, (Ut, Yt) in tests.items():
                row[k] = {kk: v for kk, v in score(grid, lambda X, gp=gp: gp.predict(X)[0], Ut, Yt, scale).items()
                          if not kk.startswith("_")}
            rows.append(row)
    summary = {}
    for m in MODELS:
        for n in budgets:
            rs = [r for r in rows if r["model"] == m and r["n_labelled"] == n]
            summary[f"{m}@{n}"] = {k: {q: float(np.median([r[k][q] for r in rs])) for q in
                                       ("err_median", "err_p95", "residual_median")} for k in tests}
    # cost of one evaluation: the Newton solve of the reference against the trained network (batch of one)
    x = dict(zip(INPUTS, 0.5 * (lo + hi), strict=True))
    t1 = time.perf_counter()
    for _ in range(200):
        steady_state(grid, x)
    t_newton = (time.perf_counter() - t1) / 200
    f0 = fitted[0]
    X1 = np.atleast_2d(0.5 * (lo + hi))
    t1 = time.perf_counter()
    for _ in range(200):
        f0.predict(X1)
    t_mlp = (time.perf_counter() - t1) / 200
    return {"budgets": list(budgets), "seeds": list(seeds), "epochs": list(epochs), "n_test": n_test,
            "models": list(MODELS), "rows": rows, "summary": summary,
            "spearman_residual_vs_error": {f"{m}/{k}": {"median": float(np.median([v for kk, v in rank.items()
                                                                                  if kk[:2] == (m, k)])),
                                                        "min": float(np.min([v for kk, v in rank.items()
                                                                             if kk[:2] == (m, k)]))}
                                           for m, k in (("pinn_box", "inside"), ("pinn_extended", "shell"))},
            "seconds": {"training_total": t_train, "newton_solve": t_newton, "mlp_eval": t_mlp},
            "residuals": list(RESIDUALS), "extended_box": EXTENDED_BOX}
