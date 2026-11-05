"""Pilot study: can a calibrated dynamic inversion of the 1X response recover the unbalance masses of MAFAULDA, at seen
and unseen speeds, better than classical phase-free order analysis and a data-driven regressor?

Tests (criteria in :data:`CRITERIA`, fixed before the first run):

* **T1 linearity** -- held-out R2 of the 1X phasors (six accelerometers) for ``A = alpha(w) m + beta(w)`` against the
  same model with a ``m^2`` term (10-fold by run);
* **T2 seen speeds** -- 10-fold by run, mass error of every model;
* **T3 unseen speeds** -- six contiguous speed bands, leave one band out: the four inner bands measure interpolation in
  speed, the two outer ones extrapolation;
* **level unbalance** -- one free complex unbalance per level (descriptive): its magnitude against the nominal mass
  (the magnitude physics) and its angle (whether the phase reference is the same for every level);
* exploratory, added after the first run showed the phase drift: the phase-nuisance models and the modal model fitted
  on re-phased data (:class:`.influence.PhaseNuisanceInfluence`). They are labelled as such in the report.

Every file is one 5 s run and one sample; no window of a run is ever split between training and test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .influence import (
    AmplitudeInfluence,
    ModalInfluence,
    PhaseNuisanceInfluence,
    ScreenedEstimator,
    SplineInfluence,
    estimate_mass,
    estimate_mass_robust,
    level_unbalance,
    nearest_level,
)

#: biweight constant of the screened estimator, selected on the development set (calibration sessions + underhang)
#: as the value of the grid 4.685 ... 48 with the smallest worst-case error; see docs/research/q1_research_program.md
SCREEN_C = 12.0

CRITERIA = {"T1_min_R2_linear": 0.90, "T1_quadratic_gain_max": 0.01, "T2_max_mae_g": 2.0, "T3_interp_max_mae_g": 3.0}
PREREGISTERED = ("spline", "modal3", "modal4", "amplitude", "rf_phase", "rf_amplitude")
EXPLORATORY = ("nuisance_mass_angle", "nuisance_tach", "modal3_rephased", "modal4_rephased")


def load_table(rows: list[dict[str, Any]], states=("normal", "imbalance")) -> dict[str, np.ndarray]:
    """Mass, speed and 1X phasors of the six accelerometers from :func:`.mafaulda.feature_table` rows (of the given
    states; ``None``: every row whose level is in grams or empty)."""
    rows = [r for r in rows if "error" not in r and r.get("tach_ok", True)
            and (r["state"] in states if states else r["level_unit"] in ("g", ""))]
    return {"m": np.array([r["level_value"] for r in rows]), "w": np.array([r["speed_hz"] for r in rows]),
            "A": np.array([r["phasors"][0, :6] for r in rows]), "paths": [r["path"] for r in rows],
            "state": np.array([r["state"] for r in rows])}


def _scores(mh, mt, levels):
    e = mh - mt
    return {"mae_g": float(np.mean(np.abs(e))), "rmse_g": float(np.sqrt(np.mean(e ** 2))),
            "within_2g": float(np.mean(np.abs(e) <= 2.0)),
            "level_accuracy": float(np.mean(nearest_level(mh, levels) == mt)), "n": int(len(mt))}


def _r2(Ah, A):
    return float(1.0 - np.sum(np.abs(A - Ah) ** 2) / np.sum(np.abs(A - A.mean(0)) ** 2))


def _rf_x(w, A, phase):
    return np.concatenate([w[:, None], np.abs(A)] + ([A.real, A.imag] if phase else []), 1)


def fit_estimate(name: str, w, m, A, w_te, A_te, seed: int = 0) -> np.ndarray:
    """Fit model ``name`` on (w, m, A) and return the estimated mass of the test runs."""
    if name == "spline":
        return estimate_mass(SplineInfluence().fit(w, m, A), w_te, A_te)
    if name in ("modal3", "modal4"):
        return estimate_mass(ModalInfluence(n_modes=int(name[-1]), seed=seed).fit(w, m, A), w_te, A_te)
    if name == "amplitude":
        return estimate_mass(AmplitudeInfluence().fit(w, m, A), w_te, A_te)
    if name == "amplitude_robust_median":
        return estimate_mass_robust(AmplitudeInfluence().fit(w, m, A), w_te, A_te)[0]
    if name == "amplitude_screened":
        return ScreenedEstimator(c=SCREEN_C, seed=seed).fit(w, m, A).estimate(w_te, A_te)[0]
    if name in ("rf_phase", "rf_amplitude"):
        from sklearn.ensemble import RandomForestRegressor
        phase = name == "rf_phase"
        rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=4)
        return rf.fit(_rf_x(w, A, phase), m).predict(_rf_x(w_te, A_te, phase))
    if name == "nuisance_mass_angle":
        return PhaseNuisanceInfluence("mass_angle").fit(w, m, A).estimate(w_te, A_te)
    if name == "nuisance_tach":
        return PhaseNuisanceInfluence("tach_reference").fit(w, m, A).estimate(w_te, A_te)
    if name in ("modal3_rephased", "modal4_rephased"):
        return PhaseNuisanceInfluence("tach_reference", backend=name[:6]).fit(w, m, A).estimate(w_te, A_te)
    raise ValueError(f"unknown model {name!r}")


def run_pilot(table: dict[str, np.ndarray], *, models=PREREGISTERED + EXPLORATORY, n_folds: int = 10, n_bands: int = 6,
              seed: int = 0, log=print) -> dict[str, Any]:
    m, w, A = table["m"], table["w"], table["A"]
    levels = sorted(set(m))
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(m)), n_folds)
    edges = np.quantile(w, np.linspace(0, 1, n_bands + 1))
    band = np.clip(np.searchsorted(edges, w, side="right") - 1, 0, n_bands - 1)
    inner = (band > 0) & (band < n_bands - 1)
    t0 = time.perf_counter()
    lin, quad = np.zeros_like(A), np.zeros_like(A)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(m)), te)
        lin[te] = SplineInfluence().fit(w[tr], m[tr], A[tr]).predict(w[te], m[te])
        quad[te] = SplineInfluence(quadratic=True).fit(w[tr], m[tr], A[tr]).predict(w[te], m[te])
    out: dict[str, Any] = {"n_runs": int(len(m)), "levels_g": levels, "speed_range_hz": [float(w.min()),
                                                                                         float(w.max())],
                           "speed_bands_hz": edges.tolist(), "criteria": CRITERIA,
                           "preregistered": list(PREREGISTERED), "exploratory": list(EXPLORATORY)}
    out["T1_linearity"] = {"heldout_R2_linear": _r2(lin, A), "heldout_R2_quadratic": _r2(quad, A),
                           "per_channel_R2_linear": [_r2(lin[:, c], A[:, c]) for c in range(A.shape[1])]}
    out["level_unbalance"] = level_unbalance(w, m, A)
    T2, T3 = {}, {}
    for name in models:
        mh = np.zeros_like(m)
        for te in folds:
            tr = np.setdiff1d(np.arange(len(m)), te)
            mh[te] = fit_estimate(name, w[tr], m[tr], A[tr], w[te], A[te], seed)
        T2[name] = _scores(mh, m, levels)
        mh3 = np.zeros_like(m)
        for b in range(n_bands):
            te, tr = np.flatnonzero(band == b), np.flatnonzero(band != b)
            mh3[te] = fit_estimate(name, w[tr], m[tr], A[tr], w[te], A[te], seed)
        T3[name] = {"interpolation": _scores(mh3[inner], m[inner], levels),
                    "extrapolation": _scores(mh3[~inner], m[~inner], levels)}
        log(f"{name:20s} seen {T2[name]['mae_g']:.2f} g | unseen: interp {T3[name]['interpolation']['mae_g']:.2f} g, "
            f"extrap {T3[name]['extrapolation']['mae_g']:.2f} g")
    out["T2_seen_speeds"], out["T3_unseen_speed_bands"] = T2, T3
    mod = ModalInfluence(n_modes=3, seed=seed).fit(w, m, A)
    out["modal3_all_runs"] = {"natural_frequencies_hz": mod.poles[:, 0].tolist(),
                              "damping_ratios": mod.poles[:, 1].tolist(), "R2": _r2(mod.predict(w, m), A)}
    out["verdict"] = verdict(out)
    out["seconds"] = time.perf_counter() - t0
    return out


def verdict(out: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered decision: each criterion, and whether a physics model beats the classical phase-free
    calibration at unseen speeds."""
    C, T1, T2, T3 = out["criteria"], out["T1_linearity"], out["T2_seen_speeds"], out["T3_unseen_speed_bands"]
    phys = [k for k in ("spline", "modal3", "modal4") if k in T2]
    best_phys_t2 = min(phys, key=lambda k: T2[k]["mae_g"])
    best_phys_t3 = min(phys, key=lambda k: T3[k]["interpolation"]["mae_g"])
    checks = {
        "linear_in_mass": T1["heldout_R2_linear"] >= C["T1_min_R2_linear"]
        and T1["heldout_R2_quadratic"] - T1["heldout_R2_linear"] <= C["T1_quadratic_gain_max"],
        "seen_speeds_physics_within_2g": T2[best_phys_t2]["mae_g"] <= C["T2_max_mae_g"],
        "unseen_speeds_physics_within_3g": T3[best_phys_t3]["interpolation"]["mae_g"] <= C["T3_interp_max_mae_g"],
        "modal_beats_amplitude_unseen": min(T3[k]["interpolation"]["mae_g"] for k in ("modal3", "modal4") if k in T3)
        < T3["amplitude"]["interpolation"]["mae_g"],
    }
    return {"checks": checks, "go": all(checks.values()), "best_physics_seen": best_phys_t2,
            "best_physics_unseen": best_phys_t3}


def write_report(out: dict[str, Any], folder: str | Path) -> dict[str, str]:
    """results.json, report.md and two figures (level unbalance; mass error by model and test)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {"results": str(folder / "results.json"), "report": str(folder / "report.md"),
             "level_unbalance": str(folder / "level_unbalance.png"), "errors": str(folder / "mass_errors.png")}
    Path(files["results"]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    lu = out["level_unbalance"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(lu["levels"], lu["magnitude"], "o-")
    ax[0].plot([0, max(lu["levels"])], [0, max(lu["levels"])], "k--", lw=0.8, label="equal to the nominal mass")
    ax[0].set_xlabel("added mass (g)")
    ax[0].set_ylabel("|equivalent unbalance| (g)")
    ax[0].legend(fontsize=8)
    ax[1].plot(lu["levels"][1:], lu["angle_deg"][1:], "o-")
    ax[1].set_xlabel("added mass (g)")
    ax[1].set_ylabel("angle of the unbalance vs 35 g (deg)")
    fig.suptitle(f"One free complex unbalance per level (R2 {lu['R2']:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(files["level_unbalance"], dpi=150)
    plt.close(fig)
    names = list(out["T2_seen_speeds"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for j, (lab, get) in enumerate((("seen speeds", lambda k: out["T2_seen_speeds"][k]["mae_g"]),
                                    ("unseen, interpolation",
                                     lambda k: out["T3_unseen_speed_bands"][k]["interpolation"]["mae_g"]),
                                    ("unseen, extrapolation",
                                     lambda k: out["T3_unseen_speed_bands"][k]["extrapolation"]["mae_g"]))):
        ax.bar(x + (j - 1) * 0.27, [get(k) for k in names], 0.27, label=lab)
    ax.axhline(out["criteria"]["T2_max_mae_g"], color="k", lw=0.8, ls="--")
    ax.axhline(out["criteria"]["T3_interp_max_mae_g"], color="0.5", lw=0.8, ls=":")
    ax.set_xticks(x, [n + (" *" if n in out["exploratory"] else "") for n in names], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("mass MAE (g)")
    ax.set_title("Unbalance mass error (* exploratory, added after the first run)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(files["errors"], dpi=150)
    plt.close(fig)
    v, T1 = out["verdict"], out["T1_linearity"]
    L = ["# MAFAULDA pilot: calibrated inversion of the 1X response for the unbalance mass", "",
         f"{out['n_runs']} runs (normal + imbalance 6-35 g), {out['speed_range_hz'][0]:.1f}-"
         f"{out['speed_range_hz'][1]:.1f} Hz, 1X phasors of the six accelerometers referred to the tachometer pulse. "
         "Every run is one sample; splits never share a run.", "",
         f"**Decision (criteria fixed before the run): {'GO' if v['go'] else 'NO GO'}**", ""]
    L += [f"- {k}: {'pass' if ok else 'fail'}" for k, ok in v["checks"].items()]
    L += ["", "## T1 linearity in mass (held-out R2 of the phasors)", "",
          f"linear {T1['heldout_R2_linear']:.3f}, with an m^2 term {T1['heldout_R2_quadratic']:.3f}; per channel "
          + ", ".join(f"{r:.2f}" for r in T1["per_channel_R2_linear"]), "",
          "## Level unbalance (one free complex value per level)", "",
          "| level (g) | magnitude (g) | angle vs 35 g (deg) |", "|---|---|---|"]
    L += [f"| {a:g} | {b:.2f} | {c:.1f} |" for a, b, c in zip(lu["levels"], lu["magnitude"], lu["angle_deg"],
                                                               strict=True) if a > 0]
    L += ["", f"R2 {lu['R2']:.3f}.", "", "## Mass error (g)", "",
          "| model | kind | seen: MAE | seen: right level | unseen interp: MAE | unseen extrap: MAE |",
          "|---|---|---|---|---|---|"]
    for k in names:
        a, b = out["T2_seen_speeds"][k], out["T3_unseen_speed_bands"][k]
        L.append(f"| {k} | {'exploratory' if k in out['exploratory'] else 'pre-registered'} | {a['mae_g']:.2f} | "
                 f"{100 * a['level_accuracy']:.0f} % | {b['interpolation']['mae_g']:.2f} | "
                 f"{b['extrapolation']['mae_g']:.2f} |")
    mo = out["modal3_all_runs"]
    L += ["", "Modal model (3 modes, all runs): natural frequencies "
          + ", ".join(f"{f:.1f}" for f in mo["natural_frequencies_hz"]) + " Hz, damping "
          + ", ".join(f"{z:.3f}" for z in mo["damping_ratios"]) + f", R2 {mo['R2']:.3f}.", "",
          "## Limits", "",
          "- The equivalent mass is at the calibration radius and angle: the radius of the trial masses and the modal "
          "masses enter only as a product, the known masses fix the scale.",
          "- The 1X phase reference differs between mass levels (table above); phase-referenced inversion is therefore "
          "not reliable on this dataset as recorded.",
          "- Exploratory models were added after the first run; they need confirmation on independent sessions (for "
          "instance the bearing-fault folders of MAFAULDA, which carry 0, 6, 20 and 35 g).", "",
          "Figures: [level unbalance](level_unbalance.png), [mass errors](mass_errors.png)"]
    Path(files["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    return files


def cross_session(cal: dict[str, np.ndarray], test: dict[str, np.ndarray], *,
                  models=("amplitude", "spline", "nuisance_tach", "rf_amplitude"), seed: int = 0) -> dict[str, Any]:
    """Step A: models calibrated on one set of recording sessions (``cal``, e.g. normal + imbalance) estimate the added
    mass of runs recorded in other sessions (``test``, e.g. the bearing-fault folders, which carry 0/6/20/35 g):
    scores overall and per state (A1); the free complex unbalance of each test state, to see whether the 1X phase
    reference drifts there too (A2); and the same phase-free model calibrated inside each test state, leave one run
    out (A3)."""
    levels = sorted(set(test["m"]))
    out: dict[str, Any] = {"n_cal": int(len(cal["m"])), "n_test": int(len(test["m"])), "test_levels_g": levels,
                           "states": sorted(set(test["state"].tolist())), "A1": {}, "A2": {}, "A3": {}}
    w_rng = (float(cal["w"].min()), float(cal["w"].max()))
    inside = (test["w"] >= w_rng[0]) & (test["w"] <= w_rng[1])
    for name in models:
        mh = fit_estimate(name, cal["w"], cal["m"], cal["A"], test["w"], test["A"], seed)
        row = {"all": _scores(mh[inside], test["m"][inside], levels)}
        for st in out["states"]:
            s_ = (test["state"] == st) & inside
            row[st] = _scores(mh[s_], test["m"][s_], levels)
        out["A1"][name] = row
    for st in out["states"]:
        s_ = test["state"] == st
        lu = level_unbalance(test["w"][s_], test["m"][s_], test["A"][s_])
        ang = [a for lv, a in zip(lu["levels"], lu["angle_deg"], strict=True) if lv > 0]
        out["A2"][st] = {**lu, "angle_span_deg": float(np.ptp(ang)) if ang else None}
        idx = np.flatnonzero(s_)
        mh = np.zeros(len(idx))
        for j, i in enumerate(idx):
            tr = np.setdiff1d(idx, [i])
            mh[j] = estimate_mass(AmplitudeInfluence().fit(test["w"][tr], test["m"][tr], test["A"][tr]),
                                  test["w"][[i]], test["A"][[i]])[0]
        out["A3"][st] = _scores(mh, test["m"][idx], levels)
    return out
