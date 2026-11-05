"""The whole study in one call: structure, isolability, controlled-case benchmark, PINN fast model at equal budget,
showcase timelines, physics checks and the trust report, written to a folder.

    from pinneapple_systems.digital_twin.dc_microgrid import run_study
    out = run_study("out", n_per_kind=20)       # summary.json, report.md, confusion.png, surrogates.png, ...
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import KINDS, run_benchmark
from .diagnose import HYPOTHESES, RESIDUALS, isolability, signature_matrix
from .pinn import MODELS, surrogate_study
from .plant import COMPONENTS, INPUTS, SENSORS, Microgrid, Scenario, quasi_steady_valid, run_plant

#: acceptance limits of the trust report, fixed before the run
LIMITS = {"kcl_A": 1e-6, "energy_W": 1e-3, "accuracy": 0.90, "false_alarm_fraction": 0.05,
          "min_isolation_angle_deg": 5.0, "kalman_inside_2std_fraction": 0.80, "pinn_residual_sigma": 1.0}


def _jsonable(x):
    from pinneapple_core.jsonable import to_jsonable
    return to_jsonable(x, nonfinite=None)


def _figures(out: Path, bench: dict, study: dict | None, iso: dict, show: dict) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = {}
    M = np.array(bench["confusion"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(M, cmap="Blues")
    for i in range(len(KINDS)):
        for j in range(len(KINDS)):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks(range(len(KINDS)), KINDS, rotation=30)
    ax.set_yticks(range(len(KINDS)), KINDS)
    ax.set_xlabel("diagnosed")
    ax.set_ylabel("true")
    ax.set_title(f"DC microgrid twin: {100 * bench['accuracy']:.1f} % of {bench['n_cases']} cases")
    fig.tight_layout()
    files["confusion"] = str(out / "confusion.png")
    fig.savefig(files["confusion"], dpi=150)
    plt.close(fig)

    A = np.array(iso["median_deg"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=90)
    ax.set_xticks(range(len(HYPOTHESES)), [h.split(":")[1] for h in HYPOTHESES], rotation=90, fontsize=7)
    ax.set_yticks(range(len(HYPOTHESES)), [h.split(":")[1] for h in HYPOTHESES], fontsize=7)
    fig.colorbar(im, ax=ax, label="median angle between whitened fault directions (deg)")
    ax.set_title("Isolability: 90 = orthogonal, 0 = indistinguishable")
    fig.tight_layout()
    files["isolability"] = str(out / "isolability.png")
    fig.savefig(files["isolability"], dpi=150)
    plt.close(fig)

    if study:
        fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
        for ax, k in zip(axs, ("inside", "shell"), strict=True):
            for m in MODELS:
                y = [study["summary"][f"{m}@{n}"][k]["err_median"] for n in study["budgets"]]
                ax.plot(study["budgets"], y, "o-", label=m)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
            ax.set_xlabel("labelled reference solves N")
            ax.set_title("inside the envelope" if k == "inside" else "shell outside the envelope")
        axs[0].set_ylabel("median RMS error (sensor sigmas)")
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        files["surrogates"] = str(out / "surrogates.png")
        fig.savefig(files["surrogates"], dpi=150)
        plt.close(fig)

    fig, axs = plt.subplots(len(show), 1, figsize=(8.0, 2.4 * len(show)), sharex=True)
    for ax, (name, d) in zip(np.atleast_1d(axs), show.items(), strict=True):
        t = d["t"]
        ax.plot(t, d["shift"], label="largest GLR gain (bias-type faults)")
        ax.axhline(d["thresholds"]["shift"], color="C0", ls="--", lw=0.8)
        ax.plot(t, d["spread"] * d["thresholds"]["shift"] / max(d["thresholds"]["spread"], 1e-12),
                label="variance excess (scaled to the same threshold)", alpha=0.8)
        ax.axvline(d["onset_s"], color="k", lw=0.8)
        ax.set_yscale("symlog")
        ax.set_title(f"{name}: truth {d['truth']}, diagnosed {d['diagnosed']}", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
    np.atleast_1d(axs)[-1].set_xlabel("time (s)")
    fig.tight_layout()
    files["showcase"] = str(out / "showcase.png")
    fig.savefig(files["showcase"], dpi=150)
    plt.close(fig)
    return files


def _showcase(diag, grid: Microgrid) -> dict[str, Any]:
    cases = [Scenario("loose joint in line 2-3", "process", 77, onset_s=150.0, target="line_23", magnitude=0.08),
             Scenario("current sensor I_12 drifting", "sensor", 78, onset_s=150.0, target="I_12", magnitude=0.5,
                      mode="drift"),
             Scenario("PV converter losing 3 % efficiency", "process", 79, onset_s=150.0, target="pv_converter",
                      magnitude=0.03)]
    out = {}
    for sc in cases:
        run = run_plant(grid, sc)
        d = diag.diagnose(run)
        out[sc.name] = {"t": run["t"], "shift": d["stats"]["shift"], "spread": d["stats"]["spread"],
                        "thresholds": d["thresholds"], "onset_s": sc.onset_s,
                        "truth": f"{sc.kind}:{sc.target}", "diagnosed": f"{d['kind']}:{d.get('target', '')}",
                        "size": d.get("size"), "size_std": d.get("size_std"), "action": d["action"],
                        "first_alarm_s": d["first_alarm_s"]}
    return out


def trust_report(grid: Microgrid, bench: dict, iso: dict, sig: dict, study: dict | None):
    """Physics and acceptance checks of the study (limits in LIMITS, fixed before the run)."""
    from pinneapple_analysis.trust.trust_report import Check, TrustReport, checagem_booleana
    L = LIMITS
    n = bench["n_cases"]
    proc = bench["process_size_rel_error"]
    checks = [
        Check("reference simulator: Kirchhoff current law", "supports" if bench["kcl_max_A"] <= L["kcl_A"]
              else "contradicts", critical=True, value=bench["kcl_max_A"], limit=L["kcl_A"],
              source="largest bus current mismatch over every sample of every case (A)"),
        Check("reference simulator: energy balance", "supports" if bench["energy_balance_max_W"] <= L["energy_W"]
              else "contradicts", critical=True, value=bench["energy_balance_max_W"], limit=L["energy_W"],
              source="sources - loads - line losses (W)"),
        checagem_booleana("quasi-steady sampling", quasi_steady_valid(grid, 1.0),
                          fonte=f"1 s sampling >= 10 x the assumed {grid.settling_s} s converter settling time"),
        checagem_booleana("structural signatures all distinct", sig["all_distinct"], critica=True,
                          fonte="every sensor and process fault enters a different set of residuals"),
        Check("isolability: smallest median angle", "supports" if iso["smallest_median_deg"] >= L[
            "min_isolation_angle_deg"] else "contradicts", value=iso["smallest_median_deg"],
              limit=L["min_isolation_angle_deg"], source=f"{iso['closest_pairs'][0]['a']} vs "
                                                         f"{iso['closest_pairs'][0]['b']}"),
        Check("diagnosis accuracy", "supports" if bench["accuracy"] >= L["accuracy"] else "contradicts",
              value=bench["accuracy"], limit=L["accuracy"], source=f"{n} controlled cases"),
        Check("cases with a false alarm", "supports" if bench["cases_with_false_alarm"] <= L[
            "false_alarm_fraction"] * n else "contradicts", value=float(bench["cases_with_false_alarm"]),
              limit=L["false_alarm_fraction"] * n, source="alarm before the onset, or any alarm on a healthy case"),
        Check("Kalman size inside 2 sigma", "supports" if proc["n"] and proc["kalman_inside_2std"] >= L[
            "kalman_inside_2std_fraction"] * proc["n"] else "contradicts", value=float(proc["kalman_inside_2std"]),
              limit=L["kalman_inside_2std_fraction"] * proc["n"], source=f"{proc['n']} process faults isolated"),
    ]
    if study:
        r = max(study["summary"][f"pinn_box@{n_}"]["inside"]["residual_median"] for n_ in study["budgets"])
        checks.append(Check("PINN fast model satisfies the network laws", "supports" if r <= L["pinn_residual_sigma"]
                            else "contradicts", value=r, limit=L["pinn_residual_sigma"],
                            source="median RMS residual of its predictions inside the envelope (residual sigmas), "
                                   "worst budget"))
    return TrustReport(checks, model="DC microgrid twin")


def _report(path: Path, s: dict, files: dict) -> None:
    b, iso, st = s["benchmark"], s["isolability"], s.get("surrogates")
    lines = ["# DC microgrid twin: process, sensor or model?", "",
             "Four-bus ring at 380 V: battery converter (droop, bus 1), PV converter in MPPT + resistive load (bus 2),",
             "second droop converter (bus 3), constant-power load (bus 4); 13 sensors, 13 physics residuals.", "",
             f"**Trust decision: {s['trust']['decisao']}** (score {s['trust']['score']:.2f})", "",
             "## Diagnosis on controlled cases", "",
             f"{b['n_cases']} cases, window {b['window']} s, thresholds from {b['n_calibration']} healthy runs.", "",
             "| true \\ diagnosed | " + " | ".join(b["kinds"]) + " |", "|---" * (len(b["kinds"]) + 1) + "|"]
    for k, row in zip(b["kinds"], b["confusion"], strict=True):
        lines.append(f"| {k} | " + " | ".join(str(x) for x in row) + " |")
    lines += ["", f"- accuracy {100 * b['accuracy']:.1f} %; cases with a false alarm: {b['cases_with_false_alarm']}",
              f"- isolation: {json.dumps(b['isolation'])}",
              f"- per component: {json.dumps(b['per_component'])}",
              f"- process size, median relative error: Kalman {b['process_size_rel_error']['kalman_median']}, "
              f"GLR {b['process_size_rel_error']['glr_median']}; Kalman inside 2 sigma "
              f"{b['process_size_rel_error']['kalman_inside_2std']}/{b['process_size_rel_error']['n']}",
              f"- median delay after onset (s): {json.dumps(b['delay_s_median'])}", "",
              "## Isolability", "",
              f"Smallest median angle between whitened fault directions: {iso['smallest_median_deg']:.1f} deg. "
              f"Sensor directions orthonormal: {iso['sensor_directions_orthonormal']}.", "",
              "| fault a | fault b | median angle (deg) | smallest angle (deg) |", "|---|---|---|---|"]
    for p in iso["closest_pairs"]:
        lines.append(f"| {p['a']} | {p['b']} | {p['median_deg']:.1f} | {p['min_deg']:.1f} |")
    lines += ["", f"Pairs closer than {iso['confusable_deg']:g} deg at some operating points (sign-aware angle; "
                  "the folded angle a two-sided test would see in brackets):", ""]
    for p in iso["confusable_pairs"] or [{"a": "none", "b": "", "confusable_fraction": 0.0,
                                         "unsigned_median_deg": float("nan")}]:
        lines.append(f"- {p['a']} / {p['b']}: {100 * p['confusable_fraction']:.0f} % of the operating points "
                     f"(folded median {p['unsigned_median_deg']:.1f} deg)")
    if st:
        lines += ["", "## Fast model at equal budget (median over seeds)", "",
                  "RMS error in sensor sigmas (median over the test points), physics residual in residual sigmas.", "",
                  "| model | N | inside: error | inside: residual | shell: error | shell: residual |",
                  "|---|---|---|---|---|---|"]
        for m in MODELS:
            for n in st["budgets"]:
                v = st["summary"][f"{m}@{n}"]
                lines.append(f"| {m} | {n} | {v['inside']['err_median']:.3g} | {v['inside']['residual_median']:.3g} | "
                             f"{v['shell']['err_median']:.3g} | {v['shell']['residual_median']:.3g} |")
        lines += ["", f"Spearman (PINN residual vs its true error): {json.dumps(st['spearman_residual_vs_error'])}",
                  f"Timing: Newton solve {1e6 * st['seconds']['newton_solve']:.0f} us, MLP evaluation "
                  f"{1e6 * st['seconds']['mlp_eval']:.0f} us (batch of one)."]
    lines += ["", "## Showcase", ""]
    for k, v in s["showcase"].items():
        lines.append(f"- {k}: diagnosed {v['diagnosed']}, first alarm {v['first_alarm_s']} s; {v['action']}")
    lines += ["", "## Trust report", "", "| check | status | value | limit | source |", "|---|---|---|---|---|"]
    for c in s["trust"]["checagens"]:
        lines.append(f"| {c['name']} | {c['status']} | {c['value']} | {c['limit']} | {c['source']} |")
    lines += ["", "## Limits", "",
              "- Simulated plant: the droop gains, PV efficiency, sensor noise and envelope are assumptions, not a "
              "measured installation; quasi-steady samples (no converter dynamics).",
              "- One fault at a time; faults that act together, or a fault inside the training of the fast model, are "
              "not covered.",
              "- For four buses the Newton solve is itself cheap: the fast model matters for larger networks or for "
              "plant models without a closed form.", "",
              "Figures: " + ", ".join(f"[{k}]({Path(p).name})" for k, p in files.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(out_dir: str | Path, *, n_per_kind: int = 20, n_calibration: int = 12, workers: int = 4,
              surrogates: bool = True, budgets: tuple[int, ...] = (8, 16, 32, 64), seeds: tuple[int, ...] = (0, 1, 2),
              epochs: tuple[int, int] = (500, 5000), log=print) -> dict[str, Any]:
    """Run every stage and write summary.json, report.md and the figures to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    grid = Microgrid()
    sig = signature_matrix(grid)
    iso = isolability(grid)
    log(f"signatures distinct: {sig['all_distinct']}; smallest median isolation angle "
        f"{iso['smallest_median_deg']:.1f} deg ({iso['closest_pairs'][0]['a']} vs {iso['closest_pairs'][0]['b']})")
    bench = run_benchmark(grid, n_per_kind=n_per_kind, n_calibration=n_calibration, workers=workers, log=log)
    log(f"diagnosis: {100 * bench['accuracy']:.1f} % of {bench['n_cases']} cases, confusion {bench['confusion']}")
    study = surrogate_study(grid, budgets=budgets, seeds=seeds, epochs=epochs, workers=workers, log=log) \
        if surrogates else None
    show = _showcase(bench["diagnoser"], grid)
    tr = trust_report(grid, bench, iso, sig, study)
    files = _figures(out, bench, study, iso, show)
    summary = {"grid": {k: v for k, v in vars(grid).items()}, "line_resistance_ohm": {k: grid.r_line(k) for k in
                                                                                       ("12", "23", "34", "41")},
               "sensors": list(SENSORS), "inputs": list(INPUTS), "residuals": list(RESIDUALS),
               "components": list(COMPONENTS), "signatures": sig, "isolability": iso,
               "benchmark": {k: v for k, v in bench.items() if k not in ("diagnoser",)},
               "surrogates": study, "showcase": {k: {kk: vv for kk, vv in v.items() if kk not in ("t", "shift",
                                                                                                    "spread")}
                                                 for k, v in show.items()},
               "trust": tr.to_dict(), "limits": LIMITS, "files": files, "seconds": time.perf_counter() - t0}
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1), encoding="utf-8")
    _report(out / "report.md", summary, files)
    files["report"] = str(out / "report.md")
    files["summary"] = str(out / "summary.json")
    log(f"{tr.decisao}: {out / 'report.md'} ({summary['seconds']:.0f} s)")
    return {"summary": summary, "trust": tr, "files": files, "benchmark": bench, "surrogates": study}
