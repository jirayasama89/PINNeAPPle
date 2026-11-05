"""DC microgrid twin: a changed network, a wrong sensor, or a fast model outside its domain?

A four-bus DC microgrid in a ring at 380 V (battery converter and a second converter in voltage droop, a PV converter
in MPPT, a resistive load and a constant-power EV charger) runs while its twin reads thirteen sensors. Then something
changes: a joint in a line loosens, the PV converter loses efficiency or a droop converter's output resistance grows
(the process); a voltage or current sensor drifts or gets noisy; or the grid is driven outside the envelope the fast
model was trained on. Thirteen physics residuals (Kirchhoff, Ohm, droop, MPPT and load laws), whitened by the
propagated sensor noise, and a likelihood ratio per fault hypothesis tell them apart; a Kalman filter sizes the
process faults; the isolability angles say which faults look alike; and a physics-informed fast model is compared
with a data-only network and POD + GP at equal budget. Everything ends in a trust report.

Run: ``python examples/dc_microgrid_twin/microgrid_twin.py [--n-per-kind 20] [--no-surrogates]``
(``--smoke``: 3 cases per kind, one budget, short training). Outputs in ``examples/dc_microgrid_twin/_out/``.
"""
from __future__ import annotations

import argparse
import os

import pinneapple_lab
from pinneapple_systems.digital_twin.dc_microgrid import run_study

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-kind", type=int, default=20)
    ap.add_argument("--no-surrogates", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    a = ap.parse_args()
    kw = dict(n_per_kind=a.n_per_kind, surrogates=not a.no_surrogates, workers=a.workers)
    if a.smoke:
        kw.update(n_per_kind=3, n_calibration=4, budgets=(8,), seeds=(0,), epochs=(100, 300))
    with pinneapple_lab.experiment(
            "examples/dc_microgrid_twin", title="DC microgrid twin: process, sensor or model?",
            description="A four-bus DC microgrid twin that tells a loose joint or a degraded converter from a drifting "
                        "or noisy sensor from a fast model outside its domain, with whitened physics residuals, a "
                        "likelihood ratio per fault, a Kalman filter and isolability angles; a PINN fast model "
                        "against a data-only network and POD + GP at equal budget; confusion matrix on controlled "
                        "cases.",
            domain="power systems", methods=["DC power flow (droop, MPPT, constant-power load)", "structured residuals",
                                             "generalised likelihood ratio", "EKF", "POD-GPR", "PINN"],
            modules=["pinneapple_systems.digital_twin.dc_microgrid", "pinneapple_neural.trainer.graybox",
                     "pinneapple_analysis.state_estimation.kalman"], out_dir=HERE, register=False,
            links={"guide": "docs/guides/dc_microgrid_twin.md",
                   "script": "examples/dc_microgrid_twin/microgrid_twin.py"}) as ex:
        out = run_study(a.out, **kw)
        b = out["benchmark"]
        s = out["summary"]
        ex.log_metric("cause_accuracy", b["accuracy"], "fraction", f"{b['n_cases']} controlled cases")
        ex.log_metric("cases_with_false_alarm", b["cases_with_false_alarm"], "cases",
                      "alarm before the fault, or any alarm on a healthy case")
        ex.log_metric("smallest_isolation_angle", s["isolability"]["smallest_median_deg"], "deg",
                      "median over 64 operating points, closest pair of fault hypotheses")
        st = out["surrogates"]
        if st:
            n0 = st["budgets"][0]
            for m in ("pinn_box", "mlp_data", "pod_gp"):
                ex.log_metric(f"{m}_inside_error_N{n0}", st["summary"][f"{m}@{n0}"]["inside"]["err_median"],
                              "sensor sigmas", f"median RMS error on {st['n_test']} held-out points, {n0} labels, "
                                               f"median over {len(st['seeds'])} seeds")
        ex.check("energy balance of the reference simulator", b["energy_balance_max_W"] <= 1e-3,
                 value=b["energy_balance_max_W"], max=1e-3, kind="physics")
        ex.check("structural signatures all distinct", s["signatures"]["all_distinct"], kind="physics")
        ex.check("trust report decision", out["trust"].decisao == "APPROVED",
                 detail=f"{out['trust'].decisao} {out['trust'].motivos}", kind="verification")
        for k, p in out["files"].items():
            if k != "summary":
                ex.add_file(p)
        print(f"\n{out['trust'].decisao}: report {out['files']['report']}")


if __name__ == "__main__":
    main()
