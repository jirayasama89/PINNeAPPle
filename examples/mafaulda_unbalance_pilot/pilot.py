"""MAFAULDA pilot: can a calibrated dynamic inversion of the 1X response recover the unbalance mass?

On the normal and imbalance runs of MAFAULDA (SpectraQuest fault simulator, 5 s at 50 kHz, 6 to 35 g), the 1X
phasors of the six accelerometers, referred to the tachometer pulse, are modelled as ``A = alpha(w) m + beta(w)``
across speeds (complex splines, a linear dissipative modal model), against the classical phase-free calibration and a
random forest, at seen speeds and in held-out speed bands, with criteria fixed before the first run.

Data (not in the repository, 2.6 GB): download and extract into one folder, e.g.

    curl -O https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/normal.tgz
    curl -O https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/imbalance.tgz
    tar -xzf normal.tgz -C data && tar -xzf imbalance.tgz -C data

Run: ``python examples/mafaulda_unbalance_pilot/pilot.py --data data`` (the order features are cached in
``_out/features.npz``; the raw CSVs can be deleted after the first run). Outputs in ``_out/``.
"""
from __future__ import annotations

import argparse
import os

import pinneapple_lab
from pinneapple_analysis.inspection.rotating import mafaulda, unbalance_study

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with normal/ and imbalance/ extracted")
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with pinneapple_lab.experiment(
            "examples/mafaulda_unbalance_pilot", title="MAFAULDA pilot: calibrated inversion of the 1X response",
            description="Unbalance mass from the 1X phasors of MAFAULDA (influence coefficients across speeds, a "
                        "linear dissipative modal model, classical phase-free calibration, random forest), at seen "
                        "speeds and in held-out speed bands, criteria fixed before the run.",
            domain="vibration diagnostics", methods=["order tracking", "influence coefficients",
                                                     "linear dissipative modal model", "random forest"],
            modules=["pinneapple_analysis.inspection.rotating"], out_dir=HERE, register=False,
            links={"script": "examples/mafaulda_unbalance_pilot/pilot.py"}) as ex:
        runs = [r for r in mafaulda.index(a.data) if r.state in ("normal", "imbalance")]
        rows = mafaulda.feature_table(runs, cache=os.path.join(a.out, "features.npz"), workers=a.workers)
        table = unbalance_study.load_table(rows)
        out = unbalance_study.run_pilot(table)
        files = unbalance_study.write_report(out, a.out)
        T2, T3 = out["T2_seen_speeds"], out["T3_unseen_speed_bands"]
        ex.log_metric("heldout_R2_linear_in_mass", out["T1_linearity"]["heldout_R2_linear"], "1",
                      "10-fold by run, 1X phasors of six accelerometers")
        for k in ("amplitude", "spline", "modal3"):
            ex.log_metric(f"{k}_seen_mae", T2[k]["mae_g"], "g", f"10-fold by run, {out['n_runs']} runs")
            ex.log_metric(f"{k}_unseen_interp_mae", T3[k]["interpolation"]["mae_g"], "g",
                          "leave one of six speed bands out, inner bands")
        ex.check("tachometer: one pulse per revolution on every run",
                 all(r.get("revolutions", 0) >= 50 for r in rows), kind="physics")
        ex.check("magnitude of the level unbalance grows with the added mass",
                 bool(all(b > a_ for a_, b in zip(out["level_unbalance"]["magnitude"][:-1],
                                                  out["level_unbalance"]["magnitude"][1:], strict=True))),
                 kind="physics")
        ex.check("pre-registered decision", out["verdict"]["go"], detail=str(out["verdict"]["checks"]),
                 kind="verification")
        for k in ("report", "level_unbalance", "errors"):
            ex.add_file(files[k])
        print(open(files["report"]).read())


if __name__ == "__main__":
    main()
