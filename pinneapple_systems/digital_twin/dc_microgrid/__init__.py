"""A DC microgrid twin that tells a changed network from a wrong sensor from a fast model outside its domain.

Four buses in a ring at 380 V (battery and second converter in voltage droop, a PV converter in MPPT, a resistive and
a constant-power load), thirteen sensors, thirteen physics residuals (Kirchhoff, Ohm, droop, MPPT, load laws),
whitened by the propagated sensor noise; a generalised likelihood ratio per fault hypothesis, a Kalman filter on the
fault parameters, a POD + GP fast model with its domain of validity, and a physics-informed fast model compared at
equal budget with a data-only network.

    from pinneapple_systems.digital_twin.dc_microgrid import run_study, run_benchmark, Microgrid, Scenario, run_plant
    out = run_study("out", n_per_kind=20)

Modules: ``plant`` (network, reference simulator, controlled faults), ``diagnose`` (residuals, isolability, fast
model, Kalman, decision), ``benchmark`` (controlled cases, confusion matrix), ``pinn`` (physics-informed fast model,
equal-budget study), ``study`` (every stage, figures, trust report).
"""
from .benchmark import KINDS, make_cases, run_benchmark
from .diagnose import (
    HYPOTHESES,
    RESIDUALS,
    Diagnoser,
    FaultTracker,
    Residuals,
    Surrogate,
    isolability,
    raw_residuals,
    signature_matrix,
)
from .pinn import EXTENDED_BOX, fit_mlp, surrogate_study, torch_residuals
from .plant import (
    COMPONENTS,
    INPUTS,
    LINES,
    NORMAL_BOX,
    SENSORS,
    SIGMA,
    Faults,
    Microgrid,
    Scenario,
    component_faults,
    quasi_steady_valid,
    readings,
    run_plant,
    steady_state,
)
from .study import LIMITS, run_study, trust_report

__all__ = ["KINDS", "make_cases", "run_benchmark", "HYPOTHESES", "RESIDUALS", "Diagnoser", "FaultTracker", "Residuals",
           "Surrogate", "isolability", "raw_residuals", "signature_matrix", "EXTENDED_BOX", "fit_mlp",
           "surrogate_study", "torch_residuals", "COMPONENTS", "INPUTS", "LINES", "NORMAL_BOX", "SENSORS", "SIGMA",
           "Faults", "Microgrid", "Scenario", "component_faults", "quasi_steady_valid", "readings", "run_plant",
           "steady_state", "LIMITS", "run_study", "trust_report"]
