# DC microgrid twin: process, sensor or model?

Four-bus ring at 380 V: battery converter (droop, bus 1), PV converter in MPPT + resistive load (bus 2),
second droop converter (bus 3), constant-power load (bus 4); 13 sensors, 13 physics residuals.

**Trust decision: APPROVED** (score 1.00)

## Diagnosis on controlled cases

80 cases, window 30 s, thresholds from 12 healthy runs.

| true \ diagnosed | normal | process | sensor | domain |
|---|---|---|---|---|
| normal | 20 | 0 | 0 | 0 |
| process | 0 | 20 | 0 | 0 |
| sensor | 0 | 0 | 20 | 0 |
| domain | 0 | 0 | 0 | 20 |

- accuracy 100.0 %; cases with a false alarm: 1
- isolation: {"process": {"correct_kind": 20, "correct_target": 20}, "sensor": {"correct_kind": 20, "correct_target": 20, "correct_mode": 20}, "domain": {"correct_kind": 20, "correct_target": 20}}
- per component: {"line_12": {"n": 5, "correct": 5}, "line_23": {"n": 2, "correct": 2}, "line_34": {"n": 4, "correct": 4}, "line_41": {"n": 3, "correct": 3}, "pv_converter": {"n": 1, "correct": 1}, "droop_1": {"n": 2, "correct": 2}, "droop_3": {"n": 3, "correct": 3}}
- process size, median relative error: Kalman 0.03946587072333669, GLR 0.0027860920066358555; Kalman inside 2 sigma 20/20
- median delay after onset (s): {"process": 6.632737171088138, "sensor": 14.230129470578817, "domain": 20.9195761619322}

## Isolability

Smallest median angle between whitened fault directions: 33.5 deg. Sensor directions orthonormal: True.

| fault a | fault b | median angle (deg) | smallest angle (deg) |
|---|---|---|---|
| process:line_23 | process:line_34 | 33.5 | 33.4 |
| process:line_12 | process:line_41 | 33.7 | 33.5 |
| sensor:I_PV | process:pv_converter | 45.4 | 45.0 |
| sensor:I_S3 | process:droop_3 | 54.9 | 54.2 |
| sensor:I_S1 | process:droop_1 | 55.0 | 54.0 |
| sensor:I_S1 | process:droop_3 | 55.0 | 54.2 |
| sensor:I_41 | process:line_34 | 55.2 | 54.9 |
| sensor:I_S3 | process:droop_1 | 55.2 | 54.3 |

Pairs closer than 5 deg at some operating points (sign-aware angle; the folded angle a two-sided test would see in brackets):

- process:line_12 / process:line_23: 25 % of the operating points (folded median 4.0 deg)

## Fast model at equal budget (median over seeds)

RMS error in sensor sigmas (median over the test points), physics residual in residual sigmas.

| model | N | inside: error | inside: residual | shell: error | shell: residual |
|---|---|---|---|---|---|
| mlp_data | 8 | 13.4 | 9.06 | 37.7 | 24 |
| mlp_data | 16 | 12.4 | 8.18 | 40.2 | 25.3 |
| mlp_data | 32 | 3.32 | 2.11 | 22.4 | 14.2 |
| mlp_data | 64 | 1.11 | 0.748 | 12.1 | 7.47 |
| pinn_box | 8 | 1.44 | 0.745 | 9.71 | 5.23 |
| pinn_box | 16 | 1.29 | 0.619 | 8.8 | 4.13 |
| pinn_box | 32 | 1.18 | 0.696 | 8.65 | 4.49 |
| pinn_box | 64 | 1.3 | 0.756 | 8.19 | 4.59 |
| pinn_extended | 8 | 1.93 | 1.18 | 3.49 | 1.54 |
| pinn_extended | 16 | 1.32 | 0.712 | 2.17 | 0.854 |
| pinn_extended | 32 | 1.37 | 0.828 | 2.52 | 0.961 |
| pinn_extended | 64 | 1.54 | 0.946 | 2.28 | 1.1 |
| pod_gp | 8 | 49.4 | 29.9 | 88.7 | 57.9 |
| pod_gp | 16 | 3.26 | 2.82 | 10.2 | 7.18 |
| pod_gp | 32 | 0.459 | 0.289 | 1.85 | 1.19 |
| pod_gp | 64 | 0.0451 | 0.0279 | 0.378 | 0.249 |

Spearman (PINN residual vs its true error): {"pinn_box/inside": {"median": 0.7623715592889824, "min": 0.612391809795245}, "pinn_extended/shell": {"median": 0.4854458861471538, "min": 0.35948848721218035}}
Timing: Newton solve 90 us, MLP evaluation 55 us (batch of one).

## Showcase

- loose joint in line 2-3: diagnosed process:line_23, first alarm 159.0 s; line_23: 0.0961 +- 0.028 ohm (Kalman); inspect the joints and terminations of the line
- current sensor I_12 drifting: diagnosed sensor:I_12, first alarm 171.0 s; I_12: bias +0.522 A -- check / recalibrate it; meanwhile estimate it from the other sensors
- PV converter losing 3 % efficiency: diagnosed process:pv_converter, first alarm 164.0 s; pv_converter: 0.0284 +- 0.0029 (efficiency) (Kalman); inspect the converter (power stage, output filter, control set-up)

## Trust report

| check | status | value | limit | source |
|---|---|---|---|---|
| reference simulator: Kirchhoff current law | supports | 2.7284841053187847e-12 | 1e-06 | largest bus current mismatch over every sample of every case (A) |
| reference simulator: energy balance | supports | 1.4242118595575448e-09 | 0.001 | sources - loads - line losses (W) |
| quasi-steady sampling | supports | None | None | 1 s sampling >= 10 x the assumed 0.02 s converter settling time |
| structural signatures all distinct | supports | None | None | every sensor and process fault enters a different set of residuals |
| isolability: smallest median angle | supports | 33.51921621061551 | 5.0 | process:line_23 vs process:line_34 |
| diagnosis accuracy | supports | 1.0 | 0.9 | 80 controlled cases |
| cases with a false alarm | supports | 1.0 | 4.0 | alarm before the onset, or any alarm on a healthy case |
| Kalman size inside 2 sigma | supports | 20.0 | 16.0 | 20 process faults isolated |
| PINN fast model satisfies the network laws | supports | 0.7562724769294662 | 1.0 | median RMS residual of its predictions inside the envelope (residual sigmas), worst budget |

## Limits

- Simulated plant: the droop gains, PV efficiency, sensor noise and envelope are assumptions, not a measured installation; quasi-steady samples (no converter dynamics).
- One fault at a time; faults that act together, or a fault inside the training of the fast model, are not covered.
- For four buses the Newton solve is itself cheap: the fast model matters for larger networks or for plant models without a closed form.

Figures: [confusion](confusion.png), [isolability](isolability.png), [surrogates](surrogates.png), [showcase](showcase.png)
