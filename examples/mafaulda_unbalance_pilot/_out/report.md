# MAFAULDA pilot: calibrated inversion of the 1X response for the unbalance mass

382 runs (normal + imbalance 6-35 g), 11.8-60.5 Hz, 1X phasors of the six accelerometers referred to the tachometer pulse. Every run is one sample; splits never share a run.

**Decision (criteria fixed before the run): NO GO**

- linear_in_mass: fail
- seen_speeds_physics_within_2g: fail
- unseen_speeds_physics_within_3g: fail
- modal_beats_amplitude_unseen: fail

## T1 linearity in mass (held-out R2 of the phasors)

linear 0.890, with an m^2 term 0.943; per channel 0.85, 0.90, 0.89, 0.30, 0.59, 0.90

## Level unbalance (one free complex value per level)

| level (g) | magnitude (g) | angle vs 35 g (deg) |
|---|---|---|
| 6 | 6.62 | 43.8 |
| 10 | 11.69 | 37.8 |
| 15 | 16.29 | 33.6 |
| 20 | 20.65 | 28.9 |
| 25 | 25.35 | 26.2 |
| 30 | 30.26 | 22.3 |
| 35 | 35.00 | 0.0 |

R2 0.954.

## Mass error (g)

| model | kind | seen: MAE | seen: right level | unseen interp: MAE | unseen extrap: MAE |
|---|---|---|---|---|---|
| spline | pre-registered | 2.67 | 74 % | 3.91 | 6.94 |
| modal3 | pre-registered | 2.74 | 72 % | 9.10 | 6.94 |
| modal4 | pre-registered | 2.72 | 72 % | 8.53 | 7.34 |
| amplitude | pre-registered | 1.70 | 85 % | 2.30 | 7.98 |
| rf_phase | pre-registered | 1.80 | 76 % | 3.83 | 9.40 |
| rf_amplitude | pre-registered | 1.71 | 78 % | 2.99 | 7.59 |
| nuisance_mass_angle | exploratory | 2.22 | 74 % | 3.55 | 7.16 |
| nuisance_tach | exploratory | 2.02 | 78 % | 3.23 | 7.31 |
| modal3_rephased | exploratory | 2.24 | 77 % | 9.50 | 9.30 |
| modal4_rephased | exploratory | 2.09 | 78 % | 9.68 | 8.38 |

Modal model (3 modes, all runs): natural frequencies 9.4, 18.8, 38.2 Hz, damping 0.040, 0.251, 0.117, R2 0.897.

## Limits

- The equivalent mass is at the calibration radius and angle: the radius of the trial masses and the modal masses enter only as a product, the known masses fix the scale.
- The 1X phase reference differs between mass levels (table above); phase-referenced inversion is therefore not reliable on this dataset as recorded.
- Exploratory models were added after the first run; they need confirmation on independent sessions (for instance the bearing-fault folders of MAFAULDA, which carry 0, 6, 20 and 35 g).

Figures: [level unbalance](level_unbalance.png), [mass errors](mass_errors.png)
