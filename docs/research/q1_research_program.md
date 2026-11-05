# Research programme: physics-based fault diagnosis on real data (working log)

This log is updated at every work session. Each paper keeps its hypotheses and decision criteria as fixed *before*
the run that tests them, the measured results with how they were measured, and what is still missing before
submission. A result that fails its criterion stays in the log. No acceptance in a given journal can be promised:
the target venues are where the work would fit once it meets the listed conditions.

## P1 -- MAFAULDA: physics-consistent severity, leakage-free, across sessions and speeds

**Question.** Can the severity of rotor faults (unbalance mass, misalignment offset) be estimated from vibration
with a physical model that holds at unseen speeds and in recording sessions it was not calibrated on, and where do
phase-referenced and learned models fail?

**Data.** MAFAULDA (SMT/COPPE/UFRJ): normal, imbalance 6-35 g, horizontal / vertical misalignment, under- and
overhang bearing faults with 0/6/20/35 g; 5 s at 50 kHz, tachometer, two triaxial accelerometers, microphone.

**Code.** `pinneapple_analysis.inspection.rotating` (`orders`, `mafaulda`, `influence`, `unbalance_study`),
example `examples/mafaulda_unbalance_pilot/`.

| Step | Status | Result |
|---|---|---|
| Pilot: calibrated 1X inversion for the unbalance mass (criteria fixed before the run) | done, NO GO | magnitude of the equivalent unbalance follows the added mass (6.6 ... 35 g); the 1X phase reference drifts between mass levels (44 to 0 deg); the classical phase-free calibration is the most accurate (1.70 g seen speeds, 2.30 g held-out speed bands); the linear dissipative modal model fails across speed bands (8.5-9.7 g); no model extrapolates in speed (about 7-9 g) |
| Tachometer quality | done | 81 underhang runs (79 ball fault, 2 cage fault) and 17 overhang runs have spurious or missing pulses (period spread up to 160 %); pulse detection now uses a minimum pulse width and a refractory period from the spectral fundamental; runs whose periods spread more than 5 % or whose speed is not 0.9-1.05 x the nominal one are excluded and counted |
| A1 cross-session, joint phase-free calibration (pre-registered) | done, FAIL | overhang (held out): 19.4 g (ball fault 54 g, cage 9.0 g, outer race 2.9 g); underhang: ball 3.1 g, cage 2.8 g, outer race 42.7 g. One overhang radial accelerometer reads up to 66 times its healthy 1X in some sessions: a measurement anomaly, not unbalance |
| A2 phase reference inside each bearing folder (pre-registered) | done, drift recurs | angle spans 20-38 deg in all six bearing folders: the 1X phase reference is not consistent between mass levels in any MAFAULDA session |
| A3 within-session calibration (reference) | done | 1.9-2.2 g (94-97 % right level) in five folders, 4.6 g in underhang ball fault (tachometer-affected) |
| A4 weighted median of channels (pre-registered on overhang) | done, FAIL | overhang 14.6 g |
| A5 screened estimator, biweight c = 12 selected on the development set (pre-registered on overhang) | done, FAIL by 0.79 g | overhang 3.79 g (90 % right level; outer race 2.17 g, cage 4.14 g, ball 5.42 g), five times below the joint calibration; healthy sessions 1.68 g against 1.70 g; it rejects the anomalous overhang radial channel in 99 % of the overhang ball-fault runs. Remaining error: the larger masses of overhang ball fault (35 g estimated 39 g) |
| B. Phase-free physics prior for speed extrapolation (pre-registered) | done, FAIL | outer bands: hold 14.3 / 1.65 g, physics w^4 / w^2 15.5 / 1.21 g, local log-slope 12.6 / 1.01 g (low band 11.8-20.2 Hz / high band 51.3-60.5 Hz). Above the resonances the mass-controlled w^2 law extrapolates; below about 20 Hz, where the modal fits put resonances, nothing does |
| Misalignment (2X, horizontal vs vertical, offset in mm) | planned | |
| Bearing faults (envelope order spectra; BPFO 2.998 and BPFI 5.002 are not resolvable from 3X and 5X in 5 s) | planned | |
| Literature: Shen et al. 2023 (HNN), Rigas et al. (KAN, arXiv 2412.01322), Sanami & Aghdam 2026 (inverse PINN, DOI to obtain), leakage-free benchmark (SBC BRESCI) | partly read | |

**Criteria of steps A and B (written 2026-11-04, before the bearing folders were opened):**

- A1 cross-session transfer: the phase-free calibration fitted on normal + imbalance estimates the added mass (0, 6,
  20, 35 g) of every run of the bearing folders with MAE <= 3 g (reported per folder and fault).
- A2 phase reference: inside each bearing folder, the angle of the free complex unbalance of 6, 20 and 35 g spans at
  most 10 deg; otherwise the phase drift of the imbalance sessions recurs.
- A3 within-session reference: leave-one-run-out inside each bearing folder with the same model, to separate the
  cost of changing session from the cost of the model.
- B speed extrapolation: continuing the phase-free calibration outside the training band with the physical power
  laws of an unbalance response (acceleration ~ w^4 below the resonances, ~ w^2 above them) brings the error of the
  two outer held-out speed bands to <= 4 g (it is 7-8 g when the calibration is held constant outside its band);
  compared with continuing the local log-slope of the data.

- A4 (written 2026-11-04 after the underhang folders, before the overhang folders were opened): the channel-robust
  estimator (weighted median of the per-channel phase-free mass estimates, weights from the calibration residuals and
  the channel sensitivity) brings A1 on the overhang folders to MAE <= 3 g, and it is at least as accurate as the
  weighted mean where no channel is abnormal.
- A5 (written 2026-11-04, after the underhang folders, before the overhang folders were opened; development set =
  calibration sessions + underhang, confirmation set = overhang): the screened estimator
  (`influence.ScreenedEstimator`: per-channel estimates with their joint-least-squares weights; distances
  standardised by the spreads measured in cross-validation on the calibration sessions; Tukey-biweight M-estimate
  of the common mass started from the unweighted median, its constant set at the 99.5 % quantile of the
  standardised calibration errors; a channel with zero weight is flagged) gives MAE <= 3 g on the overhang folders
  and stays within 0.2 g of the joint weighted estimate on the calibration sessions. Versions tried on the
  development set only, in order: weighted median (a dominant faulty channel became the centre), unweighted median
  (+0.86 g on healthy sessions), biweight with the normal-theory constant 4.685 (+0.78 g: the errors are
  heavy-tailed). Constant selected on the development set by the smallest worst-case MAE over the grid
  4.685-48 (calibration cross-validation and the three underhang folders): c = 12 (worst case 3.03 g; calibration
  1.68 g against 1.70 g for the joint estimate).

**Venue if the conditions are met:** Mechanical Systems and Signal Processing, Measurement, IEEE Trans.
Instrumentation and Measurement.

## P1b -- MAFAULDA fault classification: Fourier neural operator on order spectra vs chosen features

**Question.** Does a Fourier neural operator on the order and envelope spectra classify MAFAULDA faults better than
classical models on chosen features, and does either hold at rotation speeds it was not trained on?

**Tasks.** T1 misalignment: normal / horizontal / vertical. T2 unbalance: normal / imbalance. T3 bearings: normal /
underhang ball, cage, outer race / overhang ball, cage, outer race. T4 all: the ten states together.

**Inputs.** Chosen features per channel (seven channels): RMS, kurtosis, skewness, crest factor, peak, amplitudes at
1X, 2X, 3X, 4X, envelope amplitudes at FTF, BSF, about 3X and about 5X, plus the speed. The FNO takes the amplitude
and envelope order spectra (0-8 orders, 0.01 step, 14 channels, log-scaled) and the speed.

**Models.** Random forest and MLP on the features; FNO on the spectra (spectral layers, mean over the order axis, a
linear classifier); same splits for all.

**Protocols.** Every file is one sample (no window of a run is split). Pa: stratified 5-fold. Pb: speed bands --
runs split into three speed terciles, train on two, test on the third (the outer terciles are extrapolation in speed).

**Criteria (written 2026-11-04, before the misalignment folders were opened and before any classifier was run).**

- C1: under Pa, macro-F1 >= 0.95 for T1 and T2 with every model (expected easy; if not, the features or the
  labels are wrong).
- C2: under Pb, the FNO's macro-F1 exceeds the random forest's by at least 0.03 on at least two of T1-T4;
  otherwise no claim that the operator adds anything.
- C3: under Pa, macro-F1 >= 0.90 on T3, with its confusion matrix reported (which bearing faults look alike).
- C4: the drop of macro-F1 from Pa to Pb is reported for every model and task: a model whose drop exceeds 0.15 does
  not generalise across speed.

**Results T1-T2 (2026-11-04, macro-F1; 880 runs of normal, imbalance, horizontal and vertical misalignment).**

| Task | Model | Pa: 5 folds by run | Pb: unseen speed tercile (low / mid / high) |
|---|---|---|---|
| T1 misalignment | random forest | 0.986 | 0.874 (0.84 / 1.00 / 0.70) |
| T1 misalignment | MLP | 0.987 | 0.810 (0.85 / 0.96 / 0.58) |
| T1 misalignment | FNO on spectra | 0.944 | 0.856 (0.78 / 0.99 / 0.80) |
| T2 unbalance | random forest | 0.994 | 0.848 (0.82 / 1.00 / 0.66) |
| T2 unbalance | MLP | 0.994 | 0.879 (0.88 / 1.00 / 0.70) |
| T2 unbalance | FNO on spectra | 0.982 | 0.951 (0.88 / 1.00 / 0.96) |

C1 fails narrowly (FNO 0.944 < 0.95 on T1; the feature models pass). C2 so far: FNO gain over the random forest
under Pb is +0.10 on T2 and -0.02 on T1 (T3, T4 pending). C4: drops from Pa to Pb of 0.11-0.18 for the feature
models, 0.03-0.09 for the FNO; the MLP exceeds 0.15 on T1. The container restarted during T3-T4 (re-run).

**Results T3-T4 and decision (2026-11-04).** T3 bearings (7 classes, 1051 runs): Pa random forest 0.992, MLP
0.991, FNO 0.985; Pb 0.866 / 0.836 / 0.828. T4 all ten states (1951 runs): Pa 0.992 / 0.987 / 0.969; Pb 0.881 /
0.814 / 0.805 (FNO low-speed tercile 0.62). Against the criteria: C1 FAIL (FNO 0.944 on T1); C2 FAIL (FNO gain over
the random forest under Pb: T1 -0.02, T2 +0.10, T3 -0.04, T4 -0.08 -- only T2); C3 PASS for all three models; C4:
drops from Pa to Pb of 0.11-0.18, the FNO the most robust on T1 (0.09) and T2 (0.03) but not on T3 (0.16) and T4
(0.16); the MLP above 0.15 on three tasks. Under Pb the bearing confusions are mainly overhang cage vs overhang outer
race (41 runs) and normal vs overhang outer race / cage. Message for a paper: published scores near 99 % on MAFAULDA
hold only when the test speeds are seen in training; at unseen speeds every model loses 0.03-0.18; an FNO on order
spectra is not better in general, but it is clearly better for unbalance (0.951 against 0.848). Results file:
`docs/research/results/mafaulda_classification.json`.

**Comparison with published methods, same protocol (written 2026-11-04, before the KAN and HNN baselines were run
on MAFAULDA).** KAN: a shallow KAN on the same chosen features (Rigas et al. report F1 97.24 % with their own
features and selection); expectation if their result transfers: macro-F1 >= 0.95 under Pa on T4. HNN: energy
features of a Hamiltonian network fitted to healthy orbits, with a logistic regression (Shen et al.) and a random
forest; Shen et al. report AUC 0.78 / 0.84, so an HNN-energy model below the feature models is expected. Both are
re-implementations, not the authors' code. Seeds: 5 for random forest, MLP and KAN, 3 for the FNO; mean and 95 %
interval (t distribution) of macro-F1 per task and protocol.

**Seeds, KAN and HNN (2026-11-04).** Macro-F1, mean +- 95 % interval over 5 seeds (folds change with the seed):

| Task | Pa: RF / MLP / KAN | Pb: RF / MLP / KAN |
|---|---|---|
| T1 misalignment | 0.988+-0.003 / 0.985+-0.003 / 0.986+-0.003 | 0.876+-0.009 / 0.838+-0.029 / 0.858+-0.016 |
| T2 unbalance | 0.990+-0.006 / 0.991+-0.006 / 0.991+-0.006 | 0.852+-0.010 / 0.884+-0.015 / 0.873+-0.009 |
| T3 bearings | 0.994+-0.001 / 0.988+-0.004 / 0.992+-0.001 | 0.866+-0.010 / 0.837+-0.004 / 0.811+-0.011 |
| T4 all | 0.992+-0.001 / 0.986+-0.001 / 0.989+-0.002 | 0.878+-0.003 / 0.817+-0.006 / 0.824+-0.017 |

KAN reproduces the level reported by Rigas et al. under Pa (0.989 on T4, expectation >= 0.95 met) and loses
0.17 at unseen speeds. HNN-energy baseline (seed 0), logistic regression / random forest: T2 0.913 / 0.932 (Pa),
0.744 / 0.759 (Pb); T1 0.577 / 0.758, 0.514 / 0.629; T4 0.536 / 0.760, 0.371 / 0.508 -- below every feature model,
as expected from Shen et al. **Correction:** the FNO's unseen-speed result on T2 (0.951 with seed 0) does not hold
with seed 1 (0.813); its seed-to-seed spread is large (T1 Pa 0.944 vs 0.986), so the claim "the FNO is clearly
better for unbalance" is withdrawn until the 3-seed interval is in.

**First FNO over 3 seeds (mean +- 95 % interval, standard deviation).** Pb: T1 0.800 +- 0.137 (sd 0.055), T2
0.864 +- 0.187 (sd 0.075), T3 0.803 +- 0.059 (sd 0.024), T4 0.800 +- 0.043 (sd 0.017); random forest over 5 seeds
0.876 / 0.852 / 0.866 / 0.878 with sd <= 0.008. The first FNO is below the random forest at unseen speeds on every
task (T2 not significant) and its seed spread is 3-10 times larger. C2 is answered: no.

**OC-FNO (order-conditioned FNO), criteria written 2026-11-04 before it was run.** Changes to the FNO, each with
its reason: (1) training stabilised (lr 1e-3 cosine, gradient clipping 1, dropout 0.1, label smoothing 0.05,
per-channel normalisation estimated on the training split only -- the first FNO used a median over all runs);
(2) FiLM conditioning of every spectral block on the speed; (3) two physical order channels, a comb at the
harmonics kX and a comb at the bearing orders; (4) transfer-function augmentation (random log-normal gain per
sensor, sigma 0.3), because the order of a line does not move with speed but its amplitude follows H(w).
Evaluation under Pb only (Pa is saturated near 0.99), 5 seeds, tasks T1-T4 and COMFAULDA K1.

- O1: OC-FNO beats the random forest under Pb by >= 0.03 macro-F1, with the 95 % interval of the paired
  (same-seed) difference above 0, on at least two of the five tasks.
- O2: ablations on T4 and COMFAULDA K1 (without FiLM, without order channels, without augmentation) say which
  change carries the gain; a change whose removal costs less than 0.01 is not claimed.
- O3: the seed-to-seed spread (standard deviation of macro-F1) of OC-FNO is at most half that of the first FNO.

## P2 -- Process, sensor or model? Physics-residual diagnosis on a real PV microgrid

**Question.** Do whitened physics residuals with a likelihood ratio per hypothesis tell process faults from sensor
faults from a fast model outside its domain on laboratory data, not only in simulation?

**Data.** Simulated DC microgrid (done, `pinneapple_systems.digital_twin.dc_microgrid`); GPVS-Faults (Bakdi et al.,
Mendeley Data, DOI 10.17632/n76t439f65.1): grid-connected PV system in the laboratory, faults in the array, the
inverter, a feedback sensor, the grid and the MPPT controller.

| Step | Status | Result |
|---|---|---|
| Simulated four-bus DC microgrid twin | done | 80/80 controlled cases right (kind and target), 1 case with a false alarm; one confusable pair (joints next to bus 2, 25 % of operating points); PINN fast model 1.4 sigma with 8 labels against 13.4 sigma for the same network without physics |
| GPVS-Faults: signals, residuals that the measured quantities allow, benchmark | in progress | 16 files (F0-F7 x IPPT/MPPT), about 10 kHz in the CSVs, 14 s each, fault introduced halfway; Ipv, Vpv, Vdc, ia-ic, va-vc, PLL magnitudes and frequencies. Faults (arXiv 2112.14842): F1 inverter IGBT, F2 feedback sensor of one phase voltage, F3 grid voltage fluctuations, F4 partial shading, F5 15 % open circuit of the array, F6 boost PI gain, F7 boost PI time constant |

**Criteria of the GPVS-Faults pilot (written 2026-11-04, after reading only the column statistics of F0L, before
any fault file was analysed).** Residuals per grid cycle (20 ms): R1 = ia + ib + ic over the current RMS (three-wire
Kirchhoff, current sensors only); R2 = va + vb + vc over the voltage RMS (phase voltages of a three-wire system,
voltage sensors only); R3 = DC-link power balance Vpv Ipv - sum(v i) - d(C Vdc^2 / 2)/dt - P_loss, with C and a
loss model fitted on the first half of F0 and checked on its second half. Thresholds: 1.25 x the largest windowed
value over the first 40 % of F0L and F0M.

- G1 sensor fault: R2 exceeds its threshold after the onset in F2L and F2M, and the phase whose voltage best
  explains R2 (least squares on the three phase voltages) is the same through the faulty half.
- G2 not a sensor fault: R1 and R2 stay below their thresholds after the onset in F1, F3, F4, F5, F6 and F7 (both
  modes), while those faults are visible in the process variables (a change of Vdc, Vpv, Ipv or of the current
  waveform against the first half).
- G3 false alarms: no residual exceeds its threshold in the first 40 % of any file.
- G4 power balance: R3 has a median absolute value on the second half of F0 below 5 % of the PV power, and its
  behaviour in each fault file is reported (no criterion fixed: which faults break the power balance is the
  question).

**GPVS-Faults pilot results (2026-11-04).** Pre-registered: G1 FAIL (R2, the voltage sum, never breaks in F2: the
data show that F2 is a *current* sensor stuck at a constant -- ib at about -0.22 A in F2M, ic in F2L; correlation with
its phase voltage drops from -1.0 to -0.02 -- not a voltage sensor as the secondary description says); G2 FAIL as
written (R1 exceeds after F1M, F5L, F5M, where the inverter shuts down and its current falls below 30 %); G3 FAIL
(false alarms of R1 before the onset: 33 and 53 windows in the F5 sessions, 4 in F4L, 1 in F3L and F6L); G4 PASS
(power balance within 2 % of the PV power on the healthy halves; it explodes when the system shuts down).
Exploratory (added after the first run): with residuals evaluated only while the inverter delivers current, R1 flags
F2L and F2M in 84 % and 88 % of the faulty windows and the admittance rule (each phase current against the common
admittance of the three phases) names phase c and phase b -- the stuck sensors seen in the waveforms; in the twelve
process-fault files R1 exceeds in at most 3 % of the operating windows, so no sensor is blamed. To do: per-session
offset of the current sensors (the F5 false alarms), the IPPT/MPPT and shutdown states, a process-change detector for
the faults that keep the laws (F3, F4, F6, F7 hardly move the process variables), and a comparison with the PCA-KDE
detector of Bakdi et al.

**Venue if the conditions are met:** IEEE Trans. Industrial Electronics, IEEE Trans. Industrial Informatics,
International Journal of Electrical Power and Energy Systems.

## P3 -- COMFAULDA: combined faults recognised from single faults

**Question.** Trained on single faults only, can a phase-free physical representation recognise and size the
components of combined faults (unbalance + misalignment, two misalignments) that were never seen?

**Data.** COMFAULDA (Zenodo 8208454, CC BY 4.0, 23 GB, 2162 runs, 7 scenarios, 12-60 Hz).

| Step | Status | Result |
|---|---|---|
| Check its tachometer phase consistency and whether its bench matches MAFAULDA | done | no pulse tachometer in the CSVs, so no phase reference; six accelerometers of two types instead of two triaxial at two bearings: no direct transfer of MAFAULDA models, an independent bench |

**COMFAULDA, what the files are (2026-11-04, from reading two files of the archive).** 48 inner archives, one per
scenario and level; CSV with 9 columns (time, then 8), sampled at 51.2 kHz (not 50 kHz), 4.88 s; normal files
separated by ';', fault files by ', '. No column is a pulse tachometer; column 8 is near zero (the column users
report as not tied to a sensor); columns 2-7 are the six accelerometers. The speed is taken from the file name,
refined by the 1X peak within +-5 %; without a phase reference only phase-free features are used.

**Criteria of the COMFAULDA study (written before any COMFAULDA feature was computed).**

- K1 seven-class classification (normal, unbalance, horizontal, vertical, unbalance + horizontal, unbalance +
  vertical, vertical + horizontal), same models and protocols as P1b: under Pa macro-F1 >= 0.95 for the random
  forest; under Pb the drop is reported (P1b found 0.03-0.18 on MAFAULDA).
- K2 combined faults from single faults: one detector per fault type (unbalance, horizontal, vertical) trained on
  normal and single-fault runs only, applied to the combined runs never seen: exact-match accuracy of the predicted
  pair >= 0.70 for at least one model, per-label F1 reported. Below that, combined faults are not recognisable
  from single faults with these inputs.
- K3 the FNO's gain over the random forest under Pb on K1 is reported (the C2 question, on an independent bench).

**COMFAULDA results (2026-11-04; 2161 runs: one file has a CRC error inside the published archive, four file names
are malformed and were parsed).** K1 seven classes, macro-F1, Pa / Pb: random forest 0.944 / 0.626, MLP 0.938 /
0.695, KAN 0.938 / 0.625, FNO 0.913 / 0.690. K1 FAILS narrowly (0.944 < 0.95). The drop at unseen speeds is larger
than on MAFAULDA (0.22-0.32): the speed effect found on MAFAULDA is confirmed on an independent bench. K3: FNO -
random forest under Pb = +0.064 (MLP +0.069). K2 FAILS clearly: detectors trained on normal + single faults
recognise the exact pair of a combined fault in 1.5 % (RF), 5.2 % (MLP), 5.0 % (KAN) of the 1143 combined runs;
unbalance is still detected inside combinations (F1 0.76-0.86), misalignment is not (F1 0.38-0.53). With these
phase-free inputs, the signature of a combined fault is not the sum of the single ones -- the superposition
hypothesis is refuted on this bench. Results: `docs/research/results/comfaulda_results.json`.

**Venue if the conditions are met:** Mechanical Systems and Signal Processing, Reliability Engineering & System
Safety.

## Session log

- 2026-11-03: DC microgrid twin (simulated); MAFAULDA unbalance pilot, NO GO for phase-referenced inversion.
- 2026-11-04: envelope and order spectra added to the MAFAULDA features; robust tachometer; steps A (cross-session, six bearing folders) and B (speed extrapolation) run against their pre-registered criteria: A1, A4, A5 and B fail, A2 shows the phase drift in every session; the screened estimator divides the cross-session error by five.
- Next: re-run the development results on the features of the corrected tachometer (pipeline running); misalignment (2X); P2 data (GPVS-Faults).
