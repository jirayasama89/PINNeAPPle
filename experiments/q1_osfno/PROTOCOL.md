# Locked protocol of the Q1 programme (written 2026-11-04, before any new model was evaluated)

This file is committed before the first evaluation on held-out speed terciles. Changes after that point go to the
"Deviations" section at the end, with date and reason. Nothing in `docs/research/results/` or in
`paper/speed_generalisation_mafaulda/` is modified by this programme.

## Data
* MAFAULDA re-extracted from the published `full.zip` with the repository's own code (`rotating.mafaulda.features`,
  orders 1-4, spectra 0-12X) plus extra representations (multi-band envelopes, angle-domain signals at 64 / rev,
  first second of the raw accelerometers). Population = runs with `tach_ok` (the manuscript's rule).
* COMFAULDA re-extracted from Zenodo 8208454 with `rotating.comfaulda.features` (+ the same extras where a phase-free
  version exists).
* A run (one 5 s recording) is indivisible.

## Partitions (identical for every model)
* **Pb (main)**: speed terciles, `classify.splits(protocol="speed")` (the manuscript's partitions); each tercile is
  held out in turn; low and high = extrapolation, mid = interpolation.
* **Pa**: stratified 5-fold by run (reported, not used for selection).
* Every result file stores the run paths of each split and a hash of the partition; comparisons are only made between
  files with the same hash.

## Selection (exploration) vs confirmation
* Exploration: hyper-parameters and architectural choices are compared ONLY on inner splits of the training runs of
  each outer fold (`q1lib.inner_splits`: 4 contiguous speed blocks of the training speeds, leave-one-block-out). The
  held-out tercile is never used to choose anything.
* Confirmation: the configuration chosen on inner splits is written to `locks/<experiment>.json` (with a hash and
  a timestamp) before the outer evaluation. Outer results are produced once per lock; a configuration changed after
  seeing outer results is a new, separately reported experiment.
* Seeds: 5 (0-4) for every stochastic model in confirmation; no architecture is retained on one seed.

## Metrics
Macro-F1 pooled over the three held-out terciles (main), balanced accuracy, per-tercile macro-F1 (low / mid / high),
worst tercile, per-class F1, confusion matrices, fit and inference time, parameter count; class probabilities stored
for calibration (ECE, NLL) where the model gives them.

## Statistics
Paired comparisons on the same runs and seeds: (i) per-seed difference of pooled macro-F1 with 95 % t-interval over
seeds; (ii) record-level paired bootstrap (10 000 resamples of runs, stratified by class) of the macro-F1 difference
for seed-averaged predictions; (iii) McNemar on seed-majority predictions. Holm correction over the tasks of one
comparison family.

## Decision rules (written before the runs)
* **R1 (FiLM)**: "removing FiLM helps" is retained only if OS-FNO-B minus OS-FNO-C on Pb is > 0 with the 95 % seed
  interval above 0 on at least 3 of the 5 tasks (T1-T4, C1).
* **R2 (speed in head)**: A vs B vs E compared the same way; a speed representation is retained only if it beats A
  with the interval above 0 on ≥ 3 tasks; otherwise A (no speed) is the default.
* **R3 (superiority)**: a new model is said to outperform the baselines only if its paired difference with the best
  baseline of each task (RF, MLP, KAN, SVM, XGB, CNN) has its interval above 0 on ≥ 3 of the 5 tasks and it is not
  worse (interval below 0) on any. Otherwise the critical benchmark remains the main contribution.
* **R4 (worst case)**: every claim is also reported on the worst tercile (low speed).

## Deviations
(none yet)

## Reproducibility and history (added 2026-11-04, does not change any decision rule)
* `env/environment.txt`, `env/pip_freeze.txt`: exact environment.
* `results/data_manifest_<dataset>.json`: every cached run with the SHA-1 of its raw CSV and the SHA-256 of its cache file.
* `HISTORY.jsonl`: one line per result file registered (UTC, SHA-256, task, model, seed, partition hash, scores,
  code commit); never rewritten, only appended; a re-run of an existing file is recorded as "changed".
* `SUMMARY.md`: regenerated from HISTORY.jsonl.
* `autosave.sh`: every 10 minutes, registers new results, commits and pushes a checkpoint.
* From this point, every result JSON also carries `provenance` (code commit, dirty flag, library versions).

## Phase 6 pre-registration (written 2026-11-05, before any Phase 6 run)
- Hypothesis H6: band-limited envelope order spectra (carriers 0.5-2, 2-5, 5-10, 10-20 kHz, extracted once with the
  data, never tuned) improve the bearing classes over the single broadband envelope.
- Models fixed in advance (no tuning, no selection): `rf-mb` = Phase 1 RF + amplitudes at FTF/BSF/BPFO/BPFI of each
  band envelope and sensor; `cnn-mb` = Phase 1 cnn-spec + the band envelopes (0-8X) as extra input channels.
- Tasks T3 (bearings) and T4 (all), 5 seeds, Phase 1 partitions; comparisons rf-mb vs rf and cnn-mb vs cnn-spec.
- Decision: H6 is supported if the difference is > 0 with 95 % CI above 0 on both T3 and T4 and the bearing-class mean
  F1 increases; otherwise it is reported as a negative result.
- Fusion of RF and CNN probabilities (`fusion.py`, alpha = 0.5) is EXPLORATORY: the pair was chosen after seeing
  Phase 1 per-band results; any fusion claim requires a gate selected on inner validation (Phase 5).

## Phase 8 (added 2026-11-05, evaluation only)
- Calibration (ECE, Brier, NLL), per-class F1, cost: from stored Phase 1 probabilities (`phase8_analysis.py`).
- Robustness (`phase8_robust.py`): Phase 1 models retrained identically, evaluated on perturbed copies of the
  held-out runs only (speed error ±1/2/5 %, dead sensor, gain x2 per sensor). Clean predictions reproduce Phase 1
  exactly (checked: RF T2 seed 0 macro-F1 0.848 in both).
