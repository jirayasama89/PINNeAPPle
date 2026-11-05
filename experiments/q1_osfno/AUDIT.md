# Phase 0 audit: speed-generalisation study (MAFAULDA / COMFAULDA)

Branch audited: `claude/laughing-carson-yotf6f` at `243841cd`. Experimental branch: `exp/q1-audit-osfno`
(nothing on the audited branch is modified; the manuscript is untouched).
Status: MAFAULDA [DATA] items completed 2026-11-04 from the full re-extraction (`results/audit_data.json`); COMFAULDA items completed 2026-11-05.

## A. Traceability JSON → tables → manuscript

| Check | Result |
|---|---|
| `make_tables_figures.py` regenerates `tables/*.tex` and `stats.json` from `docs/research/results/*.json` | **PASS**: regenerated on 2026-11-04, `git diff` empty for tables and stats.json (only the PDF figures' timestamps differ) |
| Result JSONs store per-run predictions, partitions (run ids per fold) and per-seed configs | **FAIL**: only pooled macro-F1, accuracy, per-split macro-F1, confusion and time are stored. Paired per-recording comparisons (rule: same records, same partitions) are **impossible from the existing files**; every comparison in Phases 1-3 is therefore re-run with predictions saved. |
| Number of seeds per cell matches the captions | PASS for Tables seen/unseen/gain; ablation = 1 seed (stated); C1 OC-FNO = 4 seeds (stated as 0/4) |

## B. Claims in the text not supported by the tables

| Location | Claim | What the tables show | Correction |
|---|---|---|---|
| Abstract | "All models exceed 0.97 macro-F1 when test speeds are seen" | COMFAULDA Pa: RF 0.944, MLP 0.938, KAN 0.938, FNO 0.913; HNN+RF 0.758-0.932 on MAFAULDA | "On MAFAULDA, feature models and FNOs exceed 0.97; on COMFAULDA they reach 0.91-0.94; the Hamiltonian baseline 0.76-0.93" |
| Abstract, §3.2, Conclusions | "every model loses 0.11-0.18 on MAFAULDA and 0.25-0.32 on COMFAULDA" | MAFAULDA: FNO T3 0.184; HNN+RF T4 0.252, T3 0.280. COMFAULDA: FNO 0.913 → 0.690 = 0.223 | give the ranges per model family, or "0.11-0.28 / 0.22-0.32" |
| §3.2 | "The MLP is best on COMFAULDA (0.690)" | FNO also 0.690 (1 seed) | "MLP and FNO tie at 0.690 (FNO: one seed)" |
| Abstract, §3.3, §4 | "an ablation shows that its speed conditioning is what helps unbalance and what hurts the other faults" / "locates the cause" | ablation = seed 0 only, on T4 and C1 only (no T2 ablation). OC-FNO seed-to-seed sd on T4 is 0.047, the FiLM effect is 0.106: plausible but not established; nothing measured on T2 | "a single-seed ablation suggests ..." until Phase 1 (5 seeds, all tasks) |
| Title, abstract, contribution 1 | "leakage-free" | Pa is recording-disjoint but shares speeds, sessions and mounting; Pb is speed-group-disjoint but shares sessions (every folder spans all speeds) | "recording-disjoint (Pa)" and "speed-group-disjoint (Pb)"; state that sessions are shared |
| §2.1 | labels follow folders; bearing states "recorded with 0, 6, 20 and 35 g" | in T3/T4 a bearing-fault run with 35 g is labelled only as the bearing fault, although it also carries unbalance; "imbalance" is a separate exclusive class | state the label definition explicitly: bearing classes include runs with added unbalance (they are compound in MAFAULDA itself); analyse T4 errors by mass level |

## C. Representation and configuration

| Check | Finding |
|---|---|
| Order grid | spectra cached on 0-12X, step 0.01 → **1201 points** (`orders.order_spectrum`, `max_order=12`) |
| FNO / OC-FNO input | `classify.spectrum_tensor(max_order=8.0)` → **0-8X, 801 points**; the manuscript states both (§2.2 grid 0-12, §2.3 "0-8 orders") but not why; features use 1X-4X and envelope ≤ 5X, all inside 8X. Document explicitly; Phase 3 tests 8X vs 12X |
| First FNO normalisation | `spectrum_tensor(raw=False)` scales by the **median over all rows given (train + test)**: a mild transductive leak (one scalar). OC-FNO normalises per channel on the training split only. The "FNO" column is therefore not strictly train-only; re-run with train-only scaling in Phase 1 |
| Feature models | MLP: StandardScaler inside the pipeline (train only) ✓; KAN: mean/sd on train ✓; RF: no scaling ✓ |
| Speed terciles (Pb) | edges = quantiles of the speeds of the task's runs (all runs, no labels) → same edges for every model ✓; seeds do not change Pb partitions (only initialisation) ✓ |
| OC-FNO "without FiLM" | still appends the (standardised) speed to the head → it is variant **B (speed in the head)**, not "no speed"; variant A (no speed at all) has never been run |
| OC-FNO gain augmentation | same log-normal gain on a sensor's amplitude and envelope spectra ✓; applied per sensor of the 7 channels (MAFAULDA includes the microphone) |
| COMFAULDA speed | from the file name refined by the 1X peak (no tachometer) ✓; constant-speed angle; phase-free only |

## D. Data population [DATA where marked]

| Check | Finding |
|---|---|
| 1951 runs in the archive | **PASS**: full.zip lists 2004 members, 1951 CSV (folder counts in `LOG.md`) |
| 1895 used / 56 excluded | consistent arithmetic: T4 1895 = 1951 − 56; T1 547 = 49 + 197 + 301; T2 382 = 49 + 333; T3 1064 = 49 + 1071 − 56 ✓. **Recomputed: 1951 read without error; 56 fail `tach_ok` (underhang ball 37, underhang cage 2, overhang outer race 11, overhang cage 6; 40 by period spread > 5 %, 16 by both spread and speed ratio); 1895 used — exactly the manuscript's numbers.** |
| 98 flagged tachometers (research log) vs 56 exclusions (manuscript) | the log reports 81 underhang + 17 overhang runs with spurious/missing pulses **before** the pulse detector gained a minimum width and a refractory period; the manuscript's 56 (39 + 17) is after that change. The new rule gives 56, as in the manuscript. The old count (98) cannot be recomputed from the cache (raw tachometer not stored); the manuscript should say which detector produced which count |
| Duplicates | **PASS: no identical CSV (SHA-1) and no pair of runs with spectral cosine > 0.9999** |
| Folders with 50-51 files (HM 0.5 mm, UH ball 0 g, VM 0.51 mm) vs 49 nominal speeds | **PASS: no repeated nominal speed inside any folder** (the extra files are extra speeds); speed range of used runs 11.8-60.7 Hz; Pb tercile edges per task: T1 11.8/28.0/43.8/60.7, T2 11.8/27.9/43.5/60.5, T3 11.9/26.6/41.2/60.6, T4 11.8/27.0/42.3/60.7 Hz |
| Sessions | each folder (state × level) is a recording session; all sessions span all speeds → Pb never holds out a session. A session-disjoint check (e.g. underhang vs overhang folders of the same fault type) is a stronger OOD test (Phase 8) |
| COMFAULDA 2162 vs 2161 | **Confirmed: 2162 runs, one CRC error inside the published archive (`Vertical_Misalignment_1.27_mm.zip/18.78.csv`) → 2161** |
| COMFAULDA duplicates (NEW) | **Two pairs of byte-identical CSV files published under different speeds**: `Normal_1.zip/36.26.csv` = `Normal_1.zip/37.23.csv`, `Horizontal_misalignment_2_0mm.zip/18.07.csv` = `19.88.csv` (SHA-1 identical). Each pair falls inside one Pb tercile (27.0-42.4 Hz and 12.4-27.0 Hz), so Pb is not affected; under Pa (5-fold by run) a pair can be split across train and test (a leak of 1 recording each). The speed of one file of each pair is wrong (refined speed from the 1X line is the same for both). Report to the dataset authors; drop one file of each pair or keep both in the same fold |
| COMFAULDA repeated nominal speeds (NEW) | two folders contain two files with the same nominal speed: `Unbalance_6_g+Hor. Misalignment_1_mm.zip` (25.0 Hz) and `Normal_5.zip` (20.0 Hz) — distinct recordings (different SHA-1), dependent samples at the same condition |
| COMFAULDA population | 2161 used: normal 245, unbalance 336, horizontal 195, vertical 242, unbalance+horizontal 351, unbalance+vertical 351, horizontal+vertical 441; speeds 12.4-61.0 Hz; Pb tercile edges 12.35 / 27.04 / 42.42 / 61.05 Hz |

## E. Consequences and corrections (to apply only after Phase 1-3 confirmation; manuscript untouched)

1. Replace "leakage-free" by recording-disjoint / speed-group-disjoint everywhere.
2. Fix the four unsupported sentences of table B.
3. State the order range of each model (features ≤ 5X, FNO 0-8X, cache 0-12X).
4. Re-run the plain FNO with train-only scaling, or label its column.
5. Store predictions, partitions and configs for every run (done in `experiments/q1_osfno/`).
6. Do not present the FiLM ablation as causal until Phase 1 confirms it over 5 seeds and all tasks.

## F. Claims of the research journal re-examined after Phases 1-8 (2026-11-05)

| Claim (journal / manuscript) | New evidence | Status |
|---|---|---|
| COMFAULDA compound faults: "the superposition hypothesis is refuted" | `phase7.py superposition` (`results/phase7/superposition.json`): median relative error over 0-8X of Y_A + Y_B - Y_0 (nearest-speed single and normal runs) vs the real compound run: U+H 0.85, U+V 0.90, H+V 1.25; same-class noise floor over 0-8X (two runs of one single-fault class at the nearest speed): U 0.61, H 0.92, V 0.85 | **only partly supported**: for U+H and U+V the deviation is within the range of run-to-run variability of the constituent classes; for H+V (two misalignments) it exceeds both floors (1.25 vs 0.85-0.92). The claim must be restricted to the misalignment pair and stated against the noise floor |
| Binary relevance from singles fails on compounds (exact match ≈ 0.01) | `phase7.py run br-rf`, 5 seeds: exact 0.010-0.018, per-label F1 0.86 / 0.36-0.40 / 0.42 | reproduced |
| OS-FNO / OC-FNO is the strongest speed-generalising model | Phase 1, 5 tasks: only the residual CNN on order spectra beats the best feature baseline under R3; OS-FNO variants win only T2 | **not supported** (see PHASE1_REPORT.md) |
| FiLM speed conditioning helps | R1: speed in the head (B) beats FiLM (C) with CI > 0 on 4/5 tasks; R2: no speed input (A) is as good as B | **reversed**: FiLM hurts; speed input not needed |
