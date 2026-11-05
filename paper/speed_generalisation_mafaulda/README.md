# Submission package: speed generalisation of fault classifiers on MAFAULDA and COMFAULDA

Target journal: *Measurement* (Elsevier). Alternatives of similar scope: *Mechanical Systems and Signal Processing*
(would need the physics-model contribution strengthened), *IEEE Transactions on Instrumentation and Measurement*,
*Engineering Applications of Artificial Intelligence*. Check the current scope and quartile before choosing.

## Contents

| File | Purpose |
|---|---|
| `main.tex`, `main.pdf` | manuscript (Elsevier `elsarticle`, review format, line numbers) |
| `references.bib` | bibliography (`elsarticle-num` style) |
| `tables/*.tex` | tables, generated -- do not edit by hand |
| `figures/*` | figures, generated |
| `make_tables_figures.py` | builds every table, figure and quoted number from `docs/research/results/*.json` |
| `stats.json` | every number of the tables with its seed count |
| `highlights.tex` / `highlights.pdf` | highlights (separate file required by Elsevier) |
| `cover_letter.tex` / `cover_letter.pdf` | cover letter |
| `submission.zip` | the files to upload |

Rebuild: from the repository root `python paper/speed_generalisation_mafaulda/make_tables_figures.py`, then
`latexmk -pdf main.tex` in this folder.

## Before submission (to do by the authors)

1. **Authors**: names, affiliations, corresponding author, ORCID, CRediT roles, funding (placeholders in `main.tex`).
2. **References**: entries marked `VERIFY` in `references.bib` were not checked against the publisher's record
   (Shen et al. first names and pages; Rigas et al. journal version; Gao et al. full author list, volume, pages;
   Fyfe & Munck; Zhao et al.). Read Gao et al. (2025) in full: the FNO-based encoder is reported by a reviewer of the
   programme and is cited as such.
3. **Literature**: run the systematic search described in `docs/research/literature_mafaulda.md` on Scopus / Web of
   Science, in particular for (a) speed-held-out evaluations of MAFAULDA, (b) the 2026 COMFAULDA multi-fault paper a
   reviewer mentioned, (c) neural-operator fault diagnosis since 2024; add what is found to the Introduction.
4. **Pending experiments that would strengthen the paper** (not run yet, see the research log):
   OC-FNO seed 2 on COMFAULDA (4 of 5 seeds in Table 3); ablations with more than one seed; the Hamiltonian baseline
   with 5 seeds; the OC-FNO without FiLM over 5 seeds (it is the best FNO variant in the ablation).
5. **Generative AI declaration**: Elsevier requires it; adapt the statement in `main.tex` to how the tool was used.
6. **Data and code**: archive the repository branch with a DOI (for example Zenodo) and replace the GitHub link in
   `references.bib` (`pinneapple`).
7. Read the whole manuscript critically: every claim must be supported by the tables, which are generated from the
   result files.

## Where the numbers come from

All results are in `docs/research/results/` of the repository; the hypotheses and criteria written before each
experiment, and the results that failed them, are in `docs/research/q1_research_program.md`.
