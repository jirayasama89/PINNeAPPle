# Phase 9 (started): verified literature and novelty positioning

Verification = metadata opened on Crossref (DOI, title, venue, authors, year) or the proceedings page; "abstract
read" only where stated. Nothing here is claimed from memory. Last update 2026-11-04.

## Closest works (verified)

| # | Work | Venue, year, DOI | Verified | What it means for this study |
|---|---|---|---|---|
| 1 | Vieira, Bauler, Rosa, Silva et al. — *Towards a more realistic evaluation of machine learning models for bearing fault diagnosis* | MSSP 2026, 10.1016/j.ymssp.2026.114640 | Crossref | Realistic-evaluation argument (segment / condition splits over-optimistic) is published in MSSP: our benchmark cannot claim to be the first to show split optimism; it must contribute the **speed-extrapolation** analysis and its physical causes |
| 2 | Xiao, Shao, Yan, Wang, Peng — *Domain generalization for rotating machinery fault diagnosis: A survey* | Adv. Eng. Informatics 2025, 10.1016/j.aei.2024.103063 | Crossref | DG baselines expected by reviewers (invariant learning, alignment, contrastive, DRO) |
| 3 | Gao, Qi, Sun, Zhu — *Cross-Domain Open-Set Fault Diagnosis for Rotating Machinery Based on Frequency-Aware Model With Neighborhood Invariance* | IEEE IoT J. 2025, 10.1109/JIOT.2025.3565810 | Crossref | FNO-type encoder already used for cross-domain diagnosis: no "first FNO" claim |
| 4 | *Physics-informed causal learning network for fault diagnosis of rotating machinery under unseen operating conditions* | Neurocomputing 2025, 10.1016/j.neucom.2025.130187 | Crossref (title, venue) | physics + invariance under unseen conditions exists |
| 5 | *DGNSCA: A Domain Generalized Network Leveraging Similarity and Class-Conditional Contrast ...* | IEEE Sensors J. 2026, 10.1109/jsen.2025.3649745 | Crossref | class-conditional contrastive DG exists: our SupCon variant is a baseline, not a contribution |
| 6 | *Speed adaptive gate: A novel auxiliary branch for enhancing deep learning-based rotating machinery fault diagnosis* | Measurement 2023, 10.1016/j.measurement.2023.113016 | Crossref (title, venue) | **speed-conditioned gating already published in Measurement**: FiLM-on-speed is not novel; our contribution can only be the evidence of *when* speed conditioning hurts extrapolation |
| 7 | *A Hybrid Generalization Network for Intelligent Fault Diagnosis of Rotating Machinery Under Unseen Working Conditions* | IEEE TIM 2021, 10.1109/tim.2021.3088489 | Crossref | DG under unseen working conditions in TIM since 2021 |
| 8 | *Deep Semi-supervised Domain Generalization Network for Rotary Machinery Fault Diagnosis under Variable Speed* | IEEE TIM 2020, 10.1109/tim.2020.2992829 | Crossref | variable-speed DG in TIM since 2020 |
| 9 | *A domain feature decoupling network for rotating machinery fault diagnosis under unseen operating conditions* | RESS 2024, 10.1016/j.ress.2024.110449 | Crossref | feature decoupling (fault vs condition) exists: relevant to Phase 5 "sensitivity on a fault-only representation" |
| 10 | Behroozi, Shen, Kifer — SC-FNO | ICLR 2025 (proceedings page) | abstract read | sensitivity regularisation idea (Phase 5, optional) |
| 11 | Xiao et al. — Amortized FNO | NeurIPS 2024, 10.52202/079017-3651 | Crossref | kernel parameterisation (Phase 6, only if a mode limitation is diagnosed) |
| 12 | Shi, Yin, Tierney, Gao — SirenFNO | IJCAI 2026, 10.24963/ijcai.2026/535 | Crossref | idem |
| 13 | Sakarvadia et al. — False promise of zero-shot super-resolution | ICLR 2026 (proceedings page) | abstract read | resolution invariance must be trained, not assumed |

Not verified (do not cite until found): a 2026 Taylor & Francis COMFAULDA multi-fault paper (not in Crossref
searches; a 2026 Research Square preprint "Data-Scarce Compound Fault Diagnosis in Rotating Machinery Using
Multi-Sensor Spectral Fusion" 10.21203/rs.3.rs-9410125/v1 may be it — open before citing); an Aconf 2026
"spectral normalisation + FNO" bearing paper.

## Novelty table (to be filled with Phase 1-7 evidence; no claim before the confirmation runs)

| Aspect | Closest prior work | This study (status) |
|---|---|---|
| Recording-disjoint + speed-group-disjoint protocol on MAFAULDA **and** COMFAULDA, with per-run predictions released | Vieira 2026 (bearings, other datasets) | done in the manuscript; per-run predictions added here |
| Physical explanation of the speed-extrapolation failure (transfer function / resonances in the low tercile) | none found on these datasets | partly argued in §4; Phase 8 adds per-band and per-class evidence |
| When speed conditioning hurts (FiLM vs speed in head vs none vs physical descriptor) | Speed adaptive gate (Measurement 2023) shows it helps under their protocol | Phase 1/3, R1-R2 |
| Compound faults from single faults with a factorised multi-label model | multi-label DG (PHM-AP 2023, 10.36001/phmap.2023.v4i1.3651, not yet read) | Phase 7 |
